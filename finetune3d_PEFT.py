import sys
import os
sys.path.extend(['.', './../'])
os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# add plotly import and availability check
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    print("[OK] Plotly successfully imported for 3D visualization")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    print(f"[X] Plotly not available: {e}")
    print("  3D visualization will be disabled")

from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_3d_components_from_2d
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset, TemporalDataset3D
from utils.make_master_file import DATASET_DICT
from models.unet import UNet
from models.fno import FNO2d, FNO3d
from models.dpot3d_Lora import DPOTNet3D as DPOTNet3D_Lora
from models.dpot3d_adapter import DPOTNet3D as DPOTNet3D_Adapter
from models.dpot3d_fadapter import DPOTNet3D as DPOTNet3D_fAdapter
from models.dpot3d_adapter_waveact import DPOTNet3D as DPOTNet3D_AdapterWaveAct
from models.dpot3d_finverse_adapter import DPOTNet3D as DPOTNet3D_FInverseAdapter
from models.dpot3d_Film_adapter import DPOTNet3D as DPOTNet3D_FilmAdapter
from models.dpot3d_AdaloraLayer import DPOTNet3D_AdaLora

from fvcore.nn import FlopCountAnalysis
from models.dpot3d_HydraLoRA import DPOTNet3D_HydraLora
from models.dpot3d_RandLora import DPOTNet3D_RandLora
from models.dpot3d_SVFT import DPOTNet3D_SVFT
from models.dpot3d_adapter_chebyKAN import DPOTNet3D as DPOTNet3D_AdapterChebyKAN
from models.dpot3d import DPOTNet3D  # base DPOT model
from models.dpot3d_adapter_fourierkan import DPOTNet3D as DPOTNet3D_AdapterFourierKAN

# torch.manual_seed(0)
# np.random.seed(0)


################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

### currently no influence
parser.add_argument('--model', type=str, default='DPOT')
parser.add_argument('--dataset',type=str, default='ns3d')

parser.add_argument('--train_path', type=str, default='ns3d_pdb_M1_turb') #pdebench
parser.add_argument('--test_path',type=str, default='ns3d_pdb_M1_turb')
parser.add_argument('--resume_path',type=str, default='/root/finetune_checkpoints/model_H.pth')
parser.add_argument('--ntrain', type=int, default=540)
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=True)

parser.add_argument('--res', type=int, default=64)
parser.add_argument('--noise_scale',type=float, default=0.0)
# parser.add_argument('--n_channels',type=int,default=-1)

### shared params
parser.add_argument('--width', type=int, default=2048)#High
parser.add_argument('--n_layers',type=int, default=27)
parser.add_argument('--act',type=str, default='gelu')

### GNOT params
parser.add_argument('--max_nodes',type=int, default=-1)

### FNO params
parser.add_argument('--modes', type=int, default=256)
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)


### AFNO
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=4)

parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='step')#originally default='step'
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.7) #originally 0.5
parser.add_argument('--warmup_epochs',type=int, default=50)
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/root/autodl-tmp/logs')


### finetuning parameters
parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=12)
parser.add_argument('--load_components',nargs='+', type=str, default=['blocks'])

# add load_mode argument
parser.add_argument('--load_mode', type=str, default='base_only',
                  choices=['base_only', 'peft_only', 'resume_training'],
                  help='model loading mode')

# add peft_path argument
parser.add_argument('--peft_path', type=str, default='',
                  help='PEFT parameter path')

### PEFT parameters
parser.add_argument('--peft_method', type=str, default='lora',
                  choices=['none', 'lora', 'adapter', 'adalora', 'hydralora', 'randlora', 'svft', 'prompt_tuning'],
                  help='parameter-efficient finetuning method (none means base model without PEFT)')
parser.add_argument('--peft_dim', type=int, default=8,
                  help='Dimension for PEFT components (lora rank or adapter bottleneck dim)')
parser.add_argument('--peft_scale', type=float, default=1.0,
                  help='Scale for PEFT components (lora alpha or adapter scale)')

# AdaLoRA specific arguments
parser.add_argument('--adalora_update_freq', type=int, default=10,
                  help='update frequency for AdaLoRA importance scores (steps)')
parser.add_argument('--adalora_target_r', type=int, default=8,
                  help='target rank for AdaLoRA')
parser.add_argument('--adalora_tinit', type=int, default=200,
                  help='initial warmup steps for AdaLoRA')
parser.add_argument('--adalora_tfinal', type=int, default=800,
                  help='final finetuning steps for AdaLoRA')

# HydraLoRA specific arguments
parser.add_argument('--expert_num', type=int, default=4,
                  help='Number of experts for HydraLoRA')

# RandLoRA specific arguments
parser.add_argument('--proj_factor', type=int, default=4,
                  help='Projection factor for RandLoRA (determines projection dimension)')

# SVFT specific arguments
parser.add_argument('--svft_r', type=int, default=32,
                  help='rank for SVFT')
parser.add_argument('--svft_alpha', type=float, default=1.0,
                  help='scale for SVFT')
parser.add_argument('--off_diag', type=int, default=2,
                  help='off-diagonal bandwidth for SVFT')
parser.add_argument('--pattern', type=str, default='banded',
                  help='sparsity pattern for SVFT, options: banded, random, top_k')

# --- Step 2a: Update Adapter type choices ---
parser.add_argument('--adapter_type', type=str, default='original',
                  choices=['original', 'chebykan', 'fadapter', 'fourierkan', 'waveact', 'finverse', 'film'],
                  help='Type of adapter to use when peft_method is adapter')

parser.add_argument('--cheby_degree', type=int, default=8,
                    help='Degree for Chebyshev polynomial in ChebyKAN layer (only used when adapter_type is chebykan)')

# Prompt Tuning specific arguments
parser.add_argument('--prompt_dim', type=int, default=32,
                  help='prompt vector dimension')
parser.add_argument('--prompt_pos', type=str, default='both',
                  choices=['pre', 'post', 'mid', 'both'],
                  help='prompt insertion position: pre, post, mid, both')

# fAdapter / frequency-adaptive adapter parameters
parser.add_argument('--power', type=float, default=2.0,
                    help='Power parameter for fAdapter dimension scaling')
parser.add_argument('--num_bands', type=int, default=4,
                    help='Number of frequency bands for fAdapter')
parser.add_argument('--min_dim_factor', type=float, default=0.5,
                    help='Minimum dimension factor for fAdapter (relative to peft_dim)')
parser.add_argument('--max_dim_factor', type=float, default=2.0,
                    help='Maximum dimension factor for fAdapter (relative to peft_dim)')

# --- Step 2b: Remove FourierKAN specific arguments ---
# parser.add_argument('--fourierkan_gridsize', ...) # 
# parser.add_argument('--fourierkan_addbias', ...) # 
# parser.add_argument('--fourierkan_smooth_init', ...) # 
# parser.add_argument('--fourierkan_share_adapters', ...) # 
# parser.add_argument('--fourierkan_sparse_adapters', ...) # 

# energy spectrum parameters
parser.add_argument('--enable_spectrum', action='store_true', default=True,
                  help='whether to enable energy spectrum analysis')
parser.add_argument('--spectrum_bins', type=int, default=64,
                  help='number of spectral shells')
parser.add_argument('--physical_size', type=float, default=1.0,
                  help='physical domain size (isotropic assumption)')
parser.add_argument('--velocity_channels', type=int, default=3,
                  help='number of velocity components (typically 3: u,v,w). Note: ns3d_pdb_M1_turb has 5 channels')

# 3D visualization parameters
parser.add_argument('--enable_3d_vis', action='store_true', default=True,
                  help='whether to enable 3D velocity visualization')
parser.add_argument('--vis_resolution', type=int, default=30,
                  help='3D visualization resolution (for downsampling)')
parser.add_argument('--vis_isomin', type=float, default=0.2,
                  help='isosurface minimum')
parser.add_argument('--vis_isomax', type=float, default=0.8,
                  help='isosurface maximum')

args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))

print(f"Current working directory: {os.getcwd()}")

################################################################
# energy spectrum computation function
################################################################

def energy_spectrum_3d(u, bins=64, physical_size=1.0):
    """
    Compute kinetic energy spectrum of 3D velocity field
    u: Tensor [C, H, W, D] (C=3 velocity)
    bins: number of frequency shells
    physical_size: physical domain size (isotropic cube; adjust if xyz)
    Returns: k_shell (center wavenumber) [bins], E_shell [bins]
    """
    # 1. FFT -> complex coefficients
    uh = torch.fft.fftn(u, dim=(-3, -2, -1), norm='forward')     # shape same as u
    ke_density = 0.5 * (uh.real**2 + uh.imag**2).sum(0)          # [H,W,D]

    # 2. compute wavenumber magnitude |k|
    nx, ny, nz = u.shape[-3:]
    kx = torch.fft.fftfreq(nx, d=physical_size / nx, device=u.device)
    ky = torch.fft.fftfreq(ny, d=physical_size / ny, device=u.device)
    kz = torch.fft.fftfreq(nz, d=physical_size / nz, device=u.device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')        # [H,W,D]
    k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)

    # 3. Shell averaging: bin (|k|, energy) into histogram
    k_flat = k_mag.flatten()
    e_flat = ke_density.flatten()
    k_max = k_flat.max()
    hist_E = torch.zeros(bins, device=u.device)
    hist_cnt = torch.zeros_like(hist_E)
    bin_idx = torch.clamp((k_flat / k_max * (bins-1)).long(), 0, bins-1)
    hist_E.index_add_(0, bin_idx, e_flat)
    hist_cnt.index_add_(0, bin_idx, torch.ones_like(e_flat))
    E_shell = hist_E / hist_cnt.clamp_min(1)                     # energy spectrum density
    k_shell = torch.linspace(0, k_max, bins, device=u.device)    # corresponding center wavenumber
    return k_shell.cpu(), E_shell.cpu()

def create_3d_velocity_visualization(velocity_field, title, save_path, args, 
                                   isomin=0.2, isomax=0.8, resolution=30):
    """
    Create 3D velocity isosurface visualization
    velocity_field: [3, H, W, D] - 3 velocity components
    title: plot title
    save_path: save path
    """
    print(f"[INFO] [3D visualization] start creating: {title}")
    print(f"   save path: {save_path}")
    print(f"   input shape: {velocity_field.shape}")
    print(f"   PLOTLY_AVAILABLE: {PLOTLY_AVAILABLE}")
    
    if not PLOTLY_AVAILABLE:
        print("[X] [3D visualization] Plotly not installed, skip 3D visualization")
        print("   Please run: pip install plotly kaleido")
        return
    
    try:
        # Computing velocity magnitude |v| = sqrt(u^2 + v^2 + w^2)
        print(f"   Computing velocity magnitude...")
        velocity_magnitude = torch.sqrt(
            velocity_field[0]**2 + velocity_field[1]**2 + velocity_field[2]**2
        ).cpu().numpy()
        
        print(f"   velocity magnitude range: [{velocity_magnitude.min():.6f}, {velocity_magnitude.max():.6f}]")
        
        # data normalization
        v_min, v_max = velocity_magnitude.min(), velocity_magnitude.max()
        if v_max > v_min:
            velocity_magnitude = (velocity_magnitude - v_min) / (v_max - v_min)
            print(f"   normalized to [0, 1] range")
        
        # downsample to speed up visualization
        H, W, D = velocity_magnitude.shape
        step_h = max(1, H // resolution)
        step_w = max(1, W // resolution)
        step_d = max(1, D // resolution)
        
        print(f"   original size: {H}x{W}x{D}")
        print(f"   downsample stride: {step_h}x{step_w}x{step_d}")
        
        # create grid coordinates
        x, y, z = np.mgrid[0:H:step_h, 0:W:step_w, 0:D:step_d]
        
        # downsample
        velocity_sampled = velocity_magnitude[::step_h, ::step_w, ::step_d]
        print(f"   downsampled size: {velocity_sampled.shape}")
        
        # create isosurface plot
        print(f"   create isosurface plot (isomin={isomin}, isomax={isomax})...")
        fig = go.Figure(data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(), 
            z=z.flatten(),
            value=velocity_sampled.flatten(),
            isomin=isomin,
            isomax=isomax,
            caps=dict(x_show=False, y_show=False),
            showscale=True,
            colorscale='Viridis'
        ))
        
        # update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        # ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # save image
        html_path = save_path.replace('.png', '.html')
        print(f"   save HTML file: {html_path}")
        fig.write_html(html_path)
        
        try:
            print(f"   save PNG file: {save_path}")
            fig.write_image(save_path)
            print(f"[OK] [3D visualization] PNG saved successfully: {save_path}")
        except Exception as png_error:
            print(f"[WARN]  [3D visualization] PNG save failed (HTML saved): {png_error}")
            print(f"   suggest installing kaleido: pip install kaleido")
        
        print(f"[OK] [3D visualization] HTML file saved: {html_path}")
        
    except Exception as e:
        print(f"[X] [3D visualization] creation failed: {e}")
        import traceback
        print(f"   detailed error: {traceback.format_exc()}")

def create_velocity_comparison(pred_velocity, true_velocity, epoch, log_path, args):
    """3D velocity field comparison (prediction vs ground truth) visualization"""
    print(f"\n[INFO] [velocity field comparison] Epoch {epoch} - start creating 3Dvelocity field")
    print(f"   predicted data shape: {pred_velocity.shape}")
    print(f"   ground-truth data shape: {true_velocity.shape}")
    print(f"   save path: {log_path}")
    print(f"   PLOTLY_AVAILABLE: {PLOTLY_AVAILABLE}")
    print(f"   enable_3d_vis: {args.enable_3d_vis}")
    
    if not PLOTLY_AVAILABLE:
        print("[X] [velocity field comparison] Plotly not installed, skip 3D visualization")
        return
        
    if not args.enable_3d_vis:
        print("[X] [velocity field comparison] 3D visualizationnot enabled (--enable_3d_vis)")
        return
        
    # select first sample to visualize
    pred_sample = pred_velocity[0]  # [3, H, W, D]
    true_sample = true_velocity[0]  # [3, H, W, D]
    print(f"   select first sample to visualize:")
    print(f"   predicted sample shape: {pred_sample.shape}")
    print(f"   ground-truth sample shape: {true_sample.shape}")
    
    # create prediction visualization
    pred_save_path = os.path.join(log_path, f'velocity_pred_epoch_{epoch}.png')
    print(f"   [PLOT] create prediction velocity visualization...")
    create_3d_velocity_visualization(
        pred_sample, 
        f'Predicted Velocity Field - Epoch {epoch}\n{args.peft_method} (dim={args.peft_dim})',
        pred_save_path,
        args,
        isomin=args.vis_isomin,
        isomax=args.vis_isomax,
        resolution=args.vis_resolution
    )
    
    # Only create GT visualization every 100 epochs
    if epoch % 100 == 0 or epoch == args.epochs:
        print(f"   [PLOT] create ground truth velocity visualization (Epoch {epoch})...")
        true_save_path = os.path.join(log_path, f'velocity_ground_truth_epoch_{epoch}.png')
        create_3d_velocity_visualization(
            true_sample,
            f'Ground Truth Velocity Field - Epoch {epoch}',
            true_save_path, 
            args,
            isomin=args.vis_isomin,
            isomax=args.vis_isomax,
            resolution=args.vis_resolution
        )
        
    # create side-by-side comparison
    if epoch % 100 == 0 or epoch == args.epochs:  # Only create comparison when GT exists
        print(f"   [PLOT] create comparison visualization...")
        try:
            pred_mag = torch.sqrt(pred_sample[0]**2 + pred_sample[1]**2 + pred_sample[2]**2).cpu().numpy()
            true_mag = torch.sqrt(true_sample[0]**2 + true_sample[1]**2 + true_sample[2]**2).cpu().numpy()
            
            # normalize to same range
            v_min = min(pred_mag.min(), true_mag.min())
            v_max = max(pred_mag.max(), true_mag.max())
            if v_max > v_min:
                pred_mag = (pred_mag - v_min) / (v_max - v_min)
                true_mag = (true_mag - v_min) / (v_max - v_min)
            
            # create subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=['Ground Truth', 'Prediction'],
                horizontal_spacing=0.05
            )
            
            # downsample
            H, W, D = pred_mag.shape
            step = max(1, H // args.vis_resolution)
            x, y, z = np.mgrid[0:H:step, 0:W:step, 0:D:step]
            
            # add GT isosurface
            fig.add_trace(
                go.Isosurface(
                    x=x.flatten(),
                    y=y.flatten(),
                    z=z.flatten(),
                    value=true_mag[::step, ::step, ::step].flatten(),
                    isomin=args.vis_isomin,
                    isomax=args.vis_isomax,
                    showscale=False,
                    colorscale='Blues'
                ),
                row=1, col=1
            )
            
            # add prediction isosurface
            fig.add_trace(
                go.Isosurface(
                    x=x.flatten(),
                    y=y.flatten(),
                    z=z.flatten(),
                    value=pred_mag[::step, ::step, ::step].flatten(),
                    isomin=args.vis_isomin,
                    isomax=args.vis_isomax,
                    showscale=True,
                    colorscale='Reds'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f'Velocity Field Comparison - Epoch {epoch}',
                height=600
            )
            
            comparison_path = os.path.join(log_path, f'velocity_comparison_epoch_{epoch}.png')
            comparison_html = comparison_path.replace('.png', '.html')
            
            print(f"   save comparison HTML: {comparison_html}")
            fig.write_html(comparison_html)
            
            try:
                print(f"   save comparison PNG: {comparison_path}")
                fig.write_image(comparison_path)
                print(f"[OK] [velocity field comparison] comparison PNG saved: {comparison_path}")
            except Exception as png_error:
                print(f"[WARN]  [velocity field comparison] PNG save failed (HTML saved): {png_error}")
            
            print(f"[OK] [velocity field comparison] comparison HTML saved: {comparison_html}")
            
        except Exception as e:
            print(f"[X] [velocity field comparison] comparison visualization failed: {e}")
            import traceback
            print(f"   detailed error: {traceback.format_exc()}")

################################################################
# load data and dataloader
################################################################
print('args',args)

# print spectrum analysis config
if args.enable_spectrum:
    print(f"\n=== Energy spectrum configuration ===")
    print(f"Enable spectrum analysis: True")
    print(f"number of frequency shells: {args.spectrum_bins}")
    print(f"physical domain size: {args.physical_size}")
    print(f"velocity components: {args.velocity_channels}")
    print(f"Spectrum analysis will run every 100 epochs and the final epoch")
    print("=" * 30)
else:
    print(f"\nSpectrum analysis: disabled (enable with --enable_spectrum)")

# print 3D visualization config
if args.enable_3d_vis:
    print(f"\n=== 3D visualization ===")
    print(f"Enable 3D visualization: True")
    print(f"visualization resolution: {args.vis_resolution}")
    print(f"isosurface range: [{args.vis_isomin}, {args.vis_isomax}]")
    print(f"3D visualization will run every 100 epochs and the final epoch")
    print(f"PLOTLY_AVAILABLE: {PLOTLY_AVAILABLE}")
    if PLOTLY_AVAILABLE:
        print("[OK] Plotly ready")
    else:
        print("[X] Plotly not installed, Please run: pip install plotly kaleido")
    print("=" * 30)
else:
    print(f"\n3D visualization: disabled (enable with --enable_3d_vis)")

train_dataset = TemporalDataset3D(args.train_path, n_train=args.ntrain, res=args.res, t_in=args.T_in, t_ar=args.T_ar, normalize=True, train=True)
test_dataset = TemporalDataset3D(args.test_path, res=args.res, t_in=args.T_in, t_ar=args.T_ar, normalize=True, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
ntrain, ntest = len(train_dataset), len(test_dataset)
print('Train num {} test num {}'.format(len(train_dataset), len(test_dataset)))
################################################################
# load model
################################################################
if args.model == "FNO":
    model = FNO3d(args.modes, args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels).to(device)
elif args.model == "UNet":
    model = UNet(n_dim=2, in_channels=12, out_channels=args.T_bundle, in_shape=(64, 64), width=args.width, act = args.act).to(device)
elif args.model == 'DPOT':
    # select model according to PEFT method
    if args.peft_method == 'none':
        print(f"Using base DPOT model (no PEFT method)")
        model = DPOTNet3D(
            img_size=args.res,
            patch_size=args.patch_size,
            in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in,
            out_timesteps=args.T_bundle,
            out_channels=train_dataset.n_channels,
            normalize=args.normalize,
            embed_dim=args.width,
            modes=args.modes,
            depth=args.n_layers,
            n_blocks=args.n_blocks,
            mlp_ratio=args.mlp_ratio,
            act=args.act,
            n_cls=args.n_class
        ).to(device)
    elif args.peft_method == 'lora':
        print(f"Using DPOT with LoRA fine-tuning (rank={args.peft_dim}, alpha={args.peft_scale})")
        model = DPOTNet3D_Lora(
            img_size=args.res,
            patch_size=args.patch_size,
            in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in,
            out_timesteps=args.T_bundle,
            out_channels=train_dataset.n_channels,
            normalize=args.normalize,
            embed_dim=args.width,
            modes=args.modes,
            depth=args.n_layers,
            n_blocks=args.n_blocks,
            mlp_ratio=args.mlp_ratio,
            act=args.act,
            n_cls=args.n_class,
            lora_r=args.peft_dim,
            lora_alpha=args.peft_scale
        ).to(device)
    elif args.peft_method == 'adapter':
        adapter_type = getattr(args, 'adapter_type', 'original')

        if adapter_type == 'chebykan':
            print(f"Using DPOT with ChebyKAN Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale}, cheby_degree={args.cheby_degree})")
            model = DPOTNet3D_AdapterChebyKAN(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale, cheby_degree=args.cheby_degree
            ).to(device)
        elif adapter_type == 'fadapter':
            print(f"Using DPOT with fAdapter fine-tuning (base_dim={args.peft_dim}, scale={args.peft_scale}, power={args.power}, bands={args.num_bands})")
            model = DPOTNet3D_fAdapter(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale, power=args.power,
                num_bands=args.num_bands, min_dim_factor=args.min_dim_factor, max_dim_factor=args.max_dim_factor
            ).to(device)
        # --- Step 3: Instantiate FourierKAN Adapter Model (remove specific args) ---
        elif adapter_type == 'fourierkan':
            print(f"Using DPOT with FourierKAN Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_AdapterFourierKAN(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale
                # Removed: gridsize, addbias, smooth_initialization, share_adapters, sparse_adapter_layers
                # The model will use its internal defaults for these
            ).to(device)
        elif adapter_type == 'waveact':
            print(f"Using DPOT with WaveAct Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_AdapterWaveAct(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale
            ).to(device)
        elif adapter_type == 'finverse':
            print(f"Using DPOT with FInverse Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_FInverseAdapter(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale
            ).to(device)
        elif adapter_type == 'film':
            print(f"Using DPOT with FiLM Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_FilmAdapter(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale
            ).to(device)
        else:  # Use original Adapter
            print(f"Using DPOT with original Adapter fine-tuning (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_Adapter(
                img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
                in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
                normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
                n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
                adapter_dim=args.peft_dim, adapter_scale=args.peft_scale
            ).to(device)
    elif args.peft_method == 'adalora':
        print(f"Using DPOT with AdaLoRA fine-tuning (init_r={args.peft_dim}, target_r={args.adalora_target_r}, alpha={args.peft_scale})")
        model = DPOTNet3D_AdaLora(
            img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
            normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
            n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
            lora_r=args.peft_dim, lora_alpha=args.peft_scale, target_r=args.adalora_target_r,
            total_step=args.epochs * len(train_loader), tinit=args.adalora_tinit, tfinal=args.adalora_tfinal
        ).to(device)
        print("Initializing AdaLora rank allocator...")
        model.initialize_rank_allocator(total_step=args.epochs * len(train_loader))
    elif args.peft_method == 'hydralora':
        print(f"Using DPOT with HydraLora fine-tuning (rank={args.peft_dim}, alpha={args.peft_scale}, experts={args.expert_num})")
        model = DPOTNet3D_HydraLora(
            img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
            normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
            n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
            lora_r=args.peft_dim, lora_alpha=args.peft_scale, expert_num=args.expert_num
        ).to(device)
    elif args.peft_method == 'randlora':
        print(f"Using DPOT with RandLoRA fine-tuning (rank={args.peft_dim}, alpha={args.peft_scale}, proj_factor={args.proj_factor})")
        model = DPOTNet3D_RandLora(
            img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
            normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
            n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
            lora_r=args.peft_dim, lora_alpha=args.peft_scale, proj_factor=args.proj_factor
        ).to(device)
    elif args.peft_method == 'svft':
        print(f"Using DPOT with SVFT fine-tuning (rank={args.svft_r}, alpha={args.svft_alpha}, off_diag={args.off_diag}, pattern={args.pattern})")
        model = DPOTNet3D_SVFT(
            img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
            normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
            n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
            svft_r=args.svft_r, svft_alpha=args.svft_alpha, off_diag=args.off_diag, pattern=args.pattern
        ).to(device)
    elif args.peft_method == 'prompt_tuning':
        print(f"Using DPOT with Prompt Tuning (dim={args.prompt_dim}, pos={args.prompt_pos})")
        from models.dpot3d_Prompt_Tuning import DPOTNet3D as DPOTNet3D_PromptTuning
        model = DPOTNet3D_PromptTuning(
            img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels,
            in_timesteps=args.T_in, out_timesteps=args.T_bundle, out_channels=train_dataset.n_channels,
            normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers,
            n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, act=args.act, n_cls=args.n_class,
            prompt_dim=args.prompt_dim, prompt_pos=args.prompt_pos
        ).to(device)
    else:
        raise ValueError(f"Unknown PEFT method: {args.peft_method}")
else:
    raise NotImplementedError

# PEFT weight loading function
def load_peft_weights_from_checkpoint(model, state_dict, peft_method='lora'):
    """Load only PEFT-related weights"""
    model_dict = model.state_dict()
    peft_dict = {}
    loaded_param_count = 0
    total_param_values = 0

    print(f"Start loading PEFT ({peft_method}) ...")

    # Base model does not need special PEFT handling
    if peft_method == 'none':
        print("Load base model parameters without filtering PEFT weights")
        compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(compatible_state_dict)
        model.load_state_dict(model_dict)
        return model

    # define keyword list for filtering parameters
    peft_keywords = []
    if peft_method == 'lora' or peft_method == 'adalora':
        peft_keywords = ['lora']
    elif peft_method == 'adapter':
        adapter_type = getattr(args, 'adapter_type', 'original')
        if adapter_type == 'chebykan':
            peft_keywords = ['adapter', 'cheby']
        elif adapter_type == 'fadapter':
             peft_keywords = ['adapter', 'adapters_in', 'adapters_mid', 'adapters_out', 'band']
        elif adapter_type == 'fourierkan':
             peft_keywords = ['adapter', 'fourier', 'kan', 'coeff'] # Keywords for FourierKAN
        else: # original adapter
            peft_keywords = ['adapter']
    elif peft_method == 'hydralora':
        peft_keywords = ['lora', 'expert_weights']
    elif peft_method == 'randlora':
        peft_keywords = ['lora', 'randlora']
    elif peft_method == 'svft':
        peft_keywords = ['svft', 's_pre', 's_', 'gate']
    elif peft_method == 'prompt_tuning':
        peft_keywords = ['prompt']
    else:
        raise ValueError(f"Unknown PEFT method: {peft_method}")

    # filter parameters by keywords
    peft_dict = {k: v for k, v in state_dict.items()
                 if any(key in k.lower() for key in peft_keywords) and k in model_dict}

    if not peft_dict:
        print(f"Warning: no matching PEFT parameters found in checkpoint '{peft_method}' (adapter_type='{getattr(args, 'adapter_type', 'N/A')}') matching loadable parameters")
    else:
        print(f"Found {len(peft_dict)} PEFT parameters found, preparing to load...")

    # Update PEFT parameters in model
    model_dict.update(peft_dict)
    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        loaded_param_count = len(peft_dict)
        total_param_values = sum(v.numel() for v in peft_dict.values())

        # report possible missing PEFT parameters
        missing_peft_keys = [k for k in peft_dict if k in missing_keys]
        if missing_peft_keys:
            print(f"Warning: attempted PEFT parameters missing in model: {missing_peft_keys}")

        print(f"PEFT parameters loaded. Successfully loaded {loaded_param_count - len(missing_peft_keys)} / {loaded_param_count} PEFT parameters.")
        print(f"Total loaded {total_param_values:,} parameter values.")

    except Exception as e:
        print(f"Error while loading PEFT checkpoint: {str(e)}")

    return model


# --- Step 5: Update Parameter Freezing Logic ---
# Only train parameters for selected PEFT method, freeze others
for name, param in model.named_parameters():
    train_param = False # Default to False
    name_lower = name.lower()

    if args.peft_method == 'none':
        train_param = True
    elif args.peft_method == 'lora' or args.peft_method == 'adalora':
        if 'lora' in name_lower:
            train_param = True
    elif args.peft_method == 'adapter':
        adapter_type = getattr(args, 'adapter_type', 'original')
        adapter_keywords = []
        if adapter_type == 'chebykan':
            adapter_keywords = ['adapter', 'cheby']
        elif adapter_type == 'fadapter':
            adapter_keywords = ['adapter', 'adapters_in', 'adapters_mid', 'adapters_out', 'band']
        elif adapter_type == 'fourierkan':
            adapter_keywords = ['adapter', 'fourier', 'kan', 'coeff'] # Include KAN specific param names
        else: # original adapter
            adapter_keywords = ['adapter']
        if any(key in name_lower for key in adapter_keywords):
            train_param = True
    elif args.peft_method == 'hydralora':
        if any(key in name_lower for key in ['lora', 'expert_weights']):
             train_param = True
    elif args.peft_method == 'randlora':
        if any(key in name_lower for key in ['lora', 'randlora']):
             train_param = True
    elif args.peft_method == 'svft':
        if any(key in name_lower for key in ['svft', 's_pre', 's_', 'gate']):
             train_param = True
    elif args.peft_method == 'prompt_tuning':
        if 'prompt' in name_lower:
            train_param = True

    param.requires_grad = train_param # Set based on the flag

# print trainable parameter info
trainable_params = [p for p in model.parameters() if p.requires_grad]
total_trainable_params = sum(p.numel() for p in trainable_params)
total_params = sum(p.numel() for p in model.parameters())
print(f'Trainable parameter count: {total_trainable_params:,} / {total_params:,} ({total_trainable_params/total_params:.2%})')

#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(trainable_params, lr=args.lr, betas=(args.beta1, args.beta2), adam=True, debias=False, weight_decay=1e-4)
else:
    optimizer = Adam(trainable_params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.step_size, eta_min=0.) # Fixed: args.lr_step_size -> args.step_size
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.step_size * len(train_loader), mode='triangular2', cycle_momentum=False) # Fixed: args.lr_step_size -> args.step_size
elif args.lr_method == 'cosine':
    print('Using cosine annealing schedule')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )
else:
    raise NotImplementedError

# --- Step 7: Update Log Comment ---
comment = args.comment + f'_{args.peft_method}_dim{args.peft_dim}_scale{args.peft_scale}_{len(args.train_path)}_{ntrain}' # Fixed len(train_dataset) -> ntrain
if args.peft_method == 'adalora':
    comment += f'_target{args.adalora_target_r}'
elif args.peft_method == 'hydralora':
    comment += f'_experts{args.expert_num}'
elif args.peft_method == 'randlora':
    comment += f'_factor{args.proj_factor}'
elif args.peft_method == 'svft':
    comment += f'_r{args.svft_r}_alpha{args.svft_alpha}_diag{args.off_diag}_{args.pattern}'
elif args.peft_method == 'adapter':
    adapter_type = getattr(args, 'adapter_type', 'original')
    comment += f'_{adapter_type}'
    if adapter_type == 'chebykan':
        comment += f'_deg{args.cheby_degree}'
    elif adapter_type == 'fadapter':
        comment += f'_pow{args.power}_bands{args.num_bands}'
    elif adapter_type == 'fourierkan':
        comment += f'_dim{args.peft_dim}'
elif args.peft_method == 'prompt_tuning':
    comment += f'_dim{args.prompt_dim}_{args.prompt_pos}'

# add spectrum analysis flag
if args.enable_spectrum:
    comment += f'_spectrum_bins{args.spectrum_bins}'

log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0 else os.path.join('./logs', args.log_path + comment)
model_path = log_path + f'/model_{args.peft_method}.pth'
if not os.path.exists(log_path):
    os.makedirs(log_path)

# Output full path info
print(f"\n[PATH] [Path info]")
print(f"   Log path: {os.path.abspath(log_path)}")
print(f"   Model path: {os.path.abspath(model_path)}")
if args.enable_3d_vis:
    print(f"   3D visualizations will be saved to: {os.path.abspath(log_path)}/velocity_*.png|html")
if args.enable_spectrum:
    print(f"   Spectrum plots will be saved to: {os.path.abspath(log_path)}/spectrum_*.png")

if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
    log_file_path = log_path + f'/logs_{args.peft_method}.txt'
    params_file_path = log_path + f'/params_{args.peft_method}.json'
    try:
        fp = open(log_file_path, 'w+', buffering=1)
        # Redirect stdout to the log file
        sys.stdout = fp # Be careful with redirecting stdout globally
        # Save parameters
        with open(params_file_path, 'w') as params_fp:
            json.dump(vars(args), params_fp, indent=4)
        print(f"Log file: {log_file_path}")
        print(f"Params file: {params_file_path}")
    except Exception as e:
        print(f"Unable to open log or params file: {e}")
        fp = None
        writer = None # Disable writer if logging fails
else:
    writer = None
    fp = None

# Log model structure and parameter count to console and potentially file
print(model)
count_parameters(model)


if args.resume_path:
    print('Loading base model from {}'.format(args.resume_path))
    try:
        checkpoint = torch.load(args.resume_path, map_location='cuda:{}'.format(args.gpu))
    except Exception as e:
        print(f"Unable to load base model checkpoint: {e}")
        checkpoint = None # Ensure checkpoint is None if loading fails

    if checkpoint:
        # Determine PEFT path
        peft_path = args.peft_path if args.peft_path else args.resume_path
        peft_checkpoint = None
        if os.path.exists(peft_path):
             if peft_path != args.resume_path:
                 print('Loading PEFT weights from {}'.format(peft_path))
                 try:
                     peft_checkpoint = torch.load(peft_path, map_location='cuda:{}'.format(args.gpu))
                 except Exception as e:
                     print(f"Unable to load PEFT checkpoint: {e}")
                     peft_checkpoint = None
             else:
                 peft_checkpoint = checkpoint
        else:
             print(f"Warning: PEFT path {peft_path} does not exist.")


        if 'model' not in checkpoint:
            print("Error: base checkpoint missing 'model' key.")
        else:
            base_model_state_dict = checkpoint['model']

            if args.load_mode == 'base_only':
                print(f"Load base model components only: {args.load_components}")
                load_3d_components_from_2d(model, base_model_state_dict,
                                         components=args.load_components, strict=False)

            elif args.load_mode == 'peft_only':
                print(f"Load base model components: {args.load_components}")
                load_3d_components_from_2d(model, base_model_state_dict,
                                         components=args.load_components, strict=False)
                if peft_checkpoint and 'model' in peft_checkpoint:
                    print(f"load {args.peft_method} parameters from {peft_path}")
                    load_peft_weights_from_checkpoint(model, peft_checkpoint['model'], peft_method=args.peft_method)
                elif peft_checkpoint:
                     print("Warning: PEFT checkpoint missing 'model' key.")
                else:
                     print(": Unable to load PEFT checkpoint")


            elif args.load_mode == 'resume_training':
                print("Resume training: load full model state")
                # 1. load base model
                print(f"from{args.resume_path}Load base model components: {args.load_components}")
                load_3d_components_from_2d(model, base_model_state_dict,
                                         components=args.load_components, strict=False)

                # 2. load PEFT parameters
                if peft_checkpoint and 'model' in peft_checkpoint:
                     print(f"load {args.peft_method} parameters from {peft_path}")
                     load_peft_weights_from_checkpoint(model, peft_checkpoint['model'], peft_method=args.peft_method)

                     # 3. Restore optimizer state
                     if 'optimizer' in peft_checkpoint:
                         print("Restore optimizer state")
                         try:
                             optimizer.load_state_dict(peft_checkpoint['optimizer'])
                         except Exception as e:
                             print(f"Restore optimizer state failed: {e}")
                     else:
                          print("Warning: PEFT checkpoint missing optimizer state.")

                     # 4. Restore scheduler state
                     if 'scheduler' in peft_checkpoint:
                         print("Restore scheduler state")
                         try:
                             scheduler.load_state_dict(peft_checkpoint['scheduler'])
                         except Exception as e:
                              print(f"Failed to restore scheduler state: {e}")
                     else:
                         print("Warning: PEFT checkpoint missing scheduler state.")

                     # 5. Restore training epochs
                     if 'epoch' in peft_checkpoint:
                         start_epoch = peft_checkpoint['epoch'] + 1
                         print(f"continue training from epoch {start_epoch}")
                     else:
                         start_epoch = 0
                         print("Warning: PEFT checkpoint missing 'epoch'; starting from 0.")

                elif peft_checkpoint:
                     print("Warning: PEFT checkpoint missing 'model'; starting from 0.")
                     start_epoch = 0
                else:
                     print('Warning: PEFT checkpoint missing or failed to load; starting from 0.')
                     start_epoch = 0 # Ensure start_epoch is defined


            else: # Default behavior
                 print(f"Use default load mode, load base model components: {args.load_components}")
                 load_3d_components_from_2d(model, base_model_state_dict,
                                          components=args.load_components, strict=False)
    else:
        print("Unable to load base model; starting from scratch.")
        start_epoch = 0 # Ensure start_epoch is defined


    print("Model load complete")
else:
    print("Base model path not provided (--resume_path); starting from scratch.")
    start_epoch = 0 # Ensure start_epoch is defined

################################################################
# Main function for pretraining
################################################################
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')
iter = 0
for ep in tqdm(range(args.epochs), desc="Training"):
    model.train()

    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_step = 0
    train_l2_full = 0
    cls_total, cls_correct, cls_acc = 0, 0, 0.
    loss_previous = np.inf

    for batch_idx, (xx, yy, msk) in enumerate(tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}")):
        t_load += default_timer() - t_1
        t_1 = default_timer()

        loss, cls_loss = 0. , 0.
        xx = xx.to(device)  ## B, n, n, T_in, C
        yy = yy.to(device)  ## B, n, n, T_ar, C
        msk = msk.to(device)
        # cls = cls.to(device)


        ## auto-regressive training loop, support 1. noise injection, 2. long rollout backward, 3. temporal bundling prediction
        for t in range(0, yy.shape[-2], args.T_bundle):
            y = yy[..., t:t + args.T_bundle, :]

            ### auto-regressive training
            xx = xx + args.noise_scale *torch.sum(xx**2, dim=(1,2,3),keepdim=True)**0.5 * torch.randn_like(xx)
            im = model(xx)
            loss += myloss(im, y, mask=msk)


            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

        train_l2_step += loss.item()
        l2_full = myloss(pred, yy, mask=msk)
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        total_loss = loss  # + 1.0 * cls_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        # For AdaLoRA, update importance scores and rank allocation
        if args.peft_method == 'adalora':
            # Compute current global step
            global_step = ep * len(train_loader) + batch_idx
            
            # Update importance scores and rank allocation at configured frequency
            if batch_idx % args.adalora_update_freq == 0:
                updated, budget = model.update_adalora_ranks(global_step)
                if updated and args.use_writer:
                    print(f"AdaLoRA rank update: current budget {budget}")
                    writer.add_scalar("adalora_budget", budget, global_step)

        train_l2_step_avg, train_l2_full_avg = train_l2_step / ntrain / (yy.shape[-2] / args.T_bundle), train_l2_full / ntrain
        iter +=1
        if args.use_writer:
            writer.add_scalar("train_loss_step", loss.item()/(xx.shape[0] * yy.shape[-2] / args.T_bundle), iter)
            writer.add_scalar("train_loss_full", l2_full / xx.shape[0], iter)

            ## reset model
            if loss.item() > 10 * loss_previous : # or (ep > 50 and l2_full / xx.shape[0] > 0.9):
                print('loss explodes, loading model from previous epoch')
                checkpoint = torch.load(model_path,map_location='cuda:{}'.format(args.gpu))
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint["optimizer"])
                loss_previous = loss.item()

        t_train += default_timer() -  t_1
        t_1 = default_timer()


    # Run test evaluation every 100 epochs or final epoch
    test_l2_fulls, test_l2_steps = [], []
    if (ep + 1) % 100 == 0 or (ep + 1) == args.epochs:
        with torch.no_grad():
            model.eval()

            test_l2_full, test_l2_step = 0, 0
            
            # Initialize spectrum statistics
            k_bins = None
            spec_pred_sum = None
            spec_true_sum = None
            spectrum_samples = 0
            
            # 3D visualization flag - only on first batch
            first_batch_processed = False
            
            for xx, yy, msk in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                msk = msk.to(device)

                for t in range(0, yy.shape[-2], args.T_bundle):
                    y = yy[..., t:t + args.T_bundle, :]
                    im = model(xx)
                    loss += myloss(im, y, mask=msk)

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)

                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred, yy, mask=msk)
                
                # Spectrum analysis - handle last timestep predictions and targets
                if args.enable_spectrum and pred.shape[-1] >= args.velocity_channels:
                    # Data format: [B, H, W, D, T, C]
                    # For ns3d_pdb_M1_turb: C=5 (u,v,w,p,other); use first 3 channels as velocity
                    
                    # Use last timestep and first velocity_channels channels
                    u_pred = pred[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)  # [B, velocity_channels, H, W, D]
                    u_true = yy[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)    # [B, velocity_channels, H, W, D]
                    
                    # samples
                    for b in range(u_pred.shape[0]):
                        try:
                            k, spec_p = energy_spectrum_3d(u_pred[b], bins=args.spectrum_bins, physical_size=args.physical_size)
                            _, spec_t = energy_spectrum_3d(u_true[b], bins=args.spectrum_bins, physical_size=args.physical_size)
                            
                            if k_bins is None:
                                k_bins = k
                                spec_pred_sum = torch.zeros_like(spec_p)
                                spec_true_sum = torch.zeros_like(spec_t)
                            
                            spec_pred_sum += spec_p
                            spec_true_sum += spec_t
                            spectrum_samples += 1
                        except Exception as e:
                            if spectrum_samples == 0:  # Only print on first error
                                print(f"Spectrum computation warning: {e}")
                            continue

                    # 3D visualization - handle only first batch to avoid repetition
                    if args.enable_3d_vis and not first_batch_processed:  # visualize only on first batch
                        print(f"\n[RUN] [] 3D visualization - Epoch {ep + 1}")
                        print(f"   data shape: u_pred={u_pred.shape}, u_true={u_true.shape}")
                        create_velocity_comparison(u_pred, u_true, ep + 1, log_path, args)
                        first_batch_processed = True  # batch
                    elif args.enable_3d_vis and first_batch_processed:
                        print(f"   [] 3D visualization (batch)")
                    else:
                        print(f"   [] 3D visualizationnot enabled")

            test_l2_step_avg, test_l2_full_avg = test_l2_step / ntest / (yy.shape[-2] / args.T_bundle), test_l2_full / ntest
            test_l2_steps.append(test_l2_step_avg)
            test_l2_fulls.append(test_l2_full_avg)
            
            # 
            spectrum_results = {}
            if args.enable_spectrum and k_bins is not None and spectrum_samples > 0:
                # average spectrum
                spec_pred_avg = spec_pred_sum / spectrum_samples
                spec_true_avg = spec_true_sum / spectrum_samples
                
                # spectral error metric - L2
                epsilon = 1e-12
                spec_true_safe = spec_true_avg.clamp(min=epsilon)
                
                # L2 relative error in spectrum
                l2_rel_error = torch.sqrt(torch.mean(((spec_pred_avg - spec_true_avg) / spec_true_safe) ** 2)).item()
                
                # L2 logarithmic error (L2)
                spec_pred_safe = spec_pred_avg.clamp(min=epsilon)
                l2_log_error = torch.sqrt(torch.mean((torch.log10(spec_pred_safe) - torch.log10(spec_true_safe))**2)).item()
                
                # Relative error in total energy content
                total_energy_pred = torch.trapz(spec_pred_avg, k_bins).item()
                total_energy_true = torch.trapz(spec_true_avg, k_bins).item()
                energy_rel_error = abs(total_energy_pred - total_energy_true) / total_energy_true * 100
                
                spectrum_results = {
                    'l2_rel_error': l2_rel_error,
                    'l2_log_error': l2_log_error,
                    'energy_rel_error': energy_rel_error,
                    'samples_count': spectrum_samples
                }
                
                print(f"Energy spectrum analysis results (based on{spectrum_samples}samples):")
                print(f"  spectrum L2 relative error: {l2_rel_error:.6f}")
                print(f"  spectrum L2 log error: {l2_log_error:.6f}")
                print(f"  total energy relative error: {energy_rel_error:.2f}%")
                
                # Spectrum comparison plot100epochepoch
                if (ep + 1) % 100 == 0 or (ep + 1) == args.epochs:
                    spectrum_plot_path = os.path.join(log_path, f'spectrum_epoch_{ep+1}.png')
                    
                    plt.figure(figsize=(12, 8))
                    
                    # main plot-
                    plt.subplot(2, 1, 1)
                    plt.loglog(k_bins.numpy(), spec_true_avg.numpy(), 'b-', linewidth=2, label='DNS (Ground Truth)')
                    plt.loglog(k_bins.numpy(), spec_pred_avg.numpy(), 'r--', linewidth=2, label=f'Model ({args.peft_method})')
                    
                    # -5/3slope reference line
                    k_ref = k_bins[k_bins > 0]
                    if len(k_ref) > 10:
                        k_mid = k_ref[len(k_ref)//4:3*len(k_ref)//4]
                        E_ref = spec_true_avg[k_bins > 0][len(k_ref)//4:3*len(k_ref)//4]
                        if len(k_mid) > 0 and len(E_ref) > 0:
                            E_scale = E_ref[len(E_ref)//2] * (k_mid[len(k_mid)//2] ** (5/3))
                            E_53_ref = E_scale * (k_mid ** (-5/3))
                            plt.loglog(k_mid.numpy(), E_53_ref.numpy(), 'k:', alpha=0.7, linewidth=1, label=r'$k^{-5/3}$ (Kolmogorov)')
                    
                    plt.xlabel(r'Wave number $k$', fontsize=12)
                    plt.ylabel(r'Energy spectrum $E(k)$', fontsize=12)
                    plt.title(f'3D Turbulence Energy Spectrum - Epoch {ep+1}\n{args.peft_method} (dim={args.peft_dim}, scale={args.peft_scale})', fontsize=14)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    
                    # subplot
                    plt.subplot(2, 1, 2)
                    rel_error = (spec_pred_avg - spec_true_avg) / spec_true_safe * 100
                    plt.semilogx(k_bins.numpy(), rel_error.numpy(), 'g-', linewidth=2)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.xlabel(r'Wave number $k$', fontsize=12)
                    plt.ylabel('Relative Error (%)', fontsize=12)
                    plt.title('Spectrum Relative Error', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(spectrum_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Spectrum comparison plot saved to: {spectrum_plot_path}")
            
            # TensorBoard
            if args.use_writer:
                writer.add_scalar("test_loss_step_{}".format(args.test_path), test_l2_step_avg, ep)
                writer.add_scalar("test_loss_full_{}".format(args.test_path), test_l2_full_avg, ep)
                
                # TensorBoard
                if args.enable_spectrum and spectrum_results:
                    writer.add_scalar("spectrum_L2_rel_error", spectrum_results['l2_rel_error'], ep)
                    writer.add_scalar("spectrum_L2_log_error", spectrum_results['l2_log_error'], ep)
                    writer.add_scalar("spectrum_energy_rel_error", spectrum_results['energy_rel_error'], ep)

            t_test = default_timer() - t_1
            spectrum_info = ""
            if args.enable_spectrum and spectrum_results:
                spectrum_info = f", spectrum L2 rel {spectrum_results['l2_rel_error']:.5f}, energy rel {spectrum_results['energy_rel_error']:.2f}%"
            print(f"Test evaluation at epoch {ep+1}: test l2 step {test_l2_step_avg:.5f}, test l2 full {test_l2_full_avg:.5f}{spectrum_info}, test time {t_test:.5f}s")

    # checkpointPEFT
    if args.use_writer:
        # PEFT
        if args.peft_method == 'lora':
            peft_keyword = 'lora'
            peft_state_dict = {k: v for k, v in model.state_dict().items() if peft_keyword in k}
        elif args.peft_method == 'adapter':
            if args.adapter_type == 'chebykan':
                peft_state_dict = {k: v for k, v in model.state_dict().items() 
                                  if 'adapter' in k or 'cheby' in k.lower()}
            elif args.adapter_type == 'fadapter':
                peft_state_dict = {k: v for k, v in model.state_dict().items() 
                                  if 'adapter' in k or 'adapters_in' in k.lower() or 'adapters_mid' in k.lower() or 'adapters_out' in k.lower() or 'band' in k.lower()}
            elif args.adapter_type == 'fourierkan':
                peft_state_dict = {k: v for k, v in model.state_dict().items() 
                                  if 'adapter' in k or 'fourier' in k.lower() or 'kan' in k.lower() or 'coeff' in k.lower()}
            else:
                peft_state_dict = {k: v for k, v in model.state_dict().items() if 'adapter' in k}
        elif args.peft_method == 'adalora':
            peft_keyword = 'lora'
            peft_state_dict = {k: v for k, v in model.state_dict().items() if peft_keyword in k}
        elif args.peft_method == 'hydralora':
            # HydraLoRAneed to saveloraexpert_weights
            peft_state_dict = {k: v for k, v in model.state_dict().items() 
                            if 'lora' in k or 'expert_weights' in k}
        elif args.peft_method == 'randlora':
            peft_state_dict = {k: v for k, v in model.state_dict().items() 
                              if 'lora' in k.lower() or 'randlora' in k.lower()}
        elif args.peft_method == 'prompt_tuning':  # handle prompt method
            peft_state_dict = {k: v for k, v in model.state_dict().items() 
                              if 'prompt' in k.lower()}
        else:
            # By default save entire model
            peft_state_dict = model.state_dict()

        torch.save({
        'args': args, 
        'model': peft_state_dict, 
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': ep}, model_path)

    t2 = default_timer()
    lr = optimizer.param_groups[0]['lr']
    print('epoch {}, time {:.5f}, lr {:.2e}, train l2 step {:.5f} train l2 full {:.5f}, cls acc {:.5f}, time train avg {:.5f} load avg {:.5f}'.format(
        ep, t2 - t1, lr, train_l2_step_avg, train_l2_full_avg, cls_acc, t_train / len(train_loader), t_load / len(train_loader)))



# Finalize AdaLoRA ranks at end of training
if args.peft_method == 'adalora':
    print("Finalizing AdaLora rank allocation...")
    model.update_adalora_ranks(args.epochs * len(train_loader) - 1, force_update=True)

# Add FLOPS evaluation
print("----- Model performance analysis -----")
# Create random tensor matching input format
input_tensor = torch.randn(1, args.res, args.res, args.res, args.T_in, train_dataset.n_channels).to(device)
# Compute FLOPS
flops = FlopCountAnalysis(model, input_tensor)
# Output FLOPS info
print(f"Model FLOPS: {flops.total() / 1e9:.4f} B")
#test

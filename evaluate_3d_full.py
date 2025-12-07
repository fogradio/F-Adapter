import sys
import os
sys.path.append(['.','./../'])
os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

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

from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint,load_3d_components_from_2d
from utils.criterion import SimpleLpLoss
from utils.griddataset import TemporalDataset3D
from models.fno import FNO2d, FNO3d
# PEFT
from models.dpot3d import DPOTNet3D as DPOTNet3D_Base  # 
from models.dpot3d_Lora import DPOTNet3D as DPOTNet3D_Lora
from models.dpot3d_adapter import DPOTNet3D as DPOTNet3D_Adapter
from models.dpot3d_fadapter import DPOTNet3D as DPOTNet3D_fAdapter
from models.dpot3d_adapter_waveact import DPOTNet3D as DPOTNet3D_AdapterWaveAct
from models.dpot3d_finverse_adapter import DPOTNet3D as DPOTNet3D_FInverseAdapter
from models.dpot3d_Film_adapter import DPOTNet3D as DPOTNet3D_FilmAdapter
from models.dpot3d_adapter_chebyKAN import DPOTNet3D as DPOTNet3D_AdapterChebyKAN
from models.dpot3d_adapter_fourierkan import DPOTNet3D as DPOTNet3D_AdapterFourierKAN
from models.dpot3d_AdaloraLayer import DPOTNet3D_AdaLora
from models.dpot3d_HydraLoRA import DPOTNet3D_HydraLora
from models.dpot3d_RandLora import DPOTNet3D_RandLora
from models.dpot3d_SVFT import DPOTNet3D_SVFT
from models.dpot3d_Prompt_Tuning import DPOTNet3D as DPOTNet3D_PromptTuning
from fvcore.nn import FlopCountAnalysis

################################################################
# configs
################################################################

parser = argparse.ArgumentParser(description='Evaluation for 3D models with PEFT methods')

### currently no influence
parser.add_argument('--model', type=str, default='DPOT')
parser.add_argument('--dataset',type=str, default='ns3d')

parser.add_argument('--train_path', type=str, default='ns3d_pdb_M1_turb') 
parser.add_argument('--test_path',type=str, default='ns3d_pdb_M1_turb')
parser.add_argument('--resume_path',type=str, default='/root/finetune_checkpoints/model_H.pth')
parser.add_argument('--ntrain', type=int, default=540)
parser.add_argument('--ntest', type=int, default=60)
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=False)

parser.add_argument('--res', type=int, default=64)
parser.add_argument('--noise_scale',type=float, default=0.0)

### shared params
parser.add_argument('--width', type=int, default=2048)
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
parser.add_argument('--out_layer_dim', type=int, default=32)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='')

parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=12)
parser.add_argument('--load_components',nargs='+', type=str, default=['blocks'])

# PEFT
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
                  help='parameter-efficient finetuning method')
parser.add_argument('--peft_dim', type=int, default=8, 
                  help='PEFT')
parser.add_argument('--peft_scale', type=float, default=1.0,
                  help='PEFT')

# AdaLoRA specific arguments
parser.add_argument('--adalora_target_r', type=int, default=8,
                  help='target rank for AdaLoRA')

# HydraLoRA specific arguments
parser.add_argument('--expert_num', type=int, default=4,
                  help='HydraLoRA')

# RandLoRA specific arguments
parser.add_argument('--proj_factor', type=int, default=4,
                  help='RandLoRA')

# SVFT
parser.add_argument('--svft_r', type=int, default=32,
                  help='rank for SVFT')
parser.add_argument('--svft_alpha', type=float, default=1.0,
                  help='scale for SVFT')
parser.add_argument('--off_diag', type=int, default=2,
                  help='off-diagonal bandwidth for SVFT')
parser.add_argument('--pattern', type=str, default='banded',
                  help='SVFT')

# Adapter
parser.add_argument('--adapter_type', type=str, default='original',
                    choices=['original', 'chebykan', 'fadapter', 'fourierkan', 'waveact', 'finverse', 'film'],
                    help='Adapter')
parser.add_argument('--cheby_degree', type=int, default=8,
                    help='ChebyKAN')
# fAdapter 
parser.add_argument('--power', type=float, default=2.0,
                    help='fAdapter')
parser.add_argument('--num_bands', type=int, default=4,
                    help='fAdapter')
parser.add_argument('--min_dim_factor', type=float, default=0.5,
                    help='fAdapter (peft_dim)')
parser.add_argument('--max_dim_factor', type=float, default=2.0,
                    help='fAdapter (peft_dim)')

# Prompt Tuning
parser.add_argument('--prompt_dim', type=int, default=32,
                  help='prompt vector dimension')
parser.add_argument('--prompt_pos', type=str, default='both',
                  choices=['pre', 'post', 'mid', 'both'],
                  help='Prompt')

parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)

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
parser.add_argument('--enable_3d_vis', action='store_true', default=False,
                  help='whether to enable 3D velocity visualization')
parser.add_argument('--vis_resolution', type=int, default=64,
                  help='3D visualization resolution (for downsampling)')
parser.add_argument('--vis_isomin', type=float, default=0.2,
                  help='isosurface minimum')
parser.add_argument('--vis_isomax', type=float, default=0.8,
                  help='isosurface maximum')

args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu))

print(f"Current working directory: {os.getcwd()}")

# print 3D visualization config
if args.enable_3d_vis:
    print(f"\n=== 3D visualization ===")
    print(f"Enable 3D visualization: True")
    print(f"visualization resolution: {args.vis_resolution}")
    print(f"isosurface range: [{args.vis_isomin}, {args.vis_isomax}]")
    print(f"PLOTLY_AVAILABLE: {PLOTLY_AVAILABLE}")
    if PLOTLY_AVAILABLE:
        print("[OK] Plotly ready")
    else:
        print("[X] Plotly not installed, Please run: pip install plotly kaleido")
    print("=" * 30)
else:
    print(f"\n3D visualization: disabled (enable with --enable_3d_vis)")

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

################################################################
# 3D visualization
################################################################

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

def create_velocity_comparison(pred_velocity, true_velocity, save_prefix, log_path, args):
    """3D velocity field comparison (prediction vs ground truth) visualization"""
    print(f"\n[INFO] [velocity field comparison] start creating 3Dvelocity field")
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
    pred_save_path = os.path.join(log_path, f'{save_prefix}_pred.png')
    print(f"   [PLOT] create prediction velocity visualization...")
    create_3d_velocity_visualization(
        pred_sample, 
        f'Predicted Velocity Field\n{args.peft_method} (dim={args.peft_dim})',
        pred_save_path,
        args,
        isomin=args.vis_isomin,
        isomax=args.vis_isomax,
        resolution=args.vis_resolution
    )
    
    # create ground truth visualization
    print(f"   [PLOT] create ground truth velocity visualization...")
    true_save_path = os.path.join(log_path, f'{save_prefix}_ground_truth.png')
    create_3d_velocity_visualization(
        true_sample,
        f'Ground Truth Velocity Field',
        true_save_path, 
        args,
        isomin=args.vis_isomin,
        isomax=args.vis_isomax,
        resolution=args.vis_resolution
    )
        
    # create side-by-side comparison
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
            title=f'Velocity Field Comparison\n{args.peft_method} (dim={args.peft_dim})',
            height=600
        )
        
        comparison_path = os.path.join(log_path, f'{save_prefix}_comparison.png')
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
# load data and dataloader (mirror the training scriptload)
################################################################
print('args',args)

# [TOOL] data normalizationensure
print("[TOOL] Use the same data normalization strategy as training...")
train_dataset = TemporalDataset3D(args.train_path, n_train=args.ntrain, res=args.res, t_in=args.T_in, t_ar=args.T_ar, normalize=True, train=True)

# Check and print normalization statistics
if hasattr(train_dataset, 'mean_') and hasattr(train_dataset, 'std_'):
    print(f"Training datasetmean: {train_dataset.mean_}")
    print(f"Training datasetstd: {train_dataset.std_}")
else:
    print("Training datasetNormalization stats not visible, but normalize=True enabled")

# Test dataset
test_dataset = TemporalDataset3D(args.test_path, res=args.res, t_in=args.T_in, t_ar=args.T_ar, normalize=True, train=False)

# Check and print normalization statistics
if hasattr(test_dataset, 'mean_') and hasattr(test_dataset, 'std_'):
    print(f"Test datasetmean: {test_dataset.mean_}")
    print(f"Test datasetstd: {test_dataset.std_}")
else:
    print("Test datasetNormalization stats not visible, but normalize=True enabled")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
ntrain, ntest = len(train_dataset), len(test_dataset)
print('train samples {} test samples {}'.format(len(train_dataset), len(test_dataset)))

################################################################
# load model
################################################################
# select model according to PEFT method
if args.model == "FNO":
    model = FNO3d(args.modes, args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels).to(device)
elif args.model == "DPOT":
    # Choose model based on load mode and PEFT method
    if args.load_mode == 'base_only' or args.peft_method == 'none':
        print("Using base DPOT model (no PEFT)")
        model = DPOTNet3D_Base(
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
        print(f"Using DPOT with LoRA (rank={args.peft_dim}, alpha={args.peft_scale})")
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
            print(f"Using DPOT with ChebyKAN Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale}, cheby_degree={args.cheby_degree})")
            model = DPOTNet3D_AdapterChebyKAN(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale,
                cheby_degree=args.cheby_degree
            ).to(device)
        elif adapter_type == 'fadapter':
            print(f"Using DPOT with fAdapter (base_dim={args.peft_dim}, scale={args.peft_scale}, power={args.power}, bands={args.num_bands})")
            model = DPOTNet3D_fAdapter(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale,
                power=args.power,
                num_bands=args.num_bands,
                min_dim_factor=args.min_dim_factor,
                max_dim_factor=args.max_dim_factor
            ).to(device)
        elif adapter_type == 'fourierkan':
            print(f"Using DPOT with FourierKAN Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_AdapterFourierKAN(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale
            ).to(device)
        elif adapter_type == 'waveact':
            print(f"Using DPOT with WaveAct Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_AdapterWaveAct(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale
            ).to(device)
        elif adapter_type == 'finverse':
            print(f"Using DPOT with FInverse Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_FInverseAdapter(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale
            ).to(device)
        elif adapter_type == 'film':
            print(f"Using DPOT with FiLM Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_FilmAdapter(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale
            ).to(device)
        else:
            print(f"Using DPOT with original Adapter (bottleneck_dim={args.peft_dim}, scale={args.peft_scale})")
            model = DPOTNet3D_Adapter(
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
                adapter_dim=args.peft_dim,
                adapter_scale=args.peft_scale
            ).to(device)
    elif args.peft_method == 'adalora':
        print(f"Using DPOT with AdaLoRA (init_r={args.peft_dim}, target_r={args.adalora_target_r}, alpha={args.peft_scale})")
        model = DPOTNet3D_AdaLora(
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
            lora_alpha=args.peft_scale,
            target_r=args.adalora_target_r
        ).to(device)
    elif args.peft_method == 'hydralora':
        print(f"Using DPOT with HydraLoRA (rank={args.peft_dim}, alpha={args.peft_scale}, experts={args.expert_num})")
        model = DPOTNet3D_HydraLora(
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
            lora_alpha=args.peft_scale,
            expert_num=args.expert_num
        ).to(device)
    elif args.peft_method == 'randlora':
        print(f"Using DPOT with RandLoRA (rank={args.peft_dim}, alpha={args.peft_scale}, proj_factor={args.proj_factor})")
        model = DPOTNet3D_RandLora(
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
            lora_alpha=args.peft_scale,
            proj_factor=args.proj_factor
        ).to(device)
    elif args.peft_method == 'svft':
        print(f"Using DPOT with SVFT (rank={args.svft_r}, alpha={args.svft_alpha}, off_diag={args.off_diag}, pattern={args.pattern})")
        model = DPOTNet3D_SVFT(
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
            svft_r=args.svft_r,
            svft_alpha=args.svft_alpha,
            off_diag=args.off_diag,
            pattern=args.pattern
        ).to(device)
    elif args.peft_method == 'prompt_tuning':
        print(f"Using DPOT with Prompt Tuning (dim={args.prompt_dim}, pos={args.prompt_pos})")
        model = DPOTNet3D_PromptTuning(
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
            prompt_dim=args.prompt_dim,
            prompt_pos=args.prompt_pos
        ).to(device)
    else:
        raise ValueError(f"Unknown PEFT method: {args.peft_method}")
else:
    raise NotImplementedError

# PEFT keyword matching function
def get_peft_keywords(peft_method, args=None):
    """Return keyword list based on PEFT method"""
    peft_keywords = []
    
    if peft_method == 'lora' or peft_method == 'adalora':
        peft_keywords = ['lora']
    elif peft_method == 'adapter':
        adapter_type = getattr(args, 'adapter_type', 'original') if args else 'original'
        if adapter_type == 'chebykan':
            peft_keywords = ['adapter', 'cheby']
        elif adapter_type == 'fadapter':
             peft_keywords = ['adapter', 'adapters_in', 'adapters_mid', 'adapters_out', 'band']
        elif adapter_type == 'fourierkan':
             peft_keywords = ['adapter', 'fourier', 'kan', 'coeff']
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
        print(f"Warning: Unknown PEFT method {peft_method}")
        peft_keywords = [peft_method.lower()]
    
    return peft_keywords

# PEFT weight loading function
def load_peft_weights_from_checkpoint(model, state_dict, peft_method='lora', args=None):
    """Load only PEFT-related weights"""
    model_dict = model.state_dict()
    
    # Get PEFT keywords
    peft_keywords = get_peft_keywords(peft_method, args)
    print(f"PEFT '{peft_method}' : {peft_keywords}")
    
    # Filter PEFT-related parameters
    def is_peft_param(key):
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in peft_keywords)
    
    peft_dict = {k: v for k, v in state_dict.items() 
                if is_peft_param(k) and k in model_dict}
    
    if not peft_dict:
        print(f"[X] not found in checkpoint{peft_method}")
        print(f"Checkpoint10: {list(state_dict.keys())[:10]}")
        model_peft_keys = [k for k in model_dict.keys() if is_peft_param(k)]
        print(f"in model{peft_method}10: {model_peft_keys[:10]}")
        return model
    
    print(f"[OK] {len(peft_dict)}{peft_method}")
    
    # 
    mismatched_shapes = []
    for k, v in peft_dict.items():
        if k in model_dict and v.shape != model_dict[k].shape:
            mismatched_shapes.append((k, v.shape, model_dict[k].shape))
    
    if mismatched_shapes:
        print(f"[X] {len(mismatched_shapes)}parameter shape mismatches:")
        for name, ckpt_shape, model_shape in mismatched_shapes[:3]:
            print(f"  {name}: checkpoint{ckpt_shape} vs model{model_shape}")
        return model
    
    # Update PEFT parameters in modelstrict=Trueensure
    model_dict.update(peft_dict)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    
    # Check for missing or unexpected keys
    peft_missing = [k for k in missing_keys if is_peft_param(k)]
    peft_unexpected = [k for k in unexpected_keys if is_peft_param(k)]
    
    if peft_missing:
        print(f"[WARN]  {len(peft_missing)}PEFTload: {peft_missing[:5]}")
    if peft_unexpected:
        print(f"[WARN]  {len(peft_unexpected)}PEFT: {peft_unexpected[:5]}")
    
    if not peft_missing and not peft_unexpected:
        print(f"[OK] {peft_method}load")
    
    return model

# load
if args.resume_path:
    print('from{}load'.format(args.resume_path))
    checkpoint = torch.load(args.resume_path, map_location='cuda:{}'.format(args.gpu))
    
    # Determine PEFT path
    peft_path = args.peft_path if args.peft_path else args.resume_path
    if peft_path != args.resume_path:
        print('from{}loadPEFT'.format(peft_path))
        peft_checkpoint = torch.load(peft_path, map_location='cuda:{}'.format(args.gpu))
    else:
        peft_checkpoint = checkpoint
    
    if args.load_mode == 'base_only':
        # load - load
        print("load")
        load_model_from_checkpoint(model, checkpoint['model'])
        print("[OK] Model load complete")
    
    elif args.load_mode == 'peft_only':
        # [TOOL] load
        print(f"Load base model components: {args.load_components}")
        
        # Get PEFT keywords
        peft_keywords = get_peft_keywords(args.peft_method, args)
        print(f"PEFT: {peft_keywords}")
        
        # load_3d_components_from_2dPEFT
        load_3d_components_from_2d(
            model, 
            checkpoint['model'], 
            components=args.load_components, 
            strict=False,
            exclude_peft_keywords=peft_keywords
        )
        
        print(f"[OK] Model load complete{args.peft_method}")
        
        # loadcheckMLP
        base_weights_loaded = 0
        total_mlp_weights = 0
        for name, param in model.named_parameters():
            if 'mlp' in name and 'weight' in name and not any(keyword in name.lower() for keyword in peft_keywords):
                total_mlp_weights += 1
                if param.abs().sum() > 1e-8:  # 
                    base_weights_loaded += 1
        
        print(f" : {base_weights_loaded}/{total_mlp_weights} MLPload")
        if base_weights_loaded == 0 and total_mlp_weights > 0:
            print("[X] MLPload")
        elif base_weights_loaded > 0:
            print(f"[OK] load")
        
        # loadPEFT
        print(f"load {args.peft_method} parameters from {peft_path}...")
        
        if peft_checkpoint and 'model' in peft_checkpoint:
            # PEFT weight loading function
            model = load_peft_weights_from_checkpoint(
                model, 
                peft_checkpoint['model'], 
                peft_method=args.peft_method, 
                args=args
            )
            
            # PEFTload
            peft_keywords = get_peft_keywords(args.peft_method, args)
            peft_params = [(n, p) for n, p in model.named_parameters() 
                          if any(keyword in n.lower() for keyword in peft_keywords)]
            
            if peft_params:
                nonzero_count = sum(1 for n, p in peft_params if p.abs().sum() > 1e-8)
                total_norm = sum(p.norm().item() for n, p in peft_params if p.abs().sum() > 1e-8)
                
                print(f" {nonzero_count}/{len(peft_params)}{args.peft_method}")
                print(f" {args.peft_method}: {total_norm:.2f}")
                
                if nonzero_count > 0 and total_norm > 1e-3:
                    print(f"[OK] {args.peft_method}")
                else:
                    print(f"[X] {args.peft_method}load")
            else:
                print(f"[X] in model{args.peft_method}")
        else:
            print(f"[X] PEFT checkpoint'model'")
    
    else:
        # load
        print(f"Use default load mode, load base model components: {args.load_components}")
        load_3d_components_from_2d(model, checkpoint['model'], 
                                 components=args.load_components, strict=False)
    
    print("[OK] Model load complete")

print(model)
count_parameters(model)

# Add FLOPS evaluation
print("----- Model performance analysis -----")
# Create random tensor matching input format
input_tensor = torch.randn(1, args.res, args.res, args.res, args.T_in, train_dataset.n_channels).to(device)
# Compute FLOPS
flops = FlopCountAnalysis(model, input_tensor)
# Output FLOPS info
print(f"Model FLOPS: {flops.total() / 1e9:.4f} G")

################################################################
#  (mirror the training script)
################################################################
# [TOOL] fully consistent
myloss = SimpleLpLoss(size_average=False)
print(f"[TOOL] Loss function init: SimpleLpLoss(size_average=False) - ")
print(f"  Loss function type: {type(myloss)}")
print(f"  size_average: {myloss.size_average}")
print(f"  T_bundle: {args.T_bundle} (ensure)")

# Log file
comment = args.comment + f'_{args.peft_method}_dim{args.peft_dim}_scale{args.peft_scale}_{args.test_path}'
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0 else os.path.join('./logs', args.log_path + comment)

# ensureLog file
os.makedirs(log_path, exist_ok=True)

# Log file
log_file = os.path.join(log_path, f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
log_fp = open(log_file, 'w', buffering=1, encoding='utf-8')

# 
params_file = os.path.join(log_path, f'params_{args.peft_method}.json')
with open(params_file, 'w') as f:
    json.dump(vars(args), f, indent=4)

# 
class Logger:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()

# 
logger = Logger(log_fp)
sys.stdout = logger

if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
else:
    writer = None

print(f"=== Evaluation start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Log file: {log_file}")
print(f": {params_file}")
print(f"save path: {log_path}")

# Output full path info
print(f"\n[PATH] [Path info]")
print(f"   Log path: {os.path.abspath(log_path)}")
if args.enable_3d_vis:
    print(f"   3D visualizations will be saved to: {os.path.abspath(log_path)}/velocity_field_*.png|html")
if args.enable_spectrum:
    print(f"   Spectrum plots will be saved to: {os.path.abspath(log_path)}/spectrum_*.png")

# 
test_l2_fulls, test_l2_steps, time_test, total_steps = [], [], 0., 0

# Initialize spectrum statistics
k_bins = None
spec_pred_sum = None
spec_true_sum = None
spectrum_samples = 0

print("\n===  ===")
model.eval()
print(f"[OK] : {not model.training}")

# samples
print("===  (check) ===")
# [TOOL] check
print("[TOOL] check:")
print(f"  T_in: {args.T_in}")
print(f"  T_ar: {args.T_ar}")
print(f"  T_bundle: {args.T_bundle} (=1)")
print(f"  batch_size: {args.batch_size}")
print(f"  normalize: True ()")
print(f"  ntrain: {args.ntrain} (540)")

# ensureT_bundle1
if args.T_bundle != 1:
    print("[WARN] T_bundle1")
    print("   suggest --T_bundle 1 ")

with torch.no_grad():
    # batch
    first_batch = next(iter(test_loader))
    xx_diag, yy_diag, msk_diag = first_batch
    xx_diag = xx_diag.to(device)
    yy_diag = yy_diag.to(device)
    msk_diag = msk_diag.to(device)
    
    print(f"data shape: {xx_diag.shape}")
    print(f"data range: [{xx_diag.min().item():.6f}, {xx_diag.max().item():.6f}]")
    print(f"data range: [{yy_diag.min().item():.6f}, {yy_diag.max().item():.6f}]")
    
    # 
    pred_diag = model(xx_diag)
    print(f"pred output: {pred_diag.shape}")
    print(f"pred output: [{pred_diag.min().item():.6f}, {pred_diag.max().item():.6f}]")
    
    # [TOOL] exactly the same
    y_slice = yy_diag[..., 0:args.T_bundle, :]  # T_bundle
    loss_diag = myloss(pred_diag, y_slice, mask=msk_diag)
    # 
    loss_per_sample = loss_diag.item() / xx_diag.shape[0] / args.T_bundle
    print(f"samples (): {loss_per_sample:.6f}")
    
    # 
    if loss_per_sample > 1.0:  # 
        print("[X] ")
        print("   suggestcheck")
        print("   1. T_bundle")
        print("   2. data normalization")
        print("   3. load")
    else:
        print("[OK] reasonable range")

print("===  (mirror the training script) ===")
# [TOOL] fully consistent
with torch.no_grad():
    test_l2_full, test_l2_step = 0, 0
    batch_count = 0
    total_batches = len(test_loader)
    
    for xx, yy, msk in test_loader:
        batch_count += 1
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        msk = msk.to(device)

        # [TOOL] 
        for t in range(0, yy.shape[-2], args.T_bundle):
            y = yy[..., t:t + args.T_bundle, :]

            time_i = time.time()
            im = model(xx)
            time_test += time.time() - time_i

            # [TOOL] exactly the same
            loss += myloss(im, y, mask=msk)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)

            xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
            total_steps += xx.shape[0]

        # [TOOL] exactly the same
        test_l2_step += loss.item()
        test_l2_full += myloss(pred, yy, mask=msk)
        
        # Spectrum analysis - handle last timestep predictions and targets
        if args.enable_spectrum and pred.shape[-1] >= args.velocity_channels:
            # Data format: [B, H, W, D, T, C]
            # For ns3d_pdb_M1_turb: C=5 (u,v,w,p,other); use first 3 channels as velocity
            print(f"data shapecheck - pred: {pred.shape}, yy: {yy.shape}")
            print(f"use first{args.velocity_channels}velocity components")
            
            # Use last timestep and first velocity_channels channels
            u_pred = pred[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)  # [B, velocity_channels, H, W, D]
            u_true = yy[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)    # [B, velocity_channels, H, W, D]
            
            print(f"spectrum calculationinput shape - u_pred: {u_pred.shape}, u_true: {u_true.shape}")
            
            # samples
            for b in range(u_pred.shape[0]):
                try:
                    # data rangecheck
                    pred_range = (u_pred[b].min().item(), u_pred[b].max().item())
                    true_range = (u_true[b].min().item(), u_true[b].max().item())
                    
                    if spectrum_samples == 0:  # samples
                        print(f"velocity fielddata range - : [{pred_range[0]:.4f}, {pred_range[1]:.4f}], : [{true_range[0]:.4f}, {true_range[1]:.4f}]")
                    
                    k, spec_p = energy_spectrum_3d(u_pred[b], bins=args.spectrum_bins, physical_size=args.physical_size)
                    _, spec_t = energy_spectrum_3d(u_true[b], bins=args.spectrum_bins, physical_size=args.physical_size)
                    
                    if k_bins is None:
                        k_bins = k
                        spec_pred_sum = torch.zeros_like(spec_p)
                        spec_true_sum = torch.zeros_like(spec_t)
                        print(f"spectrum calculation - frequency range: [0, {k.max():.4f}], bins: {len(k)}")
                    
                    spec_pred_sum += spec_p
                    spec_true_sum += spec_t
                    spectrum_samples += 1
                except Exception as e:
                    print(f"Spectrum computation warning: {e}")
                    continue
        
        # 3D visualization - batch
        if args.enable_3d_vis and pred.shape[-1] >= args.velocity_channels and batch_count == total_batches:
            # Use last timestep and first velocity_channels channels
            u_pred_vis = pred[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)  # [B, velocity_channels, H, W, D]
            u_true_vis = yy[..., -1, :args.velocity_channels].permute(0, 4, 1, 2, 3)    # [B, velocity_channels, H, W, D]
            
            print(f"\n[RUN] [] 3D visualization (batch)")
            print(f"   batch: {batch_count}/{total_batches}")
            print(f"   data shape: u_pred={u_pred_vis.shape}, u_true={u_true_vis.shape}")
            create_velocity_comparison(u_pred_vis, u_true_vis, 'velocity_field', log_path, args)
        elif args.enable_3d_vis and batch_count < total_batches:
            print(f"   [] 3D visualization (batch {batch_count}/{total_batches})")
        elif not args.enable_3d_vis:
            if batch_count == 1:  # only notify on first batch
                print(f"   [] 3D visualizationnot enabled")

    # [TOOL] exactly the same
    test_l2_step_avg = test_l2_step / ntest / (yy.shape[-2] / args.T_bundle)
    test_l2_full_avg = test_l2_full / ntest
    test_l2_steps.append(test_l2_step_avg)
    test_l2_fulls.append(test_l2_full_avg.item())

print(f"[TOOL] Test error rate (): {test_l2_step_avg:.6f} (), {test_l2_full_avg:.6f} ()")
print(f" {time_test:.4f}s,  {total_steps},  {time_test/total_steps:.6f}s")

# [TOOL] Loss computation validation - ensurefully consistent
print("\n----- Loss computation validation () -----")
print(f": SimpleLpLoss(size_average={myloss.size_average}) - fully consistent")
print(f"T_bundle: {args.T_bundle} - ensure")
print(f"test samples: {ntest}")
print(f": {yy.shape[-2]}")
print(f": {yy.shape[-2] / args.T_bundle}")
print(f": {test_l2_step_avg:.6f}")
print(f": {test_l2_full_avg:.6f}")

# 
expected_loss_range = (0.0001, 1.0)  # 
if not (expected_loss_range[0] <= test_l2_step_avg <= expected_loss_range[1]):
    print(f"[WARN]  Notice: single-step loss {test_l2_step_avg:.6f}  {expected_loss_range}")
    print("   Possible reasons")
    print("   1. data normalization")
    print("   2. T_bundle") 
    print("   3. load")
    print("   4. ")
else:
    print(f"[OK] reasonable range")

# Energy spectrum analysis results
if args.enable_spectrum and k_bins is not None and spectrum_samples > 0:
    print(f"\n----- Energy spectrum analysis results (based on{spectrum_samples}samples) -----")
    
    # average spectrum
    spec_pred_avg = spec_pred_sum / spectrum_samples
    spec_true_avg = spec_true_sum / spectrum_samples
    
    # spectral error metric - L2
    # avoid divide-by-zeroepsilon
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
    
    print(f"spectrum L2 relative error: {l2_rel_error:.6f}")
    print(f"spectrum L2 log error: {l2_log_error:.6f}")
    print(f"total energy relative error: {energy_rel_error:.2f}%")
    
    # Spectrum comparison plot
    plt.figure(figsize=(10, 8))
    
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
            # 
            E_scale = E_ref[len(E_ref)//2] * (k_mid[len(k_mid)//2] ** (5/3))
            E_53_ref = E_scale * (k_mid ** (-5/3))
            plt.loglog(k_mid.numpy(), E_53_ref.numpy(), 'k:', alpha=0.7, linewidth=1, label=r'$k^{-5/3}$ (Kolmogorov)')
    
    plt.xlabel(r'Wave number $k$', fontsize=12)
    plt.ylabel(r'Energy spectrum $E(k)$', fontsize=12)
    plt.title(f'3D Turbulence Energy Spectrum Comparison\n{args.peft_method} (dim={args.peft_dim}, scale={args.peft_scale})', fontsize=14)
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
    
    # 
    spectrum_save_path = log_path if args.use_writer else './logs'
    if not os.path.exists(spectrum_save_path):
        os.makedirs(spectrum_save_path, exist_ok=True)
    
    # inference
    inference_folder = os.path.join(spectrum_save_path, f'spectrum_results_{args.peft_method}_dim{args.peft_dim}')
    if not os.path.exists(inference_folder):
        os.makedirs(inference_folder, exist_ok=True)
    
    # Spectrum comparison plot
    spectrum_plot_path = os.path.join(inference_folder, f'spectrum_comparison_{args.peft_method}_dim{args.peft_dim}.png')
    plt.savefig(spectrum_plot_path, dpi=300, bbox_inches='tight')
    print(f"Spectrum comparison plot saved to: {spectrum_plot_path}")
    
    # 
    spectrum_data = {
        'k_bins': k_bins.numpy().tolist(),
        'spectrum_dns': spec_true_avg.numpy().tolist(),
        'spectrum_model': spec_pred_avg.numpy().tolist(),
        'l2_rel_error': l2_rel_error,
        'l2_log_error': l2_log_error,
        'energy_rel_error': energy_rel_error,
        'samples_count': spectrum_samples,
        'peft_method': args.peft_method,
        'peft_dim': args.peft_dim,
        'peft_scale': args.peft_scale,
        'dataset': args.test_path,
        'resolution': args.res,
        'velocity_channels': args.velocity_channels
    }
    
    spectrum_data_path = os.path.join(inference_folder, f'spectrum_data_{args.peft_method}_dim{args.peft_dim}.json')
    with open(spectrum_data_path, 'w') as f:
        json.dump(spectrum_data, f, indent=4)
    print(f": {spectrum_data_path}")
    
    # 
    spectrum_arrays = {
        'k_bins': k_bins.numpy(),
        'spectrum_dns': spec_true_avg.numpy(),
        'spectrum_model': spec_pred_avg.numpy()
    }
    spectrum_npy_path = os.path.join(inference_folder, f'spectrum_arrays_{args.peft_method}_dim{args.peft_dim}.npz')
    np.savez(spectrum_npy_path, **spectrum_arrays)
    print(f"array saved to: {spectrum_npy_path}")
    
    print(f": {inference_folder}")
    
    # written to TensorBoard
    if writer is not None:
        writer.add_scalars('spectrum_error', {
            'L2_rel': l2_rel_error,
            'L2_log': l2_log_error,
            'Energy_rel_pct': energy_rel_error
        }, global_step=0)
        
        # TensorBoard
        writer.add_figure('spectrum_comparison', plt.gcf(), global_step=0)
        
        print("Energy spectrum analysis resultswritten to TensorBoard")
    
    plt.close()  # 
elif args.enable_spectrum:
    print("Warning: failed to compute spectrum")
else:
    print("Spectrum analysis disabled ( --enable_spectrum )")

# 
print(f"\n=== Evaluation end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"All results saved to: {log_path}")

# 
sys.stdout = logger.terminal
log_fp.close()

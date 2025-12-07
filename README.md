# (NeurIPS 2025) F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning

[![arXiv](https://img.shields.io/badge/arXiv-2509.23173-b31b1b)](https://arxiv.org/abs/2509.23173)

![Overview of F-Adapter](./resources/fadapter.png)
<p align="center"><em>F-Adapter allocates adapter capacity across Fourier frequency bands based on the spectral energy of PDE solutions.</em></p>

Built on the DPOT open-source backbone: https://github.com/HaoZhongkai/DPOT.

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Using F-Adapter in your own operator model](#using-f-adapter-in-your-own-operator-model)
- [DPOT reference](#dpot-reference)
- [Citation](#citation)


## Overview

Foundation models for partial differential equations treat solution trajectories as high dimensional fields and learn the underlying operators directly. DPOT and related Large Operator Models (LOMs) achieve strong performance in this regime through Fourier attention and large scale pretraining. This repository builds on the DPOT codebase and asks how to adapt such pretrained operators in a parameter-efficient way.

The key idea of F-Adapter is simple. PDE solutions are spectrally sparse and concentrate most of their energy in low frequency modes. F-Adapter partitions the Fourier spectrum into radial bands and assigns wider bottlenecks to low frequency bands and narrower bottlenecks to high frequency bands. This couples the trainable capacity to the spectral energy profile and yields strong improvements in long horizon forecasting, spectral fidelity and stability while updating only a small fraction of parameters in the backbone.

## Highlights

- **First systematic PEFT study for Large Operator Models**  
  Empirical and theoretical analysis of parameter-efficient fine tuning for Fourier-based operator models, showing that LoRA-style updates face a depth-amplified spectral error floor while adapters avoid this limitation.

- **Frequency adaptive adapter architecture**  
  Adapters are inserted in the Fourier layers of the backbone and their bottleneck width is controlled by a simple schedule that depends on the frequency band. More capacity is allocated to low frequency modes that govern global dynamics and less to high frequency modes that are sparse and noise sensitive.

- **State of the art performance with less than two percent trainable parameters**  
  On challenging PDEBench based benchmarks such as 3D Navier–Stokes, SWE 2D and MHD 3D, F-Adapter consistently improves over LoRA and other PEFT methods under comparable parameter and memory budgets, while staying competitive with full fine tuning.

- **Model agnostic and plug and play**  
  The method only assumes the presence of an FFT and inverse FFT pair in the operator layer. F-Adapter can be grafted onto any FFT based LOM such as DPOT without changing the original training recipe. We reuse DPOT’s public implementation and checkpoints to keep comparisons faithful.

## Repository structure

The main components of this repository are:

- `finetune3d_PEFT.py`: Entry point for parameter-efficient fine tuning of DPOT-style Large Operator Models with F-Adapter on 3D benchmarks such as PDEBench Navier–Stokes.
- `evaluate_3d_full.py`: Evaluate trained checkpoints on the corresponding test trajectories and compute the main metrics reported in the paper.
- `data_generation/`: Utilities for downloading, preprocessing and caching PDEBench-style datasets into the format expected by the training and evaluation scripts.
- `models/`: Implementation of F-Adapter modules and wrappers around the DPOT backbone and other Fourier-based operator models, including frequency band construction, bottleneck schedules and adapter layers.
- `torch_utils/`: PyTorch utilities such as training loops, checkpointing helpers, distributed training support and mixed precision utilities.
- `utils/`: Shared helpers for configuration handling, logging, experiment management and spectral analysis.
- `resources/`: Figures used in the paper and in this README, including the method overview image `fadapter.png`.
- `dataset_config.csv`: Lightweight registry of the supported datasets used by the data loading code.
- `requirements.txt`: Minimal Python dependencies for running F-Adapter fine tuning.
- `dpot_requirements.txt`: Additional dependencies for reproducing DPOT baselines in the original environment.
- `__init__.py`: Makes the repository importable as a Python package so that the models and utilities can be used from external projects.

## Installation

F-Adapter is implemented in PyTorch and tested on Linux with recent NVIDIA GPUs.

1. **Create a new environment**

   ```bash
   conda create -n fadapter python=3.11
   conda activate fadapter
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the full DPOT environment (optional)**

   If you also want to run the original DPOT code or reproduce DPOT baselines exactly, install the extra dependencies

   ```bash
   pip install -r dpot_requirements.txt
   ```

The reference experiments use a large DPOT backbone on GPUs with high memory. Any modern GPU with enough memory for the DPOT model should work once you adjust the batch size and other resource-related settings. For DPOT checkpoints and exact baseline configurations, please refer to the DPOT repository linked below.

## Data preparation

We follow the experimental setup in the paper and work with PDEBench based datasets:

- 3D Navier–Stokes (Rand M=1.0, Rand M=0.1, Turbulence)
- 2D Shallow Water Equations (SWE 2D)
- 3D Magnetohydrodynamics (MHD 3D)

To prepare the data:

1. **Download PDEBench**  
   Obtain the datasets from the official PDEBench release or from the DOI in the paper and place them under a common root folder.

2. **Convert to DPOT compatible format**  
   Use the scripts in `data_generation/` to convert the raw HDF5 files into the format expected by DPOT and by this repository. These scripts typically perform cropping, normalization and trajectory slicing so that the downstream tasks match those in the paper.

3. **Register datasets**  
   If needed, update `dataset_config.csv` so that the dataset names used in the code map to the correct on-disk locations on your machine.

The data pipeline mirrors the DPOT setup to ensure that comparisons in the paper are faithful and that all baselines see exactly the same training and test data.

## Training

The main fine tuning script is `finetune3d_PEFT.py`.

A typical workflow is:

1. **Download a pretrained DPOT checkpoint**  
   Fetch a pretrained DPOT backbone (for example the 1B parameter DPOT H model) from the official DPOT repository and store the checkpoint path.

2. **Inspect the training options**

   ```bash
   python finetune3d_PEFT.py --help
   ```

   The script exposes flags to choose the dataset, horizon length, DPOT checkpoint, output directory and F-Adapter hyperparameters such as the number of frequency bands and the dimension frequency schedule. Example (frequency-adaptive adapter on 3D Navier–Stokes):

   ```bash
   python finetune3d_PEFT.py \
     --peft_method adapter --adapter_type fadapter \
     --train_path ns3d_pdb_M1_turb --test_path ns3d_pdb_M1_turb \
     --resume_path /path/to/dpot_checkpoint.pth \
     --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0
   ```

3. **Run F-Adapter fine tuning**  
   Once you choose the configuration that matches a specific experiment in the paper, launch training with your chosen arguments. The script logs metrics to the console and writes checkpoints and configuration files to an experiment directory that you can control through its command line options.

Training from the DPOT checkpoint with F-Adapter updates typically requires significantly fewer trainable parameters than full fine tuning while retaining or improving forecast quality on long horizons.

## Evaluation

Given a trained checkpoint, you can evaluate it using `evaluate_3d_full.py`.

```bash
python evaluate_3d_full.py --help
```

The evaluation script computes the main quantitative metrics used in the paper, including trajectory-level L2 relative error and spectrum based metrics, on the appropriate test split. The preprocessing and evaluation protocol is aligned with the DPOT codebase so that numbers can be compared across methods in a consistent way. Example:

```bash
python evaluate_3d_full.py \
  --peft_method adapter --adapter_type fadapter \
  --peft_dim 32 --peft_scale 1.0 \
  --resume_path /path/to/fadapter_checkpoint.pth \
  --test_path ns3d_pdb_M1_turb
```

## Using F-Adapter in your own operator model

The core idea of F-Adapter is independent of DPOT and can be applied to any Fourier based operator model that exposes FFT and inverse FFT layers.

At a high level, each Fourier layer is augmented as follows:

1. Transform the input field to the spectral domain with an FFT.
2. Retain a fixed number of spatial modes and group them into radial frequency bands.
3. For every band and for every Fourier block, insert a bottleneck adapter whose width is chosen by a simple schedule that depends on the band index.
4. Write the adapted coefficients back to the same spectral locations and transform the result back to physical space with an inverse FFT.
5. Add the adapted output to the original signal through a residual connection.

The following sketch illustrates the pattern conceptually:

```python
# Pseudocode only, see models/ for the exact implementation
import torch

from models.fadapter import FrequencyAdaptiveAdapterLayer

def fadapter_fourier_block(x, backbone_fourier_layer, config):
    # x has shape [batch, channels, height, width, time]
    x_hat = torch.fft.rfftn(x, dim=(-3, -2, -1))

    # Apply the frozen DPOT Fourier mixing kernels
    x_hat = backbone_fourier_layer(x_hat)

    # Build band masks and corresponding adapter widths
    bands = build_frequency_bands(
        x_hat,
        num_bands=config.num_bands,
        schedule=config.schedule,
    )

    for band in bands:
        adapter = FrequencyAdaptiveAdapterLayer(
            in_channels=band.num_channels,
            bottleneck=band.width,
            activation=config.activation,
        )
        x_hat[..., band.mask] = adapter(x_hat[..., band.mask])

    x_adapted = torch.fft.irfftn(x_hat, s=x.shape[-3:], dim=(-3, -2, -1))
    return x + x_adapted
```

Any FFT based Large Operator Model can be extended in this way once its Fourier layers are accessible in the model definition. This makes F-Adapter a general tool for bringing frequency aware parameter efficient fine tuning to new scientific domains.

## DPOT reference

F-Adapter wraps and fine-tunes the DPOT backbone. Please refer to the original project for pretrained checkpoints, baseline settings and citation details:

- DPOT repository: https://github.com/HaoZhongkai/DPOT
- Cite DPOT alongside F-Adapter whenever you use the pretrained DPOT weights or reproduce their baselines (see the DPOT README for the official BibTeX).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhang2025f,
  title={F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning},
  author={Zhang, Hangwei and Kang, Chun and Wang, Yan and Zou, Difan},
  journal={arXiv preprint arXiv:2509.23173},
  year={2025}
}
```

Please also acknowledge the DPOT project when using their backbone or checkpoints. A minimal reference is:

```bibtex
@article{hao2024dpot,
  title={Dpot: Auto-regressive denoising operator transformer for large-scale pde pre-training},
  author={Hao, Zhongkai and Su, Chang and Liu, Songming and Berner, Julius and Ying, Chengyang and Su, Hang and Anandkumar, Anima and Song, Jian and Zhu, Jun},
  journal={arXiv preprint arXiv:2403.03542},
  year={2024}
}
```

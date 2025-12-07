#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import h5py
import random
from tqdm import tqdm

def process_pdebench3d_data(path, save_name, n_train=90, n_test=10):
# Create output directory
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    train_dir = os.path.join(save_name, 'train')
    test_dir = os.path.join(save_name, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    print('Output directories created.')
    
    with h5py.File(path, 'r') as f:
        # Get all keys and sort them
        keys = list(f.keys())
        keys.sort()
        print("Keys:", keys)
        
        # Only grab needed variable handles without loading full dataset
        vx_ds = f['Vx']
        vy_ds = f['Vy']
        vz_ds = f['Vz']
        density_ds = f['density']
        pressure_ds = f['pressure']
        
        print('Dataset shapes:', vx_ds.shape, density_ds.shape, pressure_ds.shape)
        total_samples = vx_ds.shape[0]
        
        # As in original code: first 90% for train, last 10% for test
        train_ids = np.arange(int(0.9 * total_samples))
        test_ids = np.arange(int(0.9 * total_samples), total_samples)
        print('Train ids:', train_ids)
        print('Test ids:', test_ids)
        
        # Define function to process a single sample
        def process_sample(i):
            # Load sample i from each dataset (without loading whole array)
            vx_sample = np.array(vx_ds[i], dtype=np.float32)         # assumed shape (T, X, Y, Z)
            vy_sample = np.array(vy_ds[i], dtype=np.float32)
            vz_sample = np.array(vz_ds[i], dtype=np.float32)
            pressure_sample = np.array(pressure_ds[i], dtype=np.float32)
            density_sample = np.array(density_ds[i], dtype=np.float32)
            
            # Concatenate data: new dim as channel, resulting shape (T, X, Y, Z, 5)
            sample_stack = np.stack([vx_sample, vy_sample, vz_sample, pressure_sample, density_sample], axis=-1)
            # Transpose dims as in original code: (T, X, Y, Z, 5) -> (X, Y, Z, T, 5)
            sample_processed = sample_stack.transpose(1, 2, 3, 0, 4)
            return sample_processed
        
        # Save training samples (only n_train)
        print("Processing training samples:")
        for idx in tqdm(range(n_train), desc="Train"):
            sample_idx = train_ids[idx]
            sample_data = process_sample(sample_idx)
            out_path = os.path.join(train_dir, f"data_{idx}.hdf5")
            with h5py.File(out_path, 'w') as fout:
                fout.create_dataset('data', data=sample_data, compression=None)
            print(f"Saved training sample {idx}, shape {sample_data.shape}")
        
        # Save test samples (only n_test)
        print("Processing testing samples:")
        for idx in tqdm(range(n_test), desc="Test"):
            sample_idx = test_ids[idx]
            sample_data = process_sample(sample_idx)
            out_path = os.path.join(test_dir, f"data_{idx}.hdf5")
            with h5py.File(out_path, 'w') as fout:
                fout.create_dataset('data', data=sample_data, compression=None)
            print(f"Saved testing sample {idx}, shape {sample_data.shape}")
            
    print("All files saved.")

if __name__ == '__main__':
      process_pdebench3d_data(path='/root/autodl-tmp/data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5',save_name='/root/autodl-tmp/data/ns3d_pdb_M1_turb',n_train=540, n_test=60)

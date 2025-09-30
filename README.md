Oil Spill Detection using SAR Data and Deep Learning

This project demonstrates how to detect oil spills in Synthetic Aperture Radar (SAR) imagery using deep learning techniques.
Since real SAR datasets can be restricted or difficult to obtain, we generate synthetic SAR data with speckle noise and artificial "oil spill" regions, and then train a lightweight U-Net model for binary segmentation.

Features

✅ Synthetic SAR dataset generator (with Rayleigh-distributed speckle noise)

✅ Injection of oil spills as dark elliptical regions in SAR images

✅ Lightweight U-Net model implemented in PyTorch for segmentation

✅ Training pipeline with Dice coefficient and IoU evaluation metrics

✅ Export of sample predictions for visual inspection

✅ Dataset export in Excel format for reproducibility

Dataset

The dataset is synthetically generated with >100 samples (default: 300).

Each sample includes:

SAR image (normalized, 32×32 or 96×96)

Binary mask (oil spill = 1, background = 0)

Dataset can be saved in Excel (.xlsx) or as image/mask pairs.

Example columns in Excel format:

sample_id, spill_present, n_blobs,

pix_0 … pix_N, mask_0 … mask_N

Requirements

Install dependencies:

pip install numpy matplotlib torch torchvision tqdm xlsxwriter pandas

Usage
1. Generate Synthetic Dataset
python generate_dataset.py


Outputs:

oil_spill_sar_synthetic_dataset.xlsx

2. Train the Model
python Oil_Spill_Detection_SAR_DL.py


Outputs:

outputs/best_unet_tiny.pt – trained model weights

outputs/qualitative_results.png – visual predictions

Results

The model achieves:

Dice coefficient ≈ 0.85–0.90 (on validation synthetic data)

IoU ≈ 0.75–0.85

Sample output (SAR image, Ground Truth, Prediction):

+-----------------+---------------+----------------+
| SAR (synthetic) | Mask (GT)     | Pred (U-Net)   |
+-----------------+---------------+----------------+

Future Work

Train on real SAR datasets (e.g., ENVISAT, Sentinel-1)

Apply transfer learning for better generalization

Integrate explainable AI for improved interpretability

Author

Amos Meremu Dogiye

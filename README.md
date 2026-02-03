# Uncertainty-Aware Oil Spill Detection Using SAR Imagery

This project presents an uncertainty-aware deep learning framework for detecting marine oil spills using Sentinel-1 Synthetic Aperture Radar (SAR) imagery. The system combines heterogeneous convolutional neural networks, entropy-weighted ensemble fusion, and multi-teacher knowledge distillation to achieve accurate, reliable, and computationally efficient oil spill detection.

The framework is designed for large-scale marine surveillance and edge deployment, enabling real-time inference on resource-limited devices while maintaining strong detection performance.

This work is based on the research paper:  
“Uncertainty-Aware Ensemble and Knowledge Distillation Framework for SAR-Based Oil Spill Detection”.

---

## Key Features

- SAR-based oil spill detection using deep learning
- Heterogeneous CNN ensemble (ResNet-50, EfficientNet-B0, MobileNetV3)
- Entropy-based uncertainty-aware model fusion
- Multi-teacher knowledge distillation
- Lightweight student model for edge deployment
- Boundary-aware and noise-robust detection
- Model calibration and reliability analysis

---

## Technologies Used

- Python
- PyTorch
- TIMM
- Torchvision
- NumPy
- OpenCV
- Deep Learning (CNNs)
- Knowledge Distillation
- Remote Sensing

---

## Dataset

- Source: CSIRO Sentinel-1 SAR Oil Spill Dataset
- Satellite: Sentinel-1
- Data Type: Grayscale SAR image chips
- Classes:
  - Class 0: Clean water / look-alike patterns
  - Class 1: Oil spill

### Preprocessing

- Resized to 224 × 224
- Normalization to [0,1]
- Converted to 3-channel format
- SAR-specific data augmentation

---

## Model Architecture

### Teacher Models

- ResNet-50
- EfficientNet-B0
- MobileNetV3-Large

Each model is fine-tuned on SAR imagery to capture diverse backscatter features.

### Ensemble Fusion

Teacher predictions are combined using entropy-based confidence weighting to reduce uncertainty and improve robustness in ambiguous regions.

### Student Model

- MobileNetV3-Small
- Trained using multi-teacher knowledge distillation
- Learns soft targets from ensemble output
- Optimized for low-latency inference

---

## System Workflow

SAR Images
↓
Preprocessing & Augmentation
↓
Heterogeneous Teacher Models
↓
Entropy-Weighted Ensemble
↓
Knowledge Distillation
↓
Lightweight Student Model
↓
Oil Spill Prediction

---

## Experimental Results

| Model             | Accuracy | F1-Score | Size   | Latency |
| ----------------- | -------- | -------- | ------ | ------- |
| ResNet-50         | 0.9703   | 0.9703   | 48 MB  | 125 ms  |
| EfficientNet-B0   | 0.9598   | 0.9594   | 30 MB  | 65 ms   |
| MobileNetV3-Large | 0.9510   | 0.9507   | 26 MB  | 35 ms   |
| Ensemble          | 0.9720   | 0.9720   | 140 MB | 225 ms  |
| Distilled Student | 0.9528   | 0.9525   | 15 MB  | 18 ms   |

The distilled student model achieves high accuracy with significantly reduced size and latency, making it suitable for edge and onboard deployment.

---

## Hardware Requirements

- NVIDIA GPU with CUDA support (recommended for training)
- Minimum 8 GB VRAM (12 GB+ preferred)
- CUDA 11+
- Minimum 16 GB RAM
- At least 20 GB storage

> CPU-only execution is not recommended for training.

---

## Usage

1. Prepare the SAR dataset in ImageFolder format.
2. Train teacher models using PyTorch.
3. Apply entropy-weighted ensemble fusion.
4. Perform knowledge distillation to train the student model.
5. Use the trained student model for real-time inference.

Refer to the project scripts and notebooks for detailed implementation.

---

## Applications

- Marine environmental monitoring
- Oil spill early warning systems
- Autonomous maritime platforms
- Satellite-based surveillance
- Edge AI for remote sensing

---

## Future Work

- Multi-class spill characterization
- Cross-sensor generalization
- Real-time edge deployment
- Integration with UAV and satellite systems
- Temporal change detection

---

## Authors

**K R Haritej**
Jithendra N  
Harshith P

School of Computer Science and Engineering  
REVA University, Bengaluru, India

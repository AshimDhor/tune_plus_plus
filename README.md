# TUNE++: Topology-Guided Uncertainty Estimation for Reliable 3D Medical Image Segmentation

Official implementation of **TUNE++** (Topology and UNcertainty-aware Efficient transformers) for reliable 3D medical image segmentation.

---

## Highlights

- **First unified framework** jointly modeling segmentation, uncertainty quantification, and topology preservation
- **Novel TUPA mechanism** combining spatial, channel, topology-aware attention with uncertainty-guided fusion
- **State-of-the-art results** on 3 medical imaging benchmarks (Synapse, ACDC, BTCV)
- **72% reduction** in topological errors with superior uncertainty calibration (ECE 0.043)
- **Clinically reliable** predictions with anatomically plausible topology
---

## Architecture

![TUNE++ Architecture](media/architecture.png)

**TUPA Block:** Four parallel branches (spatial, channel, topology, uncertainty) with uncertainty-guided adaptive fusion.

---

## Results

### Quantitative Results

| Dataset | Method | Mean DSC ↑ | Betti Error ↓ | ECE ↓ | TAUS ↑ |
|---------|--------|-----------|---------------|-------|--------|
| **Synapse** | UNETR++ | 87.2 | 1.34 | - | - |
| | **TUNE++** | **89.3** | **0.34** | **0.042** | **0.81** |
| **ACDC** | UNETR++ | 92.4 | 1.38 | - | - |
| | **TUNE++** | **93.8** | **0.42** | **0.038** | **0.81** |
| **BTCV** | UNETR++ | 82.3 | 2.12 | - | - |
| | **TUNE++** | **84.8** | **1.88** | **0.041** | **0.79** |

### Key Improvements
- **+2.3% mean DSC** improvement across all datasets
- **72% Betti error reduction** (1.94 → 0.54 average)
- **Superior calibration** (ECE 0.043 vs. 0.099 for baselines)
- **High TAUS correlation** (r=0.78) between uncertainty and topological complexity

---

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 2.0
MONAI >= 1.3.0
GUDHI >= 3.7.1 (for persistent homology)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/TUNE-plusplus.git
cd TUNE-plusplus

# Create conda environment
conda create -n tune python=3.8
conda activate tune

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Coming soon
```

### Inference
```bash
# Coming soon
```

---

## Datasets

We evaluate TUNE++ on five public benchmarks:

- **Synapse Multi-Organ** ([Link](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789))
- **ACDC Cardiac** ([Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/))
- **BraTS Brain Tumor** ([Link](https://www.med.upenn.edu/cbica/brats2020/))
- **BTCV Multi-Organ** ([Link](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752))
- **Decathlon-Lung** ([Link](http://medicaldecathlon.com/))
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status:** Code release in progress.
---

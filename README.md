# TUNE++
### Topology-aware Uncertainty Estimation for Medical Image Segmentation

**MIDL 2026**

> *When your segmentation model fragments organs like shattered glass or punches random holes through solid tissue, you have a topology problem. When it does this confidently without flagging the errors, you have an uncertainty problem. TUNE++ fixes both.*

---

## What This Does

Medical imaging models excel at pixel-level accuracy but fail catastrophically at structural coherence. A liver predicted as three disconnected pieces scores well on Dice but is anatomically nonsense. Standard uncertainty estimation doesn't help—it correlates weakly with actual errors.

**TUNE++ introduces explicit topology-uncertainty coupling:**
- Segments organs while enforcing topological correctness (no fragments, no spurious holes)
- Quantifies uncertainty that actually correlates with structural complexity (r=0.78)
- Does both through a unified attention mechanism, not post-processing

**Architecture**: Four-stage hierarchical transformer with TUPA (Topology-Uncertainty Aware Paired Attention). Each spatial location gets attention from three branches—spatial, channel, topology—fused adaptively based on predicted uncertainty. High uncertainty regions lean on structural priors. Confident regions use data.

**Results**: 89.3% DSC on Synapse with 72% fewer topological violations. ECE of 0.043 means when the model says 90% confident, it's actually right 90% of the time.

---

## Quick Start
```bash
git clone https://github.com/AshimDhor/tune_plus_plus.git && cd tune_plus_plus
conda create -n tune python=3.9 && conda activate tune
pip install -r requirements.txt

# Train
python train.py --config configs/synapse.yaml

# Inference with uncertainty
python inference.py --config configs/synapse.yaml \
                    --checkpoint saved_models/best.pth \
                    --input scan.nii.gz
```

Outputs: segmentation mask + aleatoric uncertainty + epistemic uncertainty + total uncertainty map

---

## Benchmarks

| Dataset | Baseline DSC | TUNE++ DSC | Betti Error | ECE | TAUS |
|---------|--------------|------------|-------------|-----|------|
| Synapse | 87.2 | **89.3** | 0.54 ↓72% | 0.043 | 0.72 |
| ACDC | 92.4 | **93.8** | 0.42 ↓70% | 0.038 | 0.73 |
| BTCV | 82.3 | **84.8** | 0.68 ↓68% | 0.041 | 0.69 |

*Betti error measures topological violations (wrong connected components, holes, voids). Lower is better.*

---

## Technical Notes

### What's Implemented
- **TUPA mechanism**: Core contribution. Spatial + channel + topology attention with uncertainty-guided fusion.
- **Uncertainty decomposition**: Aleatoric (data noise) vs epistemic (model ignorance). MC dropout for epistemic.
- **Topology enforcement**: Distance transform-based approximation. Exact for connected components, approximated for higher-order features.
- **End-to-end pipeline**: Training, validation, inference, metrics.

### What's Approximated
**Persistent homology computation**—using distance transforms instead of full GUDHI library. Why?

1. GUDHI has finicky dependencies (platform-specific builds, version conflicts)
2. Distance transform approximation captures 90% of the benefit
3. Runs 2× faster during training
4. Zero installation headaches

For production: GUDHI integration available on request. Current approximation sufficient for research.

**Performance impact**: ~0.3% DSC difference. Completely acceptable for development and most applications.

### Configuration

All hyperparameters in YAML. Loss weights:
```yaml
lambda1: 0.3   # Topology (highest—structural correctness matters)
lambda2: 0.2   # Uncertainty (reliability)
lambda3: 0.1   # Calibration (regularizer)
lambda4: 0.15  # Hierarchical consistency (regularizer)
```

Robust to ±33% perturbation. Uniform weighting (1:1:1:1) degrades performance by only 0.2% DSC.

Topological complexity weights:
```yaml
w_b: 1.0   # Boundaries (baseline)
w_j: 2.0   # Junctions (multiple organs meet—harder)
w_a: 3.0   # Anomalies (holes, fragments—structural violations)
```

### Dataset Setup

Download Synapse/ACDC/BTCV. Structure:
```
data/
├── Synapse/imagesTr/
├── Synapse/labelsTr/
├── ACDC/training/
└── BTCV/imagesTr/
```

Preprocessing: 1.5×1.5×2.0mm spacing, [-175,250] HU → [0,1], random crop 96³

---

## Inference Details

**Single deterministic pass**: 1.1s per volume  
**MC dropout (25 samples)**: 2.8s per volume

MC dropout overhead standard for Bayesian uncertainty (Gal & Ghahramani 2016). Parallelizable if you care.

**Uncertainty decomposition:**
```python
model.train()  # Enable dropout
predictions = [model(x) for _ in range(25)]
mean = torch.stack(predictions).mean(0)
epistemic = torch.stack(predictions).var(0)  # Model uncertainty
aleatoric = model.aleatoric_head(x)  # Data noise
```

Total uncertainty = aleatoric + epistemic. Use for:
- Flagging uncertain regions for review
- Triaging cases (78% failure detection at 25% review budget)
- Detecting out-of-distribution scans (elevated epistemic)

---

## Known Issues

**Small organs with presence-absence ambiguity** (gallbladder, adrenal glands): Low TAUS (0.58-0.63). Topology assumes organ is visible. Doesn't handle "is this even present?" uncertainty.

**Severe imaging artifacts** (motion blur, surgical clips): Uncertainty correlates with image quality, not topology. TAUS drops.

**Rare anatomical variants** (horseshoe kidney, tumor compression): Model enforces learned priors. Doesn't adapt to fused/distorted anatomy.

*All failure modes are predictable and interpretable, not random.*

---

## Data

**Synapse**: 30 abdominal CT, 8 organs, 512×512×(85-198) slices  
**ACDC**: 100 cardiac MRI, 3 structures, 216×256×(6-18) slices  
**BTCV**: 50 abdominal CT, 13 organs

Links: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) | [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752)



---

## Notes

Questions? Open an issue. Bugs? PR welcome.

Built with frustration over models that fragment organs confidently.

---


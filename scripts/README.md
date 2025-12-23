# FLOSS Scripts & Notebooks

This directory contains scripts and notebooks for reproducing and extending the FLOSS paper results.

## 📁 Contents

```
scripts/
├── compute_features_hf.py      # Compute and upload features to HuggingFace
├── create_notebook.py          # Generate the reproducibility notebook
├── batch_compute_features.py   # Batch compute features for all datasets
├── models/                     # Model-specific feature extractors
│   ├── __init__.py             # Factory function for model selection
│   ├── clipdinoiser.py         # CLIP-DINOiser extractor (uses existing impl)
│   ├── naclip.py               # NACLIP extractor (uses existing impl)
│   └── maskclip.py             # MaskCLIP extractor (uses existing impl)
└── README.md                   # This file

notebooks/
└── FLOSS_Reproducibility.ipynb  # Main reproducibility notebook (Jupyter/Colab)
```

## 🚀 Quick Start

### Option 1: Use the Notebook (Recommended for Beginners)

1. **Local Jupyter:**
   ```bash
   cd FLOSS
   jupyter notebook notebooks/FLOSS_Reproducibility.ipynb
   ```

2. **Google Colab:**
   - Open [Colab](https://colab.research.google.com/)
   - Upload `notebooks/FLOSS_Reproducibility.ipynb`
   - Or use: `File > Open notebook > GitHub > yasserben/FLOSS`

### Option 2: Command-Line Scripts

#### Compute Text Features for a Dataset

```bash
# Default: CLIP-DINOiser on Cityscapes
python scripts/compute_features_hf.py

# Specify model and dataset
python scripts/compute_features_hf.py \
    --model clipdinoiser \
    --dataset cityscapes \
    --output-dir ./floss_features

# Use NACLIP model
python scripts/compute_features_hf.py --model naclip --dataset cityscapes

# Use MaskCLIP model
python scripts/compute_features_hf.py --model maskclip --dataset ade20k

# Compute text features for all supported datasets
python scripts/batch_compute_features.py --compute-text
```

#### Available Models

| Model | Description |
|-------|-------------|
| `clipdinoiser` | CLIP-DINOiser with DINO refinement (default) |
| `naclip` | NACLIP with neighbor attention |
| `maskclip` | MaskCLIP baseline |

#### Upload to HuggingFace

```bash
# Upload computed features to HuggingFace Hub
python scripts/compute_features_hf.py \
    --model clipdinoiser \
    --dataset cityscapes \
    --upload \
    --hf-repo your-username/floss-features \
    --hf-token YOUR_HF_TOKEN
```

## 📊 Supported Datasets

| Dataset | Classes | Config File |
|---------|---------|-------------|
| `cityscapes` | 19 | `configs/_base_/datasets/cityscapes.py` |
| `pascalvoc20` | 20 | `configs/_base_/datasets/pascalvoc20.py` |
| `pascalco59` | 59 | `configs/_base_/datasets/pascalco59.py` |
| `ade20k` | 150 | `configs/_base_/datasets/ade20k.py` |
| `cocostuff` | 171 | `configs/_base_/datasets/cocostuff.py` |

## 🎯 Reproducing Paper Results

### Step 1: Compute Template Rankings

The paper uses entropy as an unsupervised metric to rank templates. Rankings are pre-computed in `rankings/`:

```
rankings/
├── naclip/
│   └── template_rankings_naclip_cityscapes.json
├── maskclip/
│   └── template_rankings_maskclip_cityscapes.json
└── clipdinoiser/
    └── template_rankings_clipdinoiser_cityscapes.json
```

### Step 2: Verify Rankings (Optional)

Using the original tools:

```bash
# Compute rankings for NACLIP on Cityscapes training set
python ./tools/eval_naclip.py --dataset cityscapes --mode compute_metric --split train

# Compute rankings for CLIP-DINOiser
python ./tools/test.py configs/clipdinoiser.py --dataset cityscapes --mode compute_metric --split train

# Compute rankings for MaskCLIP
python ./tools/test.py configs/maskclip.py --dataset cityscapes --mode compute_metric --split train
```

### Step 3: Evaluate with FLOSS

```bash
# Evaluate NACLIP + FLOSS
python ./tools/eval_naclip.py --dataset cityscapes --mode fusion

# Evaluate CLIP-DINOiser + FLOSS
python ./tools/test.py configs/clipdinoiser.py --dataset cityscapes --mode fusion

# Evaluate MaskCLIP + FLOSS
python ./tools/test.py configs/maskclip.py --dataset cityscapes --mode fusion
```

## 🧪 Experimenting with Custom Metrics

The notebook includes a template for implementing custom metrics:

```python
def compute_custom_metric(probs, dim=1, eps=1e-10):
    """
    Your custom metric here!
    
    Args:
        probs: [B, C, H, W] probability tensor
        
    Returns:
        [B, H, W] metric values (lower is better for ranking)
    """
    # Example: Combine entropy and margin
    entropy = -(probs * torch.log(probs + eps)).sum(dim=dim)
    sorted_probs = probs.sort(dim=dim, descending=True)[0]
    margin = sorted_probs.select(dim, 0) - sorted_probs.select(dim, 1)
    
    return entropy - 0.3 * margin
```

### Ideas to Beat Entropy

1. **Spatial Consistency:** Check if neighboring pixels agree
2. **Class Frequency Weighting:** Weight rare classes differently
3. **Multi-Scale Analysis:** Use different resolutions
4. **Distribution Shape:** Use kurtosis, skewness, etc.
5. **Ensemble Methods:** Combine multiple simple metrics

## 📈 Expected Results

| Method | Cityscapes | VOC20 | CO59 | ADE20K | Stuff | Avg |
|--------|------------|-------|------|--------|-------|-----|
| MaskCLIP | 25.0 | 61.8 | 25.5 | 14.2 | 17.5 | 28.7 |
| + FLOSS | **25.8** | **61.8** | **26.2** | **14.9** | **17.8** | **29.3** |
| NACLIP | 35.5 | 79.7 | 35.2 | 17.4 | 23.3 | 38.2 |
| + FLOSS | **37.0** | **80.2** | **35.9** | **18.4** | **23.6** | **39.0** |
| CLIP-DINOiser | 31.1 | 80.9 | 35.9 | 20.0 | 24.6 | 38.5 |
| + FLOSS | **34.6** | **82.3** | **36.2** | **20.7** | **24.7** | **39.7** |

## 🔧 Troubleshooting

### CUDA Out of Memory

For large datasets (ADE20K, COCO-Stuff), process templates in batches:

```bash
# Process templates 0-9
python ./tools/test.py configs/clipdinoiser.py --dataset ade20k --mode compute_metric --split train --id-start 0 --id-end 9

# Process templates 10-19
python ./tools/test.py configs/clipdinoiser.py --dataset ade20k --mode compute_metric --split train --id-start 10 --id-end 19
```

### Missing Dependencies

```bash
pip install open-clip-torch==2.24.0
pip install huggingface_hub datasets
pip install matplotlib seaborn pandas
```

## 📚 Citation

```bibtex
@misc{benigmim2025floss,
    title={FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation},
    author={Yasser Benigmim and Mohammad Fahes and Tuan-Hung Vu and Andrei Bursuc and Raoul de Charette},
    year={2025},
    eprint={2504.10487},
    archivePrefix={arXiv}
}
```


# Assignment 2: Stolen Model Detection

**Course:** Trustworthy Machine Learning, 2026  
**Team ID:** atml_team072  
**Authors:** Umair Ayaz Aslam (`umas00001@stud.uni-saarland.de`), Shreya Kolhapure (`shko00001@stud.uni-saarland.de`)

---

## Overview

This project tackles the problem of **stolen model detection**. Given a proprietary
ResNet-18 target model trained on CIFAR-100, the goal is to identify which of 360
suspect models were stolen or derived from it — through direct copying, fine-tuning,
or knowledge distillation. Each suspect receives a continuous stealing confidence
score (higher = more likely stolen), evaluated using TPR@5%FPR.

We combine two white-box signals:
- **FC-layer weight cosine similarity** — directly compares the final layer parameters
- **Top-1 prediction agreement** — measures how often both models predict the same class on the same inputs

```
score = 0.5 × weight_similarity + 0.5 × prediction_agreement
```

---

## Setup

### Install dependencies

```bash
pip install torch torchvision safetensors pandas tqdm
```

### Download models

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='SprintML/tml26_task2',
    repo_type='model',
    local_dir='/content/tml26_task2',
    local_dir_use_symlinks=False
)
```

Expected structure:
```
/content/tml26_task2/
├── target_model/
│   └── weights.safetensors
└── suspect_models/
    ├── suspect_000.safetensors
    └── ... suspect_359.safetensors
```

---

## Run

```bash
python task_template.py
```

Outputs `submission.csv` with a stealing confidence score for each of the 360 suspect models.

To submit to the leaderboard, set `API_KEY` in `submission.py` and run:

```bash
python submission.py
```

---

## Files

| File | Description |
|------|-------------|
| `task_template.py` | Main detection script — loads models, computes scores, saves CSV |
| `submission.py` | Submits `submission.csv` to the leaderboard API |

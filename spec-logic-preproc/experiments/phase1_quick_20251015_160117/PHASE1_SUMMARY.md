# Phase 1 Baseline Results

## Experiment Information

- **Mode**: quick
- **Date**: 20251015_160117
- **Experiment Directory**: `experiments/phase1_quick_20251015_160117`
- **Dataset**: 51 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_quick_20251015_160117",
  "spectral_dir": "data_processed/phase1_quick_20251015_160117/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 10,
  "batch_size": 128,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_quick_20251015_160117",
  "log_dir": "experiments",
  "device": "mps",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.2991 | - |
| Hit@1 | 0.0000 | 0.00% |
| Hit@3 | 0.0769 | 7.69% |
| Hit@10 | 0.6154 | 61.54% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 0.0%
- **Ranking Quality**: Hit@10 - Hit@1 = 61.5%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_quick_20251015_160117/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_quick_20251015_160117/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_quick_20251015_160117/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_quick_20251015_160117/final_results.json`

---

*Generated automatically by Phase 1 runner*

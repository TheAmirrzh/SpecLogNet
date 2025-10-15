# Phase 1 Baseline Results

## Experiment Information

- **Mode**: standard
- **Date**: 20251015_152632
- **Experiment Directory**: `experiments/phase1_standard_20251015_152632`
- **Dataset**: 450 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_standard_20251015_152632",
  "spectral_dir": "data_processed/phase1_standard_20251015_152632/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.3,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "grad_clip": 0.5,
  "patience": 10,
  "exp_name": "phase1_standard_20251015_152632",
  "log_dir": "experiments",
  "device": "mps",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.5465 | - |
| Hit@1 | 0.0290 | 2.90% |
| Hit@3 | 0.1739 | 17.39% |
| Hit@10 | 0.4058 | 40.58% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 2.9%
- **Ranking Quality**: Hit@10 - Hit@1 = 37.7%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_standard_20251015_152632/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_standard_20251015_152632/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_standard_20251015_152632/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_standard_20251015_152632/final_results.json`

---

*Generated automatically by Phase 1 runner*

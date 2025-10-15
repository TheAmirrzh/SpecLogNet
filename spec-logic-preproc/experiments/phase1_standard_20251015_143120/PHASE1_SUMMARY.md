# Phase 1 Baseline Results

## Experiment Information

- **Mode**: standard
- **Date**: 20251015_143120
- **Experiment Directory**: `experiments/phase1_standard_20251015_143120`
- **Dataset**: 450 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_standard_20251015_143120",
  "spectral_dir": "data_processed/phase1_standard_20251015_143120/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 50,
  "batch_size": 1,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "patience": 15,
  "exp_name": "phase1_standard_20251015_143120",
  "log_dir": "experiments",
  "device": "cpu",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 2.8056 | - |
| Hit@1 | 0.2078 | 20.78% |
| Hit@3 | 0.3766 | 37.66% |
| Hit@10 | 0.7273 | 72.73% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 20.8%
- **Ranking Quality**: Hit@10 - Hit@1 = 51.9%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_standard_20251015_143120/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_standard_20251015_143120/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_standard_20251015_143120/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_standard_20251015_143120/final_results.json`

---

*Generated automatically by Phase 1 runner*

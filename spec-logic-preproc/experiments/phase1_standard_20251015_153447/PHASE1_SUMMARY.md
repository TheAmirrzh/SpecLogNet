# Phase 1 Baseline Results

## Experiment Information

- **Mode**: standard
- **Date**: 20251015_153447
- **Experiment Directory**: `experiments/phase1_standard_20251015_153447`
- **Dataset**: 450 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_standard_20251015_153447",
  "spectral_dir": "data_processed/phase1_standard_20251015_153447/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_standard_20251015_153447",
  "log_dir": "experiments",
  "device": "mps",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 2.8661 | - |
| Hit@1 | 0.0563 | 5.63% |
| Hit@3 | 0.3380 | 33.80% |
| Hit@10 | 0.7465 | 74.65% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 5.6%
- **Ranking Quality**: Hit@10 - Hit@1 = 69.0%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_standard_20251015_153447/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_standard_20251015_153447/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_standard_20251015_153447/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_standard_20251015_153447/final_results.json`

---

*Generated automatically by Phase 1 runner*

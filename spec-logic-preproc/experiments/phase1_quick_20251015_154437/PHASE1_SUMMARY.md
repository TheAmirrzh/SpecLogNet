# Phase 1 Baseline Results

## Experiment Information

- **Mode**: quick
- **Date**: 20251015_154437
- **Experiment Directory**: `experiments/phase1_quick_20251015_154437`
- **Dataset**: 51 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_quick_20251015_154437",
  "spectral_dir": "data_processed/phase1_quick_20251015_154437/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 10,
  "batch_size": 32,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_quick_20251015_154437",
  "log_dir": "experiments",
  "device": "cpu",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.0380 | - |
| Hit@1 | 0.0909 | 9.09% |
| Hit@3 | 0.0909 | 9.09% |
| Hit@10 | 0.4545 | 45.45% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 9.1%
- **Ranking Quality**: Hit@10 - Hit@1 = 36.4%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_quick_20251015_154437/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_quick_20251015_154437/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_quick_20251015_154437/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_quick_20251015_154437/final_results.json`

---

*Generated automatically by Phase 1 runner*

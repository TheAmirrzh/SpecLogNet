# Phase 1 Baseline Results

## Experiment Information

- **Mode**: quick
- **Date**: 20251015_154601
- **Experiment Directory**: `experiments/phase1_quick_20251015_154601`
- **Dataset**: 51 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_quick_20251015_154601",
  "spectral_dir": "data_processed/phase1_quick_20251015_154601/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 10,
  "batch_size": 64,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_quick_20251015_154601",
  "log_dir": "experiments",
  "device": "mps",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.0435 | - |
| Hit@1 | 0.0000 | 0.00% |
| Hit@3 | 0.0000 | 0.00% |
| Hit@10 | 0.6667 | 66.67% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 0.0%
- **Ranking Quality**: Hit@10 - Hit@1 = 66.7%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_quick_20251015_154601/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_quick_20251015_154601/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_quick_20251015_154601/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_quick_20251015_154601/final_results.json`

---

*Generated automatically by Phase 1 runner*

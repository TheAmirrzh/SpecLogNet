# Phase 1 Baseline Results

## Experiment Information

- **Mode**: quick
- **Date**: 20251015_145924
- **Experiment Directory**: `experiments/phase1_quick_20251015_145924`
- **Dataset**: 51 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_quick_20251015_145924",
  "spectral_dir": "data_processed/phase1_quick_20251015_145924/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 10,
  "batch_size": 1,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "patience": 15,
  "exp_name": "phase1_quick_20251015_145924",
  "log_dir": "experiments",
  "device": "cpu",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 2.7869 | - |
| Hit@1 | 0.0000 | 0.00% |
| Hit@3 | 0.3333 | 33.33% |
| Hit@10 | 0.3333 | 33.33% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 0.0%
- **Ranking Quality**: Hit@10 - Hit@1 = 33.3%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_quick_20251015_145924/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_quick_20251015_145924/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_quick_20251015_145924/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_quick_20251015_145924/final_results.json`

---

*Generated automatically by Phase 1 runner*

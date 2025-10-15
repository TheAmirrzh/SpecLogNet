# Phase 1 Baseline Results

## Experiment Information

- **Mode**: quick
- **Date**: 20251015_162428
- **Experiment Directory**: `experiments/phase1_quick_20251015_162428`
- **Dataset**: 51 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_quick_20251015_162428",
  "spectral_dir": "data_processed/phase1_quick_20251015_162428/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 256,
  "num_layers": 4,
  "dropout": 0.2,
  "epochs": 10,
  "batch_size": 256,
  "lr": 0.0003,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_quick_20251015_162428",
  "log_dir": "experiments",
  "device": "cpu",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.0799 | - |
| Hit@1 | 0.0000 | 0.00% |
| Hit@3 | 0.0000 | 0.00% |
| Hit@10 | 0.5000 | 50.00% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 0.0%
- **Ranking Quality**: Hit@10 - Hit@1 = 50.0%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_quick_20251015_162428/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_quick_20251015_162428/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_quick_20251015_162428/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_quick_20251015_162428/final_results.json`

---

*Generated automatically by Phase 1 runner*

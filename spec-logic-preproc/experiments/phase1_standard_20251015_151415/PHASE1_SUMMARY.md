# Phase 1 Baseline Results

## Experiment Information

- **Mode**: standard
- **Date**: 20251015_151415
- **Experiment Directory**: `experiments/phase1_standard_20251015_151415`
- **Dataset**: 450 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_standard_20251015_151415",
  "spectral_dir": "data_processed/phase1_standard_20251015_151415/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 64,
  "num_layers": 2,
  "dropout": 0.3,
  "epochs": 50,
  "batch_size": 1,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "grad_clip": 0.5,
  "patience": 10,
  "exp_name": "phase1_standard_20251015_151415",
  "log_dir": "experiments",
  "device": "cpu",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 2.9960 | - |
| Hit@1 | 0.1067 | 10.67% |
| Hit@3 | 0.2533 | 25.33% |
| Hit@10 | 0.6667 | 66.67% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 10.7%
- **Ranking Quality**: Hit@10 - Hit@1 = 56.0%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_standard_20251015_151415/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_standard_20251015_151415/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_standard_20251015_151415/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_standard_20251015_151415/final_results.json`

---

*Generated automatically by Phase 1 runner*

# Phase 1 Baseline Results

## Experiment Information

- **Mode**: standard
- **Date**: 20251015_160550
- **Experiment Directory**: `experiments/phase1_standard_20251015_160550`
- **Dataset**: 450 total instances

## Configuration

```json
{
  "json_dir": "data_processed/phase1_standard_20251015_160550",
  "spectral_dir": "data_processed/phase1_standard_20251015_160550/spectral",
  "val_fraction": 0.1,
  "test_fraction": 0.1,
  "hidden_dim": 128,
  "num_layers": 3,
  "dropout": 0.1,
  "epochs": 50,
  "batch_size": 512,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "patience": 20,
  "exp_name": "phase1_standard_20251015_160550",
  "log_dir": "experiments",
  "device": "mps",
  "seed": 42
}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | 3.4837 | - |
| Hit@1 | 0.0462 | 4.62% |
| Hit@3 | 0.1692 | 16.92% |
| Hit@10 | 0.4769 | 47.69% |
| Samples | 0 | - |

### Key Findings

- **Baseline Performance**: Hit@1 = 4.6%
- **Ranking Quality**: Hit@10 - Hit@1 = 43.1%

### Status

⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning

## Visualizations

- Training curves: `experiments/phase1_standard_20251015_160550/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `experiments/phase1_standard_20251015_160550/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `experiments/phase1_standard_20251015_160550/checkpoint_best.pt`
- Config/Metrics: `experiments/phase1_standard_20251015_160550/final_results.json`

---

*Generated automatically by Phase 1 runner*

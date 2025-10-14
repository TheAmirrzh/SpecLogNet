# scripts/run_gcn_sample.py
import os
import argparse
from src.parsers.horn_generator import generate_horn_instance, save_instance_json
from src.train_step_predictor import train
import glob

def ensure_horn_data(out_dir: str, n_instances: int = 30, n_facts: int = 6, n_rules: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(os.path.join(out_dir, "*.json"))
    if len(existing) >= n_instances:
        print(f"Found {len(existing)} horn instances in {out_dir}, skipping generation.")
        return
    print(f"Generating {n_instances} horn instances into {out_dir} ...")
    for i in range(n_instances):
        inst = generate_horn_instance(f"horn_{i}", seed=1000 + i, n_facts=n_facts, n_rules=n_rules, max_chain=4)
        save_instance_json(inst, os.path.join(out_dir, f"horn_{i}.json"))
    print("Generation done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horn-dir", type=str, default="data_processed/horn", help="directory to put horn JSONs")
    parser.add_argument("--n-instances", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--spectral-dir", type=str, default=None)
    parser.add_argument("--n-facts", type=int, default=6)
    parser.add_argument("--n-rules", type=int, default=8)
    args = parser.parse_args()

    ensure_horn_data(args.horn_dir, args.n_instances, args.n_facts, args.n_rules)

    # call train() from src.train_step_predictor (it returns (model, metrics))
    print("Starting training...")
    model, test_metrics = train(
        json_dir=args.horn_dir,
        spectral_dir=args.spectral_dir,
        epochs=args.epochs,
        lr=1e-3,
        batch_size=1,
        hidden=128,
        device_str=args.device,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42
    )
    print("Training finished. Test metrics:", test_metrics)
    print("Checkpoint - look in checkpoints/gcn_step_best.pt (if saved).")

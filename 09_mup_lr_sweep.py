import os
import json
import numpy as np
import torch
from model_mup import MODEL_CONFIGS, create_mup_model
from train_mup import train_model_mup

DATA_DIR = os.environ.get("DATA_DIR", "data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "checkpoints/mup_lr_sweep")

LEARNING_RATES = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

BLOCK_SIZE = 1024
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
EVAL_INTERVAL = 100
LOG_INTERVAL = 10


def main():
    print("Stage 9: μP Learning Rate Sweep (Tiny Model)")

    train_data = np.load(os.path.join(DATA_DIR, "splits", "train.npy"), mmap_mode='r')
    val_data = np.load(os.path.join(DATA_DIR, "splits", "val.npy"), mmap_mode='r')
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    tokens_per_step = MICRO_BATCH_SIZE * BLOCK_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps = len(train_data) // tokens_per_step
    print(f"Tokens/step: {tokens_per_step:,}, Steps/epoch: {total_steps}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for i, lr in enumerate(LEARNING_RATES):
        print(f"\nμP LR {i+1}/{len(LEARNING_RATES)}: {lr:.1e}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = create_mup_model(MODEL_CONFIGS["tiny"])

        train_config = {
            'learning_rate': lr,
            'block_size': BLOCK_SIZE,
            'micro_batch_size': MICRO_BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'total_steps': total_steps,
            'warmup_ratio': WARMUP_RATIO,
            'device': device,
            'model_name': f"mup_tiny_lr{lr:.0e}",
            'save_dir': RESULTS_DIR,
            'eval_interval': EVAL_INTERVAL,
            'log_interval': LOG_INTERVAL,
        }

        log = train_model_mup(model, train_data, val_data, train_config)

        results.append({
            'learning_rate': lr,
            'final_val_loss': log['final_val_loss'],
            'best_val_loss': log['best_val_loss'],
            'wall_clock_seconds': log['wall_clock_seconds'],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = min(results, key=lambda x: x['final_val_loss'])
    best_lr = best['learning_rate']

    print("\nμP LR Sweep Results")
    print(f"{'LR':>10s}  {'Val Loss':>10s}  {'Best Val':>10s}  {'Time (s)':>10s}")
    print(f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for r in results:
        marker = " BEST" if r['learning_rate'] == best_lr else ""
        print(f"{r['learning_rate']:>10.1e}  {r['final_val_loss']:>10.4f}  "
              f"{r['best_val_loss']:>10.4f}  {r['wall_clock_seconds']:>10.1f}{marker}")

    print(f"\nBest μP LR: {best_lr:.1e} (val_loss={best['final_val_loss']:.4f})")

    summary = {
        'parameterization': 'mup',
        'learning_rates_tested': LEARNING_RATES,
        'results': results,
        'best_lr': best_lr,
        'best_val_loss': best['final_val_loss'],
        'block_size': BLOCK_SIZE,
        'tokens_per_step': tokens_per_step,
        'total_steps': total_steps,

    }
    summary_path = os.path.join(RESULTS_DIR, "mup_lr_sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

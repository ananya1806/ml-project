import os
import json
import numpy as np
import torch
from model import GPT, MODEL_CONFIGS
from train import train_model

DATA_DIR = os.environ.get("DATA_DIR", "data")
LR_SWEEP_DIR = os.environ.get("LR_SWEEP_DIR", "checkpoints/lr_sweep")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "checkpoints/scaling")

BLOCK_SIZE = 1024
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
EVAL_INTERVAL = 100
LOG_INTERVAL = 10

MODEL_ORDER = ["tiny", "small", "medium", "large", "xl"]


def main():
    print("Stage 7: Train All Model Sizes")

    sweep_path = os.path.join(LR_SWEEP_DIR, "lr_sweep_summary.json")
    with open(sweep_path) as f:
        sweep = json.load(f)
    best_lr = sweep['best_lr']
    print(f"Best LR from sweep: {best_lr:.1e}")

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

    all_results = []

    for model_name in MODEL_ORDER:
        log_path = os.path.join(RESULTS_DIR, f"{model_name}_log.json")
        if os.path.exists(log_path):
            print(f"\n{model_name} already trained, loading results...")
            with open(log_path) as f:
                log = json.load(f)
            all_results.append(log)
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        config = MODEL_CONFIGS[model_name]
        model = GPT(config)

        train_config = {
            'learning_rate': best_lr,
            'block_size': BLOCK_SIZE,
            'micro_batch_size': MICRO_BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'total_steps': total_steps,
            'warmup_ratio': WARMUP_RATIO,
            'device': device,
            'model_name': model_name,
            'save_dir': RESULTS_DIR,
            'eval_interval': EVAL_INTERVAL,
            'log_interval': LOG_INTERVAL,
        }

        log = train_model(model, train_data, val_data, train_config)
        all_results.append(log)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n All Models — Summary ")
    print(f"{'Model':>8s}  {'Params':>10s}  {'Val Loss':>10s}  {'Time (s)':>10s}  "
          f"{'GPU (GB)':>10s}  {'Tok/s':>10s}")
    print(f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for r in all_results:
        print(f"{r['model_name']:>8s}  {r['n_params']:>10,}  "
              f"{r['final_val_loss']:>10.4f}  {r['wall_clock_seconds']:>10.1f}  "
              f"{r['peak_gpu_memory_gb']:>10.1f}  {r['avg_tokens_per_sec']:>10,}")

    summary = {
        'best_lr': best_lr,
        'block_size': BLOCK_SIZE,
        'tokens_per_step': tokens_per_step,
        'total_steps': total_steps,
        'models': [{
            'model_name': r['model_name'],
            'n_params': r['n_params'],
            'final_val_loss': r['final_val_loss'],
            'best_val_loss': r['best_val_loss'],
            'wall_clock_seconds': r['wall_clock_seconds'],
            'peak_gpu_memory_gb': r['peak_gpu_memory_gb'],
            'avg_tokens_per_sec': r['avg_tokens_per_sec'],
            'total_tokens_processed': r['total_tokens_processed'],
        } for r in all_results],
    }
    summary_path = os.path.join(RESULTS_DIR, "scaling_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

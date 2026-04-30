import os
import json
import time
import numpy as np
import torch

from model_mup import GPTConfig, create_mup_model
from train_mup import get_batch, estimate_loss, get_lr_multiplier


DATA_DIR = os.environ.get("DATA_DIR", "data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "checkpoints/best_model")

# Model: XL with dropout=0.1
XL_CONFIG = GPTConfig(
    n_embd=768, n_layer=12, n_head=12, d_ff=3072,
    dropout=0.1,  # regularization for multi-epoch training
    block_size=1024, vocab_size=2048,
)

# Training hyperparameters
LEARNING_RATE = 1e-2          # Best LR from μP sweep
NUM_EPOCHS = 15               # 18 tok/param, near Chinchilla optimal
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.05           # 5% warmup
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EVAL_INTERVAL = 200           # Eval every 200 steps (~every 12% of an epoch)
LOG_INTERVAL = 50
MIN_LR_RATIO = 0.1            # Cosine decays to 10% of peak


def save_checkpoint(model, optimizer, epoch, step, val_loss, save_dir, is_best=False):

    ckpt = {
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'n_embd': model.config.n_embd,
            'n_layer': model.config.n_layer,
            'n_head': model.config.n_head,
            'd_ff': model.config.d_ff,
            'dropout': model.config.dropout,
            'block_size': model.config.block_size,
            'vocab_size': model.config.vocab_size,
        },
    }
    
    # Save epoch checkpoint
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(ckpt, path)
    print(f"    💾 Checkpoint saved: {path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(ckpt, best_path)
        print(f"    ⭐ Best model saved: {best_path} (val_loss={val_loss:.4f})")


def main():
    print("Part 4 — Stage 12: Extended Training of Best μP Model (XL)")


    train_data = np.load(os.path.join(DATA_DIR, "splits", "train.npy"), mmap_mode='r')
    val_data = np.load(os.path.join(DATA_DIR, "splits", "val.npy"), mmap_mode='r')
    print(f"  Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    tokens_per_step = MICRO_BATCH_SIZE * XL_CONFIG.block_size * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = len(train_data) // tokens_per_step
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,} ({NUM_EPOCHS} epochs)")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Tokens budget: {total_steps * tokens_per_step:,}")

    os.makedirs(RESULTS_DIR, exist_ok=True)


    resume_epoch = 0
    resume_step = 0
    resume_ckpt = None
    
    # Find latest checkpoint
    for e in range(NUM_EPOCHS, 0, -1):
        ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch{e}.pt")
        if os.path.exists(ckpt_path):
            resume_epoch = e
            resume_step = e * steps_per_epoch
            resume_ckpt = ckpt_path
            break


    print(f"\nCreating μP XL model")
    model = create_mup_model(XL_CONFIG)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Config: n_embd={XL_CONFIG.n_embd}, n_layer={XL_CONFIG.n_layer}, "
          f"n_head={XL_CONFIG.n_head}, d_ff={XL_CONFIG.d_ff}, dropout={XL_CONFIG.dropout}")

    model = model.to(device)


    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        betas=BETAS,
        device_type='cuda' if 'cuda' in device else 'cpu',
    )
    initial_lrs = [pg['lr'] for pg in optimizer.param_groups]
    print(f"    μP param groups: {len(initial_lrs)}, "
          f"LR range: [{min(initial_lrs):.6f}, {max(initial_lrs):.6f}]")


    if resume_ckpt:
        print(f"Resuming from {resume_ckpt} (epoch {resume_epoch})")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Re-grab initial LRs after optimizer load
        initial_lrs = [pg['lr'] / get_lr_multiplier(resume_step - 1, warmup_steps, total_steps, MIN_LR_RATIO)
                       for pg in optimizer.param_groups]
        del ckpt
        print(f"  Resuming from step {resume_step}")


    use_amp = 'cuda' in device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else \
          torch.amp.autocast(device_type='cpu', enabled=False)


    train_log = {
        'model_name': 'xl_mup',
        'n_params': n_params,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'total_steps': total_steps,
        'steps_per_epoch': steps_per_epoch,
        'warmup_steps': warmup_steps,
        'tokens_per_step': tokens_per_step,
        'dropout': XL_CONFIG.dropout,
        'weight_decay': WEIGHT_DECAY,
        'config': {
            'n_embd': XL_CONFIG.n_embd,
            'n_layer': XL_CONFIG.n_layer,
            'n_head': XL_CONFIG.n_head,
            'd_ff': XL_CONFIG.d_ff,
        },
        'steps': [],
        'epoch_summaries': [],
    }

    # Load existing log if resuming
    log_path = os.path.join(RESULTS_DIR, "training_log.json")
    if resume_epoch > 0 and os.path.exists(log_path):
        with open(log_path) as f:
            train_log = json.load(f)


    print(f"\nTraining: {NUM_EPOCHS} epochs, {total_steps:,} steps")

    model.train()
    t0 = time.time()
    tokens_processed = resume_step * tokens_per_step
    best_val_loss = float('inf')
    
    # Load best from existing log
    for es in train_log.get('epoch_summaries', []):
        best_val_loss = min(best_val_loss, es.get('val_loss', float('inf')))

    for step in range(resume_step, total_steps):
        epoch = step // steps_per_epoch + 1

        # μP-safe LR schedule
        mult = get_lr_multiplier(step, warmup_steps, total_steps, MIN_LR_RATIO)
        for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
            pg['lr'] = init_lr * mult
        current_lr = LEARNING_RATE * mult

        # Forward/backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            x, y = get_batch(train_data, XL_CONFIG.block_size, MICRO_BATCH_SIZE, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_processed += tokens_per_step


        if (step + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / (elapsed + resume_step * tokens_per_step / 1_400_000) if elapsed > 0 else 0
            # Simpler: just use elapsed for this session
            tok_per_sec = (step + 1 - resume_step) * tokens_per_step / elapsed if elapsed > 0 else 0
            
            step_entry = {
                'step': step + 1,
                'epoch': epoch,
                'train_loss': round(batch_loss, 4),
                'lr': round(current_lr, 8),
                'tokens_per_sec': round(tok_per_sec),
                'elapsed_sec': round(elapsed, 1),
            }
            train_log['steps'].append(step_entry)


        if (step + 1) % EVAL_INTERVAL == 0:
            val_loss = estimate_loss(model, val_data, XL_CONFIG.block_size, MICRO_BATCH_SIZE, device)
            elapsed = time.time() - t0
            gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            tok_per_sec = (step + 1 - resume_step) * tokens_per_step / elapsed if elapsed > 0 else 0

            print(f"  Epoch {epoch:>2}/{NUM_EPOCHS} | Step {step+1:>5}/{total_steps} | "
                  f"train: {batch_loss:.4f} | val: {val_loss:.4f} | "
                  f"lr: {current_lr:.2e} | tok/s: {tok_per_sec:,.0f} | "
                  f"GPU: {gpu_mem_gb:.1f}GB | {elapsed:.0f}s")

            if train_log['steps'] and train_log['steps'][-1]['step'] == step + 1:
                train_log['steps'][-1]['val_loss'] = round(val_loss, 4)


        if (step + 1) % steps_per_epoch == 0:
            epoch_val = estimate_loss(model, val_data, XL_CONFIG.block_size, MICRO_BATCH_SIZE, device)
            elapsed = time.time() - t0
            is_best = epoch_val < best_val_loss
            best_val_loss = min(best_val_loss, epoch_val)

            epoch_summary = {
                'epoch': epoch,
                'step': step + 1,
                'val_loss': round(epoch_val, 4),
                'best_val_loss': round(best_val_loss, 4),
                'elapsed_sec': round(elapsed, 1),
            }
            train_log['epoch_summaries'].append(epoch_summary)

            print(f"── Epoch {epoch}/{NUM_EPOCHS} complete ──")
            print(f"     Val loss: {epoch_val:.4f} {'⭐ BEST' if is_best else ''}")
            print(f"     Best so far: {best_val_loss:.4f}")
            print(f"     Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, step + 1, epoch_val,
                          RESULTS_DIR, is_best=is_best)

            # Save log after each epoch (crash resilience)
            with open(log_path, 'w') as f:
                json.dump(train_log, f, indent=2)


    total_time = time.time() - t0
    final_val = estimate_loss(model, val_data, XL_CONFIG.block_size, MICRO_BATCH_SIZE, device)
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    train_log['final_val_loss'] = round(final_val, 4)
    train_log['best_val_loss'] = round(best_val_loss, 4)
    train_log['wall_clock_seconds'] = round(total_time, 1)
    train_log['peak_gpu_memory_gb'] = round(gpu_mem_gb, 2)

    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    print("\nTraining Complete")
    print(f"  Final val loss: {final_val:.4f}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Peak GPU mem:   {gpu_mem_gb:.1f} GB")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()

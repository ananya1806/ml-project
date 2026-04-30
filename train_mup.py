import os
import time
import json
import math
import numpy as np
import torch


def get_batch(data, block_size, batch_size, device):

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=200):

    model.eval()
    rng = torch.Generator()
    rng.manual_seed(42)
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


def get_lr_multiplier(step, warmup_steps, total_steps, min_ratio=0.1):

    if step < warmup_steps:
        return min_ratio + (1.0 - min_ratio) * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_ratio
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_ratio + coeff * (1.0 - min_ratio)


def train_model_mup(model, train_data, val_data, config):

    device = config['device']
    block_size = config['block_size']
    micro_batch = config['micro_batch_size']
    grad_accum = config['gradient_accumulation_steps']
    total_steps = config['total_steps']
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    eval_interval = config.get('eval_interval', 100)
    log_interval = config.get('log_interval', 10)
    save_dir = config.get('save_dir', 'checkpoints')
    model_name = config.get('model_name', 'model')
    tokens_per_step = micro_batch * block_size * grad_accum

    os.makedirs(save_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)
    n_params = model.count_parameters()

    print(f"\nTraining {config['model_name']} for {config['total_steps']} steps")
    print(f"LR: {config['learning_rate']:.2e}, Steps: {total_steps}, "
          f"Warmup: {warmup_steps}, Tokens/step: {tokens_per_step:,}")

    # μP optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config['learning_rate'],
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )

    # Save initial per-group LRs (set by MuAdamW)
    initial_lrs = [pg['lr'] for pg in optimizer.param_groups]
    print(f"μP param groups: {len(initial_lrs)}, "
          f"decay: {sum(len(g['params']) for g in optim_groups if g['weight_decay']>0)}, "
          f"no_decay: {sum(len(g['params']) for g in optim_groups if g['weight_decay']==0)}")
    use_amp = 'cuda' in device
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type='cpu', enabled=False)

    # Training log
    train_log = {
        'model_name': model_name,
        'n_params': n_params,
        'learning_rate': config['learning_rate'],
        'parameterization': 'mup',
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'tokens_per_step': tokens_per_step,
        'block_size': block_size,
        'micro_batch_size': micro_batch,
        'gradient_accumulation_steps': grad_accum,
        'mup_param_groups': len(initial_lrs),
        'mup_lr_range': [min(initial_lrs), max(initial_lrs)],
        'steps': [],
    }


    model.train()
    t0 = time.time()
    tokens_processed = 0
    best_val_loss = float('inf')

    for step in range(total_steps):
        # μP-safe LR schedule: multiply initial LRs by schedule factor
        mult = get_lr_multiplier(step, warmup_steps, total_steps)
        for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
            pg['lr'] = init_lr * mult

        # Current effective base LR (for logging)
        current_lr = config['learning_rate'] * mult

        # Forward/backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = get_batch(train_data, block_size, micro_batch, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            batch_loss += loss.item()

        # Gradient clipping + optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step


        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            step_entry = {
                'step': step + 1,
                'train_loss': round(batch_loss, 4),
                'lr': round(current_lr, 8),
                'tokens_per_sec': round(tok_per_sec),
                'elapsed_sec': round(elapsed, 1),
            }
            train_log['steps'].append(step_entry)


        if (step + 1) % eval_interval == 0 or step == total_steps - 1:
            val_loss = estimate_loss(model, val_data, block_size, micro_batch, device)
            elapsed = time.time() - t0
            gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

            best_val_loss = min(best_val_loss, val_loss)

            print(f"Step {step+1:>5}/{total_steps} | "
                  f"train: {batch_loss:.4f} | val: {val_loss:.4f} | "
                  f"lr: {current_lr:.2e} | tok/s: {tok_per_sec:,.0f} | "
                  f"GPU: {gpu_mem_gb:.1f}GB | {elapsed:.0f}s")

            if train_log['steps'] and train_log['steps'][-1]['step'] == step + 1:
                train_log['steps'][-1]['val_loss'] = round(val_loss, 4)
                train_log['steps'][-1]['gpu_mem_gb'] = round(gpu_mem_gb, 2)


    total_time = time.time() - t0
    final_val_loss = estimate_loss(model, val_data, block_size, micro_batch, device)
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    train_log['final_val_loss'] = round(final_val_loss, 4)
    train_log['best_val_loss'] = round(best_val_loss, 4)
    train_log['wall_clock_seconds'] = round(total_time, 1)
    train_log['peak_gpu_memory_gb'] = round(gpu_mem_gb, 2)
    train_log['avg_tokens_per_sec'] = round(tokens_processed / total_time) if total_time > 0 else 0
    train_log['total_tokens_processed'] = tokens_processed

    # Save log
    log_path = os.path.join(save_dir, f"{model_name}_log.json")
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    print(f"\n{model_name} (μP) done: val_loss={final_val_loss:.4f}, "
          f"time={total_time:.0f}s, peak_mem={gpu_mem_gb:.1f}GB")
    print(f"Log saved: {log_path}")

    return train_log

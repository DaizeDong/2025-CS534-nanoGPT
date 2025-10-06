"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Examples:
- Single GPU:
  python train.py --batch_size=32 --compile=False --csv_path=runs/s1/metrics.csv
- 4 GPUs (1 node):
  torchrun --standalone --nproc_per_node=4 train.py --csv_path=runs/4gpu/metrics.csv
- 8 GPUs (1 node):
  torchrun --standalone --nproc_per_node=8 train.py --csv_path=runs/8gpu/metrics.csv
"""

import os
import time
import math
import pickle
import csv
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values (nanoGPT-style)
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# CSV logging (format described below); set empty to disable
csv_path = ''  # e.g., 'runs/4gpu_fixed/metrics.csv'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # sim larger batch sizes
batch_size = 12  # micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# lr scheduler
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
# allow overrides from config files / CLI
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP init
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
global_batch_size = gradient_accumulation_steps * ddp_world_size * batch_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"effective global batch size (sequences) will be: {global_batch_size:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# CSV logger with elapsed time + segment-average MFU
# CSV schema (comma-separated):
#   type,x,loss,elapsed_s,avg_mfu
# Where:
#   - type ∈ {"iter","train_step","val_step"}
#   - x    = iteration number (int)
#   - loss = float (train-step: train loss; val-step: val loss; iter: scaled lossf)
#   - elapsed_s = seconds since training loop started (float)
#   - avg_mfu   = average MFU(%) over the *segment* since the previous CSV write
#                 (we accumulate raw per-iter MFU estimates and average them;
#                  then reset the accumulator at each CSV write)
_csv_enabled = bool(csv_path) and master_process
_run_t0: float = time.time()
_seg_mfu_sum: float = 0.0
_seg_mfu_count: int = 0


def _csv_prepare():
    if not _csv_enabled:
        return
    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['type', 'x', 'loss', 'elapsed_s', 'avg_mfu'])


def _csv_flush(row_type: str, x_val: int, loss_val: float, elapsed_s: float, avg_mfu: Optional[float]):
    """Write one CSV row and reset the segment MFU accumulator."""
    global _seg_mfu_sum, _seg_mfu_count
    if not _csv_enabled:
        _seg_mfu_sum = _seg_mfu_count = 0
        return
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([row_type, int(x_val), float(loss_val), float(elapsed_s), (float(avg_mfu) if avg_mfu is not None else "")])
    # reset segment accumulator *after* writing
    _seg_mfu_sum = 0.0
    _seg_mfu_count = 0


def _seg_mfu_add(mfu_percent: float):
    """Accumulate MFU(%) for the current segment."""
    global _seg_mfu_sum, _seg_mfu_count
    _seg_mfu_sum += float(mfu_percent)
    _seg_mfu_count += 1


def _seg_mfu_avg() -> Optional[float]:
    return (_seg_mfu_sum / _seg_mfu_count) if _seg_mfu_count > 0 else None


_csv_prepare()

# -----------------------------------------------------------------------------
# data loading
data_dir = os.path.join('data', dataset)


def get_batch(split):
    # per nanoGPT guidance, re-open memmap each time to avoid leaks
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# model init
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# optimizer & scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free

if compile:
    print("compiling the model... (takes ~1 min)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# training loop
X, Y = get_batch('train')
t0 = time.time()
run_wall_t0 = t0
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

while True:
    # set LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # periodic eval (+ CSV rows for train/val)
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_loss = float(losses['train'])
        val_loss = float(losses['val'])
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        elapsed = time.time() - run_wall_t0
        avg_mfu = _seg_mfu_avg()
        _csv_flush('train_step', iter_num, train_loss, elapsed, avg_mfu)
        _csv_flush('val_step', iter_num, val_loss, elapsed, avg_mfu)

        if wandb_log:
            wandb.log({"iter": iter_num,
                       "train/loss": train_loss,
                       "val/loss": val_loss,
                       "lr": lr,
                       "mfu": running_mfu * 100})

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # gradient accumulation steps
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # prefetch next batch
        X, Y = get_batch('train')
        # backward
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    # iteration logging (+ CSV 'iter' row with elapsed + segment-avg MFU)
    if iter_num % log_interval == 0 and master_process:
        lossf = float(loss.item() * gradient_accumulation_steps)
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        # accumulate MFU for this segment (convert to %)
        if running_mfu >= 0:
            _seg_mfu_add(running_mfu * 100.0)

        # flush an 'iter' CSV row each time we print
        elapsed = time.time() - run_wall_t0
        avg_mfu = _seg_mfu_avg()
        _csv_flush('iter', iter_num, lossf, elapsed, avg_mfu)

    iter_num += 1
    local_iter_num += 1

    # termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

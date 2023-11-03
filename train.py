import copy
import math
import os
import argparse
import pathlib
import pprint
import random
import time

import numpy as np
import torch
from torch import nn

import utils
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AttentionModel
from plot import plot_schedule
from problem import JSSDataset, compute_cost
from reinforce_baselines import Baseline, Rollout
from eval import validate


def train(model, opts):
    """
    Train the model
    """
    optimizer = Adam(model.parameters(), lr=opts.lr)
    if opts.baseline == 'rollout':
        baseline = Rollout(model, opts)
    elif opts.baseline == 'no_baseline':
        baseline = Baseline()
    else:
        raise ValueError('Unknown baseline:', opts.baseline)
    print("Building validation dataset...")
    val_dataset = JSSDataset(opts.num_jobs, opts.num_machines, opts.val_size)

    for epoch in range(opts.num_epochs):
        train_epoch(model, optimizer, baseline, val_dataset, epoch, opts)
        baseline.update(model)  # update neural model-based baselines


def train_epoch(model, optimizer, baseline, val_dataset, epoch, opts):
    """
    Trains one epoch. At every epoch a new dataset is created
    """
    if epoch != 0:
        print('\n' + '─'*os.get_terminal_size().columns)

    print("Building training dataset...")
    training_dataset = JSSDataset(opts.num_jobs, opts.num_machines, opts.epoch_size)
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=opts.batch_size * max(1, torch.cuda.device_count()),
                                     num_workers=4,
                                     pin_memory=True,
                                     shuffle=True)

    print(f"Start train epoch {epoch}, lr={optimizer.param_groups[0]['lr']}, for run {opts.run_name}")
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    model.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(model, optimizer, baseline, batch, step, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print(f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))} s")

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.num_epochs - 1:
        print('Saving model and state...')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'baseline': baseline.state_dict()
        }, os.path.join(opts.save_dir, f'epoch-{epoch}.pt'))

    print(f"Validating...")
    cost = validate(model, val_dataset, opts)
    avg_cost = cost.mean()
    print(f'Validation overall for epoch {epoch} avg_cost: {avg_cost} +- {torch.std(cost) / math.sqrt(len(cost))}')

    upload_image = opts.upload_image_step > 0 and epoch % opts.upload_image_step == 0
    wandb.log({
        'val_avg_cost': avg_cost
    }, commit=not upload_image)

    if upload_image:
        # show schedule
        model.eval()
        with torch.no_grad():
            sequence, _ = model(val_dataset.data[0][None, ...].to(opts.device), decode_strategy='greedy')
            _, schedule = compute_cost(sequence, opts.num_jobs, opts.num_machines)
            fig = plot_schedule(schedule[0].cpu(), val_dataset.data[0], opts.num_jobs, opts.num_machines, dpi=80, show=False)
            wandb.log({"schedule": wandb.Image(fig)})


def train_batch(model, optimizer, baseline, batch, step, opts):
    """
    Trains one batch. At the end of every batch, the model is optimized
    """
    batch = batch.to(opts.device)

    sequence, log_likelihood = model(batch)
    cost, _ = compute_cost(sequence, opts.num_jobs, opts.num_machines)

    bl_cost = baseline(batch)

    loss = ((cost - bl_cost) * log_likelihood).mean()

    for param in model.parameters():  # zero out gradients
        param.grad = None
    loss.backward()

    grad_norms = [torch.nn.utils.clip_grad_norm_(
        group['params'], math.inf if opts.grad_clip == -1 else opts.grad_clip, norm_type=2)
        for group in optimizer.param_groups]

    optimizer.step()

    # optimize critic
    critic_loss = baseline.step(cost)

    # log
    avg_cost = cost.mean().item()

    if step % opts.log_step == 0:
        wandb.log({
            'train_avg_cost': avg_cost,
            'loss': loss,
            'grad_norm': grad_norms[0],
            'critic_loss': critic_loss
        })
# ______________________________________________________________________________________________________________________


def get_opts(args=None):
    """
    Default training options
    """
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Scheduling problems with Deep Reinforcement Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_jobs', type=int, default=6, help="Number of jobs of the scheduling problem")
    parser.add_argument('--num_machines', type=int, default=6, help="Number of machines of the scheduling problem")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=640000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='Number of instances used for validation batch')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--feed_forward_dim', type=int, default=512, help='Dimension of feed forward layers in the encoder')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads in the attention-based encoder')
    parser.add_argument('--num_attention_layers', type=int, default=3, help='Number of attention layers in the encoder network')
    parser.add_argument('--lr', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights and Biases logging')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping (-1 = no clipping)')
    parser.add_argument('--log_step', type=int, default=50, help='Log every log_step steps')
    parser.add_argument('--baseline', type=str, default='rollout', help='Reinforce baseline used [rollout, fifo, no_baseline]')
    parser.add_argument('--upload_image_step', type=int, default=-1, help='Upload image of schedule after N steps')

    opts = parser.parse_args(args)

    opts.run_name = f'{opts.run_name}_{time.strftime("%Y%m%dT%H%M%S")}'

    opts.save_dir = os.path.join(opts.output_dir, f"jss_{opts.num_jobs}_{opts.num_machines}", opts.run_name)
    pathlib.Path(opts.save_dir).mkdir(parents=True, exist_ok=True)

    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.save_opts(opts, os.path.join(opts.save_dir, 'args.json'))

    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    return opts


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True

    opts = get_opts()
    pprint.pprint(vars(opts))

    wandb.init(
        project='Deep-RL-scheduling',
        name=f'{opts.run_name}_PRETRAINING',
        group=opts.run_name,
        mode='disabled' if opts.no_wandb else 'online',
        config=vars(opts))

    model = AttentionModel(
        num_jobs=opts.num_jobs,
        num_machines=opts.num_machines,
        embedding_dim=opts.embedding_dim,
        num_attention_layers=opts.num_attention_layers,
        num_attention_heads=opts.num_attention_heads,
        feed_forward_dim=opts.feed_forward_dim
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(opts.device)
    wandb.watch(model, log_freq=50, log='all')
    train(model, opts)

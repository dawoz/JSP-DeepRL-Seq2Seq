import argparse
import math
import os
import pathlib
import pprint
import random
import time
from datetime import timedelta

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
import wandb

from model import DataParallel
from problem import compute_cost, augment_batch


def efficient_active_search_emb(model, batch_instances, i_batch, opts):
    # augment batch of instances: (batch_size * aug_size, ...)
    batch_instances = augment_batch(batch_instances, opts.num_augmentations, opts).to(opts.device)

    # reset idxs of RL search and IL
    model.reset_instances_idx(opts.num_instances_batch, opts.num_augmentations, opts.parallel_runs)

    # compute embeddings and set up optimizer
    model.requires_grad_(False)
    with torch.no_grad():
        embeddings = model.encode(batch_instances)
    embeddings.requires_grad_(True)
    optimizer = Adam([embeddings], lr=opts.lr)

    best_costs = None
    best_solutions = None  # best schedules
    incumbent_batch_idx = None  # index of the instances referred to the best solutions
    incumbent_seq = None  # sequence of actions for best solutions

    for epoch in tqdm(range(opts.num_epochs)):
        # same forward pass for RL search and IL
        sequences, permutations, log_likelihood = model.decode_and_imitate(batch_instances,
                                                                           embeddings,
                                                                           incumbent_batch_idx=incumbent_batch_idx,
                                                                           incumbent_seq=incumbent_seq)
        # compute cost (only for RL) and POMO baseline
        cost, solutions = compute_cost(permutations[:model.IL_batch_idx], opts.num_jobs, opts.num_machines)
        bl_cost = cost.view(-1, opts.parallel_runs).mean(-1).repeat_interleave(opts.parallel_runs, 0)

        # compute losses
        loss_RL = ((cost - bl_cost) * log_likelihood[:model.IL_batch_idx]).mean()
        loss_IL = - log_likelihood[model.IL_batch_idx:].mean() if incumbent_batch_idx is not None else 0
        loss = loss_RL + opts.param_lambda * loss_IL

        # zero out gradients and backpropagate
        optimizer.zero_grad(True)
        loss.backward()

        # clip grad norms and update weights
        grad_norms = [torch.nn.utils.clip_grad_norm_(
            group['params'], math.inf if opts.grad_clip == -1 else opts.grad_clip, norm_type=2)
            for group in optimizer.param_groups]
        optimizer.step()

        # update incumbent solutions
        incumbent_batch_idx = (
                cost.view(-1, opts.parallel_runs).argmin(-1) +
                opts.parallel_runs * torch.arange(opts.num_instances_batch * opts.num_augmentations,
                                                  device=batch_instances.device)
        ).cpu()
        incumbent_seq = sequences[incumbent_batch_idx].cpu()

        # update best cost and solution
        if best_solutions is None:
            best_solutions = solutions[incumbent_batch_idx]
            best_costs = cost[incumbent_batch_idx]
        else:
            best_solutions = torch.where((cost[incumbent_batch_idx] < best_costs).unsqueeze(-1),
                                         solutions[incumbent_batch_idx],
                                         best_solutions)
            best_costs = torch.minimum(cost[incumbent_batch_idx], best_costs)

        # log
        best_cost_single_instances = best_costs.view(-1, opts.num_augmentations).min(-1)[0]
        log = {f'instance_{i + opts.num_instances_batch * i_batch}/cost': c
               for i, c in enumerate(best_cost_single_instances)}
        log.update({f'batch_{i_batch}/train_avg_cost': cost.mean(),
                    f'batch_{i_batch}/loss': loss,
                    f'batch_{i_batch}/loss_RL': loss_RL,
                    f'batch_{i_batch}/loss_IL': loss_IL,
                    f'batch_{i_batch}/grad_norm': grad_norms[0]})
        wandb.log(log)

    # consider best cost from all augmentations
    best_costs = best_costs.view(-1, opts.num_augmentations)
    best_solutions = best_solutions.view(-1, opts.num_augmentations)
    j = best_costs.argmin(-1)
    best_cost_for_instance = best_costs[range(opts.num_instances_batch), j]
    best_solution_for_instance = best_solutions[range(opts.num_instances_batch), j]

    return best_cost_for_instance.cpu(), best_solution_for_instance.cpu()


# ______________________________________________________________________________________________________________________


def get_opts(args=None):
    parser = argparse.ArgumentParser(description="Run efficient active search on a test dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, help="Filename of the dataset")
    parser.add_argument('--model', type=str, help="Path to trained model to finetune")
    parser.add_argument('--num_jobs', type=int, default=6, help="Number of jobs of the scheduling problem")
    parser.add_argument('--num_machines', type=int, default=6, help="Number of machines of the scheduling problem")
    parser.add_argument('--seed', type=int, default=1234, help='Seed for computation')
    parser.add_argument('--num_instances_batch', type=int, default=10, help='Number of instances solved in parallel in a EAS iteration')
    parser.add_argument('--parallel_runs', type=int, default=12, help='Number of solutions to find per instance')
    parser.add_argument('--num_augmentations', type=int, default=1, help='Number of instance augmentations')
    parser.add_argument('--param_lambda', type=float, default=1, help='Imitation learning coefficient')
    parser.add_argument('--lr', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--num_epochs', type=int, default=40, help='The number of epochs to train')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs_eas', help='Directory to write output models to')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights and Biases logging')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping (-1 = no clipping)')
    parser.add_argument('--log_step', type=float, default=50, help='Log step for model parameters and gradients')

    opts = parser.parse_args(args)

    opts.run_name = f'{opts.run_name}_{time.strftime("%Y%m%dT%H%M%S")}'

    opts.save_dir = os.path.join(opts.output_dir, f"jss_{opts.num_jobs}_{opts.num_machines}", opts.run_name)
    pathlib.Path(opts.save_dir).mkdir(parents=True, exist_ok=True)

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.save_opts(opts, os.path.join(opts.save_dir, 'args.json'))

    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    return opts


if __name__ == '__main__':
    opts = get_opts()
    pprint.pprint(vars(opts))

    dataset = utils.load_dataset(opts.dataset)

    model, training_opts = utils.load_model(opts.model)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(opts.device)

    wandb.init(
        project='Deep-RL-scheduling',
        name=f'{training_opts.run_name}___EAS___{time.strftime("%Y%m%dT%H%M%S")}',
        group=f'{training_opts.run_name}',
        mode='disabled' if opts.no_wandb else 'online',
        config=vars(opts))
    wandb.watch(model, log_freq=opts.log_step, log='all')

    model.train()
    dataloader = DataLoader(
            dataset,
            batch_size=max(1, torch.cuda.device_count()) * opts.num_instances_batch,
            pin_memory=True,
            num_workers=4)
    tot_batches = math.ceil(dataset.size / opts.num_instances_batch)
    wall_time_start = time.time()

    batch_time_tot = 0
    costs = []

    for i, batch_instances in enumerate(dataloader):
        print(f'EAS batch {i + 1}/{tot_batches}')
        batch_time_start = time.time()

        # run eas
        solution_cost, _ = efficient_active_search_emb(model, batch_instances, i, opts)

        batch_time = time.time() - batch_time_start
        batch_time_tot += batch_time
        costs.append(solution_cost)
        print(f'Done. Took {timedelta(seconds=batch_time)}')
        if i+1 < tot_batches:
            print('─' * os.get_terminal_size().columns)

    # wandb log
    costs = torch.cat(costs)

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=100, figsize=(10, 5))
    plt.hist(costs, histtype='step', color='blue')
    plt.axvline(x=costs.mean(), color='red', linestyle='dashed', label=f'$\mu={costs.mean():.2f}$')
    plt.legend()

    wandb.log({'overall/cost': wandb.Image(fig)})
    wandb.run.summary['avg_instance_time'] = batch_time_tot / dataset.size

    print('=' * os.get_terminal_size().columns)
    print(f'Done. Took {timedelta(seconds=time.time() - wall_time_start)}')
    print(f'Average cost: {costs.mean():.5f} +- {costs.std():.5f} s')

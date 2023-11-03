import argparse
import math
import os
import pathlib
import pprint
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
import wandb
from problem import JSSDataset, compute_cost, augment_batch
from reinforce_baselines import EMABaseline


def active_search(model, baseline, instance, n_instance, opts):
    """
    Run active search on a given instance with a (pretrained) model. Returns the best cost, the best solution
    and the cost history
    """
    optimizer = Adam(model.parameters(), lr=opts.lr)

    # create dataset with instance augmentation
    train_dataset = JSSDataset.from_data(
        opts.num_jobs,
        opts.num_machines,
        augment_batch(instance.unsqueeze(0), opts.epoch_size, opts)
    )

    cost_history = []
    best_cost = torch.inf
    best_solution = None

    for epoch in range(opts.num_epochs):

        cost, solution = active_search_iteration(model, optimizer, baseline, train_dataset, epoch, n_instance, opts)
        wandb.log({f'instance_{n_instance}/cost': cost})
        cost_history.append(cost)

        if cost < best_cost:
            print('Found better solution!')
            best_cost = cost
            best_solution = solution

    cost_history = torch.tensor(cost_history)

    return best_cost, best_solution, cost_history


def active_search_iteration(model, optimizer, baseline, train_dataset, epoch, n_instance, opts):
    """
    Runs one iteration of active search on a dataset. Returns the best cost and the best solution
    """
    if epoch != 0:
        print('\n' + '. ' * (os.get_terminal_size().columns // 2))

    training_dataloader = DataLoader(train_dataset,
                                     batch_size=opts.batch_size * max(1, torch.cuda.device_count()),
                                     num_workers=4,
                                     pin_memory=True)

    print(f"Active search iteration {epoch}, lr={optimizer.param_groups[0]['lr']}, for run {opts.run_name}")
    step = epoch * np.ceil(opts.epoch_size / (opts.batch_size * max(1, torch.cuda.device_count())))
    start_time = time.time()

    model.train()

    best_cost = torch.inf
    best_solution = None

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        cost, solution = active_search_on_batch(model, optimizer, baseline, batch, step, n_instance, opts)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        step += 1

    epoch_duration = time.time() - start_time
    print(f"Finished iteration {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))} s")
    print(f'Solution cost for iteration {epoch}: {best_cost}')

    return best_cost, best_solution


def active_search_on_batch(model, optimizer, baseline, batch, step, n_instance, opts):
    """
    Processes one batch. The baseline is an exponential moving average of the previous best costs
    """
    batch = batch.to(opts.device)

    sequence, log_likelihood = model(batch)
    cost, _ = compute_cost(sequence, opts.num_jobs, opts.num_machines)

    # EMA
    bl_cost = baseline(batch)

    loss = ((cost - bl_cost) * log_likelihood).mean()

    # zero out grads and backpropagate
    optimizer.zero_grad(True)
    loss.backward()

    grad_norms = [torch.nn.utils.clip_grad_norm_(
        group['params'], math.inf if opts.grad_clip == -1 else opts.grad_clip, norm_type=2)
        for group in optimizer.param_groups]

    optimizer.step()

    # log
    j = cost.argmin()
    baseline.update_value(cost[j])

    if step % opts.log_step == 0:
        log = {f'instance_{n_instance}/train_avg_cost': cost.mean(),
               f'instance_{n_instance}/loss': loss,
               f'instance_{n_instance}/grad_norm': grad_norms[0]}
        wandb.log(log)

    return cost[j], sequence[j]


# ______________________________________________________________________________________________________________________


def get_opts(args=None):
    parser = argparse.ArgumentParser(description="Run active search (~RL finetuning) on a test dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, help="Filename of the dataset")
    parser.add_argument('--model', type=str, help="Path to trained model to finetune")
    parser.add_argument('--num_jobs', type=int, default=6, help="Number of jobs of the scheduling problem")
    parser.add_argument('--num_machines', type=int, default=6, help="Number of machines of the scheduling problem")
    parser.add_argument('--seed', type=int, default=1234, help='Seed for computation')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=10240, help='Number of instances per epoch during training')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--lr', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to train')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs_active_search', help='Directory to write output models to')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights and Biases logging')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping (-1 = no clipping)')
    parser.add_argument('--log_step', type=int, default=50, help='Log every log_step steps')
    parser.add_argument('--baseline_alpha', type=float, default=0.99, help='Exponential Moving Average coefficient for baseline')

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

    test_dataset = utils.load_dataset(opts.dataset)

    model, training_opts = utils.load_model(opts.model)

    # save model state to reset after every run of active search on one instance
    model_initial_state = model.state_dict()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(opts.device)

    wandb.init(
        project='Deep-RL-scheduling',
        name=f'{training_opts.run_name}___ACTIVE_SEARCH___{time.strftime("%Y%m%dT%H%M%S")}',
        group=f'{training_opts.run_name}',
        mode='disabled' if opts.no_wandb else 'online',
        config=vars(opts))
    wandb.watch(model, log_freq=opts.log_step, log='all')

    times_all = []
    costs_all = []
    cost_history_all = []

    for i, instance in enumerate(test_dataset.data):
        if i != 0:
            print('\n' + '─' * os.get_terminal_size().columns)

        print(f'Solving instance {i}\n')
        start_time = time.perf_counter()

        # set up baseline
        baseline = EMABaseline(opts.baseline_alpha)

        # run search
        solution_cost, solution, cost_history = active_search(model, baseline, instance, i, opts)

        times_all.append(time.perf_counter() - start_time)
        costs_all.append(solution_cost)
        cost_history_all.append(cost_history)

        print(f'\n===== Best objective found: {solution_cost} =====')

        # reset models to pretrained state
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(model_initial_state)
        else:
            model.load_state_dict(model_initial_state)

    # plots
    times_all = torch.tensor(times_all)
    costs_all = torch.tensor(costs_all)
    cost_history_all = torch.stack(cost_history_all)

    data = [[i, c] for i, c in enumerate(cost_history_all.mean(0))]
    table = wandb.Table(data=data, columns=["epoch", "avg_best_cost"])
    log = {'overall/avg_best_cost': wandb.plot.line(table, "epoch", "avg_best_cost", title="avg_best_cost")}

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(dpi=100, figsize=(10, 5))
    plt.hist(times_all, histtype='step', color='blue')
    plt.axvline(x=times_all.mean(), color='red', linestyle='dashed', label=f'$\mu={times_all.mean():.2f}$ s')
    plt.legend()
    log['overall/time'] = wandb.Image(fig)

    fig = plt.figure(dpi=100, figsize=(10, 5))
    plt.hist(costs_all, histtype='step', color='blue')
    plt.axvline(x=costs_all.mean(), color='red', linestyle='dashed', label=f'$\mu={costs_all.mean():.2f}$')
    plt.legend()
    log['overall/cost'] = wandb.Image(fig)

    wandb.log(log)

    print('\n' + '─' * os.get_terminal_size().columns)
    print(f'Instances solved. Time: {times_all.mean():.5f} +- {times_all.std():.5f} s')
    print(f'Average cost: {costs_all.mean():.5f} +- {costs_all.std():.5f} s')

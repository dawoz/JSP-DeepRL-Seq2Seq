from torch.utils.data import Dataset
import torch

import numpy as np


class JSSDataset(Dataset):
    """
    Random dataset of JSP instances generated with Taillard's method
    """
    def __init__(self, num_jobs=None, num_machines=None, size=None):
        if None in (num_jobs, num_machines, size):  # for JSSDataset.from_data() function
            return

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.size = size
        nm = num_jobs*num_machines

        # pairs (job_idx, op_idx)
        job_op = torch.cartesian_prod(torch.arange(num_jobs), torch.arange(num_machines)).repeat(size, 1)

        # random machine permutations for each job
        machines = torch.from_numpy(
            np.random.default_rng().permuted(
                np.tile(np.arange(num_machines), size*num_jobs).reshape(size*num_jobs, num_machines),
                axis=1)).reshape(-1,1)

        # random times
        times = torch.randint(0, 99, (size*nm, 1))

        ds = torch.cat((job_op, machines, times), dim=1).float() 
        self.data = [ds[i*nm:(i+1)*nm] for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_data(cls, num_jobs, num_machines, data):
        """
        Generate a JSSDataset from data, not ramdomly
        """
        inst = cls()
        inst.num_jobs = num_jobs
        inst.num_machines = num_machines
        inst.size = data.shape[0]
        inst.data = data.clone().detach()
        return inst


def compute_cost(sequence, num_jobs, num_machines):
    """
    Compute schedule and cost of a batch of permutations
    """
    sequence = sequence.long()

    # time in which machines are ready
    machine_ready = torch.zeros(sequence.shape[0], num_machines).to(sequence.device)

    # times in which next job is ready
    job_ready = torch.zeros(sequence.shape[0], num_jobs).to(sequence.device)

    # solution (batch_size, seq_len): schedule[k, j] indicates start time of operation at index j of batch element k
    schedule = torch.zeros(sequence.shape[:-1]).to(sequence.device)

    for i in range(sequence.shape[1]):
        # time in which machine (of operation at index i) AND job (of operation at index i) are ready
        ready = torch.maximum(
            torch.gather(machine_ready, 1, sequence[:, i, 2].unsqueeze(1)),
            torch.gather(job_ready, 1, sequence[:, i, 0].unsqueeze(1))
        ).squeeze(-1)

        # note ready time of the operation
        schedule.scatter_(1, (num_machines * sequence[:, i, 0] + sequence[:, i, 1]).unsqueeze(1), ready.unsqueeze(1))

        # update times when machines and jobs are free
        ready += sequence[:, i, -1]
        machine_ready.scatter_(1, sequence[:, i, 2].unsqueeze(1), ready.unsqueeze(1))
        job_ready.scatter_(1, sequence[:, i, 0].unsqueeze(1), ready.unsqueeze(1))

    return job_ready.max(1)[0], schedule


def augment_batch(batch, size, opts):
    """Augment batch by randomly shuffling job indexes (e.g. JSP with job indexes 0 and 1 swapped are identical)"""
    if size == 1:
        return batch

    # random permutation of jobs
    conseq = np.tile(np.arange(opts.num_jobs) * opts.num_machines, batch.shape[0] * size)
    random_gen = np.random.default_rng()
    randperm = random_gen.permuted(conseq.reshape(-1, opts.num_jobs), axis=-1).reshape(conseq.shape).repeat(
        opts.num_machines, axis=0)

    # contiguous offset of operations
    offset = torch.arange(opts.num_machines).repeat(opts.num_jobs * batch.shape[0] * size)

    # index machine-time
    shuffle_idx = torch.from_numpy(randperm) + offset

    # index job-op
    idx = torch.arange(batch.shape[1]).repeat(batch.shape[0] * size)

    # index batch
    batch_idx = torch.arange(batch.shape[0]).repeat_interleave(batch.shape[1] * size)

    data = torch.cat([batch[batch_idx, idx, :2], batch[batch_idx, shuffle_idx, 2:]], -1) \
           .reshape(-1, opts.num_jobs*opts.num_machines, 4)
    return data

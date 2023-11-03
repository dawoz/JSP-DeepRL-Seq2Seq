import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from problem import compute_cost


def validate(model, val_dataset, opts):
    """
    Evaluates the model on the validation set. Solutions are computes with greedy rollouts.
    """
    model.eval()
    val_dataloader = DataLoader(val_dataset,
                                opts.val_batch_size * max(1, torch.cuda.device_count()),
                                num_workers=4,
                                pin_memory=True)

    def validate_batch(batch):
        sequence, _ = model(batch.to(opts.device), decode_strategy='greedy')
        batch_cost, _ = compute_cost(sequence, opts.num_jobs, opts.num_machines)
        return batch_cost.cpu()

    with torch.no_grad():
        cost = torch.cat([validate_batch(batch) for batch in tqdm(val_dataloader, disable=opts.no_progress_bar)])

    return cost




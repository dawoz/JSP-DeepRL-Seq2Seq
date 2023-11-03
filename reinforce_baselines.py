import copy

import torch
from scipy.stats import ttest_rel

from problem import compute_cost, JSSDataset
from eval import validate


class Baseline:
    def __init__(self):
        """
        Basic baseline interface
        """
        pass

    def __call__(self, input):
        """
        Compute baseline in batched input. Returns a baseline cost
        """
        return 0

    def step(self, cost):
        """
        For critic-based baselines, optimize critic network. Returns critic's loss
        """
        return 0

    def update(self, model):
        """
        For actor-based baselines, update
        """
        pass

    def state_dict(self):
        """
        Return state dict
        """
        return {}


class Rollout(Baseline):
    def __init__(self, model, opts, dataset=None):
        """
        Rollout baseline (see Kool et al.)
        """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.opts = opts

        print('Initial baseline model validation...')
        if dataset is not None:
            self.dataset = dataset
            self.reset_dataset = False
        else:
            self.dataset = None
            self.reset_dataset = True
        self._init_dataset()

    def _init_dataset(self):
        if self.reset_dataset:
            self.dataset = JSSDataset(self.opts.num_jobs, self.opts.num_machines, self.opts.val_size)
        self.bl_cost = validate(self.model, self.dataset, self.opts)
        self.bl_mean = self.bl_cost.mean()

    def __call__(self, input):
        self.model.eval()
        with torch.no_grad():
            bl_sequence, _ = self.model(input, decode_strategy='greedy')
            bl_cost, _ = compute_cost(bl_sequence, self.opts.num_jobs, self.opts.num_machines)
        return bl_cost

    def update(self, model):
        print('Validating model for baseline\'s model update...')
        candidate_cost = validate(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_cost.mean()
        print(f'Candidate mean: {candidate_mean}, baseline mean: {self.bl_mean}, diff: {candidate_mean - self.bl_mean}')

        if self.dataset.size > 1 and candidate_mean - self.bl_mean < 0:
            t, p = ttest_rel(candidate_cost, self.bl_cost)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print(f'p-value: {p_val}')
            if p_val < 0.05:
                print('Updated baseline model')
                self.model = copy.deepcopy(model)
                self._init_dataset()
        elif candidate_mean - self.bl_mean < 0:
            print('Updated baseline model')
            self.model = copy.deepcopy(model)
            self._init_dataset()

    def state_dict(self):
        """
        Return state dict
        """
        return {'rollout': self.model.state_dict()}


class EMABaseline(Baseline):
    def __init__(self, alpha):
        """
        Exponential moving average baseline
        """
        super().__init__()
        self.value = 0
        self.alpha = alpha

    def update_value(self, value):
        self.value = self.alpha * self.value + (1 - self.alpha) * value

    def __call__(self, *args, **kwargs):
        return self.value

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


def plot_schedule(schedule, instance, num_jobs, num_machines, seed=42, dpi=150, show=True):
    """
    Plots a Gantt chart of the schedule
    """
    assert isinstance(schedule, torch.Tensor) and isinstance(instance, torch.Tensor), "schedule and instance must be torch tensors"
    assert len(schedule.shape) == 1, "schedule must be a 1-D tensor"
    assert len(instance.shape) == 2, "instance must be a 2-D tensor"
    schedule = schedule.long()
    instance = instance.long()

    rnd = random.Random(seed)
    colors = np.array([f"#{rnd.randrange(0x1000000):06x}" for __ in range(num_jobs)])

    xs = torch.unique(torch.cat((schedule, schedule[num_machines * instance[:, 0] + instance[:, 1]] + instance[:, -1])))
    makespan = xs[-1]
    fig = plt.figure(figsize=(8 if num_jobs < 10 else 20, num_machines // 2), dpi=dpi)
    plt.barh(y=-instance[:, 2],
             width=instance[:, -1],
             left=schedule[num_machines * instance[:, 0] + instance[:, 1]],
             height=1,
             linewidth=3,
             edgecolor='white',
             tick_label=instance[:, 2],
             color=colors[instance[:, 0]])
    legend_elements = [Patch(facecolor=colors[job], label=f'Job {job}') for job in torch.unique(instance[:, 0])]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.878)
    if len(xs) < 30:
        plt.xticks(xs)
        for x in xs:
            if x != 0:
                plt.axvline(x, color='black', zorder=10, linewidth=0.2, alpha=0.7)
    plt.xlabel('time')
    plt.ylabel('Machine')
    plt.title(f'Schedule makespan = {makespan}')
    if show:
        plt.show()
    return fig


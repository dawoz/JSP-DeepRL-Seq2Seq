import json
import os
import pickle
from types import SimpleNamespace

import torch
from torch import nn

from model import AttentionModel


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def save_opts(opts, filename):
    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(filename, 'w') as f:
        args = vars(opts)
        args['device'] = str(args['device'])
        json.dump(args, f)


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def load_model(path, load_optimizer_and_baseline=False):
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        epoch = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(path)
            if os.path.splitext(filename)[1] == '.pt'
        )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    with open(os.path.join(path, 'args.json'), 'r') as f:
        args = SimpleNamespace(**json.load(f))

    model = AttentionModel(
        num_jobs=args.num_jobs,
        num_machines=args.num_machines,
        embedding_dim=args.embedding_dim,
        num_attention_layers=args.num_attention_layers,
        num_attention_heads=args.num_attention_heads,
        feed_forward_dim=args.feed_forward_dim,
    )
    load_data = torch.load(model_filename, map_location=lambda storage, loc: storage)
    if all(k.startswith('module.') for k in load_data['model'].keys()):
        model = nn.DataParallel(model)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model.eval()
    if load_optimizer_and_baseline:
        return model, args, load_data.get('optimizer', None), load_data.get('baseline', None)
    else:
        return model, args

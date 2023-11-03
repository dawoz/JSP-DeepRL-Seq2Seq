import math

import torch
import torch.nn as nn


class SelfAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        """
        Self attention layer
        """
        super(SelfAttention, self).__init__(*args, **kwargs)

    def forward(self, input, *args, **kwargs):
        return super(SelfAttention, self).forward(
            query=input, key=input, value=input, need_weights=False, *args, **kwargs)[0]


class SkipConnection(nn.Module):
    def __init__(self, module):
        """
        Level that computes *input + module(input)*
        """
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        """
        Normalization layer
        """
        super(Normalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input):
        return self.normalizer(input.swapaxes(1, 2)).swapaxes(1, 2)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self, num_attention_heads, embedding_dim, feed_forward_dim):
        """
        Multi-head attention layer (Vaswani et al.)
        """
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                SelfAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    batch_first=True
                )
            ),
            Normalization(embedding_dim=embedding_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)
                )
            ),
            Normalization(embedding_dim=embedding_dim),
        )


class DataParallel(nn.DataParallel):
    """
    DataParallel that accesses to module's attributes
    """
    def __init__(self, module):
        super().__init__(module)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
# ______________________________________________________________________________________________________________________


class PointerAttention(nn.Module):
    def __init__(self, embedding_dim):
        """
        Pointing mechanism for the pointer network
        """
        super(PointerAttention, self).__init__()
        self.w1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.w2 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.v = nn.Linear(in_features=embedding_dim, out_features=1)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, encoder_hidden, decoder_hidden, mask=None, temp=1):
        a = self.v(torch.tanh(self.w1(encoder_hidden) + self.w2(decoder_hidden))).squeeze(2) / temp
        return self.softmax(a if mask is None else a.masked_fill(mask, -10000))


class PointerNetworkDecoder(nn.Module):
    """
    Pointer Network
    """
    def __init__(self, embedding_dim, num_decoder_layers):
        super(PointerNetworkDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_decoder_layers,
            bias=True,
            batch_first=True
        )
        self.attention = PointerAttention(embedding_dim=embedding_dim)

    def forward(self, input, encoder_hidden, decoder_h_c, mask=None, temp=1):
        self.lstm.flatten_parameters()
        decoder_hidden, decoder_h_c = self.lstm(input, decoder_h_c)
        return self.attention(encoder_hidden, decoder_hidden, mask, temp), decoder_h_c


# ______________________________________________________________________________________________________________________


class AttentionModel(nn.Module):
    def __init__(self,
                 num_jobs,
                 num_machines,
                 embedding_dim=128,
                 num_attention_heads=8,
                 num_attention_layers=3,
                 feed_forward_dim=512):
        """
        Pointer network with attention encoder

        :param embedding_dim: embedding dimension
        :param num_attention_heads: number of attention heads
        :param num_attention_layers: number of attention layers
        :param feed_forward_dim: dimension of the embedding for feed forward layers
        """
        super().__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        # linear layer for (job, op) and (machine, time)
        self.op_embedder = nn.Sequential(Normalization(2), nn.Linear(in_features=2, out_features=embedding_dim))
        self.mach_time_embedder = nn.Sequential(Normalization(2), nn.Linear(in_features=2, out_features=embedding_dim))

        # transformer encoder
        self.encoder = nn.Sequential(
            Normalization(embedding_dim),
            *(MultiHeadAttentionLayer(
                num_attention_heads=num_attention_heads,
                embedding_dim=embedding_dim,
                feed_forward_dim=feed_forward_dim
            ) for _ in range(num_attention_layers))
        )

        # recurrent cell and pointer mechanism
        self.lstm = nn.LSTMCell(embedding_dim, embedding_dim)
        self.attention = PointerAttention(embedding_dim)

        # start-of-sequence vectors
        self.h_sos = torch.zeros(embedding_dim)
        self.decoder_input_sos = torch.zeros(embedding_dim)

        # indexes for EAS
        self.idx_config = None
        self.idx_instances = None
        self.IL_batch_idx = None

    def reset_instances_idx(self, num_inst, num_aug, parallel_runs):
        """Indexes of instances for RL, and for IL"""
        if self.idx_config != (num_inst, num_aug, parallel_runs):
            self.idx_config = (num_inst, num_aug, parallel_runs)

            # indexes original instances (augmentations count as distinct instances)
            self.idx_instances = torch.arange(num_inst * num_aug).repeat_interleave(parallel_runs)

            # first index of incumbent solutions in batch
            self.IL_batch_idx = self.idx_instances.shape[0]

    def encode(self, input):
        """
        Computes the embeddings of the operations
        """
        op_embeddings = self.op_embedder(input[:, :, :2])
        mt_embeddings = self.mach_time_embedder(input[:, :, -2:])
        embeddings = self.encoder(op_embeddings + mt_embeddings)
        return embeddings

    def decode(self, input, encoder_hidden, decode_strategy='sample', temp=1.):
        """
        Generates the solutions of the input instances
        """
        decode = {
            'sample': lambda x: torch.distributions.Categorical(logits=x).sample(),
            'greedy': lambda x: x.max(1)[1]
        }.get(decode_strategy, None)
        assert decode is not None, "Unknown decode strategy: " + decode_strategy

        decoder_c = encoder_hidden.mean(1)
        decoder_h = self.h_sos.to(input.device).repeat(input.shape[0], 1)
        decoder_input = self.decoder_input_sos.to(input.device).repeat(input.shape[0], 1)
        log_likelihood = torch.zeros(input.shape[:-1], device=input.device)
        seq = []

        # boolean matrices for masking
        is_scheduled = torch.zeros(input.shape[:-1], device=input.device, dtype=torch.bool)
        is_not_available = input[:, :, 1] != 0

        for i in range(input.shape[1]):
            decoder_h, decoder_c = self.lstm(decoder_input, (decoder_h, decoder_c))
            logits = self.attention(encoder_hidden,
                                    decoder_h.unsqueeze(1),
                                    is_scheduled | is_not_available,  # compute mask
                                    temp)
            pred = decode(logits)

            # gather embeddings of selected ops
            decoder_input = torch.gather(encoder_hidden,
                                         1,
                                         pred[:, None, None].repeat(1, 1, encoder_hidden.shape[-1])).squeeze(1)
            log_likelihood[:, i] = torch.gather(logits, 1, pred.unsqueeze(1)).squeeze()

            # update masks and save index of selected op
            is_scheduled = is_scheduled.scatter(1, pred.unsqueeze(1), 1)
            is_not_available = is_not_available.scatter(1, (pred.unsqueeze(1) + 1).clamp_max(input.shape[1] - 1), 0)
            seq.append(pred)

        seq = torch.stack(seq, dim=1)  # (batch_size, seq_len)
        return (
            torch.gather(input, 1, seq.unsqueeze(2).expand(input.shape)),  # permutation of input
            log_likelihood.sum(1)
        )

    def decode_and_imitate(self, input, encoder_hidden, *, incumbent_batch_idx, incumbent_seq):
        """
        Generates solutions and computes the log-prob of the generation of the incumbent solutions.
        incumbent_batch_idx are the indexes of the incumbent solutions in input.
        incumbent_seq is the sequence of the incumbent solution.
        """
        # index for instances. Incumbent solutions are last
        idx = torch.cat(
            [self.idx_instances] + ([self.idx_instances[incumbent_batch_idx]] if incumbent_seq is not None else [])
        )
        log_likelihood = torch.zeros(idx.shape[0], input.shape[1], device=input.device)
        seq = []

        # boolean matrices for masking
        is_scheduled = torch.zeros(idx.shape[0], input.shape[1], device=input.device, dtype=torch.bool)
        is_not_available = input[idx, :, 1] != 0

        for i in range(input.shape[1]):
            if i == 0:  # multiple starts (POMO)
                # same log-probability for random starts
                logits = torch.where(input[idx, :, 1] == 0, 1, -10000).float().softmax(-1).log()
                p_runs = self.idx_config[-1]
                if p_runs == self.num_jobs:  # select exactly the n first operations of each job
                    pred = self.num_machines * torch.arange(self.num_jobs, device=input.device) \
                                                   .repeat(math.ceil(idx.shape[0] / self.num_jobs))[:idx.shape[0]]
                else:  # sample
                    pred = logits[:logits.shape[0] // p_runs].exp() \
                               .multinomial(p_runs, replacement=p_runs > self.num_jobs).reshape(-1)
                    pred = torch.cat([  # add idxs for incumbent solutions
                        pred, torch.zeros(idx.shape[0] - pred.shape[0], device=input.device, dtype=torch.long)
                    ])
            else:  # sample action
                decoder_h, decoder_c = self.lstm(decoder_input, (decoder_h, decoder_c) if i > 1 else None)
                logits = self.attention(encoder_hidden[idx], decoder_h.unsqueeze(1), is_scheduled | is_not_available)
                pred = logits.exp().multinomial(1).squeeze(-1)

            # teacher force incumbent solution
            if incumbent_seq is not None:
                pred[self.IL_batch_idx:] = incumbent_seq[:, i]

            # update decoder input and mask
            decoder_input = torch.gather(
                encoder_hidden[idx], 1, pred[:, None, None].repeat(1, 1, encoder_hidden.shape[-1])
            ).squeeze(1)
            log_likelihood[:, i] = torch.gather(logits, 1, pred.unsqueeze(1)).squeeze(1)
            is_scheduled = is_scheduled.scatter(1, pred.unsqueeze(1), 1)
            is_not_available = is_not_available.scatter(1, (pred.unsqueeze(1) + 1).clamp_max(input.shape[1] - 1), 0)
            seq.append(pred)

        seq = torch.stack(seq, dim=1)
        return (
            seq,  # indexes
            torch.gather(input[idx], 1, seq.unsqueeze(2).expand(idx.shape[0], *input.shape[1:])),  # permutation of input
            log_likelihood.sum(1)
        )

    def forward(self, input, decode_strategy='sample', temp=1):
        """
        Encodes the problem and creates one solution for each instance in the input
        """
        embeddings = self.encode(input)
        sequence, log_likelihood = self.decode(input, embeddings, decode_strategy, temp)
        return sequence, log_likelihood

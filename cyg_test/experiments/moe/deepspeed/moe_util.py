import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import parallelize
from alpa.model.model_util import TrainState
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig
from alpa.util import print_used_time


def compute_moe_parameter_count(num_layers,
                                hidden_size,
                                vocab_size,
                                num_expert,
                                mlp_factor=8,
                                tie_embedding=True):
    pure_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1) + \
        hidden_size * 4
    moe_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        num_expert * (hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1)) + \
        hidden_size * 4

    # embedding
    embedding_factor = 1 if tie_embedding else 2
    embedding = embedding_factor * vocab_size * (hidden_size + 1)

    if num_expert == 1:
        return pure_transformer * num_layers + embedding
    else:
        half = num_layers / 2
        return half * pure_transformer + half * moe_transformer + embedding

def compute_moe_total_flop(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       group_size,
                       vocab_size,
                       num_expert,
                       num_gpus,
                       latency,
                       mlp_factor=8,
                       checkpoint_activations=False):
    factor = 4 if checkpoint_activations else 3
    # num_layers / 2 attention block
    pure_transformer = batch_size * seq_len * (hidden_size ** 2) * (8 + 4 * mlp_factor) +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    pure_transformer = pure_transformer * factor

    # num_layers / 2 attention-moe block
    # transformer
    moe_transformer = batch_size * seq_len * (hidden_size ** 2) * 8  +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    # expert FFNs:
    # moe_transformer += 2 * batch_size * seq_len * (hidden_size ** 2) * mlp_factor * 2
    moe_transformer += 8 * batch_size * seq_len * (hidden_size**2) * mlp_factor

    # softmax
    moe_transformer += 2 * batch_size * seq_len * hidden_size * num_expert
    # top-2 gating
    moe_transformer += 2 * (batch_size * seq_len) * 2 * group_size
    # dispatch + combine
    moe_transformer += 2 * batch_size * seq_len * hidden_size * 2 * group_size * 2

    moe_transformer = moe_transformer * factor

    # vocab
    embedding = 6 * batch_size * seq_len * hidden_size * vocab_size

    total_flop = pure_transformer * num_layers / 2 + \
                 moe_transformer * num_layers / 2 + embedding
    return total_flop

def compute_moe_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = benchmark_case.model_config
    use_remat = benchmark_case.parallel_args.use_remat

    total_flop = compute_moe_total_flop(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                expert_group_size,
                                vocab_size,
                                num_experts,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_moe_parameter_count(num_layers,
                                                  hidden_size,
                                                  vocab_size,
                                                  num_experts,
                                                  mlp_factor=8)
    return total_flop, parameter_count

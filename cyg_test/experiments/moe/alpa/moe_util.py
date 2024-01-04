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

def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.adafactor(learning_rate=1e-2,
                         weight_decay_mask=weight_decay_mask)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              use_master_copy=(dtype == jnp.float16),
                              dynamic_scale=None)
    return state


def prepare_moe_input_and_model(benchmark_case,
                                add_manual_remat=None,
                                add_manual_layer_marker=None,
                                num_manual_pipeline_stages=None,
                                correct_expert_group_size=True):
    print_used_time(None)
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = model_config
    dtype = jnp.float16
    tie_word_embeddings = False

    if correct_expert_group_size:
        rang_factor = 1
        expected_expert_group_size = min(
            expert_group_size,
            batch_size * seq_len // num_micro_batches // 1 // rang_factor)
        if expected_expert_group_size != expert_group_size:
            print("- Expected expert group size should be {}, "
                  "but got {}. Will reset it".format(expected_expert_group_size,
                                                     expert_group_size))
            expert_group_size = expected_expert_group_size

    # Prepare input batch
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = FlaxMoEForLMModule(
        MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 8,  # this is specific to gspmd.
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=expert_group_size,
            expert_number=num_experts,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
        dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")
    return state, batch, rngkey



def get_train_step(parallel_method, grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=parallel_method)
    def train_step(state, batch, rng_key):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    batch["token_type_ids"],
                                    batch["position_ids"],
                                    deterministic=True,
                                    rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step
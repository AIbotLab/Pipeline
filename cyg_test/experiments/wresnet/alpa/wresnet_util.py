from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import common_utils
from jax.tree_util import PyTreeDef, tree_flatten, tree_map

import alpa
from alpa import (ShardParallel, automatic_remat, get_global_cluster,
                  parallelize, set_global_virtual_physical_mesh)
from alpa.model.wide_resnet import TrainState, get_wide_resnet
# from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time
from cyg_test.experiments.experiment_util import (
    compile_and_benchmark_pipeshard_training_executable,
    compile_and_benchmark_shard_training_executable,
    get_pipeshard_parallel_method, get_shard_parallel_method)


def compute_param_number(pytree: PyTreeDef):
    """Compute the total number of elements in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape)
    return ret

def create_learning_rate_fn():
    """Create learning rate schedule."""
    base_learning_rate = 0.1
    warmup_epochs = 5.0
    steps_per_epoch = 10000
    num_epochs = 100.0

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=warmup_epochs *
                                      steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn

def create_train_state(rngkey, model, input_images, learning_rate_fn):
    params = model.init_dummy(rngkey, input_images)
    params, batch_stats = params["params"], params["batch_stats"]

    # dynamic_scale = optim.DynamicScale()
    # from flax.training.dynamic_scale import DynamicScale
    # dynamic_scale = DynamicScale(growth_interval=10)
    dynamic_scale = None

    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=0.9,
        nesterov=True,
    )
    # print("cuiyonggan : params ",params)
    # from alpa.model.model_util import TrainState
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=dynamic_scale)
    # state = TrainState.create(apply_fn=model.apply,
    #                           params=params,
    #                           tx=tx,
    #                           use_master_copy=True,
    #                           dynamic_scale=None)
    # print("cuiyonggan : state ",state)
    return state


def prepare_wresnet_input_and_model(benchmark_case):
    print_used_time(None)
    # Model configs
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (image_size, num_layers, num_channels, width_factor, dtype) = model_config
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Prepare input batch
    num_classes = 1024
    batch = {
        "images":
            jnp.ones((batch_size, image_size, image_size, 3), dtype=dtype),
        "labels":
            jnp.ones((batch_size), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = get_wide_resnet(num_layers, width_factor, num_channels, num_classes,
                            dtype)

    rngkey = jax.random.PRNGKey(0)
    learning_rate_fn = create_learning_rate_fn()
    state = create_train_state(rngkey, model, batch["images"], learning_rate_fn)
    print_used_time("Create train state")
    return state, batch, learning_rate_fn

def cross_entropy_loss(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)

def get_train_step(learning_rate_fn,
                   use_remat,
                   num_remat_layers,
                   method,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=method)
    def train_step(state, batch):

        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {
                    "params": params,
                    "batch_stats": state.batch_stats
                },
                batch["images"],
                mutable=["batch_stats"])
            loss = cross_entropy_loss(logits, batch["labels"])
            # weight_penalty_params = jax.tree_leaves(params)
            # weight_decay = 0.0001
            # weight_l2 = sum(
            #     [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
            # weight_penalty = weight_decay * 0.5 * weight_l2
            metrics = {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
                "lr": learning_rate_fn(step)
            }
            return loss, (new_model_state, metrics)

        if isinstance(method, ShardParallel) and use_remat:
            loss_fn = automatic_remat(loss_fn, layer_num=num_remat_layers)

        step = state.step
        dynamic_scale = state.dynamic_scale

        if dynamic_scale:
            # TODO(lmzheng): handle gradient accumulation for this
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
            # dynamic loss takes care of averaging gradients across replicas
        else:
            grad_fn = grad_func(loss_fn, has_aux=True)
            grads, aux = grad_fn(state.params)
        new_model_state, metrics = aux

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"])
        if dynamic_scale:
            # if is_fin == False the gradients contain Inf/NaNs and optimizer
            # state and params should be restored (= skip this step).
            new_state = new_state.replace(
                opt_state=jax.tree_multimap(partial(jnp.where, is_fin),
                                            new_state.opt_state,
                                            state.opt_state),
                params=jax.tree_multimap(partial(jnp.where, is_fin),
                                         new_state.params, state.params))
            metrics["scale"] = dynamic_scale.scale

        # return new_state, metrics
        return new_state

    return train_step
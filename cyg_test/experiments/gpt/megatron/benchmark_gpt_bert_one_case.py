import argparse
import csv
import gc
import os
import re
import sys
import time
from functools import partial
import json

import numpy as np
import torch
from megatron import get_args, get_timers, initialize_megatron, mpu
from megatron.model import BertModel, GPTModel, ModelType
from megatron.training import setup_model_and_optimizer, train_step
from megatron.utils import average_losses_across_data_parallel_group

from cyg_test.experiments.experiment_util import benchmark_func, write_csv_row
from cyg_test.experiments.gpt.megatron.gpt_util import (
    compute_gpt_bert_statistics, get_bert_functions, get_gpt_functions)

GB = 1024**3

def benchmark_one_case(benchmark_case,output_file_name,repeat=10, niter=100,):
    # Model configs
    (model_type, global_batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = model_config
    assert parallel_mode == "uniform"
    (prefer_reduce_scatter, use_remat, dp, op, pp,
     force_batch_dim_mapping) = parallel_args

    dp_size, tensor_mp_size, pipeline_mp_size = dp, op, pp
    checkpoint_activations = use_remat

    num_gpus = dp_size * tensor_mp_size * pipeline_mp_size
    assert global_batch_size % (dp_size * num_micro_batches) == 0
    micro_batch_size = global_batch_size // dp_size // num_micro_batches

    # always use local DDP
    ddp_impl = True
    
    # Parallel configs
    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]
    sys.argv += ["--global-batch-size", str(global_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--seq-length", str(seq_len)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--DDP-impl", "local" if ddp_impl else "torch"]
    sys.argv += ["--fp16"]
    sys.argv += ["--loss-scale", "8"]
    if checkpoint_activations:
        sys.argv += ["--checkpoint-activations"]
    # sys.argv += ["--no-masked-softmax-fusion"]
    # sys.argv += ["--no-async-tensor-model-parallel-allreduce"]
    # sys.argv += ["--no-scatter-gather-tensors-in-pipeline"]
    initialize_megatron()
    args = get_args()
    args.padded_vocab_size = vocab_size
    
    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size()
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size()
    assert pipeline_mp_size == mpu.get_pipeline_model_parallel_world_size()

    # model_type = "gpt"
    # Build model
    if model_type == "gpt":
        model_provider, loss_func, forward_step = get_gpt_functions(micro_batch_size,seq_len)
    elif model_type == "bert":
        model_provider, loss_func, forward_step = get_bert_functions()
            
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, model_type=ModelType.encoder_or_decoder)

    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Benchmark step time
    for i in range(5):
        run_func()
    latencies = benchmark_func(run_func,
                        sync_func=None,
                        warmup=0,
                        repeat=repeat,
                        number=niter)
    
    # Print results
    rank = torch.distributed.get_rank()
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(0)
        json.dump([peak_mem,latencies.tolist()],open("cyg_test/experiments/gpt/megatron/tmp.json","w"))

               
if __name__ == "__main__":
    print("*****",sys.argv)
    case = eval(sys.argv[-4])
    output_file_name = sys.argv[-3]
    niter=int(sys.argv[-2])
    repeat=int(sys.argv[-1])
    del sys.argv[-1]
    del sys.argv[-1]
    del sys.argv[-1]
    del sys.argv[-1]
    benchmark_one_case(case, output_file_name, repeat, niter)
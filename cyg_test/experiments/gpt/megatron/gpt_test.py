import argparse
import json
import os
from datetime import datetime
import numpy as np

from cyg_test.experiments.experiment_util import write_csv_row
from cyg_test.experiments.gpt.megatron.gpt_benchmarks_config import (
    megatron_gpt_best_suite, megatron_gpt_search_suite)
from cyg_test.experiments.gpt.megatron.gpt_util import \
    compute_gpt_bert_statistics

gpt_suites={
    "best":megatron_gpt_best_suite,
    "search":megatron_gpt_search_suite
}

GB = 1 << 30
def run_cmd(cmd):
    # print(cmd)
    return os.system(cmd)

def benchmark_all(args):
    model = args.model
    suite = args.suite
    niter = args.niter
    repeat_num = args.repeat_num
    output_name = args.output_name
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = gpt_suites[suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.",flush=True)
        exit()

    for benchmark_case in gpt_suites[suite][num_gpus]:
        (global_batch_size, model_config, num_micro_batches, parallel_mode, parallel_args) = benchmark_case
        (seq_len, hidden_size, num_layers, num_heads, vocab_size) = model_config
        assert parallel_mode == "uniform"
        (prefer_reduce_scatter, use_remat, dp, op, pp, force_batch_dim_mapping) = parallel_args
        total_flops, parameter_count = compute_gpt_bert_statistics(benchmark_case, [0], num_gpus)
        
        benchmark_case = tuple(tuple(x) if isinstance(x, tuple) else x for x in benchmark_case)
        benchmark_case_str = str((model,) + benchmark_case)

        if args.nnodes == 1:
            # Single node
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         'cyg_test/experiments/gpt/megatron/benchmark_gpt_bert_one_case.py '
                         f'"{benchmark_case_str}" '
                         f'"{output_name}" '
                         f'"{niter}" '
                         f'"{repeat_num}"')
        else:
            # Multiple nodes
            ret = run_cmd('python3 -m torch.distributed.launch '
                         f'--nproc_per_node {args.nproc_per_node} '
                         f'--nnodes {args.nnodes} '
                         f'--node_rank {args.node_rank} '
                         f'--master_addr {args.master_addr} '
                         f'--master_port {args.master_port} '
                         'cyg_test/experiments/gpt/megatron/benchmark_gpt_bert_one_case.py '
                         f'"{benchmark_case_str}" '
                         f'"{output_name}" '
                         f'"{niter}" '
                         f'"{repeat_num}"')
        
        values = [
            model, "megatron-lm", seq_len, hidden_size, num_layers, num_heads,vocab_size, 
            num_gpus, global_batch_size, num_micro_batches, 
            prefer_reduce_scatter, use_remat, dp, op, pp,force_batch_dim_mapping,
            parameter_count/1e9,total_flops/1e12,
            ]
        
        if os.path.exists("cyg_test/experiments/gpt/megatron/tmp.json"):
            peak_mem, latencies = json.load(open("cyg_test/experiments/gpt/megatron/tmp.json"))
            os.remove("cyg_test/experiments/gpt/megatron/tmp.json")
        else:
            peak_mem = np.inf
            latencies = [np.inf]*repeat_num
                    
        values.append(peak_mem/GB)
        values.extend(latencies)
        # print(values)
        write_csv_row(output_name,values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    
    parser.add_argument("--suite", type=str, default="best")
    parser.add_argument("--niter",type=int,default=3)
    parser.add_argument("--repeat_num",type=int,default=3)
    parser.add_argument("--output_name",type=str,default="none")
    
    args = parser.parse_args()
    print("***",args)

    benchmark_all(args)

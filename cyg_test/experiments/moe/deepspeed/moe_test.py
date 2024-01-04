import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np

from cyg_test.experiments.experiment_util import write_csv_row
from cyg_test.experiments.moe.deepspeed.moe_benchmarks_config import (
    deepspeed_moe_best_suite, deepspeed_moe_search_suite)
from cyg_test.experiments.moe.deepspeed.moe_util import (
    compute_moe_parameter_count, compute_moe_total_flop)

moe_suites={
    "best":deepspeed_moe_best_suite,
    "search":deepspeed_moe_search_suite
}


def run_cmd(cmd):
    # print(cmd)
    return os.system(cmd)

def update_ds_config(filename, gradient_accumulation_steps, prefer_reduce_scatter,enable_overlapping):
    lines = list(open(filename))

    for i in range(len(lines)):
        if "gradient_accumulation_steps" in lines[i]:
            idx = lines[i].index(":")
            lines[i] = lines[i][:idx] + f": {gradient_accumulation_steps},\n"
        if "reduce_scatter" in lines[i]:
            idx = lines[i].index(":")
            lines[i] = lines[i][:idx] + f": {str(prefer_reduce_scatter).lower()},\n"
        if "overlap_comm" in lines[i]:
            idx = lines[i].index(":")
            lines[i] = lines[i][:idx] + f": {str(enable_overlapping).lower()},\n"

    with open(filename, "w") as fout:
        fout.writelines(lines)
    
def benchmark_one_case(benchmark_case,ep_size,bench_iter,nproc_per_node,nnodes,data_path,vocab_file,merge_file,config_file,output_name,enable_overlapping,use_deepspeed = True):
    (batch_size, model_config, num_micro_batches, parallel_mode, parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_expert, expert_group_size) = model_config
    (prefer_reduce_scatter, use_remat, dp_size, tp_size, pp_size, force_batch_dim_mapping) = parallel_args

    warmup_iter = 5
    gpt_options = (
        f"--model-parallel-size {tp_size} "
        f"--num-layers {num_layers} "
        f"--hidden-size {hidden_size} "
        f"--num-attention-heads {num_heads} "
        f"--seq-length {seq_len} "
        f"--max-position-embeddings {seq_len} "
        f"--batch-size {batch_size // dp_size // num_micro_batches} "
        f"--train-iters {(warmup_iter + bench_iter) * num_micro_batches} "
        f"--lr-decay-iters 320000 "
        #f"--save $CHECKPOINT_PATH "
        #f"--load $CHECKPOINT_PATH "
        f"--data-path {data_path} "
        f"--vocab-file {vocab_file} "
        f"--merge-file {merge_file} "
        f"--data-impl mmap "
        f"--split 949,50,1 "
        f"--distributed-backend nccl "
        f"--lr 1.5e-4 "
        f"--lr-decay-style cosine "
        f"--min-lr 1.0e-5 "
        f"--weight-decay 1e-2 "
        f"--clip-grad 1.0 "
        f"--warmup 0.01 "
        f"--log-interval 1 "
        f"--save-interval 10000 "
        f"--eval-interval 2000 "
        f"--eval-iters 0 "
        f"--fp16 "
        f"--fp16-lm-cross-entropy "
        f"--loss-scale 1.0 "
        f"--scattered-embeddings "
        f"--split-transformers "

        # Disable fusion optimizations because this makes
        # loading too slow.
        #f"--scaled-upper-triang-masked-softmax-fusion "
        #f"--scaled-masked-softmax-fusion "
        #f"--bias-gelu-fusion "
        #f"--bias-dropout-fusion "
    )

    if use_deepspeed:
        gpt_options += (
            "--deepspeed "
            f"--deepspeed_config {config_file} "
        )
        update_ds_config(config_file, num_micro_batches, prefer_reduce_scatter,enable_overlapping)

    if use_remat:
        gpt_options += "--checkpoint-activations "
        gpt_options += "--deepspeed-activation-checkpointing "
        gpt_options += "--checkpoint-num-layers 1 "

        # Disable other checkpoint optimizations
        # gpt_options += "--partition-activations "
        # gpt_options += "--checkpoint-in-cpu "
        # gpt_options += "--synchronize-each-layer "
        # gpt_options += "--ontigious-checkpointing "

    if num_expert > 1:
        gpt_options += "--moe "
        gpt_options += "--ep-world-size {} ".format(ep_size)
        gpt_options += "--num-experts {} ".format(str(num_expert))
        gpt_options += "--top-k 2 "
        gpt_options += "--min-capacity 4 "
        gpt_options += "--noisy-gate-policy None "
        gpt_options += "--moe-param-group "
        gpt_options += "--output_name {}".format(output_name)

    if nnodes > 1:
        host_options = "--hostfile hostfile_{}node ".format(nnodes)
    else:
        host_options = ""

    work_dir= os.environ["DEEPSPEED_PATH"] + "/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/"
    ret = run_cmd(f"PYTHONPATH={work_dir} PYTHON_VOCAB_SIZE={vocab_size} deepspeed "
                    f"{host_options}"
                    f"--num_nodes {nnodes} "
                    f"--master_port {random.randint(30000, 40000)} "
                    f"--num_gpus {nproc_per_node} "
                    f"cyg_test/experiments/moe/deepspeed/pretrain_gpt2_moe.py {gpt_options}")
    
    
def benchmark_all(args):
    model = args.model
    bench_iter = args.niter
    repeat_num = args.repeat_num
    output_name = args.output_name
    # MOE does not support stage 3
    config_file = args.config_file
    data_path = args.data_path
    vocab_file = args.vocab_file
    merge_file = args.merge_file
    nproc_per_node = args.nproc_per_node
    nnodes = args.nnodes
    enable_overlapping = args.enable_overlapping
    suite = args.suite
    
    num_gpus = args.nproc_per_node * args.nnodes
    use_deepspeed = True
    GB = 1 << 30
    
    try:
        _ = moe_suites[suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.",flush=True)
        exit()

    for benchmark_case in moe_suites[suite][num_gpus]:
        (batch_size, model_config, num_micro_batches, parallel_mode, parallel_args) = benchmark_case
        (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_expert, expert_group_size) = model_config
        
        assert parallel_mode == "uniform", "只支持 UniformParallelArgs 并行参数"
        
        (prefer_reduce_scatter, use_remat, dp_size, tp_size, pp_size, force_batch_dim_mapping) = parallel_args

        assert dp_size * tp_size == nproc_per_node*nnodes
        assert batch_size % dp_size == 0
        assert batch_size % num_micro_batches == 0
        
        if use_deepspeed:
            num_micro_batches = json.load(open(config_file))["gradient_accumulation_steps"]
        else:
            num_micro_batches = 1

        expert_group_size = batch_size * seq_len // num_micro_batches // dp_size
        mlp_factor = 8
        parameter_count = compute_moe_parameter_count(num_layers, hidden_size, vocab_size, num_expert, mlp_factor=mlp_factor)
        total_flops = compute_moe_total_flop(batch_size, seq_len, num_layers,
                                    hidden_size, expert_group_size,
                                    vocab_size, num_expert,
                                    num_gpus,
                                    0, 
                                    mlp_factor=mlp_factor)
        
        ep_size = 1
        ep_size_list = []
        while ep_size<=dp_size:
            ep_size_list.append(ep_size)
            ep_size = ep_size << 1

        for ep_size in ep_size_list:
            values = [model, "DeepSpeed", enable_overlapping, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_expert, expert_group_size,
                        num_gpus, batch_size, num_micro_batches,
                        prefer_reduce_scatter, use_remat, dp_size, tp_size, pp_size, ep_size, force_batch_dim_mapping,
                        parameter_count/1e9,total_flops/1e12]
            for i in range(1,repeat_num+1):
                print("第{}次".format(i),flush=True)
                try:
                    benchmark_one_case(benchmark_case,ep_size,bench_iter,nproc_per_node,nnodes,data_path,vocab_file,merge_file,config_file,output_name,enable_overlapping,use_deepspeed)
                    if os.path.exists("cyg_test/experiments/moe/deepspeed/tmp.json"):
                        peak_mem, latencies = json.load(open("cyg_test/experiments/moe/deepspeed/tmp.json"))
                        os.remove("cyg_test/experiments/moe/deepspeed/tmp.json")
                    peak_mem = peak_mem/GB
                except:
                    peak_mem, latencies = np.inf, [np.inf]
                if i == 1:
                    values.extend([peak_mem, latencies[0]])
                else:
                    values.append(latencies[0])
            write_csv_row(output_name,values)          
        # print(">>>>>> Alpa benchmark: sleep for 30 seconds before starting the next case.", flush=True)
        # time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="moe")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--suite", type=str, default="best")
    
    # 便于shell控制的参数
    parser.add_argument("--niter",type=int,default=3)
    parser.add_argument("--repeat_num",type=int,default=3)
    parser.add_argument("--enable_overlapping",type=str,default="true")
    parser.add_argument("--output_name",type=str,default="none")
    parser.add_argument("--config_file",type=str,default="cyg_test/experiments/moe/deepspeed/ds_zero_stage_2_moe_config.json")
    parser.add_argument("--data_path",type=str,default="cyg_test/experiments/moe/deepspeed/data/small-webtext/small-webtext")
    parser.add_argument("--vocab_file",type=str,default="cyg_test/experiments/moe/deepspeed/data/gpt2-vocab.json")
    parser.add_argument("--merge_file",type=str,default="cyg_test/experiments/moe/deepspeed/data/gpt2-merges.txt")
    
    args = parser.parse_args()

    benchmark_all(args)
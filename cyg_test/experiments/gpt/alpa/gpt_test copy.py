import argparse
import os
import re

enable_overlapping = os.environ.get("enable_overlapping", "False") == "True"
file_path = 'alpa/global_env.py'
if enable_overlapping:
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()
    # 在文件内容中进行修改（示例：将字符串 'old_text' 替换为 'new_text'）
    pattern = re.compile(r'self.enable_overlapping = .*\n')
    content = pattern.sub(r'self.enable_overlapping = True\n', content)
    content = content.replace('old_text', 'new_text')
    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.write(content)
else:
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()
    # 在文件内容中进行修改（示例：将字符串 'old_text' 替换为 'new_text'）
    pattern = re.compile(r'self.enable_overlapping = .*\n')
    content = pattern.sub('self.enable_overlapping = False\n', content)
    content = content.replace('old_text', 'new_text')
    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.write(content)
import csv

import jax
import numpy as np
import ray
from jax.lib import xla_bridge

import alpa
from alpa import (LocalPhysicalDeviceMesh, automatic_remat,
                  clear_executable_cache, get_global_cluster, global_config,
                  init, parallelize, set_global_virtual_physical_mesh,
                  shutdown)
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import (GB, disable_tqdm_globally,
                       get_num_hosts_and_num_devices, to_str_round, write_tsv)
from cyg_test.experiments.experiment_util import (
    compile_and_benchmark_pipeshard_training_executable,
    get_bpp_pipeshard_parallel_method, get_pipeshard_parallel_method,
    write_csv_row)
from cyg_test.experiments.gpt.alpa.gpt_benchmarks_config import (
    alpa_gpt_best_suite, alpa_gpt_search_suite)
from cyg_test.experiments.gpt.alpa.gpt_util import (
    compute_gpt_bert_statistics, get_train_step,
    prepare_gpt_bert_input_and_model)

gpt_suites={
    "best":alpa_gpt_best_suite,
    "search":alpa_gpt_search_suite
}

def benchmark_one_case(model,
                       benchmark_case,
                       niter,
                       num_hosts,
                       num_devices_per_host,
                       method_name,
                       disable_tqdm=False,):
    # 设置全局变量
    global_config.pipeline_sync_for_timer = False
    global_config.shard_parallel_sync_for_timer = False       
    global_config.collect_trace = False
        
    
    init(cluster="ray",num_nodes=num_hosts,num_devices_per_node=num_devices_per_host)

    
    if disable_tqdm:
        disable_tqdm_globally()
    
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    if method_name == 'alpa':
        pipeline_schedule = "1f1b"
        (method, add_manual_remat, add_manual_layer_marker,
         num_manual_pipeline_stages) = get_pipeshard_parallel_method(
                benchmark_case,
                virtual_mesh.num_devices_per_host,
                use_fine_grained_remat=True,
                pipeline_schedule=pipeline_schedule)
    elif method_name == "eager":   
        pipeline_schedule = "1f1b_overlap_friendly"
        (method, add_manual_remat, add_manual_layer_marker,
        num_manual_pipeline_stages) = get_pipeshard_parallel_method(
            benchmark_case,
            virtual_mesh.num_devices_per_host,
            use_fine_grained_remat=True,
            pipeline_schedule=pipeline_schedule)
    elif method_name == "bpp":
        pipeline_schedule = "1f1b_overlap_friendly"
        (method, add_manual_remat, add_manual_layer_marker,
        num_manual_pipeline_stages) = get_bpp_pipeshard_parallel_method(
            benchmark_case,
            virtual_mesh.num_devices_per_host,
            use_fine_grained_remat=True,
            pipeline_schedule=pipeline_schedule)
        

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model,
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages,)

    # try:
    use_grad_acc = benchmark_case.num_micro_batches > 1
    grad_func = alpa.grad if use_grad_acc else jax.grad
    train_step = get_train_step(method,grad_func)
    (latencies, max_mem_allocated, compilation_times,
    executable) = compile_and_benchmark_pipeshard_training_executable(
        benchmark_case.parallel_mode,
        niter,
        train_step,
        state, (batch, rngkey),)
    (compute_cost_file_name, forward_stage_layer_ids, submesh_shapes,
    logical_mesh_shapes, autosharding_option_dicts) = get_last_dp_result()
    metadata = {
        "forward_stage_layer_ids": forward_stage_layer_ids,
        "submesh_shapes": submesh_shapes,
        "logical_mesh_shapes": logical_mesh_shapes,
        "autosharding_option_dicts": autosharding_option_dicts,
    }
    # except:
    #     print("无解决方案")
    #     latencies = [np.inf]
    #     max_mem_allocated = np.inf
    #     clear_executable_cache()
    #     metadata = {
    #     "forward_stage_layer_ids": [],
    #     "submesh_shapes": [],
    #     "logical_mesh_shapes": [],
    #     "autosharding_option_dicts": [],}
    
    clear_executable_cache()
    shutdown()
    ray.shutdown()
    
    
    return max_mem_allocated, latencies, metadata

def benchmark_and_write_to_namespace(result_namespace, *args, **kwargs):
    result = benchmark_one_case(*args, **kwargs)
    result_namespace.result = result

def benchmark_all(args):
    model = args.model
    num_hosts = args.num_hosts
    num_devices_per_host = args.num_devices_per_host
    suite = args.suite
    niter = args.niter
    repeat_num = args.repeat_num
    method_name = args.method_name
    output_name = args.output_name
    
    disable_tqdm = True
    num_gpus = num_hosts*num_devices_per_host
    try:
        _ = gpt_suites[suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.",flush=True)
        exit()

    for benchmark_case in gpt_suites[suite][num_gpus]:
        (batch_size, model_config, num_micro_batches, parallel_mode, parallel_args) = benchmark_case
        (seq_len, hidden_size, num_layers, num_heads,vocab_size) = model_config

        values = [model, method_name, enable_overlapping, seq_len, hidden_size, num_layers, num_heads,vocab_size, 
                num_gpus, batch_size, num_micro_batches,]
        
        for i in range(1,repeat_num+1):
            print("第{}次".format(i),flush=True)
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            result_namespace = manager.Namespace()
            p = ctx.Process(target=benchmark_and_write_to_namespace,
                            args=(result_namespace, model, benchmark_case,niter,num_hosts,num_devices_per_host,method_name,disable_tqdm),
                            )
            p.start()
            p.join()
            if p.exitcode != 0:
                peak_mem, latencies, metadata = np.inf, [np.inf], {"forward_stage_layer_ids": [],
                                                                "submesh_shapes": [],
                                                                "logical_mesh_shapes": [],
                                                                "autosharding_option_dicts": [],}
            else:
                peak_mem, latencies, metadata =  result_namespace.result
            if i == 1:
                if parallel_mode == "load_solution":
                    values.extend(parallel_args)
                elif parallel_mode == "search":
                    prefer_reduce_scatter, use_remat, num_auto_layers, _ = parallel_args
                    values.extend([prefer_reduce_scatter, use_remat, num_auto_layers])
                    values.extend([metadata["forward_stage_layer_ids"],metadata["submesh_shapes"],metadata["logical_mesh_shapes"], metadata["autosharding_option_dicts"]])
                elif parallel_mode == "uniform":
                    prefer_reduce_scatter, use_remat, dp_size, tp_size, num_auto_layers, force_batch_dim_mapping = parallel_args
                    values.extend([prefer_reduce_scatter, use_remat, num_auto_layers])
                    num_mesh_devices = dp_size * tp_size
                    if num_mesh_devices <= num_devices_per_host:
                        physical_mesh_shape = (1, num_mesh_devices)
                    else:
                        assert num_mesh_devices % num_devices_per_host == 0
                        physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                            num_devices_per_host)
                    values.extend([[[i] for i in range(num_auto_layers)], [physical_mesh_shape] * num_auto_layers, [(dp_size, tp_size)] * num_auto_layers, [{}] * num_auto_layers])
                else:
                    print("{}是不支持的并行参数".format(parallel_mode))
                    exit()
                total_flop, parameter_count = compute_gpt_bert_statistics(benchmark_case, [0], num_gpus)
                values.extend([parameter_count/1e9,total_flop/1e12, peak_mem/GB,latencies[0]])
            else:
                values.append(latencies[0])
        write_csv_row(output_name,values)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--num_hosts", type=int, default=1)
    parser.add_argument("--num_devices_per_host", type=int, required=True)
    parser.add_argument("--suite", type=str, default="best")
    
    # 便于shell控制的参数
    parser.add_argument("--niter",type=int,default=3)
    parser.add_argument("--repeat_num",type=int,default=3)
    parser.add_argument("--method_name",type=str,default="alpa")
    parser.add_argument("--output_name",type=str,default="none")
  
    args = parser.parse_args()

    benchmark_all(args)

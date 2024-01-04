from collections import namedtuple

from cyg_test.experiments.experiment_util import (BenchmarkCase,
                                                  LoadSolutionParallelArgs,
                                                  SearchParallelArgs)

GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

def get_solution_case(model_spec, max_global_batch_size,num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts,
                      prefer_reduce_scatter=True,
                      use_remat=False,):
    return BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    
def get_search_cases(model_spec, 
                     max_global_batch_size, 
                     num_micro_batches_list, 
                     num_auto_layers_list,
                     auto_stage_option,
                     prefer_reduce_scatter=True,
                     use_remat=False,):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
        for num_auto_layers in num_auto_layers_list
    ]
 
gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "760-1M": GPTModelConfig(512, 768, 12, 12, 51200),
    "760-2M": GPTModelConfig(48, 768, 12, 12, 51200),
    "760-3M": GPTModelConfig(64, 768, 12, 12, 51200),
    "760-4M": GPTModelConfig(128, 768, 12, 12, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}

force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}
prefer_reduce_scatter=True
use_remat=False
max_global_batch_size = 16
# Performance test with search solutions found for p3.16xlarge
alpa_gpt_best_suite = {
    # 每个微批次的大小为3，超过之后内存会溢出
    1:
        [get_solution_case(gpt_specs["125M"], 6,2, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}],prefer_reduce_scatter,use_remat)],
    2:
        [
            get_solution_case(gpt_specs["125M"], 4,4, 12, [[0, 1, 2, 3, 4, 5,6], [7, 8, 9, 10, 11]],
                            [(1, 1)] * 2, [(1, 1)] * 2, [{}] * 2,prefer_reduce_scatter,use_remat,),
            get_solution_case(gpt_specs["125M"], 8,4, 12, [[0, 1, 2, 3, 4, 5,6], [7, 8, 9, 10, 11]],
                            [(1, 1)] * 2, [(1, 1)] * 2, [{}] * 2,prefer_reduce_scatter,use_remat,),
            get_solution_case(gpt_specs["125M"], 16,4, 12, [[0, 1, 2, 3, 4, 5,6], [7, 8, 9, 10, 11]],
                            [(1, 1)] * 2, [(1, 1)] * 2, [{}] * 2,prefer_reduce_scatter,use_remat,),
            get_solution_case(gpt_specs["125M"], 32,4, 12, [[0, 1, 2, 3, 4, 5,6], [7, 8, 9, 10, 11]],
                            [(1, 1)] * 2, [(1, 1)] * 2, [{}] * 2,prefer_reduce_scatter,use_remat,),
            get_solution_case(gpt_specs["125M"], 64,4, 12, [[0, 1, 2, 3, 4, 5,6], [7, 8, 9, 10, 11]],
                            [(1, 1)] * 2, [(1, 1)] * 2, [{}] * 2,prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-1M"], 16,8, 6, [[0, 1, 2, 3], [4, 5]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{}, {}],prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-2M"], 16,8, 12, [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [force_dp_dict] * 2,prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-3M"], 16,8, 12, [[0, 1, 2, 3, 4, 5], [6,7, 8, 9, 10, 11]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{}, {}],prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-1M"], 16,16, 12, [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{'force_batch_dim_to_mesh_dim': 0}, {}],prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-2M"], 16,16, 12, [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{}, {}],prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-3M"], 16,16, 6, [[0, 1, 2, 3], [4, 5]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{}, {}],prefer_reduce_scatter,use_remat,),
            # get_solution_case(gpt_specs["760-4M"], 16,16, 12, [[0, 1, 2, 3, 4, 5,], [6, 7, 8, 9, 10, 11]],
            #                 [(1, 1)] * 2, [(1, 1)] * 2, [{}, {}],prefer_reduce_scatter,use_remat,),
        ],
    4:
        [get_solution_case(gpt_specs["1.3B"], max_global_batch_size,128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 2)] * 2, [(2, 1)] * 2, [force_dp_dict] * 2,prefer_reduce_scatter,use_remat,),],
    8:
        [get_solution_case(gpt_specs["2.6B"], max_global_batch_size,128,
                          8, [[0, 1], [2, 3], [4, 5, 6, 7]], [(1, 2), (1, 2),
                                                              (1, 4)], [(2, 1),
                                                                        (2, 1),
                                                                        (4, 1)],
                          [force_dp_dict, {}, {}],prefer_reduce_scatter,use_remat,),],
}
import numpy as np

# Grid search on hyperparameters
auto_stage_option = {
    "submesh_physical_shape_space": "manual",
    "manually_specified_submeshes": [(1,1)],
    "submesh_logical_shape_space": "all",
    "stage_imbalance_tolerance": np.inf,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}
alpa_gpt_search_suite = {
    2: (get_search_cases(gpt_specs["760-1M"],32, [4], [12],auto_stage_option,prefer_reduce_scatter,use_remat,)
        # get_search_cases(gpt_specs["760-2M"],max_global_batch_size, [8,16], [6,12],auto_stage_option,prefer_reduce_scatter,use_remat,) +
        # get_search_cases(gpt_specs["760-3M"],max_global_batch_size, [8,16], [6,12],auto_stage_option,prefer_reduce_scatter,use_remat,) +
        # get_search_cases(gpt_specs["760-4M"],max_global_batch_size, [16,32], [6,12],auto_stage_option,prefer_reduce_scatter,use_remat,)
        ),
    # 4: (get_search_cases(gpt_specs["1.3B"],max_global_batch_size, [32, 64, 128], [6]) +
    #     get_search_cases(gpt_specs["1.3B"],max_global_batch_size, [32, 64], [12])),
    # 8: (get_search_cases(gpt_specs["2.6B"], [64, 128, 256], [8]) +
    #     get_search_cases(gpt_specs["2.6B"], [64, 128], [16])),
    # 16: get_search_cases(gpt_specs["6.7B"], [32, 64, 128, 256], [8]),
    # 32: get_search_cases(gpt_specs["15B"], [64, 128, 256, 512], [16]),
    # 64: get_search_cases(gpt_specs["39B"], [128, 256, 512, 1024], [8]),
}
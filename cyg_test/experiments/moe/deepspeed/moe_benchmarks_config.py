from collections import namedtuple

import numpy as np

from cyg_test.experiments.experiment_util import (BenchmarkCase,
                                                  UniformParallelArgs,
                                                  SearchParallelArgs,
                                                  LoadSolutionParallelArgs)


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
 
# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads, S_ = expert_group_size, E = expert_number,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,   

MoEModelConfig = namedtuple("MoEModelConfig", [
    "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size",
    "num_experts", "expert_group_size"
])
   
moe_specs = {
    #                      S,    H,   L, head, V,   E,  S_
    "380M": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048),
    "690M": MoEModelConfig(1024, 768, 8, 16, 32000, 16, 2048),
    "1.3B": MoEModelConfig(1024, 768, 16, 16, 32000, 16, 2048),
    "2.4B": MoEModelConfig(1024, 1024, 16, 16, 32000, 16, 2048),
    "7.1B": MoEModelConfig(1024, 1280, 16, 16, 32000, 32, 2048),
    "10B": MoEModelConfig(1024, 1536, 16, 16, 32000, 32, 2048),
    "27B": MoEModelConfig(1024, 2048, 16, 16, 32000, 48, 2048),
    "70B": MoEModelConfig(1024, 2048, 32, 16, 32000, 64, 2048),
    "140B": MoEModelConfig(1024, 2048, 32, 16, 32000, 128, 2048),
}

force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}
prefer_reduce_scatter=True
use_remat=False
max_global_batch_size = 64
# Performance test with search solutions found for p3.16xlarge
deepspeed_moe_best_suite = {
    1:
        [get_solution_case(moe_specs["380M"], max_global_batch_size, 512, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}],prefer_reduce_scatter,use_remat),],
    2:
        [BenchmarkCase(max_global_batch_size, moe_specs["380M"], 16, "uniform",
                      UniformParallelArgs(prefer_reduce_scatter, use_remat, 2, 1, 1, True)),],
    4:
        [get_solution_case(moe_specs["1.3B"], max_global_batch_size, 32, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 2)] * 2,
                          [(2, 1)] * 2, [force_dp_dict] * 2,prefer_reduce_scatter,use_remat),],
    8:
        [get_solution_case(moe_specs["2.4B"], max_global_batch_size, 32, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 4)] * 2,
                          [(4, 1)] * 2, [force_dp_dict] * 2,prefer_reduce_scatter,use_remat),],
}


# Grid search on hyperparameters
auto_stage_option = {
    "submesh_physical_shape_space": "manual",
    "manually_specified_submeshes": [(1,1)],
    "submesh_logical_shape_space": "all",
    "stage_imbalance_tolerance": np.inf,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}
deepspeed_moe_search_suite = {
    2: [BenchmarkCase(max_global_batch_size, moe_specs["380M"], 16, "uniform",
                      UniformParallelArgs(prefer_reduce_scatter, use_remat, 2, 1, 1, True)),
        BenchmarkCase(max_global_batch_size, moe_specs["380M"], 16, "uniform",
                      UniformParallelArgs(prefer_reduce_scatter, use_remat, 1, 2, 1, True)),
        ],
    # 4: (get_search_cases(moe_specs["1.3B"],max_global_batch_size, [32, 64, 128], [6]) +
    #     get_search_cases(moe_specs["1.3B"],max_global_batch_size, [32, 64], [12])),
    # 8: (get_search_cases(moe_specs["2.4B"], [64, 128, 256], [8]) +
    #     get_search_cases(moe_specs["2.4B"], [64, 128], [16])),
    # 16: get_search_cases(moe_specs["10B"], [32, 64, 128, 256], [8]),
    # 32: get_search_cases(moe_specs["27B"], [64, 128, 256, 512], [16]),
    # 64: get_search_cases(moe_specs["70B"], [128, 256, 512, 1024], [8]),
}

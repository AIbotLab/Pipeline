from collections import namedtuple

import numpy as np

from cyg_test.experiments.experiment_util import (BenchmarkCase,
                                                  LoadSolutionParallelArgs,
                                                  SearchParallelArgs)

WResNetModelConfig = namedtuple(
    "WResNetModelConfig",
    ["image_size", "num_layers", "num_channels", "width_factor", "dtype"])


def get_num_auto_layers(model_name):
    if wresnet_specs[model_name].num_layers == 50:
        return 16  # number of residual blocks
    elif wresnet_specs[model_name].num_layers == 101:
        return 33
    else:
        raise ValueError("Unsupported number of layers: {}".format(
            wresnet_specs[model_name].num_layers))
        

def get_solution_case(model_name, max_global_batch_size, num_micro_batches,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts,
                      prefer_reduce_scatter=True,
                      use_remat=False,):
    num_auto_layers = get_num_auto_layers(model_name)
    return BenchmarkCase(
            max_global_batch_size, wresnet_specs[model_name], num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))  

def get_search_cases(model_name, 
                     max_global_batch_size, 
                     num_micro_batches_list,
                     auto_stage_option,
                     prefer_reduce_scatter=True,
                     use_remat=False,):
    num_auto_layers = get_num_auto_layers(model_name)
    return [
        BenchmarkCase(
            max_global_batch_size, wresnet_specs[model_name], num_micro_batches, "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
    ]
    
wresnet_specs = {
    #                      I,   L,   C,   W,  dtype,
    "250M": WResNetModelConfig(224, 50, 160, 2, "fp32"),
    "500M": WResNetModelConfig(224, 50, 224, 2, "fp32"),
    "1B": WResNetModelConfig(224, 50, 320, 2, "fp32"),
    "2B": WResNetModelConfig(224, 50, 448, 2, "fp32"),
    "4B": WResNetModelConfig(224, 50, 640, 2, "fp32"),
    "6.8B": WResNetModelConfig(224, 50, 320, 16, "fp32"),
    "13B": WResNetModelConfig(224, 101, 320, 16, "fp32"),
}

force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}
prefer_reduce_scatter=True
use_remat=False
max_global_batch_size = 1024
# Performance test with search solutions found for p3.16xlarge
alpa_wresnet_best_suite = {
    1:
        [get_solution_case("250M", 1536, 24, [list(range(16))], [(1, 1)],
                          [(1, 1)], [{}],prefer_reduce_scatter,use_remat),],
    2:
        [get_solution_case("500M", 16, 4, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15]], [(1, 1), (1, 1)],
                          [(1, 1), (1, 1)], [{},{}],prefer_reduce_scatter,use_remat),],
    4:
        [get_solution_case("1B", 1536, 24, [list(range(16))], [(1, 4)], [(1, 4)],
                          [{}],prefer_reduce_scatter,use_remat),],
    8:
        [get_solution_case(
            "2B", 1536, 24,
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            [(1, 4), (1, 4)], [(4, 1), (1, 4)], [{}, force_dp_dict],prefer_reduce_scatter,use_remat),],
}

# Grid search on hyperparameters
auto_stage_option = {
    # all or power_of_two or small_power_of_two or manual
    "submesh_physical_shape_space": "small_power_of_two",
    # "manually_specified_submeshes": [(1,1)],
    # all or single_node_model_parallel or same_as_physical or data_parallel_only or model_parallel_only
    "submesh_logical_shape_space": "single_node_model_parallel",
    "stage_imbalance_tolerance": np.inf,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}
alpa_wresnet_search_suite  = {
    1:
        [],
    2:
        (get_search_cases("500M",16, [4,8,16],auto_stage_option,prefer_reduce_scatter,use_remat)
        ),
    4:
        [],
    8:
        [],
}

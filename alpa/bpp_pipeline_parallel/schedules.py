"""Generate pipeline schedules."""
import itertools
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Any, Sequence
from numpy._typing import NDArray

import numpy as np
from alpa.pipeline_parallel.schedules import PipelineSchedule
from alpa.pipeline_parallel.computation import PipelineComputation
from alpa.util import OrderedSet, cached_property

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 构建调度表
def genDependencyWithStages(
    compute_stages: List[PipelineComputation],
    apply_grad_stages: List[PipelineComputation] = ()) -> NDArray[Any]:
    """生成流水线阶段列表的依赖关系矩阵。
    修改自 gen_dependency_with_stages 函数，原始函数默认 compute_stages 按照拓扑顺序排列。
    此函数修复了此问题。

    Args:
        compute_stages (List[PipelineComputation]): 计算梯度的阶段
        apply_grad_stages (List[PipelineComputation], optional): 应用梯度的阶段. Defaults to ().

    Returns:
        NDArray[Any]: 存储流水线阶段间依赖关系的矩阵
    """    
    """Generate the dependency matrix for a list of pipeline stages."""
    n_stages = len(compute_stages) + len(apply_grad_stages)
    d = np.zeros([n_stages, n_stages], dtype=int)
    var_stage_id = {}
    # 将所有阶段的输出变量与对应阶段索引联系起来，存储到 var_stage_id
    for i, stage in enumerate(itertools.chain(compute_stages,
                                              apply_grad_stages)):
        for var in stage.outvars:
            var_stage_id[var] = i
    # 根据阶段的输入，构建依赖关系矩阵
    for i, stage in enumerate(itertools.chain(compute_stages,
                                              apply_grad_stages)):
        for var in stage.invars:
            if var in var_stage_id:
                d[i, var_stage_id[var]] = 1
            else:
                # 输入不是任何一个阶段的输出，假设 var 来自 global_invars
                pass
    return d


class PipeDreamFlushWithBackwardWeightDelay(PipelineSchedule):
    """
    生成一个 PipeDream Flush 的变形调度表（也称为1F1B）。
    其中反向传播分为关键阶段，和非关键阶段，非关键阶段可以与关键阶段的通信和计算重叠。

    """

    @property
    def name(self):
        return "1f1b_backward_weight_delay"

    def _generate_schedule(self):
        """
        这个时间表与PipeDream的时间表非常接近，但延迟反向传播，为重叠的通信和计算创造更多机会。
        F表示正向，B_C表示反向关键阶段，B_N表示反向非关键阶段
        k (i,j)     (i,j)       (i,j)
        - -------   -------     -------
        0 (0,0,F)
        1 (1,0,F)   (0,1,F)
        2 (2,0,F)   (1,1,F)     (0,2,F)
        3                       (0,2,B_C)
        4           (0,1,B_C)   (0,2,B_N)
        5 (0,0,B_C) (0,1,B_N)   (1,2,F)
        6 (0,0,B_N) (2,1,F)     (1,2,B_C)
        7 (3,0,F)   (1,1,B_C)   (1,2,B_N)
        8 (1,0,B_C) (1,1,B_N)   (2,2,F)
        9 (1,0,B_N) (3,1,F)     (2,2,B_C)
        ...
        """
        m = self.num_batch
        n = self.num_mesh

        # 完成所有前向和反向阶段需要的 clock 数
        num_clock = (m - 1) * 3 + n * 2 + 1
        # 存储前向与反向调度表
        schedules = [[None] * n for k in range(num_clock)]

        # warmup 阶段每个 mesh 上的微批数
        num_warmup_microbatches = [min(n - i - 1, m) for i in range(n)]
        # remaining 阶段每个 mesh 上的微批数
        num_microbatches_remaining = [m - i for i in num_warmup_microbatches]

        # 每个 mesh 上将执行的前向阶段微批次索引
        next_fwd_mb_idx = [0 for _ in range(n)]
        # 每个 mesh 上将执行的反向阶段微批次索引
        next_bwd_mb_idx = [0 for _ in range(n)]
        # 每个 mesh 上将执行周期索引
        next_available_clock = list(range(n))
        # 每个 mesh 每个 clock 正在执行的反向关键阶段的微批索引
        finished_bwd_critical_batch_indices = np.zeros(shape=[num_clock, n],
                                              dtype=np.int32)

        # warm-up clocks
        for i in range(n):
            for _ in range(num_warmup_microbatches[i]):
                schedules[next_available_clock[i]][i] = (next_fwd_mb_idx[i], i)
                next_available_clock[i] = next_available_clock[i] + 1
                next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1

        # run 1F1B
        for i in reversed(range(n)):
            # 从最后一个设备到第一个设备
            for _ in range(num_microbatches_remaining[i]):
                # 运行所有 remaining 的微批次
                
                # forward
                next_clock = next_available_clock[i]
                # 设置 ( batch_idx, stage_idx )
                schedules[next_clock][i] = (next_fwd_mb_idx[i], i)
                next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1
                finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1

                # backward
                # 获得执行阶段的 clock ，
                # 设置下一个可用 clock 为前一阶段刚刚完成目标微批次反向传播的时钟。
                if i + 1 < n:  # not the last device
                    # find the next possible backward clock
                    while finished_bwd_critical_batch_indices[next_clock][
                            i + 1] <= next_bwd_mb_idx[i]:
                        assert finished_bwd_critical_batch_indices[
                            next_clock - 1][i] == next_bwd_mb_idx[i]
                        finished_bwd_critical_batch_indices[next_clock][
                            i] = finished_bwd_critical_batch_indices[next_clock - 1][i]
                        next_clock = next_clock + 1
                # 关键阶段
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
                finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1
                
                # 非关键阶段
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 3 * n - 1 - i)
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_available_clock[i] = next_clock + 1
                
        # run cooldown passes
        for i in reversed(range(n)):
            for _ in range(num_warmup_microbatches[i]):
                assert i + 1 < n
                next_clock = next_available_clock[i]
                while finished_bwd_critical_batch_indices[next_clock][
                        i + 1] <= next_bwd_mb_idx[i]:
                    finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[
                        i]
                    next_clock = next_clock + 1
                # 关键阶段
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
                finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1
                
                # 非关键阶段
                schedules[next_clock][i] = (next_bwd_mb_idx[i], 3 * n - 1 - i)
                next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
                finished_bwd_critical_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_available_clock[i] = next_clock + 1
            # update status matrix for the last worker
            if i > 0:
                finished_bwd_critical_batch_indices[next_available_clock[i]:num_clock,
                                           i] = m

        # append apply_grad schedules
        scheds = [None] * n
        for stage_idx, worker in self.apply_grad_placement.items():
            scheds[worker] = (self.last_backward_batch_index, stage_idx)
        schedules.append(scheds)
        return schedules

    @property
    def first_backward_batch_index(self):
        """Return the index of the first microbatch at backward pass."""
        return 0

    @property
    def last_backward_batch_index(self):
        """Return the index of the last microbatch at backward pass."""
        return self.num_batch - 1

    def previous_backward_batch_index(self, batch_idx):
        """Return the index of the previous microbatch at backward pass."""
        assert batch_idx > 0
        return batch_idx - 1  

from typing import Any, List, Sequence

import networkx as nx
import numpy as np
from jax.core import (AbstractValue, Atom, ClosedJaxpr, DropVar, Jaxpr,
                      JaxprEqn, Literal, Primitive, ShapedArray, Var, gensym)
from numpy._typing import NDArray

from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, PipelineComputation,
    merge_marked_jaxprs_with_named_call)
from alpa.pipeline_parallel.layer_stats import eqn_flops
from alpa.pipeline_parallel.primitive_def import (mark_pipeline_jaxpreqn,
                                                  pipeline_p)
from alpa.util import (OrderedSet, clone_jaxpr, clone_jaxpr_eqn,
                       compile_dummy_zero_constant, get_compile_options,
                       get_var_mapping, jaxpr_to_hlo, new_jaxpr_eqn,
                       replicated_sharding_spec_proto, setup_computation_alias,
                       slices_to_jaxpr, undefined_sharding_spec_proto)

# 将 ClosedJaxpr 转为 networkx 中的 DiGraph

def translateJaxprToNetworkxWithoutVarAndPipelineMarker(close_jaxpr:ClosedJaxpr):
    """本函数生成jax-eqn的有向图，省略公式的输入输出，省略pipeline_mark公式。认为 eqn:outvar=1:n。
    此函数假定close_jaxpr中的公式，按照拓扑顺序存储。
    如此对每一个公式，如果其输入变量是另一个公式的输出变量，就将两个公式连接起来。

    Args:
        close_jaxpr (ClosedJaxpr): 输入的close_jaxpr
    """    
    def _addEqnNodeToGraph(eqn : JaxprEqn,eqn_index,graph,shape="rectangle",color="blue"):
        """向图graph中添加一个eqn节点

        Args:
            eqn (JaxprEqn): 要添加的公式
            eqn_index (_type_): 公式在jaxpr中的序号
            graph (_type_): 目标图
            shape (str, optional): 节点的形状. Defaults to "rectangle".
            color (str, optional): 节点的颜色. Defaults to "blue".

        Returns:
            name(str): 节点的名字
        """        
        name = eqn.primitive.name
        if 'name' in eqn.params:
            name = name +'_'+ eqn.params['name']
        if 'mark_type' in eqn.params:
            name = name +'_'+ eqn.params['mark_type']
        name = name+'_'+str(eqn_index)
        if 'pipeline_marker' not in name:
            lable = name
            graph.add_node(name,lable = lable,shape=shape,color=color)
        return name

    def _addEqnToGraph(eqn : JaxprEqn,eqn_index,graph,shape="rectangle",color="blue"):
        """向图中添加一个公式节点，和指向该节点的边。
        不添加pipeline_mark公式节点，而是将前后两个对应的节点连接起来

        Args:
            eqn (JaxprEqn): 要添加的公式
            eqn_index (_type_): 公式在jaxpr中的序号
            graph (_type_): 目标图
            shape (str, optional): 节点的形状. Defaults to "rectangle".
            color (str, optional): 节点的颜色. Defaults to "blue".

        """        
        # 添加公式节点
        eqn_name = _addEqnNodeToGraph(eqn,eqn_index,digraph)
        eqn_name_to_eqn_index[eqn_name] = eqn_index
        eqn_index_to_eqn_name[eqn_index] = eqn_name
        
        # 不添加 pipeline_marker 节点，直接讲前后对应节点相连
        if 'pipeline_marker' in eqn_name:
            for index,var in enumerate(eqn.invars):
                if(isinstance(var,DropVar) or isinstance(var,Literal)):
                    continue
                prior_eqn_name = outvar_id_to_eqn_name.get(id(var),None)
                if prior_eqn_name is not None:
                    outvar_id_to_eqn_name[id(eqn.outvars[index])] = prior_eqn_name
        else: 
            # 添加指向该公式节点的边，边权设置为上一个公式的 floats
            for var in eqn.invars:
                if(isinstance(var,DropVar) or isinstance(var,Literal)):
                    continue
                src_eqn_name = outvar_id_to_eqn_name.get(id(var),None)
                # pipeline_marker 中 Literal 输入对应的输出无响应公式，排除此情况对之后公式的影响
                if src_eqn_name is not None:
                    weight = eqn_index_to_floats[eqn_name_to_eqn_index[src_eqn_name]]
                    digraph.add_edge(src_eqn_name,eqn_name,weight = weight)
            
            # 更新 outvar_id_to_eqn_name
            for var in eqn.outvars:
                assert outvar_id_to_eqn_name.get(id(var),None) is None
                outvar_id_to_eqn_name[id(var)] = eqn_name
    
    outvar_id_to_eqn_name = dict()
    eqn_name_to_eqn_index=dict()
    eqn_index_to_eqn_name=dict()
    eqn_index_to_floats = dict()
    
    digraph = nx.DiGraph()
    # 添加起点和汇点
    digraph.add_node('s')
    eqn_name_to_eqn_index['s'] = -1
    eqn_index_to_eqn_name[-1] = 's'
    digraph.add_node('e')
    eqn_name_to_eqn_index['e'] = -2
    eqn_index_to_eqn_name[-2] = 'e'
    # 设置源点和汇点对应的floats，默认为1
    eqn_index_to_floats[-1] = 0
    eqn_index_to_floats[-2] = 0
    
    # 更新每个公式的floats，暂时设置为1
    for eqn_index,eqn in enumerate(close_jaxpr.jaxpr.eqns):
        eqn_index_to_floats[eqn_index] = 1
        # eqn_index_to_floats[eqn_index] = eqn_flops(eqn)
        
    # 设置全局输入变量为源点的输出变量，保证公式起源于源点
    for invar in close_jaxpr.jaxpr.invars:
        if outvar_id_to_eqn_name.get(id(invar),None) is None:
            outvar_id_to_eqn_name[id(invar)] = 's'
    
    # 将每一个公式添加到图中
    for eqn_index,eqn in enumerate(close_jaxpr.jaxpr.eqns):
        # 添加公式节点
        _addEqnToGraph(eqn,eqn_index,digraph)
                
    # 设置全局输出变量为汇点的输入变量，保证公式交于于汇点
    for outvar in close_jaxpr.jaxpr.outvars:
        # 添加全局输出变量到对应的输出公式到汇点的边
        src_eqn_name = outvar_id_to_eqn_name.get(id(outvar),None)
        assert src_eqn_name is not None
        weight = eqn_index_to_floats[eqn_name_to_eqn_index[src_eqn_name]]
        digraph.add_edge(src_eqn_name,'e',weight = weight)
    
    # 将图中出度为零的点连接到汇点，入度为零的点连接到源点
    for node in digraph.nodes():
        if node == 's' or node == 'e': continue
        node_in_degree = digraph.in_degree(node)
        node_out_degree = digraph.out_degree(node)
        if node_in_degree == 0:
            digraph.add_edge('s',node,weight = eqn_index_to_floats[-1])
        if node_out_degree == 0:
            node_weight = eqn_index_to_floats[eqn_name_to_eqn_index[node]]
            digraph.add_edge(node,'e',weight = node_weight)
    return digraph

def translateJaxprToNetworkxWithoutVar(close_jaxpr:ClosedJaxpr):
    """本函数生成jax-eqn的有向图，省略公式的输入输出。认为 eqn:outvar=1:n。
    此函数假定close_jaxpr中的公式，按照拓扑顺序存储。
    如此对每一个公式，如果其输入变量是另一个公式的输出变量，就将两个公式连接起来。

    Args:
        close_jaxpr (ClosedJaxpr): 输入的close_jaxpr
    """    
    def _addEqnNodeToGraph(eqn : JaxprEqn,eqn_index,graph,shape="rectangle",color="blue"):
        """向图graph中添加一个eqn节点

        Args:
            eqn (JaxprEqn): 要添加的公式
            eqn_index (_type_): 公式在jaxpr中的序号
            graph (_type_): 目标图
            shape (str, optional): 节点的形状. Defaults to "rectangle".
            color (str, optional): 节点的颜色. Defaults to "blue".

        Returns:
            name(str): 节点的名字
        """        
        name = eqn.primitive.name
        if 'name' in eqn.params:
            name = name +'_'+ eqn.params['name']
        if 'mark_type' in eqn.params:
            name = name +'_'+ eqn.params['mark_type']
        name = name+'_'+str(eqn_index)
        lable = name
        graph.add_node(name,lable = lable,shape=shape,color=color)
        return name
    
    outvar_id_to_eqn_name = dict()
    eqn_name_to_eqn_index=dict()
    eqn_index_to_eqn_name=dict()
    eqn_index_to_floats = dict()
    
    digraph = nx.DiGraph()
    # 添加起点和汇点
    digraph.add_node('s')
    eqn_name_to_eqn_index['s'] = -1
    eqn_index_to_eqn_name[-1] = 's'
    digraph.add_node('e')
    eqn_name_to_eqn_index['e'] = -2
    eqn_index_to_eqn_name[-2] = 'e'
    # 设置源点和汇点对应的floats，默认为1
    eqn_index_to_floats[-1] = 0
    eqn_index_to_floats[-2] = 0
    
    # 更新每个公式的floats，暂时设置为1
    for eqn_index,eqn in enumerate(close_jaxpr.jaxpr.eqns):
        eqn_index_to_floats[eqn_index] = 1
        
    # 设置全局输入变量为源点的输出变量，保证公式起源于源点
    for invar in close_jaxpr.jaxpr.invars:
        if outvar_id_to_eqn_name.get(id(invar),None) is None:
            outvar_id_to_eqn_name[id(invar)] = 's'
    
    # 将每一个公式添加到图中
    # logic_link_insert_flag 用于判断是否已添加逻辑连接
    logic_link_insert_flag = False
    for eqn_index,eqn in enumerate(close_jaxpr.jaxpr.eqns):
        # 添加公式节点
        eqn_name = _addEqnNodeToGraph(eqn,eqn_index,digraph)
        eqn_name_to_eqn_index[eqn_name] = eqn_index
        eqn_index_to_eqn_name[eqn_index] = eqn_name
        
        # 添加指向该公式节点的边，边权设置为上一个公式的 floats
        for var in eqn.invars:
            if(isinstance(var,DropVar) or isinstance(var,Literal)):
                continue
            src_eqn_name = outvar_id_to_eqn_name.get(id(var),None)
            assert src_eqn_name is not None
            weight = eqn_index_to_floats[eqn_name_to_eqn_index[src_eqn_name]]
            digraph.add_edge(src_eqn_name,eqn_name,weight = weight)
        
        # 更新 outvar_id_to_eqn_name
        for var in eqn.outvars:
            assert outvar_id_to_eqn_name.get(id(var),None) is None
            outvar_id_to_eqn_name[id(var)] = eqn_name
        
        # # 添加前向和反向见的逻辑连接,不需要了
        # if eqn.primitive.name == 'pipeline_marker' and 'backward' in eqn.params['name'] and not logic_link_insert_flag:
        #     src_eqn_name = eqn_index_to_eqn_name.get(eqn_index - 1,None)
        #     assert src_eqn_name is not None
        #     digraph.add_edge(src_eqn_name,eqn_name,weight = 0)
        #     logic_link_insert_flag = True
            
            
    # 设置全局输出变量为汇点的输入变量，保证公式交于于汇点
    for outvar in close_jaxpr.jaxpr.outvars:
        # 添加全局输出变量到对应的输出公式到汇点的边
        src_eqn_name = outvar_id_to_eqn_name.get(id(outvar),None)
        assert src_eqn_name is not None
        weight = eqn_index_to_floats[eqn_name_to_eqn_index[src_eqn_name]]
        digraph.add_edge(src_eqn_name,'e',weight = weight)
    
    # 将图中出度为零的点连接到汇点，入度为零的点连接到源点
    for node in digraph.nodes():
        if node == 's' or node == 'e': continue
        node_in_degree = digraph.in_degree(node)
        node_out_degree = digraph.out_degree(node)
        if node_in_degree == 0:
            digraph.add_edge('s',node,weight = eqn_index_to_floats[-1])
        if node_out_degree == 0:
            node_weight = eqn_index_to_floats[eqn_name_to_eqn_index[node]]
            digraph.add_edge(node,'e',weight = node_weight)
    return digraph
    
def translateJaxprToNetworkxWithVar(close_jaxpr:ClosedJaxpr):
    def _addVarNodeToGraph(var,graph,shape="ellipse",color="blue"):
        nonlocal drop_num,literal_num
        if(isinstance(var,DropVar)):
            name="dropvar"+var.__repr__()+str(drop_num)
            drop_num+=1
        if(isinstance(var,Literal)):
            name="literalvar_"+var.__repr__()+"_"+str(literal_num)
            literal_num+=1
        else:
            name=var.__repr__()
            
        lable=name+" "+var.aval.__repr__()
        node_id_to_name[id(var)]=name
        node_id_to_label[id(var)]=lable
        graph.add_node(name,lable = lable,shape=shape,color=color)
        
    def _addEqnNodeToGraph(eqn : JaxprEqn,eqn_index,graph,shape="rectangle",color="blue"):
        name = eqn.primitive.name
        if 'name' in eqn.params:
            name = name +'_'+ eqn.params['name']
        if 'mark_type' in eqn.params:
            name = name +'_'+ eqn.params['mark_type']
        name = name+'_'+str(eqn_index)
        lable = name
        graph.add_node(name,lable = lable,shape=shape,color=color)
        return name
    
    def _addEqnToGraph(eqn : JaxprEqn,eqn_index,graph,shape="rectangle",color="blue"):
        for var in eqn.invars:
            if node_id_to_name.get(id(var),None) is None:
                _addVarNodeToGraph(var,graph)
        eqn_name = _addEqnNodeToGraph(eqn,eqn_index,graph)
        for var in eqn.outvars:
            if node_id_to_name.get(id(var),None) is None:
                _addVarNodeToGraph(var,graph)
        for var in eqn.invars:
            graph.add_edge(node_id_to_name.get(id(var)),eqn_name,weight = 1)
        for var in eqn.outvars:
            graph.add_edge(eqn_name,node_id_to_name.get(id(var)),weight = 1)
        return eqn_name
        
    def contractedInvarAndEqn(eqn : JaxprEqn,eqn_index,graph):
        eqn_name = eqn_index_to_name[eqn_index]
        eqn_attribute = graph.nodes[eqn_name]
        for var in eqn.outvars:
            node_name = node_id_to_name.get(id(var))
            if not graph.nodes.get(node_name,None) is None:
                nx.contracted_nodes(graph,eqn_name,node_id_to_name.get(id(var)),self_loops=False,copy= False)
        graph.add_node(eqn_name, **eqn_attribute)
        del graph.nodes[eqn_name]['contraction']

    node_id_to_name=dict()
    node_id_to_label=dict()
    eqn_index_to_name=dict()
    drop_num = 1
    literal_num = 1
    digraph = nx.DiGraph()
    digraph.add_node('0')
    digraph.add_node('1')
    # 将全局输入连接到源点
    for invar in close_jaxpr.jaxpr.invars:
        if node_id_to_name.get(invar,None) is None:
            _addVarNodeToGraph(invar,digraph)
            digraph.add_edge('0',node_id_to_name.get(id(invar)))
    for eqn_index,eqn in enumerate(close_jaxpr.jaxpr.eqns):
        eqn_name = _addEqnToGraph(eqn,eqn_index,digraph)
        eqn_index_to_name[eqn_index] = eqn_name
    # 将全局输出连接到汇点
    for outvar in close_jaxpr.jaxpr.outvars:
        if node_id_to_name.get(outvar,None) is None:
            _addVarNodeToGraph(outvar,digraph)
            digraph.add_edge(node_id_to_name.get(id(outvar)),'1')
        
    # 将图中出度为零的点连接到汇点，入度为零的点连接到源点
    for node in digraph.nodes():
        if node == '0' or node == '1': continue
        node_in_degree = digraph.in_degree(node)
        node_out_degree = digraph.out_degree(node)
        if node_in_degree == 0:
            digraph.add_edge('0',node)
        if node_out_degree == 0:
            digraph.add_edge(node,'1')
    return digraph

# 从 networkx 的 DiGraph 中获得相关信息

def getCriticalPathFromNetwprkx(graph:nx.DiGraph) -> List:
    """从Networkx图中获得关键路径，

    Args:
        graph (nx.DiGraph): eqn构成的AOE网络

    Returns:
        List: eqn对应的类型，0表示非关键节点，1表示关键节点
    """    
    # 获得关键路径
    return nx.algorithms.dag_longest_path(graph)

def getCriticalNodesBasedCritialPath(graph:nx.DiGraph,critical_path:List)-> List:
    # 获得汇点前一个节点的祖先节点
    node_ancestors = [child for parent, child in nx.bfs_edges(graph, critical_path[-2], reverse=True)]
    # 添加汇点，和之前的一个节点
    node_ancestors.extend(critical_path[-2:])
    return node_ancestors

def getEqnsTypeBasedCriticalNode(close_jaxpr:ClosedJaxpr,critical_node:List)-> List:
    """并根据关键节点，关键节点指关键路径上的节点和其前驱节点，将eqn分为两类，

    Args:
        close_jaxpr (ClosedJaxpr): eqn构成的AOE网络
        critical_path (List): 关键节点构成的列表

    Returns:
        List: eqn对应的类型，0表示非关键公式，1表示关键公式, pipeline_mark 公式视为非关键公式（之后划分无用，会重新生成）
    """    
    # 根据关键路径将eqn分成两类
    eqn_type_list = [0 for i in range(len(close_jaxpr.jaxpr.eqns))]
    for critical_node_name in critical_node[1:-1]:
        if critical_node_name == 's' or critical_node_name == 'e':continue
        critical_eqn_index = int(critical_node_name.split('_')[-1])
        eqn_type_list[critical_eqn_index] = 1
    
    return eqn_type_list

def getNonCriticalNodeFromNetwprkx(graph:nx.DiGraph,critical_node:List) -> List:
    """从Networkx图中获得非关键节点列表，

    Args:
        graph (nx.DiGraph): eqn构成的AOE网络
        critical_path (List): 关键节点构成的列表

    Returns:
        List: 非关键节点构成的列表
    """    
    # 非关键节点
    non_critial_node = list()
    for node in graph.nodes():
        if node not in critical_node:
            non_critial_node.append(node)
    return  non_critial_node

# 设置 networkx DiGraph 中节点的相关信息
 
def setCriticalNodeColor(graph:nx.DiGraph,critical_node:List):
    for critical_node_name in critical_node:
        graph.nodes[critical_node_name]['color'] = 'red'

# 根据 EqnsType 将 alpa 原始 layer 划分为两部分

def transJaxPipelineComputationsToMergedJaxpr(layers: Sequence[JaxPipelineComputation]) -> ClosedJaxpr:
    """将多个 layer 合成一个 ClosedJaxpr ，以便进行公式划分。
    修改自 merge_marked_jaxprs_with_named_call 函数

    Args:
        layers (Sequence[JaxPipelineComputation]): layer的列表

    Returns:
        ClosedJaxpr: 合并后的 ClosedJaxpr
    """    
    layer_jaxprs = [layer.closed_jaxpr() for layer in layers]
    new_eqns = []
    invars = OrderedSet()
    const_dir = {}
    outvars = OrderedSet()
    for i, jaxpr in enumerate(layer_jaxprs):
        const_dir.update(zip(jaxpr.jaxpr.constvars, jaxpr.consts))
        invars.update(jaxpr.jaxpr.invars)
        new_eqns.extend(jaxpr.eqns)
        outvars.update(jaxpr.jaxpr.outvars)
    shared_vars = invars.intersection(outvars)
    invars.difference_update(shared_vars)
    outvars.difference_update(shared_vars)
    jaxpr = Jaxpr(const_dir.keys(), invars, outvars, new_eqns)
    closed_jaxpr = ClosedJaxpr(jaxpr, const_dir.values())
    return closed_jaxpr

def sliceClosedJaxprBasedEqnsType(layer_closed_jaxpr: ClosedJaxpr, eqns_type: List, gensym_func) -> Sequence[ClosedJaxpr]:
    """根据 eqns_type 将 closed_jaxpr 划分为两部分。
    要求 closed_jaxpr 中仅有一对 pipeline_mark ，且分别在开始和结束；

    Args:
        layer_closed_jaxpr (ClosedJaxpr): 将被划分的 closed_jaxpr 
        eqns_type (List): closed_jaxpr 公式的分类，0表示非关键公式，1表示关键公式, pipeline_mark 公式视为非关键公式（之后划分无用，会重新生成）
        gensym_func (func): 用于生成新变量的函数，

    Returns:
        Sequence[ClosedJaxpr]: 划分后的 closed_jaxpr 列表，第一个表示非关键 jaxpr （可并行的），第二个表示关键 jaxpr
    """    
    # 获得划分后的公式列表，不包含 pipeline_mark，第0个列表对象表示非关键公式集，第1个列表对象表示关键公式集
    sliced_eqns = [[],[]]
    for i,eqn in enumerate(layer_closed_jaxpr.eqns):
        if 'pipeline_marker' in eqn.primitive.name:
            continue
        sliced_eqns[eqns_type[i]].append(eqn)
        
    # 构建 jaxpr ，参考 slices_to_jaxpr 函数
    global_consts = dict(zip(layer_closed_jaxpr.jaxpr.constvars, layer_closed_jaxpr.consts))
    
    # 获得 sliced_eqns 的输入、输出、
    n_eqns = len(sliced_eqns)
    sliced_eqn_invars = [OrderedSet() for _ in range(n_eqns)]
    sliced_eqn_outvars = [OrderedSet() for _ in range(n_eqns)]
    var_layer_dict = {}  # Dict[var -> layer_idx]
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Literal):
                    continue
                else:
                    sliced_eqn_invars[i].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    sliced_eqn_outvars[i].add(var)
                    var_layer_dict[var] = i
        shared_vars = sliced_eqn_invars[i].intersection(sliced_eqn_outvars[i])
        sliced_eqn_invars[i].difference_update(shared_vars)
        sliced_eqn_outvars[i].difference_update(shared_vars)
    # 修补新产生的跨 layer 变量，导致的输出 sliced_eqn_outvars 缺少问题
    for eqn_invars in sliced_eqn_invars:
        for invar in eqn_invars:
            if invar in var_layer_dict:
                sliced_eqn_outvars[var_layer_dict[invar]].add(invar)
    
    # 获得原始 pipeline_mark 的输入和输出相关信息，
    pipeline_mark_start_outvar_to_invar = {}
    pipeline_mark_end_invar_to_outvar = {}
    assert 'pipeline_marker' in layer_closed_jaxpr.eqns[0].primitive.name
    assert 'pipeline_marker' in layer_closed_jaxpr.eqns[-1].primitive.name
    eqn = layer_closed_jaxpr.eqns[0]
    for var_index,invar in enumerate(eqn.invars):
        outvar = eqn.outvars[var_index]
        pipeline_mark_start_outvar_to_invar[outvar] = invar
    
    eqn = layer_closed_jaxpr.eqns[-1]
    for var_index,invar in enumerate(eqn.invars):
        outvar = eqn.outvars[var_index]
        pipeline_mark_end_invar_to_outvar[invar] = outvar   

    
    # 获得新 pipeline_mark 的输入和输出相关信息，第0个对象表示非关键公式集，第1个对象表示关键公式集，修改自 add_pipeline_marks_for_sliced_eqns
    sliced_closed_jaxpr = []
    names = [layer_closed_jaxpr.eqns[0].params['name']+'_parallel',layer_closed_jaxpr.eqns[0].params['name']]
    for i, eqns in enumerate(sliced_eqns):
        new_eqns = []
        # 存储新跨 layer 变量导致的更新变量关系
        update_eqn_vars = {}
        # 构建 pipeline_mark start 公式
        pipeline_start_invars = []
        pipeline_start_outvars = []
        for invar in sliced_eqn_invars[i]:
            new_var = gensym_func(invar.aval)
            # 如果原始 pipeline_mark 中存在与 invar 对应的输入变量，重用该关系，invar 作为新 pipeline_mark 的输出
            if invar in pipeline_mark_start_outvar_to_invar:
                pipeline_start_invars.append(pipeline_mark_start_outvar_to_invar[invar])
            # 如果 invar 同时是原始 pipeline_mark_end 的输入，使用输出作为新 pipeline_mark 的输出
            elif invar in pipeline_mark_end_invar_to_outvar:
                pipeline_start_invars.append(pipeline_mark_end_invar_to_outvar[invar])
            # 对新产生的跨 layer 的变量，生成一个新变量作为新 pipeline_mark start 的输出
            else:
                pipeline_start_invars.append(invar)
            pipeline_start_outvars.append(new_var)
            update_eqn_vars[invar] = new_var
        start_marker = mark_pipeline_jaxpreqn(pipeline_start_invars, pipeline_start_outvars, names[i], "start")
        
        # 构建 pipeline_mark end 公式
        pipeline_end_invars = []
        pipeline_end_outvars = []
        for outvar in sliced_eqn_outvars[i]:
            # 如果原始 pipeline_mark 中存在与 outvar 对应的输出变量，重用该关系，outvar 作为新 pipeline_mark 的输入
            if outvar in pipeline_mark_end_invar_to_outvar:
                pipeline_end_invars.append(outvar)
                pipeline_end_outvars.append(pipeline_mark_end_invar_to_outvar[outvar])
            # 对新产生的跨 layer 的变量，生成一个新变量作为新 pipeline_mark start 的输入
            else:
                new_var = gensym_func(outvar.aval)
                pipeline_end_invars.append(new_var)
                pipeline_end_outvars.append(outvar)
                update_eqn_vars[outvar] = new_var
        end_marker = mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars, names[i], "end")
        
        new_eqns.append(start_marker)
        # 更新每一个 eqn ，因为 pipeline_mark 可能改变了公式中的输入输出变量
        for eqn in eqns:
            eqn_new_invars = [
                get_var_mapping(update_eqn_vars, var)
                for var in eqn.invars
            ]
            eqn_new_outvars = [
                get_var_mapping(update_eqn_vars, var)
                for var in eqn.outvars
            ]
            new_eqns.append(clone_jaxpr_eqn(eqn, eqn_new_invars,eqn_new_outvars))
        
        new_eqns.append(end_marker)
        
        jaxpr_consts = {}
        jaxpr_invars = []
        jaxpr_outvars = pipeline_end_outvars
        for invar in pipeline_start_invars:
            if invar in global_consts:
                jaxpr_consts[invar] = global_consts[invar]
            else:
                jaxpr_invars.append(invar)
        jaxpr = Jaxpr(jaxpr_consts.keys(), jaxpr_invars, jaxpr_outvars, new_eqns)
        closed_jaxpr = ClosedJaxpr(jaxpr, jaxpr_consts.values())
        sliced_closed_jaxpr.append(closed_jaxpr)
    return sliced_closed_jaxpr
    
def sliceJaxPipelineComputationBasedEqnsType(layer: JaxPipelineComputation, eqns_type: List, gensym_func) -> Sequence[JaxPipelineComputation]:
    """根据 eqns_type 将 layer 划分为两部分。

    Args:
        layer (JaxPipelineComputation): 将被划分的 JaxPipelineComputation 
        eqns_type (List): JaxPipelineComputation 公式的分类，0表示非关键公式，1表示关键公式, pipeline_mark 公式视为非关键公式（之后划分无用，会重新生成）
        gensym_func (func): 用于生成新变量的函数，

    Returns:
        Sequence[JaxPipelineComputation]: 划分后的 JaxPipelineComputation 列表，第一个表示非关键 JaxPipelineComputation （可并行的），第二个表示关键 JaxPipelineComputation
    """    
    layer_closed_jaxpr = layer.closed_jaxpr()
    assert len(eqns_type) == len(layer_closed_jaxpr.eqns)
    sliced_closed_jaxprs = sliceClosedJaxprBasedEqnsType(layer_closed_jaxpr,eqns_type,gensym_func)
    sliced_jax_pipeline_computations = []
    for closed_jaxpr in sliced_closed_jaxprs:
        sliced_jax_pipeline_computations.append(JaxPipelineComputation.from_closed_jaxpr(closed_jaxpr.eqns[0].params['name'],closed_jaxpr))
    
    return sliced_jax_pipeline_computations
    
from jax.lib import xla_bridge as xb
from jax.lib import xla_client as xc


# 计算每个公式的floats
def eqn_flops(eqn: JaxprEqn) -> float:
    """Get the FLOP of a jaxpr equation."""
    if "jaxpr" in eqn.params:
        return sum(eqn_flops(x) for x in eqn.params["jaxpr"].eqns)

    # if eqn.primitive not in non_trivial_primitive:
    #     return 0

    new_inv = [inv for inv in eqn.invars if isinstance(inv, Var)]
    jaxpr = Jaxpr([], new_inv, eqn.outvars, [eqn])
    closed_jaxpr = ClosedJaxpr(jaxpr, [])
    hlo_module = jaxpr_to_hlo("tmp", closed_jaxpr, [
        False,
    ] * len(jaxpr.invars)).get_module()

    backend = xb.get_backend("cpu")
    properties = xc._xla.hlo_module_cost_analysis(  # pylint: disable=protected-access
        backend, hlo_module)
    return properties["flops"] if "flops" in properties else 1.0


def test1():
    """测试从jaxpr获得公式对应类型的相关函数
    """    
    from .jaxpr_display import (DeserializationComputeGradJaxpr,
                                SliceComputeGradJaxprToForwardAndBackward,
                                SliceJaxprToComputeAndApplyGrad)
    file_path="cyg_test/display/acc_grad_jaxpr.pkl"
    acc_grad_jaxpr=DeserializationComputeGradJaxpr(file_path)
    acc_grad_jaxpr, compute_acc_grad_jaxpr, _, _= SliceJaxprToComputeAndApplyGrad(acc_grad_jaxpr,1)
    forward_closed_jaxpr,backward_closed_jaxpr=SliceComputeGradJaxprToForwardAndBackward(compute_acc_grad_jaxpr)
    
    forward_digraph_with_acc_grad = translateJaxprToNetworkxWithVar(forward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(forward_digraph_with_acc_grad)
    critical_node = getCriticalNodesBasedCritialPath(forward_digraph_with_acc_grad,critical_path)
    setCriticalNodeColor(forward_digraph_with_acc_grad,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(forward_digraph_with_acc_grad,critical_node)
    # eqn_type_list = getEqnsTypeBasedCriticalNode(forward_closed_jaxpr,critical_node)
    forward_digraph_with_acc_grad = nx.nx_pydot.to_pydot(forward_digraph_with_acc_grad)
    forward_digraph_with_acc_grad.write("cyg_test/display/forward_digraph_with_acc_grad.png",format="png")
    print("前向传播：")
    print("\t带有var")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    # print("\t\t公式分类：", eqn_type_list)
    
    
    forward_digraph_with_acc_grad_without_var = translateJaxprToNetworkxWithoutVar(forward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(forward_digraph_with_acc_grad_without_var)
    critical_node = getCriticalNodesBasedCritialPath(forward_digraph_with_acc_grad_without_var,critical_path)
    setCriticalNodeColor(forward_digraph_with_acc_grad_without_var,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(forward_digraph_with_acc_grad_without_var,critical_node)
    eqn_type_list = getEqnsTypeBasedCriticalNode(forward_closed_jaxpr,critical_node)
    forward_digraph_with_acc_grad_without_var = nx.nx_pydot.to_pydot(forward_digraph_with_acc_grad_without_var)
    forward_digraph_with_acc_grad_without_var.write("cyg_test/display/forward_digraph_with_acc_grad_without_var.png",format="png")
    print("前向传播：")
    print("\t不带var")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    print("\t\t公式分类：", eqn_type_list)
    
    
    forward_digraph_with_acc_grad_without_var_and_pipeline_mark = translateJaxprToNetworkxWithoutVarAndPipelineMarker(forward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(forward_digraph_with_acc_grad_without_var_and_pipeline_mark)
    critical_node = getCriticalNodesBasedCritialPath(forward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_path)
    setCriticalNodeColor(forward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(forward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_node)
    eqn_type_list = getEqnsTypeBasedCriticalNode(forward_closed_jaxpr,critical_node)
    forward_digraph_with_acc_grad_without_var_and_pipeline_mark = nx.nx_pydot.to_pydot(forward_digraph_with_acc_grad_without_var_and_pipeline_mark)
    forward_digraph_with_acc_grad_without_var_and_pipeline_mark.write("cyg_test/display/forward_digraph_with_acc_grad_without_var_and_pipeline_mark.png",format="png")
    print("前向传播：")
    print("\t不带var和pipeline mark")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    print("\t\t公式分类：", eqn_type_list)
    
    backward_digraph_with_acc_grad = translateJaxprToNetworkxWithVar(backward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(backward_digraph_with_acc_grad)
    critical_node = getCriticalNodesBasedCritialPath(backward_digraph_with_acc_grad,critical_path)
    setCriticalNodeColor(backward_digraph_with_acc_grad,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(backward_digraph_with_acc_grad,critical_node)
    # eqn_type_list = getEqnsTypeBasedCriticalNode(backward_closed_jaxpr,critical_node)
    backward_digraph_with_acc_grad = nx.nx_pydot.to_pydot(backward_digraph_with_acc_grad)
    backward_digraph_with_acc_grad.write("cyg_test/display/backward_digraph_with_acc_grad.png",format="png")
    print("反向传播：")
    print("\t带有var")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    # print("\t\t公式分类：", eqn_type_list)
    
    
    backward_digraph_with_acc_grad_without_var = translateJaxprToNetworkxWithoutVar(backward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(backward_digraph_with_acc_grad_without_var)
    critical_node = getCriticalNodesBasedCritialPath(backward_digraph_with_acc_grad_without_var,critical_path)
    setCriticalNodeColor(backward_digraph_with_acc_grad_without_var,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(backward_digraph_with_acc_grad_without_var,critical_node)
    eqn_type_list = getEqnsTypeBasedCriticalNode(backward_closed_jaxpr,critical_node)
    backward_digraph_with_acc_grad_without_var = nx.nx_pydot.to_pydot(backward_digraph_with_acc_grad_without_var)
    backward_digraph_with_acc_grad_without_var.write("cyg_test/display/backward_digraph_with_acc_grad_without_var.png",format="png")
    print("反向传播：")
    print("\t不带var")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    print("\t\t公式分类：", eqn_type_list)
    
    
    backward_digraph_with_acc_grad_without_var_and_pipeline_mark = translateJaxprToNetworkxWithoutVarAndPipelineMarker(backward_closed_jaxpr)
    critical_path = getCriticalPathFromNetwprkx(backward_digraph_with_acc_grad_without_var_and_pipeline_mark)
    critical_node = getCriticalNodesBasedCritialPath(backward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_path)
    setCriticalNodeColor(backward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_node)
    non_critial_Node = getNonCriticalNodeFromNetwprkx(backward_digraph_with_acc_grad_without_var_and_pipeline_mark,critical_node)
    eqn_type_list = getEqnsTypeBasedCriticalNode(backward_closed_jaxpr,critical_node)
    backward_digraph_with_acc_grad_without_var_and_pipeline_mark = nx.nx_pydot.to_pydot(backward_digraph_with_acc_grad_without_var_and_pipeline_mark)
    backward_digraph_with_acc_grad_without_var_and_pipeline_mark.write("cyg_test/display/backward_digraph_with_acc_grad_without_var_and_pipeline_mark.png",format="png")
    print("反向传播：")
    print("\t不带var和pipeline mark")
    print("\t\t关键节点：", critical_node)
    print("\t\t非关键节点：", non_critial_Node)
    print("\t\t公式分类：", eqn_type_list)

   
if __name__ == "__main__":
    # file_path="cyg_test/display/jaxpr_before_split.pkl"
    # all_jaxpr=DeserializationComputeGradJaxpr(file_path)
    # all_jaxpr, compute_grad_jaxpr, apply_grad_jaxpr, split_eqn= SliceJaxprToComputeAndApplyGrad(all_jaxpr,1)
    # forward_closed_jaxpr,backward_closed_jaxpr=SliceComputeGradJaxprToForwardAndBackward(compute_grad_jaxpr,2)
    # digraph_with_var = translateJaxprToNetworkxWithVar(compute_grad_jaxpr)
    # digraph_without_var = translateJaxprToNetworkxWithoutVar(compute_grad_jaxpr)
    # # print(list(digraph_before_contracted.nodes(data=True)))
    # # print(list(digraph.nodes(data=True)))
    # digraph_with_var = nx.nx_pydot.to_pydot(digraph_with_var)
    # digraph_with_var.write("cyg_test/display/test1.png",format="png")
    # digraph_without_var = nx.nx_pydot.to_pydot(digraph_without_var)
    # digraph_without_var.write("cyg_test/display/test2.png",format="png")

    test1()
    
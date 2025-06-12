from task4feedback.graphs import *
from task4feedback.graphs.mesh import build_geometry, generate_quad_mesh, generate_tri_mesh
from task4feedback.graphs.mesh.partition import *
from typing import Callable 
import hydra
from omegaconf import DictConfig, OmegaConf

@dataclass
class GraphBuilder:
    config: GraphConfig
    function: Callable[[GraphConfig, DictConfig], Graph]
    
    
def graph_to_variant(graph_class):
    """
    Returns the variant class for a given graph class.
    """
    if graph_class == "JacobiGraph":
        return JacobiVariant
    elif graph_class == "DynamicJacobiGraph":
        return DynamicJacobiVariant
    else:
        raise NotImplementedError(f"Graph class {graph_class} is not implemented.")

def make_graph_function(graph_cfg: GraphConfig, cfg: DictConfig) -> Callable[[GraphConfig, DictConfig], Graph]:
    
    
    def make_graph():
        mesh_generator = globals()[cfg.mesh.generator]
        
        mesh = mesh_generator(L=graph_cfg.L, n=graph_cfg.n)
        
        if cfg.init.partitioner == "metis":
            partitioner = metis_geometry_partition
        else:
            raise NotImplementedError(f"Partitioner {cfg.init.partitioner} is not implemented.")
        
        geom = build_geometry(mesh)
        graph_class = globals()[cfg.graph_class]
        graph = graph_class(geom, graph_cfg)
        partition = partitioner(geom, nparts=cfg.init.nparts)
        
        graph.apply_variant(graph_to_variant(cfg.graph_class))
        
        if cfg.init.gpu_only:
            partition = [x + 1 for x in partition]  # offset by 1 to ignore cpu
            location_list = [i + 1 for i in range(cfg.init.nparts)]
        else:
            location_list = [i for i in range(cfg.init.nparts + 1)]  # include cpu as 0
            
        graph.set_cell_locations(partition)
            
        if cfg.init.randomize:
            graph.randomize_locations(graph_cfg.randomness, location_list=location_list)

        return graph
    
    graph_function = make_graph
    return graph_function


    return graph_function


def make_graph_builder(cfg: DictConfig) -> GraphBuilder:
    graph_info = cfg.graph
    config_class = globals()[graph_info["config_class"]]
    graph_config_info = graph_info.config
    graph_cfg = config_class(**OmegaConf.to_container(graph_config_info, resolve=True))
    
    graph_function = make_graph_function(graph_cfg, graph_info)
    return GraphBuilder(config=graph_cfg, function=graph_function)
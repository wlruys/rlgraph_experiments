# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.ppo import *
from task4feedback.interface import TaskTuple
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env

def configure_training(cfg: DictConfig):
    
    #A = PPOConfig(**OmegaConf.to_container(cfg.algorithm.config, resolve=True))
    
    #make_graph_builder(cfg)
    #env = make_env(graph_builder=make_graph_builder(cfg), cfg=cfg)
    
    print(cfg)
    
    # run = wandb.init(
    #     project=cfg.wandb.project,
    #     name=cfg.wandb.name,
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     reinit=True,
    # )

# # your existing train(...) refactored to accept cfg: DictConfig
# def train(cfg: DictConfig):
#     # Hydra has already loaded everything under cfg.*
#     # instantiate graph_config, reward_config, etc. directly from cfg
#     graph_info = cfg.graph
#     graph_class = globals()[graph_info.graph_class]
#     # build config dataclass for JacobiConfig or DynamicJacobiConfig dynamically
#     if graph_info.graph_class == "JacobiGraph":
#         graph_cfg = JacobiConfig(**OmegaConf.to_container(graph_info, resolve=True))
#         graph_fn = build_jacobi_graph
#     else:
#         graph_cfg = DynamicJacobiConfig(**OmegaConf.to_container(graph_info, resolve=True))
#         graph_fn = build_dynamic_jacobi_graph

#     # similarly unpack other cfg groups…
#     env = make_env(
#         graph_fn,
#         graph_cfg,
#         system_config=OmegaConf.to_container(cfg.system, resolve=True),
#         feature_config=OmegaConf.to_container(cfg.feature, resolve=True),
#         runtime_env_t=globals()[cfg.reward.runtime_env],
#         observer_factory_t=globals()[cfg.feature.observer_factory],
#         change_priority=cfg.env.change_priority,
#         change_locations=cfg.env.change_locations,
#         seed=cfg.env.seed,
#     )

#     # initialize wandb using cfg.wandb
#     wb = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)

#     # assemble PPOConfig from cfg.mconfig and cfg.training…
#     ppo_cfg = PPOConfig(**OmegaConf.to_container(cfg.mconfig, resolve=True),
#                         states_per_collection=len(env.simulator.input.graph)
#                                              * cfg.mconfig.graphs_per_collection)

#     # rest of your setup and run:
#     model = globals()[cfg.model.model_architecture](
#         feature_config=FeatureDimConfig.from_observer(env.observer),
#         layer_config=LayerConfig(**OmegaConf.to_container(cfg.layer, resolve=True)),
#         n_devices=cfg.system.n_devices,
#     )

#     run_ppo_torchrl(model, lambda: make_env(...), ppo_cfg)
#     # …

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    configure_training(cfg)
    print(cfg)

if __name__ == "__main__":
    main()

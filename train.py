# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.algorithms import *
from task4feedback.interface import TaskTuple
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import *
from functools import partial 

from task4feedback.ml.algorithms.ppo import run_ppo

from rich import print as rprint

def configure_training(cfg: DictConfig):
    
    
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg)
    
    rprint(graph_builder)
    
    def env_builder():
        return make_env(
            graph_builder=graph_builder,
            cfg=cfg,
        )

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    print(f"Feature config: {feature_config}")
    
    a = create_td_actor_critic_models(cfg, feature_config)

    alg_config = instantiate(cfg.algorithm)
    
    if cfg.wandb.enabled:
        logging_config = instantiate(cfg.logging)
    else:
        logging_config = None 
    
    run_ppo(
        actor_critic_module=a,
        env_constructors=[env_builder],
        logging_config=logging_config,
        ppo_config=alg_config,
    )
    
    
    
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.name,
        )
        
    configure_training(cfg)
    
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()

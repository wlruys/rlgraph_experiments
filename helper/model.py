from task4feedback.ml.models import *
from task4feedback.ml import ActorCriticModule 
from typing import Callable 
import hydra
import torch.nn as nn 
import tensordict.nn as td_nn
from tensordict import TensorDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from torchrl.envs import ExplorationType 
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule



def create_actor_critic_models(cfg: DictConfig, feature_cfg: FeatureDimConfig) -> nn.Module:
    layers = cfg.network.layers
    
    state_layer = layers.state
    output_layer = layers.output 
    
    policy_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
    )
    
    policy_output_module = instantiate(
        output_layer,
        input_dim=policy_state_module.output_dim,
        output_dim=cfg.system.config.n_devices
    )
    
    policy_module = nn.Sequential(
        policy_state_module,
        policy_output_module
    )

    critic_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        add_progress=cfg.network.progress_in_critic,
    )
    
    
    value_output_module = instantiate(
        output_layer,
        input_dim=critic_state_module.output_dim,
        output_dim=1
    )
    
    
    value_module = nn.Sequential(
        critic_state_module,
        value_output_module
    )
    
        
    return ActorCriticModule(policy_module, value_module)
    

def create_td_actor_critic_models(cfg: DictConfig, feature_cfg: FeatureDimConfig) -> nn.Module:
    
    layers = cfg.network.layers
    
    state_layer = layers.state
    output_layer = layers.output 
    
    policy_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
    )
    
    policy_output_module = instantiate(
        output_layer,
        input_dim=policy_state_module.output_dim,
        output_dim=cfg.system.config.n_devices
    )
    
    _td_policy_state = td_nn.TensorDictModule(
        policy_state_module,
        in_keys=["observation"],
        out_keys=["embed"],
    )
    
    _td_policy_output = td_nn.TensorDictModule(
        policy_output_module,
        in_keys=["embed"],
        out_keys=["logits"],
    )
    
    policy_module = td_nn.TensorDictSequential(
        _td_policy_state,
        _td_policy_output,
        inplace=True
    )
    
    probabilistic_policy = ProbabilisticActor(
        module=policy_module,
        in_keys=["observation"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        cache_dist=False,
    )
    
    
    critic_state_module = instantiate(
        state_layer,
        feature_config=feature_cfg,
        add_progress=cfg.network.progress_in_critic,
    )
    
    critic_output_module = instantiate(
        output_layer,
        input_dim=critic_state_module.output_dim,
        output_dim=1
    )
    
    _td_critic_state = td_nn.TensorDictModule(
        critic_state_module,
        in_keys=["observation"],
        out_keys=["embed"],
    )
    
    _td_critic_output = td_nn.TensorDictModule(
        critic_output_module,
        in_keys=["embed"],
        out_keys=["state_value"],
    )
    
    critic_module = td_nn.TensorDictSequential(
        _td_critic_state,
        _td_critic_output,
        inplace=True
    )
    
    value_operator = ValueOperator(
        module=critic_module,
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    

    return ActorCriticModule(probabilistic_policy, value_operator)

    
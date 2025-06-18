from task4feedback.graphs import *
from task4feedback.graphs.mesh import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.ml.env import *

from typing import Callable
from .graph import GraphBuilder
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback import fastsim2 as fastsim

from torchrl.envs import (
    TransformedEnv,
    Compose,
    InitTracker,
    StepCounter,
    TrajCounter,
    ObservationNorm,
    RewardScaling,
    ExplorationType,
    TensorDictPrimer,
)
from torchrl.modules import LSTMModule
from typing import Optional
from functools import partial


def create_system(cfg: DictConfig):
    system_info = cfg.system
    generator = globals()[system_info.generator]
    system = generator(**OmegaConf.to_container(system_info.config, resolve=True))
    return system


def create_conditions(cfg: DictConfig):
    runtime_transition_conditions = cfg.runtime["class"]
    transition_conditions = getattr(fastsim, runtime_transition_conditions)(
        **OmegaConf.to_container(cfg.runtime["config"], resolve=True)
    )
    return transition_conditions


def create_runtime_reward(cfg: DictConfig):
    runtime_reward_class = cfg.reward["class"]
    # check if config exists
    if "config" in cfg.reward:
        runtime_reward_config = OmegaConf.to_container(
            cfg.reward["config"], resolve=True
        )
    else:
        runtime_reward_config = {}

    runtime_env_t = globals()[runtime_reward_class]

    runtime_env_t = partial(runtime_env_t, **runtime_reward_config)

    return runtime_env_t


def create_observer_factory(cfg: DictConfig):
    observer_factory_class = cfg.feature["class"]
    if "config" in cfg.feature:
        observer_factory_config = OmegaConf.to_container(
            cfg.feature["config"], resolve=True
        )
    else:
        observer_factory_config = {}

    observer_factory_t = globals()[observer_factory_class]
    return observer_factory_t


def make_env(
    graph_builder: GraphBuilder, cfg: DictConfig, lstm: Optional[LSTMModule] = None
):
    gmsh.initialize()

    s = create_system(cfg)
    graph = graph_builder.function()

    d = graph.get_blocks()
    m = graph

    m.finalize_tasks()

    graph_spec = create_graph_spec(
        **OmegaConf.to_container(cfg.feature.limits, resolve=True)
    )
    transition_conditions = create_conditions(cfg)

    runtime_env_t = create_runtime_reward(cfg)
    observer_factory_t = create_observer_factory(cfg)

    print("Running make_env", flush=True)
    input = SimulatorInput(m, d, s, transition_conditions=transition_conditions)

    env = runtime_env_t(
        SimulatorFactory(input, graph_spec, observer_factory_t),
        device="cpu",
        change_priority=cfg.graph.env.change_priority,
        change_locations=cfg.graph.env.change_locations,
        seed=cfg.graph.env.seed,
    )
    env = TransformedEnv(env, StepCounter())
    env.append_transform(TrajCounter())
    env.append_transform(InitTracker())
    env.append_transform(ObservationNorm(in_keys=[("observation", "nodes", "tasks", "attr")]))

    if lstm is not None:
        print("Adding LSTM module to environment", flush=True)
        env.append_transform(lstm.make_tensordict_primer())
        
    if isinstance(env.transform, Compose):
        for transform in env.transform:
            if isinstance(transform, ObservationNorm) and not transform.initialized:
                transform.init_stats(num_iter=500, key=("observation", "nodes", "tasks", "attr"))
        
    return env

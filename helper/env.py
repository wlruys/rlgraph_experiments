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

from torchrl.envs.transforms import NoopResetEnv


class OneTimeNoopResetEnv(NoopResetEnv):
    def __init__(self, noops: int, random: bool = True):
        super().__init__(noops=noops, random=random)
        self._used = False

    def reset(self, tensordict):
        if not self._used:
            self._used = True
            return super().reset(tensordict)
        return self.parent.reset(tensordict)


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


@dataclass
class NormalizationDetails:
    task_norm: dict


def make_env(
    graph_builder: GraphBuilder,
    cfg: DictConfig,
    lstm: Optional[LSTMModule] = None,
    normalization: Optional[NormalizationDetails] = None,
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
    # env.append_transform(OneTimeNoopResetEnv(len(m), random=True))

    if lstm is not None:
        print("Adding LSTM module to environment", flush=True)
        env.append_transform(lstm.make_tensordict_primer())

    if normalization is None:
        task_norm_transform = ObservationNorm(
            in_keys=[("observation", "nodes", "tasks", "attr")],
            eps=1e-4,
            standard_normal=True,
        )
        env.append_transform(task_norm_transform)
        if isinstance(env.transform, Compose):
            for transform in env.transform:
                if isinstance(transform, ObservationNorm) and not transform.initialized:
                    transform.init_stats(
                        num_iter=1000, key=("observation", "nodes", "tasks", "attr")
                    )
        new_norm = NormalizationDetails(task_norm=task_norm_transform.state_dict())
    else:
        task_norm_transform = ObservationNorm(
            in_keys=[("observation", "nodes", "tasks", "attr")],
            eps=1e-4,
            standard_normal=True,
            loc=normalization.task_norm["loc"],
            scale=normalization.task_norm["scale"],
        )
        task_norm_transform.load_state_dict(normalization.task_norm)

        env.append_transform(task_norm_transform)
        new_norm = None

    if new_norm is not None:
        return env, new_norm

    else:
        return env

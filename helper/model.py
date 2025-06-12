from task4feedback.ml.models import *
from typing import Callable 
import hydra
from omegaconf import DictConfig, OmegaConf


def create_model(cfg: DictConfig) -> Callable:
    
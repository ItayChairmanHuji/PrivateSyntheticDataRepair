from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


class WANDBLogger:
    def __init__(self, project_name: str, entity: str, run_name: str, metric: str, configuration=None):
        config_dict = self._get_config(configuration)

        self.run = self._create_run(project_name, entity, run_name, metric, config_dict)
        self.run.define_metric(metric)
        self.run.define_metric("*", step_metric=metric)

    @staticmethod
    def _get_config(configuration):
        return OmegaConf.to_container(configuration, resolve=True) \
            if isinstance(configuration, DictConfig) else dict(configuration)

    @staticmethod
    def _create_run(project_name: str, entity: str, run_name: str, metric: str, config_dict: dict):
        return wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            group=metric,
            config=config_dict,
            reinit="create_new",
        )

    def log(self, metric: str, value: Any):
        self.run.log({metric: value})

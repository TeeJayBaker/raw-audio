from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="experiment/fm_baseline")
def main(cfg: DictConfig) -> None:
    Path(cfg.train.run_dir).mkdir(parents=True, exist_ok=True)
    get_method(str(cfg.trainer._target_))(cfg)


if __name__ == "__main__":
    main()

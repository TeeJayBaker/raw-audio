from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@hydra.main(version_base=None, config_path="configs", config_name="experiment/fm_baseline")
def main(cfg: DictConfig) -> None:
    Path(cfg.train.run_dir).mkdir(parents=True, exist_ok=True)
    get_class(str(cfg.trainer._target_))(cfg).run()


if __name__ == "__main__":
    main()

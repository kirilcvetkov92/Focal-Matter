from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from utils.config import ProjectConfig


@dataclass
class ExperimentPaths:
    run_dir: Path
    log_file: Path
    config_file: Path
    checkpoint_dir: Path
    output_dir: Path
    code_dir: Path


CODE_SUFFIXES: Tuple[str, ...] = (
    ".py",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
)
EXCLUDED_DIRS: Set[str] = {
    ".git",
    ".idea",
    "__pycache__",
    "alpha",
    "checkpoints",
    "experiments",
}


def create_experiment(config: ProjectConfig, phase: str, project_root: Optional[Path] = None) -> ExperimentPaths:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    experiment_root = Path(config.experiment.root)
    if not experiment_root.is_absolute():
        experiment_root = project_root / experiment_root

    timestamp = datetime.now().strftime(config.experiment.timestamp_format)
    run_name = "{}-{}-{}".format(config.experiment.name, timestamp, phase)
    run_dir = experiment_root / run_name

    if run_dir.exists():
        suffix = 1
        while (experiment_root / "{}-{}".format(run_name, suffix)).exists():
            suffix += 1
        run_dir = experiment_root / "{}-{}".format(run_name, suffix)

    paths = ExperimentPaths(
        run_dir=run_dir,
        log_file=run_dir / "logs" / "{}.log".format(phase),
        config_file=run_dir / "config.json",
        checkpoint_dir=run_dir / "checkpoints",
        output_dir=run_dir / "alpha",
        code_dir=run_dir / "code",
    )

    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.code_dir.mkdir(parents=True, exist_ok=True)

    if config.experiment.save_config_snapshot:
        save_config(config, paths.config_file)

    snapshot_source_code(project_root, paths.code_dir)
    return paths


def save_config(config: ProjectConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)


def setup_logger(config: ProjectConfig, paths: ExperimentPaths, phase: str) -> logging.Logger:
    logger = logging.getLogger("FocalMatter.{}.{}".format(paths.run_dir.name, phase))
    logger.setLevel(getattr(logging, config.experiment.log_level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.experiment.log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if config.experiment.log_to_file:
        file_handler = logging.FileHandler(paths.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Experiment directory: %s", paths.run_dir)
    logger.info("Code snapshot: %s", paths.code_dir)
    return logger


def snapshot_source_code(
    project_root: Path,
    destination: Path,
    suffixes: Iterable[str] = CODE_SUFFIXES,
    excluded_dirs: Iterable[str] = EXCLUDED_DIRS,
) -> None:
    project_root = project_root.resolve()
    destination = destination.resolve()
    suffix_set = {suffix.lower() for suffix in suffixes}
    excluded = set(excluded_dirs)

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(destination)
            continue
        except ValueError:
            pass
        if path.suffix.lower() not in suffix_set:
            continue

        relative_path = path.relative_to(project_root)
        if any(part in excluded for part in relative_path.parts):
            continue

        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

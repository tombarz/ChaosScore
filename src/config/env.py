from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_ENV_FILENAME = ".env"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_env_path() -> Path:
    return project_root() / DEFAULT_ENV_FILENAME


def read_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file."""
    path = Path(env_path) if env_path is not None else default_env_path()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_dotenv(env_path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Load values from .env into os.environ."""
    values = read_env_file(env_path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    env_path: Path
    scfoundation_repo: Path
    scfoundation_checkpoint: Path
    scfoundation_gene_panel: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    output_dir: Path
    test_tmp_dir: Path


def _resolve_path(raw_value: str, *, base_dir: Path) -> Path:
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _get_env_value(name: str, default: str, *, env_values: dict[str, str]) -> str:
    return os.environ.get(name, env_values.get(name, default))


def load_project_paths(env_path: str | Path | None = None, *, override: bool = False) -> ProjectPaths:
    """Load project paths from .env and environment variables."""
    env_file = Path(env_path) if env_path is not None else default_env_path()
    env_values = load_dotenv(env_file, override=override)

    configured_root = _get_env_value("PROJECT_ROOT", ".", env_values=env_values)
    root = _resolve_path(configured_root, base_dir=env_file.parent if env_file.parent.exists() else project_root())

    return ProjectPaths(
        project_root=root,
        env_path=env_file.resolve(),
        scfoundation_repo=_resolve_path(
            _get_env_value("SCFOUNDATION_REPO", "external/scFoundation", env_values=env_values),
            base_dir=root,
        ),
        scfoundation_checkpoint=_resolve_path(
            _get_env_value(
                "SCFOUNDATION_CKPT_PATH",
                "external/scFoundation/model/models.ckpt",
                env_values=env_values,
            ),
            base_dir=root,
        ),
        scfoundation_gene_panel=_resolve_path(
            _get_env_value(
                "SCFOUNDATION_GENE_PANEL_PATH",
                "external/scFoundation/model/OS_scRNA_gene_index.19264.tsv",
                env_values=env_values,
            ),
            base_dir=root,
        ),
        data_dir=_resolve_path(_get_env_value("CHAOSSCORE_DATA_DIR", "data", env_values=env_values), base_dir=root),
        raw_data_dir=_resolve_path(
            _get_env_value("CHAOSSCORE_RAW_DATA_DIR", "data/raw", env_values=env_values),
            base_dir=root,
        ),
        processed_data_dir=_resolve_path(
            _get_env_value("CHAOSSCORE_PROCESSED_DIR", "data/processed", env_values=env_values),
            base_dir=root,
        ),
        output_dir=_resolve_path(
            _get_env_value("CHAOSSCORE_OUTPUT_DIR", "outputs/scfoundation", env_values=env_values),
            base_dir=root,
        ),
        test_tmp_dir=_resolve_path(
            _get_env_value("CHAOSSCORE_TEST_TMP_DIR", "tests/tmp", env_values=env_values),
            base_dir=root,
        ),
    )


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    """Return the singleton project path configuration for this process."""
    return load_project_paths()

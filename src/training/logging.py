from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.scfoundation_utils import ensure_parent_dir


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class JsonlRunLogger:
    """Print structured events and optionally persist them as JSONL."""

    def __init__(self, path: str | Path | None, *, append: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None and not append:
            path = ensure_parent_dir(self.path)
            with path.open("w", encoding="utf-8") as handle:
                handle.write("")

    def log_event(self, event: dict[str, Any]) -> None:
        event = {"timestamp": timestamp_now(), **event}
        line = json.dumps(event, sort_keys=True)
        print(line, flush=True)
        if self.path is not None:
            with ensure_parent_dir(self.path).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def run_start(self, *, save_dir: str | Path, command_args: dict[str, Any], resume: bool) -> None:
        self.log_event(
            {
                "event": "run_start",
                "save_dir": str(Path(save_dir).resolve()),
                "command_args": command_args,
                "resume": bool(resume),
            }
        )

    def run_end(self, *, save_dir: str | Path) -> None:
        self.log_event(
            {
                "event": "run_end",
                "save_dir": str(Path(save_dir).resolve()),
                "progress_log_path": str(self.path.resolve()) if self.path is not None else None,
            }
        )

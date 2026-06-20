from __future__ import annotations

from pathlib import Path


def default_config_path(project_root: Path = Path(".")) -> Path:
    root_config = project_root / "config.yaml"
    if root_config.exists():
        return root_config
    return project_root / "configs" / "smoke.yaml"


def read_accelerator_backend(config_path: Path) -> str | None:
    if not config_path.exists():
        return None

    current_section: str | None = None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            current_section = stripped[:-1].strip()
            continue

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not value:
            continue

        if key in {"accelerator_backend", "accelerator_device"}:
            return value.lower()
        if current_section == "accelerator" and key in {"backend", "device"}:
            return value.lower()
    return None

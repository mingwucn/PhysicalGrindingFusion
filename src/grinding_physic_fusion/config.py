import os
from typing import Any, Dict, Optional


class ProjectConfig:
    """Minimal project configuration loader with dotted-path access."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        data = _load_yaml(path)
        return cls(data)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        keys = dotted_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def require(self, dotted_path: str) -> Any:
        keys = dotted_path.split(".")
        value = self._data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise KeyError(f"Missing required config key: {dotted_path}")
            value = value[key]
        return value

    def get_int(self, dotted_path: str, default: Optional[int] = None) -> Optional[int]:
        value = self.get(dotted_path, default)
        if value is None:
            return None
        return int(value)

    def get_bool(self, dotted_path: str, default: Optional[bool] = None) -> Optional[bool]:
        value = self.get(dotted_path, default)
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")

    def get_str(self, dotted_path: str, default: Optional[str] = None) -> Optional[str]:
        value = self.get(dotted_path, default)
        if value is None:
            return None
        return str(value)

    def get_list(self, dotted_path: str, default: Optional[list] = None) -> Optional[list]:
        value = self.get(dotted_path, default)
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def section(self, key: str) -> "ProjectConfig":
        data = self._data.get(key, {})
        if not isinstance(data, dict):
            data = {}
        return ProjectConfig(data)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _minimal_yaml_load(path)


def _minimal_yaml_load(path: str) -> Dict[str, Any]:
    """Bare-bones YAML loader for top-level scalars and one-level nesting."""
    import re

    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped.strip() or stripped.strip().startswith("#"):
                continue
            match = re.match(r"^(\s*)([\w_]+)\s*:\s*(.*)$", stripped)
            if not match:
                continue
            indent, key, value = match.group(1), match.group(2), match.group(3).strip()
            if indent == "":
                if value == "":
                    result[key] = {}
                    current_section = key
                else:
                    result[key] = _coerce(value)
                    current_section = None
            elif current_section is not None and indent.startswith("  "):
                if value == "":
                    result[current_section][key] = {}
                else:
                    result[current_section][key] = _coerce(value)
    return result


def _coerce(value: str) -> Any:
    lower = value.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        return value

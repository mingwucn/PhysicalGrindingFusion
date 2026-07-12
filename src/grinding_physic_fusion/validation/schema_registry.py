import os
from typing import Any, Dict, Optional


class SchemaRegistry:
    """Loads and exposes the schema_registry.yaml index."""

    def __init__(self, registry_path: Optional[str] = None) -> None:
        if registry_path is None:
            here = os.path.dirname(__file__)
            registry_path = os.path.abspath(
                os.path.join(here, "..", "..", "..", "schemas", "schema_registry.yaml")
            )
        self._registry_path = registry_path
        self._index: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            import yaml

            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        self._index = data.get("schemas", {})

    def lookup(self, record_type: str) -> Optional[Dict[str, Any]]:
        entry = self._index.get(record_type)
        if entry is None:
            return None
        return {
            "schema_path": entry.get("schema_path"),
            "template_path": entry.get("template_path"),
            "validator": entry.get("validator"),
        }

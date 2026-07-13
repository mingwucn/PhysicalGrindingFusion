import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from include_verified_related_references import CANDIDATES, INSERTIONS  # noqa: E402


def test_inclusion_manifest_is_unique_and_mapped() -> None:
    keys = [candidate.key for candidate in CANDIDATES]
    assert len(keys) == len(set(keys))
    included = {candidate.key for candidate in CANDIDATES if candidate.include}
    assert len(included) == 6
    assert all(candidate.doi for candidate in CANDIDATES if candidate.include)
    mapped = {key for _, _, _, required in INSERTIONS for key in required}
    assert mapped == included

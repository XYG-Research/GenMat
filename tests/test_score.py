from __future__ import annotations

from genmat.score import _composition_display_fields


def test_composition_display_fields_include_ratio_percent_and_counts() -> None:
    ratio, percent, counts = _composition_display_fields({"Fe": 4, "Co": 2, "Ni": 2})

    assert ratio == "Co:Fe:Ni = 1:2:1"
    assert percent == "Co=25.0000%; Fe=50.0000%; Ni=25.0000%"
    assert counts == "Co=2; Fe=4; Ni=2"

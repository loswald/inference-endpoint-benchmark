from __future__ import annotations

from inference_bench.publication_safety import scan_publication


def test_publication_safety_scans_recursively(tmp_path) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "result.csv").write_text("sampling_unit,value\nrequest,1\n", encoding="utf-8")
    assert scan_publication(safe)["passed"] is True

    (safe / "private.jsonl").write_text("{}\n", encoding="utf-8")
    result = scan_publication(safe)
    assert result["passed"] is False
    assert result["findings"] == [
        {"file": "private.jsonl", "rule": "forbidden_raw_artifact"}
    ]

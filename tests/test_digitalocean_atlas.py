import csv
from pathlib import Path

from inference_bench import digitalocean_atlas


def _write_endpoint_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("endpoint_id",))
        writer.writeheader()
        writer.writerows(
            (
                {"endpoint_id": "keep-endpoint"},
                {"endpoint_id": "remove-endpoint"},
            )
        )


def test_digitalocean_atlas_excludes_endpoint_from_every_input_table(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "summary"
    source.mkdir()
    for name in (
        "endpoint-inventory.csv",
        "capacity-summary.csv",
        "soak-cell-summary.csv",
        "capability-evidence.csv",
        "observed-limits.csv",
    ):
        _write_endpoint_csv(source / name)

    captured: dict[str, list[dict[str, str]]] = {}
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_capacity",
        lambda rows, _source, _destination: captured.setdefault("capacity", rows) and [],
    )
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_soak",
        lambda rows, _source, _destination: captured.setdefault("soak", rows) and [],
    )
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_capabilities",
        lambda rows, _destination: captured.setdefault("capabilities", rows) and Path("figure"),
    )

    def capture_pdf(
        _report,
        inventory,
        capacity,
        soak,
        capabilities,
        limits,
        _figures,
        **_kwargs,
    ) -> None:
        captured.update(
            inventory=inventory,
            capacity_pdf=capacity,
            soak_pdf=soak,
            capabilities_pdf=capabilities,
            limits=limits,
        )

    monkeypatch.setattr(digitalocean_atlas, "_build_pdf", capture_pdf)
    digitalocean_atlas.generate_digitalocean_atlas(
        source,
        tmp_path / "digitalocean-atlas",
        capacity_source="capacity-source",
        soak_source="soak-source",
        exclude_endpoints=("remove-endpoint",),
    )

    for rows in captured.values():
        assert [row["endpoint_id"] for row in rows] == ["keep-endpoint"]

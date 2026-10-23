import pandas as pd

from mibici.stations import (
    add_station_metadata_flags,
    build_station_lifecycle,
    load_nomenclature,
)


def make_trips(rows):
    """Build minimal interim trips from (origin, destination, start) tuples."""
    trips = pd.DataFrame(rows, columns=["origin_station_id", "destination_station_id", "started_at"])
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips[["origin_station_id", "destination_station_id"]] = trips[
        ["origin_station_id", "destination_station_id"]
    ].astype("Int32")
    return trips


def write_nomenclature(path):
    path.write_text(
        '"id","name","latitude","longitude","dpcapacity"\n'
        '"1","(GDL-001) Uno","20.67","-103.35","15"\n'
        '"2","ex-(GDL-002) Dos","20.68","-103.36","0"\n'
        '"9","(GDL-009) Sin viajes","20.69","-103.37","19"\n'
    )
    return path


def test_load_nomenclature_types_and_retired_prefix(tmp_path):
    stations = load_nomenclature(write_nomenclature(tmp_path / "nomenclatura_2026_09.csv"), "h")

    assert str(stations["station_id"].dtype) == "Int32"
    assert list(stations["retired_in_snapshot"]) == [False, True, False]
    assert (stations["snapshot_sha256"] == "h").all()


def test_lifecycle_detects_gaps_new_retired_and_metadata(tmp_path):
    nomenclature = load_nomenclature(write_nomenclature(tmp_path / "nomenclatura_2026_09.csv"), "h")
    trips = make_trips(
        [
            # Estación 1: activa en 2025-01 y 2025-04 -> 2 meses inactivos en medio
            (1, 1, "2025-01-10"),
            (1, 1, "2025-04-10"),
            (1, 1, "2026-06-10"),
            # Estación 2: deja de tener viajes en 2025-01 -> retirada por actividad
            (2, 1, "2025-01-15"),
            # Estación 7: nueva en 2026 y sin metadatos
            (7, 1, "2026-06-20"),
        ]
    )

    stations = build_station_lifecycle(trips, nomenclature).set_index("station_id")

    assert stations.loc[1, "n_inactive_months"] == 2 + 13
    assert stations.loc[2, "is_retired_by_activity"]
    assert not stations.loc[1, "is_retired_by_activity"]
    assert stations.loc[7, "is_new_station"]
    assert not stations.loc[7, "has_metadata"]
    # La estación 9 está en el catálogo pero no tiene viajes: se conserva
    assert stations.loc[9, "has_metadata"] and not stations.loc[9, "has_trips"]


def test_station_metadata_flags(tmp_path):
    trips = make_trips([(1, 7, "2026-06-10"), (7, 1, "2026-06-10")])

    flagged = add_station_metadata_flags(trips, pd.Series([1, 2], dtype="Int32"))

    assert list(flagged["flag_origin_without_metadata"]) == [False, True]
    assert list(flagged["flag_destination_without_metadata"]) == [True, False]

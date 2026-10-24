"""Tests for mibici.station_hour: the station x hour grid and its valid zeros.

Small synthetic trips with a known answer: for each scenario (opening mid-month,
inactive month, retired station) we know exactly which hours must exist.
"""

import pandas as pd

from mibici.station_hour import build_station_hour_base, count_departures


def make_trips(rows):
    """Build minimal interim trips from (origin, destination, start) tuples."""
    trips = pd.DataFrame(
        rows, columns=["origin_station_id", "destination_station_id", "started_at"]
    )
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips[["origin_station_id", "destination_station_id"]] = trips[
        ["origin_station_id", "destination_station_id"]
    ].astype("Int32")
    return trips


def hours_of(base, station):
    """Rows of one station in the base."""
    return base[base["station_id"] == station]


def test_departures_are_counted_per_station_and_hour():
    trips = make_trips(
        [
            # Dos salidas de la estación 1 dentro de la misma hora (08:00-08:59)
            (1, 2, "2025-03-01 08:05"),
            (1, 2, "2025-03-01 08:55"),
            # Una salida de la estación 1 en la hora siguiente
            (1, 2, "2025-03-01 09:10"),
        ]
    )

    counts = count_departures(trips).set_index(["station_id", "hour_start"])["departures"]

    assert counts.loc[(1, pd.Timestamp("2025-03-01 08:00"))] == 2
    assert counts.loc[(1, pd.Timestamp("2025-03-01 09:00"))] == 1


def test_active_hour_without_trips_is_a_valid_zero():
    # La estación 1 opera un solo día: tiene 24 horas, solo la de las 08:00 con salidas
    base = build_station_hour_base(make_trips([(1, 2, "2025-03-01 08:05")]))
    station = hours_of(base, 1)

    assert len(station) == 24
    assert station["departures"].sum() == 1
    # Las otras 23 horas existen y valen 0 (ceros válidos)
    assert (station["departures"] == 0).sum() == 23


def test_no_rows_before_first_trip_even_mid_month():
    # La estación 1 abre el 25 de marzo: no debe haber filas del 1 al 24 de marzo
    trips = make_trips([(1, 2, "2025-03-25 10:00"), (1, 2, "2025-03-31 10:00")])

    station = hours_of(build_station_hour_base(trips), 1)

    assert station["hour_start"].min() == pd.Timestamp("2025-03-25 00:00")
    # Del 25 al 31 de marzo: 7 días x 24 horas
    assert len(station) == 7 * 24


def test_inactive_month_inside_the_window_has_no_rows():
    # La estación 1 tiene viajes en enero y en marzo, pero ninguno en febrero
    trips = make_trips([(1, 2, "2025-01-31 10:00"), (1, 2, "2025-03-01 10:00")])

    station = hours_of(build_station_hour_base(trips), 1)

    # Febrero no aparece: no se inventan ceros en un periodo inactivo
    assert not (station["hour_start"].dt.month == 2).any()
    # Solo el 31 de enero y el 1 de marzo: 2 días x 24 horas
    assert len(station) == 2 * 24


def test_no_rows_after_last_trip_of_a_retired_station():
    # La estación 1 deja de operar el 2 de marzo; la estación 2 sigue hasta el 10
    trips = make_trips([(1, 2, "2025-03-02 10:00"), (2, 3, "2025-03-10 10:00")])

    base = build_station_hour_base(trips)

    assert hours_of(base, 1)["hour_start"].max() == pd.Timestamp("2025-03-02 23:00")


def test_station_that_only_receives_trips_is_active_with_zero_departures():
    # La estación 2 solo aparece como destino: existe ese día, pero no tuvo salidas
    base = build_station_hour_base(make_trips([(1, 2, "2025-03-01 08:05")]))
    station = hours_of(base, 2)

    assert len(station) == 24
    assert station["departures"].sum() == 0


def test_calendar_columns():
    # 2025-03-01 fue sábado; la estación opera ese día y el lunes 3 (sin hueco de mes)
    trips = make_trips([(1, 2, "2025-03-01 08:05"), (1, 2, "2025-03-03 23:10")])

    base = hours_of(build_station_hour_base(trips), 1).set_index("hour_start")

    saturday = base.loc[pd.Timestamp("2025-03-01 08:00")]
    monday = base.loc[pd.Timestamp("2025-03-03 23:00")]
    assert saturday["hour"] == 8 and saturday["day_of_week"] == 5 and saturday["is_weekend"]
    assert monday["hour"] == 23 and monday["day_of_week"] == 0 and not monday["is_weekend"]
    assert saturday["date"] == pd.Timestamp("2025-03-01") and saturday["month"] == 3


def test_every_departure_lands_in_the_grid_and_keys_are_unique():
    trips = make_trips(
        [
            (1, 2, "2025-03-01 08:05"),
            (2, 1, "2025-03-01 23:59"),
            (1, 1, "2025-03-02 00:00"),
        ]
    )

    base = build_station_hour_base(trips)

    # Ninguna salida se pierde al construir la grilla
    assert base["departures"].sum() == len(trips)
    # No hay combinaciones estación-hora repetidas
    assert not base.duplicated(["station_id", "hour_start"]).any()

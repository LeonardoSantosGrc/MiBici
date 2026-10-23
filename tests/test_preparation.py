import pandas as pd
import pytest

from mibici.preparation import (
    add_duration,
    add_quality_flags,
    drop_exact_duplicates,
    load_manifest,
    prepare_file,
)

UTF8_HEADER = (
    "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,"
    "Origen_Id,Destino_Id\n"
)


def write_trips(path, rows, header=UTF8_HEADER, encoding="utf-8"):
    path.write_bytes((header + "".join(rows)).encode(encoding))
    return path


def prepare_rows(tmp_path, rows):
    """Write rows to a fake raw file and run the full interim preparation on it."""
    path = write_trips(tmp_path / "datos_abiertos_2020_01.csv", rows)
    trips = add_quality_flags(add_duration(prepare_file(path, "abc123")))
    return trips


def test_load_manifest_maps_filename_to_hash(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("filename,url,size_bytes,sha256,status\nf.csv,http://x,10,abc,existing\n")

    assert load_manifest(path) == {"f.csv": "abc"}


def test_prepare_file_types_names_and_lineage(tmp_path):
    trips = prepare_rows(
        tmp_path, ["1,10,M,1990.0,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n"]
    )

    assert str(trips["trip_id"].dtype) == "Int64"
    assert str(trips["birth_year"].dtype) == "Int16"
    assert str(trips["origin_station_id"].dtype) == "Int32"
    assert trips.loc[0, "birth_year"] == 1990
    assert trips.loc[0, "source_file"] == "datos_abiertos_2020_01.csv"
    assert trips.loc[0, "source_sha256"] == "abc123"


def test_prepare_file_reads_latin1_quoted_crlf_files(tmp_path):
    header = '"Viaje_Id","Usuario_Id","Genero","A\xf1o_de_nacimiento","Inicio_del_viaje",'
    header += '"Fin_del_viaje","Origen_Id","Destino_Id"\r\n'
    path = write_trips(
        tmp_path / "datos_abiertos_2020_01.csv",
        ['"1","10","F","1985","2020-01-01 08:00:00","2020-01-01 08:10:00","5","6"\r\n'],
        header=header,
        encoding="latin-1",
    )

    trips = prepare_file(path, "abc123")

    assert trips.loc[0, "gender"] == "F"
    assert trips.loc[0, "birth_year"] == 1985


def test_both_date_formats_are_parsed_and_tagged(tmp_path):
    trips = prepare_rows(
        tmp_path,
        [
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            "2,11,M,1990,15/02/2020 9:30,15/02/2020 9:40,5,6\n",
        ],
    )

    assert trips["started_at"].notna().all()
    # Día primero: 15/02 es 15 de febrero
    assert trips.loc[1, "started_at"] == pd.Timestamp("2020-02-15 09:30")
    assert list(trips["datetime_format"]) == ["iso", "slash_dayfirst"]


def test_missing_markers_become_na(tmp_path):
    trips = prepare_rows(
        tmp_path,
        [
            "1,10,NULL,,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            "2,11,,NA,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
        ],
    )

    assert trips["gender"].isna().all()
    assert trips["birth_year"].isna().all()
    # Un faltante no es una anomalía marcada
    assert not trips["flag_birth_year_implausible"].any()


def test_duration_and_quality_flags(tmp_path):
    trips = prepare_rows(
        tmp_path,
        [
            # 0: normal (10 min, estaciones distintas)
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            # 1: duración cero
            "2,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:00:00,5,6\n",
            # 2: 30 s y regresa a la misma estación
            "3,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:00:30,5,5\n",
            # 3: 30 s entre estaciones distintas
            "4,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:00:30,5,6\n",
            # 4: más de 24 horas
            "5,10,M,1990,2020-01-01 08:00:00,2020-01-02 09:00:00,5,6\n",
            # 5: duración negativa y año de nacimiento implausible
            "6,10,M,1800,2020-01-01 08:10:00,2020-01-01 08:00:00,5,6\n",
        ],
    )

    assert list(trips["duration_s"]) == [600, 0, 30, 30, 90000, -600]
    assert list(trips["flag_duration_zero"]) == [False, True, False, False, False, False]
    assert list(trips["flag_short_same_station"]) == [False, False, True, False, False, False]
    assert list(trips["flag_short_cross_station"]) == [False, True, False, True, False, False]
    assert list(trips["flag_duration_over_24h"]) == [False, False, False, False, True, False]
    assert list(trips["flag_duration_negative"]) == [False, False, False, False, False, True]
    assert list(trips["flag_birth_year_implausible"]) == [False] * 5 + [True]
    # Los flags solo marcan: no se elimina ninguna fila
    assert len(trips) == 6


def test_drop_exact_duplicates_removes_only_identical_trips(tmp_path):
    row = "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n"
    # Mismo viaje con otro destino: no es duplicado exacto
    almost = "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,7\n"
    trips = prepare_rows(tmp_path, [row, row, almost])

    deduplicated, removed = drop_exact_duplicates(trips)

    assert removed == 1
    assert len(deduplicated) == 2


def test_prepare_file_rejects_unexpected_header(tmp_path):
    # Mismas 8 columnas pero Origen y Destino invertidos: no debe leerse en silencio
    header = UTF8_HEADER.replace("Origen_Id,Destino_Id", "Destino_Id,Origen_Id")
    path = write_trips(
        tmp_path / "datos_abiertos_2020_01.csv",
        ["1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n"],
        header=header,
    )

    with pytest.raises(ValueError, match="unexpected trip header"):
        prepare_file(path, "abc123")

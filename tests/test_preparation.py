"""Tests for mibici.preparation: the rules that turn raw trips into interim trips.

Each test writes a tiny synthetic CSV whose correct answer is known in advance
(like calibrating a scale with a known weight) and checks that the code gives
exactly that answer. ``data/raw`` is never touched.

Structure of every test: 1) prepare the data, 2) run the function, 3) assert.
"""

import pandas as pd
import pytest

from mibici.preparation import (
    add_duration,
    add_quality_flags,
    drop_exact_duplicates,
    load_manifest,
    prepare_file,
)

# Encabezado raw más común (UTF-8, sin comillas, fin de línea LF)
UTF8_HEADER = (
    "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,"
    "Origen_Id,Destino_Id\n"
)


def write_trips(path, rows, header=UTF8_HEADER, encoding="utf-8"):
    """Write a fake raw trips CSV (header + rows) with the given encoding."""
    # write_bytes permite simular archivos latin-1, igual que los raw de MiBici
    path.write_bytes((header + "".join(rows)).encode(encoding))
    return path


def prepare_rows(tmp_path, rows):
    """Write rows to a fake raw file and run the full interim preparation on it."""
    # tmp_path es una carpeta temporal que pytest crea para cada prueba y luego borra
    path = write_trips(tmp_path / "datos_abiertos_2020_01.csv", rows)
    trips = add_quality_flags(add_duration(prepare_file(path, "abc123")))
    return trips


def test_load_manifest_maps_filename_to_hash(tmp_path):
    # Manifest mínimo de una fila, con las mismas columnas que el real
    path = tmp_path / "manifest.csv"
    path.write_text("filename,url,size_bytes,sha256,status\nf.csv,http://x,10,abc,existing\n")

    # Debe devolver el diccionario nombre de archivo -> hash
    assert load_manifest(path) == {"f.csv": "abc"}


def test_prepare_file_types_names_and_lineage(tmp_path):
    # Año de nacimiento escrito como "1990.0", como en ~16 M de registros raw
    trips = prepare_rows(
        tmp_path, ["1,10,M,1990.0,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n"]
    )

    # Tipos del esquema interim (enteros nullable con I mayúscula)
    assert str(trips["trip_id"].dtype) == "Int64"
    assert str(trips["birth_year"].dtype) == "Int16"
    assert str(trips["origin_station_id"].dtype) == "Int32"
    # "1990.0" se convierte en el entero 1990
    assert trips.loc[0, "birth_year"] == 1990
    # Linaje: cada fila sabe de qué archivo y de qué versión (hash) salió
    assert trips.loc[0, "source_file"] == "datos_abiertos_2020_01.csv"
    assert trips.loc[0, "source_sha256"] == "abc123"


def test_prepare_file_reads_latin1_quoted_crlf_files(tmp_path):
    # El caso más difícil de raw: latin-1 ("\xf1" es la ñ), todo entre comillas y CRLF
    header = '"Viaje_Id","Usuario_Id","Genero","A\xf1o_de_nacimiento","Inicio_del_viaje",'
    header += '"Fin_del_viaje","Origen_Id","Destino_Id"\r\n'
    path = write_trips(
        tmp_path / "datos_abiertos_2020_01.csv",
        ['"1","10","F","1985","2020-01-01 08:00:00","2020-01-01 08:10:00","5","6"\r\n'],
        header=header,
        encoding="latin-1",
    )

    trips = prepare_file(path, "abc123")

    # Se lee igual que un archivo UTF-8 simple: sin comillas y con los tipos correctos
    assert trips.loc[0, "gender"] == "F"
    assert trips.loc[0, "birth_year"] == 1985


def test_both_date_formats_are_parsed_and_tagged(tmp_path):
    trips = prepare_rows(
        tmp_path,
        [
            # Formato ISO (casi todos los archivos)
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            # Formato con diagonal y día primero, sin segundos (2023-11, 2024-09, 2024-10)
            "2,11,M,1990,15/02/2020 9:30,15/02/2020 9:40,5,6\n",
        ],
    )

    # Ninguna fecha queda sin interpretar (NaT)
    assert trips["started_at"].notna().all()
    # Día primero: 15/02 es 15 de febrero
    assert trips.loc[1, "started_at"] == pd.Timestamp("2020-02-15 09:30")
    # Cada fila conserva con qué formato venía
    assert list(trips["datetime_format"]) == ["iso", "slash_dayfirst"]


def test_missing_markers_become_na(tmp_path):
    # Distintas formas de escribir "faltante" en raw: "NULL", vacío y "NA"
    trips = prepare_rows(
        tmp_path,
        [
            "1,10,NULL,,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            "2,11,,NA,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
        ],
    )

    # Todas terminan como el mismo marcador: <NA>
    assert trips["gender"].isna().all()
    assert trips["birth_year"].isna().all()
    # Un faltante no es una anomalía marcada
    assert not trips["flag_birth_year_implausible"].any()


def test_duration_and_quality_flags(tmp_path):
    # Una fila por cada caso que un flag debe detectar; el comentario dice qué se espera
    trips = prepare_rows(
        tmp_path,
        [
            # 0: normal (10 min, estaciones distintas)
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            # 1: duración cero (también es "corto" entre estaciones distintas)
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

    # Duración en segundos de cada fila: fin - inicio
    assert list(trips["duration_s"]) == [600, 0, 30, 30, 90000, -600]
    # Cada lista se lee por posición: la fila i debe tener el valor i
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
    # "row" aparece dos veces (1 duplicado) y "almost" una vez
    trips = prepare_rows(tmp_path, [row, row, almost])

    deduplicated, removed = drop_exact_duplicates(trips)

    # Solo se quita la repetición idéntica; el "casi igual" se conserva
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

    # pytest.raises: la prueba pasa solo si prepare_file lanza este error
    with pytest.raises(ValueError, match="unexpected trip header"):
        prepare_file(path, "abc123")

from mibici.profiling import (
    header_matches_trip_layout,
    list_raw_files,
    profile_trips,
    read_header,
    read_raw_trips,
)

UTF8_HEADER = (
    "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,"
    "Origen_Id,Destino_Id\n"
)


def write_trips(path, rows, header=UTF8_HEADER, encoding="utf-8"):
    path.write_bytes((header + "".join(rows)).encode(encoding))
    return path


def test_list_raw_files_detects_kind_period_and_irregular_names(tmp_path):
    for name in [
        "datos_abiertos_2016_01-1.csv",
        "datos_abiertos_2016_02.csv",
        "nomenclatura_2026_09.csv",
    ]:
        (tmp_path / name).write_text("x\n")

    files = list_raw_files(tmp_path).set_index("filename")

    assert files.loc["datos_abiertos_2016_01-1.csv", "irregular_name"]
    assert files.loc["datos_abiertos_2016_02.csv", "month"] == 2
    assert files.loc["nomenclatura_2026_09.csv", "kind"] == "nomenclature"


def test_header_layout_accepts_encoding_variants_of_birth_year():
    base = UTF8_HEADER.strip().split(",")
    corrupted = [*base[:3], "A}\xe4e_nacimiento", *base[4:]]

    assert header_matches_trip_layout(base)
    assert header_matches_trip_layout(corrupted)
    assert not header_matches_trip_layout(base[:-1])


def test_read_raw_trips_keeps_empty_values_and_canonical_names(tmp_path):
    header = '"Viaje_Id","Usuario_Id","Genero","A\xf1o_de_nacimiento","Inicio_del_viaje",'
    header += '"Fin_del_viaje","Origen_Id","Destino_Id"\r\n'
    path = write_trips(
        tmp_path / "datos_abiertos_2020_01.csv",
        ['"1","10","M","","2020-01-01 08:00:00","2020-01-01 08:10:00","5","6"\r\n'],
        header=header,
        encoding="latin-1",
    )

    df = read_raw_trips(path)
    info = read_header(path)

    assert df.columns[3] == "Año_de_nacimiento"
    assert df.loc[0, "Año_de_nacimiento"] == ""
    assert info["quoted_header"] and info["line_ending"] == "crlf"


def test_profile_trips_counts_quality_issues(tmp_path):
    path = write_trips(
        tmp_path / "datos_abiertos_2020_01.csv",
        [
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            "1,10,M,1990,2020-01-01 08:00:00,2020-01-01 08:10:00,5,6\n",
            "2,11,NULL,,2020-01-02 09:00:00,2020-01-02 08:59:00,5,5\n",
            "3,12,,1800.0,15/02/2020 9:30,15/02/2020 9:40,7,x\n",
        ],
    )

    profile = profile_trips(read_raw_trips(path), year=2020, month=1)

    assert profile["n_rows"] == 4
    assert profile["duplicated_rows"] == 1
    assert profile["duplicated_Viaje_Id"] == 1
    assert profile["empty_Año_de_nacimiento"] == 1
    assert profile["non_numeric_Destino_Id"] == 1
    assert profile["decimal_text_Año_de_nacimiento"] == 1
    assert profile["non_numeric_Año_de_nacimiento"] == 0
    assert profile["missing_token_Genero"] == 1
    assert profile["duration_negative"] == 1
    assert profile["format_slash_dayfirst_Inicio_del_viaje"] == 1
    assert profile["slash_first_field_gt_12"] == 1
    assert profile["start_outside_file_month"] == 1
    assert profile["birth_year_lt_1920"] == 1
    assert profile["same_origin_destination"] == 1

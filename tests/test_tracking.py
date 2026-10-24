"""Tests for mibici.tracking: the dataset fingerprint logged to MLflow."""

import pandas as pd

from mibici.tracking import dataframe_digest


def test_digest_is_stable_and_detects_changes_beyond_first_rows():
    # 20,000 filas: más que las 10,000 que usa el digest por defecto de MLflow
    df = pd.DataFrame({"trip_id": range(20_000), "station": [5] * 20_000})
    changed = df.copy()
    # Se cambia un solo valor en la última fila
    changed.loc[19_999, "station"] = 6

    # Misma tabla -> misma huella (reproducible entre ejecuciones)
    assert dataframe_digest(df) == dataframe_digest(df.copy())
    # Un cambio en cualquier fila, aunque sea la última, cambia la huella
    assert dataframe_digest(df) != dataframe_digest(changed)

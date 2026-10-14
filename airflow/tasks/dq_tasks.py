"""Data Quality checks and an HTML report for the active training window."""

from datetime import date, datetime, timezone
from html import escape

import numpy as np
import pandas as pd

from pipeline import (
    HISTORY_MONTHS, REPORT_DIR, WINDOW_MONTHS, connect,
    read_json, run_directory, shift_month, window_rows, write_json,
)


def write_report(run_id):
    """Read the window, save all check results and HTML before signalling failure.

    Outputs are deterministic paths for this run plus latest.html. Retries replace
    these files and never modify the labelled store or active model.
    """
    current, latest, earliest, rows = window_rows()
    frame = pd.DataFrame([dict(data, target=target) for _, _, data, target in rows])
    required = [
        "transaction_time", "amount", "population_city", "lat", "lon",
        "merchant_lat", "merchant_lon", "merch", "cat_id", "gender", "us_state", "jobs", "target",
    ]
    checks = []
    def check(name, ok, detail):
        checks.append({"rule": name, "passed": bool(ok), "detail": str(detail)})

    check("Окно не пустое", len(frame) > 0, f"{len(frame)} строк")
    missing = [field for field in required if field not in frame.columns]
    nulls = sum(int(frame[field].isna().sum() + frame[field].eq("").sum())
                for field in required if field in frame.columns)
    check("Обязательные поля заполнены", not missing and nulls == 0,
          f"Отсутствующие поля: {missing}; пустые значения: {nulls}")
    for column in ["amount", "population_city", "lat", "lon", "merchant_lat", "merchant_lon", "target"]:
        frame[column] = pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")
    for column in ["amount", "population_city"]:
        values = frame[column]
        bad = int((~np.isfinite(values) | (values < 0)).sum())
        check(f"{column}: конечное неотрицательное число", bad == 0, f"Нарушений: {bad}")
    geo_ok = (
        frame["lat"].between(-90, 90) & frame["merchant_lat"].between(-90, 90)
        & frame["lon"].between(-180, 180) & frame["merchant_lon"].between(-180, 180)
    )
    check("Координаты в допустимых диапазонах", geo_ok.all(), f"Нарушений: {int((~geo_ok).sum())}")
    check("Метка target равна 0 или 1", frame["target"].isin([0, 1]).all(),
          f"Нарушений: {int((~frame['target'].isin([0, 1])).sum())}")
    times = pd.to_datetime(
        frame.get("transaction_time", pd.Series(index=frame.index, dtype=object)),
        format="mixed",
        errors="coerce",
    )
    stored_times = pd.to_datetime(
        [timestamp for _, timestamp, _, _ in rows], format="mixed", errors="coerce"
    )
    check("Дата разбирается и совпадает с витриной",
          times.notna().all() and np.array_equal(times.to_numpy(), stored_times.to_numpy()),
          f"Нераспознанных дат: {int(times.isna().sum())}")
    ids = [identifier for identifier, _, _, _ in rows]
    check("Идентификаторы уникальны", len(ids) == len(set(ids)), f"{len(set(ids))} уникальных ID")
    actual_months = {date(timestamp.year, timestamp.month, 1) for _, timestamp, _, _ in rows}
    expected_months = {shift_month(earliest, n) for n in range(WINDOW_MONTHS)}
    check("Окно содержит последовательные месяцы", actual_months == expected_months,
          ", ".join(str(month) for month in sorted(actual_months)))
    train_targets = {target for _, timestamp, _, target in rows if timestamp.date() < latest}
    test_targets = {target for _, timestamp, _, target in rows if timestamp.date() >= latest}
    check("Оба класса есть в train и test", train_targets == test_targets == {0, 1},
          f"train: {sorted(train_targets)}; test: {sorted(test_targets)}")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("""
            SELECT min(transaction_time), max(transaction_time),
                   (SELECT current_month FROM training_state WHERE id = 1)
            FROM labelled_transactions
        """)
        oldest, newest, current = cursor.fetchone()
    cutoff = shift_month(month_start(newest), 1 - HISTORY_MONTHS) if newest else earliest
    # The published model pointer may lag behind the labelled store when a
    # candidate is rejected.  The append DAG is still allowed to advance the
    # store, so DQ must validate retention depth rather than compare the newest
    # row with the model's month pointer.
    check("Глубина истории соблюдена",
          oldest is not None and oldest.date() >= cutoff,
          f"Витрина: {oldest} — {newest}; указатель модели: {current}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    work = run_directory(run_id)
    results = {"window": [earliest.isoformat(), latest.isoformat()], "checks": checks,
               "passed": all(item["passed"] for item in checks)}
    write_json(work / "dq.json", results)
    table_rows = "".join(
        f"<tr><td>{escape(item['rule'])}</td><td>{'OK' if item['passed'] else 'FAIL'}</td>"
        f"<td>{escape(item['detail'])}</td></tr>" for item in checks
    )
    html = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Data Quality</title>'
        "<h1>Data Quality</h1>"
        f"<p>{escape(datetime.now(timezone.utc).isoformat())}; окно: {earliest} — {latest}</p>"
        "<table border='1' cellpadding='6'><tr><th>Правило</th><th>Результат</th><th>Детали</th></tr>"
        + table_rows + "</table></html>"
    )
    report_path = REPORT_DIR / (work.name + ".html")
    report_path.write_text(html, encoding="utf-8")
    temporary = REPORT_DIR / "latest.html.tmp"
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(REPORT_DIR / "latest.html")
    print(f"Report: {report_path}; passed: {results['passed']}")


def month_start(value):
    return date(value.year, value.month, 1)


def assert_quality(run_id):
    """Read this run's saved DQ results; fail only after the HTML is available.

    Reads dq.json, writes nothing. Repeating the task yields the same verdict.
    """
    results = read_json(run_directory(run_id) / "dq.json")
    failed = [item["rule"] for item in results["checks"] if not item["passed"]]
    if failed:
        raise ValueError("Data Quality failed: " + "; ".join(failed))

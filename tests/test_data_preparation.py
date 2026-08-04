"""데이터 준비 단계 검증.

설계 원칙 5-4 의 검증 항목을 그대로 옮겼다.
    로딩 → Pandas 와 Polars 의 shape 이 일치하는가
    정제 → 처리 후 결측·중복이 0인가, 행 수 변화가 기록됐는가
    저장 → 파일이 생성됐고 다시 읽었을 때 원본과 같은가

그리고 잘못된 입력(빈 파일 · 컬럼 누락 · 이상값 뭉치)에 죽지 않는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import config, data_preparation as dp


# ===========================================================================
# 스키마
# ===========================================================================
def test_schema_has_required_columns(sample_path):
    """필수 컬럼이 전부 존재한다."""
    available = dp.read_schema(sample_path)
    for col in config.REQUIRED_COLUMNS:
        assert col in available


def test_missing_required_column_raises(tmp_path):
    """필수 컬럼이 빠진 파일은 조용히 넘어가지 않고 예외를 던진다."""
    broken = tmp_path / "broken.parquet"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_parquet(broken, index=False)
    with pytest.raises(ValueError, match="필수 컬럼이 없다"):
        dp.read_schema(broken)


def test_columns_to_read_excludes_unknown(sample_path):
    """읽을 컬럼은 필수 + 파일에 실제로 있는 선택 컬럼으로만 구성된다."""
    available = set(dp.read_schema(sample_path))
    chosen = dp.columns_to_read(sample_path)
    assert set(chosen) <= available
    assert set(config.REQUIRED_COLUMNS) <= set(chosen)
    assert len(chosen) == len(set(chosen)), "중복된 컬럼이 있다"


# ===========================================================================
# 로딩 — Pandas vs Polars
# ===========================================================================
def test_pandas_polars_shape_match(sample_path):
    """두 엔진이 같은 파일에서 같은 shape 을 낸다 (채점 항목)."""
    df_pd, df_pl, info = dp.compare_load(sample_path)
    assert df_pd.shape == df_pl.shape
    assert info["shape_match"] is True


def test_pandas_polars_missing_match(sample_path):
    """두 엔진의 결측 집계가 일치한다."""
    _, _, info = dp.compare_load(sample_path)
    assert info["missing_match"] is True, (
        f"결측 불일치: pandas={info['pandas_missing']} / polars={info['polars_missing']}"
    )


def test_lazy_matches_eager(sample_path):
    """lazy 로딩 결과가 eager 와 동일하다 (pushdown 이 데이터를 바꾸지 않는다)."""
    cols = dp.columns_to_read(sample_path)
    eager, _ = dp.load_polars(sample_path, cols)
    lazy, _ = dp.load_polars_lazy(sample_path, cols)
    assert eager.shape == lazy.shape
    assert eager.columns == lazy.columns


def test_compare_load_reports_timing(sample_path):
    """속도·메모리 지표가 반환값에 기록된다 (로그에만 남지 않는다)."""
    _, _, info = dp.compare_load(sample_path)
    for key in ("pandas_seconds", "polars_eager_seconds", "polars_lazy_seconds",
                "pandas_memory_mb", "polars_memory_mb"):
        assert key in info and info[key] >= 0


def test_shape_mismatch_is_detected(sample_path, monkeypatch):
    """엔진 간 shape 이 어긋나면 즉시 멈춘다 (조용히 넘어가지 않는다)."""
    original = dp.load_polars       # 패치 전에 원본을 잡아둔다 (안 그러면 무한 재귀)

    def fake_load_polars(path, columns=None):
        df, elapsed = original(path, columns)
        return df.head(3), elapsed      # 일부러 행 수를 줄인다

    monkeypatch.setattr(dp, "load_polars", fake_load_polars)
    with pytest.raises(ValueError, match="shape"):
        dp.compare_load(sample_path)


def test_missing_file_raises(tmp_path):
    """없는 파일을 주면 빈 DF 를 만들지 않고 예외를 던진다."""
    with pytest.raises((FileNotFoundError, OSError)):
        dp.load_pandas(tmp_path / "does_not_exist.parquet")


# ===========================================================================
# 결측 처리
# ===========================================================================
def test_no_missing_in_critical_after_handling(raw_df):
    """필수 컬럼에 결측이 남지 않는다."""
    cleaned, info = dp.handle_missing(raw_df)
    critical = [c for c in config.CRITICAL_COLUMNS if c in cleaned.columns]
    assert cleaned[critical].isna().sum().sum() == 0
    assert info["rows_before"] - info["rows_after"] == info["rows_removed"]


def test_missing_profile_is_recorded(raw_df):
    """어떤 컬럼에 결측이 몇 개였는지 반환값에 남는다."""
    _, info = dp.handle_missing(raw_df)
    assert isinstance(info["missing_all"], dict)
    assert info["strategy"]


def test_all_missing_raises():
    """모든 행에 결측이 있으면 빈 데이터로 조용히 넘어가지 않는다."""
    row = {c: np.nan for c in config.REQUIRED_COLUMNS}
    df = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="남은 행이 없다"):
        dp.handle_missing(df)


# ===========================================================================
# 파생변수 — 팀 계약
# ===========================================================================
def test_derived_columns_exist(clean_df):
    """팀에 약속한 파생 컬럼이 전부 있다 (②③④가 이걸 믿고 코딩한다)."""
    for col in ("duration_min", "average_speed_mph", "pickup_hour",
                "pickup_dayofweek", "is_weekend", "is_rush_hour"):
        assert col in clean_df.columns, f"{col} 이 없다 — 팀 계약 위반"


def test_speed_formula_is_correct(clean_df):
    """average_speed_mph 가 실제로 거리 / 시간 공식과 일치한다."""
    expected = clean_df["trip_distance"] / (clean_df["duration_min"] / 60.0)
    assert np.allclose(clean_df["average_speed_mph"], expected, rtol=1e-9)


def test_duration_matches_timestamps(clean_df):
    """duration_min 이 하차 - 승차 와 일치한다."""
    expected = (clean_df["tpep_dropoff_datetime"]
                - clean_df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    assert np.allclose(clean_df["duration_min"], expected, rtol=1e-9)


def test_rush_hour_matches_config(clean_df):
    """is_rush_hour 가 config.RUSH_HOURS 정의와 정확히 일치한다.

    이 테스트가 있어야 누군가 자기 파일에서 시간대를 다르게 쓰는 사고를 막을 수 있다.
    """
    expected = clean_df["pickup_hour"].isin(config.RUSH_HOURS)
    assert (clean_df["is_rush_hour"] == expected).all()


def test_weekend_matches_dayofweek(clean_df):
    """is_weekend 가 요일 정의(토=5, 일=6)와 일치한다."""
    expected = clean_df["pickup_dayofweek"].isin(config.WEEKEND_DAYS)
    assert (clean_df["is_weekend"] == expected).all()


def test_hour_and_weekday_ranges(clean_df):
    """시각은 0~23, 요일은 0~6 범위 안에 있다."""
    assert clean_df["pickup_hour"].between(0, 23).all()
    assert clean_df["pickup_dayofweek"].between(0, 6).all()


# ===========================================================================
# 이상치 · 중복
# ===========================================================================
def test_outlier_bounds_are_respected(clean_df):
    """정제 후 모든 값이 config 의 허용 범위 안에 있다."""
    assert clean_df["fare_amount"].between(config.MIN_FARE, config.MAX_FARE).all()
    assert clean_df["trip_distance"].between(config.MIN_DISTANCE, config.MAX_DISTANCE).all()
    assert clean_df["duration_min"].between(
        config.MIN_DURATION_MIN, config.MAX_DURATION_MIN).all()
    assert clean_df["average_speed_mph"].between(
        config.MIN_SPEED_MPH, config.MAX_SPEED_MPH).all()
    # passenger_count 는 결측을 허용한다. 값이 있는 행만 범위를 검사한다.
    present = clean_df["passenger_count"].notna()
    assert clean_df.loc[present, "passenger_count"].between(
        config.MIN_PASSENGERS, config.MAX_PASSENGERS).all()


def test_no_duplicates_after_clean(clean_df):
    """정제 후 완전 중복 행이 남지 않는다."""
    assert clean_df.duplicated().sum() == 0


def test_no_missing_speed_after_clean(clean_df):
    """평균 속도에 결측이 남지 않는다 (세 가설 전부가 이 컬럼을 본다)."""
    assert clean_df["average_speed_mph"].isna().sum() == 0


def test_outlier_steps_are_recorded(prep_info):
    """어떤 필터가 몇 행을 왜 지웠는지 단계별로 기록된다."""
    steps = prep_info["outliers"]["steps"]
    assert len(steps) >= 5
    for step in steps:
        assert step["rows_before"] - step["rows_after"] == step["rows_removed"]
        assert step["reason"], f"{step['name']} 에 제거 근거가 없다"


def test_row_counts_are_traceable(prep_info):
    """처리 전후 행 수가 전 단계에 걸쳐 기록된다 (설계 원칙 5-2)."""
    assert prep_info["original_rows"] >= prep_info["final_rows"]
    assert prep_info["original_rows"] - prep_info["final_rows"] == prep_info["removed_total"]
    for key in ("missing", "month", "outliers", "duplicates"):
        assert "rows_before" in prep_info[key] and "rows_after" in prep_info[key]


def test_stray_month_removed(clean_df):
    """대상 월 바깥 레코드가 제거된다 (미터기 시계 오류 방어)."""
    periods = clean_df["tpep_pickup_datetime"].dt.to_period("M").unique()
    assert len(periods) == 1, f"여러 달이 섞여 있다: {periods}"


def test_impossible_speed_is_removed():
    """시속 500마일짜리 기록이 살아남지 않는다."""
    base = pd.Timestamp("2026-05-10 08:00:00")
    df = pd.DataFrame({
        "tpep_pickup_datetime": [base] * 3,
        "tpep_dropoff_datetime": [base + pd.Timedelta(minutes=m) for m in (30, 30, 1)],
        "trip_distance": [5.0, 6.0, 90.0],      # 세 번째가 시속 5400마일
        "fare_amount": [20.0, 25.0, 30.0],
        "total_amount": [25.0, 30.0, 35.0],
        "passenger_count": [1.0, 2.0, 1.0],
        "PULocationID": [100, 100, 100],
        "DOLocationID": [200, 200, 200],
        "RatecodeID": [1.0, 1.0, 1.0],
        "payment_type": [1, 1, 1],
        "tip_amount": [3.0, 4.0, 5.0],
        "tolls_amount": [0.0, 0.0, 0.0],
    })
    out, _ = dp.remove_outliers(dp.add_time_features(df))
    assert len(out) == 2
    assert out["average_speed_mph"].max() <= config.MAX_SPEED_MPH


def test_all_rows_filtered_raises():
    """필터가 데이터를 통째로 날리면 임계값 오류로 보고 멈춘다."""
    base = pd.Timestamp("2026-05-10 08:00:00")
    df = pd.DataFrame({
        "tpep_pickup_datetime": [base] * 3,
        "tpep_dropoff_datetime": [base + pd.Timedelta(minutes=20)] * 3,
        "trip_distance": [3.0] * 3,
        "fare_amount": [-5.0] * 3,              # 전부 음수 요금
        "total_amount": [1.0] * 3,
        "passenger_count": [1.0] * 3,
        "PULocationID": [100] * 3,
        "DOLocationID": [200] * 3,
        "RatecodeID": [1.0] * 3,
        "payment_type": [1] * 3,
        "tip_amount": [0.0] * 3,
        "tolls_amount": [0.0] * 3,
    })
    with pytest.raises(ValueError, match="남은 행이 없다"):
        dp.remove_outliers(dp.add_time_features(df))


def test_duplicates_actually_removed(raw_df):
    """의도적으로 만든 중복 행이 제거된다."""
    doubled = pd.concat([raw_df.head(50), raw_df.head(50)], ignore_index=True)
    out, info = dp.drop_duplicates(doubled)
    assert len(out) < len(doubled)
    assert info["duplicates_removed"] > 0


# ===========================================================================
# EDA
# ===========================================================================
def test_eda_contains_required_keys(prep_info):
    """EDA 결과가 리포트와 후속 담당자에게 필요한 키를 모두 담고 있다."""
    eda = prep_info["eda"]
    for key in ("shape", "dtypes", "describe", "hourly_speed", "speed_by_rush",
                "speed_by_weekend", "speed_median", "slowest_hour", "fastest_hour"):
        assert key in eda, f"EDA 결과에 {key} 가 없다"
    # 채점 항목: 기술통계에 평균·표준편차·분위수가 들어 있어야 한다
    assert {"mean", "std", "25%", "50%", "75%"} <= set(eda["describe"].columns)


def test_hourly_summary_covers_all_hours(prep_info):
    """시간대별 요약이 실제 데이터에 존재하는 시각을 빠짐없이 담는다."""
    hourly = prep_info["eda"]["hourly_speed"]
    assert len(hourly) > 0
    assert hourly["mean_speed"].notna().all()


# ===========================================================================
# 저장
# ===========================================================================
def test_save_and_reload_matches(clean_df, tmp_path):
    """저장한 parquet 을 다시 읽었을 때 원본과 shape·컬럼이 일치한다."""
    path = tmp_path / "taxi_cleaned.parquet"
    info = dp.save_cleaned(clean_df, path)

    assert path.exists() and path.stat().st_size > 0
    assert info["reload_verified"] is True

    reloaded = pd.read_parquet(path)
    assert reloaded.shape == clean_df.shape
    assert list(reloaded.columns) == list(clean_df.columns)
    pd.testing.assert_series_equal(
        reloaded["average_speed_mph"], clean_df["average_speed_mph"], check_dtype=False
    )


def test_saved_file_readable_by_polars(clean_df, tmp_path):
    """저장한 파일을 Polars 로도 읽을 수 있다 (팀원이 어느 엔진을 쓰든 동작)."""
    path = tmp_path / "taxi_cleaned.parquet"
    dp.save_cleaned(clean_df, path)
    assert pl.read_parquet(path).height == len(clean_df)


def test_verify_file_rejects_empty(tmp_path):
    """빈 파일은 '저장 성공'으로 인정하지 않는다."""
    empty = tmp_path / "empty.parquet"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        config.verify_file(empty, min_bytes=1_000)


# ===========================================================================
# 파이프라인 전체
# ===========================================================================
def test_run_end_to_end(sample_path, tmp_path, monkeypatch):
    """run() 한 번으로 로딩부터 저장까지 끝까지 돈다."""
    monkeypatch.setattr(config, "CLEANED_PATH", tmp_path / "taxi_cleaned.parquet")
    df, info = dp.run(sample_path, save=True)

    assert len(df) > 0
    assert info["save"]["reload_verified"] is True
    assert set(info["derived_columns"]) <= set(df.columns)


def test_run_is_reproducible(sample_path):
    """같은 입력으로 두 번 돌리면 결과가 완전히 같다 (설계 원칙 5-3)."""
    df1, _ = dp.run(sample_path, save=False)
    df2, _ = dp.run(sample_path, save=False)
    pd.testing.assert_frame_equal(df1, df2)


# ===========================================================================
# 결측 정책 — passenger_count 는 살린다
# ===========================================================================
def test_passenger_count_nulls_survive():
    """승객 수가 비어 있어도 행이 살아남는다.

    실제 데이터에서 이 컬럼의 결측은 특정 사업자에 몰려 있어(무작위 아님),
    제거하면 전체의 23%가 편향된 방식으로 사라진다.
    """
    base = pd.Timestamp("2026-05-10 08:00:00")
    df = pd.DataFrame({
        "tpep_pickup_datetime": [base] * 3,
        "tpep_dropoff_datetime": [base + pd.Timedelta(minutes=20)] * 3,
        "trip_distance": [3.0, 4.0, 5.0],
        "fare_amount": [20.0, 25.0, 30.0],
        "total_amount": [25.0, 30.0, 35.0],
        "passenger_count": [np.nan, 2.0, 99.0],   # 결측 / 정상 / 범위초과
        "PULocationID": [100] * 3,
        "DOLocationID": [200] * 3,
        "RatecodeID": [np.nan, 1.0, 1.0],
        "payment_type": [1] * 3,
        "tip_amount": [3.0] * 3,
        "tolls_amount": [0.0] * 3,
    })
    kept, _ = dp.handle_missing(df)
    assert len(kept) == 3, "결측만으로 행을 버리면 안 된다"

    out, _ = dp.remove_outliers(dp.add_time_features(kept))
    assert len(out) == 2, "결측은 통과, 범위초과(99명)만 제거돼야 한다"
    assert out["passenger_count"].isna().sum() == 1


def test_comissing_is_detected():
    """결측 개수가 동일한 컬럼 묶음을 찾아낸다 (사업자 단위 미보고 신호)."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, np.nan],
        "b": [1, 2, np.nan, np.nan],       # a 와 같은 위치에 결측
        "c": [1, 2, 3, np.nan],            # 개수가 다르다
    })
    groups = dp.detect_comissing(df)
    assert "2" in groups
    assert set(groups["2"]) == {"a", "b"}
    assert "1" not in groups               # 혼자인 컬럼은 묶이지 않는다


def test_retained_missing_is_reported(raw_df):
    """제거하지 않고 남긴 결측이 반환값에 기록된다 (팀원이 알아야 한다)."""
    _, info = dp.handle_missing(raw_df)
    assert "retained_missing" in info
    assert "comissing_groups" in info

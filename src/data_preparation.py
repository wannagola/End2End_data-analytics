"""데이터 준비 담당 모듈 — 모든 가설의 기초 데이터를 만든다.

담당 범위
    1. Pandas · Polars 로딩 및 비교
    2. 결측치 · 중복 처리
    3. 날짜 · 시간 형식 변환
    4. 운행 시간(duration_min) 계산
    5. 평균 속도(average_speed_mph) 계산
    6. 승차 시간(pickup_hour) 생성
    7. 요일(pickup_dayofweek) · 주말 여부(is_weekend) 생성
    8. 출퇴근 시간 여부(is_rush_hour) 생성
    9. 이상치 제거
   10. 정제 데이터 저장 -> data/processed/taxi_cleaned.parquet

이 모듈의 출력이 시각화 · 통계 · ML 담당의 입력이 된다.
따라서 아래 컬럼 계약은 임의로 바꾸지 않는다. 바꿔야 하면 팀에 먼저 알린다.

    [원본 유지]  tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
                trip_distance, RatecodeID, PULocationID, DOLocationID,
                payment_type, fare_amount, tip_amount, tolls_amount, total_amount
    [파생 생성]  duration_min, average_speed_mph, pickup_hour,
                pickup_dayofweek, is_weekend, is_rush_hour

설계 원칙 5-2: 데이터가 줄어드는 모든 지점에서 처리 전후 행 수를 기록한다.
설계 원칙 5-1: 각 단계는 입력과 출력이 명확한 독립 함수다. 전역 변수를 공유하지 않는다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from src import config

logger = logging.getLogger(__name__)


# ===========================================================================
# 0. 스키마 확인
# ===========================================================================
def read_schema(path: Path | None = None) -> list[str]:
    """parquet 파일을 메모리에 올리지 않고 컬럼 목록만 읽는다.

    TLC 파케이의 컬럼 구성은 연월에 따라 달라진다. 전체 목록을 하드코딩하면
    다른 달 파일로 바꿨을 때 바로 깨지므로, 실제 스키마를 먼저 읽고 필수 컬럼만 검증한다.

    Args:
        path: parquet 경로. None 이면 config.RAW_DATA_PATH.

    Returns:
        파일에 실제로 존재하는 컬럼 이름 목록.

    Raises:
        ValueError: 필수 컬럼이 하나라도 없을 때.
    """
    path = path or config.RAW_DATA_PATH
    available = list(pq.ParquetFile(path).schema_arrow.names)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in available]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없다: {missing}\n"
            f"파일에 있는 컬럼: {available}\n"
            "다른 스키마의 파일이거나 손상된 파일일 수 있다."
        )

    extras = [c for c in available if c not in config.REQUIRED_COLUMNS]
    logger.info("스키마 확인: 전체 %d컬럼 (필수 %d개 확보, 그 외 %d개)",
                len(available), len(config.REQUIRED_COLUMNS), len(extras))
    return available


def columns_to_read(path: Path | None = None) -> list[str]:
    """실제로 읽어들일 컬럼을 정한다 (필수 + 파일에 존재하는 선택 컬럼).

    전체 컬럼이 아니라 분석에 쓰는 것만 읽는다.
    400만 행 규모에서는 이 차이가 수백 MB 의 메모리로 나타난다.
    """
    available = set(read_schema(path))
    optional_present = [c for c in config.OPTIONAL_COLUMNS if c in available]
    return config.REQUIRED_COLUMNS + optional_present


# ===========================================================================
# 1. Pandas · Polars 로딩 및 비교
# ===========================================================================
def load_pandas(path: Path | None = None,
                columns: list[str] | None = None) -> tuple[pd.DataFrame, float]:
    """Pandas 로 parquet 을 읽는다 (컬럼 프로젝션 적용).

    Args:
        path:    parquet 경로. None 이면 config.RAW_DATA_PATH.
        columns: 읽을 컬럼. None 이면 columns_to_read() 로 결정.

    Returns:
        (데이터프레임, 소요시간 초).
    """
    path = path or config.RAW_DATA_PATH
    columns = columns or columns_to_read(path)

    started = time.perf_counter()
    df = pd.read_parquet(path, columns=columns)
    elapsed = time.perf_counter() - started

    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("Pandas 로딩: shape=%s, %.3f초, 메모리 %.1fMB", df.shape, elapsed, mem_mb)
    return df, elapsed


def load_polars(path: Path | None = None,
                columns: list[str] | None = None) -> tuple[pl.DataFrame, float]:
    """Polars eager 모드로 parquet 을 읽는다.

    Args:
        path:    parquet 경로.
        columns: 읽을 컬럼.

    Returns:
        (데이터프레임, 소요시간 초).
    """
    path = path or config.RAW_DATA_PATH
    columns = columns or columns_to_read(path)

    started = time.perf_counter()
    df = pl.read_parquet(path, columns=columns)
    elapsed = time.perf_counter() - started

    mem_mb = df.estimated_size() / 1024 ** 2
    logger.info("Polars(eager) 로딩: shape=%s, %.3f초, 메모리 %.1fMB", df.shape, elapsed, mem_mb)
    return df, elapsed


def load_polars_lazy(path: Path | None = None,
                     columns: list[str] | None = None) -> tuple[pl.DataFrame, float]:
    """Polars lazy 모드로 읽는다 (projection pushdown).

    `scan_parquet` 은 즉시 읽지 않고 실행 계획만 세운다. `select` 로 컬럼을 좁힌 뒤
    `collect()` 할 때 필요한 컬럼 청크만 디스크에서 가져온다.
    Pandas 대비 Polars 의 이점은 이 지점에서 가장 크게 드러난다.

    Args:
        path:    parquet 경로.
        columns: 읽을 컬럼.

    Returns:
        (데이터프레임, 소요시간 초).
    """
    path = path or config.RAW_DATA_PATH
    columns = columns or columns_to_read(path)

    started = time.perf_counter()
    df = pl.scan_parquet(path).select(columns).collect()
    elapsed = time.perf_counter() - started

    logger.info("Polars(lazy) 로딩: shape=%s, %.3f초", df.shape, elapsed)
    return df, elapsed


def _warm_up(path: Path, columns: list[str]) -> None:
    """속도 측정 전에 두 엔진을 각각 한 번씩 호출해 초기화 비용을 걷어낸다.

    1행만 읽으므로 비용은 거의 없다. 실패해도 측정 자체를 막을 이유는 없으므로
    경고만 남기고 넘어간다.
    """
    try:
        pd.read_parquet(path, columns=columns[:1]).head(1)
        pl.read_parquet(path, columns=columns[:1]).head(1)
        pl.scan_parquet(path).select(columns[:1]).head(1).collect()
    except Exception as exc:   # noqa: BLE001 — 워밍업 실패가 본 측정을 막아선 안 된다
        logger.warning("워밍업 실패 (%s): %s — 측정은 그대로 진행한다", type(exc).__name__, exc)


def compare_load(path: Path | None = None) -> tuple[pd.DataFrame, pl.DataFrame, dict[str, Any]]:
    """세 방식으로 읽어 shape · 결측 · 속도 · 메모리를 대조한다.

    두 엔진으로 읽는 것은 속도 비교만이 목적이 아니다. 서로 다른 엔진이 같은 결과를
    내는지 확인하는 것 자체가 검증이다(설계 원칙 5-4).

    Args:
        path: parquet 경로.

    Returns:
        (pandas DF, polars DF, 비교 정보 dict).

    Raises:
        ValueError: 엔진 간 shape 이 다를 때. 이후 분석이 무의미해지므로 즉시 멈춘다.
    """
    path = path or config.RAW_DATA_PATH
    columns = columns_to_read(path)

    # 워밍업 — 측정 전에 두 엔진을 각각 한 번씩 건드린다.
    # Polars 는 첫 호출에 스레드풀·메모리 할당자를 초기화하는데, 이 비용이 그대로
    # 측정에 섞이면 "Polars 가 Pandas 보다 40배 느리다" 같은 잘못된 결론이 나온다.
    # (작은 파일일수록 초기화 비용의 비중이 커서 왜곡이 심하다.)
    _warm_up(path, columns)

    df_pd, t_pd = load_pandas(path, columns)
    df_pl, t_pl = load_polars(path, columns)
    df_lazy, t_lazy = load_polars_lazy(path, columns)

    shapes = {"pandas": df_pd.shape, "polars_eager": df_pl.shape, "polars_lazy": df_lazy.shape}
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"엔진별 shape 이 다르다: {shapes}. "
            "파싱 옵션이 어긋난 것이므로 이후 단계를 진행하지 않는다."
        )

    na_pd = {k: int(v) for k, v in df_pd.isna().sum().items() if v > 0}
    na_pl = {k: int(v) for k, v in df_pl.null_count().to_dicts()[0].items() if v > 0}
    na_match = na_pd == na_pl
    if not na_match:
        # shape 이 같아도 결측 인식이 다르면 결과가 갈린다. 멈추지는 않되 반드시 남긴다.
        logger.warning("두 엔진의 결측 집계가 불일치한다: pandas=%s / polars=%s", na_pd, na_pl)

    mem_pd = float(df_pd.memory_usage(deep=True).sum() / 1024 ** 2)
    mem_pl = float(df_pl.estimated_size() / 1024 ** 2)

    logger.info("로딩 비교 | shape 일치: True | 결측 일치: %s", na_match)
    logger.info("로딩 비교 | 속도 pandas %.3f / polars %.3f / lazy %.3f (초)", t_pd, t_pl, t_lazy)
    logger.info("로딩 비교 | 메모리 pandas %.1fMB vs polars %.1fMB", mem_pd, mem_pl)

    info: dict[str, Any] = {
        "source_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "columns_read": columns,
        "n_columns_read": len(columns),
        "shape": df_pd.shape,
        "shape_match": True,
        "pandas_seconds": round(t_pd, 4),
        "polars_eager_seconds": round(t_pl, 4),
        "polars_lazy_seconds": round(t_lazy, 4),
        "eager_speedup": round(t_pd / t_pl, 2) if t_pl > 0 else float("nan"),
        "lazy_speedup": round(t_pd / t_lazy, 2) if t_lazy > 0 else float("nan"),
        "pandas_memory_mb": round(mem_pd, 1),
        "polars_memory_mb": round(mem_pl, 1),
        "memory_ratio": round(mem_pd / mem_pl, 2) if mem_pl > 0 else float("nan"),
        "pandas_missing": na_pd,
        "polars_missing": na_pl,
        "missing_match": na_match,
        "pandas_version": pd.__version__,
        "polars_version": pl.__version__,
    }
    return df_pd, df_pl, info


# ===========================================================================
# 2. 결측치 처리
# ===========================================================================
def detect_comissing(df: pd.DataFrame) -> dict[str, list[str]]:
    """결측 개수가 완전히 같은 컬럼 묶음을 찾아낸다.

    여러 컬럼의 결측 개수가 정확히 일치하면 우연이 아니다.
    보통은 **특정 사업자(vendor)가 해당 필드들을 통째로 비워 보고**한 경우다.
    이런 결측은 무작위(MCAR)가 아니므로, 그 행을 지우면 특정 사업자의 데이터만
    통째로 사라져 표본이 편향된다.

    이 함수는 판단을 대신하지 않는다. "이 컬럼들이 함께 비어 있다"는 사실만
    드러내서, 결측 처리 방식을 정할 때 근거로 쓰게 한다.

    Args:
        df: 로딩 직후 데이터프레임.

    Returns:
        {결측 개수: [컬럼, ...]} — 2개 이상 묶인 것만 담는다.
    """
    counts = {c: int(n) for c, n in df.isna().sum().items() if n > 0}

    groups: dict[int, list[str]] = {}
    for col, n in counts.items():
        groups.setdefault(n, []).append(col)

    suspicious = {n: cols for n, cols in groups.items() if len(cols) >= 2}
    for n, cols in suspicious.items():
        logger.warning(
            "결측이 동반 발생한다: %s — 각 %s개(%.2f%%)로 정확히 일치. "
            "특정 사업자가 이 필드들을 비워 보고했을 가능성이 높다(무작위 결측 아님).",
            ", ".join(cols), f"{n:,}", n / len(df) * 100 if len(df) else 0.0,
        )
    return {str(n): cols for n, cols in suspicious.items()}


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """분석에 필수인 컬럼에 결측이 있는 행을 제거한다.

    승차/하차 시각이나 거리가 비어 있으면 평균 속도를 계산할 수 없다.
    대치(imputation)로 채우면 존재하지 않는 운행을 만들어내는 셈이라, 여기서는 제거한다.

    반대로 `passenger_count` 처럼 **분석에 쓰이지 않는 컬럼의 결측 때문에 행을 버리지는
    않는다.** 실제 데이터에서 이 컬럼의 결측이 특정 사업자에 몰려 있어(detect_comissing 참조)
    제거하면 전체의 23%가 편향된 방식으로 사라졌다.

    Args:
        df: 로딩 직후 데이터프레임.

    Returns:
        (결측이 제거된 데이터프레임, 처리 정보 dict).

    Raises:
        ValueError: 제거 후 남은 행이 없을 때.
    """
    before = len(df)
    missing_all = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    comissing = detect_comissing(df)     # 동반 결측 진단 — 처리 방식의 근거가 된다

    critical = [c for c in config.CRITICAL_COLUMNS if c in df.columns]
    missing_critical = {c: int(df[c].isna().sum()) for c in critical if df[c].isna().any()}

    cleaned = df.dropna(subset=critical).reset_index(drop=True)
    after = len(cleaned)

    logger.info("결측 제거: %s행 → %s행 (%s행 제거, %.3f%%)",
                f"{before:,}", f"{after:,}", f"{before - after:,}",
                (before - after) / before * 100 if before else 0.0)
    for col, n in missing_critical.items():
        logger.info("  필수 컬럼 결측 %-22s %s개 (%.3f%%)", col, f"{n:,}", n / before * 100)
    for col, n in missing_all.items():
        if col not in critical:
            logger.info("  (유지) 비필수 결측 %-18s %s개", col, f"{n:,}")

    if after == 0:
        raise ValueError("결측 제거 후 남은 행이 없다. 필수 컬럼 목록을 확인하라.")

    return cleaned, {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "removed_ratio": round((before - after) / before, 5) if before else 0.0,
        "missing_all": missing_all,
        "missing_critical": missing_critical,
        "critical_columns": critical,
        "comissing_groups": comissing,
        # 제거하지 않고 남긴 결측 — ②③④ 가 이 컬럼을 쓸 때 알고 있어야 한다
        "retained_missing": {c: int(cleaned[c].isna().sum())
                             for c in cleaned.columns if cleaned[c].isna().any()},
        "strategy": ("필수 컬럼 결측 행만 제거 (listwise deletion). "
                     "분석에 쓰이지 않는 컬럼의 결측은 행을 살린 채 유지한다."),
    }


# ===========================================================================
# 3~8. 날짜·시간 변환과 파생변수 생성
# ===========================================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜·시간을 변환하고 가설 검증에 필요한 파생변수를 만든다.

    원본에는 소요시간도 속도도 없다. 승차·하차 시각과 거리에서 직접 계산한다.

    생성 컬럼:
        duration_min       운행 시간(분) = 하차 - 승차
        average_speed_mph  평균 속도(마일/시) = trip_distance / (duration_min / 60)
        pickup_hour        승차 시각 (0~23)
        pickup_dayofweek   승차 요일 (0=월 ~ 6=일)
        is_weekend         주말 여부 (토·일)
        is_rush_hour       출퇴근 시간 여부 (config.RUSH_HOURS)

    Args:
        df: 결측이 제거된 데이터프레임.

    Returns:
        파생변수가 추가된 새 데이터프레임 (원본은 건드리지 않는다).
    """
    out = df.copy()

    # (3) 날짜·시간 형식 변환 — parquet 은 보통 이미 datetime 이지만 명시적으로 보장한다
    out["tpep_pickup_datetime"] = pd.to_datetime(out["tpep_pickup_datetime"])
    out["tpep_dropoff_datetime"] = pd.to_datetime(out["tpep_dropoff_datetime"])

    # (4) 운행 시간
    out["duration_min"] = (
        out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # (5) 평균 속도 — 0으로 나누는 것을 막는다. 0분 운행은 뒤의 이상치 필터가 제거한다.
    out["average_speed_mph"] = np.where(
        out["duration_min"] > 0,
        out["trip_distance"] / (out["duration_min"] / 60.0),
        np.nan,
    )

    # (6) 승차 시간
    out["pickup_hour"] = out["tpep_pickup_datetime"].dt.hour.astype("int16")

    # (7) 요일과 주말 여부
    out["pickup_dayofweek"] = out["tpep_pickup_datetime"].dt.dayofweek.astype("int16")
    out["is_weekend"] = out["pickup_dayofweek"].isin(config.WEEKEND_DAYS)

    # (8) 출퇴근 시간 여부
    #
    # 주의 — 이 정의는 요일을 보지 않는다. 즉 "토요일 오전 8시"도 is_rush_hour=True 가 된다.
    # 팀에서 합의한 정의를 그대로 구현했지만, 가설 2(평일·주말 패턴 비교)를 검증할 때는
    # 이 점이 교란 요인이 될 수 있다. 통계 담당은 필요하면
    #     df[~df["is_weekend"]]  로 평일만 남긴 뒤 비교
    # 하는 방식으로 시간대 효과와 요일 효과를 분리할 수 있다.
    out["is_rush_hour"] = out["pickup_hour"].isin(config.RUSH_HOURS)

    logger.info("파생변수 6개 생성: duration_min, average_speed_mph, pickup_hour, "
                "pickup_dayofweek, is_weekend, is_rush_hour")
    logger.info("  출퇴근 시간 정의: %s (요일 무관)", config.RUSH_HOURS)
    return out


def filter_target_month(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """대상 월 바깥의 레코드를 제거한다.

    TLC 원본에는 2008년이나 2090년 같은 승차 시각이 소수 섞여 있다(미터기 시계 오류).
    대상 월을 하드코딩하지 않고 **데이터에서 가장 흔한 연-월**을 기준으로 삼는다.
    그래야 다른 달 파일로 바꿔도 코드를 고칠 필요가 없다.

    Args:
        df: 파생변수가 추가된 데이터프레임.

    Returns:
        (필터된 데이터프레임, 처리 정보 dict).
    """
    before = len(df)
    period = df["tpep_pickup_datetime"].dt.to_period("M")
    target = period.mode().iloc[0]

    kept = df[period == target].reset_index(drop=True)
    after = len(kept)

    strays = sorted({str(p) for p in period[period != target].unique()})
    logger.info("대상 월 필터(%s): %s행 → %s행 (%s행 제거)",
                target, f"{before:,}", f"{after:,}", f"{before - after:,}")
    if strays:
        logger.info("  제거된 이상 연월: %s", ", ".join(strays[:10]) + (" ..." if len(strays) > 10 else ""))

    return kept, {
        "target_period": str(target),
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "stray_periods": strays[:20],
    }


# ===========================================================================
# 9. 이상치 제거
# ===========================================================================
def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """물리적·상업적으로 불가능한 레코드를 제거한다.

    각 조건에 임계값의 근거를 붙였다. 임계값을 config 에 모아둔 이유는
    "왜 100마일인가"라는 질문이 나왔을 때 한 곳만 보면 되게 하기 위해서다.

    속도 관련 필터를 넣은 이유가 중요하다. 이 프로젝트의 세 가설이 전부
    average_speed_mph 를 본다. 시속 300마일짜리 기록이 몇 건만 남아 있어도
    평균과 t-검정 결과가 통째로 흔들린다.

    Args:
        df: 파생변수가 추가된 데이터프레임.

    Returns:
        (필터된 데이터프레임, 단계별 제거 이력 dict).

    Raises:
        ValueError: 어떤 필터가 데이터를 전부 제거했을 때(임계값 오류 신호).
    """
    steps: list[dict[str, Any]] = []
    work = df
    rows_at_start = len(df)

    conditions: list[tuple[str, str]] = [
        ("요금 범위",
         f"기본요금 ${config.MIN_FARE} 미만 또는 ${config.MAX_FARE} 초과는 기록 오류"),
        ("거리 범위",
         f"{config.MIN_DISTANCE}마일 미만(미터기 오작동) 또는 {config.MAX_DISTANCE}마일 초과(GPS 오류)"),
        ("소요시간 범위",
         f"{config.MIN_DURATION_MIN}분 미만(승차 취소) 또는 {config.MAX_DURATION_MIN}분 초과(미터기 미종료)"),
        ("평균 속도 범위",
         f"시속 {config.MIN_SPEED_MPH}마일 미만(사실상 정지) 또는 "
         f"{config.MAX_SPEED_MPH}마일 초과(뉴욕 시내에서 불가능)"),
        ("승객 수 범위",
         f"{config.MIN_PASSENGERS}~{config.MAX_PASSENGERS}명 범위를 벗어남 "
         f"(결측은 통과시킨다 — 특정 사업자가 이 필드를 비워 보고하므로)"),
        ("합계 금액",
         "총액이 0 이하인 운행은 환불/취소 기록"),
    ]

    for (name, reason) in conditions:
        before = len(work)
        if name == "요금 범위":
            mask = work["fare_amount"].between(config.MIN_FARE, config.MAX_FARE)
        elif name == "거리 범위":
            mask = work["trip_distance"].between(config.MIN_DISTANCE, config.MAX_DISTANCE)
        elif name == "소요시간 범위":
            mask = work["duration_min"].between(config.MIN_DURATION_MIN, config.MAX_DURATION_MIN)
        elif name == "평균 속도 범위":
            mask = work["average_speed_mph"].between(config.MIN_SPEED_MPH, config.MAX_SPEED_MPH)
        elif name == "승객 수 범위":
            # 결측이면 통과, 값이 있으면 범위를 확인한다.
            # between() 은 NaN 에 False 를 돌려주므로 isna() 를 OR 로 붙여야
            # "값이 없는 것"과 "값이 범위를 벗어난 것"을 구분할 수 있다.
            mask = (work["passenger_count"].isna()
                    | work["passenger_count"].between(config.MIN_PASSENGERS,
                                                      config.MAX_PASSENGERS))
        else:
            mask = work["total_amount"].gt(0)

        # 결측은 비교 연산에서 False 가 되므로 자동으로 제거된다.
        work = work[mask.fillna(False)]
        after = len(work)
        removed = before - after

        logger.info("  이상치 [%-12s] %s행 → %s행 (%s행 제거, %.3f%%) — %s",
                    name, f"{before:,}", f"{after:,}", f"{removed:,}",
                    removed / before * 100 if before else 0.0, reason)
        steps.append({"name": name, "rows_before": before, "rows_after": after,
                      "rows_removed": removed, "reason": reason})

        # 필터 하나가 데이터를 통째로 날리면 임계값이 잘못된 것이다. 조용히 넘어가지 않는다.
        if after == 0:
            raise ValueError(f"이상치 필터 '{name}' 이후 남은 행이 없다. 임계값을 확인하라.")

    work = work.reset_index(drop=True)
    logger.info("이상치 제거 합계: %s행 → %s행 (%s행 제거, %.3f%%)",
                f"{rows_at_start:,}", f"{len(work):,}", f"{rows_at_start - len(work):,}",
                (rows_at_start - len(work)) / rows_at_start * 100 if rows_at_start else 0.0)

    return work, {
        "steps": steps,
        "rows_before": rows_at_start,
        "rows_after": len(work),
        "rows_removed": rows_at_start - len(work),
    }


# ===========================================================================
# 중복 처리
# ===========================================================================
def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """완전 중복 행을 제거한다.

    운행 고유 ID 가 없으므로, 원본 컬럼이 전부 같으면 같은 운행이 두 번 적재된 것으로 본다.
    파생변수는 원본에서 계산된 값이라 판정 기준에서 제외한다 — 판정은 원본에만 근거해야 한다.

    Args:
        df: 이상치가 제거된 데이터프레임.

    Returns:
        (중복이 제거된 데이터프레임, 처리 정보 dict).
    """
    original_cols = [c for c in df.columns
                     if c in config.REQUIRED_COLUMNS + config.OPTIONAL_COLUMNS]
    before = len(df)
    n_dup = int(df.duplicated(subset=original_cols).sum())
    out = df.drop_duplicates(subset=original_cols).reset_index(drop=True)
    after = len(out)

    logger.info("중복 제거: %s행 → %s행 (%s행 제거, 판정 기준 원본 %d컬럼)",
                f"{before:,}", f"{after:,}", f"{before - after:,}", len(original_cols))
    return out, {
        "rows_before": before,
        "rows_after": after,
        "duplicates_removed": n_dup,
        "subset_columns": original_cols,
    }


# ===========================================================================
# EDA
# ===========================================================================
def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """기본 EDA 를 수행하고 결과를 dict 로 반환한다.

    화면에 찍고 끝내지 않고 값을 돌려주는 이유는, 이 값이 그대로 report.md 로
    흘러가야 하기 때문이다. 손으로 옮겨 적을 여지를 없앤다(설계 원칙 5-3).

    Args:
        df: 정제가 끝난 데이터프레임.

    Returns:
        shape · dtypes · 기술통계 · 시간대별 요약 등을 담은 dict.
    """
    numeric_cols = [c for c in ["trip_distance", "duration_min", "average_speed_mph",
                                "fare_amount", "tip_amount", "total_amount", "passenger_count"]
                    if c in df.columns]

    # 가설 1·2 담당자가 바로 쓸 수 있도록 시간대·요일별 요약을 미리 만들어 둔다
    hourly = (df.groupby("pickup_hour", observed=True)
                .agg(n=("average_speed_mph", "size"),
                     mean_speed=("average_speed_mph", "mean"),
                     median_speed=("average_speed_mph", "median"))
                .round(3))
    by_weekend = (df.groupby("is_weekend", observed=True)
                    .agg(n=("average_speed_mph", "size"),
                         mean_speed=("average_speed_mph", "mean"))
                    .round(3))
    by_rush = (df.groupby("is_rush_hour", observed=True)
                 .agg(n=("average_speed_mph", "size"),
                      mean_speed=("average_speed_mph", "mean"))
                 .round(3))

    slowest_hour = int(hourly["mean_speed"].idxmin())
    fastest_hour = int(hourly["mean_speed"].idxmax())
    speed_median = float(df["average_speed_mph"].median())

    logger.info("EDA | 정제 후 shape=%s", df.shape)
    logger.info("EDA | 평균 속도 최저 %d시 (%.2f mph) / 최고 %d시 (%.2f mph)",
                slowest_hour, hourly.loc[slowest_hour, "mean_speed"],
                fastest_hour, hourly.loc[fastest_hour, "mean_speed"])
    logger.info("EDA | 속도 중앙값 %.3f mph  ← ML 담당의 slow_trip 임계값 후보", speed_median)
    for flag, row in by_rush.iterrows():
        logger.info("EDA | 출퇴근시간=%-5s  n=%9s  평균속도 %.3f mph",
                    flag, f"{int(row['n']):,}", row["mean_speed"])
    for flag, row in by_weekend.iterrows():
        logger.info("EDA | 주말=%-5s        n=%9s  평균속도 %.3f mph",
                    flag, f"{int(row['n']):,}", row["mean_speed"])
    # 잔여 결측을 "의도적으로 남긴 것"과 "예상 못 한 것"으로 나눠 보여준다.
    # 뭉뚱그려 찍으면 정책에 따라 남긴 결측인지 버그인지 구분이 안 된다.
    retained = {c: int(df[c].isna().sum())
                for c in df.columns if c in config.NULLABLE_COLUMNS and df[c].isna().any()}
    unexpected = {c: int(df[c].isna().sum())
                  for c in numeric_cols
                  if c not in config.NULLABLE_COLUMNS and df[c].isna().any()}

    logger.info("EDA | 잔여 중복 %d행", int(df.duplicated().sum()))
    if retained:
        logger.info("EDA | 정책상 유지한 결측: %s (분석에 쓰이지 않는 컬럼)",
                    ", ".join(f"{c} {n:,}개" for c, n in retained.items()))
    if unexpected:
        logger.warning("EDA | 예상하지 못한 결측: %s — 확인이 필요하다",
                       ", ".join(f"{c} {n:,}개" for c, n in unexpected.items()))
    else:
        logger.info("EDA | 분석용 수치 컬럼에 예상 밖 결측 없음")

    return {
        "shape": df.shape,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "describe": df[numeric_cols].describe().T.round(3),
        "numeric_cols": numeric_cols,
        "remaining_missing": int(df[numeric_cols].isna().sum().sum()),
        "retained_missing": retained,
        "unexpected_missing": unexpected,
        "remaining_duplicates": int(df.duplicated().sum()),
        "hourly_speed": hourly,
        "speed_by_weekend": by_weekend,
        "speed_by_rush": by_rush,
        "slowest_hour": slowest_hour,
        "slowest_speed": round(float(hourly.loc[slowest_hour, "mean_speed"]), 3),
        "fastest_hour": fastest_hour,
        "fastest_speed": round(float(hourly.loc[fastest_hour, "mean_speed"]), 3),
        "speed_median": round(speed_median, 3),
        "rush_share": round(float(df["is_rush_hour"].mean()), 4),
        "weekend_share": round(float(df["is_weekend"].mean()), 4),
    }


# ===========================================================================
# 10. 정제 데이터 저장
# ===========================================================================
def save_cleaned(df: pd.DataFrame, path: Path | None = None) -> dict[str, Any]:
    """정제 데이터를 parquet 으로 저장하고, 다시 읽어 무결성을 확인한다.

    설계 원칙 5-2: "저장했다"를 믿지 않는다. 파일 존재와 크기를 확인하고,
    한 걸음 더 나아가 **다시 읽어 shape 과 컬럼이 일치하는지**까지 검증한다.
    이 파일이 팀원 세 명의 입력이므로, 여기서 조용히 깨지면 세 명이 함께 막힌다.

    Args:
        df:   저장할 정제 데이터프레임.
        path: 저장 경로. None 이면 config.CLEANED_PATH.

    Returns:
        저장 경로 · 크기 · 검증 결과를 담은 dict.

    Raises:
        ValueError: 다시 읽은 결과가 원본과 다를 때.
    """
    path = path or config.CLEANED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False, compression="snappy")
    size = config.verify_file(path, min_bytes=1_000)

    # 재로딩 검증
    reloaded = pd.read_parquet(path)
    if reloaded.shape != df.shape:
        raise ValueError(f"저장 후 shape 이 다르다: 원본 {df.shape} vs 재로딩 {reloaded.shape}")
    if list(reloaded.columns) != list(df.columns):
        raise ValueError("저장 후 컬럼 구성이 다르다.")

    # report.md 는 프로젝트 루트에 놓이므로 가능하면 루트 기준 상대경로로 적는다.
    try:
        display_path = str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:      # tmp 디렉터리 등 루트 바깥이면 절대경로를 그대로 쓴다
        display_path = str(path)

    logger.info("정제 데이터 저장: %s (%s bytes, shape=%s)",
                display_path, f"{size:,}", reloaded.shape)
    logger.info("  재로딩 검증 통과 — 시각화·통계·ML 담당은 이 파일을 입력으로 쓴다")

    return {
        "path": display_path,
        "bytes": size,
        "shape": reloaded.shape,
        "columns": list(reloaded.columns),
        "reload_verified": True,
    }


# ===========================================================================
# 파이프라인
# ===========================================================================
def run(path: Path | None = None,
        save: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """데이터 준비 단계 전체를 순서대로 실행한다.

        로딩 비교 → 결측 처리 → 파생변수 → 대상 월 필터
        → 이상치 제거 → 중복 제거 → EDA → 저장

    Args:
        path: 원본 parquet 경로. None 이면 config.RAW_DATA_PATH.
        save: 정제 결과를 파일로 저장할지 여부 (테스트에서는 False).

    Returns:
        (정제된 데이터프레임, 단계별 정보 dict).

    Raises:
        ValueError: 정제 후에도 중복이 남아 있거나 필수 컬럼이 없을 때(자기 검증).
    """
    logger.info("=== 데이터 준비 단계 시작 ===")
    started = time.perf_counter()

    df_pd, _df_pl, load_info = compare_load(path)
    original_rows = len(df_pd)

    df, missing_info = handle_missing(df_pd)
    df = add_time_features(df)
    df, month_info = filter_target_month(df)
    df, outlier_info = remove_outliers(df)
    df, dup_info = drop_duplicates(df)
    eda = run_eda(df)

    # --- 자기 검증 (설계 원칙 5-4) -----------------------------------------
    if eda["remaining_duplicates"] != 0:
        raise ValueError(f"정제 후에도 중복이 {eda['remaining_duplicates']}행 남아 있다.")
    contract = ["duration_min", "average_speed_mph", "pickup_hour",
                "pickup_dayofweek", "is_weekend", "is_rush_hour"]
    absent = [c for c in contract if c not in df.columns]
    if absent:
        raise ValueError(f"팀에 약속한 파생 컬럼이 없다: {absent}")
    if df["average_speed_mph"].isna().any():
        raise ValueError("average_speed_mph 에 결측이 남아 있다.")
    if eda["unexpected_missing"]:
        raise ValueError(f"정책에 없는 결측이 남아 있다: {eda['unexpected_missing']}")

    save_info = save_cleaned(df) if save else None
    elapsed = time.perf_counter() - started

    logger.info("=== 데이터 준비 완료 | %s행 → %s행 (%.2f%% 제거) | %.2f초 ===",
                f"{original_rows:,}", f"{len(df):,}",
                (original_rows - len(df)) / original_rows * 100 if original_rows else 0.0,
                elapsed)

    info: dict[str, Any] = {
        "original_rows": original_rows,
        "final_rows": int(len(df)),
        "removed_total": original_rows - int(len(df)),
        "removed_ratio": round((original_rows - len(df)) / original_rows, 5) if original_rows else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "load": load_info,
        "missing": missing_info,
        "month": month_info,
        "outliers": outlier_info,
        "duplicates": dup_info,
        "eda": eda,
        "save": save_info,
        "derived_columns": contract,
    }
    return df, info


if __name__ == "__main__":
    # 모듈 단독 실행용. import 만으로는 아무 일도 일어나지 않는다 (설계 원칙 5-5).
    from src import download

    config.ensure_dirs()
    config.setup_logging()
    run(download.ensure_data())

"""테스트용 합성 운행 데이터 생성기.

**이 파일이 만드는 데이터는 실제 TLC 기록이 아니다.** 테스트 전용 픽스처다.

왜 실제 데이터를 커밋하지 않는가 (설계 원칙 5-6):
- 원본은 66MB 다. GitHub 100MB 제한에 걸리지는 않지만, 한 번 커밋하면 히스토리에서
  지울 수 없고 clone 이 무거워진다.
- 테스트는 400만 행이 필요 없다. 수천 행이면 모든 분기를 지나간다.

합성 데이터는 실제 데이터의 **분석에 중요한 성질**을 의도적으로 재현한다:
1. 평일 출퇴근 시간대(RUSH_HOURS)의 평균 속도가 낮음 — 가설 1
2. 주말은 출퇴근 정체가 없어 시간대별 패턴이 다름 — 가설 2
3. 요금 음수 · 거리 0 · 6시간 초과 · 중복 행 등 정제가 걸러야 할 오염 레코드
4. 특정 사업자가 네 필드를 함께 비워 보고한 동반 결측 패턴
5. 대상 월(2026-05) 바깥으로 새어 들어온 소수의 레코드 — TLC 원본에도 실제로 존재한다
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# 일반 존 ID 풀 (실제 taxi_zone_lookup 은 1~265 범위)
_AIRPORT_ZONE_IDS = [1, 132, 138]   # Newark / JFK / LaGuardia
_COMMON_ZONES = [4, 13, 24, 41, 43, 48, 68, 79, 90, 100, 107, 113, 141, 142, 161,
                 162, 163, 164, 170, 186, 209, 230, 231, 236, 237, 238, 239, 246, 262, 263]


def make_synthetic(
    n_rows: int = 8_000,
    seed: int = config.RANDOM_STATE,
    dest: Path | None = None,
) -> Path:
    """합성 운행 데이터를 만들어 parquet 으로 저장한다.

    Args:
        n_rows: 생성할 정상 레코드 수 (오염 레코드가 추가로 붙는다).
        seed:   난수 시드.
        dest:   저장 경로. None 이면 config.SAMPLE_DATA_PATH.

    Returns:
        저장된 parquet 경로.
    """
    rng = np.random.default_rng(seed)
    dest = dest or config.SAMPLE_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    # --- 승차 시각: 2026년 5월 안에 고르게 뿌린다 -----------------------------
    base = pd.Timestamp("2026-05-01")
    offsets = rng.uniform(0, 31 * 24 * 3600, n_rows)
    pickup = base + pd.to_timedelta(offsets, unit="s")
    hour = pickup.hour.to_numpy()
    weekday = pickup.dayofweek.to_numpy()

    # --- 공항 운행 여부 (약 12%) ---------------------------------------------
    is_airport = rng.random(n_rows) < 0.12
    pu = rng.choice(_COMMON_ZONES, n_rows)
    do = rng.choice(_COMMON_ZONES, n_rows)
    airport_ids = np.array(list(_AIRPORT_ZONE_IDS))
    picked_airport = rng.choice(airport_ids, n_rows)
    # 공항 운행은 승차나 하차 중 한쪽이 공항 존이 되도록 만든다
    to_pu = rng.random(n_rows) < 0.5
    pu = np.where(is_airport & to_pu, picked_airport, pu)
    do = np.where(is_airport & ~to_pu, picked_airport, do)

    ratecode = np.ones(n_rows)
    ratecode = np.where(is_airport & (rng.random(n_rows) < 0.6), 2.0, ratecode)

    # --- 거리와 소요시간 ------------------------------------------------------
    distance = np.where(is_airport,
                        rng.lognormal(2.5, 0.35, n_rows),      # 공항은 장거리
                        rng.lognormal(0.85, 0.65, n_rows))     # 시내는 단거리
    distance = np.clip(distance, 0.2, 60).round(2)

    # 러시아워에는 같은 거리를 가는 데 더 오래 걸린다 (혼잡 계수)
    rush = (weekday < 5) & np.isin(hour, config.RUSH_HOURS)
    speed = np.where(rush, rng.normal(8.5, 2.0, n_rows), rng.normal(14.0, 3.5, n_rows))
    speed = np.clip(speed, 3.0, 45.0)
    duration_min = (distance / speed * 60 + rng.normal(2.5, 1.2, n_rows)).clip(1.2, 300)
    dropoff = pickup + pd.to_timedelta(duration_min * 60, unit="s")

    # --- 요금 ----------------------------------------------------------------
    fare = 3.0 + 3.4 * distance + 0.75 * duration_min + rng.normal(0, 1.5, n_rows)
    fare = np.where(ratecode == 2, 70.0 + rng.normal(0, 3, n_rows), fare)   # JFK 정액
    fare = np.clip(fare, 3.0, 400).round(2)
    tolls = np.where(is_airport & (rng.random(n_rows) < 0.5),
                     rng.choice([6.94, 11.75], n_rows), 0.0).round(2)

    # --- 결제수단 -------------------------------------------------------------
    payment = rng.choice([1, 2, 3, 4], n_rows, p=[0.70, 0.26, 0.025, 0.015])

    # --- 팁: 이 프로젝트의 핵심 --------------------------------------------
    # 카드 결제만 팁이 미터기에 기록된다. 현금 팁은 기사에게 직접 건네져 0으로 남는다.
    tip_rate = rng.normal(0.20, 0.07, n_rows)
    tip_rate += np.where(is_airport, 0.035, 0.0)        # 공항 승객이 조금 더 준다
    tip_rate += np.where(rush, -0.012, 0.0)             # 막히면 조금 덜 준다
    tip_rate = np.clip(tip_rate, 0.0, 0.6)
    tip = np.where(payment == 1, (fare * tip_rate).round(2), 0.0)

    total = (fare + tolls + tip + 1.0).round(2)         # +1.0 = 각종 부가요금 근사

    df = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n_rows).astype("int32"),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.choice([1, 1, 1, 2, 2, 3, 4, 5], n_rows).astype("float64"),
        "trip_distance": distance,
        "RatecodeID": ratecode,
        "store_and_fwd_flag": rng.choice(["N", "Y"], n_rows, p=[0.98, 0.02]),
        "PULocationID": pu.astype("int32"),
        "DOLocationID": do.astype("int32"),
        "payment_type": payment.astype("int64"),
        "fare_amount": fare,
        "extra": np.where(rush, 2.5, 0.5),
        "mta_tax": np.full(n_rows, 0.5),
        "tip_amount": tip,
        "tolls_amount": tolls,
        "improvement_surcharge": np.full(n_rows, 1.0),
        "total_amount": total,
        "congestion_surcharge": np.where(rng.random(n_rows) < 0.8, 2.5, 0.0),
        "airport_fee": np.where(is_airport, 1.75, 0.0),
    })

    # --- 결측 주입 -----------------------------------------------------------
    # 실제 데이터에서는 특정 사업자가 네 필드를 통째로 비워 보고한다.
    # 그 결과 passenger_count · RatecodeID · congestion_surcharge · airport_fee 의
    # 결측 개수가 정확히 일치한다. 무작위 결측이 아니므로 이 행들을 지우면
    # 한 사업자의 데이터만 통째로 사라져 표본이 편향된다.
    # detect_comissing() 이 이 패턴을 잡아내는지 확인하려고 동일하게 재현한다.
    null_idx = rng.choice(n_rows, size=max(1, n_rows // 40), replace=False)
    for col in ("passenger_count", "RatecodeID", "congestion_surcharge", "airport_fee"):
        df.loc[null_idx, col] = np.nan

    # --- 오염 레코드 주입 (정제 단계가 걸러내야 할 것들) ----------------------
    dirty = df.sample(n=max(8, n_rows // 100), random_state=seed).copy()
    k = len(dirty)
    dirty.iloc[0 * k // 6:1 * k // 6, dirty.columns.get_loc("fare_amount")] = -12.5   # 음수 요금
    dirty.iloc[1 * k // 6:2 * k // 6, dirty.columns.get_loc("trip_distance")] = 0.0   # 0마일
    dirty.iloc[2 * k // 6:3 * k // 6, dirty.columns.get_loc("trip_distance")] = 950.0  # GPS 오류
    dirty.iloc[3 * k // 6:4 * k // 6, dirty.columns.get_loc("passenger_count")] = 0.0  # 승객 0명
    dirty.iloc[4 * k // 6:5 * k // 6, dirty.columns.get_loc("tpep_dropoff_datetime")] = \
        dirty.iloc[4 * k // 6:5 * k // 6]["tpep_pickup_datetime"] - pd.Timedelta(minutes=5)  # 하차<승차
    dirty.iloc[5 * k // 6:, dirty.columns.get_loc("tpep_dropoff_datetime")] = \
        dirty.iloc[5 * k // 6:]["tpep_pickup_datetime"] + pd.Timedelta(hours=11)  # 미터기 미종료

    # --- 대상 월 바깥 레코드 (TLC 원본에도 실제로 섞여 있다) ------------------
    stray = df.sample(n=max(3, n_rows // 500), random_state=seed + 1).copy()
    stray["tpep_pickup_datetime"] = pd.Timestamp("2009-01-01 12:00:00")
    stray["tpep_dropoff_datetime"] = pd.Timestamp("2009-01-01 12:20:00")

    # --- 완전 중복 행 ---------------------------------------------------------
    dupes = df.head(max(5, n_rows // 400)).copy()

    out = pd.concat([df, dirty, stray, dupes], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # 순서 섞기

    out.to_parquet(dest, index=False)
    config.verify_file(dest, min_bytes=1_000)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="테스트용 합성 운행 데이터 생성")
    parser.add_argument("--rows", type=int, default=8_000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    path = make_synthetic(n_rows=args.rows, dest=args.out)
    print(f"생성 완료: {path} ({path.stat().st_size:,} bytes)")

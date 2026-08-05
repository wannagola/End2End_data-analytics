"""
NYC Yellow Taxi 통계 분석

기능
1. 정제된 택시 데이터 불러오기
2. 전체 기술통계 계산
3. 숫자형 변수 상관계수 계산
4. 출퇴근/비출퇴근 평균 속도 비교 (기술통계)
5. 출퇴근/비출퇴근 평균 속도 독립표본 t-검정(ttest_ind) 수행 및 p-value 해석
6. 가설 2: 평일/주말 기술통계 비교 및 시간대별 평균 속도 패턴 비교
7. 통계 분석 결과 저장 (CSV/JSON)
"""

from __future__ import annotations

import json

import pandas as pd
from scipy import stats

from src import config


DATA_PATH = config.CLEANED_PATH
DESCRIPTIVE_STATS_PATH = config.OUTPUT_DIR / "descriptive_statistics.csv"
CORRELATION_MATRIX_PATH = config.OUTPUT_DIR / "correlation_matrix.csv"
TTEST_RESULT_PATH = config.OUTPUT_DIR / "ttest_result.json"

ALPHA = config.ALPHA  # 유의수준 5%

# 기술통계·상관계수에 사용할 숫자형 변수
NUMERIC_COLUMNS = [
    "trip_distance", "duration_min", "average_speed_mph",
    "fare_amount", "tip_amount", "total_amount", "passenger_count",
]

# 데이터 불러오기

def load_data() -> pd.DataFrame:
    """정제된 택시 데이터 불러오기"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(f"원본 데이터 크기: {df.shape}")

    return df

# 전체 기술통계

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """숫자형 변수 전체 기술통계(개수/평균/표준편차/사분위수 등) 계산"""
    columns = [c for c in NUMERIC_COLUMNS if c in df.columns]

    desc = df[columns].describe().T.round(3)

    print("\n[전체 기술통계]")
    print(desc)

    return desc

# 숫자형 변수 상관계수

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """숫자형 변수 간 피어슨 상관계수 계산"""
    columns = [c for c in NUMERIC_COLUMNS if c in df.columns]

    corr = df[columns].corr().round(3)

    print("\n[상관계수 행렬]")
    print(corr)

    return corr

# 그룹별 평균 속도 비교

def compare_group_speed(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    """그룹(출퇴근 여부, 평일/주말 등)별 평균 속도 기술통계 비교"""
    summary = (
        df.groupby(group_col)["average_speed_mph"]
        .agg(n="count", mean="mean", std="std", median="median", min="min", max="max")
        .round(3)
    )

    print(f"\n[{label} 평균 속도 비교]")
    print(summary)

    return summary

# 가설 2: 평일/주말 시간대별 평균 속도 패턴 비교

def compare_hourly_speed_by_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간대(pickup_hour) x 평일/주말별 평균 속도 교차표

    단순 평일/주말 평균 비교로는 "주말은 출퇴근 정체가 없어 시간대별 패턴이
    다르다"는 가설 2를 확인할 수 없다 (시각화팀의 요일·시간대 히트맵과 짝을
    이루는 수치표). 평일은 출퇴근 시간대에 속도가 떨어지고 주말은 그렇지
    않은 패턴이 여기서 드러나야 한다.
    """
    hourly = (
        df.groupby(["pickup_hour", "is_weekend"])["average_speed_mph"]
        .mean()
        .unstack("is_weekend")
        .round(3)
        .rename(columns={False: "weekday_mean_speed", True: "weekend_mean_speed"})
    )

    print("\n[평일/주말 시간대별 평균 속도]")
    print(hourly)

    return hourly

# 가설 1: 출퇴근/비출퇴근 t-검정

def run_rush_hour_ttest(df: pd.DataFrame) -> dict[str, object]:
    """
    출퇴근 시간 vs 비출퇴근 시간 평균 속도 독립표본 t-검정(ttest_ind) 수행

    귀무가설(H0): 출퇴근 시간과 비출퇴근 시간의 평균 속도는 차이가 없다.
    대립가설(H1): 출퇴근 시간과 비출퇴근 시간의 평균 속도는 차이가 있다.
    """
    rush_speed = df.loc[df["is_rush_hour"], "average_speed_mph"]
    non_rush_speed = df.loc[~df["is_rush_hour"], "average_speed_mph"]

    # 두 그룹의 표본 크기·분산이 크게 다르므로 등분산을 가정하지 않는 Welch's t-test 사용
    t_stat, p_value = stats.ttest_ind(rush_speed, non_rush_speed, equal_var=False)

    is_significant = bool(p_value < ALPHA)

    print("\n[출퇴근 vs 비출퇴근 t-검정]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value    : {p_value:.6f}")
    print(
        f"판정: p-value가 유의수준({ALPHA})보다 "
        f"{'작으므로 귀무가설을 기각한다' if is_significant else '크거나 같으므로 귀무가설을 기각하지 못한다'}."
    )

    return {
        "hypothesis": "가설 1: 출퇴근 시간대는 평균 속도가 더 느리다",
        "n_rush_hour": int(rush_speed.count()),
        "n_non_rush_hour": int(non_rush_speed.count()),
        "mean_rush_hour": round(float(rush_speed.mean()), 3),
        "mean_non_rush_hour": round(float(non_rush_speed.mean()), 3),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "is_significant": is_significant,
        "interpretation": (
            f"p-value({p_value:.6f})가 유의수준({ALPHA})보다 "
            f"{'작아 출퇴근/비출퇴근 평균 속도 차이는 통계적으로 유의미하다' if is_significant else '크거나 같아 통계적으로 유의미한 차이가 없다'}."
        ),
    }

# 결과 저장

def save_descriptive_stats(
    overall: pd.DataFrame,
    rush_compare: pd.DataFrame,
    weekend_compare: pd.DataFrame,
    hourly_weekend_compare: pd.DataFrame,
) -> None:
    """전체/출퇴근/평일·주말 기술통계를 하나의 CSV 파일로 저장"""
    DESCRIPTIVE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DESCRIPTIVE_STATS_PATH.open("w", encoding="utf-8") as file:
        file.write("# 전체 기술통계\n")
        overall.to_csv(file)

        file.write("\n# 출퇴근 여부별 평균 속도 기술통계 (가설 1)\n")
        rush_compare.to_csv(file)

        file.write("\n# 평일/주말별 평균 속도 기술통계 (가설 2)\n")
        weekend_compare.to_csv(file)

        file.write("\n# 평일/주말 시간대별 평균 속도 패턴 (가설 2)\n")
        hourly_weekend_compare.to_csv(file)

    # "저장했다"를 믿지 않고 파일이 실제로 생겼는지·비어있지 않은지 확인한다.
    config.verify_file(DESCRIPTIVE_STATS_PATH, min_bytes=10)
    print(f"\n기술통계 저장 완료: {DESCRIPTIVE_STATS_PATH}")

def save_correlation_matrix(corr: pd.DataFrame) -> None:
    """상관계수 행렬 저장"""
    CORRELATION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(CORRELATION_MATRIX_PATH, encoding="utf-8")

    config.verify_file(CORRELATION_MATRIX_PATH, min_bytes=10)
    print(f"상관계수 저장 완료: {CORRELATION_MATRIX_PATH}")

def save_ttest_result(ttest_result: dict[str, object]) -> None:
    """t-검정 결과 저장"""
    TTEST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TTEST_RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(ttest_result, file, ensure_ascii=False, indent=2)

    config.verify_file(TTEST_RESULT_PATH, min_bytes=10)
    print(f"t-검정 결과 저장 완료: {TTEST_RESULT_PATH}")

# 통계 분석 전체 실행

def analyze(df: pd.DataFrame) -> dict[str, object]:
    """전체 통계 분석 과정 실행"""
    # 1. 전체 기술통계
    overall_stats = compute_descriptive_stats(df)

    # 2. 숫자형 변수 상관계수
    correlation = compute_correlation_matrix(df)

    # 3. 가설 1: 출퇴근/비출퇴근 평균 속도 비교 + t-검정
    rush_compare = compare_group_speed(df, "is_rush_hour", "출퇴근 여부")
    ttest_result = run_rush_hour_ttest(df)

    # 4. 가설 2: 평일/주말 평균 속도 기술통계 비교 + 시간대별 패턴 비교
    weekend_compare = compare_group_speed(df, "is_weekend", "평일/주말")
    hourly_weekend_compare = compare_hourly_speed_by_weekend(df)

    # 5. 결과 저장
    save_descriptive_stats(overall_stats, rush_compare, weekend_compare, hourly_weekend_compare)
    save_correlation_matrix(correlation)
    save_ttest_result(ttest_result)

    return ttest_result

# main.py 에서 호출하는 실행 함수

def run(df: pd.DataFrame | None = None) -> dict[str, object]:
    """main.py 파이프라인에서 통계 분석 단계를 실행하고 결과 정보를 반환"""
    if df is None:
        df = load_data()

    ttest_result = analyze(df)
    return {
        **ttest_result,
        "descriptive_statistics_path": str(DESCRIPTIVE_STATS_PATH),
        "correlation_matrix_path": str(CORRELATION_MATRIX_PATH),
        "ttest_result_path": str(TTEST_RESULT_PATH),
    }

# 실행 함수

def main() -> None:
    """통계 분석 전체 과정 실행"""
    result = run()

    print("\n[통계 분석 결과 요약]")
    print(f"t-statistic  : {result['t_statistic']:.4f}")
    print(f"p-value      : {result['p_value']:.6f}")
    print(f"유의미한 차이: {result['is_significant']}")


if __name__ == "__main__":
    main()

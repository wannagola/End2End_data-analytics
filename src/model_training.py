"""
NYC Yellow Taxi 저속 운행 예측 ML Pipeline

기능
1. 정제된 택시 데이터 불러오기
2. 평균 속도 하위 25%를 저속 운행으로 정의
3. 시간대와 운행 경로 관련 파생변수 생성
4. 숫자형/범주형 데이터 전처리
5. HIistGradientBoostingClassifier 모델 학습
6. Accuracy, Precision, Recall, F1-score 평가
7. 학습된 Pipeline과 평가 결과 저장
8. 저장 모델 재로딩 검증
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

_CPU_COUNT = os.cpu_count() or 1
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(_CPU_COUNT - 1, 1)))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# 기본 설정

DATA_PATH = Path("data/processed/taxi_cleaned.parquet")
MODEL_PATH = Path("models/slow_trip_hist_gradient_boosting.joblib")
METRICS_PATH = Path("outputs/model_metrics_hist_gradient_boosting.json")
REPORT_PATH = Path("outputs/classification_report_hist_gradient_boosting.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 데이터 불러오기

def load_data() -> pd.DataFrame:
    """정제된 택시 데이터 불러오기"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(f"원본 데이터 크기: {df.shape}")

    return df

# 타깃 생성

def create_target(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    평균 운행 속도 하위 25%를 저속 운행으로 정의
    
    slow_trip
    - 0: 일반 운행
    - 1: 평균 속도 하위 25% 운행
    """
    if "average_speed_mph" not in df.columns:
        raise ValueError("average_speed_mph 컬럼 없음")
    
    result = df.copy()

    speed_threshold = result["average_speed_mph"].quantile(0.25)

    result["slow_trip"] = (result["average_speed_mph"] <= speed_threshold).astype(int)

    print(
        f"저속 운행 기준: "
        f"{speed_threshold:.2f} mph 이하"
    )

    print("\n저속 운행 클래스 비율:")
    print(
        result["slow_trip"].value_counts(normalize=True).sort_index()
    )

    return result, float(speed_threshold)

# 파생변수 생성
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 사용할 파생변수 생성

    생성변수
    - pickup_time_group: 승차 시간대 구간
    - route_id: 출발지와 도착지 조합
    - is_friday: 금요일 여부
    - is_monday: 월요일 여부
    """
    result = df.copy()

    # 승차 시간을 5개 시간대로 구분
    result["pickup_time_group"] = pd.cut(
        result["pickup_hour"],
        bins = [-1, 5, 9, 15, 19 ,23],
        labels=["late_night", "morning_rush", "daytime", "evening_rush", "night"]
    )

    # 출발지와 도착지 조합 하나의 경로로 표현
    result["route_id"] = (
        result["PULocationID"].fillna(-1).astype(int).astype(str) + "_"
        + result["DOLocationID"].fillna(-1).astype(int).astype(str)
    )

    result["is_friday"] = (result["pickup_dayofweek"] == 4).astype(int)
    result["is_monday"] = (result["pickup_dayofweek"] == 0).astype(int)

    return result

# 데이터 샘플링
def sample_data(df: pd.DataFrame) -> pd.DataFrame:
    """전체 데이터 그대로 사용"""
    print(f"\n전체 데이터 {len(df):,}건을 사용합니다.")
    return df

# 모델 pipeline 생성

def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """전처리와 HistGradientBoostingClassifier를 하나의 pipeline으로 구성"""
    # 숫자형 변수는 결측치 중앙값으로 대체
    numeric_pipeline = Pipeline([
        (
            "imputer", SimpleImputer(strategy="median")
        )
    ])

    # 범주형 변수는 결측치 최빈값으로 대체 후 숫자 코드로 변환
    categorical_pipeline = Pipeline([
        (
            "imputer", SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric", numeric_pipeline, numeric_features
        ),
        (
            "categorical", categorical_pipeline, categorical_features
        )
    ])

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        (
            "preprocessor", preprocessor
        ),
        (
            "model", model
        )
    ])

    return pipeline

# 평가 지표 계산
def calculate_metrics(
        y_test: pd.Series, predictions, speed_threshold: float, train_size: int, test_size: int
) -> dict[str, object]:
    """모델 평가 지표 계산"""

    return {
        "model": "HistGradientBoostingClassifier",
        "speed_threshold_mph": speed_threshold,
        "train_rows": train_size,
        "test_rows": test_size,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": (
            confusion_matrix(y_test, predictions).tolist()
        )
    }

# 결과 출력

def print_results(metrics: dict[str, object], report: str) -> None:
    """모델 평가 결과 터미널에 출력"""
    print("\n[모델 평가]")

    print(
        f"accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
            f"precision: "
            f"{metrics['precision']:.4f}"
    )

    print(
            f"recall   : "
            f"{metrics['recall']:.4f}"
    )

    print(
        f"f1_score : "
        f"{metrics['f1_score']:.4f}"
    )

    print("\n[혼동행렬]")
    for row in metrics["confusion_matrix"]:
        print(row)

    print("\n[분류 보고서]")
    print(report)

# 결과 저장
def save_results(pipeline: Pipeline, metrics: dict[str, object], report: str) -> None:
    """학습 모델과 평가 결과 저장"""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file: json.dump(
        metrics, file, ensure_ascii=False, indent=2
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file: file.write(report)

    print(f"\n모델 저장 완료: {MODEL_PATH}")

    print(f"평가 결과 저장 완료: {METRICS_PATH}")

    print(f"분류 보고서 저장 완료: {REPORT_PATH}")

# 저장된 모델 검증

def verify_saved_model(X_test: pd.DataFrame, original_predictions) -> None:
    """저장한 모델 다시 불러와 저장 전후 예측 결과 같은지 검증"""
    loaded_pipeline = joblib.load(MODEL_PATH)

    loaded_predictions = loaded_pipeline.predict(X_test)

    if not(original_predictions == loaded_predictions).all():
        raise ValueError("저장 전후 모델 예측 결과가 다릅니다.")

    print("저장 모델 재로딩 검증 완료")

# 모델 학습 및 평가
def train_and_evaluate(df: pd.DataFrame) -> dict[str, object]:
    """전체 ML 학습과 평가 과정 실행"""
    # 1. 저속 운행 타깃 생성
    df, speed_threshold = create_target(df)

    # 2. 시간대와 운행 경로 파생변수 생성
    df = create_features(df)

    # 3. 대용량 데이터 샘플링
    df = sample_data(df)

    # 연속적인 수치 의미가 있는 변수
    numeric_features = ["passenger_count"]

    # 숫자로 저장되어 있지만 실제로는 시간/요일/지역/여부 나타내는 범주형 변수
    categorical_features = [
        "pickup_hour", "pickup_dayofweek", "pickup_time_group", "PULocationID", "DOLocationID",
        "route_id", "is_weekend", "is_rush_hour", "is_friday", "is_monday"
    ]

    feature_columns = (numeric_features + categorical_features)

    if not feature_columns:
        raise ValueError("모델 학습에 사용할 입력 변수가 없습니다.")

    print("\n[사용한 입력 변수]")
    print(feature_columns)

    X = df[feature_columns]
    y = df["slow_trip"]

    # 학습 데이터와 테스트 데이터 8:2 분리
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    )

    print(f"\n학습 데이터: {len(X_train):,}건")
    print(f"테스트 데이터: {len(X_test):,}건")

    # 전처리와 모델이 포함된 Pipeline 생성
    pipeline = build_pipeline(numeric_features, categorical_features)

    print("\n모델 학습을 시작합니다.")

    pipeline.fit(X_train, y_train)

    print("모델 학습이 완료되었습니다.")

    predictions = pipeline.predict(X_test)

    metrics = calculate_metrics(
        y_test=y_test, predictions=predictions, speed_threshold=speed_threshold,
        train_size=len(X_train), test_size=len(X_test)
    )

    report = classification_report(y_test, predictions, zero_division=0)

    print_results(metrics, report)

    save_results(pipeline, metrics, report)

    verify_saved_model(X_test, predictions)

    return metrics

# main.py 에서 호출하는 실행 함수
def run(df: pd.DataFrame | None = None) -> dict[str, object]:
    """main.py 파이프라인에서 ML 단계를 실행하고 결과 정보를 반환"""
    if df is None:
        df = load_data()

    metrics = train_and_evaluate(df)
    return {
        **metrics,
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "report_path": str(REPORT_PATH),
    }

# 실행 함수
def main() -> None:
    """ML Pipeline 전체 과정 실행"""

    metrics = run()

    print("[모델 결과]")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()

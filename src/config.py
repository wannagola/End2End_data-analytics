"""프로젝트 공유 설정: 경로 · 상수 · random_state · 로깅.

**이 파일은 팀 전원이 함께 쓴다.** 담당자가 각자 자기 파일에 상수를 적으면
같은 가설인데 사람마다 숫자가 달라진다. (예: 러시아워를 누구는 16시부터,
누구는 17시부터 잡으면 시각화와 통계 결과가 어긋난다.)
값이 필요하면 여기에 추가하고, 각자 파일에서는 `from src import config` 로 가져다 쓴다.

설계 원칙 5-3(재현 가능성):
- 절대경로를 코드에 박지 않는다. 이 파일 위치를 기준으로 상대경로를 계산한다.
- 난수 시드는 RANDOM_STATE 하나만 두고 모든 모듈이 가져다 쓴다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 경로 (전부 프로젝트 루트 기준 상대경로)
# ---------------------------------------------------------------------------
# __file__ = <root>/src/config.py 이므로 parents[1] 이 프로젝트 루트다.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"              # 원본 (git 제외)
PROCESSED_DIR: Path = DATA_DIR / "processed"  # 정제 결과 (git 제외)
SAMPLE_DIR: Path = DATA_DIR / "sample"        # 테스트용 소량 데이터 (git 포함)
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"   # 차트·지표 등 산출물
MODEL_DIR: Path = PROJECT_ROOT / "models"

RAW_DATA_PATH: Path = RAW_DIR / "yellow_tripdata_2026-05.parquet"
CLEANED_PATH: Path = PROCESSED_DIR / "taxi_cleaned.parquet"   # ①의 산출물 = ②③④의 입력
SAMPLE_DATA_PATH: Path = SAMPLE_DIR / "synthetic_trips.parquet"

LOG_PATH: Path = OUTPUT_DIR / "run.log"
REPORT_PATH: Path = PROJECT_ROOT / "report.md"

# ---------------------------------------------------------------------------
# 2. 데이터 원천
# ---------------------------------------------------------------------------
DATA_URLS: tuple[str, ...] = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet",
)
MIN_VALID_BYTES: int = 5_000_000     # 약 66MB 파일이다. 이보다 작으면 에러 페이지로 본다.

# ---------------------------------------------------------------------------
# 3. 스키마
# ---------------------------------------------------------------------------
# TLC 파케이의 컬럼 구성은 연월에 따라 바뀐다
# (airport_fee / Airport_fee 대소문자 혼용, 2025년 혼잡통행료 도입 후 cbd_congestion_fee 추가 등).
# 전체 목록을 하드코딩하면 다른 달 파일에서 바로 깨지므로, 아래 필수 컬럼만 검증하고 진행한다.
REQUIRED_COLUMNS: list[str] = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount",
]
# 있으면 읽고 없으면 넘어가는 컬럼
OPTIONAL_COLUMNS: list[str] = [
    "VendorID", "extra", "mta_tax", "improvement_surcharge",
    "congestion_surcharge", "airport_fee", "Airport_fee", "cbd_congestion_fee",
]
# 이 컬럼에 결측이 있으면 그 행은 분석에 쓸 수 없다 (속도 계산 자체가 불가능)
#
# passenger_count 를 여기서 뺀 이유:
#   실제 2026-05 데이터에서 passenger_count · RatecodeID · congestion_surcharge ·
#   Airport_fee 네 컬럼의 결측 개수가 955,371 로 **정확히 동일**했다.
#   특정 사업자가 이 필드들을 통째로 비워 보고한다는 뜻이다.
#   즉 무작위 결측(MCAR)이 아니라 한 사업자의 데이터를 전부 버리는 것이 되고,
#   그 사업자가 특정 지역·시간대에 몰려 있으면 속도 분석 자체가 편향된다.
#   승객 수는 세 가설 어디에도 쓰이지 않으므로, 결측을 남긴 채 행을 살린다.
CRITICAL_COLUMNS: list[str] = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "trip_distance", "fare_amount",
    "PULocationID", "DOLocationID",
]

# 결측을 허용하는 컬럼 — 값이 있을 때만 범위를 검사하고, 없으면 통과시킨다.
NULLABLE_COLUMNS: list[str] = ["passenger_count"]

# ---------------------------------------------------------------------------
# 4. 가설 공통 상수  ★ 팀 전원이 이 값을 쓴다 ★
# ---------------------------------------------------------------------------
# 가설 1: 출퇴근 시간대 정의 (오전 7~9시, 오후 4~7시)
RUSH_HOURS: list[int] = [7, 8, 9, 16, 17, 18, 19]

# 가설 2: 시간대별 패턴 비교용 구간
LATE_NIGHT_HOURS: list[int] = [2, 3, 4, 5]     # 정체가 없어야 하는 대조 구간
WEEKEND_DAYS: list[int] = [5, 6]               # 토(5), 일(6)

# 통계
ALPHA: float = 0.05
MIN_SAMPLES_FOR_TTEST: int = 30

# 모델
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# 5. 이상치 제거 기준 — 각 임계값에 근거를 남긴다
# ---------------------------------------------------------------------------
MIN_FARE: float = 2.50          # 기본요금 하한. 이보다 낮으면 기록 오류
MAX_FARE: float = 500.0         # 뉴욕시 일반 운행에서 현실적인 상한
MIN_DISTANCE: float = 0.1       # 마일. 0마일 운행은 미터기 오작동
MAX_DISTANCE: float = 100.0     # 이 이상은 GPS 오류
MIN_DURATION_MIN: float = 1.0   # 1분 미만은 승차 취소
MAX_DURATION_MIN: float = 360.0  # 6시간 초과는 미터기 미종료
MIN_SPEED_MPH: float = 1.0      # 시속 1마일 미만은 사실상 정지 상태 = 기록 오류
MAX_SPEED_MPH: float = 80.0     # 뉴욕 시내에서 물리적으로 불가능
MIN_PASSENGERS: int = 1
MAX_PASSENGERS: int = 6


# ---------------------------------------------------------------------------
# 6. 로깅 (설계 원칙 5-2: print 대신 logging)
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """루트 로거를 콘솔 + 파일 양쪽으로 설정하고 반환한다.

    Args:
        log_path: 로그 파일 경로. None 이면 config.LOG_PATH.
        level:    로깅 레벨 (기본 INFO).

    Returns:
        설정이 끝난 루트 로거.

    Note:
        여러 번 호출해도 핸들러가 중복되지 않도록 기존 핸들러를 먼저 제거한다.
    """
    log_path = log_path or LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # mode="w": 실행할 때마다 새로 쓴다. 이번 실행의 로그만 남아야 리포트와 짝이 맞는다.
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return root


def ensure_dirs() -> None:
    """실행에 필요한 디렉터리를 만든다. 이미 있으면 그대로 둔다."""
    for directory in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, OUTPUT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def verify_file(path: Path, min_bytes: int = 1) -> int:
    """파일이 실제로 생성됐고 크기가 0이 아닌지 확인한다.

    설계 원칙 5-2: "저장했다"를 믿지 않는다 — 존재와 크기를 확인한다.

    Args:
        path:      확인할 파일 경로.
        min_bytes: 최소 허용 바이트.

    Returns:
        파일 크기(바이트).

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError:        파일이 min_bytes 보다 작을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"생성됐어야 할 파일이 없다: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(f"파일 크기가 비정상이다: {path} ({size} bytes < {min_bytes})")
    return size


if __name__ == "__main__":
    # 모듈 최상단에 실행 코드를 두지 않는다 (설계 원칙 5-5).
    setup_logging()
    log = logging.getLogger("config")
    log.info("PROJECT_ROOT = %s", PROJECT_ROOT)
    log.info("RUSH_HOURS   = %s", RUSH_HOURS)
    log.info("RANDOM_STATE = %s", RANDOM_STATE)

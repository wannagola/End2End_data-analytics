"""데이터 확보와 공유 설정 검증.

네트워크에 의존하지 않는다. 외부 접속이 필요한 테스트는 CI 에서 간헐적으로 실패하고,
그러면 사람들이 빨간 배지를 무시하기 시작해 테스트 자체가 무의미해진다.
`file://` URL 로 다운로드 경로를 재현한다.
"""

from __future__ import annotations

import logging

import pytest

from src import config, download


# ===========================================================================
# download
# ===========================================================================
def test_existing_file_is_reused(tmp_path):
    """이미 정상 크기의 파일이 있으면 네트워크를 타지 않고 그대로 쓴다."""
    dest = tmp_path / "taxi.parquet"
    dest.write_bytes(b"x" * (config.MIN_VALID_BYTES + 10))
    # urls 를 빈 튜플로 줘도 성공한다 = 다운로드를 시도하지 않았다는 뜻
    assert download.ensure_data(dest=dest, urls=()) == dest


def test_all_sources_fail_raises(tmp_path):
    """모든 원천이 실패하면 조용히 넘어가지 않고 RuntimeError 를 던진다."""
    tiny = tmp_path / "tiny.parquet"
    tiny.write_bytes(b"404 not found")          # 크기 미달 -> 실패로 간주돼야 한다
    missing = tmp_path / "nope.parquet"

    with pytest.raises(RuntimeError, match="다운로드에 실패"):
        download.ensure_data(dest=tmp_path / "out.parquet",
                             urls=(tiny.as_uri(), missing.as_uri()))


def test_partial_file_is_not_left_behind(tmp_path):
    """실패한 다운로드가 목적지에 반쪽짜리 파일을 남기지 않는다."""
    tiny = tmp_path / "tiny.parquet"
    tiny.write_bytes(b"too small")
    dest = tmp_path / "out.parquet"

    with pytest.raises(RuntimeError):
        download.ensure_data(dest=dest, urls=(tiny.as_uri(),))

    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_from_file_url(tmp_path):
    """정상 크기의 원천이면 받아서 검증까지 통과한다."""
    source = tmp_path / "source.parquet"
    source.write_bytes(b"y" * (config.MIN_VALID_BYTES + 1))
    dest = tmp_path / "dest.parquet"

    assert download.ensure_data(dest=dest, urls=(source.as_uri(),)) == dest
    assert dest.stat().st_size == source.stat().st_size


# ===========================================================================
# config
# ===========================================================================
def test_paths_are_under_project_root():
    """모든 경로가 프로젝트 루트 아래에 있다 (절대경로 하드코딩 금지)."""
    for path in (config.RAW_DATA_PATH, config.CLEANED_PATH,
                 config.OUTPUT_DIR, config.LOG_PATH, config.REPORT_PATH):
        assert config.PROJECT_ROOT in path.parents or path.parent == config.PROJECT_ROOT


def test_rush_hours_are_valid():
    """러시아워 정의가 유효한 시각이고 중복이 없다."""
    assert all(0 <= h <= 23 for h in config.RUSH_HOURS)
    assert len(config.RUSH_HOURS) == len(set(config.RUSH_HOURS))


def test_rush_and_late_night_do_not_overlap():
    """가설 2의 두 비교 구간이 겹치지 않는다 (겹치면 대조가 성립하지 않는다)."""
    assert not (set(config.RUSH_HOURS) & set(config.LATE_NIGHT_HOURS))


def test_outlier_bounds_are_ordered():
    """이상치 임계값의 하한이 상한보다 작다 (오타로 뒤집히면 전부 걸러진다)."""
    assert config.MIN_FARE < config.MAX_FARE
    assert config.MIN_DISTANCE < config.MAX_DISTANCE
    assert config.MIN_DURATION_MIN < config.MAX_DURATION_MIN
    assert config.MIN_SPEED_MPH < config.MAX_SPEED_MPH
    assert config.MIN_PASSENGERS < config.MAX_PASSENGERS


def test_critical_columns_are_subset_of_required():
    """필수 결측 검사 대상이 필수 컬럼 목록 안에 들어 있다."""
    assert set(config.CRITICAL_COLUMNS) <= set(config.REQUIRED_COLUMNS)


def test_verify_file_missing(tmp_path):
    """없는 파일은 FileNotFoundError 를 던진다."""
    with pytest.raises(FileNotFoundError):
        config.verify_file(tmp_path / "nope.txt")


def test_verify_file_ok(tmp_path):
    """정상 파일은 크기를 돌려준다."""
    good = tmp_path / "good.txt"
    good.write_text("x" * 50, encoding="utf-8")
    assert config.verify_file(good, min_bytes=10) == 50


def test_setup_logging_writes_to_file(tmp_path):
    """로깅이 파일에도 남는다. 여러 번 호출해도 핸들러가 중복되지 않는다."""
    log_path = tmp_path / "run.log"
    config.setup_logging(log_path)
    config.setup_logging(log_path)                      # 두 번 호출
    assert len(logging.getLogger().handlers) == 2       # 콘솔 1 + 파일 1

    logging.getLogger("test").info("검증용 메시지")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "검증용 메시지" in log_path.read_text(encoding="utf-8")

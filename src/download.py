"""원본 데이터 확보 모듈.

설계 원칙 5-6: 데이터는 git 에 올리지 않는다. 대신 "받는 방법"을 올린다.
채점자가 clone 한 폴더에서 `python main.py` 만 실행해도 데이터가 자동으로 확보돼야 한다.

원본은 약 66MB 다. `.gitignore` 로 `data/raw/` 를 제외하고, 이 스크립트가 대신 받아온다.
"""

from __future__ import annotations

import logging
import socket
import urllib.error
import urllib.request
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; day2-final/1.0)"}
_TIMEOUT_SEC = 300      # 66MB 다. 회선이 느린 환경을 감안해 넉넉히 잡는다.
_CHUNK = 1024 * 1024


def _download_one(url: str, dest: Path) -> int:
    """URL 하나를 임시 파일로 스트리밍 다운로드하고 검증한 뒤 dest 로 옮긴다.

    파일 전체를 메모리에 올리지 않고 1MB 씩 나눠 쓴다.

    Args:
        url:  내려받을 주소.
        dest: 최종 저장 경로.

    Returns:
        저장된 파일 크기(바이트).

    Raises:
        urllib.error.URLError / socket.timeout / ValueError: 실패 원인을 그대로 올려보낸다.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")   # 다운로드 중 파일을 본 경로에 두지 않는다
    request = urllib.request.Request(url, headers=_HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SEC) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            next_report = 20 * 1024 * 1024      # 20MB 마다 진행 상황을 남긴다
            with tmp.open("wb") as fp:
                while chunk := response.read(_CHUNK):
                    fp.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        pct = f" ({done / total * 100:.0f}%)" if total else ""
                        logger.info("  다운로드 중... %s MB%s", f"{done // 1024 ** 2:,}", pct)
                        next_report += 20 * 1024 * 1024
    except BaseException:
        tmp.unlink(missing_ok=True)             # 중단 시 반쪽 파일을 남기지 않는다
        raise

    size = tmp.stat().st_size
    # 에러 페이지(HTML)를 데이터로 착각하지 않도록 크기로 1차 방어한다.
    if size < config.MIN_VALID_BYTES:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"받은 파일이 너무 작다 ({size:,} bytes) — 에러 페이지일 가능성")

    tmp.replace(dest)   # 검증을 통과한 뒤에야 정식 경로로 승격
    return size


def ensure_data(dest: Path | None = None, urls: tuple[str, ...] | None = None) -> Path:
    """원본 데이터를 확보한다. 이미 있으면 다시 받지 않는다.

    실패하면 예외를 던진다 — 조용히 빈 데이터로 넘어가지 않는다(설계 원칙 5-2).

    Args:
        dest: 저장 경로. None 이면 config.RAW_DATA_PATH.
        urls: 시도할 URL 목록. None 이면 config.DATA_URLS.

    Returns:
        확보된 데이터 파일 경로.

    Raises:
        RuntimeError: 모든 URL 이 실패했을 때.
    """
    dest = dest or config.RAW_DATA_PATH
    urls = urls or config.DATA_URLS
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= config.MIN_VALID_BYTES:
        logger.info("데이터 재사용: %s (%s bytes) — 다시 받지 않는다",
                    dest.name, f"{dest.stat().st_size:,}")
        return dest

    failures: list[str] = []
    for idx, url in enumerate(urls, start=1):
        try:
            logger.info("데이터 다운로드 시도 %d/%d: %s", idx, len(urls), url)
            size = _download_one(url, dest)
            logger.info("다운로드 성공: %s (%s bytes)", dest.name, f"{size:,}")
            return dest
        except (urllib.error.URLError, socket.timeout, ValueError, OSError) as exc:
            # 예외를 삼키지 않는다. 무엇을 왜 잡았는지 남기고 다음 후보로 넘어간다.
            logger.warning("실패 (%s): %s", type(exc).__name__, exc)
            failures.append(f"{url} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "모든 데이터 원천에서 다운로드에 실패했다. 네트워크 또는 URL 을 확인하라.\n  - "
        + "\n  - ".join(failures)
    )


if __name__ == "__main__":
    config.setup_logging()
    ensure_data()

"""NYC Yellow Taxi 속도 분석 — 진입점.

이 파일 하나를 실행하면 데이터 확보부터 report.md 생성까지 전체가 끝까지 돈다.

    $ python main.py              # 원본(약 66MB)을 받아 전체 실행
    $ python main.py --sample     # 합성 샘플로 빠르게 흐름만 점검 (다운로드 없음)

각 단계는 앞 단계의 출력만 입력으로 받고(전역 변수 공유 없음),
결과 정보(`*_info`)를 return 해서 마지막에 report 가 문서로 조립한다.
이 구조라야 "손으로 숫자를 옮겨 적을 여지가 없는 자동 생성"이 성립한다.

--------------------------------------------------------------------------
팀 담당 구간
--------------------------------------------------------------------------
    ① 데이터 준비  src/data_preparation.py   (완료)
    ② 시각화       src/visualization.py      가설 1·2 차트
    ③ 통계 분석    src/statistical_analysis.py  가설 1·2 검정
    ④ ML Pipeline  src/model_training.py     가설 3

아직 없는 모듈은 자동으로 건너뛴다. 각자 자기 파일을 만들어 push 하면
이 파일을 고치지 않아도 그 단계가 살아난다.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src import config, data_preparation, download

logger = logging.getLogger("main")


def _try_stage(module_name: str, label: str, *args: Any) -> dict[str, Any] | None:
    """팀원 모듈이 있으면 run() 을 호출하고, 없으면 건너뛴다.

    한 사람의 파일이 아직 없다고 해서 전체 파이프라인이 멈추면 안 된다.
    반대로 파일이 있는데 안에서 터지는 것은 조용히 넘기지 않는다 — 그건 진짜 버그다.

    Args:
        module_name: `src.` 아래 모듈 이름 (예: "visualization").
        label:       로그에 찍을 단계 이름.
        *args:       해당 모듈의 run() 에 넘길 인자.

    Returns:
        모듈의 run() 반환값. 모듈이 없으면 None.
    """
    try:
        module = importlib.import_module(f"src.{module_name}")
    except ModuleNotFoundError:
        logger.warning("[건너뜀] %s — src/%s.py 가 아직 없다", label, module_name)
        return None

    if not hasattr(module, "run"):
        logger.warning("[건너뜀] %s — src/%s.py 에 run() 이 없다", label, module_name)
        return None

    logger.info("=== %s ===", label)
    return module.run(*args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """명령행 인자를 해석한다."""
    parser = argparse.ArgumentParser(description="NYC Yellow Taxi 속도 분석 파이프라인")
    parser.add_argument(
        "--sample", action="store_true",
        help="원본 대신 합성 샘플(data/sample)로 실행한다. 다운로드 없이 흐름 점검용.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """전체 파이프라인을 순서대로 실행한다.

    Args:
        argv: 명령행 인자. None 이면 sys.argv 를 쓴다.

    Returns:
        종료 코드 (0 = 성공, 1 = 실패).
    """
    args = parse_args(argv)
    config.ensure_dirs()
    config.setup_logging()
    started = time.perf_counter()

    logger.info("=" * 78)
    logger.info("NYC Yellow Taxi 속도 분석 | seed=%d | 러시아워=%s%s",
                config.RANDOM_STATE, config.RUSH_HOURS,
                "  [합성 샘플 모드]" if args.sample else "")
    logger.info("=" * 78)

    try:
        # 0단계: 데이터 확보 (없으면 내려받고, 있으면 재사용)
        logger.info("=== 0단계: 데이터 확보 ===")
        if args.sample:
            raw_path = config.SAMPLE_DATA_PATH
            if not raw_path.exists():
                from tests.fixtures.make_synthetic import make_synthetic
                make_synthetic()
            logger.warning("--sample 모드: 합성 데이터로 실행한다. "
                           "여기서 나온 수치는 실제 분석 결과가 아니다.")
        else:
            raw_path = download.ensure_data()

        # ① 데이터 준비 — 로딩 비교 · 정제 · 파생변수 · 저장
        df, prep_info = data_preparation.run(raw_path)

        # ② 시각화 (가설 1·2)
        viz_info = _try_stage("visualization", "② 시각화", df)

        # ③ 통계 분석 (가설 1·2)
        stat_info = _try_stage("statistical_analysis", "③ 통계 분석", df)

        # ④ ML Pipeline (가설 3)
        model_info = _try_stage("model_training", "④ ML Pipeline", df)

        # 리포트 자동 생성
        runtime = time.perf_counter() - started
        report_info = _try_stage("report", "리포트 생성",
                                 prep_info, viz_info, stat_info, model_info, runtime)

    except Exception as exc:   # noqa: BLE001 — 최상위에서만 광범위 포착
        # 조용히 틀리느니 시끄럽게 멈춘다. 스택트레이스를 로그 파일에도 남긴다.
        logger.exception("파이프라인 실패 (%s): %s", type(exc).__name__, exc)
        return 1

    total = time.perf_counter() - started
    logger.info("=" * 78)
    logger.info("전체 완료 | 소요 %.2f초 | 최종 %s행", total, f"{prep_info['final_rows']:,}")
    logger.info("산출물:")
    logger.info("  - %s", prep_info["save"]["path"])
    for stage, info in (("시각화", viz_info), ("통계", stat_info),
                        ("모델", model_info), ("리포트", report_info)):
        logger.info("  - %s: %s", stage, "완료" if info else "미구현")
    logger.info("  - %s", config.LOG_PATH.relative_to(config.PROJECT_ROOT))
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

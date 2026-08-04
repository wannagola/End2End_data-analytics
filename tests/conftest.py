"""pytest 공용 픽스처.

테스트는 400만 행 원본이 아니라 합성 픽스처(수천 행)로 돈다.
- 원본은 git 에 없다. 없는 파일에 의존하는 테스트는 CI 에서 항상 실패한다.
- 수천 행이면 모든 분기를 지나가면서도 전체 테스트가 몇 초 안에 끝난다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, data_preparation as dp
from tests.fixtures.make_synthetic import make_synthetic


@pytest.fixture(scope="session")
def sample_path():
    """합성 샘플 경로. 없으면 그 자리에서 만든다."""
    if not config.SAMPLE_DATA_PATH.exists():
        make_synthetic(n_rows=4_000)
    return config.SAMPLE_DATA_PATH


@pytest.fixture(scope="session")
def raw_df(sample_path) -> pd.DataFrame:
    """샘플을 Pandas 로 읽은 원시 데이터프레임 (정제 전)."""
    df, _ = dp.load_pandas(sample_path)
    return df


@pytest.fixture(scope="session")
def clean_df(sample_path) -> pd.DataFrame:
    """정제가 끝난 데이터프레임 (파일 저장은 하지 않는다)."""
    df, _ = dp.run(sample_path, save=False)
    return df


@pytest.fixture(scope="session")
def prep_info(sample_path) -> dict:
    """정제 단계가 반환한 정보 dict."""
    _, info = dp.run(sample_path, save=False)
    return info

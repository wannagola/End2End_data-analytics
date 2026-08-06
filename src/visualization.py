"""
NYC Yellow Taxi 시각화 (가설 1 · 가설 2)

기능
1. 정제된 택시 데이터 불러오기
2. 가설 1: 시간대별 평균 속도 라인차트 (출퇴근 시간대 음영 표시)
3. 가설 1: 출퇴근/비출퇴근 평균 속도 boxplot
4. 가설 2: 평일/주말 시간대별 평균 속도 비교
5. 가설 2: 요일 x 시간대 평균 속도 Plotly 히트맵
6. 차트 제목·축 라벨·해석 문장 작성 및 저장 (PNG/HTML)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 저장만 하면 되므로 GUI 백엔드를 쓰지 않는다

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src import config

DATA_PATH = config.CLEANED_PATH
HOURLY_SPEED_PATH = config.OUTPUT_DIR / "hourly_speed.png"
RUSH_BOXPLOT_PATH = config.OUTPUT_DIR / "rush_hour_boxplot.png"
DAY_HOUR_HEATMAP_PATH = config.OUTPUT_DIR / "day_hour_heatmap.html"

DAY_LABELS = ["월", "화", "수", "목", "금", "토", "일"]  # pickup_dayofweek 0~6 순서와 일치

# matplotlib 기본 폰트(Arial 계열)에는 한글 글리프가 없어 제목·라벨이 네모로 깨진다.
# OS별로 흔한 한글 폰트 후보를 순서대로 찾아, 설치된 첫 번째 것을 쓴다.
_KOREAN_FONT_CANDIDATES = [
    "AppleGothic", "Apple SD Gothic Neo",      # macOS
    "Malgun Gothic",                            # Windows
    "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",  # Linux
]
_installed_fonts = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((f for f in _KOREAN_FONT_CANDIDATES if f in _installed_fonts), None)
if not _korean_font:
    print("경고: 한글을 지원하는 폰트를 찾지 못했다 — 차트의 한글이 깨질 수 있다. "
          "(예: 나눔고딕 설치 필요)")

# sns.set_theme() 이 자체 폰트 설정으로 rcParams 를 덮어쓰므로, 한글 폰트는 그 뒤에 적용한다.
sns.set_theme(style="whitegrid")
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트는 보통 마이너스(-) 글리프가 없다

# 데이터 불러오기

def load_data() -> pd.DataFrame:
    """정제된 택시 데이터 불러오기"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(f"원본 데이터 크기: {df.shape}")

    return df

# 가설 1 + 가설 2: 시간대별 평균 속도 (전체 / 평일·주말 비교)

def plot_hourly_speed(df: pd.DataFrame) -> str:
    """
    시간대별 평균 속도(가설 1)와 평일/주말 시간대별 평균 속도(가설 2)를
    위아래 두 개의 subplot으로 그려 하나의 PNG에 저장한다.
    """
    hourly_overall = df.groupby("pickup_hour")["average_speed_mph"].mean()
    hourly_by_weekend = (
        df.groupby(["pickup_hour", "is_weekend"])["average_speed_mph"]
        .mean()
        .unstack("is_weekend")
        .rename(columns={False: "평일", True: "주말"})
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # 가설 1: 전체 시간대별 평균 속도. 출퇴근 시간대는 음영으로 강조한다.
    ax1.plot(hourly_overall.index, hourly_overall.values, marker="o", color="steelblue")
    for hour in config.RUSH_HOURS:
        ax1.axvspan(hour - 0.5, hour + 0.5, color="salmon", alpha=0.2)
    ax1.set_title("가설 1: 시간대별 평균 운행 속도 (음영 = 출퇴근 시간대)")
    ax1.set_ylabel("평균 속도 (mph)")

    # 가설 2: 평일/주말 시간대별 평균 속도 비교
    ax2.plot(hourly_by_weekend.index, hourly_by_weekend["평일"],
              marker="o", label="평일", color="darkorange")
    ax2.plot(hourly_by_weekend.index, hourly_by_weekend["주말"],
              marker="o", label="주말", color="seagreen")
    ax2.set_title("가설 2: 평일 · 주말 시간대별 평균 운행 속도")
    ax2.set_xlabel("승차 시각 (0~23시)")
    ax2.set_ylabel("평균 속도 (mph)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(title="구분")

    fig.tight_layout()
    HOURLY_SPEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HOURLY_SPEED_PATH, dpi=150)
    plt.close(fig)

    config.verify_file(HOURLY_SPEED_PATH, min_bytes=1_000)
    print(f"\n시간대별 평균 속도 차트 저장 완료: {HOURLY_SPEED_PATH}")

    slowest_hour = int(hourly_overall.idxmin())
    fastest_hour = int(hourly_overall.idxmax())
    weekday_slowest = int(hourly_by_weekend["평일"].idxmin())
    weekend_slowest = int(hourly_by_weekend["주말"].idxmin())

    interpretation = (
        f"전체 평균 속도는 {slowest_hour}시에 가장 낮고({hourly_overall.min():.2f} mph) "
        f"{fastest_hour}시에 가장 높다({hourly_overall.max():.2f} mph). "
        f"평일은 {weekday_slowest}시에 속도가 가장 낮아 출퇴근 시간대와 맞물리는 반면, "
        f"주말은 {weekend_slowest}시에 가장 낮아 정체가 발생하는 시간대 패턴이 서로 다르게 나타난다."
    )
    print(interpretation)

    return interpretation

# 가설 1: 출퇴근/비출퇴근 boxplot

def plot_rush_hour_boxplot(df: pd.DataFrame) -> str:
    """가설 1: 출퇴근/비출퇴근 시간대 평균 속도 분포 boxplot"""
    plot_df = df[["is_rush_hour", "average_speed_mph"]].copy()
    plot_df["구분"] = plot_df["is_rush_hour"].map({True: "출퇴근 시간대", False: "비출퇴근 시간대"})

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(
        data=plot_df, x="구분", y="average_speed_mph",
        order=["출퇴근 시간대", "비출퇴근 시간대"],
        hue="구분", palette={"출퇴근 시간대": "salmon", "비출퇴근 시간대": "steelblue"},
        legend=False, ax=ax,
        showfliers=False,  # 수백만 건 규모라 이상치 점까지 찍으면 상자 모양이 묻힌다
    )
    ax.set_title("가설 1: 출퇴근 · 비출퇴근 시간대 평균 속도 분포")
    ax.set_xlabel("구분")
    ax.set_ylabel("평균 속도 (mph)")

    fig.tight_layout()
    RUSH_BOXPLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUSH_BOXPLOT_PATH, dpi=150)
    plt.close(fig)

    config.verify_file(RUSH_BOXPLOT_PATH, min_bytes=1_000)
    print(f"출퇴근 boxplot 저장 완료: {RUSH_BOXPLOT_PATH}")

    rush_median = plot_df.loc[plot_df["구분"] == "출퇴근 시간대", "average_speed_mph"].median()
    non_rush_median = plot_df.loc[plot_df["구분"] == "비출퇴근 시간대", "average_speed_mph"].median()

    if rush_median < non_rush_median:
        interpretation = (
            f"출퇴근 시간대의 중앙값 속도는 {rush_median:.2f} mph 로 "
            f"비출퇴근 시간대({non_rush_median:.2f} mph)보다 낮다. "
            "분포 전체가 아래로 이동해 있어 출퇴근 시간대의 정체가 일부 구간이 아닌 "
            "전반적인 현상임을 보여준다."
        )
    else:
        interpretation = (
            f"출퇴근 시간대 중앙값 속도({rush_median:.2f} mph)가 "
            f"비출퇴근 시간대({non_rush_median:.2f} mph)보다 낮게 나타나지 않았다. "
            "가설 1과 다른 방향이므로 통계 검정(ttest_ind) 결과와 함께 확인이 필요하다."
        )
    print(interpretation)

    return interpretation

# 가설 2: 요일 x 시간대 Plotly 히트맵

def plot_day_hour_heatmap(df: pd.DataFrame) -> str:
    """가설 2: 요일 x 시간대 평균 속도 Plotly 히트맵"""
    pivot = (
        df.groupby(["pickup_dayofweek", "pickup_hour"])["average_speed_mph"]
        .mean()
        .unstack("pickup_hour")
        .reindex(index=range(7), columns=range(24))
        .round(2)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{h}시" for h in pivot.columns],
            y=DAY_LABELS,
            colorscale="RdYlGn",
            colorbar={"title": "평균 속도 (mph)"},
            hovertemplate="요일=%{y}<br>시간=%{x}<br>평균 속도=%{z} mph<extra></extra>",
        )
    )
    fig.update_layout(
        title="가설 2: 요일 x 시간대별 평균 운행 속도",
        xaxis_title="승차 시각",
        yaxis_title="요일",
    )

    DAY_HOUR_HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(DAY_HOUR_HEATMAP_PATH, include_plotlyjs="cdn")

    config.verify_file(DAY_HOUR_HEATMAP_PATH, min_bytes=1_000)
    print(f"요일 x 시간대 히트맵 저장 완료: {DAY_HOUR_HEATMAP_PATH}")

    weekday_mean_by_hour = pivot.loc[0:4].mean(axis=0)
    weekend_mean_by_hour = pivot.loc[5:6].mean(axis=0)
    weekday_slow_hour = int(weekday_mean_by_hour.idxmin())
    weekend_slow_hour = int(weekend_mean_by_hour.idxmin())

    interpretation = (
        f"평일은 {weekday_slow_hour}시 전후 출퇴근 시간대에 정체가 뚜렷한 반면, "
        f"주말은 {weekend_slow_hour}시 전후에 속도 저하가 나타나 정체가 발생하는 시간대의 "
        "패턴이 요일 유형에 따라 다르게 형성된다."
    )
    print(interpretation)

    return interpretation

# 가설 1·2 시각화 전체 실행

def visualize(df: pd.DataFrame) -> dict[str, str]:
    """가설 1 · 가설 2에 필요한 차트를 모두 그리고 해석 문장을 모은다"""
    hourly_interpretation = plot_hourly_speed(df)
    boxplot_interpretation = plot_rush_hour_boxplot(df)
    heatmap_interpretation = plot_day_hour_heatmap(df)

    return {
        "hourly_speed_interpretation": hourly_interpretation,
        "rush_hour_boxplot_interpretation": boxplot_interpretation,
        "day_hour_heatmap_interpretation": heatmap_interpretation,
    }

# main.py 에서 호출하는 실행 함수

def run(df: pd.DataFrame | None = None) -> dict[str, object]:
    """main.py 파이프라인에서 시각화 단계를 실행하고 결과 정보를 반환"""
    if df is None:
        df = load_data()

    interpretations = visualize(df)
    return {
        **interpretations,
        "hourly_speed_path": str(HOURLY_SPEED_PATH),
        "rush_hour_boxplot_path": str(RUSH_BOXPLOT_PATH),
        "day_hour_heatmap_path": str(DAY_HOUR_HEATMAP_PATH),
    }

# 실행 함수

def main() -> None:
    """시각화 전체 과정 실행"""
    result = run()

    print("\n[시각화 결과 요약]")
    print(f"- {result['hourly_speed_path']}")
    print(f"- {result['rush_hour_boxplot_path']}")
    print(f"- {result['day_hour_heatmap_path']}")


if __name__ == "__main__":
    main()

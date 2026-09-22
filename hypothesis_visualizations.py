from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "hypothesis_charts"
DASHBOARD_PATH = BASE_DIR / "hypothesis_dashboard.png"

SURVIVAL_COLOR = "#2A9D8F"
DEATH_COLOR = "#E76F51"
SECONDARY_COLOR = "#457B9D"


def prepare_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data["성별"] = data["Sex"].map({"female": "여성", "male": "남성"})
    data["가족규모"] = data["SibSp"] + data["Parch"] + 1
    data["혼자여부"] = np.where(data["가족규모"] == 1, "혼자 탑승", "가족 동반")
    data["가족규모그룹"] = pd.cut(
        data["가족규모"],
        bins=[0, 1, 4, np.inf],
        labels=["혼자 탑승", "2~4명 가족", "5명 이상"],
    )
    data["객실정보"] = np.where(data["Cabin"].notna(), "객실 정보 있음", "객실 정보 없음")
    data["승선항구"] = data["Embarked"].map(
        {"C": "셰르부르(C)", "Q": "퀸스타운(Q)", "S": "사우샘프턴(S)"}
    )
    data["티켓인원"] = data.groupby("Ticket")["PassengerId"].transform("size")
    data["티켓그룹"] = pd.cut(
        data["티켓인원"],
        bins=[0, 1, 2, 4, np.inf],
        labels=["1명", "2명", "3~4명", "5명 이상"],
    )

    titles = data["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    data["호칭"] = titles.where(titles.isin(["Mr", "Miss", "Mrs", "Master"]), "기타")
    return data


def style_rate_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("생존율")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, rates, counts) -> None:
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{rate:.1%}\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def rate_bar(ax: plt.Axes, summary: pd.DataFrame, title: str) -> None:
    bars = ax.bar(summary.index.astype(str), summary["mean"], color=SURVIVAL_COLOR)
    annotate_bars(ax, bars, summary["mean"], summary["count"])
    style_rate_axis(ax)
    ax.set_title(title)


def plot_h1(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("성별", observed=True)["Survived"].agg(["mean", "count"])
    summary = summary.reindex(["여성", "남성"])
    rate_bar(ax, summary, "가설 1. 성별에 따른 생존율")


def plot_h2(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("Pclass")["Survived"].agg(["mean", "count"])
    summary.index = [f"{value}등실" for value in summary.index]
    rate_bar(ax, summary, "가설 2. 객실 등급에 따른 생존율")


def plot_h3(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby(["Pclass", "성별"])["Survived"].agg(["mean", "count"])
    x = np.arange(3)
    width = 0.35
    for offset, sex, color in [(-width / 2, "여성", "#8E6CBB"), (width / 2, "남성", SECONDARY_COLOR)]:
        subset = summary.xs(sex, level="성별").reindex([1, 2, 3])
        bars = ax.bar(x + offset, subset["mean"], width, label=sex, color=color)
        annotate_bars(ax, bars, subset["mean"], subset["count"])
    ax.set_xticks(x, ["1등실", "2등실", "3등실"])
    style_rate_axis(ax)
    ax.legend()
    ax.set_title("가설 3. 성별·객실 등급별 생존율")


def plot_h4(ax: plt.Axes, data: pd.DataFrame) -> None:
    age_data = data.dropna(subset=["Age"]).copy()
    age_data["연령대"] = pd.cut(
        age_data["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=["어린이\n(0~12)", "청소년\n(13~18)", "청년\n(19~35)", "중년\n(36~60)", "고령\n(61+)"],
        include_lowest=True,
    )
    summary = age_data.groupby("연령대", observed=True)["Survived"].agg(["mean", "count"])
    rate_bar(ax, summary, "가설 4. 연령대별 생존율")


def plot_h5(ax: plt.Axes, data: pd.DataFrame) -> None:
    fare_data = data.dropna(subset=["Fare"]).copy()
    fare_data["운임구간"] = pd.qcut(
        fare_data["Fare"],
        q=4,
        labels=["하위 25%", "25~50%", "50~75%", "상위 25%"],
        duplicates="drop",
    )
    summary = fare_data.groupby("운임구간", observed=True)["Survived"].agg(["mean", "count"])
    rate_bar(ax, summary, "가설 5. 운임 수준별 생존율")


def plot_h6(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("혼자여부")["Survived"].agg(["mean", "count"])
    summary = summary.reindex(["혼자 탑승", "가족 동반"])
    rate_bar(ax, summary, "가설 6. 혼자 탑승 여부와 생존율")


def plot_h7(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("가족규모그룹", observed=True)["Survived"].agg(["mean", "count"])
    rate_bar(ax, summary, "가설 7. 가족 규모별 생존율")


def plot_h8(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("객실정보")["Survived"].agg(["mean", "count"])
    summary = summary.reindex(["객실 정보 없음", "객실 정보 있음"])
    rate_bar(ax, summary, "가설 8. 객실 정보 유무와 생존율")


def plot_h9(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.dropna(subset=["승선항구"]).groupby("승선항구")["Survived"].agg(["mean", "count"])
    summary = summary.reindex(["셰르부르(C)", "퀸스타운(Q)", "사우샘프턴(S)"])
    rate_bar(ax, summary, "가설 9. 승선 항구별 생존율")


def plot_h10(ax: plt.Axes, data: pd.DataFrame) -> None:
    for pclass, color in zip((1, 2, 3), ("#264653", "#E9C46A", "#E76F51")):
        subset = data[data["Pclass"] == pclass].copy()
        subset["등급내운임구간"] = pd.qcut(
            subset["Fare"].rank(method="first"),
            q=4,
            labels=[1, 2, 3, 4],
        )
        summary = subset.groupby("등급내운임구간", observed=True)["Survived"].agg(["mean", "count"])
        ax.plot(summary.index.astype(int), summary["mean"], marker="o", linewidth=2, label=f"{pclass}등실", color=color)
    ax.set_xticks([1, 2, 3, 4], ["하위", "중하", "중상", "상위"])
    ax.set_xlabel("각 객실 등급 안에서의 운임 사분위")
    style_rate_axis(ax)
    ax.legend()
    ax.set_title("가설 10. 같은 등실 내 운임과 생존율")


def plot_h11(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = ["Mrs", "Miss", "Master", "Mr", "기타"]
    summary = data.groupby("호칭")["Survived"].agg(["mean", "count"]).reindex(order)
    rate_bar(ax, summary, "가설 11. 승객 호칭별 생존율")


def plot_h12(ax: plt.Axes, data: pd.DataFrame) -> None:
    summary = data.groupby("티켓그룹", observed=True)["Survived"].agg(["mean", "count"])
    rate_bar(ax, summary, "가설 12. 같은 티켓을 공유한 인원수별 생존율")


PLOTS = [
    ("01_sex", plot_h1),
    ("02_pclass", plot_h2),
    ("03_sex_pclass", plot_h3),
    ("04_age", plot_h4),
    ("05_fare", plot_h5),
    ("06_alone", plot_h6),
    ("07_family_size", plot_h7),
    ("08_cabin", plot_h8),
    ("09_embarked", plot_h9),
    ("10_fare_within_class", plot_h10),
    ("11_title", plot_h11),
    ("12_ticket_group", plot_h12),
]


def save_individual_charts(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, plot_function in PLOTS:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_function(ax, data)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def create_dashboard(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(21, 24))
    for ax, (_, plot_function) in zip(axes.flat, PLOTS):
        plot_function(ax, data)
    fig.suptitle("타이타닉 데이터 분석 가설 12개 시각화", fontsize=24, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=3, w_pad=2)
    fig.savefig(DASHBOARD_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    titanic = prepare_data(CSV_PATH)
    save_individual_charts(titanic)
    create_dashboard(titanic)
    print(f"개별 차트 저장 폴더: {OUTPUT_DIR}")
    print(f"종합 대시보드 저장 완료: {DASHBOARD_PATH}")

from pathlib import Path
import csv

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "family_size_survival_chart.png"

GROUPS = ("혼자 탑승", "2~4명 가족", "5명 이상 대가족")


def classify_family_size(family_size: int) -> str:
    if family_size == 1:
        return "혼자 탑승"
    if family_size <= 4:
        return "2~4명 가족"
    return "5명 이상 대가족"


def count_survival_by_family_size(
    csv_path: Path,
) -> dict[str, dict[str, int]]:
    counts = {
        group: {"사망": 0, "생존": 0}
        for group in GROUPS
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Survived", "SibSp", "Parch"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV 파일에 'Survived', 'SibSp' 또는 'Parch' 컬럼이 없습니다."
            )

        for row in reader:
            family_size = int(row["SibSp"]) + int(row["Parch"]) + 1
            group = classify_family_size(family_size)
            status = "생존" if row["Survived"].strip() == "1" else "사망"
            counts[group][status] += 1

    return counts


def create_chart(
    counts: dict[str, dict[str, int]], output_path: Path
) -> None:
    totals = [sum(counts[group].values()) for group in GROUPS]
    survived_counts = [counts[group]["생존"] for group in GROUPS]
    deceased_counts = [counts[group]["사망"] for group in GROUPS]
    survived_rates = [
        survived / total * 100
        for survived, total in zip(survived_counts, totals)
    ]
    deceased_rates = [100 - rate for rate in survived_rates]
    positions = range(len(GROUPS))

    fig, ax = plt.subplots(figsize=(10, 7))
    survived_bars = ax.bar(
        positions,
        survived_rates,
        color="#2A9D8F",
        width=0.62,
        label="생존",
    )
    deceased_bars = ax.bar(
        positions,
        deceased_rates,
        bottom=survived_rates,
        color="#E76F51",
        width=0.62,
        label="사망",
    )

    for index, (survived_bar, deceased_bar) in enumerate(
        zip(survived_bars, deceased_bars)
    ):
        ax.text(
            survived_bar.get_x() + survived_bar.get_width() / 2,
            survived_rates[index] / 2,
            f"{survived_rates[index]:.1f}%\n({survived_counts[index]}명)",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            deceased_bar.get_x() + deceased_bar.get_width() / 2,
            survived_rates[index] + deceased_rates[index] / 2,
            f"{deceased_rates[index]:.1f}%\n({deceased_counts[index]}명)",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    x_labels = [
        f"{group}\n전체 {total}명"
        for group, total in zip(GROUPS, totals)
    ]
    ax.set_xticks(list(positions), x_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("비율 (%)", fontsize=12)
    ax.set_title(
        "타이타닉 가족 규모별 생존·사망 비율",
        fontsize=18,
        pad=16,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        "가족 규모 = SibSp(형제자매·배우자) + Parch(부모·자녀) + 본인",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    family_counts = count_survival_by_family_size(CSV_PATH)
    print(f"가족 규모별 생존 여부 집계: {family_counts}")
    create_chart(family_counts, OUTPUT_PATH)
    print(f"차트 저장 완료: {OUTPUT_PATH}")

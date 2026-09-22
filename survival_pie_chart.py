from pathlib import Path
import csv

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "survival_pie_chart.png"


def count_survival_by_sex_and_class(
    csv_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count deceased and survived passengers for each sex and class."""
    counts = {
        sex: {
            pclass: {"사망": 0, "생존": 0}
            for pclass in ("1", "2", "3")
        }
        for sex in ("female", "male")
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Survived", "Sex", "Pclass"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV 파일에 'Survived', 'Sex' 또는 'Pclass' 컬럼이 없습니다."
            )

        for row in reader:
            status = row["Survived"].strip()
            sex = row["Sex"].strip().lower()
            pclass = row["Pclass"].strip()
            if sex not in counts or pclass not in counts[sex]:
                continue

            if status == "0":
                counts[sex][pclass]["사망"] += 1
            elif status == "1":
                counts[sex][pclass]["생존"] += 1

    return counts


def create_pie_charts(
    counts: dict[str, dict[str, dict[str, int]]], output_path: Path
) -> None:
    colors = ["#E76F51", "#2A9D8F"]
    sex_titles = {"female": "여성", "male": "남성"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row_index, sex in enumerate(("female", "male")):
        for column_index, pclass in enumerate(("1", "2", "3")):
            ax = axes[row_index, column_index]
            group_counts = counts[sex][pclass]
            labels = list(group_counts.keys())
            values = list(group_counts.values())
            wedges, _, autotexts = ax.pie(
                values,
                labels=None,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 11},
            )

            legend_labels = [
                f"{label} ({value}명)"
                for label, value in zip(labels, values)
            ]
            ax.legend(
                wedges,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=2,
                fontsize=9,
            )
            ax.set_title(
                f"{sex_titles[sex]} · {pclass}등실",
                fontsize=14,
                pad=10,
            )
            ax.axis("equal")

            for text in autotexts:
                text.set_color("white")
                text.set_fontweight("bold")

    fig.suptitle("타이타닉 성별·객실 등급별 생존 여부", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    survival_counts = count_survival_by_sex_and_class(CSV_PATH)
    print(f"성별·객실 등급별 생존 여부 집계: {survival_counts}")
    create_pie_charts(survival_counts, OUTPUT_PATH)
    print(f"차트 저장 완료: {OUTPUT_PATH}")

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"

st.set_page_config(
    page_title="타이타닉 생존 분석",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the Titanic data and add labels used by the dashboard."""
    data = pd.read_csv(path)
    required = {
        "PassengerId",
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")

    data = data.copy()
    data["생존 여부"] = data["Survived"].map({0: "사망", 1: "생존"})
    data["성별"] = data["Sex"].map({"female": "여성", "male": "남성"})
    data["객실 등급"] = data["Pclass"].astype(str) + "등급"
    data["가족 규모"] = data["SibSp"] + data["Parch"] + 1
    data["탑승 항구"] = data["Embarked"].map(
        {"C": "셰르부르", "Q": "퀸스타운", "S": "사우샘프턴"}
    ).fillna("미상")
    return data


def survival_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = (
        data.groupby(group, observed=True)["Survived"]
        .agg(승객="size", 생존자="sum", 생존율="mean")
        .reset_index()
    )
    summary["생존율 표시"] = summary["생존율"].map(lambda value: f"{value:.1%}")
    return summary


def rate_chart(summary: pd.DataFrame, category: str, color: str = "#2A9D8F"):
    return (
        alt.Chart(summary)
        .mark_bar(color=color, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{category}:N", title=None, sort=None),
            y=alt.Y("생존율:Q", title="생존율", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category),
                alt.Tooltip("승객:Q", title="승객 수"),
                alt.Tooltip("생존자:Q", title="생존자 수"),
                alt.Tooltip("생존율:Q", title="생존율", format=".1%"),
            ],
        )
        .properties(height=330)
    )


try:
    titanic = load_data(DATA_PATH)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()

st.title("🚢 타이타닉 생존 분석 대시보드")
st.caption("성별, 객실 등급, 나이와 가족 규모에 따른 생존 패턴을 살펴보세요.")

with st.sidebar:
    st.header("분석 조건")
    selected_sexes = st.multiselect(
        "성별",
        options=["여성", "남성"],
        default=["여성", "남성"],
    )
    selected_classes = st.multiselect(
        "객실 등급",
        options=["1등급", "2등급", "3등급"],
        default=["1등급", "2등급", "3등급"],
    )
    known_ages = titanic["Age"].dropna()
    age_range = st.slider(
        "나이 (결측값 제외)",
        min_value=int(known_ages.min()),
        max_value=int(known_ages.max()),
        value=(int(known_ages.min()), int(known_ages.max())),
    )
    include_unknown_age = st.checkbox("나이 미상 승객 포함", value=True)
    st.divider()
    st.caption("데이터: Kaggle Titanic 학습 데이터")

age_matches = titanic["Age"].between(*age_range)
if include_unknown_age:
    age_matches = age_matches | titanic["Age"].isna()

filtered = titanic[
    titanic["성별"].isin(selected_sexes)
    & titanic["객실 등급"].isin(selected_classes)
    & age_matches
].copy()

if filtered.empty:
    st.warning("선택한 조건에 해당하는 승객이 없습니다. 왼쪽 필터를 조정해 주세요.")
    st.stop()

passenger_count = len(filtered)
survivor_count = int(filtered["Survived"].sum())
survival_rate = filtered["Survived"].mean()
average_age = filtered["Age"].mean()

metric_columns = st.columns(4)
metric_columns[0].metric("승객 수", f"{passenger_count:,}명")
metric_columns[1].metric("생존자", f"{survivor_count:,}명")
metric_columns[2].metric("생존율", f"{survival_rate:.1%}")
metric_columns[3].metric(
    "평균 나이",
    f"{average_age:.1f}세" if pd.notna(average_age) else "정보 없음",
)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("성별 생존율")
    sex_summary = survival_summary(filtered, "성별")
    st.altair_chart(rate_chart(sex_summary, "성별"), width="stretch")

with right:
    st.subheader("객실 등급별 생존율")
    class_summary = survival_summary(filtered, "객실 등급")
    st.altair_chart(rate_chart(class_summary, "객실 등급", "#457B9D"), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("나이 분포")
    age_chart = (
        alt.Chart(filtered.dropna(subset=["Age"]))
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Age:Q", bin=alt.Bin(maxbins=20), title="나이"),
            y=alt.Y("count():Q", title="승객 수"),
            color=alt.Color(
                "생존 여부:N",
                scale=alt.Scale(domain=["생존", "사망"], range=["#2A9D8F", "#E76F51"]),
                title=None,
            ),
            tooltip=[alt.Tooltip("count():Q", title="승객 수")],
        )
        .properties(height=330)
    )
    st.altair_chart(age_chart, width="stretch")

with right:
    st.subheader("가족 규모별 생존율")
    family_summary = survival_summary(filtered, "가족 규모")
    family_chart = (
        alt.Chart(family_summary)
        .mark_line(point=alt.OverlayMarkDef(size=80), color="#8E6CBB", strokeWidth=3)
        .encode(
            x=alt.X("가족 규모:O", title="함께 탑승한 가족 수 (본인 포함)"),
            y=alt.Y("생존율:Q", title="생존율", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("가족 규모:O"),
                alt.Tooltip("승객:Q", title="승객 수"),
                alt.Tooltip("생존율:Q", format=".1%"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(family_chart, width="stretch")

with st.expander("필터링된 승객 데이터 보기"):
    display_columns = [
        "PassengerId",
        "Name",
        "성별",
        "Age",
        "객실 등급",
        "Fare",
        "가족 규모",
        "탑승 항구",
        "생존 여부",
    ]
    display_data = filtered[display_columns].rename(
        columns={
            "PassengerId": "승객 ID",
            "Name": "이름",
            "Age": "나이",
            "Fare": "요금",
        }
    )
    st.dataframe(display_data, width="stretch", hide_index=True)
    st.download_button(
        "CSV 다운로드",
        data=display_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="titanic_filtered.csv",
        mime="text/csv",
    )


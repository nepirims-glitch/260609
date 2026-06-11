
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="식물 성장 시뮬레이터", layout="wide")

st.title("🌱 식물 성장 시뮬레이터")
st.write("빛, 온도, 습도, CO₂ 농도를 조절하여 예상 식물 키를 예측합니다.")

@st.cache_resource
def train_model():
    df = pd.read_csv("plant_growth_dataset_10000.csv")

    X = df[["Light_PPFD", "Temperature_C", "Humidity_pct", "CO2_ppm"]]
    y = df["Plant_Height_cm"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))

    return model, score, df

model, score, df = train_model()

st.sidebar.header("환경 조건 설정")

light = st.sidebar.slider(
    "빛의 세기 (PPFD)",
    int(df["Light_PPFD"].min()),
    int(df["Light_PPFD"].max()),
    int(df["Light_PPFD"].mean())
)

temp = st.sidebar.slider(
    "온도 (°C)",
    float(df["Temperature_C"].min()),
    float(df["Temperature_C"].max()),
    float(df["Temperature_C"].mean())
)

humidity = st.sidebar.slider(
    "습도 (%)",
    float(df["Humidity_pct"].min()),
    float(df["Humidity_pct"].max()),
    float(df["Humidity_pct"].mean())
)

co2 = st.sidebar.slider(
    "CO₂ 농도 (ppm)",
    int(df["CO2_ppm"].min()),
    int(df["CO2_ppm"].max()),
    int(df["CO2_ppm"].mean())
)

sample = pd.DataFrame([{
    "Light_PPFD": light,
    "Temperature_C": temp,
    "Humidity_pct": humidity,
    "CO2_ppm": co2
}])

predicted_height = model.predict(sample)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("예상 식물 키", f"{predicted_height:.2f} cm")

with col2:
    st.metric("모델 정확도 (R²)", f"{score:.3f}")

st.subheader("현재 입력값")

st.dataframe(sample, use_container_width=True)

st.subheader("데이터 미리보기")
st.dataframe(df.head(), use_container_width=True)

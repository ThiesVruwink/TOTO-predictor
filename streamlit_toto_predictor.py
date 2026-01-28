import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Pagina instellingen (voor mobiele weergave)
st.set_page_config(page_title="TOTO Predictor", page_icon="⚽")

st.title("⚽ TOTO Value Checker")
st.write("Bereken de 'Fair Odds' op basis van de laatste 3 Eredivisie seizoenen.")
st.write("Auteur: Thies Vruwink")


# 1. DATA LADEN (met caching zodat de app snel blijft)
@st.cache_data
def get_data():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/N1.csv"
    ]
    all_data = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
            all_data.append(df)
        except:
            continue
    return pd.concat(all_data, ignore_index=True)


df = get_data()
teams = sorted(df['HomeTeam'].unique())

# 2. INPUTS
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Thuis Team", teams, index=teams.index("Ajax") if "Ajax" in teams else 0)
with col2:
    away_team = st.selectbox("Uit Team", teams, index=teams.index("Feyenoord") if "Feyenoord" in teams else 1)


# 3. LOGICA
def calculate_probs(home, away, data):
    avg_h = data['FTHG'].mean()
    avg_a = data['FTAG'].mean()

    h_games = data[data['HomeTeam'] == home]
    a_games = data[data['AwayTeam'] == away]

    h_att = h_games['FTHG'].mean() / avg_h
    h_def = h_games['FTAG'].mean() / avg_a
    a_att = a_games['FTAG'].mean() / avg_a
    a_def = a_games['FTHG'].mean() / avg_h

    l_h = h_att * a_def * avg_h
    l_a = a_att * h_def * avg_a

    h_p = [poisson.pmf(i, l_h) for i in range(10)]
    a_p = [poisson.pmf(i, l_a) for i in range(10)]

    m = np.outer(h_p, a_p)
    return np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))


# 4. RESULTATEN TONEN
if st.button("Bereken Kansen", use_container_width=True):
    w, d, l = calculate_probs(home_team, away_team, df)

    st.divider()

    # Gebruik kolommen voor een nette mobiele weergave
    c1, c2, c3 = st.columns(3)
    c1.metric("1 (Thuis)", f"{1 / w:.2f}", f"{w:.0%}")
    c2.metric("X (Gelijk)", f"{1 / d:.2f}", f"{d:.0%}")
    c3.metric("2 (Uit)", f"{1 / l:.2f}", f"{l:.0%}")

    st.info("'Fair Odds' worden weergegeven.")
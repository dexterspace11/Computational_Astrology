import numpy as np
import pandas as pd
from skyfield.api import load
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import ollama

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
ASPECTS = {'conj':0,'sext':60,'sq':90,'trine':120,'opp':180}
PLANET_ORBS = {'SU':3,'MO':1,'ME':3,'VE':3,'MA':2,'JU':3,'SA':2,'UR':2,'NE':2,'PL':2}
TRANSIT_WEIGHTS = {'SU':2,'MO':1,'ME':1.2,'VE':1.5,'MA':1.8,'JU':2.5,'SA':3,'UR':3.5,'NE':3.2,'PL':4}
OUTER_PLANETS = ['JU','SA','UR','NE','PL']
PLANETS = {
    'SU':'sun','MO':'moon','ME':'mercury','VE':'venus','MA':'mars',
    'JU':'jupiter barycenter','SA':'saturn barycenter',
    'UR':'uranus barycenter','NE':'neptune barycenter','PL':'pluto barycenter'
}
PSYCHO_MAP = {
    "structural_pressure": ['SA','PL'],
    "impulsivity": ['MA','UR'],
    "idealism_confusion": ['NE','JU'],
    "ego_activation": ['SU','MA'],
    "emotional_volatility": ['MO','UR']
}

# ────────────────────────────────────────────────
# LOAD EPHEMERIS
# ────────────────────────────────────────────────
eph = load('de421.bsp')
ts = load.timescale()

# ────────────────────────────────────────────────
# CORE UTILITIES
# ────────────────────────────────────────────────
def angular_distance(a, b):
    diff = np.abs(a - b) % 360
    return min(diff, 360 - diff)

def dynamic_orb(transit, natal):
    base = min(PLANET_ORBS[transit], PLANET_ORBS[natal])
    if transit in OUTER_PLANETS and natal in OUTER_PLANETS:
        return min(base, 2)
    return base

def get_planet_longitudes(target_date):
    t = ts.utc(target_date.year, target_date.month, target_date.day)
    chart = {}
    earth = eph['earth']
    for code, planet_name in PLANETS.items():
        planet = eph[planet_name]
        astrometric = earth.at(t).observe(planet)
        lon, lat, distance = astrometric.ecliptic_latlon()
        chart[code] = lon.degrees
    return chart

def detect_retrograde(date):
    today = get_planet_longitudes(date)
    tomorrow = get_planet_longitudes(date + timedelta(days=1))
    retro = {}
    for p in PLANETS.keys():
        diff = (tomorrow[p] - today[p]) % 360
        retro[p] = diff > 180
    return retro

# ────────────────────────────────────────────────
# TRANSIT DETECTION
# ────────────────────────────────────────────────
def compute_daily_transits(natal_chart, date):
    transit_chart = get_planet_longitudes(date)
    retro_flags = detect_retrograde(date)
    results = []
    for t_planet, t_lon in transit_chart.items():
        for n_planet, n_lon in natal_chart.items():
            angle = angular_distance(t_lon, n_lon)
            for asp_name, ideal in ASPECTS.items():
                orb = dynamic_orb(t_planet, n_planet)
                delta = angle - ideal
                if abs(delta) <= orb:
                    closeness = 1 - (abs(delta) / orb)
                    weight = TRANSIT_WEIGHTS[t_planet]
                    if retro_flags[t_planet]:
                        weight *= 1.15
                    if t_planet in OUTER_PLANETS:
                        weight *= 1.3
                    intensity = closeness * weight
                    results.append({
                        "date": date,
                        "transit": t_planet,
                        "natal": n_planet,
                        "aspect": asp_name,
                        "intensity": round(intensity, 3),
                        "retrograde": retro_flags[t_planet]
                    })
    return results

# ────────────────────────────────────────────────
# WINDOW SCAN
# ────────────────────────────────────────────────
def scan_transit_window(birth_date_str, start_date_str, days=365):
    natal_chart = get_planet_longitudes(datetime.strptime(birth_date_str,"%Y-%m-%d"))
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d")
    all_results = []
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        daily = compute_daily_transits(natal_chart, current_date)
        all_results.extend(daily)
    return pd.DataFrame(all_results)

# ────────────────────────────────────────────────
# STACKING INDEX
# ────────────────────────────────────────────────
def compute_stacking_index(df):
    stacking = df.groupby("date")["intensity"].sum()
    norm = (stacking - stacking.min()) / (stacking.max() - stacking.min())
    return norm * 100

# ────────────────────────────────────────────────
# PSYCHOLOGICAL INDEX LAYER
# ────────────────────────────────────────────────
def compute_psychological_indices(df):
    daily = df.groupby("date")
    index_df = pd.DataFrame(index=daily.size().index)
    for category, planets in PSYCHO_MAP.items():
        subset = df[df["transit"].isin(planets)]
        index_df[category] = subset.groupby("date")["intensity"].sum()
    return index_df.fillna(0)

# ────────────────────────────────────────────────
# FINANCIAL REGIME MODEL
# ────────────────────────────────────────────────
def compute_financial_regime(psych_df):
    regime = pd.DataFrame(index=psych_df.index)
    regime["volatility_score"] = psych_df["impulsivity"] + psych_df["emotional_volatility"]
    regime["contraction_score"] = psych_df["structural_pressure"]
    regime["euphoria_score"] = psych_df["idealism_confusion"]
    return regime

# ────────────────────────────────────────────────
# ALERT GENERATOR – MONTHLY TOP DAYS
# ────────────────────────────────────────────────
def generate_monthly_alerts(psych_df, top_n=3):
    alerts = []
    df = psych_df.copy()
    df['month'] = df.index.month
    for month in sorted(df['month'].unique()):
        month_data = df[df['month']==month]
        for category in PSYCHO_MAP.keys():
            top_days = month_data[category].sort_values(ascending=False).head(top_n)
            for day, intensity in top_days.items():
                alerts.append({
                    "month": month,
                    "category": category,
                    "date": day,
                    "intensity": round(intensity,3)
                })
    return pd.DataFrame(alerts)

# ────────────────────────────────────────────────
# VISUALIZATIONS
# ────────────────────────────────────────────────
def plot_psychological_heatmap(alerts_df):
    pivot = alerts_df.pivot_table(index='category', columns='month', values='intensity', aggfunc='max')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label':'Intensity'}, ax=ax)
    ax.set_title("Monthly Psychological Peaks Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Psychological Category")
    return fig

def plot_psychological_indices(psych_df):
    fig, ax = plt.subplots(figsize=(14,6))
    for col in psych_df.columns:
        ax.plot(psych_df.index, psych_df[col], label=col)
    ax.set_title("Behavioral/Psychological Indices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True)
    return fig

# ────────────────────────────────────────────────
# MASTER ENGINE FUNCTION
# ────────────────────────────────────────────────
def professional_behavioral_financial_engine_v8(birth_date, forecast_start, forecast_days=365, top_n_alerts=3):
    print("\n══════════════════════════════════════════════")
    print("🔮 PROFESSIONAL TRANSIT ENGINE v8.0")
    print("Behavioral Forecasting Tool + Monthly Alerts + Heatmap")
    print("══════════════════════════════════════════════\n")

    df = scan_transit_window(birth_date, forecast_start, forecast_days)
    stacking = compute_stacking_index(df)
    psych = compute_psychological_indices(df)
    regime = compute_financial_regime(psych)
    alerts = generate_monthly_alerts(psych, top_n=top_n_alerts)

    print("🔥 Top 10 High Intensity Days:")
    print(stacking.sort_values(ascending=False).head(10))

    print("\n🧠 Psychological Peaks:")
    print(psych.sum().sort_values(ascending=False))

    print("\n📈 Regime Model Snapshot:")
    print(regime.head())

    print("\n📣 Monthly Psychological Alerts:")
    print(alerts)

    fig_indices = plot_psychological_indices(psych)
    fig_heatmap = plot_psychological_heatmap(alerts)

    return df, stacking, psych, regime, alerts, fig_indices, fig_heatmap

# ────────────────────────────────────────────────
# OLLAMA INTERPRETATION (streamed)
# ────────────────────────────────────────────────
def ollama_interpret_results(stacking, psych, regime, alerts):
    top_days_str = stacking.sort_values(ascending=False).head(10).to_string()
    psych_peaks_str = psych.sum().sort_values(ascending=False).to_string()
    regime_head_str = regime.head().to_string()
    alerts_str = alerts.head(20).to_string()

    prompt = f"""
You are an expert astro-psychologist with deep knowledge of archetypal astrology and behavioral patterns. Interpret the following transit forecast results for a natal chart in a clear, compassionate, and balanced way. Avoid fatalism or determinism; emphasize awareness, potential growth, and free will.

Key results:

Top 10 High Intensity Days (normalized stacking index 0-100):
{top_days_str}

Psychological Peaks (total intensity over the period):
{psych_peaks_str}

Regime Model Snapshot (first few days):
{regime_head_str}

Sample Monthly Psychological Alerts (first 20 rows):
{alerts_str}

Write a concise 400-600 word narrative interpretation:
1. Overall psychological atmosphere and dominant themes of the period
2. Key high-pressure windows and what they may represent emotionally or behaviorally
3. How the strongest categories (e.g. idealism_confusion, impulsivity) might manifest
4. Practical awareness points or suggestions for navigating the period constructively
"""

    try:
        response = ollama.chat(
            model='llama3.1:8b',  # ← upgraded from 1b to 8b
            messages=[
                {'role': 'system', 'content': 'You are a wise, compassionate astro-psychologist.'},
                {'role': 'user', 'content': prompt}
            ],
            stream=True
        )

        full_response = ""
        for chunk in response:
            content = chunk['message']['content']
            full_response += content
            yield content  # for Streamlit streaming

        return full_response

    except Exception as e:
        yield f"Ollama interpretation failed: {str(e)}\n\nMake sure Ollama is running with 'ollama run llama3.1:8b'"

# ────────────────────────────────────────────────
# STREAMLIT APP
# ────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="PsychoTransit v8", layout="wide")
    st.title("🔮 PsychoTransit v8 – Behavioral Transit Forecasting Engine")
    st.markdown("""
    Computes astrological transits to a natal chart and maps psychological pressure patterns.  
    All core algorithms are unchanged. Ollama (now using Llama 3.1 8B) provides a streamed narrative interpretation.
    """)

    # Input fields
    birth_date = st.text_input("Birth Date (YYYY-MM-DD)", value="1941-05-24")
    forecast_start = st.text_input("Forecast Start Date (YYYY-MM-DD)", value="2026-01-01")
    forecast_days = st.number_input("Forecast Duration (days)", min_value=30, max_value=1095, value=365)
    top_n_alerts = st.number_input("Top N alerts per category per month", min_value=1, max_value=10, value=5)

    if st.button("Run Analysis"):
        with st.spinner("Computing transits..."):
            try:
                df, stacking, psych, regime, alerts, fig_indices, fig_heatmap = professional_behavioral_financial_engine_v8(
                    birth_date=birth_date,
                    forecast_start=forecast_start,
                    forecast_days=forecast_days,
                    top_n_alerts=top_n_alerts
                )

                # Text outputs
                st.subheader("Top 10 High Intensity Days")
                st.text(stacking.sort_values(ascending=False).head(10).to_string())

                st.subheader("Psychological Peaks (total intensity)")
                st.text(psych.sum().sort_values(ascending=False).to_string())

                st.subheader("Regime Model Snapshot (first 5 days)")
                st.text(regime.head().to_string())

                st.subheader("Monthly Psychological Alerts (first 20)")
                st.dataframe(alerts.head(20))

                # Plots
                st.subheader("Psychological Indices Over Time")
                st.pyplot(fig_indices)

                st.subheader("Monthly Psychological Peaks Heatmap")
                st.pyplot(fig_heatmap)

                # Ollama streamed interpretation
                st.subheader("🧘 Ollama Psychological Interpretation (Llama 3.1 8B)")
                interpretation_placeholder = st.empty()

                try:
                    stream_gen = ollama_interpret_results(stacking, psych, regime, alerts)
                    full_text = ""
                    for chunk in stream_gen:
                        full_text += chunk
                        interpretation_placeholder.markdown(full_text + "▌")
                    interpretation_placeholder.markdown(full_text)  # final clean output
                except Exception as e:
                    st.error(f"Ollama interpretation failed: {e}\n\nMake sure Ollama is running locally with 'ollama run llama3.1:8b'")

            except Exception as e:
                st.error(f"Error during computation: {e}")

if __name__ == "__main__":
    main()
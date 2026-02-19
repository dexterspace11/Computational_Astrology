import numpy as np
import pandas as pd
from skyfield.api import load
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

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
# HEATMAP VISUALIZATION (FIXED)
# ────────────────────────────────────────────────
def plot_psychological_heatmap(alerts_df):
    # Use pivot_table with aggregation to avoid duplicates
    pivot = alerts_df.pivot_table(index='category', columns='month', values='intensity', aggfunc='max')
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label':'Intensity'})
    plt.title("Monthly Psychological Peaks Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Psychological Category")
    plt.show()

# ────────────────────────────────────────────────
# TIME-SERIES PLOT
# ────────────────────────────────────────────────
def plot_psychological_indices(psych_df):
    plt.figure(figsize=(14,6))
    for col in psych_df.columns:
        plt.plot(psych_df.index, psych_df[col], label=col)
    plt.title("Behavioral/Psychological Indices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

# ────────────────────────────────────────────────
# MASTER ENGINE v8.0 (CORRECTED)
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

    # Visualizations
    plot_psychological_indices(psych)
    plot_psychological_heatmap(alerts)

    return df, stacking, psych, regime, alerts

# ────────────────────────────────────────────────
# EXAMPLE USAGE
# ────────────────────────────────────────────────
df, stacking, psych, regime, alerts = professional_behavioral_financial_engine_v8(
    birth_date="1975-06-11",
    forecast_start="2026-01-01",
    forecast_days=30,   # weekly/monthly flexibility
    top_n_alerts=3      # top N days per category per month
)

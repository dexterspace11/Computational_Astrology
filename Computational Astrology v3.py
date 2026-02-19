import numpy as np
import pandas as pd
from skyfield.api import load
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

ASPECTS = {'conj':0,'sext':60,'sq':90,'trine':120,'opp':180}

# PROFESSIONAL TIGHT ORBS
PLANET_ORBS = {
    'SU':3,'MO':1,'ME':3,'VE':3,'MA':2,
    'JU':3,'SA':2,'UR':2,'NE':2,'PL':2
}

TRANSIT_WEIGHTS = {
    'SU':2,'MO':1,'ME':1.2,'VE':1.5,'MA':1.8,
    'JU':2.5,'SA':3,'UR':3.5,'NE':3.2,'PL':4
}

OUTER_PLANETS = ['JU','SA','UR','NE','PL']

PLANETS = {
    'SU':'sun','MO':'moon','ME':'mercury','VE':'venus','MA':'mars',
    'JU':'jupiter barycenter','SA':'saturn barycenter',
    'UR':'uranus barycenter','NE':'neptune barycenter','PL':'pluto barycenter'
}

# ────────────────────────────────────────────────
# LOAD EPHEMERIS
# ────────────────────────────────────────────────
eph = load('de421.bsp')
ts = load.timescale()

# ────────────────────────────────────────────────
# CORE UTILITIES
# ────────────────────────────────────────────────

def angular_distance(a,b):
    diff = np.abs(a-b) % 360
    return min(diff,360-diff)

def dynamic_orb(transit, natal):
    base = min(PLANET_ORBS[transit], PLANET_ORBS[natal])
    if transit in OUTER_PLANETS and natal in OUTER_PLANETS:
        return min(base,2)
    return base

def get_planet_longitudes(target_date):
    t = ts.utc(target_date.year, target_date.month, target_date.day)
    chart = {}
    earth = eph['earth']
    for code,planet_name in PLANETS.items():
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
# PROFESSIONAL TRANSIT DETECTION
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

                    closeness = 1 - (abs(delta)/orb)
                    weight = TRANSIT_WEIGHTS[t_planet]

                    # Retrograde amplification
                    if retro_flags[t_planet]:
                        weight *= 1.15

                    # Outer planet amplification
                    if t_planet in OUTER_PLANETS:
                        weight *= 1.3

                    intensity = closeness * weight

                    results.append({
                        "date":date,
                        "transit":t_planet,
                        "natal":n_planet,
                        "aspect":asp_name,
                        "orb":orb,
                        "orb_distance":round(abs(delta),3),
                        "intensity":round(intensity,3),
                        "retrograde":retro_flags[t_planet]
                    })

    return results

# ────────────────────────────────────────────────
# APPLYING / SEPARATING
# ────────────────────────────────────────────────

def classify_phase(natal_chart, date, record):

    today = get_planet_longitudes(date)
    tomorrow = get_planet_longitudes(date + timedelta(days=1))

    t = record["transit"]
    n = record["natal"]
    ideal = ASPECTS[record["aspect"]]

    today_angle = angular_distance(today[t], natal_chart[n])
    tomorrow_angle = angular_distance(tomorrow[t], natal_chart[n])

    if abs(tomorrow_angle - ideal) < abs(today_angle - ideal):
        return "Applying"
    return "Separating"

# ────────────────────────────────────────────────
# WINDOW SCAN
# ────────────────────────────────────────────────

def scan_transit_window(birth_date_str, start_date_str, days=180):

    natal_chart = get_planet_longitudes(datetime.strptime(birth_date_str,"%Y-%m-%d"))
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d")

    all_results = []

    for d in range(days):
        current_date = start_date + timedelta(days=d)
        daily = compute_daily_transits(natal_chart, current_date)

        for record in daily:
            record["phase"] = classify_phase(natal_chart, current_date, record)
            all_results.append(record)

    df = pd.DataFrame(all_results)
    return df

# ────────────────────────────────────────────────
# WEIGHTED STACKING INDEX
# ────────────────────────────────────────────────

def compute_stacking_index(df):

    stacking = df.groupby("date")["intensity"].sum()

    # Normalize to 0–100 scale
    norm = (stacking - stacking.min()) / (stacking.max() - stacking.min())
    index = norm * 100

    return index.sort_values(ascending=False)

# ────────────────────────────────────────────────
# OUTER PLANET LIFE CYCLE FILTER
# ────────────────────────────────────────────────

def filter_major_cycles(df):
    return df[df["transit"].isin(OUTER_PLANETS)]

# ────────────────────────────────────────────────
# PROBABILISTIC TIMING MODEL
# ────────────────────────────────────────────────

def compute_volatility_probability(df):

    daily_score = df.groupby("date")["intensity"].sum()

    mean = daily_score.mean()
    std = daily_score.std()

    z_scores = (daily_score - mean) / std

    probability = 1 / (1 + np.exp(-z_scores))  # sigmoid

    return probability.sort_values(ascending=False)

# ────────────────────────────────────────────────
# MASTER FORECAST
# ────────────────────────────────────────────────

def full_professional_forecast(birth_date, forecast_start, forecast_days=180):

    print("\n══════════════════════════════════════════════")
    print("🔮 PROFESSIONAL TRANSIT ENGINE v4.0")
    print("══════════════════════════════════════════════\n")

    df = scan_transit_window(birth_date, forecast_start, forecast_days)

    if df.empty:
        print("No significant transits.")
        return

    # A) Cleaned realistic stacking
    stacking_index = compute_stacking_index(df)

    # B) Outer-planet life cycles
    major_cycles = filter_major_cycles(df)

    # C) Probabilistic model
    probability = compute_volatility_probability(df)

    print("🔥 Top 10 High-Intensity Days (Stacking Index):")
    print(stacking_index.head(10))

    print("\n🌊 Highest Volatility Probability Days:")
    print(probability.head(10))

    print("\n🪐 Major Outer-Planet Cycles:")
    print(major_cycles.sort_values("intensity",ascending=False).head(20))

    print("\n══════════════════════════════════════════════\n")

    return df, stacking_index, probability

# ────────────────────────────────────────────────
# EXAMPLE USAGE
# ────────────────────────────────────────────────

df, stacking_index, probability = full_professional_forecast(
    birth_date="1975-06-11",
    forecast_start="2026-01-01",
    forecast_days=180
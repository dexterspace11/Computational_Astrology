PsychoTransit v8
What it does

PsychoTransit v8 is an interactive, open-source astro-psychological transit analysis tool that:Calculates precise geocentric tropical transits from a given natal birth date to a future forecast period
Detects major aspects (conjunction, sextile, square, trine, opposition) with dynamic orbs and retrograde weighting
Computes daily transit intensity scores, aggregates them into a normalized stacking index (0–100) showing overall pressure peaks
Maps intensities into five archetypal psychological categories:structural_pressure (Saturn-Pluto: restriction, transformation, entrapment)
impulsivity (Mars-Uranus: sudden action, rebellion)
idealism_confusion (Neptune-Jupiter: vision, spirituality, illusion/delusion)
ego_activation (Sun-Mars: drive, charisma, wounds)
emotional_volatility (Moon-Uranus: mood swings, shocks)

Generates a simple regime model (volatility, contraction, euphoria scores)
Produces monthly alerts ranking the top pressure days per category
Visualizes results with time-series plots and heatmaps
Uses Ollama (default: Llama 3.1 8B) to automatically generate a streamed, narrative psychological interpretation of the results in natural language

The tool is designed to help users explore potential emotional, behavioral, and psychological pressure windows without any financial/trading overlay — purely as an astro-psychology research and self-awareness instrument.How it worksInput
User provides:Natal birth date (YYYY-MM-DD)
Forecast start date (YYYY-MM-DD)
Duration (days, default 365)
Number of top alerts per category/month (default 5)

Transit Calculation  
Uses Skyfield + DE421 ephemeris to compute accurate longitudes for 10 planets
Detects retrograde motion by comparing consecutive days
Computes angular distances and identifies aspects within dynamic orbs (tighter for outer-to-outer pairs)
Scores each valid transit-aspect pair with closeness × planet weight × retrograde multiplier × outer-planet boost

Aggregation & Mapping  
Daily stacking index: sum of all transit intensities, normalized 0–100
Psychological categories: intensities grouped by archetypal planet pairs
Regime model: simple sums (volatility = impulsivity + emotional_volatility, etc.)
Monthly alerts: top N days per category

Visualization  
Time-series plot of daily psychological index scores
Heatmap of monthly peak intensities per category

LLM Interpretation  
Key results (top days, peaks, regime snapshot, alerts) are fed to Ollama (Llama 3.1 8B)
Model generates a 400–600 word narrative focusing on dominant themes, key windows, category manifestations, and constructive navigation advice
Output streams live in the app (no waiting for full response)

All core astrological algorithms (orbs, weights, retrograde detection, aspect logic) remain unchanged from the original non-Streamlit version.

How to install

1.Python dependencies

pip install numpy pandas skyfield matplotlib seaborn streamlit ollama

2. Install and run Ollama (required for interpretation)Ollama runs large language models locally.
   Download & install Ollama: https://ollama.com/download
Open a terminal and pull/run the 8B model (recommended):

ollama pull llama3.1:8b
ollama run llama3.1:8b

Keep this terminal open (or run Ollama as a service/background process).
Note: 8B needs ~16 GB RAM/VRAM for smooth performance. 
If your machine is limited, use llama3.2:3b instead (change model name in script).

3. Download ephemeris file (if not already present)
The script uses de421.bsp (automatically downloaded by Skyfield on first run). If you prefer a different file (e.g., de440.bsp for wider range), download from https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ and update the load() line.How to useSave the script as psychotransit_app.py
Make sure Ollama is running with Llama 3.1 8B

Launch the app:

streamlit run Computational_Astrology_v8_8b.py

Open the browser URL shown (usually http://localhost:8501)
Enter:Birth date (e.g., 1941-05-24 for Bob Dylan)
Forecast start (e.g., 2026-01-01)
Duration (default 365 days)
Top N alerts (default 5)

Click Run Analysis
View:Top intensity days
Psychological peaks
Regime snapshot
Monthly alerts table
Time-series plot & heatmap
Streamed Ollama narrative interpretation

If Ollama is not running, the interpretation section shows an error but all other results still display.

Breakdown of Analysis & Interpretation
Raw Transit Data
Skyfield computes exact positions → detects aspects → scores intensity with dynamic orbs, retrograde, and outer-planet boosts.
Stacking Index
Daily sum of all transit intensities → normalized 0–100. Higher = more overlapping pressure/transformation energy.
Psychological Categories
Intensities grouped into five archetypes (e.g., idealism_confusion = Neptune-Jupiter total). Highest score = dominant theme.
Regime Model
Simple derived scores:Volatility = impulsivity + emotional_volatility
Contraction = structural_pressure
Euphoria = idealism_confusion

Monthly Alerts
Top N days per category per month — flags potential high-impact dates.
Ollama Interpretation
All key outputs are fed to Llama 3.1 8B with a structured prompt. It generates a narrative that:
Identifies dominant themes & planetary archetypes
Prioritizes major windows/clusters
Suggests constructive awareness & navigation strategies
Emphasizes free will and growth

The result is a complete, self-contained web app: precise astrological computation + visual summaries + natural-language psychological reading — all running locally.Enjoy!


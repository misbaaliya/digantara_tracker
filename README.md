# Space Object Tracker

Ground-based tracking analysis for LEO space objects using SGP4 propagation. Tracks satellite passes over a ground station, determines visibility windows, and generates a diagnostic plot.

Currently configured for **NORAD 63223 (2025-052P)** observed from **Svalbard, Norway**.

## What it does

The script propagates a satellite orbit using its TLE and checks three conditions every 30 seconds over a 24-hour window:

1. **Elevation** — Is the object above the station's field of regard? (≥ 20°)
2. **Sunlit** — Is the object illuminated by the Sun? (cylindrical shadow model)
3. **Station night** — Is it dark enough at the ground station? (Sun below −18°)

When all three are true simultaneously, the object is considered **visible/detectable**.

Transition times (ingress/egress) are refined to ≤ 1 second using bisection search.

## Setup
**1. Get the repository:**

Clone using Git:
```
git clone https://github.com/misbaaliya/digantara_tracker.git
cd digantara_tracker
```

Or download as ZIP from [https://github.com/misbaaliya/digantara_tracker](https://github.com/misbaaliya/digantara_tracker) → click **Code** → **Download ZIP**, then extract and open the folder.

**2. Install dependencies** (Python 3.8+ required):

```
pip install skyfield numpy matplotlib
```

The script will automatically download the DE421 planetary ephemeris (~17 MB) on first run if it's not already in the working directory.

## Usage

```
python tracker.py
```

That's it. After running, you'll find two files in the same directory as the script:

| File | Description |
|------|-------------|
| `tracking_report.txt` | Full text report with crossing events, visibility windows, and condition summary |
| `tracking_analysis.png` | Two-panel plot showing satellite elevation and solar elevation over the 24-hour window |

The same report is also printed to the console.

## Configuration

Everything is set at the top of the script. To track a different object or use a different station, just edit these values:

```python
TLE_NAME = "NORAD-63223 (2025-052P)"
TLE_LINE1 = "1 63223U 25052P   25244.59601767 ..."
TLE_LINE2 = "2 63223  97.4217 137.0451 ..."

STATION_LAT = 78.9066
STATION_LON = 11.88916
STATION_ALT = 380.0
```

## Assumptions

| # | Assumption |
|---|-----------|
| 1 | FOR = 70° half-cone from zenith → minimum elevation = 20° |
| 2 | Station night = Sun ≤ −18° (astronomical twilight) |
| 3 | Cylindrical Earth shadow model |
| 4 | Analysis window = 24 hours starting from TLE epoch |
| 5 | 30 s coarse grid, bisection-refined to ≤ 1 s |
| 6 | SGP4 propagator via skyfield |
| 7 | JPL DE421 ephemeris, TEME → GCRS conversion handled by skyfield |

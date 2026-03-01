import os
import sys
import numpy as np

try:
    from skyfield.api import load, EarthSatellite, wgs84, Loader
except ImportError:
    sys.exit("Skyfield not installed")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Matplotlib not found")

# Config

TLE_NAME = "NORAD-63223 (2025-052P)"
TLE_LINE1 = "1 63223U 25052P   25244.59601767  .00010814  00000-0  51235-3 0  9991"
TLE_LINE2 = "2 63223  97.4217 137.0451 0006365  74.2830 285.9107 15.19475170 25990"

STATION_LAT = 78.9066
STATION_LON = 11.88916
STATION_ALT = 380.0

FOR_DEG = 70.0
MIN_ELEVATION = 90.0 - FOR_DEG

NIGHT_SUN_ELEV = -18.0

STEP_SEC = 30
DURATION_SEC = 86400
REFINE_TOL = 1.0

_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(_DIR, "tracking_report.txt")
PLOT_PATH = os.path.join(_DIR, "tracking_analysis.png")


#Resource Loading

print("=" * 70)
print("Space Object Tracking - Ground Based Sensor")
print("=" * 70)

_loader = Loader(_DIR)
try:
    eph = _loader("de421.bsp")
    print("\nEphemeris (DE421) loaded from directory.")
except Exception:
    print("\nde421.bsp not found. ")
    eph = load("de421.bsp")
    print("Ephemeris downloaded and loaded.")

ts = load.timescale()
satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, TLE_NAME, ts)
station = wgs84.latlon(STATION_LAT, STATION_LON, elevation_m=STATION_ALT)

epoch_jd = satellite.epoch.tt
epoch_utc = satellite.epoch.utc_iso()
epoch_display = epoch_utc[:19] + " UTC"

end_jd = epoch_jd + DURATION_SEC / 86400.0
end_t = ts.tt_jd(end_jd)
end_display = end_t.utc_iso()[:19] + " UTC"

print(f"\nTarget: {TLE_NAME}")
print(f"Station: {STATION_LAT} N, {STATION_LON} E, {STATION_ALT} m (Svalbard, Norway)")
print(f"Window: {epoch_display}  to  {end_display}")
print(f"Duration: 24 hours from TLE epoch\n")


# Propagation

print("Propagating orbit on 30-second grid.")

n_steps = int(DURATION_SEC / STEP_SEC) + 1
sec_grid = np.arange(n_steps, dtype=float) * STEP_SEC

times_all = ts.tt_jd(epoch_jd + sec_grid / 86400.0)

topo = (satellite - station).at(times_all)
alt, _, rng = topo.altaz()
elev_deg = alt.degrees
range_km = rng.km

sunlit = satellite.at(times_all).is_sunlit(eph)

obs_pos = (eph["earth"] + station).at(times_all)
sun_apparent = obs_pos.observe(eph["sun"]).apparent()
sun_alt, _,_ = sun_apparent.altaz()
sun_elev_deg = sun_alt.degrees

print("Propagation complete.\n")


# Boolean conditions

in_FOR = elev_deg >= MIN_ELEVATION
is_sunlit = sunlit
station_night = sun_elev_deg <= NIGHT_SUN_ELEV
is_visible = in_FOR & is_sunlit & station_night


# Helpers
def _eval_elev(t):
    """Satellite elevation in degrees at a given time."""
    a, _, _ = (satellite - station).at(t).altaz()
    return a.degrees

def _eval_visible(t):
    """True if all three detection conditions are met."""
    if _eval_elev(t) < MIN_ELEVATION:
        return False
    sl = satellite.at(t).is_sunlit(eph)
    s_alt = (eph["earth"] + station).at(t).observe(eph["sun"]).apparent().altaz()[0]
    return bool(sl) and (s_alt.degrees <= NIGHT_SUN_ELEV)

def _bisect(t_false, t_true, condition_fn, tol_sec=REFINE_TOL):
    """Binary search for a False to True transition."""
    a = t_false.tt
    b = t_true.tt
    tol_jd = tol_sec / 86400.0
    while (b - a) > tol_jd:
        mid = (a + b) / 2.0
        if condition_fn(ts.tt_jd(mid)):
            b = mid
        else:
            a = mid
    return ts.tt_jd((a + b) / 2.0)


# Window Detection

def find_windows(bool_arr, times_arr, condition_fn=None):
    """Detect contiguous True blocks and refine edges."""
    windows = []
    in_win = False
    s_idx = 0

    for i, val in enumerate(bool_arr):
        if val and not in_win:
            in_win = True
            s_idx = i
        elif not val and in_win:
            in_win = False
            e_idx = i - 1

            if condition_fn and s_idx > 0:
                t_start = _bisect(
                    times_arr[s_idx - 1], times_arr[s_idx], condition_fn
                )
            else:
                t_start = times_arr[s_idx]

            if condition_fn and i < len(bool_arr):
                t_end = _bisect(
                    times_arr[e_idx], times_arr[i],
                    lambda t: not condition_fn(t)
                )
            else:
                t_end = times_arr[e_idx]

            dur = (t_end.tt - t_start.tt) * 86400.0
            windows.append(dict(
                start_t = t_start,
                end_t = t_end,
                duration_sec = dur,
                start_idx = s_idx,
                end_idx = e_idx,
            ))

    if in_win:
        e_idx = len(bool_arr) - 1
        windows.append(dict(
            start_t = times_arr[s_idx],
            end_t = times_arr[e_idx],
            duration_sec = (times_arr[e_idx].tt - times_arr[s_idx].tt) * 86400.0,
            start_idx = s_idx,
            end_idx = e_idx,
        ))

    return windows


_elev_cond = lambda t: _eval_elev(t) >= MIN_ELEVATION

print("Finding crossing windows.")
crossing_windows = find_windows(in_FOR, times_all, _elev_cond)

print("Finding visibility windows.")
visible_windows  = find_windows(is_visible, times_all, _eval_visible)

print(f"Done: {len(crossing_windows)} crossing event(s), {len(visible_windows)} visible event(s).\n")


# Report Generation
def fmt_utc(t):
    return t.utc_iso()[:19] + " UTC"

def fmt_dur(secs):
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m}m {s:02d}s"

def build_report():
    SEP = "=" * 70
    sep = "-" * 70
    lines = [
        SEP,
        "Space Object Tracking - Ground Based Sensor",
        SEP,
        f"Target: {TLE_NAME}",
        f"Station: {STATION_LAT} N, {STATION_LON} E, {STATION_ALT} m (Svalbard, Norway)",
        f"Window: {epoch_display}",
        f"to  {end_display}",
        f"FOR: {FOR_DEG} deg half-cone from zenith  (min elevation = {MIN_ELEVATION:.0f} deg)",
        f"Night: Sun <= {NIGHT_SUN_ELEV:.0f} deg (astronomical twilight)",
        "",
    ]

    lines += [
        SEP,
        "Crossing Events (object passes through for, elevation >= 20 deg)",
        SEP,
    ]

    if not crossing_windows:
        lines.append("No crossing events in the 24-hour window.")
    else:
        lines.append(f"Total crossings: {len(crossing_windows)}")
        hdr = (f"{'#':>3} {'Start (UTC)':<22} {'End (UTC)':<22}"
               f"{'Duration':>10} {'Peak Elev':>9} {'Az at Peak':>10}"
               f"{'Min Range':>9}")
        lines.append(hdr)
        lines.append("  " + "-" * 96)
        for k, w in enumerate(crossing_windows, 1):
            t_start = fmt_utc(w["start_t"])
            t_end = fmt_utc(w["end_t"])
            dur_str = fmt_dur(w["duration_sec"])
            i0, i1 = w["start_idx"], w["end_idx"]
            seg_elev = elev_deg[i0:i1+1]
            peak_idx = i0 + int(np.argmax(seg_elev))
            peak_el = elev_deg[peak_idx]

            topo_peak = (satellite - station).at(times_all[peak_idx])
            _, az_peak, _ = topo_peak.altaz()
            az_str = f"{az_peak.degrees:.1f} deg"

            min_rng = np.min(range_km[i0:i1+1])
            lines.append(
                f"  {k:>3}. {t_start:<22} {t_end:<22}"
                f" {dur_str:>10} {peak_el:>8.1f} deg"
                f" {az_str:>10} {min_rng:>7.0f} km"
            )
            lines.append(
                f"Peak elevation {peak_el:.1f} deg @ {fmt_utc(times_all[peak_idx])}"
                f" | Min range {min_rng:.0f} km"
            )

    lines += [
        "",
        SEP,
        "Visible / Detectable Events",
        "Condition: elevation >= 20 deg  and  target sunlit  and  Sun <= -18 deg",
        SEP,
    ]

    if not visible_windows:
        lines += [
            "No visible events in the 24-hour window.",
            "",
            f"Diagnostic: minimum solar elevation at station = {np.min(sun_elev_deg):.1f} deg",
        ]
        if np.min(sun_elev_deg) > NIGHT_SUN_ELEV:
            lines += [
                f"The Sun never drops below {NIGHT_SUN_ELEV:.0f} deg at this location.",
                "Svalbard (78.9 N) is experiencing near-Midnight Sun conditions.",
                "Optical detection is geometrically impossible on this date.",
            ]
    else:
        lines.append(f"Total visible events: {len(visible_windows)}")
        hdr = (f"{'#':>3} {'Start (UTC)':<22} {'End (UTC)':<22}"
               f"{'Duration':>10} {'Peak Elev':>9} {'Min Range':>9}")
        lines.append(hdr)
        lines.append(" " + "-" * 86)
        for k, w in enumerate(visible_windows, 1):
            t_start = fmt_utc(w["start_t"])
            t_end = fmt_utc(w["end_t"])
            dur_str = fmt_dur(w["duration_sec"])
            i0, i1 = w["start_idx"], w["end_idx"]
            peak_el = np.max(elev_deg[i0:i1+1])
            min_rng = np.min(range_km[i0:i1+1])
            lines.append(
                f"{k:>3}. {t_start:<22} {t_end:<22}"
                f"{dur_str:>10} {peak_el:>8.1f} deg {min_rng:>7.0f} km"
            )

    n = len(in_FOR)
    lines += [
        "",
        SEP,
        "Summary (fraction of 24-hour analysis window)",
        SEP,
        f"Samples evaluated: {n}",
        f"For (elev >= {MIN_ELEVATION:.0f} deg): "
        f"{np.sum(in_FOR):>5} / {n}  ({100*np.mean(in_FOR):.1f}%)",
        f"Target sunlit: "
        f"{np.sum(is_sunlit):>5} / {n}  ({100*np.mean(is_sunlit):.1f}%)",
        f"Station night (Sun <= {NIGHT_SUN_ELEV:.0f} deg): "
        f"{np.sum(station_night):>5} / {n}  ({100*np.mean(station_night):.1f}%)",
        f"Visible (all conditions met): "
        f"{np.sum(is_visible):>5} / {n}  ({100*np.mean(is_visible):.1f}%)",
        f"Min solar elevation in window: {np.min(sun_elev_deg):.1f} deg",
        f"Max solar elevation in window: {np.max(sun_elev_deg):.1f} deg",
        "",
        sep,
        "Assumptions taken:",
        sep,
        f"1. For (elev >= {MIN_ELEVATION:.0f} deg): {FOR_DEG} deg half-cone from zenith",
        f"2. Station night (Sun <= {NIGHT_SUN_ELEV} deg): astronomical twilight",
        "3. Sunlit check = cylindrical Earth shadow model (skyfield is_sunlit)",
        f"4. Analysis window starts at TLE epoch: {epoch_display}",
        f"5. Time resolution = {STEP_SEC} s coarse grid, bisection-refined to <= {REFINE_TOL:.0f} s",
        "6. Propagator: SGP4 via skyfield EarthSatellite",
        "7. Ephemeris: DE421 (JPL) | Frame: TEME -> GCRS (skyfield internal)",
        sep,
    ]

    return "\n".join(lines)


report_text = build_report()
print(report_text)

with open(REPORT_PATH, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"\nReport saved to {REPORT_PATH}")


# Plot

if not HAS_PLOT:
    sys.exit(0)

print(f"\nGenerating plot to {PLOT_PATH}")

from matplotlib.lines import Line2D
import datetime

epoch_dt = satellite.epoch.utc_datetime()
time_dt = [epoch_dt + datetime.timedelta(seconds=float(s)) for s in sec_grid]

C_ELEV = "#2C6FBF"
C_FOR = "#E05A2B"
C_SUN = "#C8960C"
C_NIGHT = "#6B9EC7"
C_CROSS = "#2C6FBF"
C_GRID = "#CCCCCC"

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
)
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

# Satellite elevation panel
ax1.fill_between(time_dt, elev_deg, 0,
                 where=(elev_deg >= MIN_ELEVATION),
                 color=C_CROSS, alpha=0.15, lw=0)
ax1.plot(time_dt, elev_deg, color=C_ELEV, lw=1.2, solid_capstyle="round")
ax1.axhline(MIN_ELEVATION, color=C_FOR, lw=0.9, ls=(0, (6, 4)), alpha=0.9)

for k, w in enumerate(crossing_windows, 1):
    i0, i1 = w["start_idx"], w["end_idx"]
    seg = elev_deg[i0:i1+1]
    peak_el = np.max(seg)
    peak_dt = time_dt[i0 + int(np.argmax(seg))]
    ax1.text(peak_dt, peak_el + 2.5, str(k),
             ha="center", va="bottom", fontsize=7,
             color=C_ELEV, fontweight="bold")

ax1.set_ylim(0, 95)
ax1.set_ylabel("Elevation (deg)", fontsize=8, color="#444444", labelpad=5)
ax1.set_yticks([0, 20, 45, 70, 90])
ax1.yaxis.set_tick_params(labelsize=7, colors="#666666")
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines["left"].set_color(C_GRID)
ax1.spines["bottom"].set_color(C_GRID)
ax1.spines["left"].set_linewidth(0.6)
ax1.spines["bottom"].set_linewidth(0.6)
ax1.grid(axis="y", color=C_GRID, lw=0.5, alpha=0.7)
ax1.tick_params(bottom=False, left=True, length=3, width=0.6, colors="#888888")
ax1.legend(
    handles=[
        Line2D([0], [0], color=C_ELEV, lw=1.2, label="Satellite elevation"),
        plt.Rectangle((0,0), 1, 1, fc=C_CROSS, alpha=0.25, label="Crossing window"),
        Line2D([0], [0], color=C_FOR, lw=0.9,
               ls=(0,(6,4)), label=f"FOR threshold ({MIN_ELEVATION:.0f} deg)"),
    ],
    loc="upper right", fontsize=7, frameon=False,
    labelcolor="#444444", handlelength=1.8,
)

# Solar elevation panel
ax2.plot(time_dt, sun_elev_deg, color=C_SUN, lw=1.2, solid_capstyle="round")
ax2.axhline(NIGHT_SUN_ELEV, color=C_NIGHT, lw=0.9, ls=(0,(6,4)), alpha=0.9)
ax2.axhline(0, color=C_GRID, lw=0.6, alpha=0.8)
ax2.fill_between(time_dt, sun_elev_deg, 0,
                 where=(sun_elev_deg < 0),
                 color=C_NIGHT, alpha=0.12, lw=0)
if np.any(station_night):
    ax2.fill_between(time_dt, NIGHT_SUN_ELEV, sun_elev_deg,
                     where=station_night, color=C_NIGHT, alpha=0.25, lw=0)

min_sun_val = np.min(sun_elev_deg)
min_sun_dt = time_dt[int(np.argmin(sun_elev_deg))]
ax2.annotate(
    f"{min_sun_val:.1f} deg",
    xy=(min_sun_dt, min_sun_val),
    xytext=(min_sun_dt - datetime.timedelta(hours=2.5), min_sun_val - 2.5),
    fontsize=7, color="#888888",
    arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.7),
    va="top", ha="center",
)

ax2.set_ylim(min(sun_elev_deg) - 6, max(sun_elev_deg) + 5)
ax2.set_ylabel("Sun elevation (deg)", fontsize=8, color="#444444", labelpad=5)
ax2.set_yticks([NIGHT_SUN_ELEV, 0, 10, 20])
ax2.yaxis.set_tick_params(labelsize=7, colors="#666666")
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines["left"].set_color(C_GRID)
ax2.spines["bottom"].set_color(C_GRID)
ax2.spines["left"].set_linewidth(0.6)
ax2.spines["bottom"].set_linewidth(0.6)
ax2.grid(axis="y", color=C_GRID, lw=0.5, alpha=0.7)
ax2.tick_params(length=3, width=0.6, colors="#888888")
ax2.legend(
    handles=[
        Line2D([0], [0], color=C_SUN, lw=1.2, label="Solar elevation"),
        Line2D([0], [0], color=C_NIGHT, lw=0.9,
               ls=(0,(6,4)), label=f"Astron. twilight ({NIGHT_SUN_ELEV:.0f} deg)"),
    ],
    loc="upper right", fontsize=7, frameon=False,
    labelcolor="#444444", handlelength=1.8,
)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M UTC"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.tick_params(axis="x", labelsize=7.5, colors="#666666")

fig.text(
    0.5, 0.97,
    f"{TLE_NAME}  |  Svalbard ({STATION_LAT} N)  |  "
    f"Window: {epoch_display}",
    ha="center", va="top", fontsize=8.5, color="#333333"
)

plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight",
            transparent=True, facecolor="none")
plt.close(fig)
print(f"\nPlot saved to {PLOT_PATH}\n")
print("Analysis complete.")

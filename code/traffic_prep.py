"""
traffic_prep.py — cleaning, aggregation and feature engineering for the
Trafikverket MCS dataset (E4 Essingeleden southbound, 8 portals). Raw files cover
04:00–10:00 *by timestamp*, but timestamps mix UTC and local time (see TIME BASE below);
after alignment the common window is 06:00–10:00 Swedish local time.
The ALL-day files are handled with set_window(0, 24*60) and
build_features(..., continuous=True, extra=True); traffic regimes via RegimeModel.

Pipeline (each step is a function so later notebooks can reuse it):

    raw = load_raw(path)                          # 1-min lane-level records
    lane, report = clean_lane_data(raw, lane_stats)   # full minute grid, zeros vs outages
    portal = aggregate_portal(lane, lane_stats, res="5min")   # portal-level flow/speed
    feats = build_features(wide, profile)         # supervised table, target = flow t+15 min

Everything that is *estimated from data* (lane shares, historical profiles,
scaler) is fitted on the TRAINING split only and then applied unchanged to the
evaluation / final-evaluation splits, so nothing leaks from 2022 into training.

Key cleaning finding (see notebook, section 2): the raw files contain no rows
with FLOW = 0. A missing minute on a sensor that is otherwise reporting is
almost always a minute with zero vehicles, not a sensor failure — the observed
share of missing minutes per hour matches the Poisson probability of zero
arrivals, P(0) = exp(-lambda), almost exactly. We therefore fill those minutes
with FLOW = 0 and only mark a gap as an *outage* when zero arrivals would be
statistically implausible (sum of expected arrivals over the gap >= 10, i.e.
P(all zeros) < 5e-5), or when the sensor has no data at all that day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
# Portals in downstream (driving) order, southbound: 58.140 -> 55.620
PORTALS_DOWNSTREAM = [58.140, 57.820, 57.435, 57.055, 56.780, 56.490, 56.160, 55.620]
PORTAL_COL = {km: f"p{int(round(km * 1000))}" for km in PORTALS_DOWNSTREAM}  # 58.14 -> "p58140"

# --- TIME BASE ---------------------------------------------------------------
# The sensors do NOT share a clock convention. Sensors with DP_ID < 4000 (all lanes
# at portals 57.055–55.620 except 4872/4873) are stamped in UTC; the 44xx/48xx
# sensors are stamped in Swedish local time (CET/CEST). Evidence (notebook §2):
#   * cross-correlation lag between 57.435 and 57.055 is exactly 60 min in winter
#     and 120 min in summer, and it jumps on the DST switch day (2022-03-27);
#   * in the ALL-day files the night minimum / morning ramp-up of the <4000 sensors
#     moves by one hour between summer and winter, the 44xx/48xx ones do not.
# load_raw(align_time=True) converts the UTC sensors to local time. After that
# the window where ALL sensors have data every day is 06:00–10:00 local, which
# contains the 07:30–08:30 benchmark peak plus 45 min of history.
UTC_SENSORS = {153, 351, 353, 534, 536, 539, 543, 740, 749, 751, 902, 1076, 1079, 1254, 1443}
TZ = "Europe/Stockholm"
DAY_START_MIN = 6 * 60      # analysis window starts 06:00 local
N_MIN = 4 * 60              # 06:00–09:59 local -> 240 one-minute slots

SPEED_MIN_MS = 1.0          # < 3.6 km/h with 1 vehicle at 04:00 = sensor artefact
SPEED_MAX_MS = 40.0         # > 144 km/h on a 70 km/h road = artefact
OUTAGE_EXPECTED_VEH = 10.0  # gap is an outage if expected arrivals over it >= this

# Swedish public holidays + de-facto holidays (midsummer eve, xmas eve, NYE)
# inside the data windows. Used as a "non-working day" feature.
HOLIDAYS = pd.to_datetime([
    "2021-06-06", "2021-06-25", "2021-06-26", "2021-11-06",
    "2021-12-24", "2021-12-25", "2021-12-26", "2021-12-31",
    "2022-01-01", "2022-01-06", "2022-04-15", "2022-04-17", "2022-04-18",
    "2022-05-26", "2022-06-06", "2022-06-24", "2022-06-25",
])


# ----------------------------------------------------------------------------
# 0. Analysis window (morning files: 06:00–10:00 local; ALL-day files: 00:00–24:00)
# ----------------------------------------------------------------------------
def set_window(start_min: int, n_min: int) -> None:
    """Set the local-time window every function works on, e.g.
    set_window(6*60, 4*60) for the morning files (default) or
    set_window(0, 24*60) for the ALL-day files."""
    global DAY_START_MIN, N_MIN
    DAY_START_MIN, N_MIN = int(start_min), int(n_min)


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_raw(path: str, align_time: bool = True) -> pd.DataFrame:
    """Read one of the provided CSVs (';'-separated), put every sensor on Swedish
    local time (align_time=True) and keep the current window (see set_window).

    After the UTC -> local shift some rows of the UTC sensors move to the next
    local date (e.g. 23:30 UTC = 00:30/01:30 local next day). Only local dates
    that are in the file itself are kept, so the interleaved 2022 files never
    borrow rows from each other. On the autumn DST night the local hour
    02:00–02:59 occurs twice; the first occurrence is kept.
    With align_time=False the original stamps are kept (only for diagnostics)."""
    df = pd.read_csv(path, sep=";", usecols=["DP_ID", "PORTAL", "Date", "Interval_1", "SPEED_MS_AVG", "FLOW"],
                     dtype={"DP_ID": "int32", "Date": "int32", "Interval_1": "int16",
                            "SPEED_MS_AVG": "float32", "FLOW": "float32"})
    df["portal_km"] = df["PORTAL"].str.replace("E4S ", "", regex=False) \
                                   .str.replace(",", ".", regex=False).astype(float).round(3)
    file_dates = pd.to_datetime(pd.Series(df["Date"].unique()).astype(str), format="%Y%m%d")
    ts = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["Interval_1"].astype(int), unit="min")
    if align_time:
        utc = df["DP_ID"].isin(UTC_SENSORS)
        ts = ts.copy()
        ts[utc] = ts[utc].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    df["date"] = ts.dt.normalize()
    df["minute"] = ((ts - df["date"]).dt.total_seconds() // 60).astype(int) - DAY_START_MIN
    keep = (df["minute"] >= 0) & (df["minute"] < N_MIN) & df["date"].isin(file_dates)
    df = df[keep].rename(columns={"DP_ID": "sensor", "SPEED_MS_AVG": "speed_ms", "FLOW": "flow"})
    df = df.drop_duplicates(["sensor", "date", "minute"], keep="first")
    return df[["sensor", "portal_km", "date", "minute", "flow", "speed_ms"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Lane grid (dense arrays) and lane statistics (fit on TRAINING only)
# ----------------------------------------------------------------------------
@dataclass
class LaneStats:
    sensor_portal: dict = field(default_factory=dict)   # sensor -> portal_km
    share: dict = field(default_factory=dict)            # sensor -> mean(flow_i)/mean(flow of other lanes)
    profile: dict = field(default_factory=dict)          # sensor -> list[360] mean veh/min by minute of window

    def to_json(self, path):
        json.dump({"sensor_portal": {str(k): v for k, v in self.sensor_portal.items()},
                   "share": {str(k): v for k, v in self.share.items()},
                   "profile": {str(k): v for k, v in self.profile.items()}}, open(path, "w"))

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls({int(k): v for k, v in d["sensor_portal"].items()},
                   {int(k): v for k, v in d["share"].items()},
                   {int(k): v for k, v in d["profile"].items()})



STATUS = np.array(["observed", "zero_filled", "outage", "sensor_day_missing", "corrupt"])


@dataclass
class LaneGrid:
    """Complete sensor x date x minute grid stored as dense arrays [s, d, m]."""
    sensors: np.ndarray            # sorted sensor ids
    dates: np.ndarray              # sorted local dates (datetime64)
    portal: np.ndarray             # portal_km per sensor
    flow: np.ndarray               # veh/min, NaN = unknown
    speed: np.ndarray              # m/s, NaN = unknown / no vehicles
    present: np.ndarray            # a raw row existed
    status: np.ndarray | None = None   # int8 code into STATUS

    def to_frame(self) -> pd.DataFrame:
        """Long table (sensor, date, minute, portal_km, flow, speed_ms, status) for inspection."""
        ns, nd, nm = self.flow.shape
        out = pd.DataFrame({
            "sensor": np.repeat(self.sensors, nd * nm),
            "date": np.tile(np.repeat(self.dates, nm), ns),
            "minute": np.tile(np.arange(nm), ns * nd),
            "portal_km": np.repeat(self.portal, nd * nm),
            "flow": self.flow.ravel(), "speed_ms": self.speed.ravel()})
        if self.status is not None:
            out["status"] = pd.Categorical.from_codes(self.status.ravel(), STATUS)
        return out


def _grid(raw: pd.DataFrame, sensor_portal: dict) -> LaneGrid:
    sensors = np.array(sorted(sensor_portal))
    dates = np.sort(raw["date"].unique())
    raw = raw[raw["sensor"].isin(sensors)]
    si = np.searchsorted(sensors, raw["sensor"].to_numpy())
    di = np.searchsorted(dates, raw["date"].to_numpy())
    mi = raw["minute"].to_numpy()
    shape = (len(sensors), len(dates), N_MIN)
    flow = np.full(shape, np.nan, np.float32); speed = np.full(shape, np.nan, np.float32)
    present = np.zeros(shape, bool)
    flow[si, di, mi] = raw["flow"].to_numpy(); speed[si, di, mi] = raw["speed_ms"].to_numpy()
    present[si, di, mi] = True
    return LaneGrid(sensors, dates, np.array([sensor_portal[s] for s in sensors], np.float32),
                    flow, speed, present)


def _portal_lanes(lg: LaneGrid) -> dict:
    return {round(float(k), 3): np.where(np.isclose(lg.portal, k))[0] for k in np.unique(lg.portal)}


def _others_mean(flow0: np.ndarray, lanes: dict, dayp: np.ndarray | None = None) -> np.ndarray:
    """Mean flow of the OTHER lanes at the same portal, per lane [s, d, m].
    With dayp (sensor-day present mask [s, d]) only lanes reporting that day are counted."""
    out = np.zeros_like(flow0)
    for idx in lanes.values():
        tot = flow0[idx].sum(0)
        if dayp is None:
            n = max(len(idx) - 1, 1)
        else:
            n = np.maximum(dayp[idx].sum(0)[None, :, None] - 1, 1)
        out[idx] = (tot[None] - flow0[idx]) / n
    return out


def _roll_centered(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    """Centred rolling mean / max along the last axis, min_periods = 1."""
    h = w // 2
    if how == "max":
        pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(h, h)], constant_values=False)
        return np.max(np.stack([pad[..., i:i + x.shape[-1]] for i in range(w)]), axis=0)
    pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(h, h)])
    cnt = np.pad(np.ones_like(x), [(0, 0)] * (x.ndim - 1) + [(h, h)])
    c = np.cumsum(pad, -1); n = np.cumsum(cnt, -1)
    c = np.concatenate([np.zeros_like(c[..., :1]), c], -1); n = np.concatenate([np.zeros_like(n[..., :1]), n], -1)
    return (c[..., w:] - c[..., :-w]) / (n[..., w:] - n[..., :-w])


def _roll_trailing(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    """Trailing (causal) rolling mean / max over the current and previous w−1 steps, min_periods = 1."""
    padw = [(0, 0)] * (x.ndim - 1) + [(w - 1, 0)]
    if how == "max":
        pad = np.pad(x, padw, constant_values=False)
        return np.max(np.stack([pad[..., i:i + x.shape[-1]] for i in range(w)]), axis=0)
    pad = np.pad(x, padw); cnt = np.pad(np.ones_like(x), padw)
    c = np.cumsum(pad, -1); n = np.cumsum(cnt, -1)
    c = np.concatenate([np.zeros_like(c[..., :1]), c], -1); n = np.concatenate([np.zeros_like(n[..., :1]), n], -1)
    return (c[..., w:] - c[..., :-w]) / (n[..., w:] - n[..., :-w])


def fit_lane_stats(raw_train: pd.DataFrame) -> LaneStats:
    """Relative volume of each lane vs the other lanes at its portal, and each
    lane's mean flow per minute of the window (TRAINING data only).
    Used to (a) compute the expected flow during a gap and (b) impute a failed lane."""
    sp = {int(k): round(float(v), 3) for k, v in raw_train.groupby("sensor")["portal_km"].first().items()}
    lg = _grid(raw_train, sp)
    dayp = lg.present.any(2)
    flow0 = np.nan_to_num(lg.flow)
    oth = _others_mean(flow0, _portal_lanes(lg), dayp)
    share, prof = {}, {}
    for i, s in enumerate(lg.sensors):
        f, o = flow0[i][dayp[i]], oth[i][dayp[i]]
        share[int(s)] = float(f.mean() / o.mean())
        prof[int(s)] = np.round(f.mean(0), 3).tolist()
    return LaneStats(sp, share, prof)


# ----------------------------------------------------------------------------
# 3. Lane-level cleaning
# ----------------------------------------------------------------------------
def clean_lane_data(raw: pd.DataFrame, stats: LaneStats, causal: bool = False) -> tuple[LaneGrid, dict]:
    """Return the complete 1-min lane grid with cleaned flow (veh/min, NaN = outage),
    speed (NaN = no vehicles or invalid) and a status code per cell.

    causal=False (hindsight, for TARGETS / analysis): a gap is judged with the whole gap
      and the neighbours ±2 min around it, and a sensor with no data all day is an outage.
    causal=True (real-time, for FEATURES): only minutes up to the current one are used —
      neighbours over the last 5 min, and a gap becomes an outage from the minute where the
      vehicles expected *since the gap started* reach the threshold. This is what a live
      system could compute, so features never depend on data after the prediction time."""
    rep = {"raw_rows": len(raw)}
    lg = _grid(raw, stats.sensor_portal)
    F, V, P = lg.flow, lg.speed, lg.present

    # (a) corrupt rows: FLOW missing (these rows carry a placeholder speed of 1.0)
    bad_row = P & np.isnan(F)
    rep["corrupt_rows_flow_nan"] = int(bad_row.sum())
    # (b) implausible speeds -> keep the vehicle count, drop the speed
    bad_speed = ~np.isnan(V) & ((V < SPEED_MIN_MS) | (V > SPEED_MAX_MS))
    rep["invalid_speed_set_nan"] = int(bad_speed.sum())
    V[bad_speed | bad_row] = np.nan

    # (c) missing minutes: zero flow or outage?
    flow0 = np.nan_to_num(F)
    lanes = _portal_lanes(lg)
    share = np.array([stats.share[int(s)] for s in lg.sensors], np.float32)[:, None, None]
    roll = _roll_trailing if causal else _roll_centered
    expected = roll(_others_mean(flow0, lanes), 5) * share
    # if NO lane at the portal reported anything in the window, the neighbours
    # carry no information (whole portal may be down) -> use the lane's historical profile
    any_nb = np.zeros_like(P)
    for idx in lanes.values():
        any_nb[idx] = roll(P[idx].any(0), 5, "max")[None]
    hist = np.array([stats.profile[int(s)] for s in lg.sensors], np.float32)[:, None, :]
    expected = np.where(any_nb, expected, hist)

    dayp = P.any(2, keepdims=True)
    miss = ~P if causal else (~P & dayp)             # causal: no knowledge of the rest of the day
    m1 = miss.reshape(-1, N_MIN)
    start = m1 & ~np.concatenate([np.zeros((m1.shape[0], 1), bool), m1[:, :-1]], 1)
    run_id = np.cumsum(start.ravel()) - 1
    mf = m1.ravel(); e = expected.ravel().astype(np.float64)
    outage = np.zeros(m1.size, bool)
    if causal:        # expected vehicles accumulated from the start of the gap up to this minute
        cum = np.cumsum(np.where(mf, e, 0.0))
        base = (cum - e)[start.ravel()]              # cumulative total just before each run starts
        outage[mf] = (cum[mf] - base[run_id[mf]]) >= OUTAGE_EXPECTED_VEH
    else:             # expected vehicles over the whole gap (uses the future of the gap)
        run_sum = np.bincount(run_id[mf], weights=e[mf])
        outage[mf] = run_sum[run_id[mf]] >= OUTAGE_EXPECTED_VEH
    outage = outage.reshape(miss.shape)

    st = np.zeros(F.shape, np.int8)                  # 0 observed
    st[miss & ~outage] = 1                           # zero_filled
    st[outage] = 2                                   # outage
    if not causal:
        st[~np.broadcast_to(dayp, F.shape)] = 3      # sensor_day_missing
    st[bad_row] = 4                                  # corrupt
    lg.flow = np.where(st <= 1, flow0, np.nan).astype(np.float32)
    V[st == 1] = np.nan                              # speed undefined with 0 vehicles
    lg.speed, lg.status = V, st

    counts = np.bincount(st.ravel(), minlength=len(STATUS))
    rep.update({str(STATUS[i]): int(c) for i, c in enumerate(counts)})
    rep["grid_rows"] = int(st.size)
    return lg, rep


# ----------------------------------------------------------------------------
# 4. Portal-level aggregation
# ----------------------------------------------------------------------------
def aggregate_portal(lg: LaneGrid, stats: LaneStats, res: str = "5min",
                     max_missing_frac: float = 0.2, interp_limit: int = 2,
                     min_valid_share: float = 0.6) -> pd.DataFrame:
    """Sum lanes to portal flow and compute flow-weighted mean speed, then
    resample to `res`. Returns long format: ts, date, minute_end, portal_km, flow_vph, speed_kmh, ...

    * a missing lane is imputed from the other lanes via its training share,
      only if the valid lanes normally carry >= min_valid_share of the portal flow;
    * a bin with > max_missing_frac of its minutes missing is NaN;
    * short NaN gaps (<= interp_limit bins) are filled CAUSALLY (last value carried
      forward, within the day) -> usable as real-time model inputs. The unfilled
      values are kept in *_obs columns; use those for targets / scoring.
    Timestamps are labelled at the END of the bin (value at 07:20 = 07:15–07:20),
    so a value stamped t is really known at time t (no look-ahead)."""
    b = int(pd.Timedelta(res).total_seconds() // 60)
    nb = N_MIN // b
    share_all = np.array([stats.share[int(s)] for s in lg.sensors], np.float32)
    frames = []
    for km, idx in _portal_lanes(lg).items():
        F, V = lg.flow[idx], lg.speed[idx]
        valid = ~np.isnan(F)
        sh = share_all[idx][:, None, None]
        vshare, ashare = (sh * valid).sum(0), sh.sum()
        nvalid = valid.sum(0)
        flow = np.nansum(F, 0)
        ok = vshare / ashare >= min_valid_share
        imputed = (nvalid < len(idx)) & ok
        flow = np.where(imputed, flow * ashare / np.where(vshare > 0, vshare, np.nan), flow)
        flow = np.where((nvalid < len(idx)) & ~ok, np.nan, flow)
        both = valid & ~np.isnan(V)
        wspeed = np.where(both, V * F, 0).sum(0); wflow = np.where(both, F, 0).sum(0)

        fb = flow.reshape(len(lg.dates), nb, b)
        nmiss = np.isnan(fb).sum(2)
        fsum = np.nansum(fb, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = fsum / (b - nmiss) * 60.0
            v = wspeed.reshape(-1, nb, b).sum(2) / wflow.reshape(-1, nb, b).sum(2) * 3.6
        bad = nmiss / b > max_missing_frac
        q[bad] = np.nan; v[bad] = np.nan
        v[~np.isfinite(v)] = np.nan
        qf = pd.DataFrame(q).ffill(axis=1, limit=interp_limit).to_numpy()
        vf = pd.DataFrame(v).ffill(axis=1, limit=interp_limit).to_numpy()
        minute_end = DAY_START_MIN + (np.arange(nb) + 1) * b
        dates = np.repeat(lg.dates, nb)
        frames.append(pd.DataFrame({
            "ts": dates + np.tile(minute_end, len(lg.dates)).astype("timedelta64[m]"),
            "date": dates, "minute_end": np.tile(minute_end, len(lg.dates)), "portal_km": float(km),
            "flow_vph": qf.ravel(), "speed_kmh": vf.ravel(), "flow_vph_obs": q.ravel(), "speed_kmh_obs": v.ravel(),
            "imputed": imputed.reshape(len(lg.dates), nb, b).mean(2).ravel()}))
    return pd.concat(frames).sort_values(["portal_km", "ts"]).reset_index(drop=True)


def to_wide(portal: pd.DataFrame, observed: bool = False) -> pd.DataFrame:
    """ts-indexed wide table: q_p58140, v_p58140, ... (q = flow veh/h, v = speed km/h).
    observed=True -> un-filled values (for targets / scoring)."""
    sfx = "_obs" if observed else ""
    w = portal.pivot_table(index="ts", columns="portal_km", values=["flow_vph" + sfx, "speed_kmh" + sfx], dropna=False)
    w.columns = [("q_" if a.startswith("flow") else "v_") + PORTAL_COL[k] for a, k in w.columns]
    order = [f"{p}_{PORTAL_COL[k]}" for p in ("q", "v") for k in PORTALS_DOWNSTREAM]
    return w[order]


# ----------------------------------------------------------------------------
# 5. Historical profile (fit on TRAINING only)
# ----------------------------------------------------------------------------
def day_type(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return np.where(d.dt.dayofweek.ge(5) | d.dt.normalize().isin(HOLIDAYS), "offday", "workday")


def fit_profile(wide_train: pd.DataFrame) -> pd.DataFrame:
    """Mean flow per (day type, time of day, portal) — the classic 'historical average'."""
    w = wide_train.copy()
    w["dtype"] = day_type(pd.Series(w.index, index=w.index))
    w["tod"] = w.index.hour * 60 + w.index.minute
    return w.groupby(["dtype", "tod"])[[c for c in w.columns if c.startswith("q_")]].mean()


def hist_features(F: pd.DataFrame, wide_fit: pd.DataFrame, horizon_min: int = 15,
                  oof_folds: int = 5, res_min: int = 5) -> pd.DataFrame:
    """Historical-average inputs hist_q_p* for the rows of F, from the profile of `wide_fit`
    (observed 5-min flows, mean per day type × time of day), looked up at the TARGET time.

    Leakage guard (out-of-fold target encoding): if a row's target lies inside `wide_fit`, the
    plain profile would contain the answer itself. The days of `wide_fit` are therefore split
    into `oof_folds` groups of whole weeks (week number mod K), and such a row gets the profile
    computed WITHOUT the fold that contains its target day — i.e. from the other weeks only.
    (Leave-one-day-out is avoided on purpose: within a profile cell it is an exact decreasing
    function of the row's own target, which a tree can exploit.)
    Rows whose target is outside `wide_fit` (validation rows with a train-split profile, 2022)
    get the plain profile. oof_folds=0 -> plain profile for every row."""
    qc = [c for c in wide_fit.columns if c.startswith("q_")]
    w = wide_fit[qc]
    res = pd.Timedelta(minutes=res_min)
    day_fit = (w.index - res).normalize()                      # day the 5-min bin belongs to
    d0 = day_fit.min()
    K = max(int(oof_folds), 1)
    fold_fit = ((day_fit - d0).days // 7) % K
    dt_fit = day_type(pd.Series(w.index, index=w.index)); tod_fit = w.index.hour * 60 + w.index.minute
    tt = F.index + pd.Timedelta(minutes=horizon_min)
    dt_row = day_type(pd.Series(F.index, index=F.index))       # same key as build_features
    tod = tt.hour * 60 + tt.minute
    day_t = (tt - res).normalize()
    fold_row = np.where(day_t.isin(day_fit.unique()) & (oof_folds > 0), ((day_t - d0).days // 7) % K, -1)
    key = pd.MultiIndex.from_arrays([dt_row, tod])
    key_f = pd.MultiIndex.from_arrays([dt_row, tod, np.maximum(fold_row, 0)])
    g_all = pd.MultiIndex.from_arrays([dt_fit, tod_fit]); g_f = pd.MultiIndex.from_arrays([dt_fit, tod_fit, fold_fit])
    S, N = w.groupby(g_all).sum(), w.notna().groupby(g_all).sum()
    Sf, Nf = w.groupby(g_f).sum(), w.notna().groupby(g_f).sum()
    oof = fold_row >= 0
    out = {}
    for c in qc:
        s = S[c].reindex(key).to_numpy(float); n = N[c].reindex(key).to_numpy(float)
        sf = np.nan_to_num(Sf[c].reindex(key_f).to_numpy(float)); nf = np.nan_to_num(Nf[c].reindex(key_f).to_numpy(float))
        s = np.where(oof, s - sf, s); n = np.where(oof, n - nf, n)
        out[f"hist_{c}"] = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    return pd.DataFrame(out, index=F.index)


# ----------------------------------------------------------------------------
# 6. Supervised feature table
# ----------------------------------------------------------------------------
def build_features(wide: pd.DataFrame, profile: pd.DataFrame, horizon_min: int = 15,
                   res_min: int = 5, n_lags_target: int = 6, n_lags_other: int = 3,
                   target_window=("07:30", "08:30"), targets="all",
                   wide_obs: pd.DataFrame | None = None, continuous: bool = False,
                   extra: bool = False, wide_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per prediction time t. All features use data stamped <= t.
    Targets y_q_<portal> = flow at t + horizon, taken from `wide_obs` (un-filled,
    so a missing truth stays NaN) when given. Rows are kept only where the
    TARGET time lies inside `target_window` (the morning-peak benchmark);
    pass target_window=None to keep all hours.

    continuous=True (ALL-day data): lags and targets run across midnight on a
    regular 5-min grid (gaps -> NaN) instead of being computed day by day.
    extra=True adds density k = q/v (veh/km) per portal and cyclic time of day.
    wide_flags: the un-filled version of `wide` (same cleaning) for the *_filled flags; by default
    the flags are derived from `wide_obs`. Pass it when targets come from a different
    (hindsight) cleaning than the features (causal)."""
    h = horizon_min // res_min
    out = []
    if continuous:
        grid = pd.date_range(wide.index.min().normalize(),
                             wide.index.max().normalize() + pd.Timedelta(days=1), freq=f"{res_min}min")
        chunks = [(None, wide.reindex(grid))]
        wide_obs_g = wide_obs.reindex(grid) if wide_obs is not None else None
        wide_flags_g = wide_flags.reindex(grid) if wide_flags is not None else wide_obs_g
    else:
        chunks = wide.groupby(wide.index.normalize())
        wide_obs_g = wide_obs
        wide_flags_g = wide_flags if wide_flags is not None else wide_obs
    for day, d in chunks:
        d = d.sort_index()
        f = {}                       # collect columns in a dict, build the frame once
        for c in d.columns:
            is_q = c.startswith("q_")
            nl = n_lags_target if is_q else n_lags_other
            for L in range(nl + 1):
                f[f"{c}_lag{L * res_min}"] = d[c].shift(L)
            if is_q:
                f[f"{c}_ma15"] = d[c].rolling(3, min_periods=2).mean()
                f[f"{c}_d15"] = d[c] - d[c].shift(3)                 # 15-min trend
        # spatial features: speed drop between consecutive portals (congestion front)
        vs = [f"v_{PORTAL_COL[k]}" for k in PORTALS_DOWNSTREAM]
        for up, dn in zip(vs[:-1], vs[1:]):
            f[f"dv_{up[2:]}_{dn[2:]}"] = d[up] - d[dn]
        if extra:   # density (veh/km) = flow / speed: separates free flow from congestion at equal flow
            for k in PORTALS_DOWNSTREAM:
                p = PORTAL_COL[k]
                f[f"k_{p}_lag0"] = d[f"q_{p}"] / d[f"v_{p}"].where(d[f"v_{p}"] > 3)
        o = wide_obs_g.loc[d.index] if wide_obs_g is not None else d
        if extra:   # speed at the target time: NOT a feature, only used to label the target's regime
            for c in [c for c in d.columns if c.startswith("v_")]:
                f[f"y_{c}"] = o[c].shift(-h)
        for c in [c for c in d.columns if c.startswith("q_")]:
            f[f"y_{c}"] = o[c].shift(-h)                              # TARGET
            if wide_flags_g is not None:                              # 1 = value at t is carried forward
                of = wide_flags_g.loc[d.index]
                f[f"{c}_filled"] = (of[c].isna() & d[c].notna()).astype(int)
        out.append(pd.DataFrame(f, index=d.index))
    F = pd.concat(out)
    if continuous:
        F = F.loc[F.index.isin(wide.index)]          # keep only times that exist in the data

    # calendar features
    ts_target = F.index + pd.Timedelta(minutes=horizon_min)
    tod = ts_target.hour * 60 + ts_target.minute
    F["tod_target_min"] = tod
    F["dow"] = F.index.dayofweek
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * F["dow"] / 7), np.cos(2 * np.pi * F["dow"] / 7)
    dt = day_type(pd.Series(F.index, index=F.index))
    F["is_offday"] = (dt == "offday").astype(int)
    if extra:
        F["tod_sin"], F["tod_cos"] = np.sin(2 * np.pi * tod / 1440), np.cos(2 * np.pi * tod / 1440)
    # (no month feature: training covers Jun–Dec, evaluation Jan–Jun -> trees cannot extrapolate)
    # historical average of each portal's flow at the TARGET time-of-day (training profile)
    prof = profile.copy()
    key = pd.MultiIndex.from_arrays([dt, tod])
    for c in profile.columns:
        F[f"hist_{c}"] = prof[c].reindex(key).values

    if target_window is not None:
        lo, hi = [int(x[:2]) * 60 + int(x[3:]) for x in target_window]
        F = F[(tod >= lo) & (tod <= hi)]
    if targets != "all":
        F = F.drop(columns=[c for c in F.columns if c.startswith("y_") and c not in targets])
    return F


# ----------------------------------------------------------------------------
# 6b. Traffic regimes from the fundamental diagram (fit on TRAINING only)
# ----------------------------------------------------------------------------
class RegimeModel:
    """Free-flow / congested regime per portal from the fundamental diagram.

    The 5-min speed distribution of every portal is bimodal: a narrow free-flow
    mode (65–80 km/h) and a broad congested mode (20–40 km/h). The regime boundary
    v* is the ANTIMODE — the minimum of the (kernel-smoothed) speed density between
    the two modes — i.e. a 1-D density-based two-cluster split along the speed axis
    of the fundamental diagram. (A 2-D Gaussian mixture on speed/density was tried
    first; its wide congested component also claimed very high speeds in the tails,
    so the simpler, monotone rule is used.)
    proba() returns a smooth P(congested) = logistic((v* - v) / width)."""

    def __init__(self, width: float = 3.0):
        self.width = width

    def fit(self, portal_long: pd.DataFrame, seed: int = 0, n_sample: int = 20000):
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(seed)
        self.vstar, self.info = {}, {}
        grid = np.arange(5, 95, 0.5)
        for km, x in portal_long.groupby("portal_km"):
            v = x["speed_kmh_obs"].dropna().to_numpy()
            v = v[v > 3]
            dens = gaussian_kde(rng.choice(v, min(n_sample, len(v)), replace=False), bw_method=0.08)(grid)
            main = grid[np.argmax(dens)]
            m_low = (grid >= 10) & (grid <= main - 20)
            low = grid[m_low][np.argmax(dens[m_low])]
            m_mid = (grid > low) & (grid < main)
            vstar = float(grid[m_mid][np.argmin(dens[m_mid])])
            p = PORTAL_COL[round(float(km), 3)]
            self.vstar[p] = vstar
            self.info[p] = {"free-flow mode (km/h)": float(main), "congested mode (km/h)": float(low),
                            "boundary v* (km/h)": vstar, "share congested": float((v < vstar).mean())}
        self.grid = grid
        return self

    def proba(self, p: str, q, v) -> np.ndarray:
        """P(congested) for portal p ('p55620'); NaN where speed is missing.
        (q is accepted for a uniform interface; the boundary is on speed.)"""
        v = np.asarray(v, float)
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp((v - self.vstar[p]) / self.width))

    @property
    def models(self):          # portal ids with a fitted boundary
        return self.vstar


def add_regime_features(F: pd.DataFrame, reg: RegimeModel) -> pd.DataFrame:
    """Adds r_p<id>_lag0 = P(congested) at time t for every portal (input feature),
    and y_reg_q_p<id> = regime (0/1) at the TARGET time (for evaluation only)."""
    F = F.copy()
    for p in reg.models:
        F[f"r_{p}_lag0"] = reg.proba(p, F[f"q_{p}_lag0"], F[f"v_{p}_lag0"])
        if f"y_v_{p}" in F:
            pr = reg.proba(p, F[f"y_q_{p}"], F[f"y_v_{p}"])
            F[f"y_reg_q_{p}"] = np.where(np.isnan(pr), np.nan, (pr >= 0.5).astype(float))
    return F


# ----------------------------------------------------------------------------
# 7. Scaling (fit on TRAINING rows only)
# ----------------------------------------------------------------------------
NO_SCALE_PREFIX = ("y_", "dow", "is_")


def scalable_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if not c.startswith(NO_SCALE_PREFIX) and not c.endswith("_filled")]


def fit_scaler(F_train: pd.DataFrame) -> pd.DataFrame:
    cols = scalable_columns(F_train)
    return pd.DataFrame({"mean": F_train[cols].mean(), "std": F_train[cols].std().replace(0, 1)})


def apply_scaler(F: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    G = F.copy()
    G[sc.index] = (G[sc.index] - sc["mean"]) / sc["std"]
    return G


# ----------------------------------------------------------------------------
# 7b. Masking augmentation (robustness to sensor failures)
# ----------------------------------------------------------------------------
def portal_feature_map(cols) -> dict:
    """portal id ('p55620') -> all input columns that come from that portal's sensors
    (flow/speed lags, moving average, trend, and speed differences involving it)."""
    m = {}
    for c in cols:
        if c.endswith("_filled"):
            continue
        mm = _re.match(r"^(q|v|k|r)_(p\d+)_", c)
        if mm:
            m.setdefault(mm.group(2), []).append(c)
        elif c.startswith("dv_"):
            for p in c.split("_")[1:]:
                m.setdefault(p, []).append(c)
    return m


def mask_augment(F: pd.DataFrame, x_cols, n_copies: int = 2, p_portal=(0.0, 0.9),
                 p_value=(0.0, 0.3), seed: int = 0) -> pd.DataFrame:
    """Return F plus `n_copies` copies in which sensor inputs are blanked (NaN).

    For every copied row, a portal-failure probability is drawn from U(p_portal)
    and each portal is switched off with that probability (all its features ->
    NaN, i.e. a whole-portal outage); independently, single values are dropped
    with a probability drawn from U(p_value). Mild and severe failure patterns
    are therefore both represented. Targets, calendar and historical-profile
    features are never touched (they are always available in operation).
    The original rows are kept, so the model still sees the clean case."""
    rng = np.random.default_rng(seed)
    x_cols = list(x_cols)
    idx = {c: i for i, c in enumerate(x_cols)}
    pm = portal_feature_map(x_cols)
    portals = sorted(pm)
    vcols = [idx[c] for c in x_cols if c.startswith(("q_p", "v_p", "k_p", "r_p", "dv_")) and not c.endswith("_filled")]
    out = [F]
    for _ in range(n_copies):
        n = len(F)
        M = np.zeros((n, len(x_cols)), dtype=bool)
        down = rng.random((n, len(portals))) < rng.uniform(*p_portal, n)[:, None]
        for j, p in enumerate(portals):
            M[np.ix_(down[:, j], [idx[c] for c in pm[p]])] = True
        M[:, vcols] |= rng.random((n, len(vcols))) < rng.uniform(*p_value, n)[:, None]
        G = F.copy()
        G[x_cols] = G[x_cols].mask(M)
        out.append(G)
    return pd.concat(out)


# ----------------------------------------------------------------------------
# 8. Imputation of missing INPUTS for models that cannot take NaN (ridge, NN)
# ----------------------------------------------------------------------------
import re as _re
_PORTAL_FEAT = _re.compile(r"^(q|v|k|r)_p(\d+)_(lag\d+|ma15)$")


class FeatureImputer:
    """Causal, row-wise imputation fitted on TRAINING rows.

    A missing portal value (e.g. q_p57820_lag5) is estimated from the *other*
    portals at the same time and lag, scaled by that portal's typical ratio to
    the cross-portal mean (flows along the corridor move together: r > 0.9).
    Speed differences are recomputed from the filled speeds, trends set to 0,
    anything left gets the training median. A 0/1 flag per portal
    (miss_p<id>) tells the model that its current value was estimated.
    Uses only information available at time t -> valid in real time."""

    def fit(self, F: pd.DataFrame):
        self.groups = {}
        for c in F.columns:
            m = _PORTAL_FEAT.match(c)
            if m:
                self.groups.setdefault((m.group(1), m.group(3)), []).append(c)
        self.ratio = {}
        for key, cols in self.groups.items():
            rm = F[cols].mean(axis=1)
            for c in cols:
                self.ratio[c] = float(F[c].mean() / rm.mean())
        self.portals = sorted({_PORTAL_FEAT.match(c).group(2) for cols in self.groups.values() for c in cols})
        self.median = F.median(numeric_only=True)
        return self

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        G = F.copy()
        for p in self.portals:   # flag BEFORE filling
            cols = [f"q_p{p}_lag0", f"v_p{p}_lag0"]
            G[f"miss_p{p}"] = G[[c for c in cols if c in G]].isna().any(axis=1).astype(int)
        for key, cols in self.groups.items():
            norm = pd.concat([G[c] / self.ratio[c] for c in cols], axis=1).mean(axis=1)
            for c in cols:
                G[c] = G[c].fillna(norm * self.ratio[c])
        for c in [c for c in G.columns if c.startswith("dv_")]:
            up, dn = c.split("_")[1:]
            G[c] = G[c].fillna(G[f"v_{up}_lag0"] - G[f"v_{dn}_lag0"])
        for c in [c for c in G.columns if c.endswith("_d15")]:
            G[c] = G[c].fillna(0.0)
        xcols = [c for c in G.columns if not c.startswith("y_")]
        G[xcols] = G[xcols].fillna(self.median.reindex(xcols))
        return G


# ----------------------------------------------------------------------------
# 9. Congestion features (notebook A4). All use only data stamped <= t.
#    Names follow the portal pattern "<q|v|r>_p<id>_<suffix>" so that
#    portal_feature_map / mask_augment blank them together with the portal.
# ----------------------------------------------------------------------------
def _grid_reindex(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    grid = pd.date_range(df.index.min().normalize(), df.index.max().normalize() + pd.Timedelta(days=1), freq=freq)
    return df.reindex(grid)


def nowcast_features(wide_1min: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Within-bin dynamics from the 1-min portal data (the 5-min bins average them away).
    For every portal p, at each 5-min stamp t in `index`:
      v_p_1mlast   speed in the minute ending at t (km/h)
      v_p_1mmin5   lowest 1-min speed in the last 5 min  -> catches a breakdown starting inside the bin
      v_p_1mslope5 speed change over the last 5 minutes
      v_p_1mstd15  std of 1-min speed over the last 15 min -> stop-and-go intensity
      q_p_1mlast   flow in the minute ending at t (veh/h)"""
    g = _grid_reindex(wide_1min, "1min")
    out = {}
    for k in PORTALS_DOWNSTREAM:
        p = PORTAL_COL[k]; v, q = g[f"v_{p}"], g[f"q_{p}"]
        out[f"v_{p}_1mlast"] = v
        out[f"v_{p}_1mmin5"] = v.rolling(5, min_periods=3).min()
        out[f"v_{p}_1mslope5"] = v - v.shift(5)
        out[f"v_{p}_1mstd15"] = v.rolling(15, min_periods=8).std()
        out[f"q_{p}_1mlast"] = q
    return pd.DataFrame(out, index=g.index).reindex(index)


def congestion_state_features(wide: pd.DataFrame, reg: "RegimeModel", index: pd.DatetimeIndex,
                              res_min: int = 5, cap_min: int = 240) -> pd.DataFrame:
    """Where are we in the congestion cycle? For every portal p at time t:
      v_p_d15     speed change over the last 15 min (km/h)
      r_p_run     minutes in the current regime, + if congested / − if free (capped at ±cap_min)
      r_p_today   minutes congested so far today (a 'bad day' indicator)"""
    d = _grid_reindex(wide, f"{res_min}min")
    day = (d.index - pd.Timedelta(minutes=res_min)).normalize()
    out = {}
    for k in PORTALS_DOWNSTREAM:
        p = PORTAL_COL[k]; v, q = d[f"v_{p}"], d[f"q_{p}"]
        out[f"v_{p}_d15"] = v - v.shift(15 // res_min)
        c = pd.Series(np.where(v.notna(), (reg.proba(p, q, v) >= .5).astype(float), np.nan), index=d.index)
        cc = c.ffill(limit=2)
        grp = (cc != cc.shift()).cumsum()
        run = (cc.groupby(grp).cumcount() + 1) * res_min
        out[f"r_{p}_run"] = np.where(cc == 1, run, -run).clip(-cap_min, cap_min)
        out[f"r_{p}_run"] = pd.Series(out[f"r_{p}_run"], index=d.index).where(cc.notna())
        out[f"r_{p}_today"] = (cc == 1).astype(float).groupby(day).cumsum() * res_min
    return pd.DataFrame(out, index=d.index).reindex(index)


def anomaly_features(wide: pd.DataFrame, profile: pd.DataFrame, index: pd.DatetimeIndex,
                     res_min: int = 5) -> pd.DataFrame:
    """Is today busier or quieter than normal? Deviation of the observed flow from the
    training historical profile (same day type and time of day), averaged over the
    last 60 and 180 minutes, for every portal: q_p_anom60, q_p_anom180 (veh/h)."""
    d = _grid_reindex(wide, f"{res_min}min")
    tod = d.index.hour * 60 + d.index.minute
    key = pd.MultiIndex.from_arrays([day_type(pd.Series(d.index, index=d.index)), tod])
    out = {}
    for k in PORTALS_DOWNSTREAM:
        p = PORTAL_COL[k]
        res = d[f"q_{p}"] - profile[f"q_{p}"].reindex(key).to_numpy()
        out[f"q_{p}_anom60"] = res.rolling(60 // res_min, min_periods=30 // res_min).mean()
        out[f"q_{p}_anom180"] = res.rolling(180 // res_min, min_periods=90 // res_min).mean()
    return pd.DataFrame(out, index=d.index).reindex(index)


def corridor_features(F: pd.DataFrame) -> pd.DataFrame:
    """Corridor-wide summaries computed from the per-portal columns that are present
    (so they stay consistent when portals are blanked): number of congested portals
    (queue extent), lowest speed on the corridor, mean flow anomaly, mean 1-min speed drop."""
    F = F.copy()
    ps = [PORTAL_COL[k] for k in PORTALS_DOWNSTREAM]
    r = F[[f"r_{p}_lag0" for p in ps]]
    F["corr_n_cong"] = (r >= .5).sum(axis=1).where(r.notna().any(axis=1))
    F["corr_vmin"] = F[[f"v_{p}_lag0" for p in ps]].min(axis=1)
    if f"q_{ps[0]}_anom60" in F:
        F["corr_anom60"] = F[[f"q_{p}_anom60" for p in ps]].mean(axis=1)
    if f"v_{ps[0]}_1mslope5" in F:
        F["corr_v1m_drop"] = F[[f"v_{p}_1mslope5" for p in ps]].min(axis=1)
    return F


def seq_frame(F: pd.DataFrame, channels=("q", "v", "k", "r"), res_min: int = 5) -> pd.DataFrame:
    """Per-portal 5-min series (the lag-0 feature columns) on a regular time grid (gaps = NaN)."""
    W = pd.DataFrame({f"{c}_{PORTAL_COL[k]}": F[f"{c}_{PORTAL_COL[k]}_lag0"]
                      for c in channels for k in PORTALS_DOWNSTREAM}).sort_index()
    W = W[~W.index.duplicated()]
    return W.reindex(pd.date_range(W.index.min(), W.index.max(), freq=f"{res_min}min"))


def build_sequences(F: pd.DataFrame, stats, T: int = 12, channels=("q", "v", "k", "r")) -> np.ndarray:
    """Input tensor for sequence networks: for every row of F (prediction time t) the last T
    5-min steps (t−5(T−1) … t) of every portal -> array [n, T, 8 portals, len(channels)+1].
    Channels are z-scored with the TRAINING statistics `stats` = (mean, std); the extra last
    channel is 1 where the portal's flow or speed is missing (the value itself is set to 0 = mean)."""
    G = seq_frame(F, channels)
    Z = (G - stats[0]) / stats[1]
    A = Z.to_numpy(np.float32).reshape(len(G), len(channels), len(PORTALS_DOWNSTREAM)).transpose(0, 2, 1)
    miss = np.isnan(A[..., :2]).any(-1, keepdims=True).astype(np.float32)
    A = np.concatenate([np.nan_to_num(A, nan=0.0), miss], -1)
    pad = np.zeros((T - 1,) + A.shape[1:], np.float32); pad[..., -1] = 1.0
    A = np.concatenate([pad, A])
    pos = G.index.get_indexer(F.index) + T - 1
    return A[pos[:, None] + np.arange(-T + 1, 1)[None, :]]


def nn_static(F: pd.DataFrame, target_p: str):
    """Non-sequence inputs: cyclic time of day (target time), weekday, off-day flag, historical
    profile of all portals at the target time (veh/h / 1000) and the current flow at the target
    portal ('level', falls back to the profile when missing). Returns (S [n, 14], level)."""
    tod = F["tod_target_min"].to_numpy() / 1440 * 2 * np.pi
    lvl = F[f"q_{target_p}_lag0"].fillna(F[f"hist_q_{target_p}"])
    S = np.c_[np.sin(tod), np.cos(tod), F["dow_sin"], F["dow_cos"], F["is_offday"],
              F[[f"hist_q_{PORTAL_COL[k]}" for k in PORTALS_DOWNSTREAM]] / 1000, lvl / 1000]
    return S.astype(np.float32), lvl


def mask_sequences(X: np.ndarray, S: np.ndarray, portals=None, frac_values: float = 0.0, seed: int = 0):
    """Simulated sensor failures for sequence inputs: whole portals (list of indices) and/or a
    random fraction of single (time, portal) values are set to missing (value 0, flag 1).
    If the target portal (index 7 = 55.620) is missing at the last step, the 'level' input falls
    back to its historical profile (S[:, 12]), exactly as in operation."""
    X, S = X.copy(), S.copy()
    if portals:
        X[:, :, portals, :-1] = 0.0; X[:, :, portals, -1] = 1.0
    if frac_values > 0:
        M = np.random.default_rng(seed).random(X.shape[:3]) < frac_values
        X[M, :-1] = 0.0; X[M, -1] = 1.0
    gone = X[:, -1, 7, -1] == 1.0
    S[gone, 13] = S[gone, 12]
    return X, S


def augment_sequences(X: np.ndarray, S: np.ndarray, n_copies: int = 2, p_portal=(0.0, 0.9),
                      p_value=(0.0, 0.3), seed: int = 0):
    """Masking augmentation for sequence networks (same scheme as mask_augment): each copy gets,
    per sample, a random portal-failure rate and a random single-value dropout rate."""
    rng = np.random.default_rng(seed); Xs, Ss = [X], [S]
    for _ in range(n_copies):
        Xc, Sc = X.copy(), S.copy(); n = len(X)
        down = rng.random((n, X.shape[2])) < rng.uniform(*p_portal, n)[:, None]           # [n, portal]
        M = down[:, None, :] | (rng.random(X.shape[:3]) < rng.uniform(*p_value, n)[:, None, None])
        Xc[M, :-1] = 0.0; Xc[M, -1] = 1.0
        gone = Xc[:, -1, 7, -1] == 1.0; Sc[gone, 13] = Sc[gone, 12]
        Xs.append(Xc); Ss.append(Sc)
    return np.concatenate(Xs), np.concatenate(Ss)


def online_bias_correction(pred: pd.Series, y_obs: pd.Series, horizon_steps: int = 3,
                           halflife_steps: float = 6.0, weight: float = 1.0, res_min: int = 5) -> pd.Series:
    """Causal post-processing: subtract an exponentially weighted mean of the model's own
    recent errors. The error of a prediction made at s is only known at s + horizon, so at
    time t only errors of predictions made at s <= t - horizon are used. Gaps in time
    (missing days) decay the memory automatically (weights follow clock time)."""
    grid = pd.date_range(pred.index.min(), pred.index.max(), freq=f"{res_min}min")
    err = (pred - y_obs).reindex(grid)
    known = err.shift(horizon_steps)
    bias = known.ewm(halflife=halflife_steps, ignore_na=False).mean()
    return pred - weight * bias.reindex(pred.index).fillna(0.0)

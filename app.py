# app.py
import math
import random
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="VIP Completions – Risk & Pipeline", layout="wide")

# -------------------------------
# Controls (left sidebar)
# -------------------------------
st.sidebar.header("Simulation / Capacity Settings")

trials = st.sidebar.slider("Monte Carlo trials", 500, 5000, 2000, 500)
hrs_per_fte = st.sidebar.slider("Hours per FTE per week", 30, 60, 40, 1)
prod_mean = st.sidebar.slider("Productivity mean (μ)", 0.80, 1.10, 0.90, 0.01)
prod_sd = st.sidebar.slider("Productivity stdev (σ)", 0.00, 0.20, 0.06, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("Pipeline Stage Probabilities")
# PDR/CDR/FDR stage defaults — tweak to match your funnel
stage_prob_default = {
    "PDR": st.sidebar.slider("PDR win probability", 0.05, 0.60, 0.25, 0.05),
    "CDR": st.sidebar.slider("CDR win probability", 0.20, 0.80, 0.55, 0.05),
    "FDR": st.sidebar.slider("FDR win probability", 0.40, 0.95, 0.80, 0.05),
    "COMMITTED": 1.0,
}

st.sidebar.markdown("---")
wip_limit_large = st.sidebar.slider("WIP limit: max concurrent LARGE interiors", 1, 6, 3, 1)

# -------------------------------
# Example VIP completions dataset
# -------------------------------
# Departments & headcounts (very rough demo values)
depts = pd.DataFrame([
    {"dept": "Interiors",  "headcount": 12},
    {"dept": "Engineering","headcount": 8},
    {"dept": "Cabinet",    "headcount": 7},
    {"dept": "Upholstery", "headcount": 6},
    {"dept": "Finish",     "headcount": 6},
    {"dept": "Avionics",   "headcount": 10},
    {"dept": "Structures", "headcount": 6},
    {"dept": "Maintenance","headcount": 10},
    {"dept": "Inspection", "headcount": 7},
])

dept_keys = depts["dept"].tolist()
headcount_map = dict(zip(depts["dept"], depts["headcount"]))

# Helper: classify size for WIP
def classify_aircraft(model: str) -> str:
    s = (model or "").upper()
    if any(k in s for k in ["B777", "A330", "A340", "B747", "A350"]):
        return "LARGE"
    if any(k in s for k in ["B757"]):
        return "M757"
    # Biz/VIP narrowbody
    return "SMALL"

# Confirmed / in-execution programs (dates in ISO)
confirmed = pd.DataFrame([
    {
        "number": "P8801", "customer": "Alpha Star", "aircraft": "B777-200LR VIP",
        "scope": "Green Completion – full cabin + CMS/IFE + monuments",
        "value_usd": 138_000_000,
        "induction": "2025-10-27", "delivery": "2026-07-31",
        # hours by dept (demo apportionment)
        "Interiors": 34000, "Engineering": 11000, "Cabinet": 9000, "Upholstery": 6200, "Finish": 8000,
        "Avionics": 5200, "Structures": 2500, "Maintenance": 4200, "Inspection": 2200,
        # supply-chain gate (workdays)
        "parts_p50": 60, "parts_p90": 90, "single_source": True,
    },
    {
        "number": "P8744", "customer": "Sands", "aircraft": "BBJ2",
        "scope": "Cabin refresh + CMS/IFE refit + Starlink",
        "value_usd": 32_000_000,
        "induction": "2025-11-17", "delivery": "2026-03-10",
        "Interiors": 12000, "Engineering": 3800, "Cabinet": 2500, "Upholstery": 2100, "Finish": 2600,
        "Avionics": 3000, "Structures": 600, "Maintenance": 1800, "Inspection": 900,
        "parts_p50": 35, "parts_p90": 55, "single_source": True,
    },
    {
        "number": "P8795", "customer": "Valkyrie", "aircraft": "ACJ319",
        "scope": "Bedroom reconfig + monuments + certs",
        "value_usd": 21_000_000,
        "induction": "2025-12-08", "delivery": "2026-04-15",
        "Interiors": 9000, "Engineering": 2600, "Cabinet": 1700, "Upholstery": 1200, "Finish": 1650,
        "Avionics": 1100, "Structures": 500, "Maintenance": 1200, "Inspection": 700,
        "parts_p50": 28, "parts_p90": 45, "single_source": False,
    },
])

# Pipeline / not yet won (stage = PDR/CDR/FDR)
pipeline = pd.DataFrame([
    {
        "number": "LQ901", "customer": "Polaris", "aircraft": "A330-200 VIP",
        "scope": "Green Completion – full cabin + shower + CMS/IFE",
        "value_usd": 155_000_000, "stage": "PDR",
        "target_induction": "2026-02-02", "duration_weeks_guess": 42,
        "Interiors": 39000, "Engineering": 12500, "Cabinet": 9800, "Upholstery": 7000, "Finish": 9000,
        "Avionics": 6500, "Structures": 3200, "Maintenance": 4500, "Inspection": 2400,
        "parts_p50": 70, "parts_p90": 100, "single_source": True,
    },
    {
        "number": "LQ912", "customer": "Celestial", "aircraft": "BBJ",
        "scope": "Cabin refresh + monuments + CMS upgrade",
        "value_usd": 28_000_000, "stage": "CDR",
        "target_induction": "2026-01-12", "duration_weeks_guess": 18,
        "Interiors": 10500, "Engineering": 3200, "Cabinet": 2100, "Upholstery": 1800, "Finish": 2000,
        "Avionics": 1700, "Structures": 500, "Maintenance": 1400, "Inspection": 800,
        "parts_p50": 30, "parts_p90": 50, "single_source": True,
    },
    {
        "number": "LQ927", "customer": "Ty Air", "aircraft": "ACJ320",
        "scope": "CMS/IFE refit + monuments cert update",
        "value_usd": 36_000_000, "stage": "FDR",
        "target_induction": "2025-12-15", "duration_weeks_guess": 20,
        "Interiors": 11500, "Engineering": 3600, "Cabinet": 2300, "Upholstery": 1700, "Finish": 2200,
        "Avionics": 2500, "Structures": 600, "Maintenance": 1600, "Inspection": 900,
        "parts_p50": 32, "parts_p90": 48, "single_source": False,
    },
])

for df in (confirmed, pipeline):
    df["class"] = df["aircraft"].apply(classify_aircraft)

# -------------------------------
# Helpers
# -------------------------------
def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)

def mondays_between(a: date, b: date):
    """Inclusive list of Mondays between a and b."""
    start = a - timedelta(days=a.weekday())  # monday on/ before a
    cur = start
    out = []
    while cur <= b:
        out.append(cur)
        cur += timedelta(days=7)
    return out

def workdays_between(a: date, b: date) -> int:
    """Mon–Fri inclusive count."""
    if b < a: return 0
    days = (b - a).days + 1
    weeks, rem = divmod(days, 7)
    wds = weeks * 5
    for i in range(rem):
        if (a + timedelta(days=i)).weekday() < 5:
            wds += 1
    return wds

@lru_cache(None)
def dept_capacity_per_week(dept: str) -> float:
    return headcount_map.get(dept, 0) * hrs_per_fte * prod_mean

def project_duration_weeks(row, prod_factor: float = 1.0) -> float:
    """Rough ‘critical’ duration = max over depts of (hours / cap)."""
    max_weeks = 0.0
    for d in dept_keys:
        hours = float(row.get(d, 0) or 0)
        cap = headcount_map.get(d, 0) * hrs_per_fte * prod_factor
        if cap > 0:
            max_weeks = max(max_weeks, hours / cap)
    return max_weeks

# -------------------------------
# Monte Carlo for confirmed jobs
# -------------------------------
rng = np.random.default_rng(42)

def sample_prod(size):
    # Truncated normal around mean (μ, σ), clamp to [μ-3σ, μ+3σ], min 0.5, max 1.2
    s = rng.normal(loc=prod_mean, scale=prod_sd, size=size)
    s = np.clip(s, prod_mean - 3*prod_sd, prod_mean + 3*prod_sd)
    return np.clip(s, 0.5, 1.2)

def sample_scope(hours):
    # Triangular: -10% to +20%, mode at +5%
    lo = 0.90 * hours
    hi = 1.20 * hours
    mode = 1.05 * hours
    return rng.triangular(left=lo, mode=mode, right=hi)

def parts_gate_days(p50, p90, size):
    # Lognormal calibrated from p50/p90-ish (simple demo)
    # Ensure positive; rough spread to hit p90~ given
    median = p50
    sigma = max(0.1, math.log(max(1.01, p90 / max(1, p50))) / 1.281)  # crude
    mu = math.log(max(1, median)) - 0.5 * sigma ** 2
    return np.maximum(0, rng.lognormal(mean=mu, sigma=sigma, size=size))

def simulate_project(row) -> dict:
    a = to_date(row["induction"])
    b = to_date(row["delivery"])
    horizon_wd = workdays_between(a, b)
    if horizon_wd <= 0:
        return {"p_on_time": 0.0, "p50_finish": b, "p90_finish": b, "driver": "N/A"}

    # Per-trial productivity factor
    prod = sample_prod(trials)

    # Scale departmental hours with triangular noise
    dept_hours = {d: sample_scope(float(row.get(d, 0) or 0)) for d in dept_keys}

    # Per-trial critical duration (weeks)
    durations = []
    for i in range(trials):
        max_weeks = 0.0
        for d in dept_keys:
            cap = headcount_map.get(d, 0) * hrs_per_fte * prod[i]
            if cap > 0:
                max_weeks = max(max_weeks, dept_hours[d] / cap)
        durations.append(max_weeks)
    durations = np.array(durations)

    # Convert to workdays (5 wd / wk)
    dur_wd = durations * 5.0

    # Parts gate: need latest critical component before real start
    gate = parts_gate_days(row["parts_p50"], row["parts_p90"], trials)

    finish_wd = gate + dur_wd
    p_on_time = float(np.mean(finish_wd <= horizon_wd))

    # P50/P90 finish date
    p50_days = int(np.percentile(finish_wd, 50))
    p90_days = int(np.percentile(finish_wd, 90))

    def add_workdays(d, wd):
        cur = d
        left = wd
        while left > 0:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                left -= 1
        return cur

    p50_finish = add_workdays(a, p50_days)
    p90_finish = add_workdays(a, p90_days)

    # Driver: whichever contributes most to critical path (avg share)
    # Approx by recomputing deterministic at μ and checking max dept duration
    driver = "Parts"
    max_weeks_mu = 0
    for d in dept_keys:
        cap_mu = headcount_map.get(d, 0) * hrs_per_fte * prod_mean
        if cap_mu > 0:
            w = (row.get(d, 0) or 0) / cap_mu
            if w > max_weeks_mu:
                max_weeks_mu = w
                driver = d
    # If parts gate median > 25% of total median duration, keep 'Parts'
    if np.median(gate) < 0.25 * np.median(finish_wd):
        pass  # driver already set by dept
    else:
        driver = "Parts"

    return {
        "p_on_time": p_on_time,
        "p50_finish": p50_finish,
        "p90_finish": p90_finish,
        "driver": driver
    }

# Run MC on confirmed
sim_rows = []
for _, r in confirmed.iterrows():
    s = simulate_project(r)
    rar = r["value_usd"] * (1.0 - s["p_on_time"])
    sim_rows.append({
        "number": r["number"], "customer": r["customer"], "aircraft": r["aircraft"],
        "class": r["class"], "scope": r["scope"], "value_usd": r["value_usd"],
        "induction": r["induction"], "delivery": r["delivery"],
        "p_on_time": s["p_on_time"], "p50_finish": s["p50_finish"], "p90_finish": s["p90_finish"],
        "driver": s["driver"], "Revenue_at_Risk": rar,
        "late_risk_index": 1.0 if s["p90_finish"] > to_date(r["delivery"]) else 0.0,
        "single_source": bool(r["single_source"]),
        "exposure_score": (100 * (1 - s["p_on_time"])) + (20 if r["single_source"] else 0),
    })
confirmed_risk = pd.DataFrame(sim_rows)

# -------------------------------
# Scorecard
# -------------------------------
st.title("VIP Completions – Risk & Pipeline (PDR/CDR/FDR)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Backlog Value", f"${confirmed_risk['value_usd'].sum():,.0f}")
with col2:
    ov = (confirmed_risk["p_on_time"] * confirmed_risk["value_usd"]).sum() / max(1, confirmed_risk["value_usd"].sum())
    st.metric("Overall On-Time Probability", f"{ov*100:,.1f}%")
with col3:
    st.metric("Revenue at Risk", f"${confirmed_risk['Revenue_at_Risk'].sum():,.0f}")
with col4:
    st.metric("Programs < 60% On-Time", int((confirmed_risk["p_on_time"] < 0.60).sum()))

st.subheader("Per-Program Risk Overview")
st.dataframe(
    confirmed_risk.sort_values("Revenue_at_Risk", ascending=False)[
        ["number","customer","aircraft","value_usd","p_on_time","Revenue_at_Risk","driver","late_risk_index","single_source","exposure_score"]
    ],
    use_container_width=True
)

# -------------------------------
# Risk visuals
# -------------------------------
st.subheader("Risk Heatmap (Likelihood vs Impact) + Driver Pareto")

bubble = px.scatter(
    confirmed_risk,
    x=confirmed_risk["p_on_time"].rsub(1),  # 1 - p_on_time
    y="Revenue_at_Risk",
    size="value_usd",
    color="driver",
    hover_data=["number","customer","aircraft","p_on_time"],
    labels={"x":"Likelihood Late (1 − p_on_time)", "Revenue_at_Risk":"Impact ($)"},
)
bubble.update_layout(height=420, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(bubble, use_container_width=True)

by_driver = confirmed_risk.groupby("driver", as_index=False)["Revenue_at_Risk"].sum().sort_values("Revenue_at_Risk", ascending=False)
pareto_cum = by_driver["Revenue_at_Risk"].cumsum() / by_driver["Revenue_at_Risk"].sum() * 100.0
fig_p = go.Figure()
fig_p.add_bar(x=by_driver["driver"], y=by_driver["Revenue_at_Risk"], name="RAR")
fig_p.add_scatter(x=by_driver["driver"], y=pareto_cum, name="Cumulative %", yaxis="y2")
fig_p.update_layout(
    height=420, margin=dict(l=0,r=0,t=10,b=0),
    yaxis=dict(title="Revenue at Risk ($)"),
    yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0,110])
)
st.plotly_chart(fig_p, use_container_width=True)

# -------------------------------
# Pipeline EV with PDR/CDR/FDR
# -------------------------------
st.subheader("Pipeline – Probability-Weighted by Stage (PDR → CDR → FDR)")

pipe = pipeline.copy()
pipe["stage_prob"] = pipe["stage"].map(stage_prob_default).fillna(0.0)
pipe["EV_usd"] = pipe["value_usd"] * pipe["stage_prob"]

colA, colB = st.columns([2,1])
with colA:
    fig_pipe = px.bar(
        pipe.sort_values("EV_usd", ascending=False),
        x="number", y="EV_usd", color="stage",
        hover_data=["customer","aircraft","value_usd","stage_prob"],
        labels={"EV_usd":"Expected Value ($)"}
    )
    fig_pipe.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_pipe, use_container_width=True)
with colB:
    stage_sum = pipe.groupby("stage", as_index=False)["EV_usd"].sum()
    fig_funnel = px.funnel(stage_sum, x="EV_usd", y="stage")
    fig_funnel.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_funnel, use_container_width=True)

st.caption("Tip: adjust stage probabilities in the sidebar to match your historical win rates (PDR < CDR < FDR < Committed).")

# -------------------------------
# WIP limits – Large interiors concurrency
# -------------------------------
st.subheader(f"WIP – Concurrent LARGE Interiors vs Policy Limit ({wip_limit_large})")

# Build a simple weekly timeline for confirmed
def weeks_span(a: date, weeks: float):
    end = a + timedelta(days=int(round(weeks*7)))
    return mondays_between(a, end)

timeline_rows = []
today = date.today()
for _, r in confirmed.iterrows():
    a = to_date(r["induction"])
    # deterministic duration at μ productivity (weeks)
    dur_wk = project_duration_weeks(r, prod_factor=prod_mean)
    weeks = weeks_span(a, dur_wk)
    for w in weeks:
        timeline_rows.append({"week": w, "number": r["number"], "class": r["class"], "is_large": r["class"] == "LARGE"})
tl = pd.DataFrame(timeline_rows)
if tl.empty:
    st.info("No confirmed programs on the timeline yet.")
else:
    wk = tl.groupby("week", as_index=False).agg(concurrent_large=("is_large", "sum"))
    wk = wk.sort_values("week")
    fig_wip = go.Figure()
    fig_wip.add_scatter(x=wk["week"], y=wk["concurrent_large"], mode="lines+markers", name="Concurrent LARGE")
    fig_wip.add_hline(y=wip_limit_large, line_dash="dash", line_color="red", annotation_text="WIP limit")
    fig_wip.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(title="Count"))
    st.plotly_chart(fig_wip, use_container_width=True)

# -------------------------------
# Earliest available induction window (demo)
# -------------------------------
st.subheader("Earliest Available Induction (respects LARGE WIP limit)")

# pick highest-EV pipeline candidate
if pipe.empty:
    st.info("No pipeline items.")
else:
    candidate = pipe.sort_values("EV_usd", ascending=False).iloc[0]
    cand_txt = f"{candidate['number']} · {candidate['customer']} · {candidate['aircraft']} · stage {candidate['stage']} · EV ${candidate['EV_usd']:,.0f}"
    st.write("Candidate:", cand_txt)

    # Estimate its duration at μ productivity
    dur_cand_wk = project_duration_weeks(candidate, prod_factor=prod_mean)
    cand_is_large = classify_aircraft(candidate["aircraft"]) == "LARGE"

    # Start searching next Monday out 20 weeks
    start_seed = next_monday(today)
    search_weeks = [start_seed + timedelta(days=7*i) for i in range(20)]

    def occupancy_with_candidate(start_date: date) -> bool:
        """Return True if adding candidate starting start_date never exceeds WIP limit across its run."""
        if not cand_is_large:
            return True  # only gating LARGE
        end_date = start_date + timedelta(days=int(round(dur_cand_wk*7)))
        cand_weeks = set(mondays_between(start_date, end_date))
        # Merge with existing wk concurrency
        wk_map = {w:0 for w in sorted(set(wk["week"]).union(cand_weeks))} if not tl.empty else {w:0 for w in cand_weeks}
        if not tl.empty:
            for _, row in wk.iterrows():
                wk_map[row["week"]] = int(row["concurrent_large"])
        # Add the candidate
        for w in cand_weeks:
            wk_map[w] = wk_map.get(w, 0) + 1
            if wk_map[w] > wip_limit_large:
                return False
        return True

    chosen = None
    for s in search_weeks:
        if occupancy_with_candidate(s):
            chosen = s
            break

    if chosen is None:
        st.warning("No feasible start within the next 20 weeks without breaching the LARGE WIP limit.")
    else:
        est_end = chosen + timedelta(days=int(round(dur_cand_wk*7)))
        st.success(f"Earliest feasible induction: **{chosen.isoformat()}**, with rough P50 delivery **{est_end.isoformat()}** (capacity @ μ).")

# -------------------------------
# Notes on realism
# -------------------------------
with st.expander("What makes this VIP-specific?"):
    st.markdown("""
- **Aircraft mix**: BBJ/ACJ and widebodies (A330/B777) with full green completions and heavy monument work.
- **Scopes**: CMS/IFE, Starlink, monuments, bedroom reconfigs, shower options, cert/burn testing implied by Engineering/Inspection load.
- **Supply-chain gates**: `parts_p50/p90` + `single_source` reflect veneer, monuments, CMS head-ends, seat tracks, etc.
- **Risk drivers**: “Parts” can dominate when p90 extends past the needed-by, otherwise the longest departmental duration (often **Interiors**).
- **WIP policy**: caps concurrent **LARGE** interiors programs (shop, tooling, metrology, and QA bandwidth).
""")

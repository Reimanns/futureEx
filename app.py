# exec_risk_demo.py
# Streamlit demo: Exec scorecard, Monte Carlo schedule risk, supply chain flags, risk heatmap,
# pipeline EV, WIP limits, and earliest induction window (illustrative logic).
# No external data needed.

import math
from datetime import date, datetime, timedelta
from functools import lru_cache
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page & seed -------------------------
st.set_page_config(page_title="Program Risk & Pipeline Demo", layout="wide")
np.random.seed(42)  # deterministic demo

# ------------------------- Helpers (dates/workdays) -------------------------
def to_date(x):
    if isinstance(x, (date, datetime)):  # already a date-like
        return date(x.year, x.month, x.day)
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()

def is_workday(d: date) -> bool:
    return d.weekday() < 5  # Mon-Fri

def add_workdays(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    days_left = abs(n)
    cur = d
    while days_left:
        cur += timedelta(days=step)
        if is_workday(cur):
            days_left -= 1
    return cur

def workdays_between(a: date, b: date) -> int:
    """Inclusive workdays between a..b with a<=b; returns 0 if a>b."""
    if a > b: return 0
    days = 0
    cur = a
    while cur <= b:
        if is_workday(cur):
            days += 1
        cur += timedelta(days=1)
    return days

def sunday_before(d: date) -> date:
    return d - timedelta(days=(d.weekday() + 1) % 7)

def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())

# ------------------------- Sample capacity model -------------------------
HOURS_PER_FTE = 40
PRODUCTIVITY_MEAN = 0.85
PRODUCTIVITY_SD = 0.05
PRODUCTIVITY_MIN = 0.70
PRODUCTIVITY_MAX = 1.00

DEPTS = [
    {"key": "Maintenance", "headcount": 36},
    {"key": "Structures",  "headcount": 22},
    {"key": "Avionics",    "headcount": 15},
    {"key": "Inspection",  "headcount": 10},
    {"key": "Interiors",   "headcount": 11},
    {"key": "Engineering", "headcount": 7},
    {"key": "Cabinet",     "headcount": 3},
    {"key": "Upholstery",  "headcount": 7},
    {"key": "Finish",      "headcount": 6},
]
DEPT_KEYS = [d["key"] for d in DEPTS]
HC = {d["key"]: d["headcount"] for d in DEPTS}

def weekly_capacity_by_dept(prod_factor: float) -> dict:
    return {k: HC[k] * HOURS_PER_FTE * prod_factor for k in DEPT_KEYS}

# ------------------------- Sample projects (confirmed) -------------------------
TODAY = date.today()  # use your actual "today"
# For stable demo, you can lock TODAY if you want: TODAY = date(2025, 11, 5)

PROJECTS = [
    dict(number="P7611", customer="Alpha Star", aircraftModel="A340", value=6_200_000,
         induction="2025-10-20", delivery="2025-12-04",
         hours={"Maintenance":2432.2,"Structures":1253.0,"Avionics":737.0,"Inspection":1474.1,
                "Interiors":1474.1,"Engineering":0,"Cabinet":0,"Upholstery":0,"Finish":0},
         parts=[{"pn":"CMS-CTRL","lead_p50_days":18,"p90_days":28,"required_by":"2025-11-10","single_source":True,"critical":True}]
    ),
    dict(number="P7706", customer="Valkyrie", aircraftModel="B737-MAX", value=3_800_000,
         induction="2025-10-31", delivery="2025-11-25",
         hours={"Maintenance":123.3,"Structures":349.4,"Avionics":493.2,"Inspection":164.4,
                "Interiors":698.7,"Engineering":143.8,"Cabinet":61.6,"Upholstery":0,"Finish":20.6},
         parts=[{"pn":"SAT-MODEM","lead_p50_days":12,"p90_days":20,"required_by":"2025-11-10","single_source":False,"critical":True}]
    ),
    dict(number="P7712", customer="Ty Air", aircraftModel="B737", value=4_500_000,
         induction="2025-11-04", delivery="2025-12-21",
         hours={"Maintenance":893.0,"Structures":893.0,"Avionics":476.3,"Inspection":238.1,
                "Interiors":3453.0,"Engineering":0,"Cabinet":0,"Upholstery":0,"Finish":0},
         parts=[{"pn":"SEAT-FOAM","lead_p50_days":15,"p90_days":25,"required_by":"2025-11-25","single_source":True,"critical":False}]
    ),
    dict(number="P7657", customer="Kaiser", aircraftModel="B737", value=1_200_000,
         induction="2025-11-15", delivery="2025-11-25",
         hours={"Maintenance":93.6,"Structures":240.6,"Avionics":294.1,"Inspection":120.3,
                "Interiors":494.6,"Engineering":80.2,"Cabinet":0,"Upholstery":0,"Finish":13.4},
         parts=[{"pn":"STARLINK-KIT","lead_p50_days":8,"p90_days":14,"required_by":"2025-11-18","single_source":False,"critical":True}]
    ),
    dict(number="P7645", customer="Kaiser", aircraftModel="B737", value=1_100_000,
         induction="2025-11-30", delivery="2025-12-10",
         hours={"Maintenance":93.6,"Structures":240.6,"Avionics":294.1,"Inspection":120.3,
                "Interiors":494.6,"Engineering":80.2,"Cabinet":0,"Upholstery":0,"Finish":13.4},
         parts=[{"pn":"STARLINK-KIT","lead_p50_days":8,"p90_days":14,"required_by":"2025-12-02","single_source":False,"critical":True}]
    ),
]

# ------------------------- Potential pipeline items -------------------------
PIPELINE = [
    {"number":"P7686","stage":"Negotiation","prob":0.75,"value":3_100_000,"est_induction":"2025-12-01","aircraftModel":"B777",
     "hours":{"Interiors":3800,"Structures":1200,"Inspection":600,"Maintenance":700,"Avionics":400,"Engineering":200}},
    {"number":"P7430","stage":"Proposal","prob":0.60,"value":10_000_000,"est_induction":"2025-11-10","aircraftModel":"B777",
     "hours":{"Interiors":12720,"Structures":12720,"Avionics":3180,"Inspection":3180,"Maintenance":0,"Engineering":3180}},
    {"number":"P7689","stage":"Scoping","prob":0.40,"value":6_000_000,"est_induction":"2025-12-10","aircraftModel":"B737-700",
     "hours":{"Maintenance":1200,"Structures":400,"Avionics":350,"Inspection":250,"Interiors":300}},
    {"number":"P7691","stage":"Discovery","prob":0.20,"value":5_500_000,"est_induction":"2026-01-15","aircraftModel":"B737-700",
     "hours":{"Maintenance":800,"Structures":500,"Avionics":200,"Inspection":180,"Interiors":450}},
    {"number":"P7669","stage":"Committed","prob":0.95,"value":4_000_000,"est_induction":"2025-12-08","aircraftModel":"A319-133",
     "hours":{"Maintenance":1150,"Structures":520,"Avionics":300,"Inspection":280,"Interiors":900}},
]

STAGE_ORDER = ["Discovery","Scoping","Proposal","Negotiation","Committed","Won"]
STAGE_PROBS = {"Discovery":0.20,"Scoping":0.40,"Proposal":0.60,"Negotiation":0.75,"Committed":0.95,"Won":1.00}

# ------------------------- Classification / WIP policy -------------------------
def aircraft_class(model: str) -> str:
    s = (model or "").upper()
    if s.startswith(("B777","B747","A340","A330")): return "HEAVY"
    if s.startswith("B757"): return "M757"
    if s.startswith(("B737","A319")): return "SMALL"
    return "UNKNOWN"

def large_interiors(project) -> bool:
    total_hours = sum(project["hours"].get(k,0) for k in DEPT_KEYS)
    return (
        project["hours"].get("Interiors",0) >= 3000
        or aircraft_class(project["aircraftModel"]) in {"HEAVY"}
        or total_hours >= 6000
    )

LARGE_WIP_LIMIT = 3

# ------------------------- Sidebar controls -------------------------
st.sidebar.header("Simulation Controls")
trials = st.sidebar.slider("Monte Carlo trials", 500, 10000, 4000, step=500)
HOURS_PER_FTE = st.sidebar.slider("Hours per FTE per week", 30, 60, HOURS_PER_FTE, step=1)
PRODUCTIVITY_MEAN = st.sidebar.slider("Productivity mean", 0.70, 1.00, PRODUCTIVITY_MEAN, 0.01)
PRODUCTIVITY_SD = st.sidebar.slider("Productivity stdev", 0.01, 0.20, PRODUCTIVITY_SD, 0.01)
st.sidebar.write("---")
st.sidebar.caption("Demo only: simplified uncertainty & gate logic; illustrative—not a production scheduler.")

# ------------------------- Monte Carlo engine -------------------------
def clip(x, lo, hi): return max(lo, min(hi, x))

def draw_productivity(n):
    x = np.random.normal(PRODUCTIVITY_MEAN, PRODUCTIVITY_SD, size=n)
    return np.clip(x, PRODUCTIVITY_MIN, PRODUCTIVITY_MAX)

def draw_hours_triangular(base_hours: float, n: int):
    # ~10% optimistic to +20% pessimistic spread
    a = 0.9 * base_hours
    c = 1.2 * base_hours
    b = base_hours
    return np.random.triangular(a, b, c, size=n)

def parts_gate_ready_samples(parts, induction: date, n: int):
    """Return n sample dates when parts gate is ready (max across critical parts)."""
    if not parts:
        return np.array([induction]*n)
    samples = []
    for p in parts:
        p50 = int(p.get("lead_p50_days", 0))
        p90 = int(p.get("p90_days", p50 + 5))
        # Draw lead times with right tail: Triangular with (lo=0.7*p50, mode=p50, hi=p90)
        lo = max(0, int(0.7 * p50))
        lt = np.random.triangular(lo, p50, p90, size=n)
        dates = np.array([add_workdays(induction, int(round(x))) for x in lt])
        samples.append(dates)
    # Gate ready when all critical parts are in (max date)
    arr = np.array(samples)  # shape: [parts, n]
    # Convert to ordinal for max/compare
    ords = np.vectorize(lambda d: d.toordinal())(arr)
    max_ord = ords.max(axis=0)
    return np.vectorize(lambda o: date.fromordinal(int(o)))(max_ord)

def duration_workdays_for_hours(hours_by_dept: dict, prod: np.ndarray) -> np.ndarray:
    """Critical-path-ish duration: max over dept durations (hours / cap_per_day)."""
    cap_week = {k: HC[k] * HOURS_PER_FTE * prod for k in DEPT_KEYS}  # each element is an array
    durations = []
    for k in DEPT_KEYS:
        base_h = float(hours_by_dept.get(k, 0.0))
        if base_h <= 0:
            continue
        h_samp = draw_hours_triangular(base_h, len(prod))
        # hours → workdays: (hours / (weekly_cap/5))
        wd = 5.0 * (h_samp / np.maximum(1e-6, cap_week[k]))
        durations.append(wd)
    if not durations:
        return np.zeros_like(prod)
    dur = np.max(np.vstack(durations), axis=0)  # critical driver is the max
    # Round up to integers
    return np.ceil(dur)

def simulate_project(project, ntrials: int):
    ind = to_date(project["induction"])
    due = to_date(project["delivery"])
    prod = draw_productivity(ntrials)
    dur_wd = duration_workdays_for_hours(project["hours"], prod)
    gate_dates = parts_gate_ready_samples(project.get("parts", []), ind, ntrials)
    start_dates = np.maximum(np.vectorize(lambda d: d.toordinal())(ind),
                             np.vectorize(lambda d: d.toordinal())(gate_dates))
    finish_dates = start_dates + dur_wd.astype(int)
    finish = np.vectorize(lambda o: date.fromordinal(int(o)))(finish_dates)
    on_time = (finish <= due)
    slip_days = np.maximum(0, (finish_dates - np.vectorize(lambda d: d.toordinal())(due)))
    return {
        "p_on_time": float(on_time.mean()),
        "p50_finish": pd.Series(finish).sort_values().iloc[int(0.50*ntrials)],
        "p90_finish": pd.Series(finish).sort_values().iloc[int(0.90*ntrials)],
        "exp_slip_days": float(slip_days.mean()),
        "prod_mean": float(prod.mean()),
    }

@lru_cache(maxsize=128)
def simulate_all(ntrials: int):
    rows = []
    for p in PROJECTS:
        res = simulate_project(p, ntrials)
        rows.append({
            "Project": p["number"],
            "Customer": p["customer"],
            "Value": p["value"],
            "Due": to_date(p["delivery"]),
            "p_on_time": res["p_on_time"],
            "p50_finish": res["p50_finish"],
            "p90_finish": res["p90_finish"],
            "exp_slip_days": res["exp_slip_days"],
        })
    df = pd.DataFrame(rows)
    df["Revenue_at_Risk"] = df["Value"] * (1 - df["p_on_time"])
    return df

def primary_driver(project) -> str:
    # Simple "driver": which dept contributes the max nominal duration at mean productivity
    prod = np.array([PRODUCTIVITY_MEAN])
    cap_week = {k: HC[k] * HOURS_PER_FTE * prod for k in DEPT_KEYS}
    best_k, best_wd = None, -1
    for k in DEPT_KEYS:
        h = project["hours"].get(k, 0.0)
        if h <= 0: continue
        wd = 5.0 * (h / max(1e-6, cap_week[k][0]))
        if wd > best_wd:
            best_wd = wd
            best_k = k
    # If critical parts gate clearly dominates, label "Parts" (heuristic)
    parts = project.get("parts", [])
    if parts:
        late_flags = 0
        for p in parts:
            p90 = int(p.get("p90_days", 0))
            req = to_date(p.get("required_by", project["delivery"]))
            if add_workdays(to_date(project["induction"]), p90) > req:
                late_flags += 1
        if late_flags >= 1:
            return "Parts"
    return best_k or "Mixed"

def supply_chain_flags(project):
    # Late-risk index ~ 1 if p90 > required_by, else scale down
    idx = 0.0
    ss = False
    for p in project.get("parts", []):
        p90 = int(p.get("p90_days", 0))
        req = to_date(p.get("required_by", project["delivery"]))
        p90_date = add_workdays(to_date(project["induction"]), p90)
        late = 1.0 if p90_date > req else 0.0
        idx = max(idx, late)
        ss = ss or bool(p.get("single_source", False))
    exposure = 100.0 * (0.6*idx + 0.4*(1.0 if ss else 0.0))
    return idx, ss, round(exposure,1)

# ------------------------- Scorecard (computed) -------------------------
results = simulate_all(trials)
backlog_value = results["Value"].sum()
overall_on_time_value_weighted = (results["p_on_time"] * results["Value"]).sum() / max(1e-9, backlog_value)
overall_rar = results["Revenue_at_Risk"].sum()
below_60 = int((results["p_on_time"] < 0.60).sum())

st.title("Program Risk & Pipeline Demo")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Backlog Value", f"${backlog_value:,.0f}")
c2.metric("Overall On-Time Probability", f"{overall_on_time_value_weighted*100:,.0f}%")
c3.metric("Revenue at Risk (RAR)", f"${overall_rar:,.0f}")
c4.metric("Programs < 60% On-Time", f"{below_60}")

st.caption("This demo uses illustrative uncertainty (triangular scope variance, truncated-normal productivity, parts p50/p90 gates).")

# ------------------------- Per-program table -------------------------
tbl = results.copy()
drivers = []
late_idx = []
single_src = []
exposure = []
for p in PROJECTS:
    drv = primary_driver(p); drivers.append((p["number"], drv))
    li, ss, ex = supply_chain_flags(p); late_idx.append((p["number"], li)); single_src.append((p["number"], ss)); exposure.append((p["number"], ex))

drv_map = dict(drivers)
li_map = dict(late_idx)
ss_map = dict(single_src)
ex_map = dict(exposure)
tbl["Primary Driver"] = tbl["Project"].map(drv_map)
tbl["Late-Risk Index"] = tbl["Project"].map(li_map)
tbl["Single-Source"] = tbl["Project"].map(ss_map)
tbl["Exposure (0–100)"] = tbl["Project"].map(ex_map)

st.subheader("Per-Program Risk Overview")
st.dataframe(
    tbl[["Project","Customer","Value","Due","p_on_time","Revenue_at_Risk","Primary Driver","Late-Risk Index","Single-Source","Exposure (0–100)"]]
      .sort_values("Revenue_at_Risk", ascending=False),
    use_container_width=True
)

# ------------------------- Risk Heatmap + Driver Pareto -------------------------
st.subheader("Risk Heatmap & Driver Pareto")

heat = tbl.copy()
heat["Impact ($M)"] = heat["Revenue_at_Risk"] / 1_000_000.0
heat["Likelihood"] = 1.0 - heat["p_on_time"]
heat["Size"] = heat["Value"] / 1_000_000.0

chart = alt.Chart(heat).mark_circle().encode(
    x=alt.X("Likelihood:Q", scale=alt.Scale(domain=[0,1])),
    y=alt.Y("Impact ($M):Q"),
    size=alt.Size("Size:Q", title="Project Value ($M)", legend=None),
    color=alt.Color("Primary Driver:N", legend=alt.Legend(title="Driver")),
    tooltip=["Project","Customer","Value","p_on_time","Revenue_at_Risk","Primary Driver"]
).properties(height=380)
st.altair_chart(chart, use_container_width=True)

pareto = (
    heat.groupby("Primary Driver", as_index=False)["Revenue_at_Risk"]
        .sum().sort_values("Revenue_at_Risk", ascending=False)
)
pareto["Cum %"] = pareto["Revenue_at_Risk"].cumsum() / pareto["Revenue_at_Risk"].sum() * 100

bars = alt.Chart(pareto).mark_bar().encode(
    x=alt.X("Primary Driver:N", sort="-y"),
    y=alt.Y("Revenue_at_Risk:Q", title="RAR ($)"),
    tooltip=["Primary Driver","Revenue_at_Risk"]
)
line = alt.Chart(pareto).mark_line(point=True).encode(
    x=alt.X("Primary Driver:N", sort="-y"),
    y=alt.Y("Cum %:Q", axis=alt.Axis(format="~s"), title="Cumulative %"),
)
st.altair_chart((bars + line).resolve_scale(y="independent"), use_container_width=True)

# ------------------------- Pipeline EV & Funnel -------------------------
st.subheader("Pipeline (Probability-Weighted)")

pipe = pd.DataFrame(PIPELINE)
pipe["stage_prob"] = pipe["stage"].map(STAGE_PROBS).fillna(pipe["prob"])
pipe["EV"] = pipe["value"] * pipe["stage_prob"]
funnel = (pipe.groupby("stage", as_index=False)
          .agg(count=("number","count"), nominal=("value","sum"), EV=("EV","sum")))
stage_cat = pd.Categorical(funnel["stage"], categories=STAGE_ORDER, ordered=True)
funnel = funnel.assign(stage=stage_cat).sort_values("stage")

cA, cB = st.columns((2,1))
with cA:
    plot = alt.Chart(funnel).mark_bar().encode(
        x=alt.X("stage:N", sort=STAGE_ORDER, title="Stage"),
        y=alt.Y("EV:Q", title="Expected Value ($)"),
        color=alt.Color("stage:N", legend=None)
    ).properties(height=320)
    st.altair_chart(plot, use_container_width=True)
with cB:
    st.dataframe(funnel.assign(**{
        "nominal_fmt": funnel["nominal"].map(lambda v: f"${v:,.0f}"),
        "EV_fmt": funnel["EV"].map(lambda v: f"${v:,.0f}")
    })[["stage","count","nominal_fmt","EV_fmt"]].rename(columns={"nominal_fmt":"Nominal","EV_fmt":"EV"}),
    use_container_width=True)

# ------------------------- WIP Limits (max 3 large Interiors) -------------------------
st.subheader("WIP Limits – Large Interiors Concurrency")

def active_weeks(ind, due):
    start = monday_of_week(ind)
    end = monday_of_week(due)
    weeks = []
    cur = start
    while cur <= end:
        weeks.append(cur)
        cur += timedelta(days=7)
    return weeks

# build weekly counts for "large interiors"
counts = {}
for p in PROJECTS:
    if not large_interiors(p): continue
    ind, due = to_date(p["induction"]), to_date(p["delivery"])
    for w in active_weeks(ind, due):
        counts[w] = counts.get(w, 0) + 1

if counts:
    wip_df = pd.DataFrame({"Week": list(counts.keys()), "Large_Count": list(counts.values())}).sort_values("Week")
else:
    wip_df = pd.DataFrame({"Week": [monday_of_week(TODAY)], "Large_Count":[0]})

limit_line = alt.Chart(wip_df).mark_rule(color="red").encode(y=alt.datum(LARGE_WIP_LIMIT))
wip_line = alt.Chart(wip_df).mark_line(point=True).encode(
    x=alt.X("Week:T"),
    y=alt.Y("Large_Count:Q", title="Large Interiors Concurrency"),
    tooltip=["Week","Large_Count"]
).properties(height=300)
st.altair_chart(wip_line + limit_line, use_container_width=True)
breaches = wip_df[wip_df["Large_Count"] > LARGE_WIP_LIMIT]
if not breaches.empty:
    st.warning(f"WIP breach forecast in {len(breaches)} week(s): {', '.join(breaches['Week'].dt.strftime('%Y-%m-%d'))}")

# ------------------------- Earliest Available Induction Window (demo) -------------------------
st.subheader("Earliest Available Induction Window (demo)")

# pick the next likely pipeline win (highest EV, not yet confirmed)
candidate = pipe.sort_values("EV", ascending=False).iloc[0].to_dict()
cand_hours = candidate.get("hours", {})
cand_model = candidate.get("aircraftModel","B737")
cand_large = (cand_hours.get("Interiors",0) >= 3000) or (aircraft_class(cand_model) == "HEAVY") or (sum(cand_hours.values()) >= 6000)

# sweep next 20 weeks starting Sunday before TODAY
start_sun = sunday_before(TODAY)
weeks = [start_sun + timedelta(days=7*i) for i in range(0, 20)]
def large_count_with_candidate(start_week: date) -> int:
    hypothetical = counts.copy()
    if cand_large:
        # occupancy window: estimate duration from Interiors hours at mean productivity, fallback to 4 weeks
        prod = PRODUCTIVITY_MEAN
        cap_wk_interiors = HC["Interiors"] * HOURS_PER_FTE * prod
        dur_weeks = max(1, int(math.ceil((cand_hours.get("Interiors",1e-6)) / max(1.0, cap_wk_interiors))))
        for i in range(dur_weeks):
            hypothetical[start_week + timedelta(days=7*i)] = hypothetical.get(start_week + timedelta(days=7*i), 0) + 1
    return max(hypothetical.values()) if hypothetical else 1 if cand_large else 0

feasible_week = None
for w in weeks:
    if large_count_with_candidate(monday_of_week(w)) <= LARGE_WIP_LIMIT:
        feasible_week = monday_of_week(w)
        break

if feasible_week:
    # Rough delivery P50 estimate using the same duration estimator
    prod = PRODUCTIVITY_MEAN
    # duration driven by critical dept (max of dept durations)
    crit_wd = duration_workdays_for_hours(cand_hours, np.array([prod]))[0]
    est_delivery = add_workdays(feasible_week, int(crit_wd))
    st.success(f"Earliest induction (meets WIP limit={LARGE_WIP_LIMIT}): **{feasible_week}**  → Estimated P50 delivery: **{est_delivery}**")
else:
    st.error("No feasible induction window within the next 20 weeks under current WIP policy.")

# ------------------------- Appendix: Raw data -------------------------
with st.expander("Show raw data (projects & pipeline)"):
    st.write("**Confirmed Projects**")
    st.dataframe(pd.DataFrame(PROJECTS).assign(induction=lambda d: d["induction"], delivery=lambda d: d["delivery"]))
    st.write("**Pipeline**")
    st.dataframe(pd.DataFrame(PIPELINE))

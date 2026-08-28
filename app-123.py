
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import milp, LinearConstraint, Bounds

# ----------------------------------------------------------------------------- config / theme
st.set_page_config(page_title="Turbofan Maintenance Scheduler",
                   page_icon="🛩", layout="wide")

NAVY, TEAL, AMBER, RED, SLATE, MIST = "#0B2138", "#0E7C86", "#E8A317", "#C0392B", "#33475B", "#EEF3F6"

st.markdown(f"""
<style>
  .stApp {{ background:#F7F9FB; }}
  h1,h2,h3 {{ color:{NAVY}; font-family:"Segoe UI",Arial,sans-serif; }}
  .step-eyebrow {{ color:{TEAL}; font-weight:700; letter-spacing:.14em;
                   font-size:.72rem; text-transform:uppercase; }}
  .card {{ background:#fff; border:1px solid #E2E8EE; border-radius:12px;
           padding:18px 20px; box-shadow:0 1px 3px rgba(11,33,56,.05); }}
  .kpi {{ background:#fff; border:1px solid #E2E8EE; border-left:5px solid {TEAL};
          border-radius:10px; padding:14px 16px; }}
  .kpi .v {{ font-size:1.7rem; font-weight:700; color:{NAVY}; line-height:1.1; }}
  .kpi .l {{ font-size:.78rem; color:{SLATE}; text-transform:uppercase; letter-spacing:.05em; }}
  .lock {{ background:{MIST}; border:1px dashed {TEAL}; border-radius:8px;
           padding:8px 12px; color:{NAVY}; font-size:.86rem; }}
  .stButton>button {{ background:{NAVY}; color:#fff; border:0; border-radius:8px;
                      padding:.55rem 1.1rem; font-weight:600; }}
  .stButton>button:hover {{ background:{TEAL}; color:#fff; }}
  .stDownloadButton>button {{ background:#fff; color:{NAVY}; border:1.5px solid {NAVY};
                              border-radius:8px; font-weight:600; }}
</style>""", unsafe_allow_html=True)

def eyebrow(n, txt):
    st.markdown(f"<div class='step-eyebrow'>Step {n}</div><h2 style='margin-top:-4px'>{txt}</h2>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- session state
ss = st.session_state
for k, v in dict(data=None, trained=False, pred=None, rmse=None, mae=None,
                 go_opt=False, sched=None, kpi=None, obj=None, params=None).items():
    ss.setdefault(k, v)

SENSORS_PREFIX = "S"
RUL_CAP = 125

# ----------------------------------------------------------------------------- ML helpers
def _features(df, sensors, win=5):
    df = df.sort_values(["ID", "Cycle"]).copy()
    g = df.groupby("ID")[sensors]
    rm = g.rolling(win, min_periods=1).mean().reset_index(0, drop=True).add_suffix("_rm")
    rs = g.rolling(win, min_periods=1).std().fillna(0).reset_index(0, drop=True).add_suffix("_rs")
    first = df.groupby("ID")[sensors].transform("first")
    dl = (df[sensors] - first).add_suffix("_dl")           # degradation delta (key feature)
    feats = pd.concat([df, rm, rs, dl], axis=1)
    cols = sensors + [s + "_rm" for s in sensors] + [s + "_rs" for s in sensors] + [s + "_dl" for s in sensors]
    return feats, cols

def train_predict(train, test, true_rul):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    sensors = [c for c in train.columns if c.startswith(SENSORS_PREFIX)]
    tr = train.copy(); tr["RUL"] = tr["RUL"].clip(upper=RUL_CAP)
    trf, cols = _features(tr, sensors); tef, _ = _features(test, sensors)
    sc = MinMaxScaler().fit(trf[cols])
    Xtr, ytr = sc.transform(trf[cols]), tr["RUL"].values
    try:                                                   # real XGBoost if present
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 min_child_weight=30, random_state=42, n_jobs=-1,
                                 objective="reg:squarederror")
        engine = "XGBoost"
    except ImportError:                                    # equivalent GBDT fallback
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.02, max_depth=5,
                                              l2_regularization=2.0, min_samples_leaf=40, random_state=42)
        engine = "Gradient-Boosted Trees (XGBoost-equivalent)"
    model.fit(Xtr, ytr)
    last = tef.groupby("ID").tail(1).sort_values("ID")
    pred = np.clip(model.predict(sc.transform(last[cols])), 0, RUL_CAP)
    ids = last["ID"].values.astype(int)
    rmse = float(np.sqrt(mean_squared_error(true_rul, pred))) if true_rul is not None else None
    mae = float(np.mean(np.abs(true_rul - pred))) if true_rul is not None else None
    return pd.DataFrame({"Engine ID": ids, "Predicted RUL": np.round(pred, 1)}), rmse, mae, engine

# ----------------------------------------------------------------------------- optimizer (HiGHS)
def solve_ilp(rul_values, p):
    RUL = np.asarray(rul_values, float); N = len(RUL); rho = p["rho"]
    H = int(np.ceil(RUL.max() / rho)) + 2; periods = list(range(1, H + 1)); T = len(periods)
    t_tgt = np.clip(np.ceil(p["beta"] * RUL / rho).astype(int), 1, T)
    d_i = np.clip(np.floor((RUL - p["Delta"]) / rho).astype(int), 1, T)
    C = np.array([[p["c_m"] + p["p_e"] * max(t_tgt[i] - t, 0) + p["p_l"] * max(t - t_tgt[i], 0)
                   for t in periods] for i in range(N)])
    nx = N * T; nvar = nx + N
    xi = lambda i, j: i * T + j; yi = lambda i: nx + i
    c_obj = np.concatenate([C.flatten(), np.full(N, p["p_g"])])
    A, lb, ub = [], [], []
    for i in range(N):                                     # C1 assignment
        r = np.zeros(nvar); [r.__setitem__(xi(i, j), 1) for j in range(T)]; r[yi(i)] = 1
        A.append(r); lb.append(1); ub.append(1)
    for j in range(T):                                     # C2 capacity
        r = np.zeros(nvar); [r.__setitem__(xi(i, j), 1) for i in range(N)]
        A.append(r); lb.append(0); ub.append(p["C_t"])
    for i in range(N):                                     # C3 safety deadline (MANDATORY)
        r = np.zeros(nvar); late = False
        for j, t in enumerate(periods):
            if t > d_i[i]: r[xi(i, j)] = 1; late = True
        if late: A.append(r); lb.append(0); ub.append(0)
    res = milp(c=c_obj, constraints=LinearConstraint(np.array(A), lb, ub),
               integrality=np.ones(nvar), bounds=Bounds(np.zeros(nvar), np.ones(nvar)),
               options={"mip_rel_gap": 0.0})
    x = res.x[:nx].reshape(N, T); y = res.x[nx:]
    rows = []
    for i in range(N):
        if y[i] > 0.5:
            sp, mode, cost, e, l = None, "Emergency (corrective)", p["p_g"], 0, 0
        else:
            j = int(np.argmax(x[i])); sp = periods[j]; mode = "Preventive (scheduled)"
            cost = C[i, j]; e = max(t_tgt[i] - sp, 0); l = max(sp - t_tgt[i], 0)
        rows.append({"Engine ID": i + 1, "Predicted RUL": round(RUL[i], 1),
                     "Target Period": int(t_tgt[i]), "Deadline Period": int(d_i[i]),
                     "Scheduled Period": sp, "Maintenance Mode": mode,
                     "Earliness": e, "Lateness": l, "Realized Cost ($)": float(cost)})
    df = pd.DataFrame(rows)
    df["_k"] = df["Scheduled Period"].fillna(10**9)
    df = df.sort_values(["_k", "Predicted RUL"]).drop(columns="_k").reset_index(drop=True)
    df.insert(0, "Priority Rank", range(1, N + 1))
    return df, float(res.fun)

def kpis_from(df, p):
    prev = int((df["Maintenance Mode"].str.startswith("Preventive")).sum())
    emer = len(df) - prev
    used = df["Scheduled Period"].dropna()
    load = used.value_counts()
    return {
        "Total optimal cost ($)": df["Realized Cost ($)"].sum(),
        "Engines": len(df), "Preventive": prev, "Emergency (corrective)": emer,
        "Emergency share of fleet": emer / len(df),
        "Emergency cost share": df.loc[df["Maintenance Mode"].str.startswith("Emergency"),
                                       "Realized Cost ($)"].sum() / df["Realized Cost ($)"].sum(),
        "Avg earliness (periods)": df.loc[df["Scheduled Period"].notna(), "Earliness"].mean(),
        "Avg lateness (periods)": df.loc[df["Scheduled Period"].notna(), "Lateness"].mean(),
        "Periods used": int(used.nunique()),
        "Peak workshop load": int(load.max()) if len(load) else 0,
        "Workshop capacity": p["C_t"],
    }

def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, d in sheets.items():
            d.to_excel(xl, sheet_name=name, index=False)
    return buf.getvalue()

# ----------------------------------------------------------------------------- header
st.markdown(f"<h1 style='margin-bottom:0'>🛩 Turbofan Maintenance Scheduler</h1>"
            f"<p style='color:{SLATE};margin-top:2px'>RUL-based predictive maintenance optimisation "
            f"for aircraft engines — CMAPSS FD001. Prediction → optimisation → schedule.</p>",
            unsafe_allow_html=True)
st.divider()

# ============================================================================= STEP 1 — DATA
eyebrow(1, "Load the dataset")
c1, c2 = st.columns([1, 1])
local = Path(__file__).parent / "CMAPSS_Dataset.xlsx"
with c1:
    if local.exists():
        st.download_button("⬇ Download CMAPSS dataset workbook", local.read_bytes(),
                           "CMAPSS_Dataset.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Place CMAPSS_Dataset.xlsx next to app.py to enable the download.")
with c2:
    up = st.file_uploader("Upload dataset workbook (sheets: Train, Test, RUL)", type=["xlsx"])

src = up if up is not None else (local if local.exists() else None)
if src is not None and ss.data is None:
    xls = pd.read_excel(src, sheet_name=None)
    ss.data = {"Train": xls["Train"], "Test": xls["Test"],
               "RUL": xls["RUL"].iloc[:, 0].values.astype(float) if "RUL" in xls else None}
if ss.data:
    tr = ss.data["Train"]
    st.markdown(f"<div class='card'>Loaded — <b>{tr['ID'].nunique()}</b> training engines, "
                f"<b>{ss.data['Test']['ID'].nunique()}</b> test engines, "
                f"<b>{len([c for c in tr.columns if c.startswith('S')])}</b> sensors.</div>",
                unsafe_allow_html=True)

# ============================================================================= STEP 2 — XGBOOST
if ss.data:
    st.divider(); eyebrow(2, "Predict RUL with XGBoost")
    if st.button("🚀 Run XGBoost"):
        with st.spinner("Training model…"):
            ss.pred, ss.rmse, ss.mae, eng = train_predict(ss.data["Train"], ss.data["Test"], ss.data["RUL"])
            ss.trained = True; ss.engine_name = eng
    if ss.trained:
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><div class='v'>{ss.rmse:.2f}</div>"
                    f"<div class='l'>RMSE (cycles)</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><div class='v'>{ss.mae:.2f}</div>"
                    f"<div class='l'>MAE (cycles)</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><div class='v'>{len(ss.pred)}</div>"
                    f"<div class='l'>engines predicted</div></div>", unsafe_allow_html=True)
        st.caption(f"Model: {ss.engine_name} · features: 14 sensors + rolling mean/std + degradation delta · RUL capped at {RUL_CAP}.")
        g1, g2 = st.columns([1, 1])
        with g1:
            if ss.data["RUL"] is not None:
                sc = pd.DataFrame({"True RUL": ss.data["RUL"], "Predicted RUL": ss.pred["Predicted RUL"]})
                fig = px.scatter(sc, x="True RUL", y="Predicted RUL", opacity=.7,
                                 color_discrete_sequence=[TEAL], title="Predicted vs True RUL")
                m = max(sc.max()); fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                                 line=dict(dash="dash", color=NAVY), showlegend=False))
                fig.update_layout(height=340, plot_bgcolor="white"); st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig = px.histogram(ss.pred, x="Predicted RUL", nbins=25,
                               color_discrete_sequence=[NAVY], title="Distribution of predicted RUL")
            fig.update_layout(height=340, plot_bgcolor="white"); st.plotly_chart(fig, use_container_width=True)

        # STEP 3 — download predictions
        st.markdown("<div class='step-eyebrow'>Step 3</div>", unsafe_allow_html=True)
        st.download_button("⬇ Download predicted RUL", to_excel({"Predicted RUL": ss.pred}),
                           "predicted_rul.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # STEP 4 — continue
        if st.button("Continue to optimization →"):
            ss.go_opt = True

# ============================================================================= STEP 5 — PARAMETERS
if ss.go_opt:
    st.divider(); eyebrow(5, "Optimisation parameters")
    st.markdown("<div class='lock'>🔒 Safety-deadline constraint (C3) is always enforced — "
                "no engine is scheduled after its RUL-based deadline.</div>", unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    Delta = a.number_input("Safety margin Δ (cycles)", 0.0, 60.0,
                           float(round(ss.rmse)) if ss.rmse else 14.0, 1.0)
    C_t = a.number_input("Workshop capacity per period", 1, 20, 3, 1)
    beta = b.number_input("Safety factor β", 0.50, 1.00, 0.90, 0.05)
    c_m = b.number_input("Base preventive cost ($)", 1000, 100000, 10000, 500)
    p_l = c.number_input("Lateness penalty ($/period)", 0, 10000, 1000, 50)
    p_g = c.number_input("Emergency (failure) cost ($)", 5000, 500000, 50000, 1000)
    p_e = c.number_input("Earliness penalty ($/period)", 0, 100000, 100, 50)

    # STEP 6 — run
    if st.button("▶ Run optimization"):
        params = dict(rho=1.0, beta=beta, Delta=Delta, C_t=int(C_t),
                      c_m=float(c_m), p_e=float(p_e), p_l=float(p_l), p_g=float(p_g))
        with st.spinner("Solving ILP with HiGHS…"):
            ss.sched, ss.obj = solve_ilp(ss.pred["Predicted RUL"].values, params)
            # attach real Engine IDs from predictions (align by predicted RUL order)
            ss.sched["Engine ID"] = ss.sched["Engine ID"].map(
                dict(zip(range(1, len(ss.pred) + 1), ss.pred["Engine ID"].values)))
            ss.kpi = kpis_from(ss.sched, params); ss.params = params

# ============================================================================= STEP 7 — RESULTS
if ss.sched is not None:
    st.divider(); eyebrow(7, "Schedule, KPIs & dashboard")
    k = ss.kpi
    cols = st.columns(4)
    cards = [("${:,.0f}".format(k["Total optimal cost ($)"]), "Total optimal cost"),
             (k["Preventive"], "Preventive scheduled"),
             (k["Emergency (corrective)"], "Emergency (corrective)"),
             ("{:.0%}".format(k["Emergency cost share"]), "Emergency cost share")]
    for col, (v, l) in zip(cols, cards):
        col.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>",
                     unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [(k["Peak workshop load"], f"peak load (cap {k['Workshop capacity']})"),
             (k["Periods used"], "periods used"),
             ("{:.1f}".format(k["Avg earliness (periods)"]), "avg earliness"),
             ("{:.1f}".format(k["Avg lateness (periods)"]), "avg lateness")]
    for col, (v, l) in zip(cols, cards):
        col.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>",
                     unsafe_allow_html=True)

    st.write("")
    display_cols = ["Priority Rank", "Engine ID", "Predicted RUL",
                    "Maintenance Mode", "Realized Cost ($)"]
    show = ss.sched[display_cols].copy()

    def hl(row):
        if row["Maintenance Mode"].startswith("Emergency"):
            return ["background-color:#FCE4E4"] * len(row)
        return [""] * len(row)
    st.dataframe(show.style.apply(hl, axis=1).format({"Realized Cost ($)": "${:,.0f}",
                 "Predicted RUL": "{:.1f}"}), use_container_width=True, height=430)

    # dashboard charts
    d1, d2 = st.columns([1.3, 1])
    with d1:
        sc = ss.sched.dropna(subset=["Scheduled Period"])
        fig = px.scatter(sc, x="Scheduled Period", y="Predicted RUL", color="Maintenance Mode",
                         color_discrete_map={"Preventive (scheduled)": TEAL, "Emergency (corrective)": RED},
                         title="Schedule map — when each engine is serviced vs its RUL")
        fig.update_layout(height=360, plot_bgcolor="white", legend=dict(orientation="h", y=-.25))
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        load = ss.sched["Scheduled Period"].dropna().astype(int).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=load.index, y=load.values, marker_color=NAVY, name="engines"))
        fig.add_hline(y=ss.params["C_t"], line_dash="dash", line_color=RED,
                      annotation_text="capacity")
        fig.update_layout(title="Workshop load per period", height=360, plot_bgcolor="white",
                          xaxis_title="period", yaxis_title="engines")
        st.plotly_chart(fig, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        split = ss.sched["Maintenance Mode"].value_counts()
        fig = go.Figure(go.Pie(labels=split.index, values=split.values, hole=.55,
                        marker_colors=[TEAL, RED]))
        fig.update_layout(title="Preventive vs emergency", height=320)
        st.plotly_chart(fig, use_container_width=True)
    with d4:
        cb = ss.sched.groupby("Maintenance Mode")["Realized Cost ($)"].sum()
        color_map = {"Preventive (scheduled)": TEAL, "Emergency (corrective)": RED}
        bar_colors = [color_map.get(m, SLATE) for m in cb.index]
        fig = go.Figure(go.Bar(x=cb.index, y=cb.values, marker_color=bar_colors,
                        text=[f"${v:,.0f}" for v in cb.values], textposition="outside"))
        fig.update_layout(title="Cost breakdown by mode", height=320, plot_bgcolor="white",
                          yaxis_title="$", uniformtext_minsize=10, uniformtext_mode="hide")
        fig.update_yaxes(range=[0, cb.values.max() * 1.15])
        st.plotly_chart(fig, use_container_width=True)

    # downloads
    st.divider()
    kpi_df = pd.DataFrame({"Metric": list(k.keys()), "Value": list(k.values())})
    par_df = pd.DataFrame({"Parameter": list(ss.params.keys()), "Value": list(ss.params.values())})
    e1, e2 = st.columns(2)
    e1.download_button("⬇ Download schedule (Excel)", to_excel({"Schedule": show}),
                       "maintenance_schedule.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    e2.download_button("⬇ Download full results workbook",
                       to_excel({"Schedule": show, "KPIs": kpi_df, "Parameters": par_df}),
                       "maintenance_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

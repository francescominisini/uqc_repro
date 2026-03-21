from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(page_title="UFO / TRPO Training Dashboard", layout="wide")


# ---------- Helpers ----------

def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=2)
def load_training_log(log_path: str) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "iteration" in df.columns:
        df = df.sort_values("iteration").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=2)
def load_plan_npz(plan_path: str) -> dict[str, Any] | None:
    path = Path(plan_path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        if "controls" in out:
            out["controls"] = np.asarray(out["controls"], dtype=np.float64)
        return out
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=2)
def find_plan_files(run_dir: str) -> list[str]:
    plans_dir = Path(run_dir) / "plans"
    if not plans_dir.exists():
        return []
    return sorted([str(p) for p in plans_dir.glob("*.npz")])


@st.cache_data(show_spinner=False, ttl=2)
def load_summary(run_dir: str) -> dict[str, Any] | None:
    return safe_read_json(Path(run_dir) / "summary.json")


@st.cache_data(show_spinner=False, ttl=2)
def load_controls_table(plan_path: str) -> pd.DataFrame:
    plan = load_plan_npz(plan_path)
    if not plan or "controls" not in plan:
        return pd.DataFrame()

    controls = np.asarray(plan["controls"], dtype=np.float64)
    dt_ns = float(plan.get("dt_ns", 1.0))
    t = np.arange(controls.shape[0], dtype=np.float64) * dt_ns
    cols = ["d1", "d2", "f1", "phi1", "f2", "phi2", "g"]
    df = pd.DataFrame(controls, columns=cols)
    df.insert(0, "time_ns", t)
    return df



def metric_delta(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or len(df) < 2:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) < 2:
        return None
    return float(vals.iloc[-1] - vals.iloc[-2])



def latest_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.empty:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])



def make_line_figure(df: pd.DataFrame, x: str, y_cols: list[str], title: str, yaxis_title: str = "") -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                fig.add_trace(go.Scatter(x=df[x], y=s, mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=yaxis_title,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig



def make_controls_figure(ctrl_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ["d1", "d2", "f1", "f2", "g", "phi1", "phi2"]:
        if col in ctrl_df.columns:
            fig.add_trace(go.Scatter(x=ctrl_df["time_ns"], y=ctrl_df[col], mode="lines", name=col))
    fig.update_layout(
        title="Control channels vs time",
        xaxis_title="time_ns",
        yaxis_title="MHz / phase(rad)",
        height=430,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig



def maybe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


# ---------- Sidebar ----------
st.sidebar.title("Dashboard Controls")
run_dir = st.sidebar.text_input("Run directory", value="out/trpo_fastpilot")
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (s)", min_value=1, max_value=30, value=3)
show_raw_table = st.sidebar.checkbox("Show raw training table", value=False)
show_controls_table = st.sidebar.checkbox("Show controls table", value=False)

summary = load_summary(run_dir)
log_path = str(Path(run_dir) / "training_log.jsonl")
df = load_training_log(log_path)
plan_files = find_plan_files(run_dir)

plan_choice_mode = st.sidebar.radio("Plan source", ["Best plan", "Final plan", "Latest iter plan", "Choose manually"], index=0)
manual_plan = None
if plan_choice_mode == "Choose manually" and plan_files:
    manual_plan = st.sidebar.selectbox("Plan file", options=plan_files, index=len(plan_files) - 1)

best_plan_path = str(Path(run_dir) / "best_control_plan.npz")
final_plan_path = str(Path(run_dir) / "final_control_plan.npz")

if plan_choice_mode == "Best plan":
    selected_plan_path = best_plan_path
elif plan_choice_mode == "Final plan":
    selected_plan_path = final_plan_path
elif plan_choice_mode == "Latest iter plan":
    selected_plan_path = plan_files[-1] if plan_files else best_plan_path
else:
    selected_plan_path = manual_plan or (plan_files[-1] if plan_files else best_plan_path)

plan = load_plan_npz(selected_plan_path) if selected_plan_path else None
ctrl_df = load_controls_table(selected_plan_path) if selected_plan_path else pd.DataFrame()


# ---------- Header ----------
st.title("UFO / TRPO Training Dashboard")
st.caption("Live monitor for the single-target training run and saved control plans.")

if not Path(run_dir).exists():
    st.error(f"Run directory not found: {run_dir}")
    st.stop()


# ---------- Summary / status ----------
left, right = st.columns([2, 1])
with left:
    st.subheader("Run status")
    if df.empty:
        st.warning("No training_log.jsonl entries found yet.")
    else:
        st.write(f"Iterations logged: **{len(df)}**")
        if "iteration" in df.columns:
            st.write(f"Latest iteration: **{int(df['iteration'].max())}**")

with right:
    st.subheader("Artifacts")
    st.write(f"Summary file: {'✅' if summary else '❌'}")
    st.write(f"Plan files: **{len(plan_files)}**")
    st.write(f"Selected plan: `{Path(selected_plan_path).name if selected_plan_path else 'None'}`")


# ---------- Top metrics ----------
if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eval fidelity", latest_value(df, "eval_fidelity"), metric_delta(df, "eval_fidelity"))
    c2.metric("Eval cost", latest_value(df, "eval_cost"), metric_delta(df, "eval_cost"))
    c3.metric("Avg train fidelity", latest_value(df, "avg_fidelity"), metric_delta(df, "avg_fidelity"))
    c4.metric("Avg train leakage", latest_value(df, "avg_leakage"), metric_delta(df, "avg_leakage"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg time (ns)", latest_value(df, "avg_time_ns"), metric_delta(df, "avg_time_ns"))
    c6.metric("TRPO KL", latest_value(df, "update_kl"), metric_delta(df, "update_kl"))
    c7.metric("Value loss", latest_value(df, "update_value_loss"), metric_delta(df, "update_value_loss"))
    c8.metric("Policy step accepted", latest_value(df, "update_updated"), metric_delta(df, "update_updated"))


# ---------- Charts: training / eval ----------
if not df.empty and "iteration" in df.columns:
    st.subheader("Training curves")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_line_figure(df, "iteration", ["avg_cost", "eval_cost"], "Cost", "cost"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_line_figure(df, "iteration", ["avg_fidelity", "eval_fidelity"], "Fidelity", "fidelity"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            make_line_figure(df, "iteration", ["avg_leakage", "eval_leakage"], "Leakage", "leakage"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_line_figure(df, "iteration", ["avg_time_ns", "eval_time_ns"], "Runtime", "ns"),
            use_container_width=True,
        )

    st.subheader("TRPO diagnostics")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            make_line_figure(df, "iteration", ["update_kl", "update_policy_loss"], "Policy update", "value"),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            make_line_figure(df, "iteration", ["update_value_loss", "update_updated"], "Critic / acceptance", "value"),
            use_container_width=True,
        )


# ---------- Best run summary ----------
st.subheader("Summary.json")
if summary:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Best eval fidelity", maybe_float(summary.get("best_eval_fidelity")))
    s2.metric("Best eval leakage", maybe_float(summary.get("best_eval_leakage")))
    s3.metric("Best eval time (ns)", maybe_float(summary.get("best_eval_time_ns")))
    s4.metric("Best eval cost", maybe_float(summary.get("best_eval_cost")))

    with st.expander("Raw summary"):
        st.json(summary)
else:
    st.info("No summary.json found yet.")


# ---------- Plan / controls ----------
st.subheader("Selected control plan")
if plan is None:
    st.warning("Selected plan file not found yet.")
else:
    info_cols = st.columns(5)
    info_cols[0].metric("Steps", int(np.asarray(plan["controls"]).shape[0]) if "controls" in plan else None)
    info_cols[1].metric("dt (ns)", maybe_float(plan.get("dt_ns", None)))
    info_cols[2].metric("alpha", maybe_float(plan.get("target_alpha", None)))
    info_cols[3].metric("gamma", maybe_float(plan.get("target_gamma", None)))
    bw = plan.get("filter_bandwidth_mhz", None)
    info_cols[4].metric("Filter BW (MHz)", maybe_float(bw))

    note = plan.get("note", "")
    if isinstance(note, np.ndarray) and note.shape == ():
        note = note.item()
    st.caption(f"Plan note: {note}")

    if not ctrl_df.empty:
        st.plotly_chart(make_controls_figure(ctrl_df), use_container_width=True)

        c_left, c_right = st.columns(2)
        with c_left:
            amp_cols = [c for c in ["d1", "d2", "f1", "f2", "g"] if c in ctrl_df.columns]
            st.plotly_chart(
                make_line_figure(ctrl_df, "time_ns", amp_cols, "Amplitude-like controls", "MHz"),
                use_container_width=True,
            )
        with c_right:
            phase_cols = [c for c in ["phi1", "phi2"] if c in ctrl_df.columns]
            st.plotly_chart(
                make_line_figure(ctrl_df, "time_ns", phase_cols, "Phase controls", "rad"),
                use_container_width=True,
            )

        if show_controls_table:
            with st.expander("Controls table"):
                st.dataframe(ctrl_df, use_container_width=True, height=320)


# ---------- Training table ----------
if show_raw_table and not df.empty:
    st.subheader("Raw training log")
    st.dataframe(df, use_container_width=True, height=420)


# ---------- Footer ----------
st.markdown("---")
st.caption(
    "This dashboard reads the training_log.jsonl, summary.json, and saved .npz control plans generated by your training script."
)


# ---------- Auto refresh ----------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()



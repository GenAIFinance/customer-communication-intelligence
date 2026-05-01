"""
streamlit_app.py — Customer Communication Intelligence Dashboard.

4 pages:
  - Overview       : KPI cards + engagement distribution + segment breakdown
  - Anomaly Monitor: Anomaly flags + complaint spike chart + campaign heatmap
  - Case Explorer  : Filterable table + drill-down + model scores
  - AI Summary     : Select a case → generate next-best-action card

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Customer Comm Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #0f1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
        border: 1px solid #2d3147;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-value {
        font-family: 'DM Mono', monospace;
        font-size: 2.2rem;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }
    .kpi-delta-up   { color: #f87171; font-size: 0.8rem; }
    .kpi-delta-down { color: #34d399; font-size: 0.8rem; }

    .anomaly-flag {
        background: #1f1320;
        border-left: 3px solid #f87171;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #fca5a5;
    }
    .ok-flag {
        background: #0f1f17;
        border-left: 3px solid #34d399;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #6ee7b7;
    }

    .summary-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1c2035 100%);
        border: 1px solid #3b4168;
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 16px;
    }
    .summary-text {
        font-size: 1.05rem;
        line-height: 1.8;
        color: #cbd5e1;
        font-style: italic;
    }
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .badge-openai { background: #10b981; color: #fff; }
    .badge-stub   { background: #6366f1; color: #fff; }

    .risk-high   { color: #f87171; font-weight: 600; }
    .risk-medium { color: #fbbf24; font-weight: 600; }
    .risk-low    { color: #34d399; font-weight: 600; }

    div[data-testid="stSidebarContent"] { background-color: #0d0f18; }
    .stSelectbox > div > div { background-color: #1a1d27; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load communications data from DuckDB. Cached for 5 minutes."""
    from src.utils.db import query_df
    return query_df("SELECT * FROM customer_communications")


@st.cache_resource
def load_model_and_features():
    """Load trained model and feature names once per session."""
    try:
        from src.modeling.score import load_model
        from src.modeling.train_model import load_feature_names
        return load_model(), load_feature_names()
    except FileNotFoundError:
        return None, None


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add intervention scores to a DataFrame."""
    model, feature_names = load_model_and_features()
    if model is None:
        df["intervention_score"] = 0.0
        df["risk_band"] = "Unknown"
        return df
    from src.modeling.score import score_batch
    return score_batch(df, model=model, expected_columns=feature_names)


# ── Sidebar navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 CCI Platform")
    st.markdown("*Customer Communication Intelligence*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Anomaly Monitor", "Case Explorer", "AI Summary"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Data**")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("⚠️ Synthetic data only")


# ── Load data ──────────────────────────────────────────────────────────────────

try:
    df = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load data: {e}. Run `python -m src.data.ingest` first.")
    data_ok = False
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("# Overview")
    st.markdown("High-level metrics across all customer communications.")
    st.divider()

    # ── KPI cards ──
    total        = len(df)
    intervention = df["needs_intervention"].mean()
    complaint    = df["complaint_flag"].mean()
    opt_out      = df["opt_out_flag"].mean()
    avg_eng      = df["engagement_score"].mean()
    escalation   = df["escalation_flag"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"{total:,}",         "Total Records",       ""),
        (c2, f"{intervention:.1%}", "Intervention Rate",  "kpi-delta-up"),
        (c3, f"{avg_eng:.2f}",     "Avg Engagement",      "kpi-delta-down"),
        (c4, f"{complaint:.1%}",   "Complaint Rate",      "kpi-delta-up"),
        (c5, f"{escalation:.1%}",  "Escalation Rate",     "kpi-delta-up"),
        (c6, f"{opt_out:.1%}",     "Opt-out Rate",        ""),
    ]
    for col, val, label, delta_cls in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Charts row 1 ──
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Engagement Score Distribution")
        fig = px.histogram(
            df, x="engagement_score", nbins=40,
            color_discrete_sequence=["#6366f1"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=260,
            xaxis_title="Engagement Score",
            yaxis_title="Count",
            showlegend=False,
        )
        fig.add_vline(x=avg_eng, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Mean: {avg_eng:.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Intervention by Segment")
        seg_stats = df.groupby("segment")["needs_intervention"].mean().reset_index()
        seg_stats.columns = ["Segment", "Intervention Rate"]
        fig2 = px.bar(
            seg_stats, x="Segment", y="Intervention Rate",
            color="Intervention Rate",
            color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=260,
            showlegend=False,
            coloraxis_showscale=False,
        )
        fig2.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Channel Breakdown")
        ch = df.groupby("channel").agg(
            total=("customer_id", "count"),
            intervention=("needs_intervention", "mean")
        ).reset_index()
        fig3 = px.scatter(
            ch, x="total", y="intervention",
            text="channel", size="total",
            color_discrete_sequence=["#818cf8"],
            template="plotly_dark",
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=240,
            xaxis_title="Volume", yaxis_title="Intervention Rate",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Sentiment Mix")
        sent = df["sentiment_text"].value_counts().reset_index()
        sent.columns = ["Sentiment", "Count"]
        colors = {"positive": "#34d399", "neutral": "#94a3b8",
                  "mixed": "#fbbf24", "negative": "#f87171"}
        fig4 = px.pie(
            sent, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=colors,
            template="plotly_dark", hole=0.5,
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=240,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: ANOMALY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Anomaly Monitor":
    st.markdown("# Anomaly Monitor")
    st.markdown("Real-time anomaly detection across segments, campaigns, and complaint trends.")
    st.divider()

    with st.spinner("Running anomaly detectors..."):
        from src.modeling.anomaly import run_all_detectors, anomaly_summary_text
        results = run_all_detectors(df)

    # ── Status cards ──
    for name, result in results.items():
        label = name.replace("_", " ").title()
        if result.flagged:
            st.markdown(
                f'<div class="anomaly-flag">⚠ <b>{label}</b> — {result.summary}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ok-flag">✓ <b>{label}</b> — {result.summary}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Daily Complaint Volume")
        df_dates = df.copy()
        df_dates["sent_date"] = pd.to_datetime(df_dates["sent_date"], errors="coerce")
        daily = df_dates.groupby(df_dates["sent_date"].dt.date).agg(
            complaints=("complaint_flag", "sum"),
            total=("complaint_flag", "count"),
        ).reset_index()
        daily.columns = ["Date", "Complaints", "Total"]
        daily["Rate"] = daily["Complaints"] / daily["Total"]

        fig5 = px.line(
            daily, x="Date", y="Complaints",
            template="plotly_dark",
            color_discrete_sequence=["#f87171"],
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        complaint_result = results.get("complaint_spike")
        if complaint_result and complaint_result.anomalies:
            for a in complaint_result.anomalies[:3]:
                fig5.add_vline(
                    x=a["date"], line_dash="dot", line_color="#fbbf24",
                    annotation_text=f"z={a['zscore']:.1f}",
                )
        st.plotly_chart(fig5, use_container_width=True)

    with col_b:
        st.markdown("#### Campaign Open Rate Heatmap")
        camp = df.groupby(["campaign_id", "segment"]).agg(
            open_rate=("opened", "mean")
        ).reset_index()
        pivot = camp.pivot(index="campaign_id", columns="segment", values="open_rate")
        fig6 = px.imshow(
            pivot, color_continuous_scale="RdYlGn",
            template="plotly_dark",
            aspect="auto",
        )
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Anomaly detail table ──
    camp_result = results.get("campaign_underperformance")
    if camp_result and camp_result.anomalies:
        st.markdown("#### Underperforming Campaigns")
        st.dataframe(
            pd.DataFrame(camp_result.anomalies),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CASE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Case Explorer":
    st.markdown("# Case Explorer")
    st.markdown("Filter, explore, and score individual customer communication cases.")
    st.divider()

    # ── Filters ──
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        seg_filter = st.multiselect("Segment", df["segment"].unique(),
                                    default=list(df["segment"].unique()))
    with f2:
        ch_filter = st.multiselect("Channel", df["channel"].unique(),
                                   default=list(df["channel"].unique()))
    with f3:
        int_filter = st.selectbox("Intervention Flag", ["All", "Needs Intervention", "No Intervention"])
    with f4:
        score_model = st.checkbox("Add Model Scores", value=False,
                                  help="Runs batch scoring — may take a few seconds")

    # Apply filters
    filtered = df[df["segment"].isin(seg_filter) & df["channel"].isin(ch_filter)]
    if int_filter == "Needs Intervention":
        filtered = filtered[filtered["needs_intervention"] == 1]
    elif int_filter == "No Intervention":
        filtered = filtered[filtered["needs_intervention"] == 0]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

    if score_model and len(filtered) > 0:
        with st.spinner("Scoring cases..."):
            filtered = score_dataframe(filtered)

    # ── Table ──
    display_cols = ["customer_id", "segment", "channel", "campaign_id",
                    "engagement_score", "complaint_flag", "escalation_flag",
                    "days_since_last_contact", "needs_intervention"]
    if "intervention_score" in filtered.columns:
        display_cols += ["intervention_score", "risk_band"]

    st.dataframe(
        filtered[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "engagement_score":   st.column_config.ProgressColumn("Engagement", min_value=0, max_value=1),
            "needs_intervention": st.column_config.CheckboxColumn("Intervention?"),
            "complaint_flag":     st.column_config.CheckboxColumn("Complaint"),
            "escalation_flag":    st.column_config.CheckboxColumn("Escalation"),
        }
    )

    # ── Case drill-down ──
    st.divider()
    st.markdown("#### Case Drill-down")
    selected_id = st.selectbox(
        "Select a customer ID to inspect",
        options=filtered["customer_id"].head(50).tolist(),
    )
    if selected_id:
        case = filtered[filtered["customer_id"] == selected_id].iloc[0]
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("Engagement Score", f"{case['engagement_score']:.3f}")
            st.metric("Tenure (months)", int(case["tenure_months"]))
        with dc2:
            st.metric("Days Since Contact", int(case["days_since_last_contact"]))
            st.metric("Opt-out", "Yes" if case["opt_out_flag"] else "No")
        with dc3:
            st.metric("Complaint", "Yes" if case["complaint_flag"] else "No")
            st.metric("Escalation", "Yes" if case["escalation_flag"] else "No")
        if "intervention_score" in case:
            risk = case.get("risk_band", "Unknown")
            colour = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "⚪")
            st.metric("Intervention Score", f"{case['intervention_score']:.3f}",
                      delta=f"{colour} {risk} Risk")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AI SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "AI Summary":
    st.markdown("# AI Summary")
    st.markdown("Generate next-best-action summaries for individual customer cases.")
    st.divider()

    # ── Customer selector ──
    col_sel, col_opts = st.columns([2, 1])

    with col_sel:
        intervention_cases = df[df["needs_intervention"] == 1]
        selected_id = st.selectbox(
            "Select a customer case (intervention-flagged)",
            options=intervention_cases["customer_id"].head(100).tolist(),
        )

    with col_opts:
        force_stub = st.checkbox("Use stub (no OpenAI call)", value=True)
        include_anomaly = st.checkbox("Include anomaly context", value=True)

    if selected_id:
        case = df[df["customer_id"] == selected_id].iloc[0].to_dict()

        # Show case summary
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Segment", case["segment"])
        i2.metric("Channel", case["channel"])
        i3.metric("Engagement", f"{case['engagement_score']:.2f}")
        i4.metric("Days Silent", int(case["days_since_last_contact"]))

        st.divider()

        if st.button("Generate Next-Best-Action Summary", type="primary",
                     use_container_width=True):
            with st.spinner("Generating summary..."):
                # Get anomaly context
                anomaly_text = ""
                if include_anomaly:
                    from src.modeling.anomaly import run_all_detectors, anomaly_summary_text
                    anomaly_results = run_all_detectors(df)
                    anomaly_text = anomaly_summary_text(anomaly_results)

                # Score the customer
                model, feature_names = load_model_and_features()
                score_result = None
                if model is not None:
                    from src.modeling.score import score_customer
                    score_result = score_customer(
                        case, model=model, expected_columns=feature_names
                    )

                # Generate summary
                from src.genai.summarizer import generate_summary
                result = generate_summary(
                    customer=case,
                    score_result=score_result,
                    anomaly_summary=anomaly_text,
                    force_stub=force_stub,
                )

            # ── Display result ──
            source_badge = (
                '<span class="source-badge badge-openai">OpenAI</span>'
                if result.source == "openai"
                else '<span class="source-badge badge-stub">Stub</span>'
            )

            risk_label = ""
            if score_result:
                risk_colours = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
                cls = risk_colours.get(score_result.risk_band, "")
                risk_label = f'<span class="{cls}">● {score_result.risk_band} Risk</span>'

            st.markdown(f"""
            <div class="summary-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                    <div>{source_badge} &nbsp; {risk_label}</div>
                    <div style="font-size:0.75rem; color:#475569; font-family:'DM Mono',monospace;">
                        {selected_id}
                    </div>
                </div>
                <div class="summary-text">"{result.summary}"</div>
            </div>
            """, unsafe_allow_html=True)

            if include_anomaly and anomaly_text:
                with st.expander("Anomaly context used"):
                    st.text(anomaly_text)

            if score_result:
                st.caption(
                    f"Model score: {score_result.intervention_score:.4f} · "
                    f"Threshold: 0.45 · "
                    f"Flag: {'✓ Intervention' if score_result.needs_intervention else '○ No intervention'}"
                )

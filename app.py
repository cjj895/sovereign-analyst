"""
app.py — Sovereign Analyst Cockpit
====================================
Streamlit entry point.  Run from the project root with:

    .venv/bin/streamlit run app.py

Layout
------
Sidebar   : Ticker selector, date range, refresh button
Tab 1     : Portfolio Overview — 4 KPI cards + holdings table
Tab 2     : Sovereign Signal  — Candlestick chart + AI sentiment overlay
Tab 3     : Truth Layer       — Surgical Delta diff + Source-Trace audit
Tab 4     : Audit Trail       — Full analyst notes history table
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── project root on sys.path so relative imports work when Streamlit cwd varies
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.charts import build_signal_chart
from ui.panels import render_delta_diff, render_source_trace
from ui.queries import (
    get_all_notes,
    get_chroma_trace,
    get_latest_delta,
    get_latest_note,
    get_notes,
    get_ohlcv,
    get_portfolio,
    get_tickers,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS injection ──────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Metric card border */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #8b949e;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #161b22;
    color: #f0f6fc;
    border-bottom: 2px solid #58a6ff;
}

/* Expander header */
details > summary { font-size: 0.88rem; }

/* Dataframe scrollbar */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Sovereign Analyst")
    st.markdown(
        "<p style='color:#8b949e; font-size:0.78rem; margin-top:-8px;'>"
        "Command-grade portfolio intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tickers = get_tickers()
    if not tickers:
        st.warning("No investment transactions found.")
        st.caption("Run `sovereign init` then seed your transactions.")
        st.stop()

    ticker: str = st.selectbox("Active Ticker", tickers, index=0)

    st.markdown("**Chart Range**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date: date = st.date_input(
            "From",
            value=date.today() - timedelta(days=365),
            label_visibility="collapsed",
        )
    with col_e:
        end_date: date = st.date_input(
            "To",
            value=date.today(),
            label_visibility="collapsed",
        )

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Read-only · SQLite + ChromaDB")
    st.caption(f"DB: `data/sovereign.db`")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (runs once per session / TTL window)
# ══════════════════════════════════════════════════════════════════════════════
portfolio    = get_portfolio()
holdings_df  = portfolio.get("holdings", pd.DataFrame())
summary_df   = portfolio.get("summary",  pd.DataFrame())

notes_for_ticker   = get_notes(ticker, limit=50)
latest_note_dict   = get_latest_note(ticker)
latest_delta_dict  = get_latest_delta(ticker)

# ── compute KPI values ────────────────────────────────────────────────────────
# 1. Portfolio-wide totals
total_value:      float = 0.0
unrealised_pnl:   float = 0.0
unrealised_pct:   float = 0.0
if summary_df is not None and not summary_df.empty:
    _row = summary_df.iloc[0]
    total_value    = float(_row.get("total_market_value",    0) or 0)
    unrealised_pnl = float(_row.get("total_unrealised_pnl",  0) or 0)
    unrealised_pct = float(_row.get("unrealised_pnl_pct",    0) or 0)

# 2. Ticker-level sentiment
_SENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
if notes_for_ticker:
    _scores = [_SENT_MAP.get((n.get("sentiment") or "neutral").lower(), 0.0)
               for n in notes_for_ticker]
    _avg_raw = sum(_scores) / len(_scores)
    avg_sentiment_label  = ("Positive" if _avg_raw > 0.2
                             else "Negative" if _avg_raw < -0.2
                             else "Neutral")
    avg_sentiment_delta  = f"{_avg_raw:+.2f} ({len(notes_for_ticker)} notes)"
else:
    avg_sentiment_label = "N/A"
    avg_sentiment_delta = "no notes yet"

# 3. Ticker-level audit confidence
_conf = latest_note_dict.get("confidence_score") if latest_note_dict else None
if _conf is not None:
    confidence_display = f"{_conf:.1f}%"
    confidence_delta   = ("HIGH" if _conf >= 80 else "MEDIUM" if _conf >= 60 else "LOW")
else:
    confidence_display = "—"
    confidence_delta   = "not audited"

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Overview",
    "📈 Sovereign Signal",
    "🔬 Truth Layer",
    "📋 Audit Trail",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Portfolio Value",
            f"${total_value:,.2f}",
            delta=f"${unrealised_pnl:+,.2f} unrealised",
            delta_color="normal",
        )
    with c2:
        sign = "+" if unrealised_pct >= 0 else ""
        st.metric(
            "Unrealized P&L",
            f"{sign}{unrealised_pct:.2f}%",
            delta_color="normal",
        )
    with c3:
        st.metric(
            f"Avg Sentiment · {ticker}",
            avg_sentiment_label,
            delta=avg_sentiment_delta,
            delta_color="off",
        )
    with c4:
        st.metric(
            f"Audit Confidence · {ticker}",
            confidence_display,
            delta=confidence_delta,
            delta_color="off",
        )

    st.markdown("---")
    st.subheader("Current Holdings")

    if holdings_df is not None and not holdings_df.empty:
        _display_cols = [
            col for col in [
                "asset", "quantity", "avg_cost", "capital_invested",
                "current_price", "market_value", "unrealised_pnl", "weight",
            ]
            if col in holdings_df.columns
        ]

        _fmt: dict[str, str] = {}
        if "avg_cost"           in _display_cols: _fmt["avg_cost"]           = "{:.2f}"
        if "capital_invested"   in _display_cols: _fmt["capital_invested"]   = "{:,.2f}"
        if "current_price"      in _display_cols: _fmt["current_price"]      = "{:.2f}"
        if "market_value"       in _display_cols: _fmt["market_value"]       = "{:,.2f}"
        if "unrealised_pnl"     in _display_cols: _fmt["unrealised_pnl"]     = "{:+,.2f}"
        if "weight"             in _display_cols: _fmt["weight"]             = "{:.1f}%"
        if "quantity"           in _display_cols: _fmt["quantity"]           = "{:.4f}"

        st.dataframe(
            holdings_df[_display_cols].style.format(_fmt, na_rep="—"),
            use_container_width=True,
            height=360,
        )

        # Realised P&L inline
        realised_df = portfolio.get("realised", pd.DataFrame())
        if realised_df is not None and not realised_df.empty:
            with st.expander("Realised P&L", expanded=False):
                _r_fmt: dict[str, str] = {}
                if "pnl"  in realised_df.columns: _r_fmt["pnl"]  = "{:+,.2f}"
                if "divs" in realised_df.columns: _r_fmt["divs"] = "{:,.2f}"
                st.dataframe(
                    realised_df.style.format(_r_fmt, na_rep="—"),
                    use_container_width=True,
                )
    else:
        st.info("No holdings found. Ensure transactions are loaded and the DB is seeded.")

    # Portfolio error passthrough
    if "_error" in portfolio:
        st.error(f"Pipeline error: {portfolio['_error']}")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — SOVEREIGN SIGNAL CHART
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Sovereign Signal — {ticker}")

    ohlcv_df = get_ohlcv(ticker, str(start_date), str(end_date))

    if ohlcv_df.empty:
        st.warning(
            f"No price data returned for **{ticker}** between "
            f"{start_date} and {end_date}. "
            "Check that the ticker is valid and the date range includes trading days."
        )
    else:
        fig = build_signal_chart(ticker, ohlcv_df, notes_for_ticker)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Recent notes beneath the chart
    if notes_for_ticker:
        st.markdown("---")
        with st.expander(
            f"AI Notes History ({len(notes_for_ticker)} entries)",
            expanded=False,
        ):
            for note in notes_for_ticker[:8]:
                _sent      = (note.get("sentiment") or "neutral").lower()
                _date_str  = str(note.get("created_at") or "")[:10]
                _summary   = note.get("summary") or ""
                _model     = note.get("model")   or ""
                _sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(_sent, "⚪")
                st.markdown(
                    f"**{_date_str}** &nbsp; {_sent_icon} `{_sent.upper()}` &nbsp; "
                    f"<span style='color:#8b949e; font-size:0.78rem;'>{_model}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(_summary)
                st.markdown("---")
    else:
        st.caption(f"No analyst notes found for {ticker}. Run `sovereign note {ticker}` to generate one.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — TRUTH LAYER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    delta_col, trace_col = st.columns([3, 2], gap="large")

    # ── LEFT: Surgical Delta ─────────────────────────────────────────────────
    with delta_col:
        st.subheader("Surgical Delta — Risk Factor Changes")

        if latest_delta_dict:
            _intensity   = latest_delta_dict.get("intensity_delta")
            _verdict     = latest_delta_dict.get("summary") or ""   # summary stores the verdict
            _acc_ref     = latest_delta_dict.get("accession_number") or ""
            _created     = str(latest_delta_dict.get("created_at") or "")[:10]

            # Intensity badge row
            _i_colour = "#f85149" if (_intensity or 0) > 0 else "#3fb950" if (_intensity or 0) < 0 else "#8b949e"
            _i_sign   = f"+{_intensity}" if (_intensity or 0) > 0 else str(_intensity or 0)
            st.markdown(
                f'<span style="background:{_i_colour}20; color:{_i_colour}; '
                f'border:1px solid {_i_colour}; border-radius:4px; '
                f'padding:3px 12px; font-weight:700; font-size:0.88rem;">'
                f'Intensity {_i_sign} / 10</span>'
                f'&nbsp;&nbsp;<span style="color:#8b949e; font-size:0.82rem;">{_created}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Verdict:** {_verdict}")
            st.caption(f"Filings: {_acc_ref}")
            st.markdown("---")

        render_delta_diff(latest_delta_dict)

        # ── Generate New Analysis button ─────────────────────────────────────
        st.markdown("---")
        if st.button(
            f"⚡ Generate New Delta Analysis for {ticker}",
            type="primary",
            use_container_width=True,
            help="Calls the Gemini API — uses token budget.",
        ):
            with st.spinner(f"Calling Gemini (temperature=0.1) for {ticker}..."):
                try:
                    from scripts.analyze_deltas import run_delta
                    _new_id = run_delta(ticker)
                    st.success(f"Delta saved (id={_new_id}). Refreshing data...")
                    st.cache_data.clear()
                    st.rerun()
                except SystemExit:
                    st.error(
                        "Delta generation failed. Ensure:\n"
                        "- GEMINI_API_KEY is set in `.env`\n"
                        "- At least 2 preprocessed 10-K filings exist for this ticker\n"
                        "- Run `sovereign fetch` → `sovereign preprocess` first"
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # ── RIGHT: Source-Trace Panel ────────────────────────────────────────────
    with trace_col:
        st.subheader("Source-Trace Audit")

        if latest_note_dict and latest_note_dict.get("risks"):
            try:
                _risks_list: list[str] = json.loads(latest_note_dict["risks"])
            except (json.JSONDecodeError, TypeError):
                _risks_list = []

            if _risks_list:
                st.caption(
                    f"Tracing {len(_risks_list)} risk claim(s) from note dated "
                    f"{str(latest_note_dict.get('created_at') or '')[:10]} "
                    "against ChromaDB filing chunks."
                )
                with st.spinner("Querying ChromaDB..."):
                    traces = get_chroma_trace(json.dumps(_risks_list), ticker)
                render_source_trace(traces)
            else:
                st.info("The latest note has no extracted risk claims.")
        else:
            st.info(
                f"No analyst note found for **{ticker}**.\n\n"
                "Run `sovereign note {ticker}` to generate one."
            )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Audit Trail — All Analyst Notes")

    all_notes_df = get_all_notes()

    if all_notes_df.empty:
        st.info("No analyst notes generated yet.")
    else:
        # Filter controls
        _filter_col1, _filter_col2, _ = st.columns([1, 1, 4])
        with _filter_col1:
            _ticker_filter = st.selectbox(
                "Filter ticker",
                ["All"] + sorted(all_notes_df["ticker"].dropna().unique().tolist()),
                key="audit_ticker_filter",
            )
        with _filter_col2:
            _sent_filter = st.selectbox(
                "Filter sentiment",
                ["All", "positive", "neutral", "negative"],
                key="audit_sent_filter",
            )

        _filtered = all_notes_df.copy()
        if _ticker_filter != "All":
            _filtered = _filtered[_filtered["ticker"] == _ticker_filter]
        if _sent_filter != "All":
            _filtered = _filtered[_filtered["sentiment"] == _sent_filter]

        st.caption(f"Showing {len(_filtered)} of {len(all_notes_df)} notes")

        _display_cols = [
            col for col in [
                "ticker", "created_at", "sentiment",
                "confidence_score", "intensity_delta", "summary",
            ]
            if col in _filtered.columns
        ]

        _col_config: dict[str, Any] = {
            "ticker":      st.column_config.TextColumn("Ticker", width="small"),
            "created_at":  st.column_config.TextColumn("Generated At", width="medium"),
            "sentiment":   st.column_config.TextColumn("Sentiment", width="small"),
            "summary":     st.column_config.TextColumn("Summary / Verdict", width="large"),
        }

        if "confidence_score" in _filtered.columns:
            _col_config["confidence_score"] = st.column_config.ProgressColumn(
                "Confidence",
                format="%.1f%%",
                min_value=0,
                max_value=100,
                width="medium",
            )
        if "intensity_delta" in _filtered.columns:
            _col_config["intensity_delta"] = st.column_config.NumberColumn(
                "Intensity Δ",
                format="%+d",
                min_value=-10,
                max_value=10,
                width="small",
            )

        st.dataframe(
            _filtered[_display_cols],
            column_config=_col_config,
            use_container_width=True,
            height=520,
            hide_index=True,
        )

        # Row-detail expander
        st.markdown("---")
        st.subheader("Note Detail")
        _note_ids = _filtered.get("id", pd.Series(dtype=int)).tolist() if "id" in _filtered.columns else []
        if _note_ids:
            _sel_idx = st.selectbox(
                "Select note index",
                range(len(_filtered)),
                format_func=lambda i: (
                    f"{_filtered.iloc[i].get('ticker', '')} · "
                    f"{str(_filtered.iloc[i].get('created_at', ''))[:10]} · "
                    f"{_filtered.iloc[i].get('sentiment', '')}"
                ),
                key="audit_note_detail",
            )
            _detail = _filtered.iloc[_sel_idx].to_dict()
            _c1, _c2 = st.columns(2)
            with _c1:
                st.markdown("**Summary**")
                st.markdown(_detail.get("summary") or "_empty_")
            with _c2:
                _raw_risks = _detail.get("risks") or "[]"
                st.markdown("**Extracted Risks**")
                try:
                    _risk_items: list[str] = json.loads(_raw_risks)
                    for _r in _risk_items:
                        st.markdown(f"- {_r}")
                except (json.JSONDecodeError, TypeError):
                    st.code(_raw_risks, language="json")

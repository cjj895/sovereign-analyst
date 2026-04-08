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
Tab 5     : Portfolio Manager — Trade entry, CSV import, transaction history
"""
from __future__ import annotations

import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── project root on sys.path so relative imports work when Streamlit cwd varies
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.charts import build_signal_chart
from ui.panels import render_delta_diff, render_source_trace
from ui.queries import (
    bulk_import_from_df,
    delete_transaction_by_id,
    get_all_notes,
    get_all_transactions,
    get_chroma_trace,
    get_latest_delta,
    get_latest_note,
    get_notes,
    get_ohlcv,
    get_portfolio,
    get_tickers,
    write_transaction,
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

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    font-weight: 600;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #388bfd;
}
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
    if tickers:
        ticker: str | None = st.selectbox("Active Ticker", tickers, index=0)
    else:
        st.warning("No transactions found.")
        st.caption("Open **⚙️ Portfolio Manager** to add your first trade or import a CSV.")
        ticker = None

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

    if start_date > end_date:
        st.warning("**From** date is after **To** date — chart range is inverted.")

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("SQLite + ChromaDB · Local-first")
    st.caption("DB: `data/sovereign.db`")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (runs once per session / TTL window)
# ══════════════════════════════════════════════════════════════════════════════
_NO_DATA_MSG = "No transactions loaded. Use **⚙️ Portfolio Manager** to add trades."

portfolio   = get_portfolio() if ticker else {}
holdings_df = portfolio.get("holdings", pd.DataFrame())
summary_df  = portfolio.get("summary",  pd.DataFrame())

notes_for_ticker  = get_notes(ticker, limit=50)  if ticker else []
latest_note_dict  = get_latest_note(ticker)       if ticker else None
latest_delta_dict = get_latest_delta(ticker)      if ticker else None

# ── compute KPI values ────────────────────────────────────────────────────────
total_value:    float = 0.0
unrealised_pnl: float = 0.0
unrealised_pct: float = 0.0
if summary_df is not None and not summary_df.empty:
    _row = summary_df.iloc[0]
    total_value    = float(_row.get("total_market_value",    0) or 0)
    unrealised_pnl = float(_row.get("total_unrealised_pnl",  0) or 0)
    unrealised_pct = float(_row.get("unrealised_pnl_pct",    0) or 0)

_SENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
if notes_for_ticker:
    _scores = [_SENT_MAP.get((n.get("sentiment") or "neutral").lower(), 0.0)
               for n in notes_for_ticker]
    _avg_raw = sum(_scores) / len(_scores)
    avg_sentiment_label = ("Positive" if _avg_raw > 0.2
                            else "Negative" if _avg_raw < -0.2
                            else "Neutral")
    avg_sentiment_delta = f"{_avg_raw:+.2f} ({len(notes_for_ticker)} notes)"
else:
    avg_sentiment_label = "N/A"
    avg_sentiment_delta = "no notes yet"

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Overview",
    "📈 Sovereign Signal",
    "🔬 Truth Layer",
    "📋 Audit Trail",
    "⚙️ Portfolio Manager",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not ticker:
        st.info(_NO_DATA_MSG)
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Portfolio Value",
                f"${total_value:,.2f}",
                delta=f"${unrealised_pnl:+,.2f} unrealized",
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
            if "avg_cost"         in _display_cols: _fmt["avg_cost"]         = "{:.2f}"
            if "capital_invested" in _display_cols: _fmt["capital_invested"] = "{:,.2f}"
            if "current_price"    in _display_cols: _fmt["current_price"]    = "{:.2f}"
            if "market_value"     in _display_cols: _fmt["market_value"]     = "{:,.2f}"
            if "unrealised_pnl"   in _display_cols: _fmt["unrealised_pnl"]   = "{:+,.2f}"
            if "weight"           in _display_cols: _fmt["weight"]           = "{:.1f}%"
            if "quantity"         in _display_cols: _fmt["quantity"]         = "{:.4f}"

            st.dataframe(
                holdings_df[_display_cols].style.format(_fmt, na_rep="—"),
                use_container_width=True,
                height=360,
            )

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
            st.info("No holdings found. Ensure transactions are loaded.")

        if "_error" in portfolio:
            st.error(f"Pipeline error: {portfolio['_error']}")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — SOVEREIGN SIGNAL CHART
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not ticker:
        st.info(_NO_DATA_MSG)
    else:
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

        if notes_for_ticker:
            st.markdown("---")
            with st.expander(
                f"AI Notes History ({len(notes_for_ticker)} entries)",
                expanded=False,
            ):
                for note in notes_for_ticker[:8]:
                    _sent     = (note.get("sentiment") or "neutral").lower()
                    _date_str = str(note.get("created_at") or "")[:10]
                    _summary  = note.get("summary") or ""
                    _model    = note.get("model")   or ""
                    _icon     = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(_sent, "⚪")
                    st.markdown(
                        f"**{_date_str}** &nbsp; {_icon} `{_sent.upper()}` &nbsp; "
                        f"<span style='color:#8b949e; font-size:0.78rem;'>{_model}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_summary)
                    st.markdown("---")
        else:
            st.caption(f"No analyst notes for {ticker}. Run `sovereign note {ticker}` to generate one.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — TRUTH LAYER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not ticker:
        st.info(_NO_DATA_MSG)
    else:
        delta_col, trace_col = st.columns([3, 2], gap="large")

        with delta_col:
            st.subheader("Surgical Delta — Risk Factor Changes")

            if latest_delta_dict:
                _intensity = latest_delta_dict.get("intensity_delta")
                _verdict   = latest_delta_dict.get("summary") or ""
                _acc_ref   = latest_delta_dict.get("accession_number") or ""
                _created   = str(latest_delta_dict.get("created_at") or "")[:10]

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
                    except (ValueError, FileNotFoundError, EnvironmentError) as exc:
                        st.error(f"Delta generation failed: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

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
                st.info(f"No analyst note found for **{ticker}**.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Audit Trail — All Analyst Notes")

    all_notes_df = get_all_notes()

    if all_notes_df.empty:
        st.info("No analyst notes generated yet.")
    else:
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
            "ticker":     st.column_config.TextColumn("Ticker",      width="small"),
            "created_at": st.column_config.TextColumn("Generated At", width="medium"),
            "sentiment":  st.column_config.TextColumn("Sentiment",   width="small"),
            "summary":    st.column_config.TextColumn("Summary / Verdict", width="large"),
        }
        if "confidence_score" in _filtered.columns:
            _col_config["confidence_score"] = st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%", min_value=0, max_value=100, width="medium",
            )
        if "intensity_delta" in _filtered.columns:
            _col_config["intensity_delta"] = st.column_config.NumberColumn(
                "Intensity Δ", format="%+d", min_value=-10, max_value=10, width="small",
            )

        st.dataframe(
            _filtered[_display_cols],
            column_config=_col_config,
            use_container_width=True,
            height=520,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Note Detail")
        if not _filtered.empty:
            _sel_idx = st.selectbox(
                "Select a note",
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

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — PORTFOLIO MANAGER  (Interactive data entry)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    pm_tab1, pm_tab2, pm_tab3 = st.tabs([
        "➕ Log Trade",
        "📂 Import CSV",
        "📒 Transaction History",
    ])

    # ── SUB-TAB A: Manual Trade Entry ─────────────────────────────────────────
    with pm_tab1:
        st.subheader("Log a Transaction")
        st.caption(
            "All trades are written directly to `sovereign.db` using the ACB engine. "
            "The portfolio recalculates instantly after saving."
        )
        st.markdown("---")

        # Transaction type controls which fields are active
        _tx_type = st.selectbox(
            "Transaction Type",
            ["buy", "sell", "dividend", "income", "expense", "split"],
            key="pm_tx_type",
        )

        _is_investment = _tx_type in ("buy", "sell", "dividend", "split")
        _needs_price   = _tx_type in ("buy", "sell")
        _needs_qty     = _tx_type in ("buy", "sell", "split")
        _needs_ratio   = _tx_type == "split"
        _needs_amount  = _tx_type in ("dividend", "income", "expense")

        with st.form("trade_entry_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns(2)

            with f_col1:
                f_date = st.date_input(
                    "Date *",
                    value=date.today(),
                    key="pm_f_date",
                )
                f_ticker = st.text_input(
                    "Ticker *" if _is_investment else "Ticker (optional)",
                    placeholder="e.g. AAPL",
                    key="pm_f_ticker",
                ).upper().strip()
                f_asset = st.text_input(
                    "Asset Name",
                    placeholder="e.g. Apple Inc",
                    key="pm_f_asset",
                ).strip()

            with f_col2:
                if _needs_price:
                    f_price = st.number_input(
                        "Price per Share *",
                        min_value=0.0,
                        step=0.01,
                        format="%.4f",
                        key="pm_f_price",
                    )
                else:
                    f_price = None

                if _needs_qty:
                    f_qty = st.number_input(
                        "Quantity *",
                        min_value=0.0,
                        step=0.01,
                        format="%.4f",
                        key="pm_f_qty",
                    )
                else:
                    f_qty = None

                if _needs_ratio:
                    f_ratio = st.number_input(
                        "Split Ratio * (e.g. 10 for 10-for-1)",
                        min_value=0.01,
                        step=0.01,
                        format="%.2f",
                        key="pm_f_ratio",
                    )
                else:
                    f_ratio = None

                if _needs_amount:
                    f_amount_raw = st.number_input(
                        "Amount (cash value, always positive) *",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                        key="pm_f_amount",
                    )
                else:
                    f_amount_raw = None

            f_description = st.text_input(
                "Description / Note",
                placeholder="Optional memo",
                key="pm_f_desc",
            ).strip()

            submitted = st.form_submit_button(
                "💾 Save Transaction",
                use_container_width=True,
            )

        if submitted:
            # ── Validation ────────────────────────────────────────────────────
            _errors: list[str] = []

            if not f_date:
                _errors.append("Date is required.")
            if _is_investment and not f_ticker:
                _errors.append("Ticker is required for investment transactions.")
            if _needs_price and (f_price is None or f_price <= 0):
                _errors.append("Price must be greater than 0.")
            if _needs_qty and (f_qty is None or f_qty <= 0):
                _errors.append("Quantity must be greater than 0.")
            if _needs_ratio and (f_ratio is None or f_ratio <= 0):
                _errors.append("Split ratio must be greater than 0.")
            if _needs_amount and (f_amount_raw is None or f_amount_raw < 0):
                _errors.append("Amount must be 0 or greater.")

            if _errors:
                for _e in _errors:
                    st.error(_e)
            else:
                # ── Compute amount and sign ───────────────────────────────────
                if _tx_type == "buy":
                    _amount = -(f_price * f_qty)  # type: ignore[operator]
                elif _tx_type == "sell":
                    _amount = +(f_price * f_qty)  # type: ignore[operator]
                elif _tx_type == "dividend" or _tx_type == "income":
                    _amount = +f_amount_raw       # type: ignore[operator]
                elif _tx_type == "expense":
                    _amount = -f_amount_raw       # type: ignore[operator]
                else:
                    _amount = None                # split — no cash

                _ok, _msg = write_transaction(
                    date_str=str(f_date),
                    tx_type=_tx_type,
                    ticker=f_ticker or None,
                    asset=f_asset or None,
                    price=f_price,
                    quantity=f_qty,
                    amount=_amount,
                    description=f_description or None,
                    ratio=f_ratio,
                )
                if _ok:
                    st.success(f"✅ {_msg} — {_tx_type.upper()} {f_ticker or ''} on {f_date}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Save failed: {_msg}")

    # ── SUB-TAB B: CSV Import ──────────────────────────────────────────────────
    with pm_tab2:
        st.subheader("Import Transactions from CSV")

        # ── Sample data import ───────────────────────────────────────────────
        st.markdown("#### Option 1 — Import Sample Portfolio")
        st.caption(
            "Loads **all rows** from `data/transactions_sample.csv` into the database. "
            "This is safe to run multiple times — each row is appended regardless of "
            "whether data already exists."
        )
        if st.button("📥 Import Sample Data (AAPL, NVDA, MSFT, AMZN, GOOGL…)", use_container_width=True):
            _csv_path = _ROOT / "data" / "transactions_sample.csv"
            if not _csv_path.exists():
                st.error(f"Sample CSV not found at `{_csv_path}`.")
            else:
                _sample_df = pd.read_csv(_csv_path)
                _inserted, _errs = bulk_import_from_df(_sample_df)
                if _inserted > 0:
                    st.success(f"✅ Imported {_inserted} transactions from sample CSV.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("No rows were imported.")
                if _errs:
                    with st.expander(f"{len(_errs)} row error(s)", expanded=True):
                        for _e in _errs:
                            st.caption(_e)

        st.markdown("---")

        # ── Custom CSV upload ────────────────────────────────────────────────
        st.markdown("#### Option 2 — Upload Your Own CSV")

        with st.expander("Expected CSV format", expanded=False):
            st.markdown(
                "Your CSV must have at minimum `date` and `type` columns. "
                "All other columns are optional.\n\n"
                "| Column | Required | Example |\n"
                "|---|---|---|\n"
                "| `date` | ✅ | `2026-04-01` |\n"
                "| `type` | ✅ | `buy` \\| `sell` \\| `dividend` \\| `income` \\| `expense` \\| `split` |\n"
                "| `ticker` | For buy/sell/div/split | `AAPL` |\n"
                "| `asset` | Optional | `Apple Inc` |\n"
                "| `price` | For buy/sell | `175.50` |\n"
                "| `quantity` | For buy/sell/split | `10` |\n"
                "| `amount` | For div/income/expense | `25.50` |\n"
                "| `ratio` | For split | `4.0` |\n"
                "| `description` | Optional | `Initial purchase` |\n"
            )
            st.download_button(
                "⬇️ Download Sample CSV Template",
                data=(_ROOT / "data" / "transactions_sample.csv").read_bytes(),
                file_name="sovereign_transactions_template.csv",
                mime="text/csv",
            )

        _uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Drag and drop or click to select a .csv file from your broker or spreadsheet.",
        )

        if _uploaded_file is not None:
            try:
                _upload_df = pd.read_csv(io.StringIO(_uploaded_file.getvalue().decode("utf-8")))
                st.caption(f"Preview: {len(_upload_df)} rows detected")
                st.dataframe(_upload_df.head(10), use_container_width=True, height=250)

                if st.button("⬆️ Confirm & Import", type="primary", use_container_width=True):
                    _inserted, _errs = bulk_import_from_df(_upload_df)
                    if _inserted > 0:
                        st.success(f"✅ Imported {_inserted} transactions.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning("No rows were imported — check format.")
                    if _errs:
                        with st.expander(f"{len(_errs)} row error(s)"):
                            for _e in _errs:
                                st.caption(_e)
            except Exception as _exc:
                st.error(f"Could not parse CSV: {_exc}")

    # ── SUB-TAB C: Transaction History ────────────────────────────────────────
    with pm_tab3:
        st.subheader("Transaction History")

        _all_tx = get_all_transactions()

        if _all_tx.empty:
            st.info("No transactions in the database yet. Use 'Log Trade' or 'Import CSV' to get started.")
        else:
            # Filter row
            _hist_col1, _hist_col2, _ = st.columns([1, 1, 3])
            with _hist_col1:
                _hist_ticker = st.selectbox(
                    "Filter by ticker",
                    ["All"] + sorted(_all_tx["ticker"].dropna().unique().tolist()),
                    key="hist_ticker_filter",
                )
            with _hist_col2:
                _hist_type = st.selectbox(
                    "Filter by type",
                    ["All", "buy", "sell", "dividend", "income", "expense", "split"],
                    key="hist_type_filter",
                )

            _tx_filtered = _all_tx.copy()
            if _hist_ticker != "All":
                _tx_filtered = _tx_filtered[_tx_filtered["ticker"] == _hist_ticker]
            if _hist_type != "All":
                _tx_filtered = _tx_filtered[_tx_filtered["type"] == _hist_type]

            st.caption(f"{len(_tx_filtered)} of {len(_all_tx)} transactions")

            _tx_display_cols = [
                c for c in ["id", "date", "type", "ticker", "asset",
                             "price", "quantity", "amount", "description"]
                if c in _tx_filtered.columns
            ]
            _tx_col_cfg: dict[str, Any] = {
                "id":          st.column_config.NumberColumn("ID",       width="small"),
                "date":        st.column_config.TextColumn("Date",       width="small"),
                "type":        st.column_config.TextColumn("Type",       width="small"),
                "ticker":      st.column_config.TextColumn("Ticker",     width="small"),
                "asset":       st.column_config.TextColumn("Asset",      width="medium"),
                "price":       st.column_config.NumberColumn("Price",    format="$%.2f", width="small"),
                "quantity":    st.column_config.NumberColumn("Qty",      format="%.4f",  width="small"),
                "amount":      st.column_config.NumberColumn("Amount",   format="$%.2f", width="small"),
                "description": st.column_config.TextColumn("Note",       width="medium"),
            }

            st.dataframe(
                _tx_filtered[_tx_display_cols],
                column_config=_tx_col_cfg,
                use_container_width=True,
                height=420,
                hide_index=True,
            )

            # ── Delete a transaction ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Delete a Transaction")
            st.caption(
                "Deleting a transaction recalculates ACB for all subsequent trades of that ticker. "
                "Use with care."
            )
            _del_col1, _del_col2 = st.columns([1, 3])
            with _del_col1:
                _del_id = st.number_input(
                    "Transaction ID to delete",
                    min_value=1,
                    step=1,
                    key="pm_del_id",
                )
            with _del_col2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete Transaction", type="secondary"):
                    _del_ok, _del_msg = delete_transaction_by_id(int(_del_id))
                    if _del_ok:
                        st.success(_del_msg)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(_del_msg)

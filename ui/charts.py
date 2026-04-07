"""
ui/charts.py
------------
Plotly figure builders for the Sovereign Cockpit.

build_signal_chart:
    Candlestick price chart with AI sentiment markers overlaid as bubble
    annotations.  Hovering a bubble reveals the AI executive summary for
    that note date.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour constants ──────────────────────────────────────────────────────────
_COLOUR_BG = "#0d1117"
_COLOUR_GRID = "#21262d"
_COLOUR_TEXT = "#c9d1d9"
_COLOUR_UP = "#3fb950"        # green candle
_COLOUR_DOWN = "#f85149"      # red candle
_COLOUR_VOLUME = "#30363d"
_COLOUR_POSITIVE = "#3fb950"  # sentiment bubble — positive
_COLOUR_NEGATIVE = "#f85149"  # sentiment bubble — negative
_COLOUR_NEUTRAL = "#8b949e"   # sentiment bubble — neutral
_COLOUR_LINE = "#58a6ff"      # MA line


def _sentiment_colour(sentiment: str) -> str:
    mapping = {
        "positive": _COLOUR_POSITIVE,
        "negative": _COLOUR_NEGATIVE,
        "neutral":  _COLOUR_NEUTRAL,
    }
    return mapping.get((sentiment or "neutral").lower(), _COLOUR_NEUTRAL)


def _truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text


def build_signal_chart(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    notes: list[dict[str, Any]],
) -> go.Figure:
    """
    Build an interactive candlestick chart for *ticker* with sentiment
    markers and an optional 20-day moving-average overlay.

    Parameters
    ----------
    ticker:   Ticker symbol (used for axis/title labels).
    ohlcv_df: DataFrame with DatetimeIndex and Open/High/Low/Close/Volume columns.
    notes:    List of analyst_notes row dicts from AnalystNoteStore.
              Each dict must contain: created_at, sentiment, summary.

    Returns a Plotly Figure ready for ``st.plotly_chart``.
    """
    # ── guard: empty data ─────────────────────────────────────────────────────
    if ohlcv_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker} — no price data",
            paper_bgcolor=_COLOUR_BG,
            plot_bgcolor=_COLOUR_BG,
        )
        return fig

    # ── normalise columns ─────────────────────────────────────────────────────
    ohlcv = ohlcv_df.copy()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    # Flatten MultiIndex if present (yfinance sometimes returns one)
    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = ohlcv.columns.get_level_values(0)

    has_volume = "Volume" in ohlcv.columns and ohlcv["Volume"].notna().any()

    # ── subplots: price (top) + volume (bottom, if available) ─────────────────
    row_heights = [0.75, 0.25] if has_volume else [1.0]
    rows = 2 if has_volume else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ── candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name=ticker,
            increasing_line_color=_COLOUR_UP,
            decreasing_line_color=_COLOUR_DOWN,
            increasing_fillcolor=_COLOUR_UP,
            decreasing_fillcolor=_COLOUR_DOWN,
            line_width=1,
            whiskerwidth=0.8,
        ),
        row=1, col=1,
    )

    # ── 20-day simple moving average ──────────────────────────────────────────
    if len(ohlcv) >= 20:
        ma20 = ohlcv["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ma20,
                mode="lines",
                name="MA20",
                line=dict(color=_COLOUR_LINE, width=1.2, dash="dot"),
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # ── volume bars ───────────────────────────────────────────────────────────
    if has_volume:
        colours = [
            _COLOUR_UP if c >= o else _COLOUR_DOWN
            for o, c in zip(ohlcv["Open"], ohlcv["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=ohlcv.index,
                y=ohlcv["Volume"],
                name="Volume",
                marker_color=colours,
                marker_opacity=0.4,
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── sentiment markers ─────────────────────────────────────────────────────
    if notes:
        notes_df = pd.DataFrame(notes)
        notes_df["_date"] = pd.to_datetime(notes_df["created_at"]).dt.normalize()

        # Map each note to the nearest available trading-day close price
        ohlcv_dates = ohlcv.index.normalize()
        price_series = pd.Series(
            ohlcv["Close"].values,
            index=ohlcv_dates,
            name="close",
        )

        marker_x: list = []
        marker_y: list = []
        marker_colours: list[str] = []
        hover_texts: list[str] = []

        for _, note in notes_df.iterrows():
            note_date = note["_date"]
            # Find closest trading day at or before note date
            candidates = ohlcv_dates[ohlcv_dates <= note_date]
            if candidates.empty:
                continue
            closest = candidates[-1]
            price = float(price_series.loc[closest])

            sentiment = str(note.get("sentiment", "neutral")).lower()
            summary = str(note.get("summary", "")).strip()
            model = str(note.get("model", ""))
            created = str(note.get("created_at", ""))[:10]

            marker_x.append(closest)
            marker_y.append(price)
            marker_colours.append(_sentiment_colour(sentiment))
            hover_texts.append(
                f"<b>{created}</b><br>"
                f"<span style='color:{_sentiment_colour(sentiment)}'>"
                f"{sentiment.upper()}</span><br><br>"
                f"{_truncate(summary)}<br><br>"
                f"<i>Model: {model}</i>"
            )

        if marker_x:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    name="AI Signal",
                    marker=dict(
                        color=marker_colours,
                        size=14,
                        symbol="circle",
                        line=dict(width=2, color=_COLOUR_BG),
                        opacity=0.92,
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts,
                ),
                row=1, col=1,
            )

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> — Price & AI Signals",
            font=dict(color=_COLOUR_TEXT, size=16),
        ),
        paper_bgcolor=_COLOUR_BG,
        plot_bgcolor=_COLOUR_BG,
        font=dict(color=_COLOUR_TEXT, family="monospace"),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            bgcolor=_COLOUR_GRID,
            bordercolor=_COLOUR_GRID,
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_rangeslider_visible=False,
    )

    # ── axis styling ──────────────────────────────────────────────────────────
    axis_common = dict(
        gridcolor=_COLOUR_GRID,
        zerolinecolor=_COLOUR_GRID,
        color=_COLOUR_TEXT,
        tickfont=dict(size=11),
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)

    # Price axis label
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

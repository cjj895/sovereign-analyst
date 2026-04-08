"""
ui/panels.py
------------
HTML-rich panel renderers for the Truth Layer tab.

render_delta_diff:
    Parses the `delta_summary` JSON from an analyst_notes row and renders
    ADDED / REMOVED / SOFTENED risk items with colour-coded backgrounds.

render_source_trace:
    Displays the ChromaDB source chunks that back each risk claim, with
    a HIGH / MEDIUM / LOW confidence badge per claim.
"""
from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

# ── colour palette (matches .streamlit/config.toml dark theme) ────────────────
_GREEN_BG   = "#0d2818"
_GREEN_FG   = "#3fb950"
_RED_BG     = "#2d1117"
_RED_FG     = "#f85149"
_AMBER_BG   = "#2d1f00"
_AMBER_FG   = "#d29922"
_GREY_FG    = "#8b949e"
_BORDER     = "#30363d"


# ── internal helpers ──────────────────────────────────────────────────────────

def _badge(text: str, bg: str, fg: str) -> str:
    """Return an inline HTML badge span."""
    return (
        f'<span style="background:{bg}; color:{fg}; border:1px solid {fg}; '
        f'border-radius:4px; padding:2px 8px; font-size:0.75rem; '
        f'font-weight:700; letter-spacing:0.05em;">{text}</span>'
    )


def _card(content: str, bg: str, border: str) -> str:
    """Return a styled card div wrapping *content* HTML."""
    return (
        f'<div style="background:{bg}; border-left:3px solid {border}; '
        f'border-radius:4px; padding:10px 14px; margin:6px 0; '
        f'font-size:0.88rem; line-height:1.55;">{content}</div>'
    )


def _grade_badge(grade: str, score: int) -> str:
    if grade == "HIGH":
        return _badge(f"HIGH {score}%", _GREEN_BG, _GREEN_FG)
    if grade == "MEDIUM":
        return _badge(f"MED {score}%", _AMBER_BG, _AMBER_FG)
    return _badge(f"LOW {score}%", _RED_BG, _RED_FG)


# ── public renderers ──────────────────────────────────────────────────────────

def render_delta_diff(delta_note: dict[str, Any] | None) -> None:
    """
    Render the Surgical Delta analysis as colour-coded HTML blocks.

    Expected source:
        AnalystNoteStore.get_latest_delta(ticker) → dict with
        `delta_summary` column containing the Gemini JSON string.

    Sections rendered:
        ADDED   — green background (new risk disclosures)
        REMOVED — red  background + strikethrough (dropped disclosures)
        SOFTENED — amber side-by-side before/after
    """
    if not delta_note:
        st.info("No delta analysis available for this ticker.")
        return

    raw_json = delta_note.get("delta_summary") or delta_note.get("raw_response", "")
    if not raw_json:
        st.warning("Delta note exists but contains no structured data.")
        return

    try:
        parsed: dict[str, Any] = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        st.warning("Delta data could not be parsed as JSON.")
        st.code(raw_json[:800], language="text")
        return

    added    = parsed.get("added",    [])
    removed  = parsed.get("removed",  [])
    softened = parsed.get("softened", [])

    # ── ADDED ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"{_badge(f'+ ADDED  {len(added)}', _GREEN_BG, _GREEN_FG)} "
        f"<span style='color:{_GREY_FG}; font-size:0.8rem;'>  New disclosures in the latest filing</span>",
        unsafe_allow_html=True,
    )
    if added:
        html_blocks = "".join(
            _card(f"+ &nbsp;{html.escape(str(item))}", _GREEN_BG, _GREEN_FG)
            for item in added
        )
        st.markdown(html_blocks, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p style="color:{_GREY_FG}; font-size:0.85rem; margin:4px 0 12px 4px;">'
            "None detected.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── REMOVED ───────────────────────────────────────────────────────────────
    st.markdown(
        f"{_badge(f'− REMOVED  {len(removed)}', _RED_BG, _RED_FG)} "
        f"<span style='color:{_GREY_FG}; font-size:0.8rem;'>  Highest-signal category — quietly dropped risks</span>",
        unsafe_allow_html=True,
    )
    if removed:
        html_blocks = "".join(
            _card(
                f"<span style='text-decoration:line-through; color:{_RED_FG};'>"
                f"− &nbsp;{html.escape(str(item))}</span>",
                _RED_BG,
                _RED_FG,
            )
            for item in removed
        )
        st.markdown(html_blocks, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p style="color:{_GREY_FG}; font-size:0.85rem; margin:4px 0 12px 4px;">'
            "None detected.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SOFTENED ─────────────────────────────────────────────────────────────
    st.markdown(
        f"{_badge(f'~ SOFTENED  {len(softened)}', _AMBER_BG, _AMBER_FG)} "
        f"<span style='color:{_GREY_FG}; font-size:0.8rem;'>  Language became less alarming without underlying risk changing</span>",
        unsafe_allow_html=True,
    )
    if softened:
        for item in softened:
            original       = html.escape(str(item.get("original",       "")))
            revised        = html.escape(str(item.get("revised",        "")))
            interpretation = item.get("interpretation", "")
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown(
                    _card(
                        f"<b style='color:{_GREY_FG}; font-size:0.75rem;'>BEFORE</b><br>"
                        f"<span style='color:{_RED_FG};'>{original}</span>",
                        _RED_BG,
                        _RED_FG,
                    ),
                    unsafe_allow_html=True,
                )
            with right_col:
                st.markdown(
                    _card(
                        f"<b style='color:{_GREY_FG}; font-size:0.75rem;'>AFTER</b><br>"
                        f"<span style='color:{_AMBER_FG};'>{revised}</span>",
                        _AMBER_BG,
                        _AMBER_FG,
                    ),
                    unsafe_allow_html=True,
                )
            if interpretation:
                st.caption(f"Why it matters: {interpretation}")
    else:
        st.markdown(
            f'<p style="color:{_GREY_FG}; font-size:0.85rem; margin:4px 0 12px 4px;">'
            "None detected.</p>",
            unsafe_allow_html=True,
        )


def render_source_trace(traces: list[dict[str, Any]]) -> None:
    """
    Render the Source-Trace audit panel.

    Each *trace* dict (produced by ``ui.queries.get_chroma_trace``) must contain:
        risk        — original risk claim string
        best_chunk  — 350-char preview of the closest filing chunk
        claim_score — int 0-100
        grade       — "HIGH" | "MEDIUM" | "LOW"
        distance    — cosine distance float
        year        — filing year string (optional)

    If the list contains a single ``{"error": "..."}`` entry, the error is
    displayed instead.
    """
    if not traces:
        st.info("No risks to trace. Generate an analyst note first.")
        return

    # Check for error sentinel
    if len(traces) == 1 and "error" in traces[0]:
        st.warning(traces[0]["error"])
        return

    for i, trace in enumerate(traces, start=1):
        score  = trace.get("claim_score", 0)
        grade  = trace.get("grade", "LOW")
        risk   = trace.get("risk", "")
        chunk  = trace.get("best_chunk", "(no chunk)")
        dist   = trace.get("distance", 1.0)
        year   = trace.get("year", "")

        badge_html = _grade_badge(grade, score)
        year_label = f" · {year}" if year else ""

        with st.expander(
            f"Claim {i}: {risk[:70]}{'…' if len(risk) > 70 else ''}",
            expanded=(i == 1),
        ):
            st.markdown(
                f"{badge_html} "
                f"<span style='color:{_GREY_FG}; font-size:0.78rem;'>"
                f"cosine distance {dist:.4f}{year_label}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:{_BORDER}; border-left:3px solid #58a6ff; '
                f'border-radius:4px; padding:10px 14px; margin-top:8px; '
                f'font-size:0.82rem; font-family:monospace; line-height:1.6; '
                f'color:#c9d1d9; white-space:pre-wrap;">'
                f'{html.escape(str(chunk))}'
                f'</div>',
                unsafe_allow_html=True,
            )

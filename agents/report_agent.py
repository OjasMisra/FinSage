"""
FinSage Report Agent
====================
Assembles validated charts + Cortex analyses into a professional
branded PDF report using reportlab.

Design: Option B — Midnight Teal
    Header/Footer:  #0f2027  (near black)
    Accent:         #00b4d8  (electric teal)
    Signal BULLISH: #06d6a0  (mint green)
    Signal BEARISH: #ef476f  (coral red)
    Signal NEUTRAL: #94a3b8  (slate)

PDF Structure (15-20 pages):
    Page 1    — Cover page
    Page 2    — Table of Contents
    Page 3-4  — Executive Summary + key metrics table + thesis
    Pages 5-10 — One dedicated page per chart (6 charts)
    Page 11-12 — Financial Metrics Summary (detailed table)
    Page 13-14 — Risk Factors (categorized)
    Page 15    — Investment Recommendation
    Page 16-17 — Appendix + data sources + disclaimer

Usage:
    Called by orchestrator.py — not run standalone in production.
    Dev test: python agents/report_agent.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.platypus.flowables import KeepTogether, Flowable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Section marker for dynamic TOC page numbers
# ──────────────────────────────────────────────────────────────
class SectionMarker(Flowable):
    """Zero-height flowable that records its page number when drawn."""
    width = 0
    height = 0

    def __init__(self, section_key: str, page_map: dict):
        super().__init__()
        self.section_key = section_key
        self.page_map = page_map

    def draw(self):
        page_num = self.canv.getPageNumber()
        self.page_map[self.section_key] = page_num

# ──────────────────────────────────────────────────────────────
# Midnight Teal Palette
# ──────────────────────────────────────────────────────────────
C_DARK       = colors.HexColor("#0f2027")   # header / footer bg
C_TEAL       = colors.HexColor("#00b4d8")   # accent / section headers
C_TEAL_LIGHT = colors.HexColor("#e0f7fa")   # subtle teal background
C_BULLISH    = colors.HexColor("#06d6a0")   # BULLISH badge
C_BEARISH    = colors.HexColor("#ef476f")   # BEARISH badge
C_NEUTRAL    = colors.HexColor("#94a3b8")   # NEUTRAL badge
C_WHITE      = colors.white
C_BLACK      = colors.HexColor("#0f172a")
C_GRAY       = colors.HexColor("#64748b")
C_LIGHT_GRAY = colors.HexColor("#f1f5f9")
C_DIVIDER    = colors.HexColor("#e2e8f0")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ──────────────────────────────────────────────────────────────
# Signal helpers
# ──────────────────────────────────────────────────────────────

def _fmt_money(v) -> str:
    """Format a raw money amount as $B / $M string."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v > 1e9:
        return f"${v/1e9:.1f}B"
    if v > 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


def _validate_report_data(charts: list, analysis: dict, ticker: str) -> dict:
    """Pre-assembly sanity checks on report data.

    Logs warnings for inconsistencies that would surface as wrong numbers in
    the final PDF.  Does NOT raise — the report still renders, but the build
    log captures every data-quality concern.

    Returns a dict of corrections applied, e.g.
    ``{"net_margin_override": (old_val, new_val)}``.
    """
    corrections: dict = {}
    chart_map = {c["chart_id"]: c.get("data_summary", {}) for c in charts}
    fin = chart_map.get("financial_health", {})
    eps = chart_map.get("eps_trend", {})
    price = chart_map.get("price_sma", {})
    co_facts = analysis.get("company_overview", {}).get("key_facts", {})

    # 1. Net margin consistency: SEC-derived (financial_health) vs Yahoo (key_facts)
    sec_margin = fin.get("net_margin_pct")
    yf_margin = co_facts.get("net_margin") or co_facts.get("profit_margin")
    if sec_margin is not None and yf_margin is not None:
        try:
            s, y = float(sec_margin), float(yf_margin)
            # Normalise Yahoo decimal to pct if needed
            if abs(y) < 1:
                y = y * 100
            if abs(s - y) > 10 and 5 <= y <= 60:
                logger.warning(
                    "[%s] Net margin mismatch: SEC=%.2f%% vs Yahoo=%.2f%% "
                    "(diff %.2f pp) — overriding SEC value with Yahoo",
                    ticker, s, y, abs(s - y),
                )
                corrections["net_margin_override"] = (s, y)
                fin["net_margin_pct"] = round(y, 4)
            elif abs(s - y) > 5:
                logger.warning(
                    "[%s] Net margin mismatch: SEC=%.2f%% vs Yahoo=%.2f%% "
                    "(diff %.2f pp) — PDF will use SEC value",
                    ticker, s, y, abs(s - y),
                )
        except (ValueError, TypeError):
            pass

    # 2. D/E ratio sanity + cross-source consistency
    de_sec = fin.get("debt_to_equity_ratio")
    de_yf = co_facts.get("debt_to_equity")
    for label, val in [("SEC", de_sec), ("Yahoo", de_yf)]:
        if val is not None:
            try:
                v = float(val)
                if v < 0:
                    logger.warning(
                        "[%s] Negative D/E from %s: %.4f", ticker, label, v,
                    )
                elif v > 20:
                    logger.warning(
                        "[%s] Extreme D/E from %s: %.4f — likely unscaled "
                        "percentage; will be clamped", ticker, label, v,
                    )
            except (ValueError, TypeError):
                pass
    # Reconcile: if both exist and differ substantially, log it so report is aware
    if de_sec is not None and de_yf is not None:
        try:
            ds, dy = float(de_sec), float(de_yf)
            if dy > 0 and abs(ds - dy) / dy > 0.50:
                logger.warning(
                    "[%s] D/E cross-source divergence: SEC=%.4f vs Yahoo=%.4f "
                    "(>50%% relative diff) — using SEC value for consistency",
                    ticker, ds, dy,
                )
        except (ValueError, TypeError):
            pass

    # 3. Stock price sanity (should be positive and < $100k)
    cur_price = price.get("current_price")
    if cur_price is not None:
        try:
            p = float(cur_price)
            if p <= 0 or p > 100_000:
                logger.warning(
                    "[%s] Suspicious stock price: $%.2f", ticker, p,
                )
        except (ValueError, TypeError):
            pass

    # 4. Executive-summary N/A audit — flag metrics that will render as "N/A"
    na_fields = []
    if cur_price is None:
        na_fields.append("Current Price")
    if fin.get("net_margin_pct") is None and yf_margin is None:
        na_fields.append("Net Margin")
    if de_sec is None and de_yf is None:
        na_fields.append("Debt/Equity")
    if eps.get("latest_eps") is None:
        na_fields.append("Latest EPS")
    rev_growth = chart_map.get("revenue_growth", {})
    if rev_growth.get("latest_revenue_growth_yoy") is None:
        na_fields.append("Revenue Growth (YoY)")
    if na_fields:
        logger.warning(
            "[%s] Executive summary will show N/A for: %s",
            ticker, ", ".join(na_fields),
        )

    # 5. EPS vs revenue directional check
    eps_val = eps.get("latest_eps")
    rev_val = fin.get("total_revenue")
    if eps_val is not None and rev_val is not None:
        try:
            if float(eps_val) < 0 and float(rev_val) > 0:
                logger.warning(
                    "[%s] Negative EPS ($%.2f) with positive revenue — "
                    "verify net income data", ticker, float(eps_val),
                )
        except (ValueError, TypeError):
            pass

    return corrections


def _patch_analysis_text(analysis: dict, old_val: float, new_val: float) -> int:
    """Replace stale net-margin values in LLM-generated analysis prose.

    After ``_validate_report_data`` overrides the structured data_summary,
    the already-generated LLM text may still contain the old (bogus) value.
    This function performs a targeted regex substitution so the PDF prose
    is consistent with the corrected metric.

    Returns the total number of substitutions made.
    """
    import re as _re

    old_s1 = f"{old_val:.1f}"   # e.g. "1.9"
    old_s2 = f"{old_val:.2f}"   # e.g. "1.90"
    new_s = f"{new_val:.1f}"    # e.g. "38.0"

    # Match "1.9%" / "1.90%" / "1.9 %" / "1.9 percent" — but not mid-number
    pattern = _re.compile(
        rf"(?<!\d)({_re.escape(old_s2)}|{_re.escape(old_s1)})\s*(%|percent)",
        _re.IGNORECASE,
    )
    replacement = f"{new_s}%"
    count = 0

    for ca in analysis.get("chart_analyses", []):
        text = ca.get("analysis_text", "")
        new_text, n = pattern.subn(replacement, text)
        if n:
            ca["analysis_text"] = new_text
            count += n

    thesis = analysis.get("investment_thesis", "")
    if thesis:
        new_thesis, n = pattern.subn(replacement, thesis)
        if n:
            analysis["investment_thesis"] = new_thesis
            count += n

    return count


def get_signal(chart_id: str, data_summary: dict) -> tuple:
    """
    Derive a BULLISH / BEARISH / NEUTRAL signal from chart data_summary.
    Returns (label, color).
    """
    if chart_id == "price_sma":
        sig = data_summary.get("trend_signal", "NEUTRAL")
        if "BULL" in sig:
            return "BULLISH ▲", C_BULLISH
        elif "BEAR" in sig:
            return "BEARISH ▼", C_BEARISH
        return "NEUTRAL →", C_NEUTRAL

    elif chart_id == "volatility":
        vol = data_summary.get("volatility_30d_pct", 0)
        if vol < 1.5:
            return "LOW VOL ▲", C_BULLISH
        elif vol > 3.0:
            return "HIGH VOL ▼", C_BEARISH
        return "MOD VOL →", C_NEUTRAL

    elif chart_id in ("revenue_growth", "eps_trend"):
        sig = data_summary.get("fundamental_signal",
              data_summary.get("eps_growth_yoy_pct", 0))
        if isinstance(sig, str):
            if "STRONG" in sig or "MODERATE" in sig:
                return "GROWTH ▲", C_BULLISH
            elif "DECLIN" in sig:
                return "DECLINING ▼", C_BEARISH
            return "MIXED →", C_NEUTRAL
        return ("GROWTH ▲", C_BULLISH) if float(sig) > 0 else ("DECLINING ▼", C_BEARISH)

    elif chart_id == "sentiment":
        label = data_summary.get("sentiment_label", "NEUTRAL")
        if "BULL" in label:
            return "BULLISH ▲", C_BULLISH
        elif "BEAR" in label:
            return "BEARISH ▼", C_BEARISH
        return "NEUTRAL →", C_NEUTRAL

    elif chart_id == "financial_health":
        health = data_summary.get("financial_health", "FAIR")
        if health in ("EXCELLENT", "HEALTHY"):
            return "HEALTHY ▲", C_BULLISH
        elif health == "UNPROFITABLE":
            return "WEAK ▼", C_BEARISH
        return "FAIR →", C_NEUTRAL

    elif chart_id == "margin_trend":
        trend = data_summary.get("margin_trend", "insufficient data")
        if trend == "expanding":
            return "EXPANDING ▲", C_BULLISH
        elif trend == "compressing":
            return "COMPRESSING ▼", C_BEARISH
        return "FLAT →", C_NEUTRAL

    elif chart_id == "balance_sheet":
        equity_pct = data_summary.get("equity_pct")
        if equity_pct is not None and equity_pct > 40:
            return "STRONG ▲", C_BULLISH
        elif equity_pct is not None and equity_pct < 20:
            return "WEAK ▼", C_BEARISH
        return "MODERATE →", C_NEUTRAL

    return "NEUTRAL →", C_NEUTRAL


def clean_llm_text(text: str) -> str:
    """Strip markdown syntax that LLMs emit so it renders cleanly in reportlab.

    - Remove **bold** markers (keep the inner text)
    - Remove leading # headers
    - Convert leading '- ' bullet markers to '• '
    - Strip markdown links [text](url) → text
    - Strip inline code backticks
    - Collapse runs of whitespace
    """
    if not text:
        return ""
    import re as _re
    out_lines = []
    for raw in str(text).splitlines():
        line = raw.rstrip()
        line = _re.sub(r"#+\s+", "", line)                          # headers anywhere
        line = _re.sub(r"\*\*(.+?)\*\*", r"\1", line)               # **bold**
        line = _re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", line)      # *italic*
        line = _re.sub(r"`([^`]+)`", r"\1", line)                    # `code`
        line = _re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)      # [text](url)
        line = _re.sub(r"^\s*[-*]\s+", "• ", line)                   # bullet
        line = _re.sub(r"^\s*\d+\.\s+", "• ", line)                 # numbered list
        out_lines.append(line)
    cleaned = "\n".join(out_lines)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def overall_signal(charts: list) -> tuple:
    """Derive overall report signal from all chart signals."""
    bullish = 0
    bearish = 0
    for c in charts:
        label, _ = get_signal(c["chart_id"], c.get("data_summary", {}))
        if "▲" in label:
            bullish += 1
        elif "▼" in label:
            bearish += 1
    if bullish > bearish + 1:
        return "BULLISH ▲", C_BULLISH
    elif bearish > bullish + 1:
        return "BEARISH ▼", C_BEARISH
    return "NEUTRAL →", C_NEUTRAL


# ──────────────────────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_ticker": ParagraphStyle(
            "cover_ticker",
            fontName="Helvetica-Bold",
            fontSize=42,
            textColor=C_WHITE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cover_company": ParagraphStyle(
            "cover_company",
            fontName="Helvetica",
            fontSize=18,
            textColor=C_TEAL,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            fontName="Helvetica",
            fontSize=13,
            textColor=C_DARK,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cover_date": ParagraphStyle(
            "cover_date",
            fontName="Helvetica",
            fontSize=11,
            textColor=C_GRAY,
            alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_WHITE,
            alignment=TA_LEFT,
            leftIndent=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_BLACK,
            alignment=TA_JUSTIFY,
            leading=16,
            spaceAfter=8,
        ),
        "exec_label": ParagraphStyle(
            "exec_label",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_GRAY,
            alignment=TA_LEFT,
        ),
        "exec_value": ParagraphStyle(
            "exec_value",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=C_BLACK,
            alignment=TA_LEFT,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=C_GRAY,
            alignment=TA_JUSTIFY,
            leading=11,
        ),
        "appendix_header": ParagraphStyle(
            "appendix_header",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_TEAL,
            spaceAfter=6,
        ),
        "appendix_body": ParagraphStyle(
            "appendix_body",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_GRAY,
            leading=14,
        ),
        "risk_body": ParagraphStyle(
            "risk_body",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_BLACK,
            alignment=TA_JUSTIFY,
            leading=16,
            spaceAfter=8,
        ),
        "page_title": ParagraphStyle(
            "page_title",
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=C_DARK,
            spaceAfter=12,
        ),
    }
    return styles


# ──────────────────────────────────────────────────────────────
# Page templates — header/footer canvas callbacks
# ──────────────────────────────────────────────────────────────

def draw_cover_bg(canvas, doc):
    """Cover page chrome: dark header + footer bars on a white page.

    Draw rects FIRST with both fill=1 and stroke=1, then text on top.
    """
    canvas.saveState()
    report_date = getattr(doc, "report_date", datetime.now().strftime("%B %d, %Y"))

    header_h = 22 * mm
    footer_h = 18 * mm

    # Header rect (fill + stroke, both dark)
    canvas.setFillColor(C_DARK)
    canvas.setStrokeColor(C_DARK)
    canvas.rect(0, PAGE_H - header_h, PAGE_W, header_h, fill=1, stroke=1)
    # Teal divider under header
    canvas.setFillColor(C_TEAL)
    canvas.setStrokeColor(C_TEAL)
    canvas.rect(0, PAGE_H - header_h - 2, PAGE_W, 2, fill=1, stroke=1)

    # Header text ON TOP of rect
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(MARGIN, PAGE_H - header_h + 8 * mm,
                      "FinSage | AI-Powered Financial Research")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - header_h + 8 * mm,
                           report_date)

    # Footer rect
    canvas.setFillColor(C_DARK)
    canvas.setStrokeColor(C_DARK)
    canvas.rect(0, 0, PAGE_W, footer_h, fill=1, stroke=1)
    canvas.setFillColor(C_TEAL)
    canvas.setStrokeColor(C_TEAL)
    canvas.rect(0, footer_h, PAGE_W, 2, fill=1, stroke=1)

    # Footer text ON TOP of rect
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(
        PAGE_W / 2, footer_h / 2 + 3,
        f"EQUITY RESEARCH REPORT  \u2014  {report_date}"
    )
    canvas.setFillColor(colors.HexColor("#aaaaaa"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        PAGE_W / 2, footer_h / 2 - 10,
        "AI-generated. Not financial advice."
    )
    canvas.restoreState()


def draw_content_page(canvas, doc):
    """Header + footer for all content pages."""
    canvas.saveState()

    # ── Header bar ──────────────────────────────────────────
    canvas.setFillColor(C_DARK)
    canvas.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)

    # Teal accent line under header
    canvas.setFillColor(C_TEAL)
    canvas.rect(0, PAGE_H - 14 * mm - 1.5, PAGE_W, 2, fill=1, stroke=0)

    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(C_WHITE)
    canvas.drawString(MARGIN, PAGE_H - 9 * mm, "FinSage")

    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(C_TEAL)
    ticker = getattr(doc, "_finsage_ticker", "")
    canvas.drawString(MARGIN + 38, PAGE_H - 9 * mm, f"| {ticker} Equity Research Report")

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_NEUTRAL)
    canvas.drawRightString(
        PAGE_W - MARGIN, PAGE_H - 9 * mm,
        datetime.now().strftime("%B %d, %Y")
    )

    # ── Footer bar ──────────────────────────────────────────
    canvas.setFillColor(C_DARK)
    canvas.rect(0, 0, PAGE_W, 12 * mm, fill=1, stroke=0)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_NEUTRAL)
    canvas.drawString(MARGIN, 4 * mm, "Confidential — For Internal Use Only")
    canvas.drawRightString(
        PAGE_W - MARGIN, 4 * mm,
        f"Page {doc.page}"
    )

    # Teal accent line above footer
    canvas.setFillColor(C_TEAL)
    canvas.rect(0, 12 * mm, PAGE_W, 1.5, fill=1, stroke=0)

    canvas.restoreState()


# ──────────────────────────────────────────────────────────────
# Flowable builders
# ──────────────────────────────────────────────────────────────

def investment_rating(charts: list) -> tuple:
    """
    Derive Buy/Hold/Sell rating from chart signals.
    Returns (rating, price_target_multiplier).
    Uses overall_signal() so the cover page and Section 7 always agree.
    """
    signal_label, _ = overall_signal(charts)
    if "BULLISH" in signal_label:
        return "BUY", 1.12
    elif "BEARISH" in signal_label:
        return "SELL", 0.88
    return "HOLD", 1.03


def build_cover(ticker: str, company_name: str, styles: dict,
                overall_sig: tuple, charts: list = None) -> list:
    """Cover page: dark header bar, teal ticker, BUY badge, 3 metric boxes, dark footer."""
    elements = []

    # Space for dark header bar drawn by canvas callback
    elements.append(Spacer(1, 30 * mm))

    # FinSage wordmark
    elements.append(Paragraph(
        '<font color="#00b4d8" size="11"><b>FinSage</b></font>'
        '<font color="#64748b" size="9">  |  AI-Powered Financial Research</font>',
        ParagraphStyle("wm", fontName="Helvetica", fontSize=11,
                       textColor=C_BLACK, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 16 * mm))

    # Ticker — 48pt teal centered
    elements.append(Paragraph(
        ticker,
        ParagraphStyle("ct", fontName="Helvetica-Bold", fontSize=48,
                       textColor=C_TEAL, alignment=TA_CENTER, leading=56)
    ))
    elements.append(Paragraph(company_name, styles["cover_company"]))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph("EQUITY RESEARCH REPORT", styles["cover_subtitle"]))
    elements.append(Paragraph(
        datetime.now().strftime("%B %d, %Y"), styles["cover_date"]
    ))

    elements.append(Spacer(1, 12 * mm))

    # BUY/HOLD/SELL rounded badge with colored fill
    if charts:
        rating, multiplier = investment_rating(charts)
        rating_fills = {
            "BUY":  "#06d6a0",
            "SELL": "#ef476f",
            "HOLD": "#fbbf24",
        }
        fill_hex = rating_fills.get(rating, "#94a3b8")

        badge = Table(
            [[Paragraph(
                f'<font color="#0f2027" size="22"><b>{rating}</b></font>',
                ParagraphStyle("bg", fontName="Helvetica-Bold", fontSize=22,
                               alignment=TA_CENTER, leading=28)
            )]],
            colWidths=[70 * mm],
        )
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill_hex)),
            ("ROUNDEDCORNERS", [10, 10, 10, 10]),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        wrapper = Table([[badge]], colWidths=[PAGE_W - 2 * MARGIN])
        wrapper.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements.append(wrapper)

        elements.append(Spacer(1, 14 * mm))

        # 3-box metrics: Current Price, Price Target, 30D Volatility
        price_d = {}
        vol_d = {}
        for c in charts:
            if c["chart_id"] == "price_sma":
                price_d = c.get("data_summary", {})
            elif c["chart_id"] == "volatility":
                vol_d = c.get("data_summary", {})

        cur_price = price_d.get("current_price")
        price_target = f"${float(cur_price) * multiplier:.2f}" if cur_price else "N/A"
        cur_price_s = f"${cur_price:.2f}" if cur_price else "N/A"
        vol_30d = vol_d.get("volatility_30d_pct")
        vol_s = f"{vol_30d:.2f}%" if vol_30d is not None else "N/A"

        def _box(label, val):
            return Table(
                [[Paragraph(label,
                            ParagraphStyle("bl", fontName="Helvetica", fontSize=8,
                                           textColor=C_GRAY, alignment=TA_CENTER))],
                 [Paragraph(f"<b>{val}</b>",
                            ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=16,
                                           textColor=C_DARK, alignment=TA_CENTER))]],
                colWidths=[(PAGE_W - 2 * MARGIN) * 0.30],
            )

        box_row = Table(
            [[_box("CURRENT PRICE", cur_price_s),
              _box("PRICE TARGET", price_target),
              _box("30D VOLATILITY", vol_s)]],
            colWidths=[(PAGE_W - 2 * MARGIN) / 3] * 3,
        )
        box_row.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
            ("BOX", (0, 0), (0, 0), 0.5, C_DIVIDER),
            ("BOX", (1, 0), (1, 0), 0.5, C_DIVIDER),
            ("BOX", (2, 0), (2, 0), 0.5, C_DIVIDER),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(box_row)

    elements.append(PageBreak())
    return elements


def build_toc(charts: list, styles: dict, page_map: dict = None,
              detail_level: str = "full") -> list:
    """Build Table of Contents page. Uses page_map for dynamic page numbers.

    When detail_level == "summary", sections skipped during summary rendering
    (Financial Metrics Summary and Appendix) are excluded from the TOC.
    """
    elements = []
    elements.append(Paragraph("Table of Contents", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=C_TEAL, spaceAfter=12
    ))

    toc_style = ParagraphStyle(
        "toc_entry",
        fontName="Helvetica",
        fontSize=11,
        textColor=C_BLACK,
        leading=22,
        leftIndent=8,
    )
    toc_section_style = ParagraphStyle(
        "toc_section",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=C_DARK,
        leading=26,
        leftIndent=0,
        spaceBefore=10,
    )

    # Build dotted-leader TOC as a 3-column Table
    pm = page_map or {}
    toc_rows_data = [
        ("1.", "Executive Summary", str(pm.get("exec_summary", "?")), True),
        ("2.", "Company Overview", str(pm.get("company_overview", "?")), True),
        ("3.", "Analysis Sections", str(pm.get("analysis_sections", "?")), True),
    ]
    chart_sub_rows = []
    for i, c in enumerate(charts):
        key = f"chart_{c['chart_id']}"
        chart_sub_rows.append((f"3.{i+1}", c.get("title", c["chart_id"]),
                               str(pm.get(key, "?")), False))

    is_summary = detail_level == "summary"
    # In summary mode, skip Financial Metrics Summary (section 4) and
    # Appendix & Data Sources (last section) since those pages aren't rendered.
    more_rows_data = []
    sec_num = 4
    if not is_summary:
        more_rows_data.append((f"{sec_num}.", "Financial Metrics Summary",
                               str(pm.get("financial_metrics", "?")), True))
        sec_num += 1
    more_rows_data.append((f"{sec_num}.", "Peer Comparison",
                           str(pm.get("peer_comparison", "?")), True))
    sec_num += 1
    more_rows_data.append((f"{sec_num}.", "Risk Factors",
                           str(pm.get("risk_factors", "?")), True))
    sec_num += 1
    more_rows_data.append((f"{sec_num}.", "Investment Recommendation",
                           str(pm.get("recommendation", "?")), True))
    sec_num += 1
    if not is_summary:
        more_rows_data.append((f"{sec_num}.", "Appendix & Data Sources",
                               str(pm.get("appendix", "?")), True))

    entry_style = ParagraphStyle("te", fontName="Helvetica", fontSize=11,
                                  textColor=C_BLACK, leading=14)
    entry_bold = ParagraphStyle("teb", fontName="Helvetica-Bold", fontSize=11,
                                 textColor=C_DARK, leading=14)
    page_style = ParagraphStyle("tp", fontName="Helvetica-Bold", fontSize=11,
                                 textColor=C_TEAL, leading=14, alignment=TA_RIGHT)

    table_rows = []
    for num, title, pg, is_section in toc_rows_data + chart_sub_rows + more_rows_data:
        indent = "" if is_section else "    "
        label = Paragraph(f"<b>{num}</b> &nbsp; {indent}{title}",
                           entry_bold if is_section else entry_style)
        page_p = Paragraph(pg, page_style)
        # Middle cell is empty — we draw a bottom border via TableStyle for the leader
        table_rows.append([label, "", page_p])

    toc_table = Table(table_rows, colWidths=[380, 80, 40], rowHeights=[18] * len(table_rows))
    toc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # Dotted leader: bottom border on middle column only
        ("LINEBELOW", (1, 0), (1, -1), 0.5, C_GRAY, 1, (1, 2)),
    ]))
    elements.append(toc_table)

    elements.append(Spacer(1, 20 * mm))

    # Methodology callout box (teal left accent, gray background)
    style_small_gray = ParagraphStyle(
        "meth_small", fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#555555"), leading=14, alignment=TA_JUSTIFY,
    )
    methodology_text = (
        "This report was generated using the FinSage CAVM "
        "(Chart-Analysis-Validation-Model) pipeline. Charts undergo a 3-iteration "
        "VLM refinement loop using Snowflake Cortex VLM (claude-sonnet-4-6) for visual critique. "
        "Analysis is powered by Cortex LLM (claude-opus-4-6) with grounding in Snowflake ANALYTICS "
        "layer data. SEC filing summaries leverage Cortex SUMMARIZE on 10-K/10-Q filings."
    )
    content_w = PAGE_W - 2 * MARGIN
    meth_table = Table(
        [["",
          Paragraph(
              '<b><font color="#00b4d8">Methodology</font></b><br/>' + methodology_text,
              style_small_gray,
          )]],
        colWidths=[6, content_w - 6],
    )
    meth_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), C_TEAL),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f0f0f0")),
        ("TOPPADDING", (1, 0), (1, 0), 8),
        ("BOTTOMPADDING", (1, 0), (1, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(Spacer(1, 20))
    elements.append(KeepTogether(meth_table))
    logger.info("TOC story items: %d", len(elements))

    elements.append(PageBreak())
    return elements


def build_executive_summary(ticker: str, charts: list,
                             analysis: dict, styles: dict) -> list:
    """Build executive summary page with key metrics grid."""
    elements = []
    elements.append(Paragraph("1. Executive Summary", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    # Key metrics table — pull from data_summaries
    chart_map = {c["chart_id"]: c for c in charts}

    price_data = chart_map.get("price_sma", {}).get("data_summary", {})
    vol_data   = chart_map.get("volatility", {}).get("data_summary", {})
    fund_data  = chart_map.get("revenue_growth", {}).get("data_summary", {})
    eps_data   = chart_map.get("eps_trend", {}).get("data_summary", {})
    sent_data  = chart_map.get("sentiment", {}).get("data_summary", {})
    fin_data   = chart_map.get("financial_health", {}).get("data_summary", {})

    def metric_cell(label, value):
        return [
            Paragraph(label, styles["exec_label"]),
            Paragraph(str(value), styles["exec_value"]),
        ]

    def _fmt_val(v, fmt="{}", suffix="", prefix=""):
        """Format a metric value, returning 'N/A' for None."""
        if v is None:
            return "N/A"
        return f"{prefix}{fmt.format(v)}{suffix}"

    # Net margin: prefer chart data, fall back to analysis company_overview
    net_margin_val = fin_data.get("net_margin_pct")
    if net_margin_val is None:
        co_facts = analysis.get("company_overview", {}).get("key_facts", {})
        net_margin_val = co_facts.get("net_margin") or co_facts.get("profit_margin")
        # DIM_COMPANY.PROFIT_MARGIN is decimal (0.27), FCT_FUNDAMENTALS_GROWTH is pct (29.28)
        if net_margin_val is not None:
            try:
                v = float(net_margin_val)
                if abs(v) < 1:
                    net_margin_val = v * 100
            except (ValueError, TypeError):
                net_margin_val = None

    # D/E: prefer SEC-derived (financial_health chart), fall back to analysis facts
    de_val = fin_data.get("debt_to_equity_ratio")
    if de_val is None:
        co_facts = analysis.get("company_overview", {}).get("key_facts", {})
        de_val = co_facts.get("debt_to_equity")
    # Clamp obviously wrong D/E values (Yahoo Finance unscaled percentages)
    if de_val is not None:
        try:
            if float(de_val) > 50:
                de_val = None
        except (ValueError, TypeError):
            pass

    metrics = [
        [
            metric_cell("Current Price",
                _fmt_val(price_data.get('current_price'), prefix="$")),
            metric_cell("Trend Signal",
                price_data.get("trend_signal") or "N/A"),
            metric_cell("30D Volatility",
                _fmt_val(vol_data.get('volatility_30d_pct'), suffix="%")),
        ],
        [
            metric_cell("Revenue Growth (YoY)",
                _fmt_val(fund_data.get('latest_revenue_growth_yoy'), suffix="%")),
            metric_cell("Latest EPS",
                _fmt_val(eps_data.get('latest_eps'), prefix="$")),
            metric_cell("EPS Growth (YoY)",
                _fmt_val(eps_data.get('eps_growth_yoy_pct'), suffix="%")),
        ],
        [
            metric_cell("Net Margin",
                _fmt_val(net_margin_val, suffix="%")),
            metric_cell("Debt/Equity",
                _fmt_val(de_val)),
            metric_cell("News Sentiment",
                sent_data.get("sentiment_label") or "N/A"),
        ],
    ]

    col_w = (PAGE_W - 2 * MARGIN) / 3

    for row in metrics:
        t = Table(
            [[cell for cell in row]],
            colWidths=[col_w, col_w, col_w]
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
            ("BOX",        (0, 0), (-1, -1), 0.5, C_DIVIDER),
            ("INNERGRID",  (0, 0), (-1, -1), 0.5, C_DIVIDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 8 * mm))

    # Investment thesis — brief summary only (full text in Recommendation section)
    thesis = clean_llm_text(analysis.get("investment_thesis", ""))
    if thesis and "not available" not in thesis.lower():
        elements.append(Paragraph(
            "Investment Thesis (Summary)", styles["appendix_header"]
        ))
        elements.append(HRFlowable(
            width="100%", thickness=0.5,
            color=C_DIVIDER, spaceAfter=6
        ))
        # Build a brief that preserves opening + first 2-3 numbered agreement points.
        import re as _re
        flat = _re.sub(r"\s+", " ", thesis).strip()
        # Pull numbered bullets like "1. ... 2. ... 3. ..."
        bullets = _re.findall(r"\b([1-9])\.\s+([^1-9][^.?!]*[.?!])", flat)
        opening = flat.split(":")[0].strip()
        if not opening.endswith((".", "?", "!")):
            opening += "."
        if bullets:
            kept = " ".join(f"{n}. {txt.strip()}" for n, txt in bullets[:3])
            brief = f"{opening}: {kept}"
        else:
            # Fallback: first 3 sentences or first 600 chars
            sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
            brief = " ".join(sentences[:3]) if sentences else flat[:600]
        if len(brief) > 900:
            brief = brief[:900].rsplit(" ", 1)[0] + "…"
        if not brief.endswith("."):
            brief += "."
        brief += " (See Section 7 for full analysis.)"
        elements.append(Paragraph(brief, styles["body"]))
        elements.append(Spacer(1, 6 * mm))

    # MD&A summary paragraph
    elements.append(Paragraph(
        "Management Commentary Summary", styles["appendix_header"]
    ))
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=C_DIVIDER, spaceAfter=6
    ))
    mda = clean_llm_text(analysis.get("mda_summary", "Not available."))
    elements.append(Paragraph(mda, styles["body"]))
    elements.append(Spacer(1, 6 * mm))

    # Market context
    elements.append(Paragraph(
        "Market Context &amp; Data Coverage", styles["appendix_header"]
    ))
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=C_DIVIDER, spaceAfter=6
    ))

    # Summary signal table
    sig_rows = [["Category", "Signal", "Status"]]
    for c in charts:
        label, color = get_signal(c["chart_id"], c.get("data_summary", {}))
        c_hex = "#%02x%02x%02x" % (int(color.red * 255), int(color.green * 255), int(color.blue * 255)) if hasattr(color, "red") else "#94a3b8"
        sig_rows.append([
            c.get("title", c["chart_id"]),
            Paragraph(f'<font color="{c_hex}"><b>{label}</b></font>',
                      ParagraphStyle("es", fontName="Helvetica-Bold", fontSize=9)),
            "Validated" if c.get("validated") else "Review",
        ])

    content_width = PAGE_W - 2 * MARGIN
    sig_table = Table(sig_rows, colWidths=[content_width * 0.45, content_width * 0.30, content_width * 0.25])
    sig_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(sig_table)

    elements.append(PageBreak())
    return elements


def build_chart_section(chart: dict, analysis_text: str,
                         styles: dict, chart_number: int = 0,
                         detail_level: str = "detailed") -> list:
    """Build one chart section on a dedicated page: teal header bar + chart image + analysis + key metrics."""
    elements = []

    chart_id = chart["chart_id"]
    title = chart.get("title", chart_id)
    data_summary = chart.get("data_summary", {})
    sig_label, sig_color = get_signal(chart_id, data_summary)

    # ── Section header bar ───────────────────────────────────
    sig_hex = "#%02x%02x%02x" % (int(sig_color.red * 255), int(sig_color.green * 255), int(sig_color.blue * 255)) if hasattr(sig_color, "red") else "#94a3b8"

    section_title = f"3.{chart_number} &nbsp; {title}" if chart_number else title

    # Pill-shaped signal badge (inner colored cell with rounded corners)
    badge_inner = Table(
        [[Paragraph(
            f'<font color="#ffffff"><b>{sig_label}</b></font>',
            ParagraphStyle("badge", fontName="Helvetica-Bold",
                           fontSize=9, textColor=C_WHITE, alignment=TA_CENTER)
        )]],
        colWidths=[(PAGE_W - 2 * MARGIN) * 0.22],
    )
    badge_inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(sig_hex)),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    header_data = [[
        Paragraph(section_title, styles["section_header"]),
        badge_inner,
    ]]

    content_width = PAGE_W - 2 * MARGIN
    header_table = Table(
        header_data,
        colWidths=[content_width * 0.75, content_width * 0.25]
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (0, 0),   10),
        ("RIGHTPADDING",  (1, 0), (1, 0),   10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (1, 0),   "RIGHT"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 4))

    # ── "Data unavailable" callout when all metrics are null ──
    _numeric_vals = [v for k, v in data_summary.items()
                     if k not in ("financial_health",) and v is not None]
    if data_summary and not _numeric_vals:
        callout = Table(
            [[Paragraph(
                '<font color="#92400e"><b>⚠ Underlying data unavailable</b> — '
                'The structured data for this chart could not be retrieved from '
                'the data warehouse. The chart image and LLM analysis below may '
                'not reflect current figures.</font>',
                ParagraphStyle("callout", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#92400e"),
                               leading=11)
            )]],
            colWidths=[content_width],
        )
        callout.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(callout)
        elements.append(Spacer(1, 4))

    # ── Chart image ──────────────────────────────────────────
    file_path = chart.get("file_path", "")
    if file_path and os.path.exists(file_path):
        img_width = content_width
        img_height = img_width * (6 / 14)  # preserve 14:6 aspect ratio
        img = Image(file_path, width=img_width, height=img_height)
        elements.append(img)
    else:
        elements.append(Paragraph(
            f"[Chart image not available: {file_path}]",
            styles["body"]
        ))

    elements.append(Spacer(1, 5))

    # ── Teal accent line ─────────────────────────────────────
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=C_TEAL, spaceAfter=6
    ))

    # ── Key metrics from data_summary (inline table) ─────────
    if data_summary and detail_level != "summary":
        metric_items = []
        for k, v in data_summary.items():
            label = k.replace("_", " ").title()
            if v is None:
                val_str = "N/A"
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and k in (
                "total_revenue", "net_income", "total_assets", "total_equity",
                "operating_income", "revenue", "market_cap"
            ):
                val_str = _fmt_money(v)
            elif isinstance(v, float):
                val_str = f"{v:,.2f}"
            else:
                val_str = str(v)
            metric_items.append([
                Paragraph(f"<b>{label}</b>", ParagraphStyle("mk", fontName="Helvetica-Bold", fontSize=8, textColor=C_GRAY)),
                Paragraph(val_str, ParagraphStyle("mv", fontName="Helvetica", fontSize=9, textColor=C_BLACK)),
            ])

        # Arrange in 2-column layout
        if len(metric_items) > 1:
            rows = []
            for i in range(0, len(metric_items), 2):
                row = metric_items[i]
                if i + 1 < len(metric_items):
                    row = row + metric_items[i + 1]
                else:
                    row = row + [Paragraph("", styles["body"]), Paragraph("", styles["body"])]
                rows.append(row)

            half = content_width * 0.5
            kv_table = Table(rows, colWidths=[half * 0.45, half * 0.55, half * 0.45, half * 0.55])
            kv_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
                ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(kv_table)
            elements.append(Spacer(1, 4))

    # ── Analysis paragraph ───────────────────────────────────
    elements.append(Paragraph(clean_llm_text(analysis_text), styles["body"]))

    # Page break after each chart section
    elements.append(PageBreak())

    return elements


def build_company_overview(ticker: str, analysis: dict, styles: dict,
                           charts: list = None) -> list:
    """Build Company Overview page with AI description + key facts table."""
    elements = []
    elements.append(Paragraph("2. Company Overview", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    overview = analysis.get("company_overview", {})
    content_width = PAGE_W - 2 * MARGIN

    # AI-generated company description
    description = overview.get("company_description", f"Company overview not available for {ticker}.")
    elements.append(Paragraph(description, styles["body"]))
    elements.append(Spacer(1, 6 * mm))

    # Key Facts table
    elements.append(Paragraph("Key Facts", styles["appendix_header"]))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))

    facts = overview.get("key_facts", {})

    # Use financial_health chart data as authoritative source for margin/D/E
    # to stay consistent with the Executive Summary on page 3.
    _fin_chart = {}
    for c in (charts or []):
        if c["chart_id"] == "financial_health":
            _fin_chart = c.get("data_summary", {})
            break

    # Backfill facts from chart data_summaries when analysis agent returned empty.
    # Uses direct assignment instead of setdefault to overwrite None values.
    # Chart keys: financial_health → total_revenue, net_margin_pct, debt_to_equity_ratio
    #             eps_trend → latest_eps, eps_growth_yoy_pct
    #             revenue_growth → latest_revenue_growth_yoy, latest_net_income_growth_yoy
    if charts:
        _chart_map = {c["chart_id"]: c.get("data_summary", {}) for c in charts}
        fin_health = _chart_map.get("financial_health", {})
        if not facts.get("revenue"):
            _rev_val = fin_health.get("total_revenue")
            if _rev_val is not None:
                facts["revenue"] = _rev_val
        if not facts.get("eps"):
            eps_summary = _chart_map.get("eps_trend", {})
            _eps_val = eps_summary.get("latest_eps")
            if _eps_val is not None:
                facts["eps"] = _eps_val
        if not facts.get("net_income"):
            # Derive net_income from revenue × net_margin if both available
            _rev = fin_health.get("total_revenue")
            _margin = fin_health.get("net_margin_pct")
            if _rev and _margin:
                facts["net_income"] = round(_rev * _margin / 100)
        if not facts.get("latest_quarter"):
            # Backfill from eps_trend or revenue_growth chart data
            for _cid in ("eps_trend", "revenue_growth"):
                _ds = _chart_map.get(_cid, {})
                _lq = _ds.get("latest_quarter") or _ds.get("fiscal_quarter")
                if _lq:
                    facts["latest_quarter"] = _lq
                    break

    def _fmt_mc(v):
        if not v or not isinstance(v, (int, float)):
            return "N/A"
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        return f"${v/1e6:.0f}M"

    def _fmt_pct(v):
        if v is None:
            return "N/A"
        v = float(v)
        # DIM_COMPANY.PROFIT_MARGIN is decimal (0.27), everything else is pct (29.28)
        if abs(v) < 1:
            return f"{v*100:.1f}%"
        return f"{v:.1f}%"

    _de_val = _fin_chart.get("debt_to_equity_ratio") or facts.get("debt_to_equity")
    # Clamp obviously wrong D/E values (Yahoo Finance unscaled percentages)
    if _de_val is not None:
        try:
            if float(_de_val) > 50:
                _de_val = None
        except (ValueError, TypeError):
            _de_val = None

    facts_rows = [
        ["Metric", "Value"],
        ["Market Capitalization", _fmt_mc(facts.get("market_cap"))],
        ["Market Cap Category", str(facts.get("market_cap_category", "N/A"))],
        ["P/E Ratio", f"{facts['pe_ratio']:.1f}" if facts.get("pe_ratio") else "N/A"],
        ["Net Margin", _fmt_pct(_fin_chart.get("net_margin_pct") or facts.get("net_margin") or facts.get("profit_margin"))],
        ["Debt-to-Equity", f"{float(_de_val):.2f}" if _de_val else "N/A"],
        ["Latest Quarter", str(facts.get("latest_quarter", "N/A"))],
        ["Revenue (Latest Q)", _fmt_mc(facts.get("revenue"))],
        ["Net Income (Latest Q)", _fmt_mc(facts.get("net_income"))],
        ["EPS (Latest Q)", f"${facts['eps']:.2f}" if facts.get("eps") else "N/A"],
        ["SEC CIK", str(facts.get("cik", "N/A"))],
    ]

    facts_table = Table(facts_rows, colWidths=[content_width * 0.45, content_width * 0.55])
    facts_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(facts_table)
    elements.append(Spacer(1, 6 * mm))

    # Business segments (from Bedrock KB, if available)
    segments = overview.get("business_segments", "")
    if segments:
        elements.append(Paragraph("Business Segments &amp; Revenue Drivers", styles["appendix_header"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))
        elements.append(Paragraph(segments, styles["body"]))

    elements.append(PageBreak())
    return elements


def build_peer_comparison(ticker: str, analysis: dict, styles: dict) -> list:
    """Build Peer Comparison page with metrics table + AI comparison summary."""
    elements = []
    elements.append(Paragraph("5. Peer Comparison", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    peer_data = analysis.get("peer_comparison", {})
    peers = peer_data.get("peers", [])
    content_width = PAGE_W - 2 * MARGIN

    if not peers:
        elements.append(Paragraph(
            f"Peer comparison data not available for {ticker}.",
            styles["body"]
        ))
        elements.append(PageBreak())
        return elements

    # Comparison table
    def _fmt_mc(v):
        if not v or not isinstance(v, (int, float)):
            return "N/A"
        if v >= 1e12:
            return f"${v/1e12:.1f}T"
        if v >= 1e9:
            return f"${v/1e9:.0f}B"
        return f"${v/1e6:.0f}M"

    def _fmt_pct(v):
        if v is None:
            return "N/A"
        v = float(v)
        # DIM_COMPANY.PROFIT_MARGIN is decimal (0.27), everything else is pct (29.28)
        if abs(v) < 1:
            return f"{v*100:.1f}%"
        return f"{v:.1f}%"

    header = ["Ticker", "Market Cap", "P/E", "Net Margin", "D/E", "EPS"]
    table_rows = [header]
    for p in peers:
        is_target = p["ticker"] == ticker.upper()
        pe = f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") else "N/A"
        margin = p.get("net_margin") or p.get("profit_margin")
        eps = f"${p['eps']:.2f}" if p.get("eps") else "N/A"
        de_raw = p.get("debt_to_equity")
        # Clamp obviously wrong D/E values (Yahoo Finance unscaled percentages)
        if de_raw is not None:
            try:
                de_raw = float(de_raw)
                if de_raw > 50:
                    de_raw = None
            except (ValueError, TypeError):
                de_raw = None
        de = f"{de_raw:.2f}" if de_raw is not None else "N/A"
        row = [
            f"{p['ticker']} *" if is_target else p["ticker"],
            _fmt_mc(p.get("market_cap")),
            pe,
            _fmt_pct(margin),
            de,
            eps,
        ]
        table_rows.append(row)

    col_w = content_width / 6
    peer_table = Table(table_rows, colWidths=[col_w] * 6)

    # Highlight target ticker row
    target_row_idx = None
    for i, p in enumerate(peers):
        if p["ticker"] == ticker.upper():
            target_row_idx = i + 1  # +1 for header
            break

    table_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    if target_row_idx is not None:
        table_style_cmds.append(("BACKGROUND", (0, target_row_idx), (-1, target_row_idx), C_TEAL_LIGHT))
        table_style_cmds.append(("FONTNAME", (0, target_row_idx), (-1, target_row_idx), "Helvetica-Bold"))

    peer_table.setStyle(TableStyle(table_style_cmds))
    elements.append(peer_table)
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"* {ticker.upper()} is the subject of this report",
        ParagraphStyle("note", fontName="Helvetica", fontSize=7.5, textColor=C_GRAY)
    ))
    excluded = peer_data.get("excluded_peers") or []
    if excluded:
        elements.append(Paragraph(
            f"Note: {', '.join(excluded)} excluded due to unavailable peer data.",
            ParagraphStyle("note2", fontName="Helvetica-Oblique", fontSize=7.5, textColor=C_GRAY)
        ))
    elements.append(Spacer(1, 8 * mm))

    # AI comparison summary
    summary = peer_data.get("comparison_summary", "")
    if summary and "not available" not in summary.lower():
        elements.append(Paragraph("Comparative Analysis", styles["appendix_header"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))
        elements.append(Paragraph(summary, styles["body"]))

    elements.append(PageBreak())
    return elements


def build_risk_section(risk_summary: str, charts: list, styles: dict, analysis: dict = None) -> list:
    """Build expanded risk factors page with categorized risks."""
    elements = []
    elements.append(Paragraph("6. Risk Factors", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_BEARISH, spaceAfter=10
    ))

    content_width = PAGE_W - 2 * MARGIN

    # ── SEC-derived risk summary ─────────────────────────────
    risk_header = Table(
        [[Paragraph("SEC Filing Risk Factors (10-K / 10-Q)", styles["section_header"])]],
        colWidths=[content_width]
    )
    risk_header.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(risk_header)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(clean_llm_text(risk_summary), styles["risk_body"]))
    elements.append(Spacer(1, 6 * mm))

    # ── Data-driven risk indicators ──────────────────────────
    risk_cat_header = Table(
        [[Paragraph("Data-Driven Risk Indicators", styles["section_header"])]],
        colWidths=[content_width]
    )
    risk_cat_header.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(risk_cat_header)
    elements.append(Spacer(1, 8))

    chart_map = {c["chart_id"]: c for c in charts}
    risk_items = []

    # Volatility risk
    vol_data = chart_map.get("volatility", {}).get("data_summary", {})
    vol_pct = vol_data.get("volatility_30d_pct", 0)
    vol_level = "HIGH" if vol_pct > 3.0 else "MODERATE" if vol_pct > 1.5 else "LOW"
    vol_color = C_BEARISH if vol_level == "HIGH" else C_NEUTRAL if vol_level == "MODERATE" else C_BULLISH
    risk_items.append(("Market Volatility", vol_level, f"30-day volatility at {vol_pct:.1f}%. "
                       "Elevated volatility increases short-term downside risk and may indicate "
                       "market uncertainty about future earnings or macro headwinds."))

    # Sentiment risk
    sent_data = chart_map.get("sentiment", {}).get("data_summary", {})
    sent_label = sent_data.get("sentiment_label", "NEUTRAL")
    sent_level = "HIGH" if "BEAR" in sent_label else "MODERATE" if "NEUTRAL" in sent_label else "LOW"
    risk_items.append(("Sentiment / Headline Risk", sent_level,
                       f"Current sentiment: {sent_label}. "
                       "Negative media coverage or analyst downgrades could pressure the stock "
                       "beyond what fundamentals justify."))

    # Financial health risk
    fin_data = chart_map.get("financial_health", {}).get("data_summary", {})
    dte = fin_data.get("debt_to_equity_ratio")
    margin = fin_data.get("net_margin_pct")
    # Fall back to analysis company_overview if chart data is missing
    if margin is None and analysis:
        co_facts = analysis.get("company_overview", {}).get("key_facts", {})
        margin = co_facts.get("net_margin") or co_facts.get("profit_margin")
        # Normalize decimal (0.27) to percentage (27.0)
        if margin is not None:
            try:
                margin = float(margin)
                if abs(margin) < 1:
                    margin = margin * 100
            except (ValueError, TypeError):
                margin = None
    # Clamp obviously wrong D/E values (Yahoo Finance unscaled percentages)
    if dte is not None:
        try:
            if float(dte) > 50:
                dte = None
        except (ValueError, TypeError):
            dte = None
    if margin is None or dte is None:
        fin_level = "MODERATE"  # Insufficient data — don't flag as HIGH
        dte_str = f"{dte:.2f}" if dte is not None else "N/A"
        margin_str = f"{margin:.1f}%" if margin is not None else "N/A"
    else:
        fin_level = "HIGH" if (dte > 2.0 or margin < 0) else "MODERATE" if dte > 1.0 else "LOW"
        dte_str = f"{dte:.2f}"
        margin_str = f"{margin:.1f}%"
    risk_items.append(("Balance Sheet / Leverage Risk", fin_level,
                       f"Debt-to-equity ratio: {dte_str}, net margin: {margin_str}. "
                       "Companies with elevated leverage face higher interest expense "
                       "and refinancing risk in rising-rate environments."))

    # Growth risk
    fund_data = chart_map.get("revenue_growth", {}).get("data_summary", {})
    rev_growth = fund_data.get("latest_revenue_growth_yoy")
    if rev_growth is None:
        growth_level = "MODERATE"  # Insufficient data
        growth_str = "N/A"
    else:
        growth_level = "HIGH" if rev_growth < -5 else "MODERATE" if rev_growth < 5 else "LOW"
        growth_str = f"{rev_growth:.1f}%"
    risk_items.append(("Revenue Growth Risk", growth_level,
                       f"Latest YoY revenue growth: {growth_str}. "
                       "Slowing or negative growth may signal market share loss, "
                       "demand softening, or sector-wide headwinds."))

    risk_label_style = ParagraphStyle("risk_label", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK)
    risk_level_style = ParagraphStyle("risk_level", fontName="Helvetica-Bold", fontSize=9, alignment=TA_CENTER)
    risk_desc_style = ParagraphStyle("risk_desc", fontName="Helvetica", fontSize=9, textColor=C_GRAY, leading=14)

    for name, level, desc in risk_items:
        level_color = {"HIGH": "#ef476f", "MODERATE": "#ffaa00", "LOW": "#06d6a0"}.get(level, "#94a3b8")
        row_data = [[
            Paragraph(name, risk_label_style),
            Paragraph(f'<font color="{level_color}"><b>{level}</b></font>', risk_level_style),
        ]]
        row_table = Table(row_data, colWidths=[content_width * 0.75, content_width * 0.25])
        row_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
            ("BOX", (0, 0), (-1, -1), 0.5, C_DIVIDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(row_table)
        elements.append(Paragraph(desc, risk_desc_style))
        elements.append(Spacer(1, 4))

    return elements


def build_financial_metrics_page(charts: list, styles: dict) -> list:
    """Build a detailed financial metrics summary page."""
    elements = []
    # No PageBreak needed — previous chart section already ends with one
    elements.append(Paragraph("4. Financial Metrics Summary", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    content_width = PAGE_W - 2 * MARGIN
    chart_map = {c["chart_id"]: c for c in charts}

    # ── Price & Trading Metrics ──────────────────────────────
    elements.append(Paragraph("Price &amp; Trading Metrics", styles["appendix_header"]))
    price_data = chart_map.get("price_sma", {}).get("data_summary", {})
    vol_data = chart_map.get("volatility", {}).get("data_summary", {})

    price_rows = [
        ["Metric", "Value"],
        ["Current Close Price", f"${price_data.get('current_price', 'N/A')}"],
        ["7-Day SMA", f"${price_data.get('sma_7d', 'N/A')}"],
        ["30-Day SMA", f"${price_data.get('sma_30d', 'N/A')}"],
        ["90-Day SMA", f"${price_data.get('sma_90d', 'N/A')}"],
        ["Trend Signal", str(price_data.get('trend_signal', 'N/A'))],
        ["30-Day Volatility", f"{vol_data.get('volatility_30d_pct', 'N/A')}%"],
        ["Avg Daily Volume", f"{vol_data.get('avg_volume', 'N/A'):,}" if isinstance(vol_data.get('avg_volume'), (int, float)) else "N/A"],
        ["Date Range", str(price_data.get('date_range', 'N/A'))],
    ]
    price_table = Table(price_rows, colWidths=[content_width * 0.55, content_width * 0.45])
    price_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(price_table)
    elements.append(Spacer(1, 6 * mm))

    # ── Fundamental Metrics ──────────────────────────────────
    elements.append(Paragraph("Fundamental Metrics", styles["appendix_header"]))
    fund_data = chart_map.get("revenue_growth", {}).get("data_summary", {})
    eps_data = chart_map.get("eps_trend", {}).get("data_summary", {})
    fin_data = chart_map.get("financial_health", {}).get("data_summary", {})

    def fmt_val(v, suffix="", prefix=""):
        if v is None or v == "N/A":
            return "N/A"
        if isinstance(v, float):
            return f"{prefix}{v:,.2f}{suffix}"
        return f"{prefix}{v}{suffix}"

    fund_rows = [
        ["Metric", "Value"],
        ["Revenue Growth (YoY)", fmt_val(fund_data.get('latest_revenue_growth_yoy'), "%")],
        ["Net Income Growth (YoY)", fmt_val(fund_data.get('latest_net_income_growth_yoy'), "%")],
        ["Fundamental Signal", str(fund_data.get('fundamental_signal', 'N/A'))],
        ["Latest EPS", fmt_val(eps_data.get('latest_eps'), prefix="$")],
        ["EPS Growth (YoY)", fmt_val(eps_data.get('eps_growth_yoy_pct'), "%")],
        ["EPS Growth (QoQ)", fmt_val(eps_data.get('eps_growth_qoq_pct'), "%")],
        ["Net Margin", fmt_val(fin_data.get('net_margin_pct'), "%")],
        ["Operating Margin", fmt_val(fin_data.get('operating_margin_pct'), "%") if fin_data.get('operating_margin_pct') is not None else "Not reported"],
        ["Debt/Equity Ratio", fmt_val(fin_data.get('debt_to_equity_ratio'))],
        ["Financial Health", str(fin_data.get('financial_health', 'N/A'))],
    ]
    fund_table = Table(fund_rows, colWidths=[content_width * 0.55, content_width * 0.45])
    fund_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(fund_table)
    elements.append(Spacer(1, 6 * mm))

    # ── Sentiment Metrics ────────────────────────────────────
    elements.append(Paragraph("Sentiment &amp; News Metrics", styles["appendix_header"]))
    sent_data = chart_map.get("sentiment", {}).get("data_summary", {})
    sent_rows = [
        ["Metric", "Value"],
        ["7-Day Avg Sentiment", fmt_val(sent_data.get('sentiment_score_7d_avg'))],
        ["Sentiment Label", str(sent_data.get('sentiment_label', 'N/A'))],
        ["Sentiment Trend", str(sent_data.get('sentiment_trend', 'N/A'))],
        ["Articles (30D)", str(sent_data.get('total_articles_30d', 'N/A'))],
    ]
    sent_table = Table(sent_rows, colWidths=[content_width * 0.55, content_width * 0.45])
    sent_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(sent_table)

    return elements


def build_recommendation_page(ticker: str, charts: list, analysis: dict,
                               styles: dict) -> list:
    """Build investment recommendation page with overall signal and thesis."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("7. Investment Recommendation", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    content_width = PAGE_W - 2 * MARGIN

    # Overall signal
    sig_label, sig_color = overall_signal(charts)
    sig_hex = "#%02x%02x%02x" % (int(sig_color.red * 255), int(sig_color.green * 255), int(sig_color.blue * 255)) if hasattr(sig_color, "red") else "#94a3b8"

    # Big signal badge
    sig_bg = colors.HexColor("#065f46") if "BULLISH" in sig_label else \
             colors.HexColor("#4c0519") if "BEARISH" in sig_label else \
             colors.HexColor("#1e293b")

    signal_box = Table(
        [[Paragraph(
            f'<font color="{sig_hex}" size="20"><b>{sig_label}</b></font>',
            ParagraphStyle("big_sig", fontName="Helvetica-Bold", fontSize=20,
                           alignment=TA_CENTER)
        )]],
        colWidths=[content_width]
    )
    signal_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sig_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(signal_box)
    elements.append(Spacer(1, 8 * mm))

    # Signal breakdown table
    elements.append(Paragraph("Signal Breakdown by Category", styles["appendix_header"]))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))

    breakdown_rows = [["Category", "Signal", "Key Metric"]]
    for c in charts:
        label, color = get_signal(c["chart_id"], c.get("data_summary", {}))
        color_hex = "#%02x%02x%02x" % (int(color.red * 255), int(color.green * 255), int(color.blue * 255)) if hasattr(color, "red") else "#94a3b8"
        ds = c.get("data_summary", {})
        # Pick a key metric for each chart
        key_metric = ""
        if c["chart_id"] == "price_sma":
            key_metric = f"Price: ${ds.get('current_price', 'N/A')}"
        elif c["chart_id"] == "volatility":
            key_metric = f"Vol: {ds.get('volatility_30d_pct', 'N/A')}%"
        elif c["chart_id"] == "revenue_growth":
            key_metric = f"Rev Growth: {ds.get('latest_revenue_growth_yoy', 'N/A')}%"
        elif c["chart_id"] == "eps_trend":
            key_metric = f"EPS: ${ds.get('latest_eps', 'N/A')}"
        elif c["chart_id"] == "sentiment":
            key_metric = f"Sentiment: {ds.get('sentiment_label', 'N/A')}"
        elif c["chart_id"] == "financial_health":
            key_metric = f"D/E: {ds.get('debt_to_equity_ratio', 'N/A')}"

        breakdown_rows.append([
            c.get("title", c["chart_id"]),
            Paragraph(f'<font color="{color_hex}"><b>{label}</b></font>',
                      ParagraphStyle("bl", fontName="Helvetica-Bold", fontSize=9)),
            key_metric,
        ])

    bd_table = Table(breakdown_rows, colWidths=[content_width * 0.40, content_width * 0.25, content_width * 0.35])
    bd_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(bd_table)
    elements.append(Spacer(1, 8 * mm))

    # Investment thesis
    thesis = clean_llm_text(analysis.get("investment_thesis", ""))
    if thesis and "not available" not in thesis.lower():
        elements.append(Paragraph("Investment Thesis", styles["appendix_header"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))
        elements.append(Paragraph(thesis, styles["body"]))
    else:
        elements.append(Paragraph("Summary Assessment", styles["appendix_header"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))
        bullish_count = sum(1 for c in charts if "BULLISH" in get_signal(c["chart_id"], c.get("data_summary", {}))[0] or "GROWTH" in get_signal(c["chart_id"], c.get("data_summary", {}))[0] or "HEALTHY" in get_signal(c["chart_id"], c.get("data_summary", {}))[0] or "LOW VOL" in get_signal(c["chart_id"], c.get("data_summary", {}))[0])
        total = len(charts)
        elements.append(Paragraph(
            f"Based on analysis of {total} key indicators, {bullish_count} show positive signals. "
            f"This multi-factor assessment considers price momentum, earnings trajectory, "
            f"revenue growth, market sentiment, volatility profile, and balance sheet health. "
            f"Investors should consider these signals alongside their own risk tolerance, "
            f"time horizon, and portfolio diversification requirements.",
            styles["body"]
        ))

    # Disclaimer footer
    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_DIVIDER, spaceAfter=6))
    elements.append(Paragraph(
        "This recommendation is generated by AI and does not constitute financial advice. "
        "Past performance is not indicative of future results. Always consult a qualified "
        "financial advisor before making investment decisions.",
        styles["disclaimer"]
    ))

    return elements


_SECTION_NAMES = {"mda_summary": "MD&A Summary"}


def _section_name(label: str) -> str:
    return _SECTION_NAMES.get(label, label.replace("_", " ").title())


def build_guardrail_disclosure(guardrail_audit: Optional[dict], styles: dict,
                               content_width: float) -> list:
    """
    Appendix block stating what the content-safety guardrails actually did.

    Guardrails fail open (the text is kept if the check cannot run), so the
    reader needs to see which sections went out unvalidated.
    """
    elements = [Paragraph("Guardrail Checks", styles["appendix_header"])]

    if guardrail_audit is None:
        elements.append(Paragraph(
            "Guardrail status was not recorded for this report.",
            styles["appendix_body"],
        ))
        elements.append(Spacer(1, 8 * mm))
        return elements

    skipped = guardrail_audit.get("skipped_sections", [])
    blocked = guardrail_audit.get("blocked_sections", [])
    rows = [
        ["Outcome", "Checks", "Sections"],
        ["Passed", str(guardrail_audit.get("passed", 0)), "—"],
        ["Blocked (replaced with notice)", str(guardrail_audit.get("blocked", 0)),
         ", ".join(_section_name(s) for s in blocked) or "—"],
        ["Skipped (not validated)", str(guardrail_audit.get("skipped", 0)),
         ", ".join(_section_name(s) for s in skipped) or "—"],
    ]
    table = Table(
        rows,
        colWidths=[content_width * 0.32, content_width * 0.12, content_width * 0.56]
    )
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 0), (1, -1), "CENTER"),
    ]))
    elements.append(table)

    if skipped:
        elements.append(Spacer(1, 3 * mm))
        elements.append(Paragraph(
            "Sections marked skipped were included without content-safety "
            "validation because the guardrail service was unavailable or "
            "returned an error. Review them before relying on this report.",
            styles["appendix_body"],
        ))
    elements.append(Spacer(1, 8 * mm))
    return elements


def build_appendix(ticker: str, charts: list, styles: dict,
                   guardrail_audit: Optional[dict] = None) -> list:
    """Build appendix + disclaimer page."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("8. Appendix &amp; Data Sources", styles["page_title"]))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=C_TEAL, spaceAfter=10
    ))

    # Data sources table
    elements.append(Paragraph("Data Sources", styles["appendix_header"]))
    sources = [
        ["Source", "Data Type", "Layer"],
        ["Yahoo Finance", "OHLCV Price Data, Fundamentals", "RAW → STAGING → ANALYTICS"],
        ["NewsAPI", "Financial News Articles", "RAW → STAGING → ANALYTICS"],
        ["SEC EDGAR (XBRL)", "Financial Statement Concepts", "RAW → STAGING → ANALYTICS"],
        ["SEC EDGAR (HTML)", "10-K/10-Q Filing Text", "RAW (S3)"],
        ["Snowflake Cortex", "LLM Analysis & Summarization", "CAVM Pipeline"],
    ]

    content_width = PAGE_W - 2 * MARGIN
    src_table = Table(
        sources,
        colWidths=[content_width * 0.25, content_width * 0.45, content_width * 0.30]
    )
    src_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BACKGROUND",    (0, 1), (-1, -1), C_LIGHT_GRAY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    elements.append(src_table)
    elements.append(Spacer(1, 8 * mm))

    # Chart generation metadata
    elements.append(Paragraph("Chart Generation Metadata", styles["appendix_header"]))
    chart_meta = [["Chart", "Method", "Refinement Iterations", "Validated"]]
    for c in charts:
        method = "LLM + VLM Refinement" if c.get("refinement_count", 0) > 0 else "Fallback"
        chart_meta.append([
            c.get("title", c["chart_id"]),
            method,
            str(c.get("refinement_count", 0)),
            "✓" if c.get("validated") else "✗",
        ])

    meta_table = Table(
        chart_meta,
        colWidths=[content_width * 0.35, content_width * 0.35,
                   content_width * 0.18, content_width * 0.12]
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8 * mm))

    elements += build_guardrail_disclosure(guardrail_audit, styles, content_width)

    # CAVM Pipeline Architecture
    elements.append(Paragraph("CAVM Pipeline Architecture", styles["appendix_header"]))
    pipeline_rows = [
        ["Stage", "Agent", "Technology", "Purpose"],
        ["1. Chart", "chart_agent", "Cortex LLM + VLM", "3-iteration chart generation with visual critique refinement loop"],
        ["2. Validation", "validation_agent", "Cortex VLM (claude-sonnet-4-6)", "Chart quality assurance — file integrity, dimensions, VLM scoring"],
        ["3. Analysis", "analysis_agent", "Cortex LLM (claude-opus-4-6)", "Per-chart financial analysis + SEC MD&A/Risk summarization"],
        ["4. Report", "report_agent", "reportlab", "Branded PDF assembly with executive summary and recommendations"],
    ]
    pipe_table = Table(
        pipeline_rows,
        colWidths=[content_width * 0.12, content_width * 0.18,
                   content_width * 0.22, content_width * 0.48]
    )
    pipe_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(pipe_table)
    elements.append(Spacer(1, 6 * mm))

    # Data architecture
    elements.append(Paragraph("Data Architecture", styles["appendix_header"]))
    arch_rows = [
        ["Layer", "Schema", "Tables", "Purpose"],
        ["RAW", "RAW", "4 tables", "Raw data ingestion from Yahoo Finance, NewsAPI, SEC EDGAR"],
        ["STAGING", "STAGING", "4 views", "Data cleaning, type casting, deduplication (dbt views)"],
        ["ANALYTICS", "ANALYTICS", "5 tables", "Fact/dimension tables for analysis (dbt tables)"],
    ]
    arch_table = Table(
        arch_rows,
        colWidths=[content_width * 0.12, content_width * 0.15,
                   content_width * 0.15, content_width * 0.58]
    )
    arch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(arch_table)
    elements.append(Spacer(1, 10 * mm))

    # Disclaimer — keep together to avoid orphaned lines on a new page
    disclaimer_elements = []
    disclaimer_elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=C_DIVIDER, spaceAfter=6
    ))
    disclaimer_elements.append(Paragraph(
        "DISCLAIMER",
        ParagraphStyle("disc_hdr", fontName="Helvetica-Bold",
                       fontSize=8, textColor=C_GRAY, spaceAfter=4)
    ))
    disclaimer_elements.append(Paragraph(
        f"This report was generated automatically by FinSage, an AI-powered financial research "
        f"system developed at Northeastern University (DAMG 7374). The information contained "
        f"herein is derived from publicly available data sources including Yahoo Finance, "
        f"NewsAPI, and SEC EDGAR. This report does not constitute financial advice, investment "
        f"recommendations, or an offer to buy or sell any security. All data is provided for "
        f"informational and educational purposes only. FinSage and its authors make no "
        f"representations or warranties regarding the accuracy, completeness, or timeliness "
        f"of the information contained in this report. Past performance is not indicative of "
        f"future results. Generated on {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}.",
        styles["disclaimer"]
    ))
    elements.append(KeepTogether(disclaimer_elements))

    return elements


# ──────────────────────────────────────────────────────────────
# Main PDF builder
# ──────────────────────────────────────────────────────────────

def build_pdf(
    ticker: str,
    charts: list,
    analysis: dict,
    output_path: str,
    company_name: str = None,
    detail_level: str = "detailed",
) -> str:
    """
    Assemble the full FinSage PDF report.

    Args:
        ticker:       Stock ticker symbol
        charts:       List of ChartResult dicts from chart_agent/validation_agent
        analysis:     Dict from analysis_agent:
                        {chart_analyses: [...], mda_summary: str, risk_summary: str}
        output_path:  Full path for output PDF file
        company_name: Optional display name (defaults to ticker)
        detail_level: "detailed" for full 15-20 page report,
                      "summary" for condensed 8-10 page report

    Returns:
        output_path (str)
    """
    company_name = company_name or ticker
    styles = build_styles()

    # Pre-assembly data-quality checks (logs warnings, may override metrics)
    corrections = _validate_report_data(charts, analysis, ticker)

    # Patch stale LLM-generated prose if net margin was overridden
    if corrections.get("net_margin_override"):
        old_val, new_val = corrections["net_margin_override"]
        n = _patch_analysis_text(analysis, old_val, new_val)
        if n:
            logger.info(
                "[%s] Patched %d stale net-margin references in analysis text "
                "(%.1f%% → %.1f%%)", ticker, n, old_val, new_val,
            )

    # Build analysis lookup (after patching so text is corrected)
    analysis_map = {
        a["chart_id"]: a["analysis_text"]
        for a in analysis.get("chart_analyses", [])
    }

    # Overall signal for cover
    sig = overall_signal(charts)

    # ── Doc setup ────────────────────────────────────────────
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
        title=f"FinSage — {ticker} Equity Research Report",
        author="FinSage AI",
        subject=f"{ticker} Financial Analysis",
    )
    doc._finsage_ticker = ticker  # used by header callback
    doc.report_date = datetime.now().strftime("%B %d, %Y")

    # ── Page templates ───────────────────────────────────────
    cover_frame = Frame(
        0, 0, PAGE_W, PAGE_H,
        leftPadding=MARGIN, rightPadding=MARGIN,
        topPadding=0, bottomPadding=0,
        id="cover"
    )
    content_frame = Frame(
        MARGIN, 14 * mm,
        PAGE_W - 2 * MARGIN, PAGE_H - 32 * mm,
        topPadding=4 * mm,
        bottomPadding=2 * mm,
        id="content"
    )

    doc.addPageTemplates([
        PageTemplate(id="Cover",   frames=[cover_frame],
                     onPage=draw_cover_bg),
        PageTemplate(id="Content", frames=[content_frame],
                     onPage=draw_content_page),
    ])

    # ── Assemble elements (two-pass for dynamic TOC) ───────────
    page_map = {}  # populated by SectionMarker flowables during first pass

    def _build_elements(pm):
        """Build the full element list. Called twice: first to discover page
        numbers, second to render with correct TOC."""
        elems = []

        # Page 1: Cover (with Buy/Hold/Sell rating)
        elems.append(NextPageTemplate("Cover"))
        cover_elems = build_cover(ticker, company_name, styles, sig, charts=charts)
        cover_elems.insert(-1, NextPageTemplate("Content"))
        elems += cover_elems

        # Page 2: Table of Contents
        elems += build_toc(charts, styles, page_map=pm,
                           detail_level=detail_level)

        # Pages 3-4: Executive Summary
        elems.append(SectionMarker("exec_summary", pm))
        elems += build_executive_summary(ticker, charts, analysis, styles)

        # Page 5: Company Overview
        elems.append(SectionMarker("company_overview", pm))
        elems += build_company_overview(ticker, analysis, styles, charts=charts)

        # Pages 6-11: Chart sections (one per page)
        elems.append(SectionMarker("analysis_sections", pm))
        elems.append(Paragraph("3. Analysis Sections", styles["page_title"]))
        elems.append(HRFlowable(
            width="100%", thickness=1.5,
            color=C_TEAL, spaceAfter=12
        ))

        for i, chart in enumerate(charts, 1):
            chart_id = chart["chart_id"]
            elems.append(SectionMarker(f"chart_{chart_id}", pm))
            analysis_text = analysis_map.get(
                chart_id,
                "Analysis not available for this chart."
            )
            elems += build_chart_section(chart, analysis_text, styles,
                                          chart_number=i,
                                          detail_level=detail_level)

        # Financial Metrics Summary (skip in summary mode)
        if detail_level != "summary":
            elems.append(SectionMarker("financial_metrics", pm))
            elems += build_financial_metrics_page(charts, styles)

        # Peer Comparison
        elems.append(SectionMarker("peer_comparison", pm))
        elems += build_peer_comparison(ticker, analysis, styles)

        # Risk Factors
        elems.append(SectionMarker("risk_factors", pm))
        elems += build_risk_section(
            analysis.get("risk_summary", "Risk factors not available."),
            charts,
            styles,
            analysis=analysis
        )

        # Investment Recommendation
        elems.append(SectionMarker("recommendation", pm))
        elems += build_recommendation_page(ticker, charts, analysis, styles)

        # Appendix (skip in summary mode)
        if detail_level != "summary":
            elems.append(SectionMarker("appendix", pm))
            elems += build_appendix(ticker, charts, styles,
                                    guardrail_audit=analysis.get("guardrail_audit"))

        return elems

    # Pass 1: render to /dev/null to collect page numbers
    import io
    pass1_buf = io.BytesIO()
    doc1 = BaseDocTemplate(
        pass1_buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=14 * mm,
    )
    doc1.addPageTemplates([
        PageTemplate(id="Cover",   frames=[cover_frame],
                     onPage=draw_cover_bg),
        PageTemplate(id="Content", frames=[content_frame],
                     onPage=draw_content_page),
    ])
    doc1.build(_build_elements(page_map))
    pass1_buf.close()
    logger.info("TOC pass 1 complete — page_map: %s", page_map)

    # Pass 2: render final PDF with correct TOC page numbers
    pass1_pages = dict(page_map)  # snapshot Pass 1 page numbers
    elements = _build_elements(page_map)

    # ── Build PDF ────────────────────────────────────────────
    logger.info("Building PDF: %s", output_path)
    doc.build(elements)

    # Validate: Pass 2 page numbers (now in page_map) should match Pass 1
    for key in pass1_pages:
        if pass1_pages[key] != page_map.get(key):
            logger.warning(
                "TOC drift: section '%s' was page %s in pass 1 but %s in pass 2",
                key, pass1_pages[key], page_map.get(key),
            )
    file_size = os.path.getsize(output_path)
    logger.info("✅ PDF complete: %s (%.1f KB)", output_path, file_size / 1024)

    return output_path


# ──────────────────────────────────────────────────────────────
# Main entry point (called by orchestrator)
# ──────────────────────────────────────────────────────────────

def generate_report(
    ticker: str,
    charts: list,
    analysis: dict,
    output_dir: str = None,
    company_name: str = None,
    detail_level: str = "detailed",
) -> str:
    """
    Generate the FinSage PDF report.

    Args:
        ticker:       Stock ticker symbol
        charts:       Validated ChartResult list from chart_agent
        analysis:     Output from analysis_agent.run_analysis()
        output_dir:   Directory to save PDF (defaults to outputs/)
        company_name: Display name for cover page
        detail_level: "detailed" for full report, "summary" for condensed

    Returns:
        Full path to generated PDF
    """
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_Summary" if detail_level == "summary" else ""
    filename = f"{ticker}_FinSage_Report{suffix}_{date_str}.pdf"
    output_path = os.path.join(output_dir, filename)

    logger.info("═" * 50)
    logger.info("Report Agent starting for %s (detail_level=%s)", ticker, detail_level)
    logger.info("Output: %s", output_path)
    logger.info("═" * 50)

    return build_pdf(
        ticker=ticker,
        charts=charts,
        analysis=analysis,
        output_path=output_path,
        company_name=company_name,
        detail_level=detail_level,
    )


# ──────────────────────────────────────────────────────────────
# Dev test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

    from snowflake_connection import get_session
    import json

    # Load real charts from most recent chart_agent run
    output_folders = sorted(
        glob.glob(str(OUTPUT_DIR / "AAPL_*")),
        reverse=True
    )

    charts_to_use = []
    latest = None
    for folder in output_folders:
        manifest_path = os.path.join(folder, "chart_manifest.json")
        if os.path.exists(manifest_path):
            latest = folder
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Re-attach file_paths (not stored in manifest)
            for entry in manifest:
                entry["file_path"] = os.path.join(folder, f"{entry['chart_id']}.png")
            charts_to_use = manifest
            break

    if not charts_to_use:
        print("❌ No chart output found — run chart_agent.py first")
        sys.exit(1)

    print(f"✅ Using real charts from: {latest}")
    print(f"   {len(charts_to_use)} charts loaded")

    # Run analysis agent with real Cortex
    session = get_session()

    from agents.analysis_agent import run_analysis
    analysis = run_analysis(session, charts_to_use, "AAPL")
    session.close()

    # Generate report
    pdf_path = generate_report(
        ticker="AAPL",
        charts=charts_to_use,
        analysis=analysis,
        company_name="Apple Inc.",
    )

    print(f"\n✅ Report generated: {pdf_path}")
    print(f"   Open with: open \"{pdf_path}\"")

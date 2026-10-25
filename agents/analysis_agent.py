"""
FinSage Analysis Agent

Uses Snowflake Cortex to generate written analysis for each validated chart
and summarize SEC filing text (MD&A + Risk Factors).

AWS Bedrock integrations (optional, graceful fallback to Cortex-only):
    - Guardrails: validates all LLM output for safety before PDF inclusion
    - Bedrock KB RAG: semantic search over SEC filings for targeted context
    - Multi-Model Consensus: multiple Bedrock models for investment thesis

Inputs:
    - validated chart list (from validation_agent / mock_data during dev)
    - Snowflake session
    - ticker symbol

Outputs:
    - List of analysis dicts: {chart_id, title, analysis_text}
    - SEC summary dict: {mda_summary, risk_summary}
    - Investment thesis (single-model or multi-model consensus)
"""

import os
import logging
import re
import threading
import contextvars
from typing import Optional
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Cortex model to use
# ──────────────────────────────────────────────────────────────
CORTEX_MODEL = os.getenv("CORTEX_MODEL_LLM", "claude-opus-4-6")

# Max characters of SEC text to pass into SUMMARIZE
MAX_SEC_TEXT_CHARS = 40_000

# Canonical analysis order for progressive Chain-of-Analysis.
ANALYSIS_ORDER = [
    "price_sma",        # Price trend — foundation
    "volatility",       # Risk/volume — builds on price
    "revenue_growth",   # Fundamentals — explains price moves
    "eps_trend",        # Earnings — confirms fundamentals
    "financial_health", # Balance sheet — deeper fundamental
    "margin_trend",     # Profitability trajectory — extends financial health
    "balance_sheet",    # Capital structure — completes fundamental picture
    "sentiment",        # Market perception — ties it all together
]

# Bedrock KB semantic search queries per chart type
KB_QUERIES = {
    "price_sma":        "stock price performance market outlook forward guidance",
    "volatility":       "market risk volatility trading volume investor activity",
    "revenue_growth":   "revenue growth drivers business segments sales performance",
    "eps_trend":        "earnings per share profitability income growth quality",
    "financial_health": "balance sheet strength debt equity ratio financial condition liquidity",
    "margin_trend":     "operating margin gross margin profitability cost structure efficiency",
    "balance_sheet":    "total assets liabilities stockholders equity capital structure solvency",
    "sentiment":        "competitive position market perception brand strategy outlook",
}


# ──────────────────────────────────────────────────────────────
# Lazy Bedrock client singletons (initialized on first use)
# Thread-safe via double-checked locking.
# ──────────────────────────────────────────────────────────────
_guardrail = None
_guardrail_checked = False
_guardrail_lock = threading.Lock()

_bedrock_kb = None
_bedrock_kb_checked = False
_bedrock_kb_lock = threading.Lock()

_multi_model = None
_multi_model_checked = False
_multi_model_lock = threading.Lock()


def _get_guardrail():
    """Lazy-init GuardedLLM. Returns None if BEDROCK_GUARDRAIL_ID not set."""
    global _guardrail, _guardrail_checked
    if _guardrail_checked:
        return _guardrail
    with _guardrail_lock:
        if _guardrail_checked:
            return _guardrail
        _guardrail_checked = True
        if not os.getenv("BEDROCK_GUARDRAIL_ID"):
            logger.info("BEDROCK_GUARDRAIL_ID not set — guardrails disabled")
            return None
        try:
            from sec_filings.guardrails import GuardedLLM
            _guardrail = GuardedLLM()
            logger.info("Guardrails initialized")
        except Exception as e:
            logger.warning("Could not initialize guardrails: %s", e)
    return _guardrail


def _get_bedrock_kb():
    """Lazy-init BedrockKB. Returns None if BEDROCK_KB_ID not set."""
    global _bedrock_kb, _bedrock_kb_checked
    if _bedrock_kb_checked:
        return _bedrock_kb
    with _bedrock_kb_lock:
        if _bedrock_kb_checked:
            return _bedrock_kb
        _bedrock_kb_checked = True
        if not os.getenv("BEDROCK_KB_ID"):
            logger.info("BEDROCK_KB_ID not set — Bedrock KB RAG disabled")
            return None
        try:
            from sec_filings.bedrock_kb import BedrockKB
            _bedrock_kb = BedrockKB()
            logger.info("Bedrock KB initialized")
        except Exception as e:
            logger.warning("Could not initialize Bedrock KB: %s", e)
    return _bedrock_kb


def _get_multi_model():
    """Lazy-init MultiModelAnalyzer. Returns None if AWS unavailable."""
    global _multi_model, _multi_model_checked
    if _multi_model_checked:
        return _multi_model
    with _multi_model_lock:
        if _multi_model_checked:
            return _multi_model
        _multi_model_checked = True
        try:
            from sec_filings.multi_model import MultiModelAnalyzer
            _multi_model = MultiModelAnalyzer()
            logger.info("MultiModelAnalyzer initialized")
        except Exception as e:
            logger.warning("Could not initialize MultiModelAnalyzer: %s", e)
    return _multi_model


# ──────────────────────────────────────────────────────────────
# Guardrails validation helper (Integration 1)
# ──────────────────────────────────────────────────────────────
GUARDRAIL_FALLBACK = (
    "This analysis section has been withheld because it did not pass "
    "content safety validation. Please refer to the underlying data "
    "for your own assessment."
)


class GuardrailAudit:
    """
    Per-report record of every guardrail check: passed, blocked, or skipped.

    Guardrails fail open by design (a report should not die at minute 12
    because Bedrock timed out), so a skipped check must be visible in the
    report itself, not just in the logs. The report appendix renders this.
    Thread-safe: Stage 3b/3c record into the same audit from worker threads.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.passed: list = []
        self.blocked: list = []
        self.skipped: list = []

    def record(self, outcome: str, label: str) -> None:
        with self._lock:
            getattr(self, outcome).append(label)

    def summary(self) -> dict:
        with self._lock:
            return {
                "passed": len(self.passed),
                "blocked": len(self.blocked),
                "skipped": len(self.skipped),
                "blocked_sections": sorted(set(self.blocked)),
                "skipped_sections": sorted(set(self.skipped)),
            }


# One audit per report run. A ContextVar (not a module global) because the
# API can generate several reports concurrently in the same process.
_guardrail_audit: contextvars.ContextVar = contextvars.ContextVar(
    "guardrail_audit", default=None
)


def start_guardrail_audit() -> GuardrailAudit:
    """Begin a fresh audit for the current report run and return it."""
    audit = GuardrailAudit()
    _guardrail_audit.set(audit)
    return audit


def _record_guardrail(outcome: str, label: str) -> None:
    audit = _guardrail_audit.get()
    if audit is not None:
        audit.record(outcome, label)


def _validate_with_guardrails(text: str, label: str = "output") -> str:
    """
    Validate text through Bedrock Guardrails if available.
    Returns the original text if guardrails are unavailable or text passes.
    Returns GUARDRAIL_FALLBACK if text is blocked.
    Every outcome is recorded in the current GuardrailAudit (if one is active),
    including skips, so the report can disclose unvalidated sections.
    """
    guard = _get_guardrail()
    if guard is None:
        _record_guardrail("skipped", label)
        return text
    try:
        result = guard.check_output(text)
        if result.get("blocked"):
            logger.warning(
                "Guardrail BLOCKED %s: %s", label, result.get("details", [])
            )
            _record_guardrail("blocked", label)
            return GUARDRAIL_FALLBACK
        # check_output() swallows service errors and returns action="ERROR"
        # with blocked=False, so an error here means the text was never checked.
        if result.get("action") == "ERROR":
            logger.warning(
                "Guardrail check errored for %s: %s — using original text",
                label, result.get("details", []),
            )
            _record_guardrail("skipped", label)
            return text
        _record_guardrail("passed", label)
        return text
    except Exception as e:
        logger.warning("Guardrail check failed for %s: %s — using original text", label, e)
        _record_guardrail("skipped", label)
        return text


# ──────────────────────────────────────────────────────────────
# Per-chart prompt templates
# ──────────────────────────────────────────────────────────────
CHART_PROMPTS = {
    "price_sma": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following stock price and moving average data for {ticker}. Write a focused 4-6 sentence
analysis covering: (1) current trend direction and strength based on SMA alignment, (2) short-term
momentum implied by the 7D vs 30D SMA relationship, (3) medium-term trend health from the 30D vs 90D
SMA, and (4) actionable implications for positioning. Reference specific price levels and percentages.
Avoid generic statements — every sentence must add analytical value.

Data:
- Current Price: ${current_price}
- 7-Day SMA: ${sma_7d}
- 30-Day SMA: ${sma_30d}
- 90-Day SMA: ${sma_90d}
- Trend Signal: {trend_signal}
- Date Range: {date_range}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "volatility": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following volume and volatility data for {ticker}. Write a focused 4-6 sentence analysis
covering: (1) current volatility level relative to what is typical for large-cap equities, (2) what
the daily range implies about intraday price risk, (3) volume context and liquidity profile, and
(4) overall risk characterization for portfolio allocation purposes. Be quantitative.
Avoid generic statements — every sentence must add analytical value.

Data:
- Average Daily Volume: {avg_volume:,}
- 30-Day Volatility: {volatility_30d_pct}%
- Average Daily Range: {daily_range_pct_avg}%

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "revenue_growth": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following revenue and net income growth data for {ticker}. Write a focused 4-6 sentence
analysis covering: (1) revenue growth trajectory and whether it is accelerating or decelerating,
(2) net income growth relative to revenue growth (margin expansion or compression), (3) what the
fundamental signal implies about earnings quality, and (4) forward-looking sustainability of these
growth rates. Reference the specific percentages provided. Avoid generic statements.

Data:
- Latest Revenue Growth (YoY): {latest_revenue_growth_yoy}%
- Latest Net Income Growth (YoY): {latest_net_income_growth_yoy}%
- Fundamental Signal: {fundamental_signal}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "eps_trend": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following earnings per share data for {ticker}. Write a focused 4-6 sentence analysis
covering: (1) current EPS level and what it implies about per-share profitability, (2) sequential
(QoQ) vs year-over-year growth and whether earnings momentum is improving, (3) earnings quality
signals (e.g., is EPS growth driven by margin expansion, revenue growth, or buybacks), and
(4) implications for forward earnings estimates. Reference the specific figures provided.
Avoid generic statements.

Data:
- Latest EPS: ${latest_eps}
- EPS Growth (YoY): {eps_growth_yoy_pct}%
- EPS Growth (QoQ): {eps_growth_qoq_pct}%

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "sentiment": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following news sentiment data for {ticker}. Write a focused 4-6 sentence analysis
covering: (1) current sentiment level and where it falls on the bearish-to-bullish spectrum,
(2) whether sentiment is trending positively or negatively and at what pace, (3) media coverage
volume and what it implies about institutional attention, and (4) how sentiment aligns with or
diverges from the stock's fundamental and technical profile. Avoid vague characterizations —
quantify sentiment strength and direction. Avoid generic statements.

Data:
- 7-Day Average Sentiment Score: {sentiment_score_7d_avg} (scale: -1 bearish to +1 bullish)
- Sentiment Label: {sentiment_label}
- Sentiment Trend: {sentiment_trend}
- Total Articles (Last 30 Days): {total_articles_30d}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "financial_health": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following financial health data for {ticker} sourced from SEC filings. Write a focused
4-6 sentence analysis covering: (1) revenue scale and profitability as measured by net margin,
(2) balance sheet leverage via debt-to-equity and what it implies about financial flexibility,
(3) the overall health rating and whether it is justified by the underlying metrics, and
(4) key risks or strengths in the capital structure. If operating margin data is available,
comment on operational efficiency. Reference the specific figures provided. Avoid generic statements.

Data:
- Total Revenue: ${total_revenue:,.0f}
- Net Margin: {net_margin_pct}%
- Operating Margin: {operating_margin_pct}%
- Debt-to-Equity Ratio: {debt_to_equity_ratio}
- Financial Health Rating: {financial_health}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "margin_trend": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following profitability margin trend data for {ticker}. Write a focused 4-6 sentence
analysis covering: (1) the current level of net and operating margins and what they imply about
cost structure, (2) whether margins are expanding or compressing and the likely drivers, (3) how
the spread between operating and net margin reflects non-operating items (interest, tax), and
(4) sustainability of current margin levels relative to industry norms. Be quantitative.
Avoid generic statements.

Data:
- Quarters Covered: {num_quarters}
- Latest Net Margin: {latest_net_margin_pct}%
- Latest Operating Margin: {latest_operating_margin_pct}%
- Margin Trend: {margin_trend}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "balance_sheet": """
You are a senior equity research analyst writing a section of an institutional-grade financial report.
Analyze the following balance sheet composition data for {ticker}. Write a focused 4-6 sentence
analysis covering: (1) total asset scale and what it implies about the company's size, (2) the
liability-to-equity mix and what it reveals about leverage and financial risk, (3) the equity
percentage and whether it suggests conservative or aggressive financing, and (4) implications
for the company's ability to weather economic downturns or fund growth. Reference the specific
dollar amounts. Avoid generic statements.

Data:
- Total Assets: ${total_assets_b}B
- Total Liabilities: ${total_liabilities_b}B
- Stockholders' Equity: ${stockholders_equity_b}B
- Equity as % of Assets: {equity_pct}%
- Quarters Covered: {num_quarters}

Write in third person, professional tone. Do not use bullet points or headers.
""",
}


# ──────────────────────────────────────────────────────────────
# Summary-mode prompt templates (shorter output for condensed PDF)
# ──────────────────────────────────────────────────────────────
CHART_PROMPTS_SUMMARY = {
    "price_sma": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of the stock price
and moving average data for {ticker}. Cover the dominant trend direction and the key actionable
implication. Reference specific price levels.

Data:
- Current Price: ${current_price}
- 7-Day SMA: ${sma_7d}
- 30-Day SMA: ${sma_30d}
- 90-Day SMA: ${sma_90d}
- Trend Signal: {trend_signal}
- Date Range: {date_range}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "volatility": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of the volume and
volatility profile for {ticker}. Focus on current volatility level and risk characterization.

Data:
- Average Daily Volume: {avg_volume:,}
- 30-Day Volatility: {volatility_30d_pct}%
- Average Daily Range: {daily_range_pct_avg}%

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "revenue_growth": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of revenue and
net income growth for {ticker}. Focus on the growth trajectory and earnings quality signal.

Data:
- Latest Revenue Growth (YoY): {latest_revenue_growth_yoy}%
- Latest Net Income Growth (YoY): {latest_net_income_growth_yoy}%
- Fundamental Signal: {fundamental_signal}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "eps_trend": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of earnings per
share data for {ticker}. Focus on EPS level and momentum direction.

Data:
- Latest EPS: ${latest_eps}
- EPS Growth (YoY): {eps_growth_yoy_pct}%
- EPS Growth (QoQ): {eps_growth_qoq_pct}%

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "sentiment": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of news sentiment
for {ticker}. Focus on the current sentiment level and whether it aligns with fundamentals.

Data:
- 7-Day Average Sentiment Score: {sentiment_score_7d_avg} (scale: -1 bearish to +1 bullish)
- Sentiment Label: {sentiment_label}
- Sentiment Trend: {sentiment_trend}
- Total Articles (Last 30 Days): {total_articles_30d}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "financial_health": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of the financial
health of {ticker}. Focus on profitability and leverage.

Data:
- Total Revenue: ${total_revenue:,.0f}
- Net Margin: {net_margin_pct}%
- Operating Margin: {operating_margin_pct}%
- Debt-to-Equity Ratio: {debt_to_equity_ratio}
- Financial Health Rating: {financial_health}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "margin_trend": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of the profitability
margin trend for {ticker}. Focus on whether margins are expanding or compressing.

Data:
- Quarters Covered: {num_quarters}
- Latest Net Margin: {latest_net_margin_pct}%
- Latest Operating Margin: {latest_operating_margin_pct}%
- Margin Trend: {margin_trend}

Write in third person, professional tone. Do not use bullet points or headers.
""",

    "balance_sheet": """
You are a senior equity research analyst. Write a concise 2-3 sentence summary of the balance
sheet composition for {ticker}. Focus on leverage and financial flexibility.

Data:
- Total Assets: ${total_assets_b}B
- Total Liabilities: ${total_liabilities_b}B
- Stockholders' Equity: ${stockholders_equity_b}B
- Equity as % of Assets: {equity_pct}%
- Quarters Covered: {num_quarters}

Write in third person, professional tone. Do not use bullet points or headers.
""",
}


# ──────────────────────────────────────────────────────────────
# Cortex helpers
# ──────────────────────────────────────────────────────────────
def _cortex_complete(session: Session, prompt: str) -> str:
    """
    Call Snowflake Cortex COMPLETE() with the given prompt.
    Returns the generated text string.
    """
    safe_prompt = prompt.replace("'", "''")

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{CORTEX_MODEL}',
            '{safe_prompt}'
        ) AS response
    """
    result = session.sql(sql).collect()
    if result and result[0]["RESPONSE"]:
        return result[0]["RESPONSE"].strip()
    return ""


def _cortex_summarize(session: Session, text: str) -> str:
    """
    Call Snowflake Cortex SUMMARIZE() on the given text.
    Returns a condensed summary string.
    """
    if not text or len(text.strip()) < 100:
        return ""

    truncated = text[:MAX_SEC_TEXT_CHARS]
    safe_text = truncated.replace("'", "''")

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
            '{safe_text}'
        ) AS summary
    """
    result = session.sql(sql).collect()
    if result and result[0]["SUMMARY"]:
        return result[0]["SUMMARY"].strip()
    return ""


# ──────────────────────────────────────────────────────────────
# Chart analysis
# ──────────────────────────────────────────────────────────────
def analyze_chart(session: Session, chart: dict, ticker: str,
                  prior_analyses: list = None,
                  detail_level: str = "detailed") -> dict:
    """
    Generate a written analysis paragraph for a single validated chart.
    Supports Chain-of-Analysis: prior analyses provide cross-referencing context.
    Optionally enriched with SEC filing context from Bedrock KB (Integration 2).
    Output validated by Guardrails (Integration 1).

    Args:
        detail_level: "detailed" for full 4-6 sentence analysis,
                      "summary" for concise 2-3 sentence analysis.
    """
    chart_id = chart["chart_id"]
    data_summary = chart.get("data_summary", {})

    prompts = CHART_PROMPTS_SUMMARY if detail_level == "summary" else CHART_PROMPTS
    prompt_template = prompts.get(chart_id) or CHART_PROMPTS.get(chart_id)
    if not prompt_template:
        logger.warning("No prompt template for chart_id '%s', skipping", chart_id)
        return {
            "chart_id": chart_id,
            "title": chart.get("title", ""),
            "analysis_text": "Analysis not available for this chart.",
        }

    try:
        # Fill in the prompt template with actual data values
        prompt = prompt_template.format(ticker=ticker, **data_summary)

        # Integration 2: Enrich with targeted SEC filing context from Bedrock KB
        kb = _get_bedrock_kb()
        if kb is not None:
            try:
                kb_query = KB_QUERIES.get(chart_id, "")
                if kb_query:
                    chunks = kb.retrieve(kb_query, ticker=ticker, max_results=2)
                    if chunks:
                        sec_context = "\n".join(c["text"][:500] for c in chunks[:2])
                        prompt += (
                            f"\n\nRelevant SEC filing context:\n{sec_context}\n\n"
                            "Reference this SEC filing data in your analysis where relevant."
                        )
                        logger.info("KB enrichment: %d chunks for %s/%s", len(chunks), ticker, chart_id)
            except Exception as e:
                logger.debug("KB enrichment skipped for %s: %s", chart_id, e)

        # Chain-of-Analysis: append prior analyses as context
        if prior_analyses:
            context_lines = []
            for pa in prior_analyses:
                text = pa["analysis_text"][:500]
                if len(pa["analysis_text"]) > 500:
                    text += "..."
                context_lines.append(f"- {pa['title']}: {text}")

            prompt += (
                "\n\nPrior analysis context (you MUST cross-reference at least one prior finding):\n"
                + "\n".join(context_lines)
                + "\n\nIntegrate the prior findings into your analysis. Specifically:\n"
                "- Reference at least one prior observation by name (e.g., 'As noted in the Price & SMA analysis...')\n"
                "- Identify whether this chart's data confirms, contradicts, or adds nuance to earlier findings\n"
                "- Use connective phrases like 'consistent with', 'in contrast to', 'further supported by', "
                "'this adds context to the earlier observation that...'\n"
                "- If this chart reveals a divergence from prior trends, explicitly call it out as a risk or opportunity"
            )

        logger.info("Generating analysis for chart: %s", chart_id)

        analysis_text = _cortex_complete(session, prompt)

        if not analysis_text:
            logger.warning("Cortex returned empty response for chart '%s'", chart_id)
            analysis_text = "Insufficient data to generate analysis for this chart."

        logger.info("Analysis generated for %s (%d chars)", chart_id, len(analysis_text))

        return {
            "chart_id": chart_id,
            "title": chart.get("title", ""),
            "analysis_text": analysis_text,
        }

    except Exception as e:
        logger.error("Failed to analyze chart '%s': %s", chart_id, e)
        return {
            "chart_id": chart_id,
            "title": chart.get("title", ""),
            "analysis_text": "Analysis could not be generated due to a processing error.",
        }


def analyze_all_charts(session: Session, charts: list, ticker: str,
                       detail_level: str = "detailed") -> list:
    """
    Generate analysis for all validated charts using Chain-of-Analysis.
    Charts are analyzed in canonical order so each analysis can
    cross-reference prior findings progressively.
    """
    logger.info("Starting Chain-of-Analysis for %s (%d charts, detail_level=%s)",
                ticker, len(charts), detail_level)

    ordered = sorted(
        charts,
        key=lambda c: (
            ANALYSIS_ORDER.index(c["chart_id"])
            if c["chart_id"] in ANALYSIS_ORDER else 99
        ),
    )

    analyses = []
    for chart in ordered:
        if not chart.get("validated", True):
            logger.warning("Skipping unvalidated chart: %s", chart.get("chart_id"))
            continue
        analysis = analyze_chart(session, chart, ticker, prior_analyses=analyses,
                                 detail_level=detail_level)
        analyses.append(analysis)

    logger.info("Completed Chain-of-Analysis for %d/%d charts", len(analyses), len(charts))
    return analyses


# ──────────────────────────────────────────────────────────────
# SEC filing retrieval (Integration 2: Bedrock KB RAG)
# ──────────────────────────────────────────────────────────────
def _fetch_sec_text_rag(ticker: str, chart_id: str = None) -> dict:
    """
    Fetch SEC filing context via Bedrock KB semantic search.
    Returns dict with mda_text and risk_text assembled from retrieved chunks.
    """
    kb = _get_bedrock_kb()
    if kb is None:
        return {"mda_text": "", "risk_text": ""}

    try:
        query = KB_QUERIES.get(chart_id, "financial performance and risk factors")
        chunks = kb.retrieve(query, ticker=ticker, max_results=5)

        mda_chunks = []
        risk_chunks = []
        for chunk in chunks:
            section = chunk.get("section", "")
            if section == "Risk Factors":
                risk_chunks.append(chunk["text"])
            else:
                mda_chunks.append(chunk["text"])

        mda_text = "\n\n".join(mda_chunks) if mda_chunks else ""
        risk_text = "\n\n".join(risk_chunks) if risk_chunks else ""

        logger.info(
            "Bedrock KB retrieved %d chunks for %s: MD&A=%d chars, Risk=%d chars",
            len(chunks), ticker, len(mda_text), len(risk_text),
        )
        return {"mda_text": mda_text, "risk_text": risk_text}

    except Exception as e:
        logger.warning("Bedrock KB retrieval failed for %s: %s", ticker, e)
        return {"mda_text": "", "risk_text": ""}


def _extract_section(text: str, heading: str, max_chars: int = 50_000) -> str:
    """Extract a named section from full SEC filing text.

    Uses case-insensitive matching to find sections like
    "Management's Discussion and Analysis" or "Risk Factors".
    Returns the text between the heading and the next major heading,
    truncated to *max_chars*.
    """
    if not text:
        return ""
    # Common SEC filing section headings used as stop markers
    stop_headings = [
        r"item\s+\d",
        r"part\s+(?:i{1,3}|iv|[1-4])",
        r"signatures",
        r"financial\s+statements",
        r"quantitative\s+and\s+qualitative",
        r"controls\s+and\s+procedures",
        r"legal\s+proceedings",
        r"mine\s+safety",
        r"unresolved\s+staff\s+comments",
    ]
    # Find the start of the target section
    pattern = re.compile(re.escape(heading), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    # Find the next major heading after the start
    remaining = text[start:start + max_chars + 10_000]
    stop_pattern = re.compile(
        r"\n\s*(?:" + "|".join(stop_headings) + r")\b",
        re.IGNORECASE,
    )
    stop_match = stop_pattern.search(remaining, pos=200)  # skip at least 200 chars
    if stop_match:
        section = remaining[:stop_match.start()]
    else:
        section = remaining
    return section.strip()[:max_chars]


def fetch_sec_text(session: Session, ticker: str, chart_id: str = None) -> dict:
    """
    Fetch SEC filing context for the given ticker.

    Strategy:
        1. If Bedrock KB available: semantic search for relevant sections
        2. Fallback: SQL query for extracted filings from RAW_SEC_FILING_DOCUMENTS
        3. Final fallback: parse MD&A/Risk from full text in RAW_SEC_FILING_TEXT
    """
    # Try Bedrock KB RAG first
    kb_result = _fetch_sec_text_rag(ticker, chart_id=chart_id)
    if kb_result["mda_text"] or kb_result["risk_text"]:
        return kb_result

    # Fallback: original SQL approach
    logger.info("Using SQL fallback for SEC text (%s)", ticker)
    safe_ticker = ticker.upper().strip()
    if not safe_ticker.isalpha() or len(safe_ticker) > 10:
        logger.warning("Invalid ticker for SEC text: %s", ticker)
        return {"mda_text": "", "risk_text": ""}
    sql = f"""
        SELECT MDA_TEXT, RISK_FACTORS_TEXT
        FROM RAW.RAW_SEC_FILING_DOCUMENTS
        WHERE TICKER = '{safe_ticker}'
          AND EXTRACTION_STATUS = 'extracted'
          AND MDA_TEXT IS NOT NULL
        ORDER BY FILING_DATE DESC
        LIMIT 1
    """
    try:
        rows = session.sql(sql).collect()
        if rows:
            return {
                "mda_text": rows[0]["MDA_TEXT"] or "",
                "risk_text": rows[0]["RISK_FACTORS_TEXT"] or "",
            }
    except Exception as e:
        logger.warning("Could not fetch SEC text for %s: %s", ticker, e)

    # Final fallback: parse RAW_SEC_FILING_TEXT (populated by SECFilingLoader)
    logger.info("Trying RAW_SEC_FILING_TEXT fallback for %s", ticker)
    try:
        fallback_sql = f"""
            SELECT FILING_TEXT::STRING AS FILING_TEXT
            FROM RAW.RAW_SEC_FILING_TEXT
            WHERE TICKER = '{safe_ticker}'
              AND FORM_TYPE IN ('10-K', '10-Q')
              AND FILING_TEXT IS NOT NULL
            ORDER BY FILING_DATE DESC
            LIMIT 1
        """
        fb_rows = session.sql(fallback_sql).collect()
        if fb_rows and fb_rows[0]["FILING_TEXT"]:
            full_text = fb_rows[0]["FILING_TEXT"]
            mda = _extract_section(full_text, "management's discussion and analysis")
            risk = _extract_section(full_text, "risk factors")
            if mda or risk:
                logger.info(
                    "Extracted from RAW_SEC_FILING_TEXT for %s: MD&A=%d chars, Risk=%d chars",
                    ticker, len(mda), len(risk),
                )
                return {"mda_text": mda, "risk_text": risk}
    except Exception as e:
        logger.warning("RAW_SEC_FILING_TEXT fallback failed for %s: %s", ticker, e)

    return {"mda_text": "", "risk_text": ""}


# ──────────────────────────────────────────────────────────────
# SEC filing summarization
# ──────────────────────────────────────────────────────────────
def summarize_sec_filings(session: Session, ticker: str) -> dict:
    """
    Fetch and summarize SEC filing text (MD&A + Risk Factors) using Cortex.
    Output validated by Guardrails (Integration 1).
    """
    logger.info("Fetching SEC filing text for %s", ticker)
    sec_text = fetch_sec_text(session, ticker, chart_id=None)

    mda_summary = ""
    risk_summary = ""

    if sec_text["mda_text"]:
        logger.info("Summarizing MD&A text (%d chars)", len(sec_text["mda_text"]))
        mda_summary = _cortex_summarize(session, sec_text["mda_text"])
        if mda_summary and len(mda_summary) >= 100:
            mda_summary = _validate_with_guardrails(mda_summary, label="mda_summary")
            logger.info("MD&A summary generated (%d chars)", len(mda_summary))
        else:
            logger.warning(
                "MD&A summary too short or empty for %s (%d chars), discarding",
                ticker, len(mda_summary or ""),
            )
            mda_summary = ""
    else:
        logger.warning("No extracted MD&A text found for %s", ticker)

    if sec_text["risk_text"]:
        logger.info("Summarizing Risk Factors text (%d chars)", len(sec_text["risk_text"]))
        risk_summary = _cortex_summarize(session, sec_text["risk_text"])
        if risk_summary and len(risk_summary) >= 100:
            risk_summary = _validate_with_guardrails(risk_summary, label="risk_summary")
            logger.info("Risk summary generated (%d chars)", len(risk_summary))
        else:
            logger.warning(
                "Risk summary too short or empty for %s (%d chars), discarding",
                ticker, len(risk_summary or ""),
            )
            risk_summary = ""
    else:
        logger.warning("No extracted Risk Factors text found for %s", ticker)

    return {
        "mda_summary": mda_summary or f"MD&A summary not available for {ticker}.",
        "risk_summary": risk_summary or f"Risk factors summary not available for {ticker}.",
    }


# ──────────────────────────────────────────────────────────────
# Investment thesis synthesis (Integration 3: Multi-Model)
# ──────────────────────────────────────────────────────────────
def synthesize_analyses(session: Session, analyses: list, ticker: str,
                        detail_level: str = "detailed") -> str:
    """
    Generate an overall investment thesis synthesizing all individual chart analyses.

    Strategy:
        1. If MultiModelAnalyzer available: multi-model consensus (higher confidence)
        2. Fallback: single Cortex call (original behavior)

    Output validated by Guardrails (Integration 1).
    """
    if not analyses:
        return "Insufficient data to generate investment thesis."

    combined = "\n\n".join(
        f"[{a['title']}]: {a['analysis_text']}" for a in analyses
    )

    if detail_level == "summary":
        thesis_question = (
            f"Synthesize the following individual analyses for {ticker} into a concise "
            f"investment thesis of 3-4 sentences: (1) the dominant narrative (growth, value, "
            f"or transitional), (2) the key bull and bear arguments in one sentence each, "
            f"(3) a forward-looking conviction level (high/moderate/low). "
            f"Do not use bullet points or headers. Write in third person, professional tone. "
            f"Reference specific data from the analyses below."
        )
    else:
        thesis_question = (
            f"Synthesize the following individual analyses for {ticker} into a cohesive "
            f"investment thesis of 6-8 sentences structured as follows: "
            f"(1) Open with the dominant narrative — is this a growth, value, or transitional story? "
            f"(2) Summarize the bull case in 1-2 sentences citing specific metrics from the analyses. "
            f"(3) Summarize the bear case or key risks in 1-2 sentences. "
            f"(4) Note any contradictions between technical, fundamental, and sentiment signals. "
            f"(5) Conclude with a forward-looking outlook and conviction level (high/moderate/low). "
            f"Do not use bullet points or headers. Write in third person, professional tone. "
            f"Every sentence must reference specific data from the analyses below."
        )

    # Integration 3: Try multi-model consensus first
    mm = _get_multi_model()
    if mm is not None:
        try:
            logger.info("Running multi-model consensus for %s investment thesis", ticker)
            result = mm.consensus(thesis_question, context=combined)
            consensus_text = result.get("consensus", "")
            succeeded = result.get("summary", {}).get("succeeded", 0)
            if consensus_text and succeeded >= 2:
                total = result.get("summary", {}).get("total_models", 0)
                logger.info(
                    "Multi-model consensus generated (%d chars, %d/%d models)",
                    len(consensus_text), succeeded, total,
                )
                return consensus_text
            else:
                logger.warning("Multi-model consensus empty/failed, falling back to Cortex")
        except Exception as e:
            logger.warning("Multi-model consensus failed: %s — falling back to Cortex", e)

    # Fallback: single Cortex call
    prompt = (
        f"You are a senior equity research analyst writing the executive summary for a "
        f"{ticker} equity research report. {thesis_question}\n\n{combined}"
    )

    thesis = _cortex_complete(session, prompt)
    if thesis:
        logger.info("Investment thesis generated via Cortex (%d chars)", len(thesis))
    else:
        logger.warning("Cortex returned empty investment thesis for %s", ticker)
        thesis = f"Investment thesis not available for {ticker}."

    return thesis


# ──────────────────────────────────────────────────────────────
# Company Overview (called by orchestrator)
# ──────────────────────────────────────────────────────────────

# In-memory peer-group cache, seeded with well-known tickers.
# resolve_peers() populates this dynamically for unknown tickers.
_PEER_CACHE = {
    "AAPL":  ["MSFT", "GOOGL", "AMZN"],
    "MSFT":  ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["AAPL", "MSFT", "META"],
    "AMZN":  ["AAPL", "MSFT", "GOOGL"],
    "TSLA":  ["NIO", "RIVN", "F"],
    "NVDA":  ["AMD", "INTC", "AVGO"],
    "META":  ["GOOGL", "SNAP", "PINS"],
    "JPM":   ["BAC", "GS", "MS"],
    "BAC":   ["JPM", "GS", "C"],
    "GS":    ["JPM", "MS", "BAC"],
}

# Sector-level fallback peers when yfinance lookup fails
_SECTOR_DEFAULTS = {
    "Technology":        ["AAPL", "MSFT", "GOOGL"],
    "Communication Services": ["GOOGL", "META", "NFLX"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD"],
    "Financial Services": ["JPM", "BAC", "GS"],
    "Healthcare":        ["JNJ", "UNH", "PFE"],
    "Consumer Defensive": ["PG", "KO", "PEP"],
    "Industrials":       ["CAT", "HON", "UNP"],
    "Energy":            ["XOM", "CVX", "COP"],
    "Utilities":         ["NEE", "DUK", "SO"],
    "Real Estate":       ["AMT", "PLD", "CCI"],
    "Basic Materials":   ["LIN", "APD", "ECL"],
}


def resolve_peers(ticker: str, max_peers: int = 3) -> list[str]:
    """Resolve peer tickers for comparison analysis.

    Resolution order:
        1. In-memory cache
        2. yfinance sector/industry lookup → find peers in same sector
        3. Sector-level default peers
        4. Generic large-cap fallback
    """
    ticker = ticker.upper().strip()

    # Method 1: cache
    if ticker in _PEER_CACHE:
        return _PEER_CACHE[ticker][:max_peers]

    # Method 2: yfinance sector lookup
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "")

        if sector and sector in _SECTOR_DEFAULTS:
            peers = [p for p in _SECTOR_DEFAULTS[sector] if p != ticker][:max_peers]
            if peers:
                _PEER_CACHE[ticker] = peers
                logger.info("Resolved peers for %s via sector '%s': %s", ticker, sector, peers)
                return peers
    except Exception as exc:
        logger.debug("yfinance peer lookup failed for %s: %s", ticker, exc)

    # Method 3: generic large-cap fallback
    fallback = [p for p in ["AAPL", "MSFT", "GOOGL"] if p != ticker][:max_peers]
    _PEER_CACHE[ticker] = fallback
    logger.info("Using fallback peers for %s: %s", ticker, fallback)
    return fallback


def _query_company_facts(session: Session, ticker: str) -> dict:
    """Query DIM_COMPANY and FCT_FUNDAMENTALS_GROWTH for key company facts."""
    safe_ticker = ticker.upper().strip()
    facts = {}

    # DIM_COMPANY
    try:
        rows = session.sql(f"""
            SELECT MARKET_CAP, PE_RATIO, PROFIT_MARGIN, DEBT_TO_EQUITY,
                   MARKET_CAP_CATEGORY, CIK, TOTAL_TRADING_DAYS, DATA_SOURCES_AVAILABLE
            FROM ANALYTICS.DIM_COMPANY
            WHERE TICKER = '{safe_ticker}'
            LIMIT 1
        """).collect()
        if rows:
            r = rows[0]
            facts["market_cap"] = r["MARKET_CAP"]
            facts["pe_ratio"] = r["PE_RATIO"]
            facts["profit_margin"] = r["PROFIT_MARGIN"]
            facts["debt_to_equity"] = r["DEBT_TO_EQUITY"]
            facts["market_cap_category"] = r["MARKET_CAP_CATEGORY"]
            facts["cik"] = r["CIK"]
    except Exception as e:
        logger.warning("Could not query DIM_COMPANY for %s: %s", ticker, e)

    # Latest fundamentals
    try:
        rows = session.sql(f"""
            SELECT FISCAL_QUARTER, REVENUE, NET_INCOME, EPS, NET_MARGIN_PCT
            FROM ANALYTICS.FCT_FUNDAMENTALS_GROWTH
            WHERE TICKER = '{safe_ticker}'
            ORDER BY FISCAL_QUARTER DESC
            LIMIT 1
        """).collect()
        if rows:
            r = rows[0]
            facts["latest_quarter"] = r["FISCAL_QUARTER"]
            facts["revenue"] = r["REVENUE"]
            facts["net_income"] = r["NET_INCOME"]
            facts["eps"] = r["EPS"]
            facts["net_margin"] = r["NET_MARGIN_PCT"]
    except Exception as e:
        logger.warning("Could not query FCT_FUNDAMENTALS_GROWTH for %s: %s", ticker, e)

    # Override D/E with SEC-derived ratio (DIM_COMPANY value from Yahoo Finance
    # can be an unscaled percentage, e.g. 102.63 instead of the actual ratio 3.30)
    try:
        sec_rows = session.sql(f"""
            SELECT DEBT_TO_EQUITY_RATIO, NET_MARGIN_PCT
            FROM ANALYTICS.FCT_SEC_FINANCIAL_SUMMARY
            WHERE TICKER = '{safe_ticker}'
              AND REPORTING_FREQUENCY = 'quarterly'
              AND DEBT_TO_EQUITY_RATIO IS NOT NULL
            ORDER BY FISCAL_YEAR DESC, FISCAL_PERIOD DESC
            LIMIT 1
        """).collect()
        if sec_rows:
            r = sec_rows[0]
            if r["DEBT_TO_EQUITY_RATIO"] is not None:
                facts["debt_to_equity"] = float(r["DEBT_TO_EQUITY_RATIO"])
            if r["NET_MARGIN_PCT"] is not None and facts.get("net_margin") is None:
                facts["net_margin"] = float(r["NET_MARGIN_PCT"])
    except Exception as e:
        logger.warning("Could not query FCT_SEC_FINANCIAL_SUMMARY for D/E override for %s: %s",
                        ticker, e)

    return facts


def generate_company_overview(session: Session, ticker: str,
                              detail_level: str = "detailed") -> dict:
    """
    Generate a Company Overview section for the PDF report.

    Queries DIM_COMPANY + FCT_FUNDAMENTALS_GROWTH for key facts,
    uses Cortex COMPLETE for an AI-generated company description,
    and optionally pulls business segment info from Bedrock KB.

    Returns:
        {
            "company_description": str,
            "key_facts": dict,
            "business_segments": str,
        }
    """
    logger.info("Generating company overview for %s", ticker)

    facts = _query_company_facts(session, ticker)

    # Format market cap for prompt
    mc = facts.get("market_cap")
    if mc and isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12:
            mc_str = f"${mc/1e12:.2f} trillion"
        elif mc >= 1e9:
            mc_str = f"${mc/1e9:.1f} billion"
        else:
            mc_str = f"${mc/1e6:.0f} million"
    else:
        mc_str = "N/A"

    pe = facts.get("pe_ratio")
    pe_str = f"{pe:.1f}" if pe else "N/A"
    margin = facts.get("net_margin") or facts.get("profit_margin")
    margin_str = f"{float(margin)*100:.1f}%" if margin and margin < 1 else (
        f"{float(margin):.1f}%" if margin else "N/A"
    )

    # AI-generated company description via Cortex
    if detail_level == "summary":
        prompt = (
            f"Write a concise 2-3 sentence company overview for {ticker}. Cover the core business "
            f"and key competitive position. "
            f"Key facts: Market cap {mc_str}, P/E ratio {pe_str}, net margin {margin_str}. "
            f"Write in third person, professional tone. Do not use bullet points or headers."
        )
    else:
        prompt = (
            f"Write a comprehensive 6-8 sentence company overview for {ticker} suitable for "
            f"an institutional equity research report. Cover:\n"
            f"1. Core business description and key products/services\n"
            f"2. Total addressable market (TAM) and industry positioning\n"
            f"3. Competitive advantages and moats (brand, ecosystem, IP, scale)\n"
            f"4. Key growth drivers and strategic initiatives\n"
            f"Key facts: Market cap {mc_str}, P/E ratio {pe_str}, net margin {margin_str}. "
            f"Write in third person, professional tone. Do not use bullet points or headers."
        )

    description = _cortex_complete(session, prompt)
    if not description:
        description = f"Company overview not available for {ticker}."
    else:
        description = _validate_with_guardrails(description, label="company_overview")

    # Competitive landscape via Cortex
    peers = resolve_peers(ticker)
    peer_str = ", ".join(peers[:4]) if peers else "major industry peers"
    if detail_level == "summary":
        comp_prompt = (
            f"Write a 1-2 sentence competitive positioning statement for {ticker} "
            f"relative to {peer_str}. Write in professional third-person tone."
        )
    else:
        comp_prompt = (
            f"Write a concise 3-4 sentence competitive landscape analysis for {ticker} "
            f"in the context of its primary competitors ({peer_str}). Cover relative market "
            f"position, key differentiators, and competitive threats. "
            f"Write in professional third-person tone."
        )
    competitive_landscape = _cortex_complete(session, comp_prompt)
    if competitive_landscape:
        competitive_landscape = _validate_with_guardrails(
            competitive_landscape, label="competitive_landscape"
        )
    else:
        competitive_landscape = ""

    # Business segments from Bedrock KB
    segments = ""
    kb = _get_bedrock_kb()
    if kb is not None:
        try:
            chunks = kb.retrieve(
                f"{ticker} business segments revenue breakdown products services",
                ticker=ticker, max_results=3
            )
            if chunks:
                raw_segment_text = "\n\n".join(c["text"] for c in chunks[:3])
                # Use Cortex to synthesize a clean summary from raw KB chunks
                seg_prompt = (
                    f"Based on the following SEC filing excerpts for {ticker}, write a concise "
                    f"2-3 sentence summary of the company's key business segments and their "
                    f"revenue contributions. If specific percentages are available, include them. "
                    f"Write in third person, professional tone.\n\n{raw_segment_text[:2000]}"
                )
                segments = _cortex_complete(session, seg_prompt)
                if segments:
                    segments = _validate_with_guardrails(segments, label="business_segments")
                else:
                    segments = raw_segment_text[:600]
                logger.info("Retrieved business segment context for %s (%d chars)", ticker, len(segments))
        except Exception as e:
            logger.debug("KB segment retrieval skipped for %s: %s", ticker, e)

    result = {
        "company_description": description,
        "key_facts": facts,
        "business_segments": segments,
        "competitive_landscape": competitive_landscape,
    }
    logger.info("Company overview generated for %s", ticker)
    return result


# ──────────────────────────────────────────────────────────────
# Peer Comparison (called by orchestrator)
# ──────────────────────────────────────────────────────────────

def generate_peer_comparison(session: Session, ticker: str,
                             detail_level: str = "detailed") -> dict:
    """
    Generate a Peer Comparison section for the PDF report.

    Queries DIM_COMPANY + FCT_FUNDAMENTALS_GROWTH for the target ticker
    and its peer group, then uses Cortex COMPLETE for a comparison summary.

    Returns:
        {
            "ticker": str,
            "peers": [{"ticker": str, "market_cap": float, ...}, ...],
            "comparison_summary": str,
        }
    """
    logger.info("Generating peer comparison for %s", ticker)
    safe_ticker = ticker.upper().strip()
    peer_tickers = resolve_peers(safe_ticker)

    if not peer_tickers:
        logger.warning("No peer group defined for %s", ticker)
        return {
            "ticker": ticker,
            "peers": [],
            "comparison_summary": f"Peer comparison not available for {ticker}.",
        }

    all_tickers = [safe_ticker] + peer_tickers
    ticker_list = ", ".join(f"'{t}'" for t in all_tickers)

    peers_data = []

    # Query DIM_COMPANY for all tickers
    try:
        rows = session.sql(f"""
            SELECT TICKER, MARKET_CAP, PE_RATIO, PROFIT_MARGIN, DEBT_TO_EQUITY,
                   MARKET_CAP_CATEGORY
            FROM ANALYTICS.DIM_COMPANY
            WHERE TICKER IN ({ticker_list})
        """).collect()

        company_map = {}
        for r in rows:
            company_map[r["TICKER"]] = {
                "ticker": r["TICKER"],
                "market_cap": r["MARKET_CAP"],
                "pe_ratio": r["PE_RATIO"],
                "profit_margin": r["PROFIT_MARGIN"],
                "debt_to_equity": r["DEBT_TO_EQUITY"],
                "market_cap_category": r["MARKET_CAP_CATEGORY"],
            }
    except Exception as e:
        logger.warning("Could not query DIM_COMPANY for peers: %s", e)
        company_map = {}

    # Query latest fundamentals for all tickers
    try:
        rows = session.sql(f"""
            SELECT TICKER, FISCAL_QUARTER, REVENUE, NET_INCOME, EPS, NET_MARGIN_PCT
            FROM ANALYTICS.FCT_FUNDAMENTALS_GROWTH
            WHERE TICKER IN ({ticker_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY FISCAL_QUARTER DESC) = 1
        """).collect()

        for r in rows:
            t = r["TICKER"]
            if t in company_map:
                company_map[t]["revenue"] = r["REVENUE"]
                company_map[t]["net_income"] = r["NET_INCOME"]
                company_map[t]["eps"] = r["EPS"]
                company_map[t]["net_margin"] = r["NET_MARGIN_PCT"]
                company_map[t]["latest_quarter"] = r["FISCAL_QUARTER"]
    except Exception as e:
        logger.warning("Could not query FCT_FUNDAMENTALS_GROWTH for peers: %s", e)

    # Override D/E for ALL tickers with SEC-derived ratio for consistency
    try:
        sec_rows = session.sql(f"""
            SELECT TICKER, DEBT_TO_EQUITY_RATIO
            FROM ANALYTICS.FCT_SEC_FINANCIAL_SUMMARY
            WHERE TICKER IN ({ticker_list})
              AND REPORTING_FREQUENCY = 'quarterly'
              AND DEBT_TO_EQUITY_RATIO IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY TICKER ORDER BY FISCAL_YEAR DESC, FISCAL_PERIOD DESC
            ) = 1
        """).collect()
        for r in sec_rows:
            t = r["TICKER"]
            if t in company_map and r["DEBT_TO_EQUITY_RATIO"] is not None:
                company_map[t]["debt_to_equity"] = float(r["DEBT_TO_EQUITY_RATIO"])
    except Exception as e:
        logger.warning("Could not query FCT_SEC_FINANCIAL_SUMMARY for D/E override: %s", e)

    # Clamp D/E outliers: values beyond ±20 are likely data issues
    for t, data in company_map.items():
        de = data.get("debt_to_equity")
        if de is not None and abs(de) > 20:
            logger.warning("Clamping extreme D/E for %s: %.2f → None", t, de)
            data["debt_to_equity"] = None

    # Build peers list (target first, then peers) — exclude peers with no data
    excluded_peers = []
    for t in all_tickers:
        if t in company_map:
            p = company_map[t]
            if t != safe_ticker and not any(
                p.get(k) is not None for k in ("market_cap", "pe_ratio", "net_margin", "eps")
            ):
                excluded_peers.append(t)
                continue
            peers_data.append(p)
        else:
            if t == safe_ticker:
                peers_data.append({"ticker": t, "market_cap": None, "pe_ratio": None,
                                   "profit_margin": None, "debt_to_equity": None,
                                   "revenue": None, "eps": None, "net_margin": None})
            else:
                excluded_peers.append(t)

    # Generate comparison summary via Cortex
    def _fmt_mc(v):
        if not v or not isinstance(v, (int, float)):
            return "N/A"
        if v >= 1e12:
            return f"${v/1e12:.1f}T"
        if v >= 1e9:
            return f"${v/1e9:.0f}B"
        return f"${v/1e6:.0f}M"

    # Check if we have any actual peers (not just the target ticker)
    actual_peers = [p for p in peers_data if p["ticker"] != safe_ticker]

    if not actual_peers:
        logger.warning(
            "No peer data available for %s — all peers excluded: %s",
            safe_ticker, excluded_peers,
        )
        summary = (
            f"Peer comparison is not available for {safe_ticker}. "
            f"Peer tickers ({', '.join(excluded_peers)}) are not currently tracked "
            f"in the data warehouse. Add them to the tracked tickers list and run "
            f"the data pipeline to enable peer comparison analysis."
        )
    else:
        comparison_lines = []
        for p in peers_data:
            mc = _fmt_mc(p.get("market_cap"))
            pe = f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") else "N/A"
            margin = p.get("net_margin") or p.get("profit_margin")
            m_str = f"{float(margin)*100:.1f}%" if margin and float(margin) < 1 else (
                f"{float(margin):.1f}%" if margin else "N/A"
            )
            eps = f"${p['eps']:.2f}" if p.get("eps") else "N/A"
            comparison_lines.append(
                f"{p['ticker']}: Market Cap {mc}, P/E {pe}, Net Margin {m_str}, EPS {eps}"
            )

        # Determine factual market-cap ranking to ground LLM claims
        _mc_ranked = sorted(
            [(p["ticker"], p.get("market_cap") or 0) for p in peers_data],
            key=lambda x: x[1], reverse=True,
        )
        _mc_leader = _mc_ranked[0] if _mc_ranked else None
        _mc_note = ""
        if _mc_leader and _mc_leader[1] > 0:
            _mc_note = (
                f"\n\nIMPORTANT: Based on the data above, {_mc_leader[0]} has the largest "
                f"market cap ({_fmt_mc(_mc_leader[1])}) in this peer group. Do NOT claim "
                f"any other company is 'the largest' or 'the most valuable' unless the "
                f"data supports it. Avoid unsupported superlatives.\n"
            )

        if detail_level == "summary":
            prompt = (
                f"You are a senior equity research analyst. "
                f"Compare {safe_ticker} against its peer group in 1-2 sentences. "
                f"Highlight the most notable relative strength or weakness.\n\n"
                f"Peer Group Data:\n" + "\n".join(comparison_lines) + _mc_note + "\n\n"
                f"Write in third person, professional tone. Do not use bullet points."
            )
        else:
            prompt = (
                f"You are a senior equity research analyst writing a peer comparison section. "
                f"Compare {safe_ticker} against its peer group and write a concise 3-4 sentence "
                f"comparative analysis. Highlight relative strengths and weaknesses in valuation, "
                f"profitability, and scale. Be specific with numbers.\n\n"
                f"Peer Group Data:\n" + "\n".join(comparison_lines) + _mc_note + "\n\n"
                f"Write in third person, professional tone. Do not use bullet points."
            )

        summary = _cortex_complete(session, prompt)
        if not summary:
            summary = f"Peer comparison summary not available for {ticker}."
        else:
            summary = _validate_with_guardrails(summary, label="peer_comparison")

    result = {
        "ticker": ticker,
        "peers": peers_data,
        "comparison_summary": summary,
        "excluded_peers": excluded_peers,
        "excluded_reason": (
            f"Peers {', '.join(excluded_peers)} are not in the tracked tickers list "
            f"and have no warehouse data. Add them to config/tickers.yaml and run "
            f"the data pipeline to enable peer comparison."
        ) if excluded_peers else "",
    }
    logger.info("Peer comparison generated for %s (%d peers)", ticker, len(peers_data) - 1)
    return result


# ──────────────────────────────────────────────────────────────
# Financial Deep Dive (called by orchestrator)
# ──────────────────────────────────────────────────────────────

def generate_financial_deep_dive(session: Session, ticker: str) -> dict:
    """
    Generate a Financial Deep Dive section with multi-quarter trend analysis.

    Queries FCT_SEC_FINANCIAL_SUMMARY for balance sheet, income statement,
    and cash position data across quarters, then uses Cortex to produce
    narrative commentary on trends.

    Returns:
        {
            "quarterly_data": list[dict],  # rows of quarterly financials
            "narrative": str,              # LLM-generated trend analysis
            "balance_sheet_summary": str,  # LLM commentary on balance sheet
        }
    """
    logger.info("Generating financial deep dive for %s", ticker)

    # Query multi-quarter SEC financials
    query = f"""
    SELECT FISCAL_YEAR, FISCAL_PERIOD, TOTAL_REVENUE, NET_INCOME, OPERATING_INCOME,
           TOTAL_ASSETS, TOTAL_LIABILITIES, STOCKHOLDERS_EQUITY,
           CASH_AND_EQUIVALENTS, NET_MARGIN_PCT, OPERATING_MARGIN_PCT, RETURN_ON_EQUITY_PCT, RETURN_ON_ASSETS_PCT,
           DEBT_TO_EQUITY_RATIO, REVENUE_GROWTH_YOY_PCT, NET_INCOME_GROWTH_YOY_PCT
    FROM FINSAGE_DB.ANALYTICS.FCT_SEC_FINANCIAL_SUMMARY
    WHERE TICKER = '{ticker}'
    ORDER BY FISCAL_YEAR DESC, FISCAL_PERIOD DESC
    LIMIT 8
    """
    try:
        df = session.sql(query).to_pandas()
    except Exception as e:
        logger.warning("Financial deep dive query failed for %s: %s", ticker, e)
        df = None

    quarterly_data = []
    data_text = "No quarterly financial data available."

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            qd = {}
            for col in df.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                qd[col.lower()] = val
            quarterly_data.append(qd)

        # Format data for LLM prompt
        lines = []
        for q in quarterly_data[:6]:
            period = f"{q.get('fiscal_period', 'N/A')} {int(q.get('fiscal_year', 0))}"
            rev = q.get('revenue')
            ni = q.get('net_income')
            oi = q.get('operating_income')
            ta = q.get('total_assets')
            tl = q.get('total_liabilities')
            cash = q.get('cash_and_equivalents')
            roe = q.get('roe')
            de = q.get('debt_to_equity')
            rev_str = f"${rev/1e9:.2f}B" if rev and rev > 1e6 else "N/A"
            ni_str = f"${ni/1e9:.2f}B" if ni and ni > 1e6 else "N/A"
            oi_str = f"${oi/1e9:.2f}B" if oi and oi > 1e6 else "N/A"
            ta_str = f"${ta/1e9:.1f}B" if ta and ta > 1e6 else "N/A"
            tl_str = f"${tl/1e9:.1f}B" if tl and tl > 1e6 else "N/A"
            cash_str = f"${cash/1e9:.1f}B" if cash and cash > 1e6 else "N/A"
            roe_str = f"{roe:.1f}%" if roe else "N/A"
            de_str = f"{de:.2f}" if de else "N/A"
            lines.append(
                f"{period}: Revenue {rev_str}, Net Income {ni_str}, Operating Income {oi_str}, "
                f"Total Assets {ta_str}, Total Liabilities {tl_str}, Cash {cash_str}, "
                f"ROE {roe_str}, D/E {de_str}"
            )
        data_text = "\n".join(lines)

    # Generate narrative analysis
    narrative_prompt = (
        f"You are a senior equity research analyst. Analyze the following quarterly financial "
        f"data for {ticker} and write a 3-4 paragraph analysis covering:\n"
        f"1. Revenue and profitability trends over recent quarters\n"
        f"2. Operating efficiency (operating margin trends)\n"
        f"3. Key observations and inflection points\n\n"
        f"Data:\n{data_text}\n\n"
        f"Write in professional third-person tone suitable for an institutional research report. "
        f"Reference specific numbers and quarters. Do not use bullet points or headers."
    )
    narrative = _cortex_complete(session, narrative_prompt)
    if not narrative:
        narrative = "Financial trend analysis not available."
    else:
        narrative = _validate_with_guardrails(narrative, label="financial_deep_dive")

    # Generate balance sheet commentary
    bs_prompt = (
        f"Based on this financial data for {ticker}, write a concise 2-paragraph assessment "
        f"of the company's balance sheet strength and cash position. Cover:\n"
        f"1. Asset/liability composition and leverage (debt-to-equity trends)\n"
        f"2. Cash position adequacy and capital allocation capacity\n\n"
        f"Data:\n{data_text}\n\n"
        f"Write in professional tone. Reference specific figures."
    )
    balance_sheet_summary = _cortex_complete(session, bs_prompt)
    if not balance_sheet_summary:
        balance_sheet_summary = "Balance sheet analysis not available."
    else:
        balance_sheet_summary = _validate_with_guardrails(
            balance_sheet_summary, label="balance_sheet_summary"
        )

    logger.info("Financial deep dive generated for %s (%d quarters)", ticker, len(quarterly_data))
    return {
        "quarterly_data": quarterly_data,
        "narrative": narrative,
        "balance_sheet_summary": balance_sheet_summary,
    }


# ──────────────────────────────────────────────────────────────
# Valuation Analysis (called by orchestrator)
# ──────────────────────────────────────────────────────────────

def generate_valuation_analysis(session: Session, ticker: str) -> dict:
    """
    Generate a relative valuation analysis comparing the ticker's
    key valuation multiples against its peer group.

    Returns:
        {
            "ticker_metrics": dict,        # P/E, margin, D/E for target
            "peer_metrics": list[dict],     # same metrics for each peer
            "valuation_narrative": str,     # LLM-generated valuation commentary
        }
    """
    logger.info("Generating valuation analysis for %s", ticker)

    peers = resolve_peers(ticker)
    all_tickers = [ticker] + [p for p in peers if p != ticker]
    ticker_list = ", ".join(f"'{t}'" for t in all_tickers)

    query = f"""
    SELECT d.TICKER, d.MARKET_CAP, d.PE_RATIO, d.PROFIT_MARGIN, d.DEBT_TO_EQUITY,
           f.REVENUE, f.NET_INCOME, f.EPS, f.REVENUE_GROWTH_YOY_PCT, f.RETURN_ON_ASSETS_PCT
    FROM FINSAGE_DB.ANALYTICS.DIM_COMPANY d
    LEFT JOIN (
        SELECT TICKER, REVENUE, NET_INCOME, EPS, REVENUE_GROWTH_YOY_PCT, RETURN_ON_ASSETS_PCT,
               ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY FISCAL_QUARTER DESC) as rn
        FROM FINSAGE_DB.ANALYTICS.FCT_FUNDAMENTALS_GROWTH
    ) f ON d.TICKER = f.TICKER AND f.rn = 1
    WHERE d.TICKER IN ({ticker_list})
    """
    try:
        df = session.sql(query).to_pandas()
    except Exception as e:
        logger.warning("Valuation query failed for %s: %s", ticker, e)
        return {
            "ticker_metrics": {},
            "peer_metrics": [],
            "valuation_narrative": "Valuation analysis not available.",
        }

    ticker_metrics = {}
    peer_metrics = []

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            metrics = {}
            for col in df.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                metrics[col.lower()] = val
            if metrics.get("ticker") == ticker:
                ticker_metrics = metrics
            else:
                peer_metrics.append(metrics)

    # Format for LLM
    def fmt_metrics(m):
        t = m.get("ticker", "?")
        mc = m.get("market_cap")
        mc_s = f"${mc/1e12:.2f}T" if mc and mc > 1e12 else (f"${mc/1e9:.1f}B" if mc and mc > 1e9 else "N/A")
        pe = m.get("pe_ratio")
        pe_s = f"{pe:.1f}x" if pe else "N/A"
        pm = m.get("profit_margin")
        pm_s = f"{float(pm)*100:.1f}%" if pm and float(pm) < 1 else (f"{float(pm):.1f}%" if pm else "N/A")
        rg = m.get("revenue_growth_yoy")
        rg_s = f"{rg:.1f}%" if rg else "N/A"
        return f"{t}: Market Cap {mc_s}, P/E {pe_s}, Margin {pm_s}, Rev Growth {rg_s}"

    comp_lines = [fmt_metrics(ticker_metrics)] if ticker_metrics else []
    for pm in peer_metrics:
        comp_lines.append(fmt_metrics(pm))
    comp_text = "\n".join(comp_lines)

    val_prompt = (
        f"You are a financial data analyst. Write a 2-3 paragraph factual comparison "
        f"of {ticker}'s valuation multiples relative to its sector peers. Cover:\n"
        f"1. How {ticker}'s P/E ratio compares numerically to peers and possible reasons "
        f"for any differences (growth profile, margin structure, market position)\n"
        f"2. Whether {ticker}'s revenue growth rate is consistent with its multiple relative to peers\n"
        f"3. A factual summary of where {ticker} sits in the peer group on each metric\n\n"
        f"Comparative data:\n{comp_text}\n\n"
        f"IMPORTANT: This is a factual data comparison only. Do NOT provide investment advice, "
        f"price targets, buy/sell/hold recommendations, or predict future price movements. "
        f"Stick to describing the data and observable differences. "
        f"Write in professional analytical tone. Reference specific multiples and peers by name."
    )
    valuation_narrative = _cortex_complete(session, val_prompt)
    if not valuation_narrative:
        valuation_narrative = "Valuation analysis not available."
    else:
        valuation_narrative = _validate_with_guardrails(
            valuation_narrative, label="valuation_analysis"
        )

    logger.info("Valuation analysis generated for %s vs %d peers", ticker, len(peer_metrics))
    return {
        "ticker_metrics": ticker_metrics,
        "peer_metrics": peer_metrics,
        "valuation_narrative": valuation_narrative,
    }


# ──────────────────────────────────────────────────────────────
# Main entry point (called by orchestrator)
# ──────────────────────────────────────────────────────────────
def run_analysis(session: Session, charts: list, ticker: str,
                 detail_level: str = "detailed") -> dict:
    """
    Full analysis pipeline:
        1. Generate written analysis for each validated chart (with CoA + KB enrichment)
        2. Summarize SEC filing MD&A and Risk Factors (KB RAG + Cortex)
        3. Synthesize investment thesis (multi-model or single Cortex)
        4. All outputs validated by Guardrails
    """
    logger.info("=" * 50)
    logger.info("Analysis Agent starting for %s (detail_level=%s)", ticker, detail_level)
    logger.info("=" * 50)

    # Log which Bedrock integrations are active
    if _get_guardrail():
        logger.info("  Guardrails: ACTIVE")
    if _get_bedrock_kb():
        logger.info("  Bedrock KB RAG: ACTIVE")
    if _get_multi_model():
        logger.info("  Multi-Model Consensus: ACTIVE")

    chart_analyses = analyze_all_charts(session, charts, ticker,
                                        detail_level=detail_level)
    sec_summaries = summarize_sec_filings(session, ticker)
    investment_thesis = synthesize_analyses(session, chart_analyses, ticker,
                                            detail_level=detail_level)

    result = {
        "chart_analyses": chart_analyses,
        "mda_summary": sec_summaries["mda_summary"],
        "risk_summary": sec_summaries["risk_summary"],
        "investment_thesis": investment_thesis,
    }

    logger.info("Analysis Agent complete for %s", ticker)
    return result


# ──────────────────────────────────────────────────────────────
# Dev test (uses mock data — remove before production)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    import logging

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

    from snowflake_connection import get_session
    from mock_data import MOCK_CHARTS

    session = get_session()
    result = run_analysis(session, MOCK_CHARTS, "AAPL")

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    for a in result["chart_analyses"]:
        print(f"\n{a['title']}")
        print(f"{a['analysis_text']}")
        print("-" * 40)

    print(f"\nMD&A Summary:\n{result['mda_summary']}")
    print(f"\nRisk Summary:\n{result['risk_summary']}")
    print(f"\nInvestment Thesis:\n{result['investment_thesis']}")

    session.close()

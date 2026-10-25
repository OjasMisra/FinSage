"""Tests for the per-report guardrail audit and its appendix disclosure."""

import contextvars
import io
import threading
from unittest.mock import MagicMock, patch

import pytest
from reportlab.platypus import SimpleDocTemplate, Table

import analysis_agent
from analysis_agent import (
    GUARDRAIL_FALLBACK,
    _validate_with_guardrails,
    start_guardrail_audit,
)
from report_agent import build_guardrail_disclosure, build_styles


def _guard_returning(result):
    guard = MagicMock()
    guard.check_output.return_value = result
    return guard


@pytest.fixture
def fresh_context():
    """Run each test in its own context so audits never leak between tests."""
    return contextvars.copy_context()


class TestValidateWithGuardrails:

    def _run(self, fresh_context, guard, text="some analysis", label="mda_summary"):
        def body():
            audit = start_guardrail_audit()
            with patch.object(analysis_agent, "_get_guardrail", return_value=guard):
                out = _validate_with_guardrails(text, label=label)
            return out, audit.summary()
        return fresh_context.run(body)

    def test_passed(self, fresh_context):
        out, summary = self._run(fresh_context, _guard_returning({"action": "NONE", "blocked": False}))
        assert out == "some analysis"
        assert summary["passed"] == 1 and summary["skipped"] == 0

    def test_blocked_returns_fallback(self, fresh_context):
        out, summary = self._run(fresh_context, _guard_returning({"action": "GUARDRAIL_INTERVENED", "blocked": True}))
        assert out == GUARDRAIL_FALLBACK
        assert summary["blocked"] == 1
        assert summary["blocked_sections"] == ["mda_summary"]

    def test_service_error_is_recorded_as_skipped(self, fresh_context):
        # check_output() swallows exceptions and reports action="ERROR", blocked=False
        out, summary = self._run(fresh_context, _guard_returning({"action": "ERROR", "blocked": False}))
        assert out == "some analysis"
        assert summary["passed"] == 0
        assert summary["skipped_sections"] == ["mda_summary"]

    def test_exception_is_recorded_as_skipped(self, fresh_context):
        guard = MagicMock()
        guard.check_output.side_effect = RuntimeError("timeout")
        out, summary = self._run(fresh_context, guard)
        assert out == "some analysis"
        assert summary["skipped"] == 1

    def test_guardrails_not_configured_is_recorded_as_skipped(self, fresh_context):
        out, summary = self._run(fresh_context, None)
        assert out == "some analysis"
        assert summary["skipped_sections"] == ["mda_summary"]

    def test_no_active_audit_still_validates(self, fresh_context):
        guard = _guard_returning({"action": "NONE", "blocked": False})

        def body():
            with patch.object(analysis_agent, "_get_guardrail", return_value=guard):
                return _validate_with_guardrails("text", label="x")
        assert fresh_context.run(body) == "text"


class TestAuditAcrossThreads:

    def test_copied_context_records_into_same_audit(self, fresh_context):
        guard = _guard_returning({"action": "ERROR", "blocked": False})

        def body():
            audit = start_guardrail_audit()
            with patch.object(analysis_agent, "_get_guardrail", return_value=guard):
                workers = [
                    threading.Thread(
                        target=contextvars.copy_context().run,
                        args=(_validate_with_guardrails, "t", f"section_{i}"),
                    )
                    for i in range(4)
                ]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
            return audit.summary()

        summary = fresh_context.run(body)
        assert summary["skipped"] == 4
        assert len(summary["skipped_sections"]) == 4

    def test_separate_runs_do_not_share_an_audit(self):
        guard = _guard_returning({"action": "NONE", "blocked": False})

        def one_report():
            audit = start_guardrail_audit()
            with patch.object(analysis_agent, "_get_guardrail", return_value=guard):
                _validate_with_guardrails("t", label="a")
            return audit.summary()["passed"]

        assert contextvars.copy_context().run(one_report) == 1
        assert contextvars.copy_context().run(one_report) == 1


class TestGuardrailDisclosure:

    def _render(self, elements):
        doc = SimpleDocTemplate(io.BytesIO())
        doc.build(list(elements))  # raises if any flowable is malformed

    def _texts(self, elements):
        out = []
        for e in elements:
            if isinstance(e, Table):
                out.extend(str(cell) for row in e._cellvalues for cell in row)
            elif hasattr(e, "text"):
                out.append(e.text)
        return " ".join(out)

    def test_skipped_sections_are_named_with_a_warning(self):
        audit = {"passed": 7, "blocked": 0, "skipped": 2,
                 "blocked_sections": [], "skipped_sections": ["mda_summary", "risk_summary"]}
        elements = build_guardrail_disclosure(audit, build_styles(), 400)
        text = self._texts(elements)
        assert "MD&A Summary, Risk Summary" in text
        assert "without content-safety validation" in text
        self._render(elements)

    def test_clean_run_has_no_warning(self):
        audit = {"passed": 9, "blocked": 0, "skipped": 0,
                 "blocked_sections": [], "skipped_sections": []}
        elements = build_guardrail_disclosure(audit, build_styles(), 400)
        assert "without content-safety validation" not in self._texts(elements)
        self._render(elements)

    def test_missing_audit_says_so(self):
        elements = build_guardrail_disclosure(None, build_styles(), 400)
        assert "not recorded" in self._texts(elements)
        self._render(elements)

"""
Context Manager for Test Orchestration
Handles rolling summaries and maintains execution context
"""

import json
import time
from typing import Dict, List, Optional, Any
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestStepResult:
    """Result of a test step execution"""
    step_id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    mcp_response: Dict[str, Any]
    snapshot_data: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    passed: Optional[bool]
    execution_time: float
    timestamp: str
    error: Optional[str] = None


@dataclass
class TestSession:
    """Complete test session information"""
    session_id: str
    test_objective: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    inconclusive_steps: int = 0
    status: str = "running"  # running, completed, failed, aborted


class ContextManager:
    """Manages test execution context with rolling summaries"""
    
    def __init__(self, max_context_items: int = 20, summary_threshold: int = 15):
        self.max_context_items = max_context_items
        self.summary_threshold = summary_threshold
        
        # Current session
        self.current_session: Optional[TestSession] = None
        
        # Step results (recent ones kept in full detail)
        self.step_results: List[TestStepResult] = []
        
        # Rolling summary of older steps
        self.context_summary = ""
        
        # Key insights and patterns
        self.insights: List[str] = []
        
        # Current test state
        self.current_state: Dict[str, Any] = {}
        
        # Lightweight DOM/page-state cache
        self.dom_cache: Dict[str, Any] = {
            "url": None,
            "title": None,
            "snapshot_text": None,
            "snapshot_hash": None,
            "last_updated": None,
        }
        
    def start_session(self, test_objective: str) -> str:
        """Start a new test session"""
        session_id = f"test_{int(time.time())}"
        
        self.current_session = TestSession(
            session_id=session_id,
            test_objective=test_objective,
            start_time=datetime.now().isoformat()
        )
        
        # Reset context for new session
        self.step_results = []
        self.context_summary = ""
        self.insights = []
        self.current_state = {
            "session_id": session_id,
            "objective": test_objective,
            "started_at": self.current_session.start_time
        }
        
        logger.info(f"Started test session: {session_id}")
        return session_id
    
    def add_step_result(self, step_result: TestStepResult):
        """Add a test step result to the context"""
        if not self.current_session:
            raise RuntimeError("No active test session")
        
        # Add to results
        self.step_results.append(step_result)
        
        # Update session statistics
        self.current_session.total_steps += 1
        if step_result.passed is True:
            self.current_session.passed_steps += 1
        elif step_result.passed is False:
            self.current_session.failed_steps += 1
        else:
            self.current_session.inconclusive_steps += 1
        
        # Update current state
        self.current_state.update({
            "last_step_id": step_result.step_id,
            "last_action": step_result.action_type,
            "last_result": "passed" if step_result.passed else "failed" if step_result.passed is False else "inconclusive",
            "total_steps": self.current_session.total_steps,
            "passed_steps": self.current_session.passed_steps,
            "failed_steps": self.current_session.failed_steps
        })
        
        # Manage context length
        self._manage_context_length()
        
        # Extract insights
        self._extract_insights(step_result)
        
        logger.info(f"Added step result: {step_result.step_id} - {step_result.passed}")
    
    def _manage_context_length(self):
        """Manage context length with rolling summary"""
        if len(self.step_results) > self.max_context_items:
            # Move older results to summary
            old_results = self.step_results[:-self.summary_threshold]
            self.step_results = self.step_results[-self.summary_threshold:]
            
            # Create summary of old results
            if old_results:
                summary = self._create_summary(old_results)
                if self.context_summary:
                    self.context_summary = f"{self.context_summary}\n\n{summary}"
                else:
                    self.context_summary = summary
                
                logger.debug(f"Created rolling summary for {len(old_results)} old results")
    
    def _create_summary(self, results: List[TestStepResult]) -> str:
        """Create a summary of test step results"""
        if not results:
            return ""
        
        # Basic statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed is True)
        failed = sum(1 for r in results if r.passed is False)
        inconclusive = total - passed - failed
        
        summary = f"Summary of {total} previous steps:\n"
        summary += f"- Passed: {passed}\n"
        summary += f"- Failed: {failed}\n"
        summary += f"- Inconclusive: {inconclusive}\n"
        
        # Key actions performed
        actions = {}
        for result in results:
            action = result.action_type
            actions[action] = actions.get(action, 0) + 1
        
        summary += f"- Actions performed: {dict(actions)}\n"
        
        # Notable failures or issues
        failures = [r for r in results if r.passed is False]
        if failures:
            summary += f"- Notable failures:\n"
            for failure in failures[-3:]:  # Last 3 failures
                summary += f"  * {failure.step_id}: {failure.description}\n"
        
        # Time range
        start_time = results[0].timestamp
        end_time = results[-1].timestamp
        summary += f"- Time range: {start_time} to {end_time}\n"
        
        return summary
    
    def _extract_insights(self, step_result: TestStepResult):
        """Extract insights from step results"""
        # Pattern detection
        if step_result.passed is False:
            error_pattern = f"Failed {step_result.action_type} with {step_result.parameters}"
            if error_pattern not in self.insights:
                self.insights.append(f"Error pattern: {error_pattern}")
        
        # Performance insights
        if step_result.execution_time > 10:  # Slow execution
            self.insights.append(f"Slow execution: {step_result.step_id} took {step_result.execution_time:.2f}s")
        
        # Keep only recent insights
        if len(self.insights) > 10:
            self.insights = self.insights[-10:]
    
    def get_current_context(self) -> str:
        """Get current context for LLM analysis"""
        if not self.current_session:
            return "No active test session"
        
        context = f"Test Session: {self.current_session.session_id}\n"
        context += f"Objective: {self.current_session.test_objective}\n"
        context += f"Status: {self.current_session.status}\n"
        context += f"Progress: {self.current_session.total_steps} steps "
        context += f"({self.current_session.passed_steps} passed, {self.current_session.failed_steps} failed)\n\n"
        
        # Add rolling summary if available
        if self.context_summary:
            context += f"Previous Context:\n{self.context_summary}\n\n"
        
        # Add recent step results
        if self.step_results:
            context += "Recent Steps:\n"
            for result in self.step_results[-5:]:  # Last 5 steps
                status = "✅" if result.passed else "❌" if result.passed is False else "❓"
                context += f"{status} {result.step_id}: {result.description}\n"
                if result.error:
                    context += f"   Error: {result.error}\n"
        
        # Add insights
        if self.insights:
            context += f"\nKey Insights:\n"
            for insight in self.insights[-5:]:  # Last 5 insights
                context += f"- {insight}\n"
        
        # Add current state
        context += f"\nCurrent State:\n"
        for key, value in self.current_state.items():
            context += f"- {key}: {value}\n"
        
        return context
    
    def get_step_history(self, limit: int = 10) -> List[TestStepResult]:
        """Get recent step history"""
        return self.step_results[-limit:] if self.step_results else []
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if not self.current_session:
            return {}
        
        return {
            "session": asdict(self.current_session),
            "recent_steps": len(self.step_results),
            "context_summary_length": len(self.context_summary),
            "insights_count": len(self.insights),
            "current_state": self.current_state
        }
    
    def end_session(self, status: str = "completed"):
        """End the current test session"""
        if not self.current_session:
            return
        
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.status = status
        
        logger.info(f"Ended test session: {self.current_session.session_id} with status: {status}")
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data"""
        return {
            "session": asdict(self.current_session) if self.current_session else None,
            "step_results": [asdict(result) for result in self.step_results],
            "context_summary": self.context_summary,
            "insights": self.insights,
            "current_state": self.current_state,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def save_to_file(self, filepath: str):
        """Save session data to file"""
        data = self.export_session_data()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session data saved to: {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load session data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore session
        if data.get("session"):
            self.current_session = TestSession(**data["session"])
        
        # Restore step results
        self.step_results = [TestStepResult(**result) for result in data.get("step_results", [])]
        
        # Restore other data
        self.context_summary = data.get("context_summary", "")
        self.insights = data.get("insights", [])
        self.current_state = data.get("current_state", {})
        
        logger.info(f"Session data loaded from: {filepath}")
    
    def reset(self):
        """Reset all context data"""
        self.current_session = None
        self.step_results = []
        self.context_summary = ""
        self.insights = []
        self.current_state = {}
        logger.info("Context manager reset") 

    # ---------------------------
    # DOM Cache helpers
    # ---------------------------
    def update_dom_cache(self, url: Optional[str], title: Optional[str], snapshot_text: Optional[str]) -> None:
        """Update cached page state with latest URL/title/snapshot text."""
        try:
            has_changed = False
            if snapshot_text and snapshot_text != self.dom_cache.get("snapshot_text"):
                snap_hash = hashlib.md5(snapshot_text.encode("utf-8", errors="ignore")).hexdigest()
                self.dom_cache["snapshot_text"] = snapshot_text
                self.dom_cache["snapshot_hash"] = snap_hash
                has_changed = True
            if url and url != self.dom_cache.get("url"):
                self.dom_cache["url"] = url
                has_changed = True
            if title and title != self.dom_cache.get("title"):
                self.dom_cache["title"] = title
                has_changed = True
            if has_changed:
                self.dom_cache["last_updated"] = datetime.now().isoformat()
        except Exception:
            pass

    def get_minimal_dom_context(self, max_lines: int = 80) -> str:
        """Return a compact textual context from cached DOM."""
        url = self.dom_cache.get("url") or ""
        title = self.dom_cache.get("title") or ""
        text = self.dom_cache.get("snapshot_text") or ""
        header: List[str] = []
        if url:
            header.append(f"Page URL: {url}")
        if title:
            header.append(f"Page Title: {title}")
        if not text:
            return "\n".join(header)
        lines = text.splitlines()
        preferred: List[str] = []
        roles = (
            "heading",
            "textbox",
            "button",
            "link",
            "contentinfo",
            "banner",
            "dialog",
            "alert",
            "modal",
            "overlay",
            "menu",
            "listbox",
            "combobox",
        )
        for ln in lines:
            if "ref=" in ln and any(r in ln for r in roles):
                preferred.append(ln)
        chosen = preferred if preferred else lines
        chosen = chosen[:max_lines]
        header.append("Page Snapshot (compact):")
        return "\n".join(header + chosen)

    def get_full_dom_snapshot(self, max_chars: int = 8000) -> str:
        """Return the full cached snapshot text (trimmed), including URL and title.

        Useful as a fallback when minimal context is insufficient (e.g., modals/popups).
        """
        url = self.dom_cache.get("url") or ""
        title = self.dom_cache.get("title") or ""
        text = self.dom_cache.get("snapshot_text") or ""
        header: List[str] = []
        if url:
            header.append(f"Page URL: {url}")
        if title:
            header.append(f"Page Title: {title}")
        if text and len(text) > max_chars:
            text = text[:max_chars] + "\n..."
        if text:
            header.append("Page Snapshot:")
            header.append(text)
        return "\n".join(header)


# ---------------------------
# Structured session log recorder
# ---------------------------

class SessionLogRecorder:
    """Record structured events for an orchestrator session and persist to JSON.

    The recorder writes a new JSON file per session under mcp-client/reports/ by default.
    """

    def __init__(self, reports_dir: Optional[str] = None, report_file: Optional[str] = None):
        from pathlib import Path
        import os

        base_dir = Path(reports_dir) if reports_dir else Path(__file__).parent.parent / "reports"
        base_dir.mkdir(parents=True, exist_ok=True)

        if report_file is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"session-{ts}.json"
            self.report_path = base_dir / filename
        else:
            # Expand environment variables and user home, then anchor bare filenames to base_dir
            expanded = os.path.expandvars(str(report_file))
            candidate_path = Path(expanded).expanduser()
            if not candidate_path.is_absolute() and candidate_path.parent == Path("."):
                # Only a filename provided: write under dynamic reports directory
                self.report_path = base_dir / candidate_path.name
            else:
                # Respect absolute paths or relative paths with directories
                self.report_path = candidate_path

        # Ensure parent directory exists
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.meta: Dict[str, Any] = {"created_at": datetime.now().isoformat()}
        self.events: List[Dict[str, Any]] = []
        self.final_report: Optional[Dict[str, Any]] = None
        # Token and cost tracking
        self.token_totals: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.cost_totals_usd: float = 0.0

    # ----- Session lifecycle -----
    def start_session(self, session_id: str, test_objective: str, extra: Optional[Dict[str, Any]] = None) -> None:
        import os
        self.meta.update({
            "session_id": session_id,
            "test_objective": test_objective,
            "started_at": datetime.now().isoformat(),
        })
        if extra:
            self.meta.update(extra)
        self.events.append({
            "ts": datetime.now().isoformat(),
            "type": "session_start",
            "data": {
                "session_id": session_id,
                "test_objective": test_objective,
                "env": {k: os.getenv(k) for k in ["MCP_SERVER_URL", "LLM_MODEL"] if os.getenv(k)}
            }
        })

    def end_session(self, status: str) -> None:
        self.meta.update({"ended_at": datetime.now().isoformat(), "status": status})
        self.events.append({"ts": datetime.now().isoformat(), "type": "session_end", "data": {"status": status}})

    def attach_final_report(self, report: Dict[str, Any]) -> None:
        self.final_report = report
        # Also surface high-signal context into meta to reduce duplication in final_report
        try:
            if isinstance(report, dict):
                if "final_context" in report:
                    self.meta["final_context"] = report["final_context"]
                if "execution_summary" in report:
                    self.meta["execution_summary"] = report["execution_summary"]
                if "insights" in report:
                    self.meta["insights"] = report["insights"]
        except Exception:
            pass

    # ----- LLM -----
    def record_llm_call(self, phase: str, provider: str, model: str, messages: List[Dict[str, str]], response_text: str, usage: Optional[Dict[str, Any]] = None) -> None:
        event: Dict[str, Any] = {
            "ts": datetime.now().isoformat(),
            "type": "llm_call",
            "phase": phase,
            "provider": provider,
            "model": model,
            "request": messages,
            "response": response_text,
        }
        if usage is not None:
            event["usage"] = usage
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = int(usage.get("total_tokens", prompt + completion))
            self.token_totals["prompt_tokens"] += prompt
            self.token_totals["completion_tokens"] += completion
            self.token_totals["total_tokens"] += total
            cost = self._estimate_cost_usd(provider, model, prompt, completion)
            event["estimated_cost_usd"] = cost
            self.cost_totals_usd += cost
        self.events.append(event)

    def _estimate_cost_usd(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        try:
            provider = (provider or "").lower()
            model = (model or "").lower()
            if provider == "openai":
                pricing_per_million = {
                    "gpt-4o": {"input": 5.0, "output": 15.0},
                    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
                    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                }
                matched = None
                for key in pricing_per_million.keys():
                    if key in model:
                        matched = pricing_per_million[key]
                        break
                if matched is None and model in pricing_per_million:
                    matched = pricing_per_million[model]
                if matched is None:
                    return 0.0
                input_rate = matched["input"] / 1_000_000.0
                output_rate = matched["output"] / 1_000_000.0
                return prompt_tokens * input_rate + completion_tokens * output_rate
            if provider == "anthropic":
                input_rate = 3.0 / 1_000_000.0
                output_rate = 15.0 / 1_000_000.0
                return prompt_tokens * input_rate + completion_tokens * output_rate
        except Exception:
            pass
        return 0.0

    # ----- JSON-RPC / MCP -----
    def record_json_rpc_request(self, request_id: str, method: str, params: Dict[str, Any], url: str, headers: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "ts": datetime.now().isoformat(),
            "type": "json_rpc_request",
            "id": request_id,
            "method": method,
            "params": params,
            "url": url,
            "headers": headers or {},
        })

    def record_json_rpc_response(self, request_id: str, status: int, headers: Dict[str, Any], result: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None, content_type: Optional[str] = None) -> None:
        self.events.append({
            "ts": datetime.now().isoformat(),
            "type": "json_rpc_response",
            "id": request_id,
            "status": status,
            "content_type": content_type,
            "headers": headers,
            "result": result,
            "error": error,
        })

    def record_sse_event(self, data: str) -> None:
        self.events.append({"ts": datetime.now().isoformat(), "type": "sse_event", "data": data})

    # ----- Tools / Steps -----
    def record_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        self.events.append({
            "ts": datetime.now().isoformat(),
            "type": "tool_call",
            "tool": tool_name,
            "parameters": parameters,
        })

    def record_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        self.events.append({
            "ts": datetime.now().isoformat(),
            "type": "snapshot",
            "data": snapshot_data,
        })

    def record_step_input(self, nl_prompt: str) -> None:
        self.events.append({"ts": datetime.now().isoformat(), "type": "step_input", "prompt": nl_prompt})

    def record_structured_step(self, step: Any) -> None:
        try:
            payload = asdict(step)
        except Exception:
            payload = step if isinstance(step, dict) else {"repr": repr(step)}
        self.events.append({"ts": datetime.now().isoformat(), "type": "structured_step", "data": payload})

    def record_step_result(self, step_result: Any) -> None:
        try:
            payload = asdict(step_result)
        except Exception:
            payload = step_result if isinstance(step_result, dict) else {"repr": repr(step_result)}
        self.events.append({"ts": datetime.now().isoformat(), "type": "step_result", "data": payload})

    # ----- Persistence -----
    def save(self) -> None:
        self.meta["usage_summary"] = {
            "prompt_tokens": self.token_totals["prompt_tokens"],
            "completion_tokens": self.token_totals["completion_tokens"],
            "total_tokens": self.token_totals["total_tokens"],
        }
        self.meta["cost_summary_usd"] = {
            "estimated_total_cost_usd": round(self.cost_totals_usd, 6),
        }
        doc: Dict[str, Any] = {
            "meta": self.meta,
            "events": self.events,
        }
        if self.final_report is not None:
            doc["final_report"] = self.final_report
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    @property
    def filepath(self):
        return self.report_path

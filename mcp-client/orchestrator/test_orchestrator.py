"""
Test Orchestrator - Main coordination system
Orchestrates LLM-driven test automation with MCP integration
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import base64
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from .llm_service import LLMService, TestStep, LLMResponse
from .mcp_http_client import MCPHTTPClient, create_mcp_http_client
from .context_manager import ContextManager, TestStepResult, SessionLogRecorder

logger = logging.getLogger(__name__)
console = Console()


class TestOrchestrator:
    """Main orchestrator for LLM-driven test automation"""
    
    def __init__(self, 
                 mcp_server_url: str,
                 openai_api_key: str,
                 llm_model: str = "gpt-4",
                 max_steps: int = 50,
                 step_timeout: int = 30,
                 record_session_logs: bool = True,
                 report_file: Optional[str] = None):
        
        self.mcp_server_url = mcp_server_url
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        
        # Initialize components
        self.session_recorder: Optional[SessionLogRecorder] = SessionLogRecorder(report_file=report_file) if record_session_logs else None
        self.llm_service = LLMService(api_key=openai_api_key, model=llm_model, recorder=self.session_recorder)
        self.context_manager = ContextManager()
        self.mcp_client: Optional[MCPHTTPClient] = None
        
        # State tracking
        self.is_running = False
        self.current_step = 0
        self.keep_alive_task: Optional[asyncio.Task] = None
        self.last_page_state: Optional[str] = None  # Store last browser page state
        
    def _extract_page_state(self, mcp_response: Dict[str, Any], snapshot_data: Dict[str, Any]):
        """Extract page state with element references from MCP response"""
        try:
            # Look for page state in mcp_response content first
            if 'content' in mcp_response:
                for content in mcp_response['content']:
                    if content.get('type') == 'text':
                        text = content.get('text', '')
                        # Look for page snapshot section
                        if 'Page Snapshot:' in text and 'ref=' in text:
                            # Extract the YAML section
                            lines = text.split('\n')
                            yaml_start = -1
                            yaml_end = -1
                            
                            for i, line in enumerate(lines):
                                if 'Page Snapshot:' in line:
                                    yaml_start = i + 1
                                elif yaml_start > -1 and line.strip() == '```' and '```yaml' in lines[yaml_start-1]:
                                    yaml_end = i
                                    break
                            
                            if yaml_start > -1 and yaml_end > -1:
                                yaml_content = '\n'.join(lines[yaml_start+1:yaml_end])
                                self.last_page_state = yaml_content
                                logger.debug(f"🎯 Extracted page state with {yaml_content.count('ref=')} element references")
                                return
            
            # Fallback: look in snapshot_data
            if snapshot_data and 'tool_result' in snapshot_data:
                tool_result = snapshot_data['tool_result']
                if 'content' in tool_result:
                    for content in tool_result['content']:
                        if content.get('type') == 'text':
                            text = content.get('text', '')
                            if 'ref=' in text:
                                self.last_page_state = text
                                logger.debug(f"🎯 Extracted page state from snapshot_data")
                                return
                                
        except Exception as e:
            logger.warning(f"Failed to extract page state: {e}")
            self.last_page_state = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize MCP client (attach recorder if available)
        self.mcp_client = await create_mcp_http_client(self.mcp_server_url)
        if isinstance(self.mcp_client, MCPHTTPClient) and self.session_recorder:
            self.mcp_client.recorder = self.session_recorder
        
        # Start keep-alive task
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Stop keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close MCP client
        if self.mcp_client:
            await self.mcp_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _keep_alive_loop(self):
        """Keep MCP connection alive"""
        try:
            while True:
                await asyncio.sleep(60)  # Every minute
                if self.mcp_client:
                    await self.mcp_client.keep_alive()
        except asyncio.CancelledError:
            logger.info("Keep-alive loop cancelled")

    def _write_screenshot_file(self, screenshot_result: Dict[str, Any], screenshot_path: Path) -> bool:
        """Best-effort: extract image data from tool result and write to file."""
        try:
            # Look for content entries
            if isinstance(screenshot_result, dict):
                content = screenshot_result.get("content") or screenshot_result.get("result", {}).get("content")
                if isinstance(content, list):
                    for entry in content:
                        if not isinstance(entry, dict):
                            continue
                        # Direct image payload
                        if entry.get("type") == "image":
                            data = entry.get("data") or entry.get("image")
                            if isinstance(data, str):
                                img_bytes = base64.b64decode(data)
                                screenshot_path.write_bytes(img_bytes)
                                return True
                        # Data URI embedded in text
                        if entry.get("type") == "text" and isinstance(entry.get("text"), str):
                            text = entry["text"]
                            marker = "data:image/"
                            if marker in text:
                                # naive extract after comma
                                b64_part = text.split("data:image/")[1]
                                if "," in b64_part:
                                    b64_data = b64_part.split(",", 1)[1].split("\n")[0].strip()
                                    try:
                                        img_bytes = base64.b64decode(b64_data)
                                        screenshot_path.write_bytes(img_bytes)
                                        return True
                                    except Exception:
                                        pass
            return False
        except Exception:
            return False
    
    async def run_test_sequence(self, 
                              test_objective: str,
                              initial_prompt: str,
                              pdf_path: Optional[str] = None,
                              interactive: bool = False) -> Dict[str, Any]:
        """
        Run a complete test sequence
        
        Args:
            test_objective: Overall goal of the test
            initial_prompt: First natural language test step
            pdf_path: Path to PDF file for testing (if applicable)
            interactive: Whether to pause for user input between steps
        """
        
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")
        
        # Start test session
        session_id = self.context_manager.start_session(test_objective)
        if self.session_recorder:
            try:
                self.session_recorder.start_session(session_id, test_objective)
            except Exception:
                pass
        self.is_running = True
        self.current_step = 0
        
        console.print(Panel(f"🚀 Starting Test Session: {session_id}", style="bold green"))
        console.print(f"📋 Objective: {test_objective}")
        console.print(f"🎯 Initial Step: {initial_prompt}")
        
        try:
            # Get available tools
            available_tools = self.mcp_client.get_tool_names()
            console.print(f"🔧 Available Tools: {', '.join(available_tools)}")
            
            # Start with initial prompt
            current_prompt = initial_prompt
            
            # Main test loop
            while self.is_running and self.current_step < self.max_steps and current_prompt:
                
                self.current_step += 1
                console.print(f"\n{'='*60}")
                console.print(f"📍 Step {self.current_step}: {current_prompt}")
                
                # Execute single test step
                step_result = await self._execute_test_step(
                    current_prompt, 
                    available_tools, 
                    pdf_path
                )
                
                # Add to context
                self.context_manager.add_step_result(step_result)
                
                # Display step result
                self._display_step_result(step_result)
                
                # Interactive pause
                if interactive:
                    user_input = console.input("\n⏸️  Press Enter to continue, 'q' to quit, or type next step: ")
                    if user_input.lower() == 'q':
                        break
                    elif user_input.strip():
                        current_prompt = user_input.strip()
                        continue
                
                # Generate next step using LLM
                current_context = self.context_manager.get_current_context()
                next_prompt = await self.llm_service.generate_next_step(
                    current_context, 
                    test_objective
                )
                
                if next_prompt:
                    current_prompt = next_prompt
                    console.print(f"🤖 LLM suggests next step: {next_prompt}")
                else:
                    console.print("✅ Test sequence completed by LLM")
                    break
                
                # Small delay between steps
                await asyncio.sleep(1)
            
            # End session
            self.context_manager.end_session("completed")
            
            # Generate final report
            final_report = self._generate_final_report()
            if self.session_recorder:
                try:
                    self.session_recorder.attach_final_report(final_report)
                    self.session_recorder.end_session("completed")
                    self.session_recorder.save()
                    console.print(f"📝 Session log written to: {self.session_recorder.filepath}")
                except Exception:
                    pass
            
            console.print(Panel("🎉 Test Session Completed", style="bold green"))
            
            return final_report
            
        except Exception as e:
            logger.error(f"Test sequence failed: {e}")
            self.context_manager.end_session("failed")
            console.print(Panel(f"❌ Test Session Failed: {str(e)}", style="bold red"))
            raise
        
        finally:
            self.is_running = False
            if self.session_recorder and self.context_manager.current_session:
                try:
                    self.session_recorder.end_session(self.context_manager.current_session.status)
                    self.session_recorder.save()
                    console.print(f"📝 Session log written to: {self.session_recorder.filepath}")
                except Exception:
                    pass
    
    async def _execute_test_step(self, 
                               nl_prompt: str, 
                               available_tools: List[str],
                               pdf_path: Optional[str] = None) -> TestStepResult:
        """Execute a single test step"""
        
        start_time = time.time()
        step_id = f"step_{self.current_step:03d}_{int(start_time)}"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # Parse natural language step
                task1 = progress.add_task("🧠 Parsing step with LLM...", total=None)
                test_step = await self.llm_service.parse_natural_language_step(
                    nl_prompt, available_tools, self.last_page_state
                )
                if self.session_recorder:
                    try:
                        self.session_recorder.record_step_input(nl_prompt)
                        self.session_recorder.record_structured_step(test_step)
                    except Exception:
                        pass
                progress.update(task1, completed=True)
                
                # Add PDF path if needed and not specified
                if pdf_path and "pdf_path" not in test_step.parameters:
                    test_step.parameters["pdf_path"] = pdf_path
                
                # Execute MCP request
                task2 = progress.add_task(f"🔧 Executing {test_step.action_type}...", total=None)
                
                # Enhanced logging for LLM structured output
                logger.info(f"🤖 LLM Structured Output:")
                logger.info(f"   Step ID: {test_step.step_id}")
                logger.info(f"   Description: {test_step.description}")
                logger.info(f"   Action Type: {test_step.action_type}")
                logger.info(f"   Parameters: {test_step.parameters}")
                logger.info(f"   Expected Outcome: {test_step.expected_outcome}")
                if hasattr(test_step, 'additional_parameters') and test_step.additional_parameters:
                    logger.info(f"   Additional Parameters: {test_step.additional_parameters}")
                
                # Debug logging
                logger.debug(f"Executing tool: {test_step.action_type}")
                logger.debug(f"Parameters: {test_step.parameters}")
                if hasattr(test_step, 'additional_parameters') and test_step.additional_parameters:
                    logger.debug(f"Additional parameters: {test_step.additional_parameters}")
                if isinstance(test_step.parameters, dict):
                    logger.debug(f"Parameter types: {[(k, type(v)) for k, v in test_step.parameters.items()]}")
                else:
                    logger.debug(f"Parameters is not a dict, type: {type(test_step.parameters)}, value: {test_step.parameters}")
                
                try:
                    # Execute primary tool call
                    if self.session_recorder:
                        try:
                            self.session_recorder.record_tool_call(test_step.action_type, test_step.parameters)
                        except Exception:
                            pass
                    mcp_response = await asyncio.wait_for(
                        self.mcp_client.call_tool(test_step.action_type, test_step.parameters),
                        timeout=self.step_timeout
                    )
                    
                    # Execute additional parameter sets if present
                    if hasattr(test_step, 'additional_parameters') and test_step.additional_parameters:
                        logger.info(f"Executing {len(test_step.additional_parameters)} additional parameter sets")
                        for i, additional_params in enumerate(test_step.additional_parameters):
                            logger.debug(f"Executing additional parameter set {i+1}: {additional_params}")
                            # Execute additional calls and aggregate responses if needed
                            additional_response = await asyncio.wait_for(
                                self.mcp_client.call_tool(test_step.action_type, additional_params),
                                timeout=self.step_timeout
                            )
                            # For now, we use the primary response, but in the future could aggregate
                            logger.debug(f"Additional response {i+1} completed")
                    
                    progress.update(task2, completed=True)
                    
                    # Log MCP tool response
                    logger.info(f"🔧 MCP Tool Response:")
                    logger.info(f"   Tool: {test_step.action_type}")
                    logger.info(f"   Status: {mcp_response.get('status', 'unknown')}")
                    logger.info(f"   Response Keys: {list(mcp_response.keys())}")
                    if '_metadata' in mcp_response:
                        logger.info(f"   Metadata: {mcp_response['_metadata']}")
                    
                    # Get snapshot data
                    task3 = progress.add_task("📸 Capturing snapshot...", total=None)
                    snapshot_data = await self.mcp_client.get_snapshot_data(
                        test_step.action_type, test_step.parameters, mcp_response
                    )
                    if self.session_recorder:
                        try:
                            self.session_recorder.record_snapshot(snapshot_data)
                        except Exception:
                            pass
                    
                    # Extract page state for next step context
                    # Log snapshot data
                    logger.info(f"📸 Snapshot Data:")
                    logger.info(f"   Tool Name: {snapshot_data.get('tool_name', 'unknown')}")
                    logger.info(f"   Execution Time: {snapshot_data.get('execution_time', 'unknown')}")
                    logger.info(f"   Parameters Used: {snapshot_data.get('parameters_used', 'unknown')}")
                    if 'tool_result' in snapshot_data:
                        tool_result = snapshot_data['tool_result']
                        logger.info(f"   Tool Result Keys: {list(tool_result.keys())}")
                        if 'content' in tool_result:
                            content_preview = str(tool_result['content'])[:200] + "..." if len(str(tool_result['content'])) > 200 else str(tool_result['content'])
                            logger.info(f"   Content Preview: {content_preview}")
                    
                    self._extract_page_state(mcp_response, snapshot_data)
                    
                    progress.update(task3, completed=True)
                    
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Step timed out after {self.step_timeout} seconds")
                
                # Analyze result with LLM
                task4 = progress.add_task("🤖 Analyzing result...", total=None)
                llm_analysis = await self.llm_service.analyze_step_result(
                    test_step, mcp_response, snapshot_data
                )
                progress.update(task4, completed=True)

                # If step failed per LLM analysis, take a screenshot
                if llm_analysis.pass_fail is False and self.mcp_client and self.mcp_client.has_tool("browser_take_screenshot"):
                    try:
                        # Prepare output path in mcp-client/output
                        output_dir = Path(__file__).parent.parent / "output"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path = output_dir / f"{self.context_manager.current_session.session_id}_step_{self.current_step:03d}.png"

                        if self.session_recorder:
                            try:
                                self.session_recorder.record_tool_call("browser_take_screenshot", {"filename": str(screenshot_path), "raw": True})
                            except Exception:
                                pass

                        screenshot_result = await self.mcp_client.call_tool(
                            "browser_take_screenshot",
                            {"raw": True, "filename": str(screenshot_path), "fullPage": True}
                        )
                        # If no content returned, try without filename to get payload inline
                        if not isinstance(screenshot_result, dict) or not screenshot_result.get("content"):
                            screenshot_result = await self.mcp_client.call_tool(
                                "browser_take_screenshot",
                                {"raw": True, "fullPage": True}
                            )
                        # Try to write image locally from result content if server didn't save to our FS
                        wrote = self._write_screenshot_file(screenshot_result, screenshot_path)
                        # Attach screenshot result into snapshot_data for logging
                        if isinstance(snapshot_data, dict):
                            snapshot_data.setdefault("failure_artifacts", {})["screenshot"] = screenshot_result
                            snapshot_data["failure_artifacts"]["screenshot_path"] = str(screenshot_path)
                        if wrote or screenshot_path.exists():
                            console.print(f"📸 Failure screenshot saved: {screenshot_path}")
                        else:
                            console.print(f"📸 Screenshot captured (not saved by server); wrote locally: { 'yes' if wrote else 'no' }")
                    except Exception as _sse:
                        logger.warning(f"Screenshot on failure failed: {_sse}")
            
            execution_time = time.time() - start_time
            
            # Create step result
            step_result = TestStepResult(
                step_id=step_id,
                description=test_step.description,
                action_type=test_step.action_type,
                parameters=test_step.parameters,
                mcp_response=mcp_response,
                snapshot_data=snapshot_data,
                llm_analysis=llm_analysis.__dict__,
                passed=llm_analysis.pass_fail,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            if self.session_recorder:
                try:
                    self.session_recorder.record_step_result(step_result)
                except Exception:
                    pass
            
            # Add to LLM context
            self.llm_service.add_to_context(
                "assistant", 
                f"Executed {test_step.action_type}: {llm_analysis.content}"
            )
            
            return step_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Attempt screenshot on unexpected failure as well
            if self.mcp_client and self.mcp_client.has_tool("browser_take_screenshot"):
                try:
                    output_dir = Path(__file__).parent.parent / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = output_dir / f"{self.context_manager.current_session.session_id or 'session'}_step_{self.current_step:03d}_exception.png"

                    if self.session_recorder:
                        try:
                            self.session_recorder.record_tool_call("browser_take_screenshot", {"filename": str(screenshot_path), "raw": True})
                        except Exception:
                            pass

                    screenshot_result = await self.mcp_client.call_tool(
                        "browser_take_screenshot",
                        {"raw": True, "filename": str(screenshot_path), "fullPage": True}
                    )
                    if not isinstance(screenshot_result, dict) or not screenshot_result.get("content"):
                        screenshot_result = await self.mcp_client.call_tool(
                            "browser_take_screenshot",
                            {"raw": True, "fullPage": True}
                        )
                    wrote = self._write_screenshot_file(screenshot_result, screenshot_path)
                    if wrote or screenshot_path.exists():
                        console.print(f"📸 Exception screenshot saved: {screenshot_path}")
                except Exception as _sse:
                    logger.warning(f"Screenshot on exception failed: {_sse}")
            
            # Create failed step result
            step_result = TestStepResult(
                step_id=step_id,
                description=nl_prompt,
                action_type="unknown",
                parameters={},
                mcp_response={"error": error_msg},
                snapshot_data={},
                llm_analysis={"error": error_msg},
                passed=False,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                error=error_msg
            )
            
            logger.error(f"Step execution failed: {error_msg}")
            return step_result
    
    def _display_step_result(self, step_result: TestStepResult):
        """Display step result in a formatted way"""
        
        # Status indicator
        if step_result.passed is True:
            status = "✅ PASSED"
            style = "green"
        elif step_result.passed is False:
            status = "❌ FAILED"
            style = "red"
        else:
            status = "❓ INCONCLUSIVE"
            style = "yellow"
        
        # Create result table
        table = Table(title=f"Step {self.current_step} Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Status", status)
        table.add_row("Action", step_result.action_type)
        table.add_row("Execution Time", f"{step_result.execution_time:.2f}s")
        
        if step_result.error:
            table.add_row("Error", step_result.error)
        
        # Add LLM analysis if available
        if "reasoning" in step_result.llm_analysis:
            table.add_row("LLM Analysis", step_result.llm_analysis["reasoning"][:100] + "...")
        
        console.print(table)
        
        # Show snapshot summary
        if step_result.snapshot_data:
            console.print(f"📸 Snapshot: {len(step_result.snapshot_data)} data points captured")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final test report"""
        
        session_summary = self.context_manager.get_session_summary()
        session_data = self.context_manager.export_session_data()
        
        report = {
            "session_id": session_data["session"]["session_id"] if session_data["session"] else "unknown",
            "test_objective": session_data["session"]["test_objective"] if session_data["session"] else "unknown",
            "execution_summary": {
                "total_steps": self.current_step,
                "passed_steps": session_data["session"]["passed_steps"] if session_data["session"] else 0,
                "failed_steps": session_data["session"]["failed_steps"] if session_data["session"] else 0,
                "inconclusive_steps": session_data["session"]["inconclusive_steps"] if session_data["session"] else 0,
                "execution_time": session_data["session"]["end_time"] if session_data["session"] else None,
            },
            "insights": session_data["insights"],
            "context_summary": session_data["context_summary"],
            "step_details": session_data["step_results"],
            "final_context": self.context_manager.get_current_context()
        }
        
        # Display summary table
        summary_table = Table(title="📊 Test Execution Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Steps", str(report["execution_summary"]["total_steps"]))
        summary_table.add_row("Passed", str(report["execution_summary"]["passed_steps"]))
        summary_table.add_row("Failed", str(report["execution_summary"]["failed_steps"]))
        summary_table.add_row("Inconclusive", str(report["execution_summary"]["inconclusive_steps"]))
        
        if report["insights"]:
            summary_table.add_row("Key Insights", str(len(report["insights"])))
        
        console.print(summary_table)
        
        return report
    
    async def run_single_step(self, 
                            nl_prompt: str, 
                            pdf_path: Optional[str] = None) -> TestStepResult:
        """Run a single test step (useful for testing)"""
        
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")
        
        available_tools = self.mcp_client.get_tool_names()
        return await self._execute_test_step(nl_prompt, available_tools, pdf_path)
    
    async def run_test_from_file(self,
                               test_file_path: str,
                               pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """Run automated test sequence from a file with step-by-step instructions"""
        
        # Read test file
        try:
            with open(test_file_path, 'r') as f:
                test_steps = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to read test file {test_file_path}: {e}")
        
        if not test_steps:
            raise RuntimeError(f"Test file {test_file_path} is empty or contains no valid steps")
        
        # Extract test objective from file name or first step
        test_objective = f"Automated test from {Path(test_file_path).name}"
        initial_prompt = test_steps[0]
        
        console.print(f"📋 Running test from file: {test_file_path}")
        console.print(f"📝 Found {len(test_steps)} test steps")
        console.print(f"🎯 Test objective: {test_objective}")
        
        # Start test session
        session_id = self.context_manager.start_session(test_objective)
        if self.session_recorder:
            try:
                self.session_recorder.start_session(session_id, test_objective)
            except Exception:
                pass
        self.is_running = True
        self.current_step = 0
        
        console.print(Panel(f"🚀 Starting Test Session: {session_id}", style="bold green"))
        console.print(f"📋 Objective: {test_objective}")
        console.print(f"🎯 Initial Step: {initial_prompt}")
        
        try:
            # Execute each step from the file
            for i, step_instruction in enumerate(test_steps, 1):
                self.current_step = i
                console.print(f"\n🔄 Step {i}/{len(test_steps)}: {step_instruction}")
                
                if not self.mcp_client:
                    raise RuntimeError("MCP client not initialized")
                
                available_tools = self.mcp_client.get_tool_names()
                step_result = await self._execute_test_step(step_instruction, available_tools, pdf_path)
                
                # Store step result
                self.context_manager.add_step_result(step_result)
                
                # Check if step failed and should stop
                if step_result.passed is False:
                    console.print(f"❌ Step {i} failed: {step_result.error}")
                    console.print("🛑 Stopping test execution due to step failure")
                    self.context_manager.end_session("failed")
                    final_report = self._generate_final_report()
                    if self.session_recorder:
                        try:
                            self.session_recorder.attach_final_report(final_report)
                            self.session_recorder.end_session("failed")
                            self.session_recorder.save()
                            console.print(f"📝 Session log written to: {self.session_recorder.filepath}")
                        except Exception:
                            pass
                    return final_report
                elif step_result.passed is True:
                    console.print(f"✅ Step {i} passed")
                else:
                    console.print(f"❓ Step {i} result inconclusive")
                
                # Brief pause between steps
                await asyncio.sleep(0.5)
            
            # All steps completed successfully
            self.context_manager.end_session("completed")
            final_report = self._generate_final_report()
            if self.session_recorder:
                try:
                    self.session_recorder.attach_final_report(final_report)
                    self.session_recorder.end_session("completed")
                    self.session_recorder.save()
                    console.print(f"📝 Session log written to: {self.session_recorder.filepath}")
                except Exception:
                    pass
            return final_report
        
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            error_step = TestStepResult(
                step_id="error_step",
                description="Test file execution failed",
                action_type="file_test_execution",
                parameters={"test_file": test_file_path},
                mcp_response={},
                snapshot_data={},
                llm_analysis={},
                passed=False,
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            self.context_manager.add_step_result(error_step)
            self.context_manager.end_session("failed")
            final_report = self._generate_final_report()
            if self.session_recorder:
                try:
                    self.session_recorder.attach_final_report(final_report)
                    self.session_recorder.end_session("failed")
                    self.session_recorder.save()
                    console.print(f"📝 Session log written to: {self.session_recorder.filepath}")
                except Exception:
                    pass
            return final_report
        
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the test execution"""
        self.is_running = False
        console.print("🛑 Test execution stopped by user")


# Factory function for easy orchestrator creation
async def create_test_orchestrator(mcp_server_url: str, 
                                 openai_api_key: str,
                                 **kwargs) -> TestOrchestrator:
    """Create test orchestrator (caller must initialize with async context manager)"""
    orchestrator = TestOrchestrator(mcp_server_url, openai_api_key, **kwargs)
    # DO NOT call __aenter__() here - let the caller handle it with async context manager
    return orchestrator 
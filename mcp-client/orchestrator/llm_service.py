"""
LLM Service for Test Orchestration
Handles OpenAI and Anthropic API integration with context management
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM analysis"""
    content: str
    reasoning: str
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    pass_fail: Optional[bool] = None


@dataclass
class TestStep:
    """Represents a test step"""
    step_id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    additional_parameters: Optional[List[Dict[str, Any]]] = None  # For multi-element actions


class LLMService:
    """Service for LLM integration with context management"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", recorder: Optional["SessionLogRecorder"] = None):
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.context_summary = ""
        self.max_context_length = 8000  # tokens
        self.recorder = recorder
        
        # Determine provider based on model name
        if model.startswith('claude'):
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic library not installed. Run: pip install anthropic")
            self.provider = 'anthropic'
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            self.provider = 'openai'
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            self.client = AsyncOpenAI(api_key=self.api_key)
        
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Make API call to LLM provider with retry logic"""
        try:
            if self.provider == 'anthropic':
                # Convert OpenAI format to Anthropic format
                system_message = ""
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        user_messages.append(msg)
                
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.1,
                    system=system_message,
                    messages=user_messages
                )
                usage = getattr(response, "usage", None)
                # Anthropic may not always provide usage; normalize if present
                usage_dict = None
                if usage is not None:
                    try:
                        usage_dict = {
                            "prompt_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0),
                            "total_tokens": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
                        }
                    except Exception:
                        usage_dict = None
                return response.content[0].text, usage_dict
            else:
                # OpenAI
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                usage_dict = None
                try:
                    if getattr(response, "usage", None) is not None:
                        usage = response.usage
                        usage_dict = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(usage, "completion_tokens", 0),
                            "total_tokens": getattr(usage, "total_tokens", 0),
                        }
                except Exception:
                    usage_dict = None
                return response.choices[0].message.content, usage_dict
        except Exception as e:
            logger.error(f"{self.provider.upper()} API call failed: {e}")
            raise
    
    # Keep backward compatibility
    async def _call_openai(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Legacy method for backward compatibility"""
        return await self._call_llm(messages)
    
    async def parse_natural_language_step(self, nl_prompt: str, available_tools: List[str], page_state: Optional[str] = None) -> TestStep:
        """Parse natural language prompt into structured test step"""
        tools_list = ', '.join(available_tools)
        # Build compact page context if available
        page_context = ""
        if page_state:
            # Truncate very long snapshots; rely on DOM cache to pass minimal sections
            compact_state = page_state
            if len(compact_state) > 4000:
                compact_state = compact_state[:4000] + "\n..."
            page_context = f"""

CURRENT PAGE STATE (compact):
{compact_state}

Use the [ref=XXX] values to target elements.
"""

        system_prompt = f"""
You are a test automation expert. Parse the natural language test instruction into structured test steps.

Available MCP tools: {tools_list}
{page_context}
You can return either:
1. A single JSON object for simple actions
2. An array of JSON objects for complex multi-step actions

Each JSON object should have:
- step_id: unique identifier  
- description: clear description of what to test
- action_type: MUST be one of the exact tool names from the available MCP tools list above
- parameters: JSON object (dictionary) for the MCP tool
- expected_outcome: what should happen if the step passes

IMPORTANT: 
- action_type MUST exactly match one of the tool names from the list above
- parameters MUST be a JSON object (dictionary), NOT an array
- For complex actions involving multiple elements, focus on the FIRST element only
- If instructions has scoll in it, its likely for visual validation for user DON'T verify playwright code or the console logs for it
- If instructions has wait in it, its likely for visual validation for user, ignore any resource load issues and DON'T verify playwright code and DON'T check the console logs for it
 - For waits, DO NOT include unsupported keys like "element", "ref", or "timeout". Use ONLY: 
   - "time" (seconds to wait), or
   - "text" (wait until text appears), or
   - "textGone" (wait until text disappears).
            
            Tool Usage:
            - For navigation: "browser_navigate" with {{"url": "https://example.com"}}
            - For clicking: "browser_click" with {{"element": "button name", "ref": "eXX"}}
            - For typing: "browser_type" with {{"element": "input field", "ref": "eXX", "text": "value"}}
            - For waiting: "browser_wait_for" with one of:
              • {{"time": 2}}
              • {{"text": "Your Profile"}}
              • {{"textGone": "Loading profile..."}}
            - For screenshot: "browser_take_screenshot" with {{"raw": true}}

Examples:

Simple action:
Input: "Navigate to github.com"
Output: {{
    "step_id": "nav_001",
    "description": "Navigate to github.com",
    "action_type": "browser_navigate",
    "parameters": {{"url": "https://github.com"}},
    "expected_outcome": "Page should load github.com successfully"
}}

Multi-step action:
Input: "Fill email and password then click login"
Output: [
    {{
        "step_id": "fill_email_001",
        "description": "Fill email field",
        "action_type": "browser_type",
        "parameters": {{"element": "Email", "ref": "e16", "text": "user@example.com"}},
        "expected_outcome": "Email field should be filled"
    }},
    {{
        "step_id": "fill_password_001",
        "description": "Fill password field",
        "action_type": "browser_type",
        "parameters": {{"element": "Password", "ref": "e17", "text": "password123"}},
        "expected_outcome": "Password field should be filled"
    }},
    {{
        "step_id": "click_001",
        "description": "Click login button",
        "action_type": "browser_click",
        "parameters": {{"element": "Login", "ref": "e20"}},
        "expected_outcome": "User should be logged in"
    }}
]

For complex single-line instructions like "Fill creds ab@g and 12345":
- Focus on the FIRST element mentioned
- Return parameters as a dictionary, not an array
- Example: {{"element": "Email", "ref": "e16", "text": "ab@g"}}

Wait example:
Input: "Wait until 'Your Profile' is visible for 2 seconds"
Output: {{
    "step_id": "wait_001",
    "description": "Wait until 'Your Profile' is visible for 2 seconds",
    "action_type": "browser_wait_for",
    "parameters": {{"text": "Your Profile", "time": 2}},
    "expected_outcome": "'Your Profile' text appears within 2 seconds"
}}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this test step: {nl_prompt}"}
        ]
        
        response_text, usage = await self._call_llm(messages)
        # Record LLM call (parse phase)
        if self.recorder:
            try:
                self.recorder.record_llm_call(
                    phase="parse_natural_language_step",
                    provider=self.provider,
                    model=self.model,
                    messages=messages,
                    response_text=response_text,
                    usage=usage,
                )
            except Exception:
                pass
        
        try:
            # Clean up the response
            response = response_text.strip()
            logger.info(f"🤖 Raw LLM Response:")
            logger.info(f"   Response: {response}")
            
            # Try to parse as JSON
            parsed_data = json.loads(response)
            logger.info(f"📊 Parsed JSON Data:")
            logger.info(f"   Type: {type(parsed_data)}")
            logger.info(f"   Value: {parsed_data}")
            
            # Handle multiple JSON objects (array response)
            if isinstance(parsed_data, list):
                logger.info(f"Multiple test steps returned: {len(parsed_data)}")
                # For now, return the first step and log that there are more
                # In the future, the orchestrator could handle multiple steps
                step_data = parsed_data[0]
                if len(parsed_data) > 1:
                    logger.warning(f"Multiple steps returned, using first step. Remaining steps: {len(parsed_data) - 1}")
            else:
                step_data = parsed_data
            
            logger.debug(f"🔍 Step data after array handling: {step_data}")
            logger.debug(f"🔍 Step data type: {type(step_data)}")
            if 'parameters' in step_data:
                logger.debug(f"🔍 Parameters type: {type(step_data['parameters'])}")
                logger.debug(f"🔍 Parameters value: {step_data['parameters']}")
            
            # Handle parameters as array for multi-element actions
            if 'parameters' in step_data and isinstance(step_data['parameters'], list):
                logger.info(f"Parameters array detected with {len(step_data['parameters'])} elements: {step_data['parameters']}")
                # For tools like browser_type, we might need to handle multiple elements
                # For now, convert to a special format or handle the first element
                params_list = step_data['parameters']
                if len(params_list) > 0:
                    # Use the first parameter set and log others
                    first_params = params_list[0]
                    if isinstance(first_params, dict):
                        step_data['parameters'] = first_params
                        if len(params_list) > 1:
                            logger.info(f"Multiple parameter sets detected, using first one. Additional sets: {len(params_list) - 1}")
                            # Store additional parameters for future use
                            step_data['additional_parameters'] = params_list[1:]
                    else:
                        logger.error(f"First parameter set is not a dict: {type(first_params)}, value: {first_params}")
                        # Fallback: create a basic parameter structure
                        step_data['parameters'] = {"raw_params": params_list}
                else:
                    logger.warning("Parameters array is empty, using empty dict")
                    step_data['parameters'] = {}
            
            final_test_step = TestStep(**step_data)
            logger.info(f"✅ Final TestStep Created:")
            logger.info(f"   Step ID: {final_test_step.step_id}")
            logger.info(f"   Description: {final_test_step.description}")
            logger.info(f"   Action Type: {final_test_step.action_type}")
            logger.info(f"   Parameters: {final_test_step.parameters}")
            logger.info(f"   Expected Outcome: {final_test_step.expected_outcome}")
            if hasattr(final_test_step, 'additional_parameters') and final_test_step.additional_parameters:
                logger.info(f"   Additional Parameters: {final_test_step.additional_parameters}")
            
            return final_test_step
            
        except json.JSONDecodeError as e:
            # Handle malformed JSON - try to extract the first valid JSON object
            logger.warning(f"JSON decode error, attempting to extract first valid JSON: {e}")
            try:
                # Find the first complete JSON object
                brace_count = 0
                start_pos = response.find('{')
                if start_pos == -1:
                    raise ValueError("No JSON object found in response")
                
                for i in range(start_pos, len(response)):
                    char = response[i]
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            first_json = response[start_pos:i+1]
                            step_data = json.loads(first_json)
                            logger.info("✅ Successfully extracted first JSON object")
                            logger.info(f"   Extracted JSON: {step_data}")
                            final_test_step = TestStep(**step_data)
                            logger.info(f"✅ Final TestStep Created (from extraction):")
                            logger.info(f"   Step ID: {final_test_step.step_id}")
                            logger.info(f"   Description: {final_test_step.description}")
                            logger.info(f"   Action Type: {final_test_step.action_type}")
                            logger.info(f"   Parameters: {final_test_step.parameters}")
                            logger.info(f"   Expected Outcome: {final_test_step.expected_outcome}")
                            return final_test_step
                
                raise ValueError("Could not find complete JSON object")
                
            except Exception as fallback_error:
                logger.error(f"Failed to parse LLM response as JSON: {response}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")
                
        except Exception as e:
            logger.error(f"Failed to create TestStep from data: {step_data if 'step_data' in locals() else 'N/A'}")
            raise ValueError(f"Invalid TestStep data: {e}")
    
    async def analyze_step_result(self, step: TestStep, mcp_response: Dict[str, Any], 
                                snapshot_data: Optional[Dict[str, Any]] = None,
                                analysis_hints: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Analyze MCP response to determine if step passed or failed"""
        
        context = self._build_analysis_context(step, mcp_response, snapshot_data, analysis_hints)
        
        system_prompt = """
You are a test result analyzer. Analyze the MCP response and determine if the test step passed or failed.

Return ONLY a raw JSON object with:
- content: summary of what happened
- reasoning: detailed explanation of your analysis
- action_type: the MCP action that was performed
- parameters: parameters that were used
- confidence: confidence level (0.0 to 1.0)
- pass_fail: true if passed, false if failed, null if inconclusive

CRITICAL: Return ONLY the JSON object. Do NOT use markdown code blocks, do NOT add explanations, do NOT wrap in ```json```. Just return the raw JSON.

Be thorough in your analysis and provide clear reasoning.
"""
        
        # Provide helpful guidance to treat fresh snapshots/UI changes as evidence of action execution
        reasoning_hint = (
            "First, verify the expected element label/ref from the step parameters is present in the snapshot when the action targets an element (click/type/select/hover/drag/press/wait for). "
            "If the expected element label/ref is missing in the trimmed snapshot, assume the context may be truncated and fallback to the full snapshot data (if provided) before concluding. "
            "Only if element evidence is present (or not required for this action type), then consider diffs/new snapshot signals. "
            "Treat a new or changed snapshot, increases in 'ref=' lines, added UI nodes, or new UI keywords as supporting evidence, not the sole basis."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this test result (note: {reasoning_hint}):\n{context}"}
        ]
        
        response_text, usage = await self._call_openai(messages)
        if self.recorder:
            try:
                self.recorder.record_llm_call(
                    phase="analyze_step_result",
                    provider=self.provider,
                    model=self.model,
                    messages=messages,
                    response_text=response_text,
                    usage=usage,
                )
            except Exception:
                pass
        
        try:
            result_data = json.loads(response_text)
            return LLMResponse(**result_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {response_text}")
            # Fallback response
            return LLMResponse(
                content=response_text,
                reasoning="Failed to parse structured response",
                action_type=step.action_type,
                parameters=step.parameters,
                confidence=0.5,
                pass_fail=None
            )
    
    async def generate_next_step(self, current_context: str, test_objective: str) -> Optional[str]:
        """Generate the next test step based on current context"""
        
        system_prompt = f"""
You are a test automation expert. Based on the current test context and overall objective, 
suggest the next logical test step in natural language.

Test Objective: {test_objective}

If the test sequence is complete, return "COMPLETE".
If you need more information, return "NEED_INFO: <what you need>".
Otherwise, return a natural language instruction for the next step.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current context:\n{current_context}\n\nWhat should be the next step?"}
        ]
        
        response = await self._call_openai(messages)
        
        if response.strip().upper() == "COMPLETE":
            return None
        elif response.startswith("NEED_INFO:"):
            logger.info(f"LLM needs more info: {response}")
            return None
        
        return response.strip()
    
    def _build_analysis_context(self, step: TestStep, mcp_response: Dict[str, Any], 
                              snapshot_data: Optional[Dict[str, Any]],
                              analysis_hints: Optional[Dict[str, Any]] = None) -> str:
        """Build compact context string for analysis"""
        # Trim mcp_response to essentials
        trimmed_resp: Dict[str, Any] = {}
        if isinstance(mcp_response, dict):
            if "content" in mcp_response:
                content = mcp_response.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and first.get("type") == "text":
                        txt = first.get("text", "")
                        if len(txt) > 1200:
                            txt = txt[:1200] + "\n..."
                        trimmed_resp["content"] = [{"type": "text", "text": txt}]
            # Always keep metadata if available
            if "_metadata" in mcp_response:
                trimmed_resp["_metadata"] = mcp_response["_metadata"]

        trimmed_snapshot: Optional[Dict[str, Any]] = None
        if isinstance(snapshot_data, dict):
            # Keep only tool_name, parameters_used and first 800 chars of any text content
            trimmed_snapshot = {
                "tool_name": snapshot_data.get("tool_name"),
                "parameters_used": snapshot_data.get("parameters_used"),
            }
            tool_result = snapshot_data.get("tool_result", {})
            if isinstance(tool_result, dict):
                cont = tool_result.get("content")
                if isinstance(cont, list) and cont:
                    first = cont[0]
                    if isinstance(first, dict) and first.get("type") == "text":
                        txt = first.get("text", "")
                        if len(txt) > 800:
                            txt = txt[:800] + "\n..."
                        trimmed_snapshot["content"] = [{"type": "text", "text": txt}]
                # Provide full snapshot text for fallback if needed
                # (We include it under a separate key to avoid bloating the primary content signal.)
                try:
                    if isinstance(cont, list) and cont:
                        first_full = cont[0]
                        if isinstance(first_full, dict) and first_full.get("type") == "text":
                            trimmed_snapshot["full_text_fallback"] = first_full.get("text", "")
                except Exception:
                    pass

        context = f"""
Test Step:
- ID: {step.step_id}
- Description: {step.description}
- Action: {step.action_type}
- Parameters: {json.dumps(step.parameters, indent=2)}
- Expected: {step.expected_outcome}

MCP Response:
{json.dumps(trimmed_resp or mcp_response, indent=2)}
"""
        
        if trimmed_snapshot or snapshot_data:
            context += f"\n\nSnapshot Data:\n{json.dumps(trimmed_snapshot or snapshot_data, indent=2)}"
        
        if analysis_hints:
            try:
                context += f"\n\nUI Change Signals:\n{json.dumps(analysis_hints, indent=2)}"
            except Exception:
                pass
        
        return context
    
    def add_to_context(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        self._manage_context_length()
    
    def _manage_context_length(self):
        """Manage context length with rolling summary"""
        # Simple token estimation (rough)
        total_tokens = sum(len(msg["content"].split()) * 1.3 for msg in self.conversation_history)
        
        if total_tokens > self.max_context_length:
            # Create rolling summary of older messages
            old_messages = self.conversation_history[:-10]  # Keep last 10 messages
            self.conversation_history = self.conversation_history[-10:]
            
            # Update context summary
            if old_messages:
                summary_content = "\n".join([f"{msg['role']}: {msg['content'][:200]}..." 
                                           for msg in old_messages])
                self.context_summary = f"Previous context summary:\n{summary_content}\n\n{self.context_summary}"
    
    def get_full_context(self) -> str:
        """Get full context including summary and recent history"""
        context = ""
        if self.context_summary:
            context += f"{self.context_summary}\n\n"
        
        context += "Recent conversation:\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            context += f"{msg['role']}: {msg['content']}\n"
        
        return context
    
    def reset_context(self):
        """Reset conversation context"""
        self.conversation_history = []
        self.context_summary = "" 
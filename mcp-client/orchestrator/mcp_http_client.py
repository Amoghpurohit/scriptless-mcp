"""
Enhanced MCP HTTP Client with Keep-Alive Support
Provides HTTP JSON-RPC communication with persistent connections
"""

import asyncio
import json
import uuid
import urllib.parse
from typing import Dict, List, Optional, Any
import aiohttp
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from .context_manager import SessionLogRecorder


@dataclass
class MCPRequest:
    """MCP JSON-RPC request"""
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id
        }


@dataclass
class MCPResponse:
    """MCP JSON-RPC response"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error")
        )
    
    @property
    def is_success(self) -> bool:
        return self.error is None
    
    @property
    def error_message(self) -> str:
        if self.error:
            return self.error.get("message", "Unknown error")
        return ""


class MCPHTTPClient:
    """Enhanced MCP client with HTTP JSON-RPC and keep-alive support"""
    
    def __init__(self, server_url: str, timeout: int = 30, recorder: Optional[SessionLogRecorder] = None, browser_args: Optional[List[str]] = None):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.browser_args = browser_args
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.connection_id = str(uuid.uuid4())
        self.session_id: Optional[str] = None  # Add session ID tracking
        self.sse_connection: Optional[aiohttp.ClientResponse] = None  # Keep SSE connection alive
        self.sse_task: Optional[asyncio.Task] = None  # Background task for SSE
        self.pending_responses: Dict[str, asyncio.Future] = {}  # Track pending tool responses
        self.recorder: Optional[SessionLogRecorder] = recorder
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Create persistent session with keep-alive
        connector = aiohttp.TCPConnector(
            keepalive_timeout=300,  # 5 minutes
            enable_cleanup_closed=True,
            limit=10,
            limit_per_host=5
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
                'Connection': 'keep-alive',
                'X-MCP-Client-ID': self.connection_id
            }
        )
        
        # Initialize connection and discover tools
        await self._initialize_connection()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Stop SSE task
        if self.sse_task:
            self.sse_task.cancel()
            try:
                await self.sse_task
            except asyncio.CancelledError:
                pass
        
        # Close SSE connection
        if self.sse_connection:
            self.sse_connection.close()
        
        # Close session
        if self.session:
            await self.session.close()
    
    async def _initialize_connection(self):
        """Initialize MCP connection and discover tools"""
        try:
            # Establish persistent SSE connection
            logger.info("🔗 Establishing persistent SSE connection...")
            
            self.sse_connection = await self.session.get(
                f"{self.server_url}/sse",
                headers={'Accept': 'text/event-stream'}
            )
            
            if self.sse_connection.status != 200:
                raise RuntimeError(f"Failed to connect to SSE endpoint: HTTP {self.sse_connection.status}")
            
            # Start background task to handle SSE events
            self.sse_task = asyncio.create_task(self._handle_sse_events())
            
            # Wait for session ID from SSE events
            for _ in range(50):  # Wait up to 5 seconds
                if self.session_id:
                    break
                await asyncio.sleep(0.1)
            
            if not self.session_id:
                raise RuntimeError("Failed to get session ID from SSE endpoint")
            
            logger.info(f"✅ Got session ID: {self.session_id}")
            
            # Now send initialize request with session ID
            init_request = MCPRequest(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "test-orchestrator",
                        "version": "1.0.0"
                    }
                }
            )
            
            response, response_headers = await self._send_request_with_headers(init_request)
            if not response.is_success:
                raise RuntimeError(f"Failed to initialize MCP connection: {response.error_message}")
            
            logger.info("✅ MCP connection initialized")
            
            # Discover available tools
            await self._discover_tools()
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connection: {e}")
            raise
    
    async def _handle_sse_events(self):
        """Background task to handle SSE events (robust parser for large payloads)."""
        try:
            buffer = ""
            async for chunk in self.sse_connection.content.iter_any():
                if not chunk:
                    continue
                try:
                    buffer += chunk.decode('utf-8', errors='ignore')
                except Exception:
                    # Fallback: skip undecodable bytes
                    continue

                # Process complete SSE events separated by blank lines
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    lines = raw_event.splitlines()

                    event_name = None
                    data_lines = []
                    for ln in lines:
                        if ln.startswith('event:'):
                            event_name = ln.split(':', 1)[1].strip()
                        elif ln.startswith('data:'):
                            # Support both 'data: ' and 'data:'
                            payload = ln.split(':', 1)[1]
                            if payload.startswith(' '):
                                payload = payload[1:]
                            data_lines.append(payload)

                    if not data_lines:
                        continue

                    data_payload = "\n".join(data_lines)

                    # Session endpoint announcement
                    if event_name == 'endpoint' or 'sessionId=' in data_payload:
                        try:
                            parsed = urllib.parse.urlparse(data_payload)
                            query_params = urllib.parse.parse_qs(parsed.query)
                            if 'sessionId' in query_params:
                                self.session_id = query_params['sessionId'][0]
                                logger.debug(f"Extracted session ID: {self.session_id}")
                                continue
                        except Exception:
                            # Not a URL, ignore
                            pass

                    # Record raw SSE event
                    if self.recorder:
                        try:
                            self.recorder.record_sse_event(data_payload)
                        except Exception:
                            pass

                    # Try parse JSON-RPC message
                    try:
                        response_data = json.loads(data_payload)
                    except json.JSONDecodeError:
                        # Not JSON
                        continue

                    request_id = response_data.get('id')
                    if request_id and request_id in self.pending_responses:
                        logger.debug(f"🎯 Received execution result for request {request_id}")
                        future = self.pending_responses.pop(request_id)
                        if not future.done():
                            future.set_result(response_data)
                        continue

                    # tools/list response via SSE
                    if 'result' in response_data and 'tools' in response_data.get('result', {}):
                        tools = response_data['result']['tools']
                        self.available_tools = tools
                        tool_names = [tool.get("name", "unknown") for tool in tools]
                        logger.info(f"✅ Discovered {len(tools)} tools via SSE: {tool_names}")
                        continue
                        
        except asyncio.CancelledError:
            logger.debug("SSE event handler cancelled")
        except Exception as e:
            logger.error(f"Error in SSE event handler: {e}")
    
    async def _discover_tools(self):
        """Discover available tools from the server"""
        try:
            tools_request = MCPRequest(
                method="tools/list",
                params={}
            )
            
            response = await self._send_request(tools_request)
            if response.is_success and response.result:
                self.available_tools = response.result.get("tools", [])
                tool_names = [tool.get("name", "unknown") for tool in self.available_tools]
                logger.info(f"✅ Discovered {len(self.available_tools)} tools: {tool_names}")
            else:
                logger.warning(f"Failed to discover tools: {response.error_message}")
                
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
    
    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send JSON-RPC request to MCP server using Streamable HTTP protocol"""
        response, _ = await self._send_request_with_headers(request)
        return response
    
    async def _send_request_with_headers(self, request: MCPRequest) -> tuple[MCPResponse, Dict[str, str]]:
        """Send JSON-RPC request and return both response and headers"""
        if not self.session:
            raise RuntimeError("Client session not initialized")
        
        try:
            request_data = request.to_dict()
            logger.debug(f"Sending MCP request: {request.method}")
            
            # Prepare URL with session ID
            if self.session_id:
                url = f"{self.server_url}/sse?sessionId={self.session_id}"
            else:
                url = self.server_url
            # Record outgoing JSON-RPC request
            if self.recorder:
                try:
                    self.recorder.record_json_rpc_request(request.id, request.method, request.params, url)
                except Exception:
                    pass
            
            async with self.session.post(
                url,
                json=request_data
            ) as response:
                
                if response.status == 202:
                    # HTTP 202 Accepted - for notifications/responses that don't expect a reply
                    logger.info(f"Request accepted (202) for method: {request.method}")
                    # Create a success response for 202
                    mcp_response = MCPResponse(
                        id=request.id,
                        result={"status": "accepted"},
                        error=None
                    )
                    response_headers = {k.lower(): v for k, v in response.headers.items()}
                    # Record response
                    if self.recorder:
                        try:
                            self.recorder.record_json_rpc_response(request.id, response.status, response_headers, result=mcp_response.result, error=mcp_response.error, content_type=response.headers.get('content-type'))
                        except Exception:
                            pass
                    return mcp_response, response_headers
                    
                elif response.status != 200:
                    body_text = await response.text()
                    # Record error response
                    if self.recorder:
                        try:
                            self.recorder.record_json_rpc_response(request.id, response.status, {k.lower(): v for k, v in response.headers.items()}, error={"message": body_text}, content_type=response.headers.get('content-type'))
                        except Exception:
                            pass
                    raise RuntimeError(f"HTTP {response.status}: {body_text}")
                
                # Extract response headers
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    # Simple JSON response
                    response_data = await response.json()
                    mcp_response = MCPResponse.from_dict(response_data)
                    # Record response
                    if self.recorder:
                        try:
                            self.recorder.record_json_rpc_response(request.id, response.status, response_headers, result=response_data.get("result"), error=response_data.get("error"), content_type=content_type)
                        except Exception:
                            pass
                    
                elif 'text/event-stream' in content_type:
                    # SSE stream response - read the first JSON event
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            json_data = line[6:]  # Remove 'data: ' prefix
                            if json_data and json_data != '[DONE]':
                                try:
                                    response_data = json.loads(json_data)
                                    mcp_response = MCPResponse.from_dict(response_data)
                                    # Record response event
                                    if self.recorder:
                                        try:
                                            self.recorder.record_json_rpc_response(request.id, response.status, response_headers, result=mcp_response.result, error=mcp_response.error, content_type=content_type)
                                        except Exception:
                                            pass
                                    break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        raise RuntimeError("No valid JSON data found in SSE stream")
                        
                else:
                    raise RuntimeError(f"Unexpected content type: {content_type}")
                
                if not mcp_response.is_success:
                    logger.error(f"MCP error: {mcp_response.error_message}")
                
                return mcp_response, response_headers
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise RuntimeError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise RuntimeError(f"Invalid server response: {e}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Call a specific MCP tool and wait for execution completion.

        If timeout is provided, it bounds the SSE wait for the execution result.
        Otherwise, a conservative timeout derived from the client's timeout is used.
        """
        if not self.has_tool(tool_name):
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {self.get_tool_names()}")
        
        # Debug logging for parameters
        logger.debug(f"📞 Calling tool: {tool_name}")
        logger.debug(f"🔧 Parameters type: {type(parameters)}")
        logger.debug(f"🔧 Parameters value: {parameters}")
        
        if isinstance(parameters, list):
            logger.error(f"❌ Parameters is a list instead of dict! This will cause MCP server error.")
            logger.error(f"❌ List contents: {parameters}")
        
        # Add browser args to browser-related tools if available
        tool_params = parameters.copy()
        if self.browser_args and tool_name.startswith('browser_'):
            tool_params['args'] = self.browser_args
            logger.debug(f"🔧 Added browser args to {tool_name}: {self.browser_args}")
        
        request = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": tool_params
            }
        )
        
        # Create a future to wait for the actual execution result
        loop = asyncio.get_running_loop()
        execution_future = loop.create_future()
        self.pending_responses[request.id] = execution_future
        
        try:
            # Send the initial request
            response = await self._send_request(request)
            
            if not response.is_success:
                # Remove the pending response on error
                self.pending_responses.pop(request.id, None)
                raise RuntimeError(f"Tool call failed: {response.error_message}")
            
            # If we got immediate result (not just HTTP 202), return it
            if response.result and response.result.get("status") != "accepted":
                self.pending_responses.pop(request.id, None)
                result = response.result
                result["_metadata"] = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "request_id": request.id,
                    "timestamp": asyncio.get_event_loop().time()
                }
                return result
            
            # Wait for the actual execution result via SSE
            logger.debug(f"⏳ Waiting for execution result for {tool_name} (request {request.id})")
            
            try:
                # Use provided timeout if specified; otherwise be conservative for large payloads
                effective_timeout = timeout if timeout is not None else max(self.timeout * 2, self.timeout + 30)
                execution_response = await asyncio.wait_for(
                    execution_future, 
                    timeout=effective_timeout
                )
                
                logger.debug(f"✅ Received execution result for {tool_name}")
                
                # Parse the execution result
                if execution_response.get('result'):
                    result = execution_response['result']
                elif execution_response.get('error'):
                    raise RuntimeError(f"Tool execution failed: {execution_response['error']}")
                else:
                    # Fallback: use the accepted status if no specific result
                    result = {"status": "accepted"}
                
                # Add metadata
                result["_metadata"] = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "request_id": request.id,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout waiting for {tool_name} execution result")
                # Fallback to accepted status
                result = {"status": "accepted"}
                result["_metadata"] = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "request_id": request.id,
                    "timestamp": asyncio.get_event_loop().time()
                }
                return result
                
        finally:
            # Clean up pending response
            self.pending_responses.pop(request.id, None)
    
    async def get_snapshot_data(self, tool_name: str, parameters: Dict[str, Any], tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get snapshot data from already executed tool result"""
        # FIXED: Use the provided tool_result instead of calling the tool again
        # This prevents duplicate tool executions that were causing multiple browser tabs
        
        # DEBUG: Log the tool result to see what browser_snapshot returns
        if tool_name == "browser_snapshot":
            logger.debug(f"🔍 browser_snapshot tool_result keys: {list(tool_result.keys())}")
            logger.debug(f"🔍 browser_snapshot content preview: {str(tool_result)[:500]}...")
        
        snapshot_data = {
            "tool_result": tool_result,
            "execution_time": tool_result.get("_metadata", {}).get("timestamp"),
            "parameters_used": parameters,
            "tool_name": tool_name
        }
        
        return snapshot_data
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.get("name", "unknown") for tool in self.available_tools]
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        return tool_name in self.get_tool_names()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        for tool in self.available_tools:
            if tool.get("name") == tool_name:
                return tool
        return None
    
    async def health_check(self) -> bool:
        """Check if the connection is still healthy"""
        try:
            # Simple ping request
            request = MCPRequest(
                method="ping",
                params={}
            )
            
            response = await self._send_request(request)
            return response.is_success
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def keep_alive(self):
        """Send keep-alive signal to maintain connection"""
        try:
            await self.health_check()
        except Exception as e:
            logger.warning(f"Keep-alive failed: {e}")


# Factory function for easy client creation
async def create_mcp_http_client(server_url: str, timeout: int = 30, browser_args: Optional[List[str]] = None) -> MCPHTTPClient:
    """Create and initialize MCP HTTP client"""
    client = MCPHTTPClient(server_url, timeout, browser_args=browser_args)
    await client.__aenter__()
    return client 
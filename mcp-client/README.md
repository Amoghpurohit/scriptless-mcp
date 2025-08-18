# Test Orchestrator for Playwright MCP

A powerful test orchestration system that combines LLM-driven natural language test steps with Playwright's browser automation capabilities via the Model Context Protocol (MCP).

## Overview

This test orchestrator provides multiple interfaces to interact with Playwright MCP:
- **Natural Language Interface**: Write tests in plain English
- **LLM-Powered Analysis**: Automatic test step parsing and validation
- **Persistent Connections**: Efficient SSE + HTTP JSON-RPC hybrid protocol
- **Rich Browser Automation**: Full access to Playwright's capabilities

## Installation

```bash
cd mcp-client
pip install -r requirements.txt
```

## Configuration

The orchestrator needs two main configurations:
1. **OpenAI API Key** for LLM-powered test analysis
2. **Playwright MCP Server** connection details

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Optional: Configure MCP server URL (defaults to http://localhost:3000)
export MCP_SERVER_URL=http://localhost:3000
```

## Available Browser Automation Tools

The orchestrator provides access to all Playwright MCP tools:

### Navigation & Interaction
- `browser_navigate`: Navigate to URLs
- `browser_click`: Click elements
- `browser_type`: Enter text
- `browser_press_key`: Keyboard input
- `browser_hover`: Mouse hover
- `browser_drag`: Drag and drop

### Browser Control
- `browser_tab_new`: Open new tabs
- `browser_tab_select`: Switch tabs
- `browser_tab_close`: Close tabs
- `browser_resize`: Window resizing
- `browser_close`: Close browser

### Observation & Validation
- `browser_snapshot`: Get page state
- `browser_take_screenshot`: Capture screenshots
- `browser_console_messages`: Get console output
- `browser_network_requests`: Monitor network
- `browser_wait_for`: Wait for conditions

## Usage

### Test Connection

```bash
python test_orchestrator.py test-connection
```

### Run Interactive Tests

```bash
python test_orchestrator.py interactive
```

Example natural language commands:
```
> Navigate to google.com
> Click the search box
> Type "Playwright automation"
> Press Enter
> Wait for search results
```

### Run Automated Test Sequence

```bash
python test_orchestrator.py run --test-file tests/search.txt
```

### Detailed JSON Session Logs

You can write a full JSON session log (LLM prompts/responses, JSON-RPC requests/responses, SSE events, tool calls, snapshots, step results) to `mcp-client/reports/`.

```bash
# Auto-named report under mcp-client/reports/session-<timestamp>.json
python test_orchestrator.py run "Verify site" "Navigate to example.com"

# Custom path
python test_orchestrator.py run "Verify site" "Navigate to example.com" --report-file mcp-client/reports/my_run.json

# Available for interactive and single-step as well
python test_orchestrator.py interactive "Explore" "Open homepage" --report-file mcp-client/reports/interactive.json
python test_orchestrator.py single-step "Navigate to github.com" --report-file mcp-client/reports/single.json
```

### Single Step Execution

```bash
python3 test_orchestrator.py single-step "Navigate to github.com and click the Sign In button"
```

## Architecture

### Protocol Implementation
- **SSE Connection**: Persistent event stream for server messages
- **HTTP JSON-RPC**: Request/response for tool calls
- **Session Management**: Automatic session handling and keep-alive

### Components
```
mcp-client/
├── orchestrator/
│   ├── __init__.py
│   ├── test_orchestrator.py    # Main orchestration logic
│   ├── llm_service.py         # LLM integration
│   ├── mcp_http_client.py     # MCP protocol client
│   └── context_manager.py     # Test context tracking
├── test_orchestrator.py       # CLI interface
└── requirements.txt           # Dependencies
```

## Key Features

### Test Automation
- **Natural Language Processing**: Write tests in plain English
- **Context Awareness**: Maintains test state and history
- **Automatic Validation**: LLM-powered pass/fail analysis
- **Rich Snapshots**: Detailed page state capture

### Protocol Support
- **Hybrid Protocol**: SSE + HTTP JSON-RPC
- **Connection Efficiency**: Single persistent connection
- **Automatic Recovery**: Connection retry and error handling
- **Session Management**: Automatic session tracking

### Development Features
- **Async/Sync APIs**: Both programming models supported
- **Type Hints**: Full Python type annotations
- **Rich Logging**: Detailed debug information
- **Error Handling**: Comprehensive error management

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM features | Required |
| `MCP_SERVER_URL` | Playwright MCP server URL | http://localhost:3000 |
| `LLM_MODEL` | OpenAI model to use | gpt-4 |
| `LOG_LEVEL` | Logging verbosity | INFO |

## Error Handling

The orchestrator provides detailed error information:
- **Protocol Errors**: Connection and communication issues
- **Tool Errors**: Browser automation failures
- **LLM Errors**: Natural language processing issues
- **Validation Errors**: Test step verification failures

## Troubleshooting

### Connection Issues
1. Verify Playwright MCP server is running: `npx @playwright/mcp@latest`
2. Check server URL configuration
3. Ensure proper network connectivity

### LLM Issues
1. Verify OpenAI API key is set
2. Check API rate limits
3. Ensure internet connectivity

### Browser Automation
1. Check if browser is installed: `npx playwright install`
2. Verify element selectors
3. Adjust timeouts for slow pages
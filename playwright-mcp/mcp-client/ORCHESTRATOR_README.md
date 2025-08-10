# 🤖 Test Orchestrator - LLM-Driven Test Automation

A sophisticated test automation system that uses Large Language Models (LLMs) to drive test execution through natural language prompts, integrated with Model Context Protocol (MCP) servers for action execution.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Test Orchestrator│◄──►│   LLM Service   │    │   MCP Client    │
│                 │    │                 │    │                 │
│ • NL Prompt     │    │ • OpenAI API    │    │ • HTTP JSON-RPC │
│ • Loop Control  │    │ • Context Mgmt  │    │ • Keep-Alive    │
│ • State Mgmt    │    │ • Rolling Summary│    │ • Snapshot Data │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   MCP Server    │
                    │                 │
                    │ • PDF Analysis  │
                    │ • Browser Tools │
                    │ • Action Exec   │
                    └─────────────────┘
```

## 🚀 Key Features

### **LLM Integration**
- **Natural Language Processing**: Convert human-readable test instructions into structured actions
- **Intelligent Analysis**: LLM analyzes results to determine pass/fail status
- **Context Management**: Rolling summary system maintains context across long test sequences
- **Adaptive Testing**: LLM suggests next steps based on current context and test objectives

### **MCP Integration**
- **HTTP JSON-RPC**: Persistent connections with keep-alive support
- **Tool Discovery**: Automatic discovery of available MCP tools
- **Snapshot Capture**: Detailed snapshot data from each action
- **Error Handling**: Robust error handling and retry mechanisms

### **Test Orchestration**
- **Loop Control**: Automated test loop with configurable limits
- **Interactive Mode**: User can override LLM suggestions
- **Session Management**: Complete test session tracking and reporting
- **Rich Output**: Beautiful console output with progress indicators

## 📦 Installation

1. **Install Dependencies**:
```bash
cd mcp-client
pip install -r requirements.txt
```

2. **Set Up Environment**:
```bash
cp orchestrator_config.env .env
# Edit .env with your OpenAI API key and MCP server URL
```

3. **Required Environment Variables**:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MCP_SERVER_URL="http://localhost:8000"  # Optional, defaults to localhost
```

## 🎯 Usage

### **Basic Commands**

**Test Connection**:
```bash
python test_orchestrator.py test-connection
```

**Run Automated Test**:
```bash
python test_orchestrator.py run "Verify PDF contains invoice data" "Extract text from first page"
```

**Interactive Testing**:
```bash
python test_orchestrator.py interactive "Test PDF workflow" "Start by extracting text" --pdf-path document.pdf
```

**Single Step Execution**:
```bash
python test_orchestrator.py single-step "Extract text from page 2 and check for errors" --pdf-path test.pdf
```

### **Advanced Usage**

**Custom MCP Server**:
```bash
python test_orchestrator.py --mcp-server-url http://remote-server:8000 run "Test objective" "First step"
```

**Save Detailed Report**:
```bash
python test_orchestrator.py run "Comprehensive test" "Extract and analyze" --output report.json
```

**Verbose Logging**:
```bash
python test_orchestrator.py --verbose run "Debug test" "Extract text"
```

## 🔄 Test Loop Flow

```
1. 📝 Natural Language Prompt
   ↓
2. 🧠 LLM Analysis (Parse → Structured Action)
   ↓
3. 🔧 MCP Client Request (HTTP JSON-RPC)
   ↓
4. ⚙️  Server Process + Action Execution
   ↓
5. 📸 Response + Snapshot Data
   ↓
6. 🤖 LLM Analysis (Pass/Fail Determination)
   ↓
7. 📊 Context Update (Rolling Summary)
   ↓
8. 🔄 Next Step Generation (or Complete)
```

## 🧩 Components

### **1. LLM Service** (`orchestrator/llm_service.py`)
- **OpenAI Integration**: Async API calls with retry logic
- **Prompt Engineering**: Structured prompts for parsing and analysis
- **Context Management**: Rolling summary to handle long conversations
- **Response Parsing**: JSON parsing with fallback handling

### **2. MCP HTTP Client** (`orchestrator/mcp_http_client.py`)
- **Persistent Connections**: Keep-alive HTTP sessions
- **JSON-RPC Protocol**: Full MCP protocol implementation
- **Tool Discovery**: Automatic tool enumeration
- **Snapshot Capture**: Detailed execution metadata

### **3. Context Manager** (`orchestrator/context_manager.py`)
- **Session Tracking**: Complete test session lifecycle
- **Rolling Summaries**: Intelligent context compression
- **Insight Extraction**: Pattern detection and analysis
- **Data Export**: JSON export for reporting

### **4. Test Orchestrator** (`orchestrator/test_orchestrator.py`)
- **Main Coordination**: Orchestrates all components
- **Loop Control**: Manages test execution flow
- **Rich UI**: Beautiful console output
- **Error Handling**: Comprehensive error management

## 📋 Example Test Scenarios

### **PDF Analysis Workflow**
```bash
python test_orchestrator.py run \
  "Verify PDF invoice contains required data" \
  "Extract text from first page and check for invoice number" \
  --pdf-path invoice.pdf
```

**Expected Flow**:
1. Extract text from page 1
2. LLM analyzes text for invoice patterns
3. If found, verify format and completeness
4. Check for required fields (date, amount, etc.)
5. Generate final validation report

### **Interactive Debugging**
```bash
python test_orchestrator.py interactive \
  "Debug PDF processing issues" \
  "Start by extracting text elements" \
  --pdf-path problematic.pdf
```

**Interactive Features**:
- Press Enter to accept LLM suggestions
- Type custom commands to override
- Type 'q' to quit at any time
- Full context maintained throughout session

### **Batch Testing**
```bash
# Test multiple aspects
python test_orchestrator.py run \
  "Comprehensive PDF validation" \
  "Extract text and analyze document structure" \
  --pdf-path document.pdf \
  --output comprehensive_report.json
```

## ⚙️ Configuration

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `MCP_SERVER_URL` | MCP server endpoint | `http://localhost:8000` |
| `LLM_MODEL` | OpenAI model to use | `gpt-4` |
| `MAX_TEST_STEPS` | Maximum steps per session | `50` |
| `STEP_TIMEOUT` | Timeout per step (seconds) | `30` |
| `LOG_LEVEL` | Logging level | `INFO` |

### **CLI Options**

```bash
python test_orchestrator.py --help

Options:
  --mcp-server-url TEXT     MCP server URL
  --openai-api-key TEXT     OpenAI API key
  --llm-model TEXT          LLM model to use
  --max-steps INTEGER       Maximum number of test steps
  --step-timeout INTEGER    Timeout for each step in seconds
  --verbose                 Enable verbose logging
```

## 📊 Output and Reporting

### **Console Output**
- **Rich UI**: Beautiful progress indicators and tables
- **Real-time Status**: Live updates during execution
- **Color-coded Results**: ✅ Pass, ❌ Fail, ❓ Inconclusive
- **Execution Metrics**: Timing and performance data

### **JSON Reports**
```json
{
  "session_id": "test_1703123456",
  "test_objective": "Verify PDF contains invoice data",
  "execution_summary": {
    "total_steps": 5,
    "passed_steps": 4,
    "failed_steps": 1,
    "inconclusive_steps": 0
  },
  "step_details": [...],
  "insights": [...],
  "final_context": "..."
}
```

## 🔧 Development

### **Adding New MCP Tools**
1. Implement tool in your MCP server
2. Orchestrator automatically discovers new tools
3. LLM learns to use tools through examples

### **Custom LLM Prompts**
Modify prompts in `llm_service.py`:
- `parse_natural_language_step()`: Step parsing
- `analyze_step_result()`: Result analysis
- `generate_next_step()`: Next step generation

### **Extending Context Management**
Add custom insight extraction in `context_manager.py`:
```python
def _extract_insights(self, step_result: TestStepResult):
    # Add your custom pattern detection
    if "custom_pattern" in step_result.mcp_response:
        self.insights.append("Custom insight detected")
```

## 🚨 Troubleshooting

### **Common Issues**

**❌ OpenAI API Key Missing**
```bash
export OPENAI_API_KEY="your_key_here"
# or use --openai-api-key flag
```

**❌ MCP Server Connection Failed**
```bash
# Check server is running
curl http://localhost:8000/health

# Test connection
python test_orchestrator.py test-connection
```

**❌ Step Timeout**
```bash
# Increase timeout
python test_orchestrator.py --step-timeout 60 run "..." "..."
```

**❌ LLM Parsing Errors**
- Check your prompts are clear and specific
- Use `--verbose` flag to see LLM responses
- Try different LLM models (gpt-3.5-turbo, gpt-4)

### **Debug Mode**
```bash
# Enable verbose logging
python test_orchestrator.py --verbose test-connection

# Single step debugging
python test_orchestrator.py single-step "debug step" --pdf-path test.pdf
```

## 🎯 Best Practices

### **Writing Effective Test Objectives**
- Be specific about what you want to verify
- Include success criteria
- Example: "Verify PDF invoice contains valid data and required fields"

### **Crafting Initial Prompts**
- Start with simple, clear actions
- Be specific about parameters
- Example: "Extract text from page 1 and look for invoice number"

### **Interactive Testing Tips**
- Use interactive mode for exploration
- Let LLM suggest next steps, override when needed
- Save successful sequences for automation

### **Performance Optimization**
- Use appropriate timeouts for your use case
- Monitor context length for long sessions
- Use single-step mode for debugging

## 📚 API Reference

### **TestOrchestrator Class**
```python
async with await create_test_orchestrator(
    mcp_server_url="http://localhost:8000",
    openai_api_key="your_key",
    llm_model="gpt-4",
    max_steps=50,
    step_timeout=30
) as orchestrator:
    
    report = await orchestrator.run_test_sequence(
        test_objective="Your test goal",
        initial_prompt="First step",
        pdf_path="optional_file.pdf",
        interactive=False
    )
```

### **Single Step Execution**
```python
result = await orchestrator.run_single_step(
    "Extract text from page 1",
    pdf_path="document.pdf"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

## 🎉 Getting Started

1. **Set up your environment**:
```bash
export OPENAI_API_KEY="your_key_here"
```

2. **Test the connection**:
```bash
python test_orchestrator.py test-connection
```

3. **Run your first test**:
```bash
python test_orchestrator.py run "Test PDF analysis" "Extract text from first page"
```

4. **Try interactive mode**:
```bash
python test_orchestrator.py interactive "Explore PDF" "Start with text extraction"
```

**You're ready to automate tests with natural language! 🚀** 
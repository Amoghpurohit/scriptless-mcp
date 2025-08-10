## Project Commands

Concise reference of runnable commands in this repository. Run these from the repo root unless stated otherwise.

## Start Playwright MCP server

- Recommended: run the server in a dedicated terminal and keep it running:
  - macOS/Linux: `npx @playwright/mcp@latest --port 5174`
  - Windows (PowerShell): `npx @playwright/mcp@latest --port 5174`
- Optional background start:
  - macOS/Linux: `npx @playwright/mcp@latest --port 5174 &`
  - Windows (PowerShell): `Start-Process npx -ArgumentList '@playwright/mcp@latest --port 5174'`

## Development

- **Run tests**: `pytest`
- **Format code**: `black .`
- **Lint code**: `flake8`

## Important quick run

- Ensure the MCP server is running (see section above), then from the `mcp-client` directory:
  - `python test_orchestrator.py --llm-model gpt-4o run --test-file tests/sample.txt`
    - Runs an entire test sequence from `tests/sample.txt` using the `gpt-4o` model.

## Test Orchestrator CLI (`mcp-client/test_orchestrator.py`)

Global options:
- **--mcp-server-url <URL>**: MCP server URL (default `http://localhost:8000`)
- **--openai-api-key <KEY>**: OpenAI API key (or set `OPENAI_API_KEY`)
- **--llm-model <MODEL>**: LLM model (default `gpt-4`)
- **--max-steps <N>**: Maximum number of steps (default 50)
- **--step-timeout <seconds>**: Timeout per step (default 30)
- **--verbose, -v**: Verbose logging

Commands:
- **test-connection**: Verify MCP server and LLM connectivity
  - Example (from repo root): `python mcp-client/test_orchestrator.py test-connection`

- **run "<test_objective>" "<initial_prompt>" [--test-file|-f <path>] [--output|-o <report.json>] [--report-file|-rf <log.json>]**: Run an automated test sequence
  - Use either `--test-file` or the two quoted arguments
  - Examples:
    - From repo root: `python mcp-client/test_orchestrator.py --llm-model gpt-4o run -f mcp-client/tests/sample.txt`
    - From `mcp-client`: `python test_orchestrator.py --llm-model gpt-4o run --test-file tests/sample.txt`

- **interactive "<test_objective>" "<initial_prompt>" [--output|-o <report.json>] [--report-file|-rf <log.json>]**: Run an interactive session
  - Example: `python mcp-client/test_orchestrator.py interactive "Explore" "Open homepage" --report-file mcp-client/reports/interactive.json`

- **single-step "<nl_prompt>" [--report-file|-rf <log.json>]**: Execute a single natural-language step
  - Example: `python mcp-client/test_orchestrator.py single-step "Navigate to github.com and click Sign in" --report-file mcp-client/reports/single.json`

- **examples**: Print usage examples
  - Example: `python mcp-client/test_orchestrator.py examples`

## Reports

- Detailed JSON session logs are written to `mcp-client/reports/` by default. Use `--report-file` to set a custom path/name.

## Setup Utilities

- **Setup client environment**: `python mcp-client/setup_client.py`
  - Creates `venv`, installs dependencies from `mcp-client/requirements.txt`, creates `.env`, and prepares directories.

## Test Utilities and Demos

- **Orchestrator demo suite**: `python mcp-client/test_orchestrator_demo.py`
- **Enhanced snapshot test**: `python mcp-client/test_enhanced_snapshot.py`
- **Debug snapshot (direct)**: `python mcp-client/debug_snapshot_direct.py`
- **Debug snapshot (alternative server)**: `python mcp-client/debug_snapshot_alternative.py`

## Miscellaneous

- **Hello world entrypoint**: `python src/main.py`
  - Prints a greeting; useful to verify Python environment is working.

## Environment Variables (commonly used)

- **OPENAI_API_KEY**: Required for orchestrator LLM features
- **MCP_SERVER_URL**: MCP server URL for orchestrator (default `http://localhost:8000`)

## Troubleshooting: Kill lingering MCP/Playwright Chrome profiles (fixes multiple tabs opening)

These commands help when multiple browser tabs keep opening due to orphaned MCP/Playwright processes or temporary Chrome profiles.

- **List suspicious processes (macOS/Linux):**
  - `pgrep -fal "mcp-chrome-profile"`
  - `pgrep -fal "@playwright/mcp\|playwright.*mcp"`
  - `pgrep -fal "Chromium .*--user-data-dir=.*mcp\|Chrome .*--user-data-dir=.*mcp"`

- **Kill MCP/Playwright servers:**
  - `pkill -f "@playwright/mcp"`
  - `pkill -f "playwright.*mcp"`

- **Kill Chrome/Chromium instances launched with MCP profiles:**
  - `pkill -f "mcp-chrome-profile"`
  - `pkill -f "--user-data-dir=.*mcp"`

- **If ports are stuck (replace PORT with actual):**
  - `lsof -i :PORT | awk 'NR>1 {print $2}' | xargs -r kill -9`

- **Start Playwright MCP server manually (background on port 5174):**
  - `cd /Users/amogh/Onedrive/desktop/playwright-mcp 4 && npx @playwright/mcp@latest --port 5174 &`
  - Starts the official MCP server in the background; useful when the orchestrator can't discover a running server.

Notes:
- Prefer the targeted pkill patterns above to avoid killing your personal Chrome sessions.
- If you know the exact profile name (e.g., `mcp-chrome-profile-XXXX`), you can further narrow the match: `pkill -f "mcp-chrome-profile-XXXX"`.


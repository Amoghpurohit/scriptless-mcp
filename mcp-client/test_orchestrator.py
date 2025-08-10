#!/usr/bin/env python
"""
Test Orchestrator - Main CLI Script
LLM-driven test automation with MCP integration

Usage:
    python test_orchestrator.py --help
    python test_orchestrator.py run "Verify PDF contains invoice data" "Extract text from page 1"
    python test_orchestrator.py interactive "Test PDF analysis workflow" "Start by extracting text"
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
import logging
import click
from rich.console import Console
from rich.logging import RichHandler

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import TestOrchestrator, create_test_orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@click.group()
@click.option('--mcp-server-url', 
              default=lambda: os.getenv('MCP_SERVER_URL', 'http://localhost:8000'),
              help='MCP server URL')
@click.option('--openai-api-key',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--llm-model',
              default=lambda: os.getenv('LLM_MODEL', 'gpt-4'),
              help='LLM model to use')
@click.option('--max-steps',
              type=int,
              default=lambda: int(os.getenv('MAX_STEPS', '50')),
              help='Maximum number of test steps')
@click.option('--step-timeout',
              type=int,
              default=lambda: int(os.getenv('STEP_TIMEOUT', '30')),
              help='Timeout for each step in seconds')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, mcp_server_url, openai_api_key, llm_model, max_steps, step_timeout, verbose):
    """Test Orchestrator - LLM-driven test automation"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from environment if not provided
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            console.print("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-api-key")
            sys.exit(1)
    
    ctx.ensure_object(dict)
    ctx.obj.update({
        'mcp_server_url': mcp_server_url,
        'openai_api_key': openai_api_key,
        'llm_model': llm_model,
        'max_steps': max_steps,
        'step_timeout': step_timeout,
        'verbose': verbose
    })


@cli.command()
@click.argument('test_objective', required=False)
@click.argument('initial_prompt', required=False)
@click.option('--test-file', '-f',
              type=click.Path(exists=True),
              help='Path to test file with step-by-step instructions')
@click.option('--pdf-path', '-p',
              type=click.Path(exists=True),
              help='Path to PDF file for testing')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for test report (JSON)')
@click.option('--report-file', '-rf',
              type=click.Path(),
              help='Write detailed session log JSON to this path (defaults to mcp-client/reports/session-<ts>.json)')
@click.pass_context
def run(ctx, test_objective, initial_prompt, test_file, pdf_path, output, report_file):
    """Run automated test sequence"""
    
    # Validate arguments
    if test_file and (test_objective or initial_prompt):
        console.print("❌ Cannot use both --test-file and test_objective/initial_prompt arguments")
        sys.exit(1)
    elif not test_file and (not test_objective or not initial_prompt):
        console.print("❌ Must provide either --test-file or both test_objective and initial_prompt")
        sys.exit(1)
    
    async def _run():
        # Derive report file from --test-file name when not provided
        derived_report_file = report_file
        if test_file and not report_file:
            from pathlib import Path as _P
            reports_dir = _P(__file__).parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            derived_report_file = str(reports_dir / f"{_P(test_file).stem}.json")

        async with await create_test_orchestrator(
            ctx.obj['mcp_server_url'],
            ctx.obj['openai_api_key'],
            llm_model=ctx.obj['llm_model'],
            max_steps=ctx.obj['max_steps'],
            step_timeout=ctx.obj['step_timeout'],
            report_file=derived_report_file
        ) as orchestrator:
            
            if test_file:
                # Run test from file
                report = await orchestrator.run_test_from_file(
                    test_file_path=test_file,
                    pdf_path=pdf_path
                )
            else:
                # Run test with manual arguments
                report = await orchestrator.run_test_sequence(
                    test_objective=test_objective,
                    initial_prompt=initial_prompt,
                    pdf_path=pdf_path,
                    interactive=False
                )
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2)
                console.print(f"📄 Report saved to: {output}")
            
            return report
    
    try:
        report = asyncio.run(_run())
        
        # Small delay to allow cleanup
        import time
        time.sleep(0.1)
        
        # Exit with appropriate code
        failed_steps = report.get('execution_summary', {}).get('failed_steps', 0)
        if failed_steps > 0:
            console.print(f"⚠️  Test completed with {failed_steps} failed steps")
            sys.exit(1)
        else:
            console.print("✅ All tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        console.print("\n🛑 Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"❌ Test execution failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('test_objective')
@click.argument('initial_prompt')
@click.option('--pdf-path', '-p',
              type=click.Path(exists=True),
              help='Path to PDF file for testing')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for test report (JSON)')
@click.option('--report-file', '-rf',
              type=click.Path(),
              help='Write detailed session log JSON to this path (defaults to mcp-client/reports/session-<ts>.json)')
@click.pass_context
def interactive(ctx, test_objective, initial_prompt, pdf_path, output, report_file):
    """Run interactive test sequence with user control"""
    
    async def _run():
        async with await create_test_orchestrator(
            ctx.obj['mcp_server_url'],
            ctx.obj['openai_api_key'],
            llm_model=ctx.obj['llm_model'],
            max_steps=ctx.obj['max_steps'],
            step_timeout=ctx.obj['step_timeout'],
            report_file=report_file
        ) as orchestrator:
            
            console.print("🎮 Interactive mode enabled")
            console.print("   - Press Enter to continue with LLM suggestion")
            console.print("   - Type 'q' to quit")
            console.print("   - Type custom step to override LLM")
            
            report = await orchestrator.run_test_sequence(
                test_objective=test_objective,
                initial_prompt=initial_prompt,
                pdf_path=pdf_path,
                interactive=True
            )
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2)
                console.print(f"📄 Report saved to: {output}")
            
            return report
    
    try:
        report = asyncio.run(_run())
        console.print("✅ Interactive session completed!")
        
    except KeyboardInterrupt:
        console.print("\n🛑 Interactive session interrupted")
        sys.exit(130)
    except Exception as e:
        console.print(f"❌ Interactive session failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('nl_prompt')
@click.option('--pdf-path', '-p',
              type=click.Path(exists=True),
              help='Path to PDF file for testing')
@click.option('--report-file', '-rf',
              type=click.Path(),
              help='Write detailed session log JSON to this path (defaults to mcp-client/reports/session-<ts>.json)')
@click.pass_context
def single_step(ctx, nl_prompt, pdf_path, report_file):
    """Execute a single test step"""
    
    async def _run():
        async with await create_test_orchestrator(
            ctx.obj['mcp_server_url'],
            ctx.obj['openai_api_key'],
            llm_model=ctx.obj['llm_model'],
            step_timeout=ctx.obj['step_timeout'],
            report_file=report_file
        ) as orchestrator:
            
            result = await orchestrator.run_single_step(nl_prompt, pdf_path)
            
            console.print("\n📋 Step Result:")
            console.print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED' if result.passed is False else '❓ INCONCLUSIVE'}")
            console.print(f"Action: {result.action_type}")
            console.print(f"Time: {result.execution_time:.2f}s")
            
            if result.error:
                console.print(f"Error: {result.error}")
            
            if 'reasoning' in result.llm_analysis:
                console.print(f"Analysis: {result.llm_analysis['reasoning']}")
            
            return result
    
    try:
        result = asyncio.run(_run())
        
        # Small delay to allow cleanup
        import time
        time.sleep(0.1)
        
        sys.exit(0 if result.passed else 1)
        
    except Exception as e:
        console.print(f"❌ Step execution failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connection to MCP server and LLM"""
    
    async def _test():
        console.print("🔍 Testing connections...")
        
        try:
            # Test MCP connection
            from orchestrator.mcp_http_client import create_mcp_http_client
            
            async with await create_mcp_http_client(ctx.obj['mcp_server_url']) as client:
                tools = client.get_tool_names()
                console.print(f"✅ MCP Server: Connected ({len(tools)} tools available)")
                console.print(f"   Tools: {', '.join(tools)}")
        
        except Exception as e:
            console.print(f"❌ MCP Server: Connection failed - {e}")
            return False
        
        try:
            # Test LLM connection
            from orchestrator.llm_service import LLMService
            
            llm = LLMService(api_key=ctx.obj['openai_api_key'], model=ctx.obj['llm_model'])
            test_step = await llm.parse_natural_language_step(
                "Test connection", 
                ["extract_text"]
            )
            console.print(f"✅ LLM Service: Connected (model: {ctx.obj['llm_model']})")
            
        except Exception as e:
            console.print(f"❌ LLM Service: Connection failed - {e}")
            return False
        
        console.print("🎉 All connections successful!")
        return True
    
    try:
        success = asyncio.run(_test())
        sys.exit(0 if success else 1)
        
    except Exception as e:
        console.print(f"❌ Connection test failed: {e}")
        sys.exit(1)


@cli.command()
def examples():
    """Show usage examples"""
    
    examples_text = """
🚀 Test Orchestrator Examples

1. Basic automated test:
   python test_orchestrator.py run "Verify PDF invoice data" "Extract text from first page"

2. Test with specific PDF:
   python test_orchestrator.py run "Check inspection report" "Extract text from page 1" --pdf-path report.pdf

3. Interactive testing:
   python test_orchestrator.py interactive "PDF analysis workflow" "Start by extracting text" -p document.pdf

4. Single step execution:
   python test_orchestrator.py single-step "Extract text from page 2 and check for errors" -p test.pdf

5. Test connections:
   python test_orchestrator.py test-connection

6. Custom MCP server:
   python test_orchestrator.py --mcp-server-url http://remote-server:8000 run "Test objective" "First step"

7. Save detailed report:
   python test_orchestrator.py run "Comprehensive test" "Extract and analyze" -o report.json

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- MCP_SERVER_URL: Default MCP server URL (optional)

Example Test Objectives:
- "Verify that the PDF contains valid invoice data"
- "Check if the inspection report has any critical issues"
- "Analyze document structure and extract key information"
- "Validate that all required fields are present in the form"

Example Initial Prompts:
- "Extract text from the first page"
- "Check the color of cells in the Critical column"
- "Get text elements and analyze their formatting"
- "Extract text and look for specific keywords"
"""
    
    console.print(examples_text)


if __name__ == '__main__':
    cli() 
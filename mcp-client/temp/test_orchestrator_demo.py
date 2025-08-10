#!/usr/bin/env python
"""
Test Orchestrator Demo Script
Demonstrates the test orchestrator functionality
"""

import asyncio
import os
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import create_test_orchestrator
from rich.console import Console

console = Console()


async def demo_basic_functionality():
    """Demo basic orchestrator functionality"""
    
    console.print("🚀 Test Orchestrator Demo", style="bold green")
    console.print("=" * 50)
    
    # Check environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("❌ OPENAI_API_KEY environment variable required")
        console.print("Set it with: export OPENAI_API_KEY='your_key_here'")
        return False
    
    mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    try:
        console.print(f"🔗 Connecting to MCP server: {mcp_server_url}")
        console.print(f"🤖 Using OpenAI API key: {api_key[:10]}...")
        
        async with await create_test_orchestrator(
            mcp_server_url=mcp_server_url,
            openai_api_key=api_key,
            llm_model="gpt-4",
            max_steps=5,  # Limit for demo
            step_timeout=30
        ) as orchestrator:
            
            console.print("✅ Orchestrator initialized successfully")
            
            # Demo 1: Single step execution
            console.print("\n📋 Demo 1: Single Step Execution")
            console.print("-" * 30)
            
            result = await orchestrator.run_single_step(
                "List available tools and describe what they do"
            )
            
            console.print(f"Step Result: {'✅ PASSED' if result.passed else '❌ FAILED' if result.passed is False else '❓ INCONCLUSIVE'}")
            console.print(f"Action: {result.action_type}")
            console.print(f"Time: {result.execution_time:.2f}s")
            
            # Demo 2: Short test sequence
            console.print("\n📋 Demo 2: Short Test Sequence")
            console.print("-" * 30)
            
            report = await orchestrator.run_test_sequence(
                test_objective="Demonstrate basic MCP tool usage",
                initial_prompt="Test connection to MCP server and list available tools",
                interactive=False
            )
            
            console.print("\n📊 Final Report Summary:")
            summary = report.get('execution_summary', {})
            console.print(f"Total Steps: {summary.get('total_steps', 0)}")
            console.print(f"Passed: {summary.get('passed_steps', 0)}")
            console.print(f"Failed: {summary.get('failed_steps', 0)}")
            
            return True
            
    except Exception as e:
        console.print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_interactive_mode():
    """Demo interactive mode (simulated)"""
    
    console.print("\n🎮 Demo 3: Interactive Mode Simulation")
    console.print("-" * 40)
    
    api_key = os.getenv('OPENAI_API_KEY')
    mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    try:
        async with await create_test_orchestrator(
            mcp_server_url=mcp_server_url,
            openai_api_key=api_key,
            max_steps=3
        ) as orchestrator:
            
            console.print("🎯 Simulating interactive session...")
            console.print("(In real interactive mode, you would provide input)")
            
            # Simulate a few manual steps
            steps = [
                "Check what tools are available",
                "Test a simple tool call",
                "Analyze the results"
            ]
            
            for i, step in enumerate(steps, 1):
                console.print(f"\n📍 Step {i}: {step}")
                result = await orchestrator.run_single_step(step)
                console.print(f"Result: {'✅' if result.passed else '❌' if result.passed is False else '❓'}")
            
            console.print("✅ Interactive simulation completed")
            return True
            
    except Exception as e:
        console.print(f"❌ Interactive demo failed: {e}")
        return False


async def demo_error_handling():
    """Demo error handling capabilities"""
    
    console.print("\n🚨 Demo 4: Error Handling")
    console.print("-" * 30)
    
    api_key = os.getenv('OPENAI_API_KEY')
    mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    try:
        async with await create_test_orchestrator(
            mcp_server_url=mcp_server_url,
            openai_api_key=api_key,
            step_timeout=5  # Short timeout to trigger errors
        ) as orchestrator:
            
            console.print("🎯 Testing error handling with invalid requests...")
            
            # Test with invalid tool call
            result = await orchestrator.run_single_step(
                "Call a non-existent tool called 'invalid_tool'"
            )
            
            console.print(f"Error handling result: {'✅' if result.error else '❓'}")
            if result.error:
                console.print(f"Error captured: {result.error[:100]}...")
            
            console.print("✅ Error handling demo completed")
            return True
            
    except Exception as e:
        console.print(f"❌ Error handling demo failed: {e}")
        return False


async def main():
    """Run all demos"""
    
    console.print("🤖 Test Orchestrator Complete Demo Suite")
    console.print("=" * 60)
    
    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        console.print("❌ Missing OPENAI_API_KEY environment variable")
        console.print("\nTo run this demo:")
        console.print("1. Set your OpenAI API key:")
        console.print("   export OPENAI_API_KEY='your_key_here'")
        console.print("2. Ensure MCP server is running (optional for some demos)")
        console.print("3. Run: python test_orchestrator_demo.py")
        return
    
    results = []
    
    # Run demos
    results.append(await demo_basic_functionality())
    results.append(await demo_interactive_mode())
    results.append(await demo_error_handling())
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("📊 Demo Results Summary")
    console.print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    console.print(f"Demos Passed: {passed}/{total}")
    
    if passed == total:
        console.print("🎉 All demos completed successfully!")
        console.print("\nNext steps:")
        console.print("1. Try the CLI: python test_orchestrator.py --help")
        console.print("2. Run a real test: python test_orchestrator.py run 'Test objective' 'First step'")
        console.print("3. Try interactive mode: python test_orchestrator.py interactive 'Explore' 'Start'")
    else:
        console.print("⚠️  Some demos failed. Check the output above for details.")
        console.print("\nTroubleshooting:")
        console.print("1. Ensure OPENAI_API_KEY is set correctly")
        console.print("2. Check MCP server is running if using MCP tools")
        console.print("3. Verify network connectivity")


if __name__ == '__main__':
    asyncio.run(main()) 
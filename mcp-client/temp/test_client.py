#!/usr/bin/env python3
"""
Test script for MCP Client
Verifies that the client can connect and communicate with the MCP server
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_client import create_client, analyze_pdf_async
from config import MCPClientConfig


async def test_basic_connection():
    """Test basic connection to MCP server"""
    print("🔍 Testing basic connection...")
    
    try:
        # Auto-detect server path
        server_path = Path(__file__).parent.parent / "server" / "main.py"
        
        if not server_path.exists():
            print(f"❌ Server script not found at: {server_path}")
            return False
        
        async with await create_client(str(server_path)) as client:
            print("✅ Successfully connected to MCP server")
            
            # Check tools discovered during initialization
            tool_names = client.get_tool_names()
            print(f"✅ Available tools: {tool_names}")
            
            # Verify expected tools are present
            expected_tools = ['extract_text', 'text_elements', 'color_at_position']
            missing_tools = [tool for tool in expected_tools if not client.has_tool(tool)]
            
            if missing_tools:
                print(f"⚠️  Missing expected tools: {missing_tools}")
            else:
                print("✅ All expected tools are available")
            
            # Test explicit list_tools call
            try:
                tools = await client.list_tools()
                print(f"✅ Explicit list_tools call returned {len(tools)} tools")
            except Exception as e:
                print(f"⚠️  Could not call list_tools: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def test_with_sample_pdf():
    """Test with a sample PDF if available"""
    print("\n🔍 Testing with sample PDF...")
    
    # Look for sample PDFs in examples directory
    examples_dir = Path(__file__).parent.parent / "examples"
    sample_pdfs = []
    
    if examples_dir.exists():
        sample_pdfs = list(examples_dir.glob("*.pdf"))
    
    if not sample_pdfs:
        print("⚠️  No sample PDFs found in examples directory")
        print("   Create a sample PDF to test full functionality")
        return True
    
    sample_pdf = sample_pdfs[0]
    print(f"📄 Using sample PDF: {sample_pdf}")
    
    try:
        server_path = Path(__file__).parent.parent / "server" / "main.py"
        
        # Test text extraction
        async with await create_client(str(server_path)) as client:
            result = await client.extract_text(str(sample_pdf), 0)
            if result.startswith("Error:"):
                print(f"⚠️  Text extraction had issues: {result[:100]}...")
            else:
                print(f"✅ Text extraction successful: {len(result)} characters")
        
        # Test comprehensive analysis
        results = await analyze_pdf_async(
            str(server_path),
            str(sample_pdf),
            ['extract_text', 'text_elements']
        )
        
        print(f"✅ Comprehensive analysis completed: {len(results)} operations")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF analysis failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        config = MCPClientConfig.from_env()
        print(f"✅ Configuration loaded")
        print(f"   Server script: {config.server_script_path}")
        print(f"   Timeout: {config.timeout}")
        print(f"   Log level: {config.log_level}")
        
        # Test Jenkins config
        jenkins_config = MCPClientConfig.for_jenkins()
        print(f"✅ Jenkins configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting MCP Client Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Basic Connection", test_basic_connection),
        ("Sample PDF Analysis", test_with_sample_pdf)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - check configuration and setup")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
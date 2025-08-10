#!/usr/bin/env python
"""
Debug script to directly test browser_snapshot tool
"""
import asyncio
import os
import json
from orchestrator.mcp_http_client import MCPHTTPClient

async def test_snapshot():
    # Setup
    mcp_server_url = "http://localhost:5175"
    
    # Create client
    client = MCPHTTPClient(mcp_server_url)
    
    try:
        async with client:
            print("🔗 Connected to MCP server")
            
            # Wait for tools to be discovered
            print("⏳ Waiting for tool discovery...")
            await asyncio.sleep(3)
            
            print(f"🔧 Available tools: {client.get_tool_names()}")
            
            if not client.has_tool("browser_navigate"):
                print("❌ browser_navigate not available")
                return
                
            # Navigate to Google first
            print("🌐 Navigating to Google...")
            nav_result = await client.call_tool("browser_navigate", {"url": "https://google.com"})
            print(f"📍 Navigation result: {nav_result}")
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Try different tools that might return page structure
            tools_to_test = [
                ("browser_snapshot", {}),
                ("browser_evaluate", {"function": "() => document.documentElement.outerHTML"}),
                ("browser_evaluate", {"function": "() => document.title"}),
                ("browser_console_messages", {}),
                ("browser_take_screenshot", {})
            ]
            
            for tool_name, params in tools_to_test:
                print(f"\n📸 Testing {tool_name}...")
                try:
                    result = await client.call_tool(tool_name, params)
                    print(f"🔍 {tool_name} result keys:", list(result.keys()))
                    
                    # Check for content
                    result_str = str(result)
                    if len(result_str) > 200:  # More than just metadata
                        print(f"🔍 {tool_name} content preview (first 500 chars):")
                        print("-" * 30)
                        print(result_str[:500])
                        print("-" * 30)
                    else:
                        print(f"🔍 {tool_name} - minimal response:", result_str)
                        
                    # Check for element references
                    if "ref=" in result_str:
                        print(f"🎯 Found 'ref=' in {tool_name}!")
                        
                except Exception as e:
                    print(f"❌ {tool_name} failed: {e}")
                    
                await asyncio.sleep(1)
            
            # Check for specific fields
            if 'content' in snapshot_result:
                print("📋 Found 'content' field")
                content = snapshot_result['content']
                print(f"📋 Content type: {type(content)}")
                print(f"📋 Content preview: {str(content)[:500]}...")
            
            if 'result' in snapshot_result:
                print("📋 Found 'result' field")
                result = snapshot_result['result']
                print(f"📋 Result type: {type(result)}")
                print(f"📋 Result preview: {str(result)[:500]}...")
                
            # Check for any field containing "ref=" or element references
            for key, value in snapshot_result.items():
                value_str = str(value)
                if "ref=" in value_str:
                    print(f"🎯 Found 'ref=' in field '{key}'!")
                    print(f"🎯 Field content: {value_str[:1000]}...")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_snapshot())
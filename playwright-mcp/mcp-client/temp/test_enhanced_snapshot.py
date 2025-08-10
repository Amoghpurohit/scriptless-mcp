#!/usr/bin/env python
"""
Test enhanced MCP client with actual execution results
"""
import asyncio
import json
from orchestrator.mcp_http_client import MCPHTTPClient

async def test_enhanced_snapshot():
    """Test if the enhanced client gets actual browser_snapshot results"""
    mcp_server_url = "http://localhost:5174"
    
    client = MCPHTTPClient(mcp_server_url)
    
    try:
        async with client:
            print("🔗 Connected to MCP server")
            
            # Wait for tools
            await asyncio.sleep(3)
            print(f"🔧 Available tools: {len(client.get_tool_names())} tools")
            
            # Navigate to Google
            print("🌐 Navigating to Google...")
            nav_result = await client.call_tool("browser_navigate", {"url": "https://google.com"})
            print(f"📍 Navigation result: {json.dumps(nav_result, indent=2)}")
            
            # Take snapshot
            print("\n📸 Taking browser snapshot...")
            snapshot_result = await client.call_tool("browser_snapshot", {})
            print(f"🔍 Snapshot result: {json.dumps(snapshot_result, indent=2)}")
            
            # Check for element references
            snapshot_str = str(snapshot_result)
            if 'ref=' in snapshot_str:
                print("✅ FOUND element references in snapshot!")
                refs = [part for part in snapshot_str.split() if 'ref=' in part]
                print(f"🎯 Found {len(refs)} element references: {refs[:10]}...")  # Show first 10
            else:
                print("❌ No element references found in snapshot")
                
            # Check for page content
            if 'page_snapshot' in snapshot_result.get('result', {}):
                print("✅ FOUND page_snapshot data!")
            elif any(key in str(snapshot_result).lower() for key in ['page', 'url', 'title', 'html']):
                print("✅ FOUND page content data!")
            else:
                print("❌ No detailed page content found")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_snapshot())
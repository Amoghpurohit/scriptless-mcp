import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64
from pathlib import Path


class PDFAnalyzerMCPClient:
    """MCP Client for PDF Analyzer tools"""
    
    def __init__(self, server_command: List[str], is_remote: bool = False, remote_url: Optional[str] = None):
        self.server_command = server_command
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.is_remote = is_remote
        self.remote_url = remote_url
        
    async def __aenter__(self):
        """Async context manager entry"""
        if self.is_remote:
            # For remote servers, we'll implement HTTP-based MCP later
            # For now, fall back to local mode
            print("⚠️  Remote server mode not fully implemented yet, using local mode")
        
        server_params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:] if len(self.server_command) > 1 else []
        )
        
        # stdio_client returns an async context manager
        self.stdio = stdio_client(server_params)
        self.read, self.write = await self.stdio.__aenter__()
        
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        
        # Initialize the session (handshake)
        await self.session.initialize()
        
        # Discover available tools (proper MCP protocol)
        try:
            tools_response = await self.session.list_tools()
            self.available_tools = [tool.model_dump() for tool in tools_response.tools]
        except Exception as e:
            print(f"Warning: Could not list tools: {e}")
            self.available_tools = []
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if hasattr(self, 'stdio'):
            await self.stdio.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _upload_file_if_needed(self, pdf_path: str) -> str:
        """Upload file to remote server if needed, return path to use"""
        if not self.is_remote:
            # Local server - use original path
            return pdf_path
        
        # Remote server - upload file and get remote path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Local PDF file not found: {pdf_path}")
        
        print(f"📤 Uploading {Path(pdf_path).name} to remote server...")
        
        # Read file content
        with open(pdf_path, 'rb') as f:
            file_content = f.read()
        
        # Encode for transport
        file_data = base64.b64encode(file_content).decode('utf-8')
        filename = Path(pdf_path).name
        
        # Call upload tool on server
        try:
            result = await self.session.call_tool("upload_pdf", {
                "filename": filename,
                "file_data": file_data
            })
            
            # Extract remote path from result
            remote_path = result.content[0].text.strip()
            print(f"✅ File uploaded as: {remote_path}")
            return remote_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to upload file: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        if self.available_tools:
            # Return cached tools discovered during initialization
            return self.available_tools
        
        try:
            # Call the proper MCP protocol method
            tools_response = await self.session.list_tools()
            self.available_tools = [tool.model_dump() for tool in tools_response.tools]
            return self.available_tools
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []
    
    def get_tool_names(self) -> List[str]:
        """Get just the names of available tools"""
        return [tool.get('name', 'unknown') for tool in self.available_tools]
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        return tool_name in self.get_tool_names()
    
    async def extract_text(self, pdf_path: str, page_num: int = 0) -> str:
        """Extract text from PDF page"""
        try:
            # Handle file upload for remote servers
            server_path = await self._upload_file_if_needed(pdf_path)
            
            result = await self.session.call_tool("extract_text", {
                "pdf_path": server_path,
                "page_num": page_num
            })
            return result.content[0].text if result.content else "No content returned"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def get_text_elements(self, pdf_path: str, page_num: int = 0) -> str:
        """Get text elements with properties"""
        try:
            # Handle file upload for remote servers
            server_path = await self._upload_file_if_needed(pdf_path)
            
            result = await self.session.call_tool("text_elements", {
                "pdf_path": server_path,
                "page_num": page_num
            })
            return result.content[0].text if result.content else "No content returned"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def get_color_at_position(self, pdf_path: str, column_name: str, cell_number: int, page_num: int = 0) -> str:
        """Get color information for a specific cell"""
        try:
            # Handle file upload for remote servers
            server_path = await self._upload_file_if_needed(pdf_path)
            
            result = await self.session.call_tool("color_at_position", {
                "pdf_path": server_path,
                "column_name": column_name,
                "cell_number": cell_number,
                "page_num": page_num
            })
            return result.content[0].text if result.content else "No content returned"
        except Exception as e:
            return f"Error: {str(e)}"


class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass


async def create_client(server_script_path: str, remote_url: Optional[str] = None) -> PDFAnalyzerMCPClient:
    """Factory function to create MCP client"""
    is_remote = remote_url is not None
    
    if is_remote:
        # For remote servers, we might use different connection method
        # For now, we'll still use subprocess but with upload capability
        print(f"🌐 Configuring for remote server: {remote_url}")
    
    python_path = sys.executable
    server_command = [python_path, server_script_path]
    return PDFAnalyzerMCPClient(server_command, is_remote=is_remote, remote_url=remote_url)


# Convenience functions for direct usage
async def analyze_pdf_async(server_script_path: str, pdf_path: str, operations: List[str]) -> Dict[str, Any]:
    """
    Analyze PDF with specified operations
    
    Args:
        server_script_path: Path to your MCP server script
        pdf_path: Path to PDF file to analyze
        operations: List of operations to perform ['extract_text', 'text_elements', 'color_analysis']
    
    Returns:
        Dictionary with results of each operation
    """
    results = {}
    
    async with await create_client(server_script_path) as client:
        if 'extract_text' in operations:
            results['text_content'] = await client.extract_text(pdf_path)
        
        if 'text_elements' in operations:
            results['text_elements'] = await client.get_text_elements(pdf_path)
        
        if 'color_analysis' in operations:
            # Example color analysis - you might want to make this configurable
            results['color_analysis'] = {}
            for column in ['Minor', 'Major', 'Critical']:
                for cell_num in range(1, 6):  # Check first 5 cells
                    try:
                        key = f"{column}_cell_{cell_num}"
                        results['color_analysis'][key] = await client.get_color_at_position(
                            pdf_path, column, cell_num
                        )
                    except Exception as e:
                        results['color_analysis'][key] = f"Error: {str(e)}"
    
    return results


def analyze_pdf(server_script_path: str, pdf_path: str, operations: List[str]) -> Dict[str, Any]:
    """Synchronous wrapper for PDF analysis"""
    return asyncio.run(analyze_pdf_async(server_script_path, pdf_path, operations))
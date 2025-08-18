#!/usr/bin/env python3
"""
CLI interface for PDF Analyzer MCP Client
Perfect for Jenkins pipelines and automation
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from dotenv import load_dotenv

from .mcp_client import analyze_pdf_async, create_client, MCPClientError
from .config import config


# Load environment variables
load_dotenv()


@click.group() # creating a main command (cli)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--remote-url', help='Remote server URL (overrides environment)')
@click.pass_context # Ask for context object (ctx)
def cli(ctx, verbose: bool, remote_url: str):
    """PDF Analyzer MCP Client - Works with both local and remote servers"""
    ctx.ensure_object(dict)
    
    # Override remote URL if provided
    if remote_url:
        config.remote_server_url = remote_url
        config.is_local_server = False
    
    ctx.obj['verbose'] = verbose or config.verbose
    
    if ctx.obj['verbose']:
        config.print_config(verbose=True)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--page', '-p', default=0, help='Page number to extract (0-based)')
@click.option('--output', '-o', type=click.File('w'), help='Output file (default: stdout)')
@click.pass_context # Ask for context object (ctx) in the subcommand 
def extract_text(ctx, pdf_path: str, page: int, output):
    """Extract text from a PDF page"""
    async def _extract():
        async with await create_client(ctx.obj['server_path']) as client: # create client with server path from context object stored in cli(main command)
            result = await client.extract_text(pdf_path, page)
            return result
    
    result = asyncio.run(_extract())
    
    if output:
        output.write(result)
    else:
        click.echo(result)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--page', '-p', default=0, help='Page number to analyze (0-based)')
@click.option('--output', '-o', type=click.File('w'), help='Output file (default: stdout)')
@click.pass_context
def text_elements(ctx, pdf_path: str, page: int, output):
    """Get text elements with properties from a PDF page"""
    async def _analyze():
        async with await create_client(ctx.obj['server_path']) as client:
            result = await client.get_text_elements(pdf_path, page)
            return result
    
    result = asyncio.run(_analyze())
    
    if output:
        output.write(result)
    else:
        click.echo(result)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='Column name (e.g., Minor, Major, Critical)')
@click.option('--cell', '-n', required=True, type=int, help='Cell number (1-based)')
@click.option('--page', '-p', default=0, help='Page number to analyze (0-based)')
@click.option('--output', '-o', type=click.File('w'), help='Output file (default: stdout)')
@click.pass_context
def color_analysis(ctx, pdf_path: str, column: str, cell: int, page: int, output):
    """Get color information for a specific cell"""
    async def _analyze():
        async with await create_client(ctx.obj['server_path']) as client:
            result = await client.get_color_at_position(pdf_path, column, cell, page)
            return result
    
    result = asyncio.run(_analyze())
    
    if output:
        output.write(result)
    else:
        click.echo(result)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--operations', '-op', 
              multiple=True,
              type=click.Choice(['extract_text', 'text_elements', 'color_analysis']),
              default=['extract_text'],
              help='Operations to perform (can specify multiple)')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'text']),
              default='json',
              help='Output format')
@click.pass_context
def analyze(ctx, pdf_path: str, operations: List[str], output: Optional[str], output_format: str):
    """Comprehensive PDF analysis with multiple operations"""
    
    async def _analyze():
        return await analyze_pdf_async(ctx.obj['server_path'], pdf_path, list(operations))
    
    results = asyncio.run(_analyze())
    
    if output_format == 'json':
        output_data = json.dumps(results, indent=2)
    else:
        # Text format
        output_data = ""
        for operation, result in results.items():
            output_data += f"=== {operation.upper()} ===\n"
            output_data += str(result)
            output_data += "\n\n"
    
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connection to MCP server and list available tools"""
    verbose = ctx.obj['verbose']
    
    async def test():
        try:
            if verbose:
                print("🔗 Testing connection to PDF Analyzer MCP server...")
                config.print_config()
            
            server_config = config.get_server_config()
            client = await create_client(
                server_config['server_script_path'],
                server_config['remote_url']
            )
            
            async with client:
                tools = await client.list_tools()
                tool_names = client.get_tool_names()
                
                print("✅ Successfully connected to MCP server")
                print(f"✅ Available tools: {len(tools)}")
                
                if verbose:
                    print("\n🔧 Available Tools:")
                    for i, name in enumerate(tool_names, 1):
                        print(f"   {i}. {name}")
                        
                        # Find tool details
                        tool_details = next((t for t in tools if t.get('name') == name), {})
                        if tool_details.get('description'):
                            print(f"      {tool_details['description']}")
                    
                    print(f"\n📊 Server Configuration:")
                    for key, value in server_config.items():
                        print(f"   {key}: {value}")
                
                return 0
                
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    result = asyncio.run(test())
    sys.exit(result)


# Jenkins-specific command
@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output-file', '-o', help='Output file for results')
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'xml', 'junit']), help='Output format')
@click.pass_context
def jenkins_analyze(ctx, pdf_path: str, output_file: str, output_format: str):
    """Jenkins-optimized analysis with structured output"""
    verbose = ctx.obj['verbose']
    
    async def jenkins_analysis():
        try:
            # Validate file for upload if needed
            config.validate_file_for_upload(pdf_path)
            
            if verbose:
                print(f"🚀 Jenkins analysis of: {pdf_path}")
                if not config.is_local_server:
                    print(f"📤 Will upload file to remote server first...")
            
            server_config = config.get_server_config()
            client = await create_client(
                server_config['server_script_path'],
                server_config['remote_url']
            )
            
            # Collect all analysis data
            analysis_results = {}
            
            async with client:
                # Extract text
                text_result = await client.extract_text(pdf_path, 0)
                analysis_results['text_content'] = text_result
                
                # Get text elements
                elements_result = await client.get_text_elements(pdf_path, 0)
                analysis_results['text_elements'] = elements_result
                
                # Try color analysis for common columns
                color_results = {}
                for column in ['Minor', 'Major', 'Critical']:
                    try:
                        color_result = await client.get_color_at_position(pdf_path, column, 1, 0)
                        color_results[column] = color_result
                    except:
                        color_results[column] = "No data available"
                
                analysis_results['color_analysis'] = color_results
                analysis_results['file_info'] = {
                    'file_path': pdf_path,
                    'file_name': Path(pdf_path).name,
                    'server_mode': 'remote' if not config.is_local_server else 'local'
                }
            
            # Format output
            if output_format == 'json':
                formatted_output = json.dumps(analysis_results, indent=2)
            elif output_format == 'xml':
                formatted_output = _format_as_xml(analysis_results)
            elif output_format == 'junit':
                formatted_output = _format_as_junit(analysis_results)
            else:
                formatted_output = str(analysis_results)
            
            # Output results
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(formatted_output)
                print(f"✅ Results written to: {output_file}")
            else:
                print(formatted_output)
            
            return 0
            
        except Exception as e:
            print(f"❌ Jenkins analysis failed: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 2  # Jenkins failure code
    
    result = asyncio.run(jenkins_analysis())
    sys.exit(result)


@cli.command()
@click.pass_context
def config_info(ctx: click.Context):
    """Show current configuration"""
    verbose = ctx.obj['verbose']
    config.print_config(verbose=True)
    
    print(f"\n🌐 Server Connection Test:")
    # Test if we can reach the server
    test_ctx = click.Context(test_connection)
    test_ctx.obj = ctx.obj
    test_connection.invoke(test_ctx)


def _format_as_xml(data: Dict[str, Any]) -> str:
    """Format analysis results as XML"""
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<pdf_analysis>\n'
    
    for key, value in data.items():
        xml += f'  <{key}>\n'
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                xml += f'    <{subkey}><![CDATA[{subvalue}]]></{subkey}>\n'
        else:
            xml += f'    <![CDATA[{value}]]>\n'
        xml += f'  </{key}>\n'
    
    xml += '</pdf_analysis>'
    return xml


def _format_as_junit(data: Dict[str, Any]) -> str:
    """Format analysis results as JUnit XML"""
    junit = '<?xml version="1.0" encoding="UTF-8"?>\n'
    junit += '<testsuite name="PDF Analysis" tests="3" failures="0" errors="0">\n'
    
    # Text extraction test
    junit += '  <testcase name="Text Extraction" classname="PDFAnalysis">\n'
    junit += f'    <system-out><![CDATA[{data.get("text_content", "")}]]></system-out>\n'
    junit += '  </testcase>\n'
    
    # Text elements test
    junit += '  <testcase name="Text Elements Analysis" classname="PDFAnalysis">\n'
    junit += f'    <system-out><![CDATA[{data.get("text_elements", "")}]]></system-out>\n'
    junit += '  </testcase>\n'
    
    # Color analysis test
    junit += '  <testcase name="Color Analysis" classname="PDFAnalysis">\n'
    color_output = json.dumps(data.get("color_analysis", {}), indent=2)
    junit += f'    <system-out><![CDATA[{color_output}]]></system-out>\n'
    junit += '  </testcase>\n'
    
    junit += '</testsuite>'
    return junit


if __name__ == '__main__':
    cli()
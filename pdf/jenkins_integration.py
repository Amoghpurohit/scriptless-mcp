"""
Jenkins Integration Module
Provides utilities for seamless Jenkins pipeline integration
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import tempfile

from config import MCPClientConfig, is_jenkins
from mcp_client import analyze_pdf


class JenkinsReporter:
    """Handles Jenkins-specific reporting and artifacts"""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or os.getenv('WORKSPACE', os.getcwd())
        self.artifacts_dir = Path(self.workspace_dir) / "artifacts"
        self.reports_dir = Path(self.workspace_dir) / "reports"
        
        # Ensure directories exist
        self.artifacts_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def create_junit_report(self, test_results: Dict[str, Any], output_file: str = "pdf-analysis-results.xml"):
        """Create JUnit-style XML report for Jenkins"""
        junit_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="PDF Analysis" tests="{len(test_results)}" failures="0" errors="0" time="0">
"""
        
        for test_name, result in test_results.items():
            status = "passed" if not str(result).startswith("Error:") else "failed"
            junit_content += f"""    <testcase classname="PDFAnalysis" name="{test_name}" time="0">
"""
            if status == "failed":
                junit_content += f"""        <failure message="Analysis failed">{result}</failure>
"""
            junit_content += """    </testcase>
"""
        
        junit_content += "</testsuite>"
        
        output_path = self.reports_dir / output_file
        with open(output_path, 'w') as f:
            f.write(junit_content)
        
        return str(output_path)
    
    def create_html_report(self, analysis_results: Dict[str, Any], pdf_path: str, output_file: str = "analysis-report.html"):
        """Create HTML report for Jenkins"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PDF Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .error {{ color: red; background-color: #ffe6e6; }}
        .success {{ color: green; background-color: #e6ffe6; }}
        pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PDF Analysis Report</h1>
        <p class="metadata">File: {pdf_path}</p>
        <p class="metadata">Generated: {analysis_results.get('timestamp', 'Unknown')}</p>
        <p class="metadata">Status: {analysis_results.get('status', 'Unknown')}</p>
    </div>
"""
        
        # Add results sections
        for operation, result in analysis_results.get('results', {}).items():
            section_class = "error" if str(result).startswith("Error:") else "success"
            html_content += f"""
    <div class="section {section_class}">
        <h2>{operation.replace('_', ' ').title()}</h2>
        <pre>{result}</pre>
    </div>
"""
        
        # Add errors section if any
        if analysis_results.get('errors'):
            html_content += """
    <div class="section error">
        <h2>Errors</h2>
        <ul>
"""
            for error in analysis_results['errors']:
                html_content += f"            <li>{error}</li>\n"
            html_content += """        </ul>
    </div>
"""
        
        html_content += """
</body>
</html>"""
        
        output_path = self.reports_dir / output_file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def publish_artifacts(self, files: List[str]):
        """Copy files to Jenkins artifacts directory"""
        for file_path in files:
            if os.path.exists(file_path):
                dest_path = self.artifacts_dir / Path(file_path).name
                import shutil
                shutil.copy2(file_path, dest_path)
                print(f"Published artifact: {dest_path}")


class JenkinsIntegration:
    """Main Jenkins integration class"""
    
    def __init__(self, config: Optional[MCPClientConfig] = None):
        self.config = config or MCPClientConfig.for_jenkins()
        self.reporter = JenkinsReporter()
        
    def run_analysis_pipeline(self, pdf_path: str, operations: List[str] = None) -> Dict[str, Any]:
        """Run complete analysis pipeline optimized for Jenkins"""
        operations = operations or ['extract_text', 'text_elements', 'color_analysis']
        
        # Ensure directories exist
        self.config.ensure_directories()
        
        print(f"🔄 Starting PDF analysis pipeline for: {pdf_path}")
        print(f"📊 Operations: {', '.join(operations)}")
        
        try:
            # Run analysis
            results = analyze_pdf(self.config.server_script_path, pdf_path, operations)
            
            # Create comprehensive report
            report = {
                "pdf_path": pdf_path,
                "timestamp": self._get_timestamp(),
                "status": "success",
                "results": results,
                "errors": [],
                "jenkins_info": self._get_jenkins_info()
            }
            
            # Check for errors in results
            for op, result in results.items():
                if isinstance(result, str) and result.startswith("Error:"):
                    report["errors"].append(f"{op}: {result}")
            
            if report["errors"]:
                report["status"] = "partial_success"
            
            print(f"✅ Analysis completed with status: {report['status']}")
            
            return report
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return {
                "pdf_path": pdf_path,
                "timestamp": self._get_timestamp(),
                "status": "failed",
                "results": {},
                "errors": [f"Pipeline failed: {str(e)}"],
                "jenkins_info": self._get_jenkins_info()
            }
    
    def generate_reports(self, analysis_report: Dict[str, Any], pdf_path: str):
        """Generate all report formats"""
        print("📝 Generating reports...")
        
        # JSON report
        json_report_path = self.reporter.reports_dir / "analysis-report.json"
        with open(json_report_path, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        # HTML report
        html_report_path = self.reporter.create_html_report(analysis_report, pdf_path)
        
        # JUnit XML for test results
        junit_report_path = self.reporter.create_junit_report(analysis_report.get('results', {}))
        
        print(f"📄 JSON Report: {json_report_path}")
        print(f"🌐 HTML Report: {html_report_path}")
        print(f"🧪 JUnit Report: {junit_report_path}")
        
        # Publish artifacts
        self.reporter.publish_artifacts([
            str(json_report_path),
            str(html_report_path),
            str(junit_report_path)
        ])
        
        return {
            "json_report": str(json_report_path),
            "html_report": str(html_report_path),
            "junit_report": str(junit_report_path)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _get_jenkins_info(self) -> Dict[str, Any]:
        """Get Jenkins environment information"""
        jenkins_vars = [
            'BUILD_NUMBER', 'BUILD_ID', 'BUILD_URL', 'JOB_NAME', 
            'WORKSPACE', 'NODE_NAME', 'EXECUTOR_NUMBER'
        ]
        
        info = {}
        for var in jenkins_vars:
            value = os.getenv(var)
            if value:
                info[var.lower()] = value
        
        return info


def main():
    """Main entry point for Jenkins integration"""
    if len(sys.argv) < 2:
        print("Usage: python jenkins_integration.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Initialize Jenkins integration
    integration = JenkinsIntegration()
    
    # Run analysis pipeline
    analysis_report = integration.run_analysis_pipeline(pdf_path)
    
    # Generate reports
    report_paths = integration.generate_reports(analysis_report, pdf_path)
    
    # Print summary for Jenkins console
    print("\n" + "="*50)
    print("📊 ANALYSIS SUMMARY")
    print("="*50)
    print(f"Status: {analysis_report['status']}")
    print(f"PDF: {pdf_path}")
    print(f"Operations: {len(analysis_report['results'])}")
    
    if analysis_report['errors']:
        print(f"Errors: {len(analysis_report['errors'])}")
        for error in analysis_report['errors']:
            print(f"  ❌ {error}")
    
    print("\n📋 Reports Generated:")
    for report_type, path in report_paths.items():
        print(f"  📄 {report_type}: {path}")
    
    # Exit with appropriate code
    if analysis_report['status'] == 'failed':
        sys.exit(1)
    elif analysis_report['status'] == 'partial_success':
        sys.exit(2)  # Warning
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
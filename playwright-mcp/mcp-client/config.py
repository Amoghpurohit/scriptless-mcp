"""
Configuration management for MCP Client
Handles environment-specific settings for different deployment contexts
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class MCPClientConfig:
    """Configuration for MCP Client"""
    server_script_path: str
    server_host: Optional[str] = None
    server_port: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3
    log_level: str = "INFO"
    output_dir: str = "output"
    temp_dir: str = "temp"
    
    @classmethod
    def from_env(cls) -> 'MCPClientConfig':
        """Create configuration from environment variables"""
        load_dotenv()
        
        # Auto-detect server script path
        server_script = os.getenv('MCP_SERVER_SCRIPT')
        if not server_script:
            # Try to find it relative to this file
            current_dir = Path(__file__).parent
            server_script = str(current_dir.parent / "server" / "main.py")
        
        return cls(
            server_script_path=server_script,
            server_host=os.getenv('MCP_SERVER_HOST'),
            server_port=int(os.getenv('MCP_SERVER_PORT', '0')) or None,
            timeout=int(os.getenv('MCP_TIMEOUT', '30')),
            max_retries=int(os.getenv('MCP_MAX_RETRIES', '3')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            output_dir=os.getenv('OUTPUT_DIR', 'output'),
            temp_dir=os.getenv('TEMP_DIR', 'temp')
        )
    
    @classmethod
    def for_jenkins(cls) -> 'MCPClientConfig':
        """Create Jenkins-optimized configuration"""
        config = cls.from_env()
        
        # Jenkins-specific overrides
        config.log_level = "DEBUG" if os.getenv('JENKINS_DEBUG') else "INFO"
        config.output_dir = os.getenv('WORKSPACE', 'output')
        config.temp_dir = os.path.join(config.output_dir, 'temp')
        
        return config
    
    def ensure_directories(self):
        """Ensure output and temp directories exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[MCPClientConfig] = None


def get_config() -> MCPClientConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = MCPClientConfig.from_env()
    return _config


def set_config(config: MCPClientConfig):
    """Set the global configuration instance"""
    global _config
    _config = config


# Environment detection
def is_jenkins() -> bool:
    """Check if running in Jenkins environment"""
    return bool(os.getenv('JENKINS_URL') or os.getenv('BUILD_NUMBER'))


def is_ci() -> bool:
    """Check if running in any CI environment"""
    ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'JENKINS_URL', 'GITHUB_ACTIONS', 'GITLAB_CI']
    return any(os.getenv(indicator) for indicator in ci_indicators)


def get_environment_type() -> str:
    """Get the current environment type"""
    if is_jenkins():
        return "jenkins"
    elif is_ci():
        return "ci"
    else:
        return "local"


class Config:
    """Configuration management for PDF Analyzer MCP Client"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Base paths
        self.workspace_root = self._find_workspace_root()
        self.server_script_path = self._find_server_script()
        
        # Server configuration - initialize remote_server_url first
        self.remote_server_url = os.getenv('PDF_ANALYZER_REMOTE_URL')
        self.is_local_server = self._detect_local_server()
        
        # Server type selection
        self.use_simple_server = os.getenv('PDF_ANALYZER_USE_SIMPLE_SERVER', 'false').lower() == 'true'
        self.simple_server_path = self._find_simple_server_script()
        
        # Environment detection
        self.is_jenkins = self._is_jenkins_environment()
        self.is_ci = self._is_ci_environment()
        
        # Logging and output
        self.verbose = os.getenv('PDF_ANALYZER_VERBOSE', 'false').lower() == 'true'
        self.output_format = os.getenv('PDF_ANALYZER_OUTPUT_FORMAT', 'text')  # text, json, xml
        
        # File handling for remote servers
        self.max_upload_size_mb = int(os.getenv('PDF_ANALYZER_MAX_UPLOAD_MB', '50'))
        self.upload_timeout_seconds = int(os.getenv('PDF_ANALYZER_UPLOAD_TIMEOUT', '300'))
    
    def _find_workspace_root(self) -> Path:
        """Find the workspace root directory"""
        current = Path.cwd()
        
        # Look for markers that indicate workspace root
        markers = ['.git', 'Jenkinsfile', 'setup_client.py', 'pdf-analyzer']
        
        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        
        # Default to current directory
        return Path.cwd()
    
    def _find_server_script(self) -> str:
        """Find the server script path"""
        # Try different possible locations
        possible_paths = [
            self.workspace_root / "pdf-analyzer" / "server" / "main.py",
            self.workspace_root / "server" / "main.py",
            Path.cwd() / "server" / "main.py",
            Path.cwd() / "pdf-analyzer" / "server" / "main.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path.absolute())
        
        # Default fallback
        return str((self.workspace_root / "server" / "main.py").absolute())
    
    def _find_simple_server_script(self) -> str:
        """Find the simple FastMCP server script path"""
        # Try different possible locations for simple server
        possible_paths = [
            self.workspace_root / "pdf-analyzer" / "server" / "simple_server.py",
            self.workspace_root / "server" / "simple_server.py",
            Path.cwd() / "server" / "simple_server.py",
            Path.cwd() / "pdf-analyzer" / "server" / "simple_server.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path.absolute())
        
        # Default fallback
        return str((self.workspace_root / "server" / "simple_server.py").absolute())
    
    def get_active_server_path(self) -> str:
        """Get the path to the server that should be used"""
        if self.is_local_server and self.use_simple_server:
            # Use simple FastMCP server for local development
            return self.simple_server_path
        else:
            # Use full MCP server for hosting or when simple server is not preferred
            return self.server_script_path
    
    def get_server_type_description(self) -> str:
        """Get human-readable description of current server configuration"""
        if not self.is_local_server:
            return "🌐 Remote (Full MCP with Upload)"
        elif self.use_simple_server:
            return "🏠 Local Simple (FastMCP)"
        else:
            return "🏠 Local Full (Standard MCP)"
    
    def _detect_local_server(self) -> bool:
        """Detect if we should use local server or remote"""
        # Explicit remote URL overrides local detection
        if self.remote_server_url:
            return False
        
        # Check if server script exists locally
        return Path(self.server_script_path).exists()
    
    def _is_jenkins_environment(self) -> bool:
        """Detect if running in Jenkins"""
        jenkins_indicators = [
            'JENKINS_URL',
            'BUILD_NUMBER',
            'JOB_NAME',
            'WORKSPACE'
        ]
        return any(os.getenv(var) for var in jenkins_indicators)
    
    def _is_ci_environment(self) -> bool:
        """Detect if running in any CI environment"""
        ci_indicators = [
            'CI',
            'CONTINUOUS_INTEGRATION',
            'GITHUB_ACTIONS',
            'GITLAB_CI',
            'CIRCLECI',
            'TRAVIS',
            'APPVEYOR'
        ]
        return any(os.getenv(var) for var in ci_indicators) or self.is_jenkins
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration for client initialization"""
        return {
            'server_script_path': self.get_active_server_path(),
            'is_local': self.is_local_server,
            'remote_url': self.remote_server_url,
            'max_upload_size_mb': self.max_upload_size_mb,
            'upload_timeout': self.upload_timeout_seconds,
            'server_type': self.get_server_type_description(),
            'use_simple_server': self.use_simple_server
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging"""
        return {
            'workspace_root': str(self.workspace_root),
            'server_script_path': self.server_script_path,
            'simple_server_path': self.simple_server_path,
            'active_server_path': self.get_active_server_path(),
            'server_script_exists': Path(self.server_script_path).exists(),
            'simple_server_exists': Path(self.simple_server_path).exists(),
            'is_local_server': self.is_local_server,
            'use_simple_server': self.use_simple_server,
            'remote_server_url': self.remote_server_url,
            'is_jenkins': self.is_jenkins,
            'is_ci': self.is_ci,
            'verbose': self.verbose,
            'output_format': self.output_format,
            'max_upload_size_mb': self.max_upload_size_mb,
            'upload_timeout_seconds': self.upload_timeout_seconds
        }
    
    def validate_file_for_upload(self, file_path: str) -> bool:
        """Validate file can be uploaded to remote server"""
        if self.is_local_server:
            return True  # No restrictions for local server
        
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_upload_size_mb:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max: {self.max_upload_size_mb}MB)")
        
        # Check file type
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")
        
        return True
    
    def print_config(self, verbose: bool = None):
        """Print current configuration"""
        if verbose is None:
            verbose = self.verbose
        
        print("🔧 PDF Analyzer Configuration:")
        print(f"   Server Mode: {self.get_server_type_description()}")
        
        if self.is_local_server:
            active_server = self.get_active_server_path()
            print(f"   Server Script: {active_server}")
            exists = Path(active_server).exists()
            print(f"   Script Exists: {'✅' if exists else '❌'}")
            
            if self.use_simple_server:
                print("   📝 Using FastMCP (simple, local-only)")
            else:
                print("   🔧 Using Standard MCP (full features)")
        else:
            print(f"   Remote URL: {self.remote_server_url}")
            print(f"   Max Upload: {self.max_upload_size_mb}MB")
        
        print(f"   Environment: {self._get_env_description()}")
        
        if verbose:
            print("\n📊 Detailed Environment Info:")
            for key, value in self.get_environment_info().items():
                print(f"   {key}: {value}")
    
    def _get_env_description(self) -> str:
        """Get human-readable environment description"""
        if self.is_jenkins:
            return "Jenkins CI"
        elif self.is_ci:
            return "CI/CD Pipeline"
        else:
            return "Local Development"


# Global config instance
config = Config()
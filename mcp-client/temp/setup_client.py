#!/usr/bin/env python3
"""
Setup script for PDF Analyzer MCP Client
Prepares the environment for running the client
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr and check:
        print(f"Error: {result.stderr}")
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    
    return result


def setup_virtual_environment():
    """Set up Python virtual environment"""
    print("🐍 Setting up Python virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("   Virtual environment already exists")
        return str(venv_path)
    
    # Create virtual environment
    run_command(f"{sys.executable} -m venv venv")
    
    print("✅ Virtual environment created")
    return str(venv_path)


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine pip command
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    try:
        # Try to upgrade pip, but don't fail if it doesn't work
        try:
            run_command(f"{python_cmd} -m pip install --upgrade pip", check=False)
            print("✅ Pip upgrade successful")
        except Exception as e:
            print("⚠️  Pip upgrade failed, continuing with existing version")

        # Install main dependencies from mcp-client/requirements.txt
        req_file = Path("requirements.txt")
        if req_file.exists():
            run_command(f"{pip_cmd} install -r requirements.txt")
            print("✅ Project dependencies installed")
        else:
            print("⚠️  requirements.txt not found in mcp-client/")

        # Install the package in development mode (if needed)
        # If you have a setup.py or pyproject.toml, uncomment the next two lines:
        # run_command(f"{pip_cmd} install -e .")
        # print("✅ Project package installed (editable mode)")

        return python_cmd
    except Exception as e:
        print(f"⚠️  Error during dependency installation: {e}")
        print("Please try installing dependencies manually:")
        print(f"1. {python_cmd} -m pip install --upgrade pip")
        print(f"2. {pip_cmd} install -r requirements.txt")
        # print(f"3. {pip_cmd} install -e .")
        raise


def create_environment_file():
    """Create .env file from template"""
    print("⚙️  Setting up environment configuration...")
    
    env_file = Path("client/.env")
    env_example = Path("client/.env.example")
    
    if env_file.exists():
        print("   .env file already exists")
        return
    
    if env_example.exists():
        # Copy example file
        shutil.copy(env_example, env_file)
        
        # Update with actual paths
        content = env_file.read_text()
        
        # Replace placeholder paths with actual paths
        server_path = Path("server/main.py").resolve()
        content = content.replace(
            "MCP_SERVER_SCRIPT=/path/to/your/server/main.py",
            f"MCP_SERVER_SCRIPT={server_path}"
        )
        
        env_file.write_text(content)
        print("✅ Environment file created")
    else:
        print("⚠️  .env.example not found")


def test_setup():
    """Test the setup"""
    print("🧪 Testing setup...")
    
    # Determine python command
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    try:
        # Test client
        result = run_command(f"{python_cmd} client/test_client.py", check=False)
        
        if result.returncode == 0:
            print("✅ Setup test passed!")
            return True
        else:
            print("⚠️  Setup test had issues - check output above")
            return False
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs = [
        "output",
        "temp",
        "artifacts",
        "reports",
        "client/output",
        "client/temp"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")


def main():
    """Main setup function"""
    print("🚀 PDF Analyzer MCP Client Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    try:
        # Setup steps
        setup_virtual_environment()
        python_cmd = install_dependencies()
        create_environment_file()
        create_directories()
        
        print("\n" + "=" * 50)
        print("🎉 Setup completed successfully!")
        print("=" * 50)
        
        print("📋 Next steps:")
        print("1. Activate virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        
        print("2. Test the setup:")
        print("   python client/test_client.py")
        
        print("3. Try the CLI:")
        print("   python client/cli.py test-connection")
        
        print("4. For Jenkins, use:")
        print("   python client/jenkins_integration.py path/to/your.pdf")
        
        # Run test
        print("\n🧪 Running setup test...")
        if test_setup():
            print("\n✅ Ready to use!")
        else:
            print("\n⚠️  Setup completed but tests had issues")
            print("   Check the output above and your configuration")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
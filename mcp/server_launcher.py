"""
server_launcher.py

A simple script to launch the MCP server with proper environment setup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def launch_server():
    """Launch the MCP server with environment variables loaded"""
    
    # Load environment variables
    env_file = Path(__file__).parent / ".env"
    print(f"env_file: {env_file}")
    if env_file.exists():
        load_dotenv(env_file)
    
    # Check required environment variables
    required_vars = ["GROQ_API_KEY", "SUPABASE_DB_CONNECTION_STRING"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Please create a .env file with the required variables.")
        sys.exit(1)
    
    # Launch the MCP server
    server_path = Path(__file__).parent / "multiagent_mcp_server.py"
    
    try:
        print("Starting Multi-Agent Education Assistant MCP Server...")
        subprocess.run([sys.executable, str(server_path)], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    launch_server()
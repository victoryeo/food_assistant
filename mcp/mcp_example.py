import asyncio
import os
import sys
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from typing import Any

# Define a simple MCP server with a single tool
mcp_server = FastMCP("example")

@mcp_server.tool("get_greeting")
def get_greeting(name: str) -> str:
    """Returns a personalized greeting."""
    return f"Hello, {name}!"

@mcp_server.tool("add_numbers")
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

# Server configuration class
class ServerConfig:
    def __init__(self, command: str, args: list[str]):
        self.command = command
        self.args = args
        self.env = os.environ.copy()
        self.cwd = os.getcwd()
        # Add missing attributes expected by MCP client
        self.encoding = "utf-8"
        self.encoding_error_handler = "strict"
        self.stderr = None  # Let subprocess handle stderr normally

# Server runner function using FastMCP's synchronous run
def run_server_sync():
    """Run the MCP server using FastMCP synchronously"""
    print("🚀 Starting MCP server...", file=sys.stderr)
    try:
        # FastMCP's run() method manages its own event loop
        mcp_server.run()
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        raise

# Alternative server using direct MCP Server
async def run_server_direct():
    """Direct MCP server implementation"""
    print("🚀 Starting MCP server (direct implementation)...", file=sys.stderr)
    
    try:
        from mcp.server.stdio import stdio_server
        from mcp.server import Server
        from mcp.types import Tool, TextContent
        
        # Create a basic MCP server
        server = Server("example-server")
        
        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="get_greeting",
                    description="Returns a personalized greeting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name to greet"}
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="add_numbers", 
                    description="Adds two numbers together",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                )
            ]
        
        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "get_greeting":
                result = f"Hello, {arguments['name']}!"
                return [TextContent(type="text", text=result)]
            elif name == "add_numbers":
                result = arguments['a'] + arguments['b']
                return [TextContent(type="text", text=str(result))]
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        # Run the server with stdio
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)
            
    except Exception as e:
        print(f"❌ Direct server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise

# Manual stdio server implementation
async def run_server_manual():
    """Manual stdio server implementation"""
    print("🚀 Starting MCP server (manual stdio)...", file=sys.stderr)
    
    import json
    import sys
    
    async def handle_request(request_data):
        try:
            method = request_data.get("method")
            request_id = request_data.get("id")
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "example-server",
                            "version": "1.0.0"
                        }
                    }
                }
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "get_greeting",
                                "description": "Returns a personalized greeting",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Name to greet"}
                                    },
                                    "required": ["name"]
                                }
                            },
                            {
                                "name": "add_numbers",
                                "description": "Adds two numbers together", 
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "number", "description": "First number"},
                                        "b": {"type": "number", "description": "Second number"}
                                    },
                                    "required": ["a", "b"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                params = request_data.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                if name == "get_greeting":
                    result = f"Hello, {arguments['name']}!"
                elif name == "add_numbers":
                    result = str(arguments['a'] + arguments['b'])
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_data.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    try:
        # Simple stdio loop
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                line = line.strip()
                if not line:
                    break
                    
                request_data = json.loads(line)
                response = await handle_request(request_data)
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                continue
            except EOFError:
                break
                
    except Exception as e:
        print(f"❌ Manual server error: {e}", file=sys.stderr)
        raise

# Client function
async def run_client():
    """Run the MCP client"""
    server_file_path = os.path.abspath(__file__)
    
    server_config = ServerConfig(
        command=sys.executable,
        args=[server_file_path, "--run-server"]
    )
    
    print(f"🔗 Connecting to MCP server...")
    print(f"   Command: {server_config.command}")
    print(f"   Args: {server_config.args}")
    
    try:
        async with stdio_client(server_config) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("🔄 Initializing session...")
                await session.initialize()
                print("✅ Connection to server successful!")

                # List available tools
                print("📋 Listing tools from server...")
                tools_result = await session.list_tools()
                if hasattr(tools_result, 'tools'):
                    tool_names = [tool.name for tool in tools_result.tools]
                    print(f"   Available tools: {tool_names}")
                else:
                    print(f"   Tools result: {tools_result}")

                # Call the greeting tool
                print("🛠️ Calling the 'get_greeting' tool...")
                response = await session.call_tool("get_greeting", {"name": "Alice"})
                print(f"📨 Response: {response}")
                
                # Call the add_numbers tool
                print("🛠️ Calling the 'add_numbers' tool...")
                response2 = await session.call_tool("add_numbers", {"a": 5, "b": 3})
                print(f"📨 Response: {response2}")
    
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    return True

# Main execution logic
def main():
    if '--run-server' in sys.argv:
        # Try different server implementations in order of preference
        try:
            # FastMCP handles its own event loop
            run_server_sync()
        except Exception as e:
            print(f"⚠️ FastMCP failed ({e}), trying async alternatives...", file=sys.stderr)
            # Use asyncio for the fallback servers
            async def run_fallback():
                try:
                    await run_server_direct()
                except Exception as e2:
                    print(f"⚠️ Direct server failed ({e2}), trying manual server...", file=sys.stderr)
                    await run_server_manual()
            
            asyncio.run(run_fallback())
    else:
        # Run as client
        async def client_main():
            success = await run_client()
            if success:
                print("✅ Client execution completed successfully!")
            else:
                print("❌ Client execution failed!")
                sys.exit(1)
        
        asyncio.run(client_main())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
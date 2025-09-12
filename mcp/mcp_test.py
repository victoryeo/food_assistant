import asyncio
import json
import os
import sys
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from typing import Any, Dict
from mcp import ClientSession, StdioServerParameters

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
        self.capabilities = None

async def example_client_usage():
    """Example client usage demonstrating basic MCP operations"""
    
    print("=== MCP Client Usage Example ===\n")
    # Get the absolute path to the server script
    server_script = os.path.abspath("multiagent_mcp_server.py")
    
    # Create server config
    server_config = ServerConfig(
        command=sys.executable,
        args=[server_script]
    )
    async with stdio_client(server_config) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session
            await session.initialize()
            print("✅ Session initialized")
            
            # List available tools
            tools = await session.list_tools()
            print(f"📋 Available tools: {[tool.name for tool in tools.tools]}")
            
            # List available resources
            resources = await session.list_resources()
            print(f"📚 Available resources: {[res.name for res in resources.resources]}")

            # Example 1: Process a learning request
            print("\n--- Example 1: Process Learning Request ---")
            result = await session.call_tool(
                "process_message",
                {
                    "user_input": "I want to learn about photosynthesis",
                    "user_id": "demo_user",
                    "category": "biology"
                }
            )
            # Handle the response content
            if not result.content:
                print("Error: Empty response received from server")
                return
                
            response_text = result.content[0].text
            print(f"\n--- Raw Response ---\n{response_text}\n--- End Raw Response ---\n")
            
            # Check for validation errors in the response
            if "validation error" in response_text.lower() or "validation_error" in response_text.lower():
                print("⚠️ Validation error in response:")
                print(response_text)
                return
                
            try:
                # Try to parse as JSON
                response_data = json.loads(response_text)
                if isinstance(response_data, dict):
                    if 'response' in response_data:
                        print(f"✅ Agent response: {response_data['response'][:200]}...")
                    elif 'message' in response_data:
                        print(f"✅ Message: {response_data['message']}")
                    else:
                        print(f"✅ Response data: {json.dumps(response_data, indent=2)}")
                else:
                    print(f"✅ Response: {response_data}")
            except json.JSONDecodeError:
                print(f"📝 Raw response (non-JSON): {response_text[:500]}...")
            
            # Example 2: Create a specific task
            print("\n---Example 2: Creating Task ---")
            try:
                task_result = await session.call_tool(
                    "create_task",
                    {
                        "title": "Study cell structure diagram",
                        "user_id": "demo_user",
                        "description": "Review and memorize parts of plant and animal cells",
                        "category": "biology",
                        "priority": "high"
                    }
                )
                
                if not task_result.content:
                    print("Error: Empty response for task creation")
                    return
                    
                task_response = task_result.content[0].text
                print(f"\n--- Task Response ---\n{task_response}\n--- End Task Response ---\n")
                
                try:
                    task_data = json.loads(task_response)
                    if 'message' in task_data:
                        print(f"✅ {task_data['message']}")
                    elif 'created_task' in task_data:
                        print(f"✅ Created task: {task_data['created_task'].get('title', 'Untitled')}")
                        print(f"   Task ID: {task_data['created_task'].get('id', 'N/A')}")
                        print(f"   Status: {task_data['created_task'].get('status', 'unknown')}")
                    else:
                        print(f"✅ Task created: {json.dumps(task_data, indent=2)}")
                except json.JSONDecodeError:
                    print(f"📝 Task response (non-JSON): {task_response[:500]}...")
                    
            except Exception as e:
                print(f"❌ Error creating task: {str(e)}")
            
            # Example 3: Get tasks
            print("\n--- Example 3: Get User Tasks ---")
            tasks_result = await session.call_tool(
                "get_tasks",
                {
                    "user_id": "demo_user",
                    "category": "biology"
                }
            )
            tasks = json.loads(tasks_result.content[0].text)
            print(f"Found {len(tasks)} tasks for user")
            
            # Example 4: Read agent capabilities
            print("\n--- Example 4: Read Agent Capabilities ---")
            try:
                response = await session.read_resource("mcp://agent_capabilities")
                print(f"Response type: {type(response).__name__}")
                cap_data = json.loads(response.contents[0].text)
                print(f"Server supports {len(cap_data['supported_intents'])} intent types")

                # Debug the response object
                print("\n=== Response Debug ===")
                print(f"Response: {response}")
                print(f"Dir: {dir(response)}")
                                
            except Exception as e:
                print(f"❌ Error reading agent capabilities: {str(e)}")
            
            print("\n✅ Example usage completed successfully!")


async def test_mcp_server():
    """Test script for the MCP server"""
    
    test_cases = [
        {
            "name": "Create Math Task",
            "tool": "process_message",
            "params": {
                "user_input": "Create a task to practice quadratic equations",
                "user_id": "test_user",
                "category": "math"
            }
        },
        {
            "name": "Get Task List",
            "tool": "get_tasks", 
            "params": {
                "user_id": "test_user",
                "category": "math"
            }
        },
        {
            "name": "Intent Analysis",
            "tool": "analyze_intent",
            "params": {
                "user_input": "I need help with my physics homework",
                "user_id": "test_user",
                "category": "physics"
            }
        },
        {
            "name": "Agent Status",
            "tool": "get_agent_status",
            "params": {
                "user_id": "test_user",
                "category": "math"
            }
        },
        {
            "name": "Create Explicit Task",
            "tool": "create_task",
            "params": {
                "title": "Complete physics problem set #3",
                "user_id": "test_user",
                "description": "Work through momentum and energy problems",
                "category": "physics",
                "priority": "high"
            }
        },
        {
            "name": "Education Query Processing",
            "tool": "process_message",
            "params": {
                "user_input": "What are the laws of thermodynamics?",
                "user_id": "test_user",
                "category": "physics"
            }
        }
    ]
    
    print("=== MCP Server Test Suite ===\n")
        
    # Get the absolute path to the server script
    server_script = os.path.abspath("multiagent_mcp_server.py")
    
    # Create server config
    server_config = ServerConfig(
        command=sys.executable,
        args=[server_script]
    )

    print(f"🔗 Connecting to MCP server...")
    print(f"   Command: {server_config.command}")
    print(f"   Args: {server_config.args}")

    # Connect to server and run tests
    try:
        # Create client session using stdio_client with ServerConfig
        async with stdio_client(server_config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Server connection established\n")
                
                success_count = 0
                
                for test_case in test_cases:
                    print(f"Testing: {test_case['name']}")
                    try:
                        result = await session.call_tool(
                            test_case['tool'],
                            test_case['params']
                        )
                        print(f"✅ Success: {test_case['name']}")
                        
                        # Parse and display relevant parts of response
                        response_text = result.content[0].text
                        if len(response_text) > 200:
                            print(f"Response preview: {response_text[:200]}...\n")
                        else:
                            print(f"Response: {response_text}\n")
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"❌ Failed: {test_case['name']}")
                        print(f"Error: {e}\n")
                
                print(f"=== Test Results: {success_count}/{len(test_cases)} passed ===\n")
                    
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("Make sure the server is running: python multiagent_mcp_server.py")

# Main execution
if __name__ == "__main__":    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "example":
            asyncio.run(example_client_usage())
        elif mode == "test":
            asyncio.run(test_mcp_server())
        else:
            print("Usage: python test_client.py [example|test|comprehensive|interactive]")
    else:
        # Run default test suite
        asyncio.run(test_mcp_server())
        
    print("\n=== Usage Instructions ===")
    print("1. Set up your .env file with required API keys")
    print("2. Install requirements: pip install -r requirements.txt") 
    print("3. Run server: python multiagent_mcp_server.py")
    print("4. Run tests: python mcp_test.py [mode]")
    print("   Modes: example, test")
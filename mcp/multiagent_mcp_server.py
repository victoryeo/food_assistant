"""
MCP Server for Multi-Agent Education Assistant

This server exposes the MultiAgentEducationAssistant capabilities through the MCP protocol,
allowing other applications to interact with the multi-agent system.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import os
import uuid
import sys
from mcp.server.lowlevel.server import Server
from mcp.types import Resource
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    CallToolResult,
    ReadResourceResult,
    TextResourceContents
)
# Import MCP types for initialization
from mcp.server import InitializationOptions
from pydantic import AnyUrl

# Get the absolute path of the parent directory of Django app
current_dir = os.path.dirname(os.path.abspath(__file__))
django_backend_path = os.path.join(current_dir, '..', 'django-backend/djangoapp')
sys.path.append(django_backend_path)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing MultiAgentEducationAssistant
MultiAgentEducationAssistant = None
try:
    from multi_agent_assistant import MultiAgentEducationAssistant
    logger.info("Successfully imported MultiAgentEducationAssistant from multi_agent_assistant")
except Exception as e:
    logger.warning(f"Could not import MultiAgentEducationAssistant: {str(e)}.")
    MultiAgentEducationAssistant = None
# Logger already configured at the top

class MultiAgentMCPServer:
    """MCP Server wrapper for MultiAgentEducationAssistant"""
    
    def __init__(self):
        logger.info("Initializing MultiAgentMCPServer...")
        self.agents: Dict[str, Dict[str, Any]] = {}  # agent_key -> agent_data
        self.active_sessions: Dict[str, str] = {}  # session_id -> agent_key

    def _get_agent_key(self, user_id: str, category: str) -> str:
        """Generate unique agent key"""
        return f"{user_id}_{category}"
    
    async def _get_or_create_agent(self, user_id: str, category: str, role_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Get existing agent or create new one"""
        logger.info(f"🔑 _get_or_create_agent called with user_id={user_id}, category={category}")
        agent_key = self._get_agent_key(user_id, category)
        logger.info(f"🔑 Agent key: {agent_key}")
        
        if agent_key not in self.agents:
            logger.info(f"🆕 Creating new agent for user {user_id}, category {category}")
            try:
                logger.info("🔍 Checking MultiAgentEducationAssistant class...")
                if MultiAgentEducationAssistant is not None:
                    logger.info("✅ Using real MultiAgentEducationAssistant implementation")
                    try:
                        # Use real implementation
                        logger.info("🛠️  Creating MultiAgentEducationAssistant instance...")
                        try:
                            logger.info("🔄 Attempting to create MultiAgentEducationAssistant instance...")
                            agent = MultiAgentEducationAssistant(
                                user_id=user_id,
                                category=category,
                                role_prompt=role_prompt or f"You are an educational assistant specializing in {category}."
                            )
                            logger.info("✅ Successfully created MultiAgentEducationAssistant instance")
                        except TypeError as e:
                            logger.error(f"❌ TypeError creating agent: {str(e)}")
                            logger.error("This might be due to incorrect parameters. Available parameters: user_id, category, role_prompt")
                            raise
                        except Exception as e:
                            logger.error(f"❌ Unexpected error creating agent: {str(e)}")
                            raise
                        
                        # Store agent data
                        logger.info("💾 Storing agent data...")
                        self.agents[agent_key] = {
                            "instance": agent,
                            "conversation_history": [],
                            "tasks": [],
                            "created_at": datetime.now().isoformat(),
                            "last_accessed": datetime.now().isoformat()
                        }
                        logger.info("✅ Agent data stored successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to create agent: {str(e)}")
                        raise
                else:
                    # Use mock implementation
                    self.agents[agent_key] = {
                        "instance": None,
                        "user_id": user_id,
                        "category": category,
                        "role_prompt": role_prompt,
                        "tasks": [],
                        "conversation_history": [],
                        "created_at": datetime.now().isoformat()
                    }
                
                logger.info(f"Created new agent for user {user_id}, category {category}")
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise
        
        return self.agents[agent_key]

# Create server instance
logger.info("Creating server instance...")
server = Server("multi-agent-education-assistant")

# Initialize the multi-agent server
mcp_server = MultiAgentMCPServer()

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="process_message",
            description="Process a message through the multi-agent education system",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "The user's message/query"},
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"},
                    "role_prompt": {"type": "string", "description": "Optional custom role prompt"}
                },
                "required": ["user_input", "user_id"]
            }
        ),
        Tool(
            name="get_tasks",
            description="Get tasks for a user in a specific category",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"},
                    "status": {"type": "string", "description": "Optional status filter"}
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="create_task",
            description="Create a new task",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "description": {"type": "string", "default": "", "description": "Task description"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"},
                    "priority": {"type": "string", "default": "medium", "description": "Task priority"}
                },
                "required": ["title", "user_id"]
            }
        ),
        Tool(
            name="update_task",
            description="Update an existing task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to update"},
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"},
                    "title": {"type": "string", "description": "New task title"},
                    "description": {"type": "string", "description": "New task description"},
                    "status": {"type": "string", "description": "New task status"},
                    "priority": {"type": "string", "description": "New task priority"}
                },
                "required": ["task_id", "user_id"]
            }
        ),
        Tool(
            name="get_agent_status",
            description="Get status of all agents for a user/category",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"}
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="analyze_intent",
            description="Analyze user intent without full processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "The user's message/query"},
                    "user_id": {"type": "string", "description": "Unique identifier for the user"},
                    "category": {"type": "string", "default": "general", "description": "Category/subject area"}
                },
                "required": ["user_input", "user_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Handle tool calls"""
    logger.info(f"🔧 call_tool called with name={name}, arguments={arguments}")
    try:
        if name == "process_message":
            logger.info("📨 Processing process_message request")
            user_input = arguments["user_input"]
            user_id = arguments["user_id"]
            category = arguments.get("category", "general")
            role_prompt = arguments.get("role_prompt")
            logger.info(f"📝 Message details - user_input: {user_input[:50]}..., user_id: {user_id}, category: {category}")
            
            logger.info("🔍 Getting or creating agent...")
            
            try:
                agent_data = await mcp_server._get_or_create_agent(user_id, category, role_prompt)
                logger.info("✅ Successfully got/created agent")
            except Exception as e:
                logger.error(f"❌ Failed to get/create agent: {str(e)}")
                raise
            
            # Add to conversation history
            agent_data["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "user",
                "content": user_input
            })
            
            if agent_data["instance"] and MultiAgentEducationAssistant:
                # Use real agent processing
                try:
                    response, created_tasks = await agent_data["instance"].process_message(user_input)
                    
                    # Add created tasks to agent data
                    if created_tasks:
                        agent_data["tasks"].extend(created_tasks)
                except Exception as e:
                    response = f"Agent processing error: {str(e)}"
                    created_tasks = []
            else:
                # Mock response
                response = f"Mock response for '{user_input}' in category '{category}'"
                created_tasks = []
                
                # Mock task creation based on intent
                if any(keyword in user_input.lower() for keyword in ["create", "add", "new", "todo"]):
                    mock_task = {
                        "id": str(uuid.uuid4()),
                        "title": f"Task from: {user_input[:50]}...",
                        "description": f"Auto-generated task from user input",
                        "category": category,
                        "priority": "medium",
                        "status": "pending",
                        "created_at": datetime.now().isoformat(),
                        "user_id": user_id
                    }
                    agent_data["tasks"].append(mock_task)
                    created_tasks = [mock_task]
            
            # Add response to conversation history
            agent_data["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "assistant", 
                "content": response
            })
            
            result = {
                "response": response,
                "created_tasks": created_tasks,
                "timestamp": datetime.now().isoformat()
            }
            
            # Debug logging
            result_str = json.dumps(result, indent=2)
            logger.info(f"DEBUG - Sending response: {result_str}")
            
            # Create a properly formatted TextContent object
            # Ensure the response is always a dictionary with a 'response' field
            if not isinstance(result, dict):
                result = {"response": str(result)}
            
            # Convert to JSON string for the text content
            response_json = json.dumps(result, indent=2)
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": response_json,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": result,
                "isError": False
            }
        
        elif name == "get_tasks":
            user_id = arguments["user_id"]
            category = arguments.get("category", "general")
            status = arguments.get("status")
            
            agent_data = await mcp_server._get_or_create_agent(user_id, category)
            tasks = agent_data["tasks"]
            
            if status:
                tasks = [t for t in tasks if t.get('status') == status]
            
            # Prepare the result dictionary
            result = {
                "tasks": tasks,
                "user_id": user_id,
                "category": category,
                "count": len(tasks),
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert result to JSON string for the response
            result_str = json.dumps(result, indent=2)
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": result_str,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": result,
                "isError": False
            }
        
        elif name == "create_task":
            title = arguments["title"]
            user_id = arguments["user_id"]
            description = arguments.get("description", "")
            category = arguments.get("category", "general")
            priority = arguments.get("priority", "medium")
            
            agent_data = await mcp_server._get_or_create_agent(user_id, category)
            
            # Create new task
            new_task = {
                "id": str(uuid.uuid4()),
                "title": title,
                "description": description,
                "category": category,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            agent_data["tasks"].append(new_task)
            
            result = {
                "task": new_task,
                "message": f"Created task: {title}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert result to JSON string for the response
            result_str = json.dumps(result, indent=2)
            logger.info(f"DEBUG - Sending response: {result_str}")
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": result_str,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": result,
                "isError": False
            }
        
        elif name == "update_task":
            task_id = arguments["task_id"]
            user_id = arguments["user_id"]
            category = arguments.get("category", "general")
            
            agent_data = await mcp_server._get_or_create_agent(user_id, category)
            
            # Find and update task
            task_found = False
            for task in agent_data["tasks"]:
                if task["id"] == task_id:
                    task_found = True
                    
                    # Update fields if provided
                    if "title" in arguments:
                        task["title"] = arguments["title"]
                    if "description" in arguments:
                        task["description"] = arguments["description"]
                    if "status" in arguments:
                        task["status"] = arguments["status"]
                    if "priority" in arguments:
                        task["priority"] = arguments["priority"]
                    
                    task["updated_at"] = datetime.now().isoformat()
                    
            if task_found:
                result = {
                    "task": task,
                    "message": f"Updated task {task_id}",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {
                    "error": True,
                    "message": f"Task {task_id} not found",
                    "timestamp": datetime.now().isoformat()
                }
                
            # Convert result to JSON string for the response
            result_str = json.dumps(result, indent=2)
            logger.info(f"DEBUG - Sending response: {result_str}")
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": result_str,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": result,
                "isError": result.get("error", False)
            }
        
        elif name == "get_agent_status":
            user_id = arguments["user_id"]
            category = arguments.get("category", "general")
            
            agent_data = await mcp_server._get_or_create_agent(user_id, category)
            
            result = {
                "user_id": user_id,
                "category": category,
                "active": True,
                "task_count": len(agent_data.get("tasks", [])),
                "last_activity": agent_data.get("last_activity", "Never"),
                "conversation_length": len(agent_data.get("conversation_history", [])),
                "agent_key": mcp_server._get_agent_key(user_id, category),
                "created_at": agent_data.get("created_at", "Unknown"),
                "has_real_agent": agent_data.get("instance") is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert result to JSON string for the response
            result_str = json.dumps(result, indent=2)
            logger.info(f"DEBUG - Sending response: {result_str}")
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": result_str,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": result,
                "isError": False
            }
        
        else:
            error_result = {
                "error": True,
                "message": f"Unknown tool: {name}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert error to JSON string for the response
            error_str = json.dumps(error_result, indent=2)
            
            # Create TextContent with proper structure
            text_content = {
                "type": "text",
                "text": error_str,
                "annotations": None,
                "meta": {"mimeType": "application/json"}
            }
            
            # Return as a dictionary that can be properly serialized
            return {
                "content": [text_content],
                "structuredContent": error_result,
                "isError": True
            }
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        error_result = {
            "error": True,
            "message": f"Error in tool {name}: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert error to JSON string for the response
        error_str = json.dumps(error_result, indent=2)
        
        # Create TextContent with proper structure
        text_content = {
            "type": "text",
            "text": error_str,
            "annotations": None,
            "meta": {"mimeType": "application/json"}
        }
        
        # Return as a dictionary that can be properly serialized
        return {
            "content": [text_content],
            "structuredContent": error_result,
            "isError": True
        }

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="mcp://tasks/{user_id}/{category}",
            name="User Tasks",
            description="Get all tasks for a specific user and category",
            mimeType="application/json"
        ),
        Resource(
            uri="mcp://conversation_history/{user_id}/{category}",
            name="Conversation History",
            description="Get conversation history for a specific user and category",
            mimeType="application/json"
        ),
        Resource(
            uri="mcp://agent_capabilities",
            name="Agent Capabilities",
            description="Information about multi-agent system capabilities",
            mimeType="application/json"
        ),
        Resource(
            uri="mcp://active_agents",
            name="Active Agents",
            description="List of all currently active agents",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: AnyUrl) -> str | bytes:
    """Read a specific resource"""
    try:
        logger.info(f"Reading resource: {uri}")
        if "agent_capabilities" in str(uri):
            capabilities = {
                "supported_intents": [
                    "create", "update", "complete", "summary",
                    "schedule", "education", "query", "question"
                ],
                "version": "1.0",
            }
            
            logger.info("Sending agent capabilities...")
            # Create a simple response with just the capabilities
            return json.dumps(capabilities, indent=2)
        
        elif "active_agents" in str(uri):
            logger.info("Reading active agents...")
            agents_info = []
            for agent_key, agent_data in mcp_server.agents.items():
                agents_info.append({
                    "agent_key": agent_key,
                    "user_id": agent_data["user_id"],
                    "category": agent_data["category"],
                    "created_at": agent_data["created_at"],
                    "tasks_count": len(agent_data["tasks"]),
                    "conversation_length": len(agent_data["conversation_history"]),
                    "has_real_instance": agent_data["instance"] is not None
                })
            
            return json.dumps(agents_info, indent=2)
        
        elif uri.startswith("tasks/"):
            logger.info("Reading tasks...")
            # Parse URI: tasks/{user_id}/{category}
            parts = uri.split("/")
            if len(parts) >= 3:
                user_id = parts[1]
                category = parts[2] if len(parts) > 2 else "general"
                
                agent_data = await mcp_server._get_or_create_agent(user_id, category)
                tasks = agent_data["tasks"]
                
                return json.dumps(tasks, indent=2)
            else:
                raise ValueError(f"Invalid URI: {uri}")
        
        elif uri.startswith("conversation_history/"):
            logger.info("Reading conversation history...")
            # Parse URI: conversation_history/{user_id}/{category}
            parts = uri.split("/")
            if len(parts) >= 3:
                user_id = parts[1]
                category = parts[2] if len(parts) > 2 else "general"
                
                agent_data = await mcp_server._get_or_create_agent(user_id, category)
                history = agent_data["conversation_history"]
                
                return json.dumps(history, indent=2)
            else:
                raise ValueError(f"Invalid URI: {uri}")
        
        else:
            logger.info("Resource not found: {uri}")
            return f"Resource not found: {uri}"
        
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return f"Error reading resource {uri}: {e}"

async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Multi-Agent Education Assistant MCP Server...")
    
    try:
        # Create initialization options with required fields
        init_options = InitializationOptions(
            server_name="multi-agent-education-assistant",
            server_version="1.0.0",
            capabilities={
                "supported_intents": [
                    "create", "update", "complete", "summary",
                    "schedule", "education", "query", "question"
                ],
                "version": "1.0",
            }
        )
        
        # Use stdio for communication
        async with stdio_server() as (read_stream, write_stream):
            # Run the server with the stdio streams and initialization options
            try:
                await server.run(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    initialization_options=init_options,
                    raise_exceptions=True
                )
            except Exception as e:
                logger.error(f"Error in server.run(): {e}", exc_info=True)
                raise
    except Exception as e:
        logger.error(f"Error in main server loop: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
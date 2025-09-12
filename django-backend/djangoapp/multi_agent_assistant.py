from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv
import operator
import psycopg2
import os
import re
import json
import uuid
import traceback
from bson import ObjectId

load_dotenv()

# Enhanced state for multi-agent system
class MultiAgentTaskState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    user_input: str
    tasks: List[Dict[str, Any]]
    created_tasks: List[Dict[str, Any]]
    updated_tasks: List[Dict[str, Any]]
    intent: str
    category: str
    user_id: str
    response: str
    extracted_info: Dict[str, Any]
    retrieved_content: List[Document]
    # Multi-agent specific fields
    current_agent: str
    agent_outputs: Dict[str, Any]  # Store outputs from each agent
    collaboration_context: Dict[str, Any]  # Context for agent collaboration
    routing_decision: str  # Which agent should handle next
    agent_conversations: Dict[str, List[dict]]  # Conversations between agents

# Base Agent class
class BaseAgent:
    def __init__(self, name: str, role: str, category: str, llm, embeddings, db_connection_string: str, collection_name: str, user_id: str):
        self.name = name
        self.role = role
        self.category = category
        self.llm = llm
        self.embeddings = embeddings
        self.db_connection_string = db_connection_string
        self.collection_name = f"tasks_{category}_{user_id}"
        self.knowledge_collection_name = f"knowledge_{category}_{user_id}"
        self.setup_pgvector_store()
    
    def setup_pgvector_store(self):
        """Setup vector store for this agent"""
        try:
            self.vector_store = PGVector(
                collection_name=self.collection_name,
                connection=self.db_connection_string,
                embeddings=self.embeddings,
                distance_strategy="cosine",
                use_jsonb=True
            )
            self.knowledge_store = PGVector(
                collection_name=self.knowledge_collection_name,
                connection=self.db_connection_string,
                embeddings=self.embeddings,
                distance_strategy="cosine",
                use_jsonb=True
            )
        except Exception as e:
            print(f"❌ Vector store setup failed for {self.name}: {e}")
    
    def process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Base processing method - to be overridden by specific agents"""
        raise NotImplementedError

# Task Management Agent
class TaskManagerAgent(BaseAgent):
    def __init__(self, category: str, llm, embeddings, db_connection_string: str, collection_name: str, user_id: str):
        self.user_id = user_id
        print("Setting up TaskManagerAgent")
        super().__init__(
            name="task_manager",
            role="Responsible for creating, updating, and managing tasks",
            category=category,
            llm=llm,
            embeddings=embeddings,
            db_connection_string=db_connection_string,
            collection_name=collection_name,
            user_id=user_id
        )
    
    def process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process task management requests"""
        user_input = state["user_input"]
        intent = state["intent"]
        
        task_prompt = f"""
        You are a Task Management Agent. Your role is to efficiently handle task operations.
        
        Current Intent: {intent}
        User Input: {user_input}
        
        Based on the user input, extract task information and if there is sufficient task information, create new task.
        The new task shall include task title, task description, priority, deadline.
        Focus on task creation, deadlines, and priorities.
        
        Respond in the following JSON format only. Do not include any other text or explanations:
        
        {{
            "tasks": [
                {{
                    "title": "task title here",
                    "description": "task description here", 
                    "priority": "high|medium|low",
                    "deadline": "YYYY-MM-DD or null"
                }}
            ]
        }}
        
        JSON Response:
        """
        
        try:
            print("Preparing task manager response with user_input: ", user_input)
            response = self.llm.invoke([HumanMessage(content=task_prompt)])
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                extracted_info = json.loads(json_match.group(1))
            else:
                extracted_info = self._basic_extraction(user_input)
            
            print("Extracted info: ", extracted_info)
            state["agent_outputs"][self.name] = {
                "extracted_info": extracted_info,
                "processing_result": "Task information extracted successfully"
            }

            if extracted_info.get("tasks"):
                self._create_task(extracted_info)
            
        except Exception as e:
            print(f"TaskManager error: {e}")
            state["agent_outputs"][self.name] = {
                "error": str(e),
                "extracted_info": self._basic_extraction(user_input)
            }
        
        return state
    
    def _basic_extraction(self, user_input: str) -> Dict[str, Any]:
        """Basic fallback extraction"""
        return {
            "tasks": [{
                "title": user_input[:100],
                "description": user_input,
                "priority": "medium"
            }]
        }

    def _create_task(self, extracted_info: Dict[str, Any]):
        """Create new tasks with educational context"""
        created_tasks = []
        print(f"Creating tasks: {extracted_info}")

        for task in extracted_info.get("tasks"):
            task = {
                'id': str(uuid.uuid4()),
                'name': task.get('title', ''),
                'description': task.get('description', ''),
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'category': self.category,
                'user_id': str(self.user_id),
                'completed': False,
            }
            
            # Add optional fields
            if task.get('deadline'):
                task['deadline'] = task['deadline']
            if task.get('priority'):
                task['priority'] = task['priority']
            
            # Add to vector store
            print(f"Created task: {task}")
            self.vector_store.add_documents([Document(page_content=json.dumps(task))])
            created_tasks.append(task)
    
        return created_tasks

# Educational Content Agent
class EducationAgent(BaseAgent):
    def __init__(self, category: str, llm, embeddings, db_connection_string: str, collection_name: str, role_prompt: str, user_id: str):
        print("Setting up EducationAgent")
        self.user_id = user_id
        self.role_prompt = role_prompt
        super().__init__(
            name="education_specialist",
            role="Provides educational context and learning recommendations",
            category=category,
            llm=llm,
            embeddings=embeddings,
            db_connection_string=db_connection_string,
            collection_name=collection_name,
            user_id=user_id
        )
        self._load_menu_content()
 
    def _load_menu_content(self):
        """Load KFC menu data"""
        print("Loading KFC menu data...")
        menu_path = os.path.join(os.path.dirname(__file__), 'kfc_menu.json')
        if os.path.exists(menu_path):
            print("Found KFC menu data, loading...")
            try:
                with open(menu_path, 'r', encoding='utf-8') as f:
                    menu_data = json.load(f)
                
                # Process menu data into documents
                menu_docs = []
                for category, items in menu_data.items():
                    for item in items:
                        # Create a document for each menu item
                        try:
                            price = (item.get('price', ''))
                            price_str = f"${price}"
                        except (TypeError, ValueError):
                            price = 0
                            price_str = "Price not available"
                            
                        content = f"{item.get('name', '')}: {item.get('description', '')} - {price_str}"
                        doc = Document(
                            page_content=content,
                            metadata={
                                'category': category,
                                'name': item.get('name', ''),
                                'price': price,
                                'source': 'kfc_menu.json'
                            }
                        )
                        menu_docs.append(doc)
                
                if menu_docs:
                    # Add to vector store
                    self.knowledge_store.add_documents(menu_docs)
                    print(f"Loaded {len(menu_docs)} menu items from kfc_menu.json")
                    return menu_docs
                
                return []
                
            except Exception as e:
                print(f"❌ Failed to load KFC menu data: {e}")
                return []
        else:
            print("❌ kfc_menu.json not found")
            return []
    
    def process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process educational content requests"""
        user_input = state["user_input"]
        print("Preparing educational assistant response with user_input: ", user_input)
        # Retrieve relevant educational content
        try:
            relevant_docs = self.knowledge_store.similarity_search(user_input, k=3)
            
            education_prompt = f"""
            role prompt: {self.role_prompt}
            
            You are an Education Specialist Agent. Analyze the user request and provide educational insights according to role prompt.
            
            User Request: {user_input}
            
            Retrieved Educational Content:
            {self._format_documents(relevant_docs)}
            
            Provide an educational response that incorporates the retrieved learning materials.
            Offer insights, learning recommendations, and contextual information.
            Limit the response to 300 words.
            """
            
            response = self.llm.invoke([HumanMessage(content=education_prompt)])
            
            state["agent_outputs"][self.name] = {
                "educational_context": response.content,
                "retrieved_documents": len(relevant_docs),
                "recommendations": self._extract_recommendations(response.content)
            }
            
            state["retrieved_content"].extend(relevant_docs)
            
        except Exception as e:
            print(f"EducationAgent error: {e}")
            state["agent_outputs"][self.name] = {
                "error": str(e),
                "educational_context": "Unable to provide educational context at this time."
            }
        
        return state
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents for prompt"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"Document {i}: {doc.page_content[:300]}...")
        
        return "\n\n".join(formatted)
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from content"""
        # Simple extraction - could be enhanced
        lines = content.split('\n')
        recommendations = []
        for line in lines:
            if 'recommend' in line.lower() or 'suggest' in line.lower():
                recommendations.append(line.strip())
        return recommendations

# Scheduler Agent
class SchedulerAgent(BaseAgent):
    def __init__(self, category, llm, embeddings, db_connection_string: str, collection_name: str, user_id: str):
        print("SchedulerAgent initialized")
        super().__init__(
            name="scheduler",
            role="Manages deadlines, priorities, and task scheduling",
            category=category,
            llm=llm,
            embeddings=embeddings,
            db_connection_string=db_connection_string,
            collection_name=collection_name,
            user_id=user_id
        )
    
    def process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process scheduling and prioritization requests"""
        user_input = state["user_input"]
        existing_tasks = state.get("tasks", [])
        
        schedule_prompt = f"""
        You are a Scheduler Agent. Analyze the current tasks and user request for optimal scheduling.
        
        User Request: {user_input}
        Existing Tasks: {len(existing_tasks)}
        
        Provide scheduling recommendations including:
        - Optimal deadlines
        - Priority assignments
        - Time management suggestions
        - Conflict identification
        
        Consider work-life balance and realistic time constraints.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=schedule_prompt)])
            
            state["agent_outputs"][self.name] = {
                "scheduling_analysis": response.content,
                "priority_recommendations": self._analyze_priorities(existing_tasks),
                "deadline_suggestions": self._suggest_deadlines(user_input)
            }
            
        except Exception as e:
            print(f"SchedulerAgent error: {e}")
            state["agent_outputs"][self.name] = {
                "error": str(e),
                "scheduling_analysis": "Unable to provide scheduling analysis."
            }
        
        return state
    
    def _analyze_priorities(self, tasks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze task priorities"""
        priority_count = {"high": 0, "medium": 0, "low": 0}
        for task in tasks:
            priority = task.get("priority", "medium")
            priority_count[priority] = priority_count.get(priority, 0) + 1
        return priority_count
    
    def _suggest_deadlines(self, user_input: str) -> List[str]:
        """Suggest deadlines based on input"""
        suggestions = []
        now = datetime.now()
        
        if "urgent" in user_input.lower():
            suggestions.append((now + timedelta(days=1)).date().isoformat())
        elif "week" in user_input.lower():
            suggestions.append((now + timedelta(days=7)).date().isoformat())
        else:
            suggestions.append((now + timedelta(days=3)).date().isoformat())
        
        return suggestions

# Coordinator Agent
class CoordinatorAgent(BaseAgent):
    def __init__(self, category, llm, embeddings, db_connection_string: str, collection_name: str, user_id: str):
        print("CoordinatorAgent initialized")
        super().__init__(
            name="coordinator",
            role="Coordinates between agents and synthesizes responses",
            category=category,
            llm=llm,
            embeddings=embeddings,
            db_connection_string=db_connection_string,
            collection_name=collection_name,
            user_id=user_id
        )
    
    def process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Coordinate agent outputs and create final response"""
        agent_outputs = state["agent_outputs"]
        user_input = state["user_input"]
        
        coordination_prompt = f"""
        You are a Coordinator Agent. Synthesize the outputs from specialist agents into a coherent response.
        
        User Request: {user_input}
        
        Agent Outputs:
        {json.dumps(agent_outputs, indent=2, default=str)}
        
        Create a comprehensive response that:
        1. Addresses the user's request
        2. Incorporates insights from all agents
        3. Provides actionable recommendations
        4. Maintains educational context
        
        Keep the response concise but informative.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=coordination_prompt)])
            
            state["response"] = response.content
            state["agent_outputs"][self.name] = {
                "final_synthesis": response.content,
                "coordination_success": True
            }
            
        except Exception as e:
            print(f"CoordinatorAgent error: {e}")
            state["response"] = self._create_fallback_response(agent_outputs)
            state["agent_outputs"][self.name] = {
                "error": str(e),
                "coordination_success": False
            }
        
        return state
    
    def _create_fallback_response(self, agent_outputs: Dict[str, Any]) -> str:
        """Create fallback response if coordination fails"""
        response_parts = []
        
        if "task_manager" in agent_outputs:
            response_parts.append("Task information has been processed.")
        
        if "education_specialist" in agent_outputs:
            response_parts.append("Educational context has been considered.")
        
        if "scheduler" in agent_outputs:
            response_parts.append("Scheduling analysis has been completed.")
        
        return " ".join(response_parts) if response_parts else "Request processed successfully."

# Multi-Agent Education Assistant
class MultiAgentEducationAssistant:
    def __init__(self, role_prompt: str, category: str, user_id: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        self.collection_name = f"tasks_{category}_{user_id}" # Used as table name
        self.db_connection_string = os.getenv("SUPABASE_DB_CONNECTION_STRING")
        
        if not self.db_connection_string:
            raise ValueError("SUPABASE_DB_CONNECTION_STRING environment variable is not set.")
        
        # LLM setup
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
        else:
            raise ValueError("GROQ_API_KEY is not set")
        
        # Setup embeddings
        print("Setting up embeddings")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize agents
        print("Setting up agents")
        collection_base = f"{category}_{user_id}"
        self.agents = {
            "task_manager": TaskManagerAgent(self.category, self.llm, self.embeddings, self.db_connection_string, collection_base, self.user_id),
            "education_specialist": EducationAgent(self.category, self.llm, self.embeddings, self.db_connection_string, collection_base, self.role_prompt, self.user_id),
            "scheduler": SchedulerAgent(self.category, self.llm, self.embeddings, self.db_connection_string, collection_base, self.user_id),
            "coordinator": CoordinatorAgent(self.category, self.llm, self.embeddings, self.db_connection_string, collection_base, self.user_id)
        }
        
        # Task storage
        self.tasks = []
        self.conversation_history = []
        
        # Create workflow
        self.workflow = self._create_multi_agent_workflow()
    
    def _create_multi_agent_workflow(self):
        """Create multi-agent LangGraph workflow"""
        workflow = StateGraph(MultiAgentTaskState)
        
        # Add nodes for each agent and coordination
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("route_to_agents", self._route_to_agents)
        workflow.add_node("task_manager_process", self._task_manager_process)
        workflow.add_node("education_specialist_process", self._education_specialist_process)
        workflow.add_node("scheduler_process", self._scheduler_process)
        workflow.add_node("coordinate_response", self._coordinate_response)
        workflow.add_node("finalize_tasks", self._finalize_tasks)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        # Add edges
        workflow.add_edge("analyze_intent", "route_to_agents")
        
        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "route_to_agents",
            self._decide_agent_routing,
            {
                "task_focused": "task_manager_process",
                "education_focused": "education_specialist_process",
                "schedule_focused": "scheduler_process",
                "comprehensive": "task_manager_process"  # Start with task manager for comprehensive requests
            }
        )
        
        # Agent processing chains
        workflow.add_edge("task_manager_process", "education_specialist_process")
        workflow.add_edge("education_specialist_process", "scheduler_process")
        workflow.add_edge("scheduler_process", "coordinate_response")
        workflow.add_edge("coordinate_response", "finalize_tasks")
        workflow.add_edge("finalize_tasks", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_intent(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Analyze user intent for multi-agent routing"""
        user_input = state["user_input"].lower()
        
        # Determine intent
        if any(keyword in user_input for keyword in ["create", "add", "new", "todo"]):
            intent = "create"
        elif any(keyword in user_input for keyword in ["update", "modify", "change", "edit"]):
            intent = "update"
        elif any(keyword in user_input for keyword in ["complete", "done", "finished", "mark"]):
            intent = "complete"
        elif any(keyword in user_input for keyword in ["summary", "list", "show", "overview"]):
            intent = "summary"
        elif any(keyword in user_input for keyword in ["schedule", "deadline", "priority", "when"]):
            intent = "schedule"
        elif any(keyword in user_input for keyword in ["learn", "study", "education", "course"]):
            intent = "education"
        else:
            intent = "query"
        
        state["intent"] = intent
        state["agent_outputs"] = {}
        state["collaboration_context"] = {"primary_intent": intent}
        
        return state
    
    def _route_to_agents(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Determine routing strategy for agents"""
        print("Routing to agents...")
        intent = state["intent"]
        user_input = state["user_input"].lower()
        
        # Determine routing decision
        if intent in ["schedule"] or any(word in user_input for word in ["deadline", "priority", "urgent"]):
            routing = "schedule_focused"
        elif intent in ["education"] or any(word in user_input for word in ["learn", "study", "course"]):
            routing = "education_focused"
        elif intent in ["create", "update", "complete"]:
            routing = "task_focused"
        else:
            routing = "comprehensive"  # Engage multiple agents
        
        state["routing_decision"] = routing
        return state
    
    def _decide_agent_routing(self, state: MultiAgentTaskState) -> str:
        """Decision function for agent routing"""
        return state["routing_decision"]
    
    def _task_manager_process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process through Task Manager Agent"""
        print("Processing through Task Manager Agent")
        state["current_agent"] = "task_manager"
        return self.agents["task_manager"].process(state)
    
    def _education_specialist_process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process through Education Specialist Agent"""
        print("Processing through Education Specialist Agent")
        state["current_agent"] = "education_specialist"
        return self.agents["education_specialist"].process(state)
    
    def _scheduler_process(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Process through Scheduler Agent"""
        print("Processing through Scheduler Agent")
        state["current_agent"] = "scheduler"
        return self.agents["scheduler"].process(state)
    
    def _coordinate_response(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Coordinate final response through Coordinator Agent"""
        print("Processing through Coordinator Agent")
        state["current_agent"] = "coordinator"
        return self.agents["coordinator"].process(state)
    
    def _finalize_tasks(self, state: MultiAgentTaskState) -> MultiAgentTaskState:
        """Finalize task operations based on agent outputs"""
        print("Finalizing tasks...")
        task_manager_output = state["agent_outputs"].get("task_manager", {})
        extracted_info = task_manager_output.get("extracted_info", {})
        
        if state["intent"] == "create" and "tasks" in extracted_info:
            created_tasks = []
            for task_info in extracted_info["tasks"]:
                task = {
                    'id': str(uuid.uuid4()),
                    'title': task_info.get('title', ''),
                    'description': task_info.get('description', ''),
                    'status': 'pending',
                    'created_at': datetime.now().isoformat(),
                    'category': self.category,
                    'user_id': self.user_id,
                    'completed': False,
                    'priority': task_info.get('priority', 'medium')
                }
                
                # Add educational context from education agent
                education_output = state["agent_outputs"].get("education_specialist", {})
                if education_output.get("educational_context"):
                    task['educational_context'] = education_output["educational_context"][:200]
                
                # Add scheduling suggestions from scheduler agent
                scheduler_output = state["agent_outputs"].get("scheduler", {})
                if scheduler_output.get("deadline_suggestions"):
                    task['suggested_deadline'] = scheduler_output["deadline_suggestions"][0]
                
                self.tasks.append(task)
                created_tasks.append(task)
            
            state["created_tasks"] = created_tasks
        
        state["tasks"] = self.tasks
        return state
    
    async def process_message(self, user_input: str):
        """Process user message through multi-agent workflow"""
        print(f"Processing message with multi-agent system: {user_input}")
        
        initial_state = MultiAgentTaskState(
            messages=[],
            user_input=user_input,
            tasks=self.tasks,
            created_tasks=[],
            updated_tasks=[],
            intent="",
            category=self.category,
            user_id=self.user_id,
            response="",
            extracted_info={},
            retrieved_content=[],
            current_agent="",
            agent_outputs={},
            collaboration_context={},
            routing_decision="",
            agent_conversations={}
        )
        
        config = {"configurable": {"thread_id": f"{self.user_id}_{self.category}_multiagent"}}
        
        try:
            final_state = await self.workflow.ainvoke(initial_state, config)
        except Exception as e:
            print(f"ERROR: Multi-agent processing failed: {e}")
            traceback.print_exc()
            return "I apologize, but I encountered an error processing your request.", []
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=final_state["response"]))
        
        return final_state["response"], final_state.get("created_tasks", [])
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "active_agents": list(self.agents.keys()),
            "total_tasks": len(self.tasks),
            "agent_health": {name: "active" for name in self.agents.keys()}
        }
    
    def get_tasks_summary(self) -> Dict[str, Any]:
        """Get comprehensive tasks summary"""
        if not self.tasks:
            return {"message": "No tasks currently tracked."}
        
        now = datetime.now()
        today = now.date()
        
        summary = {
            "total_tasks": len(self.tasks),
            "completed": len([t for t in self.tasks if t.get('completed', False)]),
            "pending": len([t for t in self.tasks if not t.get('completed', False)]),
            "high_priority": len([t for t in self.tasks if t.get('priority') == 'high']),
            "with_educational_context": len([t for t in self.tasks if t.get('educational_context')])
        }
        
        return summary

    def _get_all_tasks_from_vector_store(self) -> List[Dict[str, Any]]:
        """Get all tasks from PGVector, filtered by category and user_id"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            # First, find the collection_id for your logical collection_name
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            print("DEBUG: Collection Name: ", self.collection_name)
            print("DEBUG: Collection UUID Row: ", collection_uuid_row)
            if not collection_uuid_row:
                print(f"WARNING: Collection '{self.collection_name}' not found in langchain_pg_collection. No tasks to retrieve.")
                cur.close()
                conn.close()
                return []

            collection_uuid = collection_uuid_row[0]

            # Then, retrieve documents from langchain_pg_embedding for that collection_id
            # The metadata is stored in the 'cmetadata' column
            cur.execute(f"SELECT document FROM langchain_pg_embedding WHERE collection_id = %s;", (str(collection_uuid),))
            
            all_tasks_metadata = cur.fetchall()
            cur.close()
            conn.close()
            
            tasks = []
            for row in all_tasks_metadata:
                document = row[0]
                documentJson = json.loads(document)
                print("DEBUG: Row: ", documentJson)
                # The 'document' column in langchain_pg_embedding IS your metadata
                # So it should contain 'category', 'user_id' directly.
                if documentJson['user_id'].strip() == str(self.user_id):
                    print("User IDs match.")
                else:
                    print("User IDs do not match.")
                if documentJson['category'] == self.category:
                    print("Categories match.")
                else:
                    print("Categories do not match.")
                if (documentJson['category'] == self.category and 
                    documentJson['user_id'].strip() == str(self.user_id)):
                    tasks.append(documentJson)
                else:
                    print(f"DEBUG: Skipping task from DB due to category/user_id mismatch in document: {documentJson}")
            
            print(f'DEBUG: Found {len(tasks)} total tasks in LangChain PGVector for {self.category}/{self.user_id}')
            return tasks
        except Exception as e:
            print(f"ERROR: Error getting all tasks from LangChain PGVector (direct query): {e}")
            traceback.print_exc()
            return [t for t in self.tasks if t.get('user_id') == self.user_id and t.get('category') == self.category]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks, refreshed from vector store"""
        self.tasks = self._get_all_tasks_from_vector_store()
        return self.tasks

    def _update_task_in_vector_store(self, task: Dict[str, Any]):
        task_id = str(task['id'])
        print(f"DEBUG: _update_task_in_vector_store START for task_id: {task_id}")
        
        # --- Direct SQL Delete Approach (if LangChain's delete fails) ---
        """try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            # Get the collection UUID (needed to filter rows belonging to this logical collection)
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if collection_uuid_row:
                collection_uuid = collection_uuid_row[0]
                print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                
                delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s::uuid AND (document::jsonb)->>'id' = %s::text;"
                cur.execute(delete_sql, (str(collection_uuid), str(task_id)))
                rows_deleted = cur.rowcount
                conn.commit() # <<< Explicit COMMIT is crucial for direct SQL
                print(f"DEBUG: Direct SQL DELETE completed. Rows deleted: {rows_deleted} for task ID {task_id}.")
                if rows_deleted == 0:
                    print(f"WARNING: Direct SQL delete found 0 rows for task ID {task_id}. Check ID, collection, or if already deleted.")
            else:
                print(f"WARNING: Collection '{self.collection_name}' not found during direct SQL delete attempt. No delete performed.")

            cur.close()
            conn.close()
        except Exception as e:
            print(f"ERROR: Exception during direct SQL delete for task {task_id}: {e}")
            traceback.print_exc()
            # If deletion failed, you might still want to try to add the new version."""

        # --- Use task manager vector store ---
        print(f"DEBUG: Deleting/Adding updated task to vector store for task_id: {task_id}")
        task_manager = self.agents.get('task_manager')
        if not task_manager or not hasattr(task_manager, 'vector_store'):
            print("ERROR: Task manager or its vector store not available")
        else:
            # Delete existing task with this ID
            task_manager.vector_store.delete(
                filter_={"id": task_id}
            )
            # Add the updated task
            task_manager.vector_store.add_documents([Document(page_content=json.dumps(task))])
            print(f"DEBUG: _update_task_in_vector_store END for task_id: {task_id}")

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a task by its ID, check LangChain PGVector.
        """
        # Search LangChain PGVector directly using the ID in metadata
        print(f"DEBUG: Searching LangChain PGVector for collection: {self.collection_name}")
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            # Find the collection_id for your logical collection_name
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if not collection_uuid_row:
                print(f"WARNING: Collection '{self.collection_name}' not found during get_task_by_id.")
                cur.close()
                conn.close()
                return None
            
            collection_uuid = collection_uuid_row[0]

            # Query langchain_pg_embedding table, filtering by collection_id and document->>'id'
            # Cast both sides to text for proper comparison
            cur.execute("""
                SELECT document 
                FROM langchain_pg_embedding 
                WHERE collection_id = %s::uuid 
                AND (document::jsonb)->>'id' = %s::text
                LIMIT 1;
            """, (str(collection_uuid), str(task_id)))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result and result[0]:
                found_task = result[0] # cmetadata is the first (and only) column selected
                print(f"DEBUG: Task {task_id} found in LangChain PGVector.")
                return found_task
            else:
                print(f"DEBUG: Task {task_id} not found in LangChain PGVector.")
                return None
        except Exception as e:
            print(f"ERROR: Error searching for task {task_id} in LangChain PGVector: {e}")
            traceback.print_exc()
            return None

    def update_task_manual(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Manually update a task"""
        # Fetch the task, prioritize from memory, then DB
        task = None
        for i, t in enumerate(self.tasks):
            if t['id'] == task_id:
                task = t
                break
        
        if not task:
            task = self.get_task_by_id(task_id)
        
        print(f"DEBUG: update_task_manual - task type: {type(task)}, content: {task}")
        if isinstance(task, str):
            task = json.loads(task)
        if task:
            # Update in vector store (DB)
            self._update_task_in_vector_store(task)
            return task
        
        return None
    
    def mark_task_complete(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Mark a task as complete"""
        return self.update_task_manual(task_id, {
            'completed': True,
            'completed_at': datetime.now().isoformat()
        })

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the vector store.
        Args: task_id: The ID of the task to delete
        Returns: bool: True if deletion was successful, False otherwise
        """
        print(f"delete_task called for task_id: {task_id}")
        try:
            # Get the task manager agent
            task_manager = self.agents.get('task_manager')
            if not task_manager or not hasattr(task_manager, 'vector_store'):
                print("ERROR: Task manager or its vector store not available")
                return False
                
            # First verify the task exists
            task = self.get_task_by_id(task_id)
            if not task:
                print(f"Task {task_id} not found")
                return False
                
            # Delete from vector store
            try:
                # Try to delete using the vector store's delete method
                task_manager.vector_store.delete(
                    filter_={"id": str(task_id)}
                )
                print(f"Successfully deleted task {task_id}")
                # Direct SQL delete
                try:
                    conn = psycopg2.connect(self.db_connection_string)
                    cur = conn.cursor()
                    
                    # Get the collection UUID (needed to filter rows belonging to this logical collection)
                    cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
                    collection_uuid_row = cur.fetchone()
                    
                    if collection_uuid_row:
                        collection_uuid = collection_uuid_row[0]
                        print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                        
                        delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s::uuid AND (document::jsonb)->>'id' = %s::text;"
                        cur.execute(delete_sql, (str(collection_uuid), str(task_id)))
                        rows_deleted = cur.rowcount
                        conn.commit() # <<< Explicit COMMIT is crucial for direct SQL
                        print(f"DEBUG: Direct SQL DELETE completed. Rows deleted: {rows_deleted} for task ID {task_id}.")
                        if rows_deleted == 0:
                            print(f"WARNING: Direct SQL delete found 0 rows for task ID {task_id}. Check ID, collection, or if already deleted.")                   
                    cur.close()
                    conn.close()
                except Exception as e:
                    print(f"ERROR: Exception during direct SQL delete for task {task_id}: {e}")
                # End of Direct SQL delete
                return True
                
            except Exception as e:
                print(f"Error deleting task {task_id} from vector store: {str(e)}")
                return False
                        
        except Exception as e:
            print(f"Unexpected error in delete_task: {str(e)}")
            traceback.print_exc()
            return False

class EducationManager:
    def __init__(self):
        # A nested dictionary to store assistants:
        # self.assistants = {
        #   'user_lance_email@example.com': {
        #     'work': TaskAssistant3_instance_for_work,
        #     'personal': TaskAssistant3_instance_for_personal
        #   },
        # }
        self.assistants: Dict[str, Dict[str, MultiAgentEducationAssistant]] = {}
        self.qdrant_url = os.getenv("QDRANT_URL")
        print(f"QDRANT_URL: {self.qdrant_url}")

        # Parent assistant configuration
        self.parent_role = """You are a friendly and organized parent task assistant powered by LangGraph workflows. Your main focus is helping parents stay on top of their children's tasks and commitments through structured processing. Specifically:

- Help track and organize children's tasks using advanced workflow capabilities
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance
- Use intelligent task extraction and intent recognition

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?"""

        # Student assistant configuration
        self.student_role = """You are a focused and efficient student task assistant powered by LangGraph workflows. 

Your main focus is helping students manage their student commitments with realistic timeframes through structured processing. 

Specifically:

- Help track and organize student tasks using advanced workflow capabilities
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:
  • Developer Relations features: typically 1 day
  • Course lesson reviews/feedback: typically 2 days
  • Documentation sprints: typically 3 days
- Help prioritize tasks based on deadlines and team dependencies
- Maintain a professional tone while helping the user stay accountable
- Use intelligent task extraction and intent recognition

Your communication style should be supportive but practical. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?"""

    def get_assistant(self, category: str, user_id: str) -> MultiAgentEducationAssistant:
        """
        Retrieves a TaskAssistant instance for a given user and category.
        If it doesn't exist, a new one is created.
        """
        # Ensure the user's dictionary exists
        if user_id not in self.assistants:
            self.assistants[user_id] = {}
            print(f"INFO: New user '{user_id}' detected. Initializing assistant dictionary.")
        
        # Check if the specific assistant exists for this user/category
        if category not in self.assistants[user_id]:
            print(f"INFO: Creating new '{category}' assistant for user '{user_id}'.")
            
            # Parent assistant configuration
            if category == 'parent':
                role_prompt = self.parent_role
            # Student assistant configuration
            elif category == 'student':
                role_prompt = self.student_role
            else:
                raise ValueError(f"Unknown assistant category: {category}")

            try:
                # Instantiate and store the new assistant, passing the user_id
                self.assistants[user_id][category] = MultiAgentEducationAssistant(
                    role_prompt=role_prompt,
                    category=category,
                    user_id=user_id # Crucially, pass the unique user ID
                )
                print(f"INFO: New '{category}' assistant created for user '{user_id}'.")
            except Exception as e:
                print(f"ERROR: Failed to create '{category}' assistant for user '{user_id}': {e}")
                raise
            
        return self.assistants[user_id][category]

# Test function
async def test_multi_agent_system():
    """Test the multi-agent system"""
    education_manager = EducationManager()
    assistant = education_manager.get_assistant('parent', 'user123')
    if assistant:
        response, tasks = await assistant.process_message(
            "I need to create a study plan for learning Python programming with deadlines"
        )
        print(f"Response: {response}")
        print(f"Created tasks: {len(tasks)}")
        print(f"Agent status: {assistant.get_agent_status()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_multi_agent_system())
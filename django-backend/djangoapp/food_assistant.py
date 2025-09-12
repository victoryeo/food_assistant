from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
#from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv
import operator
import psycopg2 # For direct DB operations if needed, or rely on PGVector
import os
import re
import json
import uuid
import traceback
from bson import ObjectId

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
os.environ["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
os.environ["SUPABASE_DB_CONNECTION_STRING"] = os.getenv("SUPABASE_DB_CONNECTION_STRING")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# State definition for LangGraph
class TaskState2(TypedDict):
    messages: Annotated[List[dict], operator.add]
    user_input: str
    tasks: List[Dict[str, Any]]
    created_tasks: List[Dict[str, Any]]
    updated_tasks: List[Dict[str, Any]]
    intent: str  # create, update, complete, summary, query
    category: str  # work, personal
    user_id: str
    response: str
    extracted_info: Dict[str, Any]
    retrieved_content: List[Document]  # New field for RAG content

# NOTE: Langgraph education assistant with Agentic RAG
class EducationAssistant:
    def __init__(self, role_prompt: str, category: str, user_id: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        self.db_connection_string = os.getenv("SUPABASE_DB_CONNECTION_STRING")
        if not self.db_connection_string:
            raise ValueError("SUPABASE_DB_CONNECTION_STRING environment variable is not set.")

        # LLM setup
        print("Info: LLM setup")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                #self.llm = ChatOpenAI(model="gpt-4")
                print("OpenAI API key is set")
            else:
                raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is set")

        # Task tracking
        self.tasks = []  # In-memory storage, kept for immediate operations
        self.conversation_history = []
        
        # Setup embeddings for vector store
        print("Setup embeddings for vector store")
        self.embeddings = self._setup_embeddings()
        
        # Initialize PGVector client and table
        self.collection_name = f"tasks_{category}_{user_id}" # Used as table name
        self.knowledge_collection_name = f"knowledge_{category}_{user_id}"

        # Test PostgreSQL connection
        try:
            conn = psycopg2.connect(self.db_connection_string)
            print("✅ PostgreSQL connection successful!")
            conn.close()
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")

        # Initialize PGVector vector store
        print("Initialize PGVector client and table")
        try:
            self.vector_store = PGVector(
                collection_name=self.collection_name,
                connection=self.db_connection_string,
                embeddings=self.embeddings,
                distance_strategy="cosine", # Align with Qdrant's COSINE
                use_jsonb=True # <--- ADD THIS LINE TO EXPLICITLY SET JSONB
            )
            
            self.knowledge_store = PGVector(
                collection_name=self.knowledge_collection_name,
                connection=self.db_connection_string,
                embeddings=self.embeddings,
                distance_strategy="cosine",
                use_jsonb=True
            )
            print(f"✅ PGVector initialized with collections: {self.collection_name}, {self.knowledge_collection_name}")
        except Exception as e:
            print(f"❌ PGVector initialization failed: {e}")

        # Load educational content
        self._load_educational_datasets()
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _setup_embeddings(self):
        """Setup embeddings for vector store"""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings
    
    def _load_exam_data(self, file_path: str) -> List[Document]:
        """Load and process data from exam JSON file"""
        try:
            print(f"Attempting to load exam data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    # Remove JavaScript-style comments (// ...)
                    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
                    # Remove multi-line comments (/* ... */)
                    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    raise
                
            documents = []
            for student in data.get('students', []):
                # Create a document for the student's overall information
                student_doc = {
                    'page_content': f"Student: {student.get('name')} (ID: {student.get('student_id')})\n                    Personality: {', '.join(student.get('initial_personality', []))}\n"
                }
                
                # Add scores information
                if 'math_scores' in student and student['math_scores']:
                    scores = student['math_scores']
                    avg_score = sum(s['score']/s['max_score'] for s in scores) / len(scores)
                    topics = set(s['topic'] for s in scores)
                    
                    student_doc['page_content'] += f"\nMath Performance (Average: {avg_score:.1%})\n"
                    student_doc['page_content'] += f"Topics covered: {', '.join(topics)}\n"
                    
                    # Add recent scores, handling potential None values in timestamps
                    try:
                        recent_scores = sorted(
                            scores,
                            key=lambda x: x.get('timestamp') or '1970-01-01T00:00:00Z',
                            reverse=True
                        )[:3]
                        student_doc['page_content'] += "\nRecent scores:\n"
                        for score in recent_scores:
                            student_doc['page_content'] += f"- {score.get('topic', 'Unknown')}: {score.get('score', '?')}/{score.get('max_score', '?')} ({score.get('type', 'Exercise')})\n"
                    except Exception as e:
                        print(f"Warning: Error processing scores for student {student.get('name', 'Unknown')}: {e}")
                        student_doc['page_content'] += "\n(Score information not available)\n"
                
                # Add metadata
                student_doc['metadata'] = {
                    'source': 'deepseek_data',
                    'student_id': student.get('student_id'),
                    'student_name': student.get('name'),
                    'category': self.category,
                    'user_id': self.user_id,
                    'loaded_at': datetime.now().isoformat()
                }

                print(student_doc)
                
                documents.append(Document(**student_doc))
                
            print(f"✅ Loaded {len(documents)} student records from exam data")
            return documents
            
        except Exception as e:
            print(f"❌ Failed to load exam data: {e}")
            return []
    
    def _load_educational_datasets(self):
        """Load educational datasets from various sources and store in vector database"""
        print("Loading educational datasets...")
        
        # Load from deepseek.json if available
        examdata_path = os.path.join(os.path.dirname(__file__), 'deepseek_json_20250820_1cc700.json')
        if os.path.exists(examdata_path):
            print("Found examdata data, loading...")
            examdata_docs = self._load_exam_data(examdata_path)
            if examdata_docs:
                try:
                    for doc in examdata_docs:
                        # Ensure metadata exists
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc.metadata = {}

                        # Convert UUIDs to strings in metadata
                        if hasattr(doc, 'metadata') and doc.metadata:
                            clean_metadata = {}
                            for k, v in doc.metadata.items():
                                if isinstance(v, (uuid.UUID, ObjectId)):
                                    clean_metadata[k] = str(v)
                                elif isinstance(v, dict):
                                    # Handle nested dictionaries
                                    clean_metadata[k] = {}
                                    for nk, nv in v.items():
                                        if isinstance(nv, (uuid.UUID, ObjectId)):
                                            clean_metadata[k][nk] = str(nv)
                                        else:
                                            clean_metadata[k][nk] = nv
                            doc.metadata = clean_metadata
                    self.knowledge_store.add_documents(examdata_docs)
                    print(f"✅ Added {len(examdata_docs)} documents from exam data")
                except Exception as e:
                    print(f"❌ Failed to add exam data to knowledge store: {e}")

        # Define HuggingFace datasets to load
        datasets_config = [
            {
                "path": "wikipedia",
                "name": "20220301.en",
                "page_content_column": "text",
                "max_docs": 500  # Reduced from 1000 to balance with exam data
            },
            {
                "path": "squad",
                "page_content_column": "context",
                "max_docs": 300  # Reduced from 500 to balance with exam data
            }
        ]
        
        # Load from HuggingFace datasets
        print(f"Loading HuggingFace datasets: {datasets_config}")
        for config in datasets_config:
            try:
                print(f"Loading dataset: {config.get('path')} - {config.get('name', '')}")
            
                # Use try/except with different parameter combinations
                loader_params = {
                    "path": config["path"],
                    "page_content_column": config["page_content_column"]
                }
            
                # Add name if specified
                if config.get("name"):
                    loader_params["name"] = config["name"]
            
                print(f"Loading documents from HuggingFace dataset: {config['path']}")
                loader = HuggingFaceDatasetLoader(**loader_params)

                # Load documents with error handling
                try:
                    documents = loader.load()
                    print(f"Successfully loaded {len(documents)} documents")
                except Exception as load_error:
                    print(f"Error loading documents: {load_error}")
                    continue
            
                # Limit documents if specified
                if config.get("max_docs") and len(documents) > config["max_docs"]:
                    documents = documents[:config["max_docs"]]
                    print(f"Limited to {len(documents)} documents")
                
                # Add metadata
                for doc in documents:
                    doc.metadata.update({
                        "source": config["path"],
                        "subset": config.get("name", ""),
                        "category": self.category,
                            "user_id": self.user_id,
                            "loaded_at": datetime.now().isoformat()
                        })
                    
                # Store in knowledge vector store
                if documents:
                    # First serialize all documents to handle UUIDs and other non-serializable types
                    for doc in documents:
                        try:
                            # Ensure metadata exists
                            if not hasattr(doc, 'metadata') or doc.metadata is None:
                                doc.metadata = {}
    
                            # Handle ID
                            if 'id' not in doc.metadata and '_id' in doc.metadata:
                                doc.metadata['id'] = str(doc.metadata['_id'])
                            elif 'id' not in doc.metadata:
                                doc.metadata['id'] = str(uuid.uuid4())
    
                            # Ensure required fields
                            if not hasattr(doc, 'page_content') and hasattr(doc, 'content'):
                                doc.page_content = doc.content
    
                            # Convert UUIDs to strings in metadata
                            if hasattr(doc, 'metadata') and doc.metadata:
                                clean_metadata = {}
                                for k, v in doc.metadata.items():
                                    if isinstance(v, (uuid.UUID, ObjectId)):
                                        clean_metadata[k] = str(v)
                                    elif isinstance(v, dict):
                                        # Handle nested dictionaries
                                        clean_metadata[k] = {}
                                        for nk, nv in v.items():
                                            if isinstance(nv, (uuid.UUID, ObjectId)):
                                                clean_metadata[k][nk] = str(nv)
                                            elif isinstance(nv, datetime):
                                                clean_metadata[k][nk] = nv.isoformat()
                                        else:
                                            clean_metadata[k][nk] = nv
                            elif isinstance(v, datetime):
                                clean_metadata[k] = v.isoformat()
                            else:
                                clean_metadata[k] = v
                            doc.metadata = clean_metadata
        
                        except Exception as e:
                            print(f"⚠️ Error processing document: {e}")
                            continue

                    # Store in knowledge vector store
                    try:
                        # Convert to Document objects if they aren't already
                        from langchain_core.documents import Document
                        docs_to_store = []
                        
                        for doc in documents:
                            try:
                                if not isinstance(doc, Document):
                                    # Create a new Document with required fields
                                    doc_obj = Document(
                                        page_content=getattr(doc, 'page_content', ''),
                                        metadata=getattr(doc, 'metadata', {})
                                    )
                                    docs_to_store.append(doc_obj)
                                else:
                                    docs_to_store.append(doc)
                            except Exception as e:
                                print(f"⚠️ Error creating Document object: {e}")
                                continue
                
                        if docs_to_store:
                            self.knowledge_store.add_documents(docs_to_store)
                            print(f"✅ Successfully stored {len(docs_to_store)} documents from {config['path']}")
                        else:
                            print("⚠️ No valid documents to store after processing")
                    except Exception as e:
                        print(f"❌ Error adding documents to vector store: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"⚠️ No documents to store for {config['path']}")
                        
            except Exception as e:
                print(f"❌ Failed to process dataset {config.get('path')}: {e}")
                traceback.print_exc()
        
    def _retrieve_educational_content(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant educational content using RAG"""
        try:
            results = self.knowledge_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error retrieving educational content: {e}")
            return []
    
    def _task_to_document(self, task: Dict[str, Any]) -> Document:
        """Convert a task dictionary to a Document for vector storage"""
        content_parts = [
            f"Title: {task.get('title', '')}",
            f"Description: {task.get('description', '')}",
            f"Category: {task.get('category', '')}",
            f"Status: {task.get('status', 'pending')}",
            f"Priority: {task.get('priority', 'medium')}",
        ]
        
        if task.get('tags'):
            content_parts.append(f"Tags: {', '.join(task['tags'])}")
        
        if task.get('deadline'):
            content_parts.append(f"Deadline: {task['deadline']}")
        
        content = "\n".join(content_parts)
        
        # Store full task data in metadata
        metadata = task.copy()
        metadata['searchable_content'] = content # Redundant, but kept for consistency
        
        # PGVector typically handles the UUID internally if not provided,
        # but we use task['id'] as the primary key for direct control.
        return Document(
            page_content=content,
            metadata=metadata
        )
    
    def _add_task_to_vector_store(self, task: Dict[str, Any]) -> str:
        """Add a task to the vector store and return the document ID (primary key)"""
        document = self._task_to_document(task)
        
        try:
            # PGVector's add_documents can directly handle this.
            # It will use the 'id' from metadata if specified, or generate one.
            doc_ids = self.vector_store.add_documents([document])
            doc_id = doc_ids[0] if doc_ids else None # Get the first ID
            if doc_id:
                print(f"DEBUG: Successfully added task to vector store with ID: {doc_id}")
            else:
                print(f"DEBUG: PGVector.add_documents returned no ID for task: {task.get('id')}. This is unexpected if no error occurred.")
            return doc_id
        except Exception as e:
            print(f"ERROR: Failed to add task to vector store: {e}")
            traceback.print_exc() # Print full traceback for more details
            return None # Indicate failure

    def _update_task_in_vector_store(self, task: Dict[str, Any]):
        task_id = str(task['id'])
        print(f"DEBUG: _update_task_in_vector_store START for task_id: {task_id}")
        
        # --- Direct SQL Delete Approach (if LangChain's delete fails) ---
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            # Get the collection UUID (needed to filter rows belonging to this logical collection)
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if collection_uuid_row:
                collection_uuid = collection_uuid_row[0]
                print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                
                delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s;"
                cur.execute(delete_sql, (str(collection_uuid), task_id))
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
            # If deletion failed, you might still want to try to add the new version.

        # --- Add the updated document ---
        print(f"DEBUG: Calling _add_task_to_vector_store for task_id: {task_id} after delete attempt.")
        self._add_task_to_vector_store(task)
        print(f"DEBUG: _update_task_in_vector_store END for task_id: {task_id}")

    def _search_tasks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search tasks using vector similarity in PGVector"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            tasks = []
            for doc in results:
                # PGVector's results typically return Document objects with metadata
                if 'id' in doc.metadata: # Ensure your original task ID is in metadata
                    tasks.append(doc.metadata)
                else:
                    print(f"WARNING: Document found without 'id' in metadata: {doc.metadata}")
            print(f"DEBUG: Found {len(tasks)} tasks via similarity search.")
            return tasks
        except Exception as e:
            print(f"ERROR: Search error in LangChain PGVector: {e}")
            traceback.print_exc()
            return [t for t in self.tasks if t.get('user_id') == self.user_id and t.get('category') == self.category] # Fallback to in-memory, filtered

    def _get_all_tasks_from_vector_store(self) -> List[Dict[str, Any]]:
        """Get all tasks from PGVector, filtered by category and user_id"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            # First, find the collection_id for your logical collection_name
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if not collection_uuid_row:
                print(f"WARNING: Collection '{self.collection_name}' not found in langchain_pg_collection. No tasks to retrieve.")
                cur.close()
                conn.close()
                return []

            collection_uuid = collection_uuid_row[0]

            # Then, retrieve documents from langchain_pg_embedding for that collection_id
            # The metadata is stored in the 'cmetadata' column
            cur.execute(f"SELECT cmetadata FROM langchain_pg_embedding WHERE collection_id = %s;", (str(collection_uuid),))
            
            all_tasks_metadata = cur.fetchall()
            cur.close()
            conn.close()
            
            tasks = []
            for row in all_tasks_metadata:
                metadata = row[0] 
                # The 'cmetadata' column in langchain_pg_embedding IS your metadata
                # So it should contain 'id', 'category', 'user_id' directly.
                if (metadata.get('category') == self.category and 
                    metadata.get('user_id') == self.user_id):
                    tasks.append(metadata)
                else:
                    print(f"DEBUG: Skipping task from DB due to category/user_id mismatch in cmetadata: {metadata}")
            
            print(f'DEBUG: Found {len(tasks)} total tasks in LangChain PGVector for {self.category}/{self.user_id}')
            return tasks
        except Exception as e:
            print(f"ERROR: Error getting all tasks from LangChain PGVector (direct query): {e}")
            traceback.print_exc()
            return [t for t in self.tasks if t.get('user_id') == self.user_id and t.get('category') == self.category]
        
    def _create_workflow(self):
        """Create the LangGraph workflow for task processing with RAG"""
        workflow = StateGraph(TaskState2)
        
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("retrieve_educational_content", self._retrieve_educational_content_node)
        workflow.add_node("extract_task_info", self._extract_task_info)
        workflow.add_node("create_tasks", self._create_tasks)
        workflow.add_node("update_tasks", self._update_tasks)
        workflow.add_node("complete_tasks", self._complete_tasks)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("generate_response", self._generate_response)
        
        workflow.set_entry_point("analyze_intent")
        
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_by_intent,
            {
                "create": "retrieve_educational_content",
                "update": "retrieve_educational_content", 
                "complete": "complete_tasks",
                "summary": "generate_summary",
                "query": "retrieve_educational_content"
            }
        )
        
        workflow.add_edge("retrieve_educational_content", "extract_task_info")
        
        workflow.add_conditional_edges(
            "extract_task_info",
            self._route_by_action,
            {
                "create": "create_tasks",
                "update": "update_tasks",
                "query": "generate_response",
                "complete": "complete_tasks"
            }
        )
        
        workflow.add_edge("create_tasks", "generate_response")
        workflow.add_edge("update_tasks", "generate_response")
        workflow.add_edge("complete_tasks", "generate_response")
        workflow.add_edge("generate_summary", "generate_response")
        
        workflow.add_edge("generate_response", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_intent(self, state: TaskState2) -> TaskState2:
        """Analyze user intent from the input"""
        user_input = state["user_input"].lower()
        
        if any(keyword in user_input for keyword in ["create", "add", "new", "todo"]):
            intent = "create"
        elif any(keyword in user_input for keyword in ["update", "modify", "change", "edit"]):
            intent = "update"
        elif any(keyword in user_input for keyword in ["complete", "done", "finished", "mark"]):
            intent = "complete"
        elif any(keyword in user_input for keyword in ["summary", "list", "show", "overview"]):
            intent = "summary"
        else:
            intent = "query"
        
        state["intent"] = intent
        return state
    
    def _retrieve_educational_content_node(self, state: TaskState2) -> TaskState2:
        """Retrieve relevant educational content using RAG"""
        user_input = state["user_input"]
        
        # Enhanced query generation for better retrieval
        enhanced_query = self._generate_enhanced_query(user_input, state["intent"])
        
        # Retrieve relevant educational content
        retrieved_content = self._retrieve_educational_content(enhanced_query, k=3)
        
        state["retrieved_content"] = retrieved_content
        state["messages"].append({
            "role": "system", 
            "content": f"Retrieved {len(retrieved_content)} educational documents for query: {enhanced_query}"
        })
        
        return state
    
    def _generate_enhanced_query(self, user_input: str, intent: str) -> str:
        """Generate enhanced query for better educational content retrieval"""
        query_prompt = f"""
        Based on the user input and intent, generate an enhanced search query for educational content.
        
        User Input: {user_input}
        Intent: {intent}
        Category: {self.category}
        
        Generate a concise, informative query that would help find relevant educational materials.
        Focus on key concepts, subjects, and learning objectives.
        
        Enhanced Query:
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=query_prompt)])
            return response.content.strip()
        except:
            return user_input  # Fallback to original input
    
    def _route_by_intent(self, state: TaskState2) -> str:
        """Route based on detected intent"""
        return state["intent"]
    
    def _route_by_action(self, state: TaskState2) -> str:
        """Route based on the action needed after info extraction"""
        return state["intent"]
    
    def _extract_task_info(self, state: TaskState2) -> TaskState2:
        """Extract task information using LLM with RAG context"""
        print(f"Extracting task info: {state['user_input']}")
        print(f"Retrieved content: {state['retrieved_content']}")
        extraction_prompt = """
        Extract task information from the user input, considering the retrieved educational content.
        
        Retrieved Educational Content:
        {retrieved_content}
        
        Extract a JSON object with the following structure:
        {{
            "tasks": [
                {{
                    "title": "task title",
                    "description": "optional description with educational context",
                    "deadline": "ISO format datetime if mentioned",
                    "educational_context": "how this relates to learning objectives"
                }}
            ]
        }}
        
        User input: {user_input}
        """

        # Format retrieved content for the prompt
        content_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content[:500]}..."
            for doc in state.get("retrieved_content", [])
        ])

        print(extraction_prompt.format(user_input=state["user_input"], retrieved_content=content_text))
        
        try:
            print("Extracting task info...")
            response = self.llm.invoke([
                HumanMessage(content=extraction_prompt.format(
                    retrieved_content=content_text,
                    user_input=state["user_input"]
                ))
            ])
            
            print(f"Response: {response}")
            json_match = re.findall(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            print(f"JSON match: {json_match}")
            if len(json_match) == 0:
                raise Exception("No JSON match found")
            print(f"last match: {json_match[-1]}")
            if json_match:
                print(f"JSON match:1")
                # Remove comments from JSON string before parsing
                json_str = json_match[-1]
                # Remove single-line comments
                json_str = re.sub(r'//.*?$|#.*?$', '', json_str, flags=re.MULTILINE)
                # Remove any remaining whitespace
                json_str = json_str.strip()
                extracted_info = json.loads(json_str)
            else:
                # Fallback to basic extraction
                print(f"JSON match:2")
                extracted_info = self._basic_task_extraction(state["user_input"])
            
            state["extracted_info"] = extracted_info
            print(f"Extracted task info: {state['extracted_info']}")
        except Exception as e:
            print(f"Extraction failed: {str(e)}")
            # Fallback to basic extraction
            state["extracted_info"] = self._basic_task_extraction(state["user_input"])
            state["messages"].append({"role": "system", "content": f"Extraction fallback used: {str(e)}"})
        
        return state
    
    def _basic_task_extraction(self, user_input: str) -> Dict[str, Any]:
        """Fallback basic task extraction"""
        tasks = []
        lines = user_input.split('\n')
        
        for line in lines:
            if any(marker in line for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']):
                task_text = line.strip()
                # Remove numbering/bullets
                for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']:
                    task_text = task_text.replace(marker, '').strip()
                
                if task_text:
                    task = {"title": task_text}
                    deadline = self._extract_deadline(task_text)
                    if deadline:
                        task["deadline"] = deadline
                    tasks.append(task)
        
        return {"tasks": tasks}
    
    def _create_tasks(self, state: TaskState2) -> TaskState2:
        """Create new tasks with educational context"""
        created_tasks = []
        print(f"Creating tasks: {state['user_input']}")

        for task_info in state["extracted_info"].get("tasks", []):
            task = {
                'id': str(uuid.uuid4()),
                'title': task_info.get('title', ''),
                'description': task_info.get('description', ''),
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'category': self.category,
                'user_id': self.user_id,
                'completed': False,
                'educational_context': task_info.get('educational_context', '')
            }
            
            # Add optional fields
            if task_info.get('deadline'):
                task['deadline'] = task_info['deadline']
            if task_info.get('priority'):
                task['priority'] = task_info['priority']
            if task_info.get('tags'):
                task['tags'] = task_info['tags']
            
            # Add to both in-memory list and vector store
            self.tasks.append(task)
            print(f"Created task: {task}")
            self._add_task_to_vector_store(task)
            created_tasks.append(task)
        
        state["created_tasks"] = created_tasks
        state["tasks"] = self.tasks
        return state
    
    def _update_tasks(self, state: TaskState2) -> TaskState2:
        """Update existing tasks with educational context"""
        print(f"Updating tasks: {state['user_input']}")
        updated_tasks = []
        user_input = state["user_input"].lower()
        
        # Search for tasks to update using vector similarity
        potential_tasks = self._search_tasks(user_input, k=10)
        
        for task in potential_tasks:
            if not task.get('completed', False):
                updated = False
                # original_task = task.copy() # not used
                
                # Update deadline if mentioned
                new_deadline = self._extract_deadline(user_input)
                if new_deadline and new_deadline != task.get('deadline'):
                    task['deadline'] = new_deadline
                    updated = True
                
                # Update priority if mentioned
                if 'high priority' in user_input or 'urgent' in user_input:
                    task['priority'] = 'high'
                    updated = True
                elif 'low priority' in user_input:
                    task['priority'] = 'low'
                    updated = True
                
                # Update educational context if relevant content was retrieved
                if state.get("retrieved_content"):
                    task['educational_context'] = "Updated with new learning materials"
                    updated = True
                
                if updated:
                    task['updated_at'] = datetime.now().isoformat()

                    # Update in both memory and vector store
                    for i, mem_task in enumerate(self.tasks):
                        if mem_task['id'] == task['id']:
                            self.tasks[i] = task
                            break
                    
                    self._update_task_in_vector_store(task)
                    updated_tasks.append(task)
                    break  # Update only first matching task
        
        state["updated_tasks"] = updated_tasks
        state["tasks"] = self.tasks
        return state
    
    def _complete_tasks(self, state: TaskState2) -> TaskState2:
        """Mark tasks as complete"""
        print(f"Completing tasks: {state['user_input']}")
        user_input = state["user_input"].lower()
        completed_tasks = []
        
        # Search for tasks to complete using vector similarity
        potential_tasks = self._search_tasks(user_input, k=10)
        
        for task in potential_tasks:
            if not task.get('completed', False):
                # A more robust check might be needed here to confirm the exact task,
                # perhaps with an LLM call or ID extraction.
                # For simplicity, if title or ID is in input, we consider it a match.
                if task['title'].lower() in user_input or task['id'] in user_input:
                    task['completed'] = True
                    task['completed_at'] = datetime.now().isoformat()
                    
                    # Update in both memory and vector store
                    for i, mem_task in enumerate(self.tasks):
                        if mem_task['id'] == task['id']:
                            self.tasks[i] = task
                            break
                    
                    self._update_task_in_vector_store(task)
                    completed_tasks.append(task)
        
        state["updated_tasks"] = completed_tasks # Reusing updated_tasks for completed tasks
        state["tasks"] = self.tasks
        return state
    
    def _generate_summary(self, state: TaskState2) -> TaskState2:
        """Generate a task summary with educational insights"""
        print(f"Generating summary: {state['user_input']}")
        tasks_summary = self.get_tasks_summary()
        
        summary_prompt = """
        Based on the task summary and retrieved educational content, create a comprehensive overview.
        Include learning progress insights and recommendations based on the educational materials.
        
        Task Summary:
        {tasks_summary}
        
        Retrieved Educational Content:
        {retrieved_content}
        
        Category: {category}
        """
        
        content_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content[:300]}..."
            for doc in state.get("retrieved_content", [])
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=summary_prompt.format(
                tasks_summary=json.dumps(tasks_summary, indent=2),
                retrieved_content=content_text,
                category=self.category
            ))
        ])
        
        state["response"] = response.content
        return state
    
    def _generate_response(self, state: TaskState2) -> TaskState2:
        """Generate final response with educational context"""
        print(f"Generating response: {state['user_input']}")
        if state.get("response"):
            print("Response already generated")
            return state  # Already have a response from summary
        
        context = {
            "intent": state["intent"],
            "created_tasks": state.get("created_tasks", []),
            "updated_tasks": state.get("updated_tasks", []),
            "category": self.category,
            "retrieved_content": state.get("retrieved_content", [])
        }
        
        response_prompt = f"""
        {self.role_prompt}
        
        Context: Educational task management for {context["category"]}.
        Intent: {context["intent"]}
        
        Created tasks: {len(context["created_tasks"])}
        Updated tasks: {len(context["updated_tasks"])}
        
        Retrieved Educational Content:
        {self._format_retrieved_content(context["retrieved_content"])}
        
        Original user input: {state["user_input"]}
        Extracted info: {state["extracted_info"]}        
        Provide an educational response that incorporates the retrieved learning materials.
        Offer insights, learning recommendations, and contextual information.
        Limit the response to 100 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=response_prompt)])
        print(f"Response: {response}")
        state["response"] = response.content
        return state

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a task by its ID, checking in-memory first, then LangChain PGVector.
        """
        # Check in-memory tasks first
        for task in self.tasks:
            if task.get('id') == task_id:
                print(f"DEBUG: Task {task_id} found in in-memory list.")
                return task

        # If not in memory, search LangChain PGVector directly using the ID in metadata
        print(f"DEBUG: Task {task_id} not in memory. Searching LangChain PGVector for collection: {self.collection_name}")
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

            # Query langchain_pg_embedding table, filtering by collection_id and metadata->>'id'
            cur.execute(f"SELECT cmetadata FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s LIMIT 1;", 
                        (str(collection_uuid), task_id))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result and result[0]:
                found_task = result[0] # cmetadata is the first (and only) column selected
                print(f"DEBUG: Task {task_id} found in LangChain PGVector.")
                # Optionally add to in-memory list if not already there
                if found_task not in self.tasks:
                    self.tasks.append(found_task)
                return found_task
            else:
                print(f"DEBUG: Task {task_id} not found in LangChain PGVector.")
                return None
        except Exception as e:
            print(f"ERROR: Error searching for task {task_id} in LangChain PGVector: {e}")
            traceback.print_exc()
            return None
  
    def _format_retrieved_content(self, retrieved_content: List[Document]) -> str:
        """Format retrieved content for the prompt"""
        if not retrieved_content:
            return "No relevant educational content found."
        
        formatted = []
        for i, doc in enumerate(retrieved_content, 1):
            formatted.append(f"Document {i} (Source: {doc.metadata.get('source', 'unknown')}):")
            formatted.append(f"{doc.page_content[:200]}...")
            formatted.append("---")
        
        return "\n".join(formatted)

    async def process_message(self, user_input: str):
        """Process user message through the LangGraph workflow"""
        print(f"Processing message: {user_input}")
        initial_state = TaskState2(
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
            retrieved_content=[]
        )
        
        config = {"configurable": {"thread_id": f"{self.user_id}_{self.category}"}}
        try:
            final_state = await self.workflow.ainvoke(initial_state, config)
        except Exception as e:
            print(f"ERROR: Error processing message: {e}")
            traceback.print_exc()
            return "", []
        
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=final_state["response"]))
        
        return final_state["response"], final_state.get("created_tasks", [])
    
    def get_tasks_summary(self):
        print(f"Getting tasks summary from PGVector")
        all_tasks = self._get_all_tasks_from_vector_store() # Fetches tasks from DB
        
        if not all_tasks:
            return "No tasks currently tracked."
        
        now = datetime.now()
        today = now.date()
        this_week_end = today + timedelta(days=(6 - today.weekday()))
        
        overdue = []
        due_today = []
        due_this_week = []
        future = []
        no_deadline = []
        
        for task in all_tasks:
            if task.get('completed', False):
                continue
                
            deadline = task.get('deadline')
            if not deadline:
                no_deadline.append(task)
                continue
            
            try:
                deadline_date = datetime.fromisoformat(deadline).date()
                if deadline_date < today:
                    overdue.append(task)
                elif deadline_date == today:
                    due_today.append(task)
                elif deadline_date <= this_week_end:
                    due_this_week.append(task)
                else:
                    future.append(task)
            except:
                no_deadline.append(task)
        
        return {
            'overdue': overdue,
            'due_today': due_today,
            'due_this_week': due_this_week,
            'future': future,
            'no_deadline': no_deadline
        }
    
    def search_tasks_by_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tasks by content using vector similarity in PGVector"""
        return self._search_tasks(query, k=limit)
    
    def add_task_manual(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manually add a task (useful for direct API calls)"""
        task['id'] = task.get('id', str(uuid.uuid4()))
        task['created_at'] = task.get('created_at', datetime.now().isoformat())
        task['category'] = self.category
        task['user_id'] = self.user_id
        
        self.tasks.append(task) # Add to in-memory
        self._add_task_to_vector_store(task) # Add to DB
        return task
    
    def update_task_manual(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Manually update a task"""
        # Fetch the task, prioritize from memory, then DB
        task = None
        for i, t in enumerate(self.tasks):
            if t['id'] == task_id:
                task = t
                break
        
        if not task:
            task = self.get_task_by_id(task_id) # Use the unified get_task_by_id
        
        if task:
            task.update(updates)
            task['updated_at'] = datetime.now().isoformat()
            
            # Update in memory (if found there)
            for i, t in enumerate(self.tasks):
                if t['id'] == task_id:
                    self.tasks[i] = task
                    break
            else: # If not found in memory, add it
                self.tasks.append(task)
            
            # Update in vector store (DB)
            self._update_task_in_vector_store(task)
            return task
        
        return None
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from both memory and LangChain PGVector.
        This method will also use the internal ID lookup for reliability.
        """
        print(f"DEBUG: delete_task START for task_id: {task_id}")
        task_deleted_from_db = False
        
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if collection_uuid_row:
                collection_uuid = collection_uuid_row[0]
                print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                
                delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s;"
                cur.execute(delete_sql, (str(collection_uuid), task_id))
                rows_deleted = cur.rowcount
                conn.commit() # <<< Explicit COMMIT is crucial for direct SQL
                print(f"DEBUG: Direct SQL DELETE completed. Rows deleted: {rows_deleted} for task ID {task_id}.")
                if rows_deleted == 0:
                    print(f"WARNING: Direct SQL delete found 0 rows for task ID {task_id}. Check ID, collection, or if already deleted.")
            else:
                print(f"WARNING: Collection '{self.collection_name}' not found during direct SQL delete attempt. No delete performed.")

            cur.close()
            conn.close()
            task_deleted_from_db = True

        except Exception as e:
            print(f"ERROR: Exception during DB deletion for task {task_id}: {e}")
            traceback.print_exc()
        
        # Remove from in-memory storage
        initial_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.get('id') != task_id]
        task_was_in_memory = len(self.tasks) < initial_count
        print(f"DEBUG: delete_task END. DB deleted={task_deleted_from_db}, In-memory deleted={task_was_in_memory}")

        return task_deleted_from_db or task_was_in_memory

    def _extract_deadline(self, text: str) -> str:
        """Extract deadline from text - basic implementation"""
        text_lower = text.lower()
        now = datetime.now()
        
        # Example basic extraction (can be expanded)
        if "today" in text_lower:
            return now.date().isoformat()
        elif "tomorrow" in text_lower:
            return (now + timedelta(days=1)).date().isoformat()
        elif "next week" in text_lower:
            # Assuming end of next week
            next_week_end = now + timedelta(days=(6 - now.weekday() + 7))
            return next_week_end.date().isoformat()
        
        # Simple regex for YYYY-MM-DD or MM/DD/YYYY (can be improved)
        date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text)
        if date_match:
            try:
                # Attempt to parse
                date_str = date_match.group(0)
                if '-' in date_str:
                    return datetime.strptime(date_str, '%Y-%m-%d').date().isoformat()
                elif '/' in date_str:
                    return datetime.strptime(date_str, '%m/%d/%Y').date().isoformat()
            except ValueError:
                pass
        
        return "" # No deadline found

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks, refreshed from vector store"""
        self.tasks = self._get_all_tasks_from_vector_store()
        return self.tasks

    def mark_task_complete(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Mark a task as complete"""
        return self.update_task_manual(task_id, {
            'completed': True,
            'completed_at': datetime.now().isoformat()
        })

class EducationManager:
    def __init__(self):
        # A nested dictionary to store assistants:
        # self.assistants = {
        #   'user_lance_email@example.com': {
        #     'work': TaskAssistant3_instance_for_work,
        #     'personal': TaskAssistant3_instance_for_personal
        #   },
        # }
        self.assistants: Dict[str, Dict[str, EducationAssistant]] = {}
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

    def get_assistant(self, category: str, user_id: str) -> EducationAssistant:
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

            # Instantiate and store the new assistant, passing the user_id
            self.assistants[user_id][category] = EducationAssistant(
                role_prompt=role_prompt,
                category=category,
                user_id=user_id # Crucially, pass the unique user ID
            )
            print(f"INFO: New '{category}' assistant created for user '{user_id}'.")
            
        return self.assistants[user_id][category]

# Usage example
async def main():
    education_manager = EducationManager()
    
    # Get parent assistant
    current_user_id = 'lance_doe_123'
    print(f"=== Getting Education Assistant for User: {current_user_id} ===")
    parent_assistant = education_manager.get_assistant('parent', user_id=current_user_id)
    
    # Create parent tasks
    print("=== Creating Education Tasks with LangGraph ===")
    response, created_tasks = await parent_assistant.process_message(
        "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday."
    )
    print("Assistant:", response)
    print(f"Created {len(created_tasks)} tasks")
    
    # Get student assistant
    # --- Get and use the student assistant for the same user ---
    print(f"\n=== Getting Student Assistant for User: {current_user_id} ===")
    student_assistant = education_manager.get_assistant('student', user_id=current_user_id)
    
    # Create student tasks
    print("\n=== Creating Student Tasks with LangGraph ===")
    response, created_tasks = await student_assistant.process_message(
        "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points."
    )
    print("Assistant:", response)
    
    # Get todo summary
    print("\n=== Student Todo Summary ===")
    response, _ = await student_assistant.process_message("Give me a todo summary.")
    print("Assistant:", response)
    
    # Test task completion
    print("\n=== Completing a Task ===")
    if student_assistant.tasks:
        task_to_complete = student_assistant.tasks[0]
        response, _ = await student_assistant.process_message(
            f"Mark task '{task_to_complete['title']}' as complete"
        )
        print("Assistant:", response)
    
    # Test semantic search capabilities
    print("\n=== Semantic Search Test ===")
    search_results = parent_assistant.search_tasks_by_content("video filming module")
    print(f"Found {len(search_results)} tasks matching 'video filming module':")
    for task in search_results[:3]:
        print(f"- {task['title']} (Score-based match)")
    
    # Test task search and update
    print("\n=== Task Search and Update ===")
    response, _ = parent_assistant.process_message(
        "Find the task about filming and update its priority to high"
    )
    print("Assistant:", response)
    
    # Show vector store persistence
    print("\n=== Vector Store Persistence Info ===")
    print(f"Parent tasks in vector store: {len(parent_assistant._get_all_tasks_from_vector_store())}")
    print(f"Student tasks in vector store: {len(student_assistant._get_all_tasks_from_vector_store())}")
    print(f"Task-to-Document mappings: {len(parent_assistant.task_id_to_doc_id)}")
    
    # Print workflow visualization info
    print("\n=== LangGraph Workflow Info ===")
    print("Parent Assistant Workflow Nodes:", list(parent_assistant.workflow.graph.nodes.keys()))
    print("Student Assistant Workflow Nodes:", list(student_assistant.workflow.graph.nodes.keys()))

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
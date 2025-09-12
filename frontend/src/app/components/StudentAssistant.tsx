'use client';

import { useState, useRef, useEffect } from 'react';
import { chatWithAssistant, fetchTasks, completeTask, deleteTask, Task } from '../../services/api/assistantApi';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
};

type AssistantProps = {
  onBack: () => void;
};

export default function StudentAssistant({ onBack }: AssistantProps) {
  const [assistantMessages, setAssistantMessages] = useState<Message>(null);
  const [question, setQuestion] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoadingTasks, setIsLoadingTasks] = useState(false);
  const [isCompletingTask, setIsCompletingTask] = useState<string | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [taskError, setTaskError] = useState<string | null>(null);
  const [showTasks, setShowTasks] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFetchTasks = async () => {
    try {
      setIsLoadingTasks(true);
      setTaskError(null);
      const fetchedTasks = await fetchTasks('student');
      setTasks(fetchedTasks);
      setShowTasks(true);
      
      // Add a system message to show the tasks
      const taskList = fetchedTasks.map(task => 
        `- ${task.title} (${task.status})`
      ).join('\n');
      
      const systemMessage: Message = {
        id: `tasks-${Date.now()}`,
        text: `Here are your current tasks:\n${taskList}`,
        sender: 'assistant',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, systemMessage]);
    } catch (err) {
      console.error('Error fetching tasks:', err);
      setTaskError('Failed to fetch tasks. Please try again.');
    } finally {
      setIsLoadingTasks(false);
    }
  };

  const handleCompleteTask = async (taskId: string) => {
    try {
      setIsCompletingTask(taskId);
      const updatedTask = await completeTask('student', taskId);
      
      // Update the task in the local state
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === updatedTask.id ? { ...task, completed: true } : task
        )
      );
      
      // Show success message
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          text: `Task marked as complete: ${updatedTask.title}`,
          sender: 'assistant',
          timestamp: new Date(),
        },
      ]);
    } catch (err) {
      console.error('Error completing task:', err);
      setTaskError('Failed to complete task. Please try again.');
    } finally {
      setIsCompletingTask(null);
    }
  };

  const handleDeleteTask = async (taskId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent triggering the task completion
    if (!window.confirm('Are you sure you want to delete this task?')) {
      return;
    }

    try {
      setDeletingTaskId(taskId);
      const result = await deleteTask('student', taskId);
      
      if (result.success) {
        // Remove the task from the local state
        setTasks(prevTasks => prevTasks.filter(task => task.id !== taskId));
        
        // Show success message
        setMessages(prev => [
          ...prev,
          {
            id: Date.now().toString(),
            text: result.message || 'Task deleted successfully',
            sender: 'assistant',
            timestamp: new Date(),
          },
        ]);
      } else {
        throw new Error(result.message || 'Failed to delete task');
      }
    } catch (err) {
      console.error('Error deleting task:', err);
      setTaskError('Failed to delete task. Please try again.');
    } finally {
      setDeletingTaskId(null);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    setShowTasks(false); // Hide tasks when sending a new message

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion(inputValue)
    setInputValue('');
    setAssistantMessages(null)

    try {
      // Call the API to get assistant's response
      const response = await chatWithAssistant({
        message: inputValue,
        assistant_type: 'student',
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.response,
        sender: 'assistant',
        timestamp: new Date(response.timestamp),
      };
      console.log(assistantMessage)

      setMessages((prev) => [...prev, assistantMessage]);
      setAssistantMessages(assistantMessage)

      // Fetch tasks after getting assistant's response
      handleFetchTasks();
    } catch (error) {
      console.error('Error getting response from assistant:', error);
      
      // Show error message to user
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: error.message,
        sender: 'assistant',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="bg-blue-600 text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <button
              onClick={onBack}
              className="mr-4 p-2 rounded-full hover:bg-blue-700"
              aria-label="Back to menu"
            >
              ←
            </button>
            <h1 className="text-xl font-bold">Student Assistant</h1>
          </div>
          <button
            onClick={handleFetchTasks}
            disabled={isLoadingTasks}
            className="px-4 py-2 bg-white text-blue-600 rounded-md font-medium hover:bg-blue-50 disabled:opacity-50"
          >
            {isLoadingTasks ? 'Loading...' : 'My Tasks'}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {showTasks && tasks.length > 0 && (
          <div className="mb-4 p-4 bg-white rounded-lg shadow">
            <h2 className="text-lg font-semibold mb-2">Your Tasks</h2>
            <ul className="space-y-2">
              {tasks.map((task) => (
                <div 
                  key={task.id} 
                  className="group flex items-center p-2 hover:bg-gray-50 rounded cursor-pointer"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        checked={task.completed}
                        onClick={() => !task.completed && handleCompleteTask(task.id)}
                        className="h-4 w-4 text-blue-600 rounded"
                      />
                      <span className={`truncate ${task.completed ? 'line-through text-gray-500' : 'text-gray-800'}`}>
                        {task.title} -
                      </span>
                      {task.description ? (
                        <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-yellow-100 text-yellow-800 whitespace-nowrap">
                          &nbsp;{task.description}
                        </span>
                      ) : " null description"}
                      {task.completed ? (
                          <svg width="1.5rem" height="1.5rem" className="w-1.5 h-1.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                      ): null
                      }
                      <button
                        onClick={(e) => handleDeleteTask(task.id, e)}
                        disabled={deletingTaskId === task.id}
                        className="ml-2 p-1 text-gray-400 hover:text-red-500 rounded-full hover:bg-red-50 disabled:opacity-50"
                        title="Delete task"
                      >
                        {deletingTaskId === task.id ? (
                          <svg width="1.5rem" height="1.5rem" className="w-1.5 h-1.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                        ) : (
                          <svg width="1.5rem" height="1.5rem" className="w-1.5 h-1.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        )}
                        <span className="text-sm font-medium">Delete</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </ul>
          </div>
        )}
        {tasks.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-sm mt-1">Ask me anything about your exams</p>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-sm mt-1">Above are your tasks - {new Date().toLocaleTimeString()}</p>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center">
          <div>
            <p className="text-sm text-gray-500">How can I help you today?</p>
          </div>
        </div>
      </div>
      {/* Input */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <form onSubmit={handleSendMessage} className="flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={!inputValue.trim()}
              className={`absolute right-1 top-1/2 -translate-y-1/2 px-3 py-2 rounded-lg transition-colors flex items-center gap-1 ${inputValue.trim() ? 'text-white bg-blue-500 hover:bg-blue-600' : 'text-gray-400 bg-gray-100'}`}
              aria-label="Send message"
            >
              <span className="text-sm font-medium">Send</span>
            </button>
          </div>
        </form>
        {/* Assistant's Response */}
        {question && (
          <div className="mt-2 p-3 bg-gray-50 rounded-lg">
            <p className="text-gray-800">Question: {question}</p>
          </div>
        )}
        {assistantMessages && (
          <table className="max-w-xl">
            <tbody className="bg-white">
            <tr>
            <td key={assistantMessages.id} className="p-4 text-center">
            Answer: {assistantMessages.text} - {assistantMessages.timestamp.toLocaleTimeString()}
            </td>
            </tr>
            </tbody>
          </table>
        )}          
      </div>
    </div>
  );
}

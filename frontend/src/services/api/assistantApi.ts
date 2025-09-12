import { debug } from 'console';
import { apiRequest } from './client';

export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  assistant_type: 'student' | 'parent';
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  timestamp: string;
}

export interface Task {
  id: string;
  title: string;
  description?: string;
  status: string;
  created_at: string;
  updated_at?: string;
  completed?: boolean;
  deadline?: string;
  priority?: string;
  tags?: string[];
}

export const fetchTasks = async (assistantType: 'student' | 'parent'): Promise<Task[]> => {
  const endpoint = `/${assistantType}/tasks`;
  
  const response = await apiRequest<{ tasks: Task[] }>(endpoint, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    },
  });

  if (response.error) {
    throw new Error(response.error);
  }

  return response.data?.tasks || [];
};

export const completeTask = async (assistantType: 'student' | 'parent', taskId: string): Promise<Task> => {
  const endpoint = `/${assistantType}/tasks/${taskId}/complete`;
  
  const response = await apiRequest<{ task: Task }>(endpoint, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    },
  });

  if (response.error) {
    throw new Error(response.error);
  }

  return response.data?.task;
};

export const deleteTask = async (assistantType: 'student' | 'parent', taskId: string): Promise<{ success: boolean; message: string }> => {
  const endpoint = `/${assistantType}/tasks/${taskId}/delete`;
  
  const response = await apiRequest<{ success: boolean; message: string }>(endpoint, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    },
  });

  console.log('deleteTask response', response)
  if (response.error) {
    throw new Error(response.error);
  }

  return response.data || { success: false, message: 'Unknown error' };
};

export const chatWithAssistant = async (data: ChatRequest): Promise<ChatResponse> => {
  const endpoint = data.assistant_type === 'student' ? '/student/tasks/' : '/parent/tasks/';
  
  console.log('access_token', localStorage.getItem('access_token'))
  const response = await apiRequest<any>(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: data.message,
      user_id: 'test_user' // You might want to get this from auth context
    }),
  });

  if (response.error) {
    throw new Error(response.error);
  }

  console.log(data.conversation_id || `conv_${Date.now()}`)
  console.log(response.data)
  // Map the response to match the expected ChatResponse interface
  return {
    response: response.data.assistant_response,
    conversation_id: data.conversation_id || `conv_${Date.now()}`,
    timestamp: new Date().toISOString()
  };
};

export const getChatHistory = async (conversationId: string): Promise<ChatMessage[]> => {
  const response = await apiRequest<ChatMessage[]>(`/api/chat/history/${conversationId}`, {
    method: 'GET',
  });

  if (response.error) {
    throw new Error(response.error);
  }

  // Convert string timestamps to Date objects
  return response.data!.map(msg => ({
    ...msg,
    timestamp: new Date(msg.timestamp)
  }));
};

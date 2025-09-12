'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import ParentAssistant from './components/ParentAssistant';
import UserProfileTable from './components/UserProfileTable';
import StudentAssistant from './components/StudentAssistant';
import { useSession, signOut } from 'next-auth/react';
import { useAuth } from './contexts/AuthContext';

type AssistantType = 'student' | 'parent' | null;

interface Assistant {
  id: string;
  name: string;
  emoji: string;
  color: string;
}

export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [activeAssistant, setActiveAssistant] = useState<AssistantType>(null);
  const { 
    isLoggedIn, 
    userInfo, 
    isLoading, 
    error, 
    setError, 
    setIsLoggedIn, 
    setUserInfo,
    setIsLoading
  } = useAuth();
  
  // Debug userInfo changes
  useEffect(() => {
    console.log('userInfo in page component:', userInfo);
    if (userInfo?.picture) {
      console.log('Picture URL:', userInfo.picture);
    }
  }, [userInfo]);
  const { data: session, status } = useSession();

  const assistants: Assistant[] = [
    { id: 'parent', name: 'Parent Assistant', emoji: '🏠', color: 'bg-green-100 hover:bg-green-200' },
    { id: 'student', name: 'Student Assistant', emoji: '🎓', color: 'bg-blue-100 hover:bg-blue-200' },
  ];

  // Handle OAuth callback parameters
  useEffect(() => {
    const handleAuthCallback = async () => {
      // Check for existing token first
      const existingToken = localStorage.getItem('access_token');
      const existingUserInfo = localStorage.getItem('user_info');
      console.log(existingUserInfo)
      if (existingToken) {
        setIsLoggedIn(true);
        if (existingUserInfo) {
          try {
            setUserInfo(JSON.parse(existingUserInfo));
          } catch (e) {
            console.warn('Failed to parse stored user info:', e);
          }
        }
        setIsLoading(false);
        return;
      }

      const urlParams = new URLSearchParams(window.location.search);
      const accessToken = urlParams.get('access_token');
      const tokenType = urlParams.get('token_type');
      const authError = urlParams.get('error');

      if (authError) {
        setError(`Authentication failed: ${decodeURIComponent(authError)}`);
        // Clear error params from URL
        window.history.replaceState({}, document.title, window.location.pathname);
        setIsLoading(false);
        return;
      }

      if (accessToken) {
        // Store the tokens
        console.log('Access token:', accessToken);
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('token_type', tokenType || 'bearer');
        
        // Try to get user info from token (decode JWT) or make API call
        try {
          // Decode JWT to get user info (basic approach)
          const tokenParts = accessToken.split('.');
          if (tokenParts.length === 3) {
            const payload = JSON.parse(atob(tokenParts[1]));
            console.log(payload);
            if (payload.email) {
              // Store basic user info from token
              const basicUserInfo = { 
                name: payload.name, 
                email: payload.email, 
                picture: payload.picture 
              };
              localStorage.setItem('user_info', JSON.stringify(basicUserInfo));
              
              // Save user info to file
              try {
                const response = await fetch('/api/users', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    name: basicUserInfo.name,
                    email: basicUserInfo.email,
                    picture: basicUserInfo.picture
                  }),
                });
                
                if (!response.ok) {
                  const error = await response.json();
                  console.error('Failed to save user info:', error);
                }
              } catch (error) {
                console.error('Error saving user info:', error);
                // Non-blocking error - we still want to log the user in
              }
              
              setUserInfo(basicUserInfo);
              setIsLoggedIn(true);
              setError(''); // Clear any previous errors
              console.log('Login successful!');
              
              // Clear OAuth params from URL
              window.history.replaceState({}, document.title, window.location.pathname);
            }
          }
        } catch (e) {
          console.warn('Failed to decode token:', e);
          setError('Failed to process authentication. Please try again.');
        }
        
        // Clear the URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);
      }
      setIsLoading(false);
    };

    handleAuthCallback();
  }, [setError, setIsLoggedIn, setUserInfo]);

  const handleBack = () => {
    setActiveAssistant(null);
  };

  const handleGoogleLogin = async () => {
    try {
      setError(''); // Clear any previous errors
      // Redirect to your FastAPI Google OAuth endpoint
      window.location.href = 'http://localhost:8000/auth/google';
    } catch (error) {
      console.error('Google login error:', error);
      setError('Failed to initiate Google login. Please try again.');
    }
  };

  const handleLogout = () => {
    // Clear tokens from localStorage
    localStorage.removeItem('access_token');
    localStorage.removeItem('token_type');
    localStorage.removeItem('user_info');
    
    setIsLoggedIn(false);
    setActiveAssistant(null);
    setUserInfo(null);
    setError('');
  };

  if (isLoading) {
    return (
      <div className="flex flex-1 flex-col justify-center items-center min-h-screen w-full h-full bg-white">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
        <p className="mt-4 text-gray-600">Loading...</p>
      </div>
    );
  }

  if (!isLoggedIn) {
    return (
      <div className="flex flex-1 flex-col justify-center items-center min-h-screen w-full h-full bg-white p-4">
        <h1 className="text-4xl font-bold mb-8 text-center text-gray-800">
          Welcome to Your Education Assistant
        </h1>
        <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
          <h2 className="text-2xl font-semibold mb-6 text-center text-gray-800">Sign in to continue</h2>
          
          {error && (
            <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}
          
          <div className="mb-8">
            <button
              onClick={handleGoogleLogin}
              className="w-full flex items-center justify-center gap-3 bg-white border border-gray-300 rounded-lg shadow-sm px-6 py-3 text-sm font-medium text-gray-800 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              <span>Continue with Google</span>
            </button>
          </div>
          
          <div className="flex items-center justify-center mb-6">
            <span className="px-4 text-sm text-gray-500">or</span>
          </div>
          
          <div className="w-full">
            <table className="w-full border-collapse">
              <tbody>
                <tr>
                  <td className="w-1/2 pr-2">
                    <button
                      type="button"
                      onClick={() => router.push('/auth/signin')}
                      className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      Sign In
                    </button>
                  </td>
                  <td className="w-1/2 pl-2">
                    <button
                      type="button"
                      onClick={() => router.push('/auth/register')}
                      className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                    >
                      Register
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col items-center min-h-screen max-w-1/2 h-full bg-white">
      <div className="bg-white shadow-sm mx-auto px-4 sm:px-6 lg:px-8 flex justify-center h-16 items-center">
        <h1 className="text-xl font-bold text-gray-800">AI Assistant</h1>
        
        {/* User Info Table */}
        {userInfo && (
          <div className="bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            <UserProfileTable userInfo={userInfo} onLogout={handleLogout} />
          </div>
        )}
      </div>
      
      <div className="bg-white shadow-sm mx-auto px-4 sm:px-6 lg:px-8 flex justify-center h-16 items-center">
        <h1 className="text-4xl font-bold mb-16 text-center text-gray-800">
          Welcome to Your Education Assistant
        </h1>
        {!activeAssistant ? (
          <div className="w-full max-w-6xl mx-auto">
            <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
              <table className="min-w-full">
                <tbody className="bg-white">
                  <tr>
                    {assistants.map((assistant) => (
                      <td key={assistant.id} className="p-4 text-center">
                        <button
                          onClick={() => setActiveAssistant(assistant.id as AssistantType)}
                          className={`p-8 rounded-2xl shadow-md transition-all duration-200 flex flex-col items-center justify-center h-64 w-72 ${assistant.color} border-2 border-transparent hover:border-blue-300 transform hover:scale-105`}
                          aria-label={`Open ${assistant.name} assistant`}
                        >
                          <span className="text-6xl mb-6" role="img" aria-hidden="true">
                            {assistant.emoji}
                          </span>
                          <h2 className="text-2xl font-semibold text-gray-800">
                            {assistant.name}
                          </h2>
                          <p className="mt-2 text-gray-600">Click to start chatting</p>
                        </button>
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        ) : activeAssistant === 'parent' ? (
          <ParentAssistant onBack={handleBack} />
        ) : (
          <StudentAssistant onBack={handleBack} />
        )}
      </div>
    </div>
  );
}
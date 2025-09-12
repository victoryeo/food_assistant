'use client';

import { createContext, useContext, ReactNode, useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';

interface AuthContextType {
  isLoggedIn: boolean;
  userInfo: any;
  isLoading: boolean;
  error: string;
  setIsLoggedIn: (value: boolean) => void;
  setUserInfo: (userInfo: any) => void;
  setError: (error: string) => void;
  setIsLoading: (isLoading: boolean) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userInfo, setUserInfo] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const { data: session, status } = useSession();

  // Update localStorage whenever userInfo changes
  useEffect(() => {
    if (userInfo) {
      console.log("userInfo updated",userInfo)
      localStorage.setItem('user_info', JSON.stringify(userInfo));
    }
  }, [userInfo]);

  useEffect(() => {
    const checkAuth = () => {
      const existingToken = localStorage.getItem('access_token');
      const existingUserInfo = localStorage.getItem('user_info');
      
      if (existingToken) {
        setIsLoggedIn(true);
        if (existingUserInfo) {
          try {
            const parsedUserInfo = JSON.parse(existingUserInfo);
            // Ensure picture URL is properly formatted
            if (parsedUserInfo.picture && !parsedUserInfo.picture.startsWith('http')) {
              parsedUserInfo.picture = `/${parsedUserInfo.picture}`.replace(/\/+/g, '/');
            }
            setUserInfo(parsedUserInfo);
          } catch (e) {
            console.warn('Failed to parse stored user info:', e);
          }
        }
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  return (
    <AuthContext.Provider 
      value={{ 
        isLoggedIn, 
        userInfo, 
        isLoading, 
        error, 
        setIsLoggedIn, 
        setUserInfo, 
        setError,
        setIsLoading 
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

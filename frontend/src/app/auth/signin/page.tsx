// File: app/auth/signin/page.js
'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { handleSignIn } from '../../actions/auth'
import { useAuth } from '../../contexts/AuthContext';

export default function SignIn() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const { 
    isLoggedIn, 
    userInfo, 
    isLoading, 
    setIsLoggedIn, 
    setUserInfo,
  } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    const formData = new FormData(e.target)
    const result = await handleSignIn(formData)
    console.log(result)
    if (result?.error) {
      setError(result.error)
      setLoading(false)
      return
    }
    
    const response = await fetch('/api/users', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    const data = await response.json()
    console.log(data)
    
    const user = data.find((u: any) => u.email === email)
    console.log(user)

    if (user) {
      // Ensure picture URL is properly formatted
      const userInfo = {
        name: user.name,
        email: user.email,
        picture: user.picture
      };
      
      // If picture is a relative path, ensure it starts with a forward slash
      if (userInfo.picture && !userInfo.picture.startsWith('http') && !userInfo.picture.startsWith('/')) {
        userInfo.picture = `/${userInfo.picture}`.replace(/\/+/g, '/');
      }
      
      setUserInfo(userInfo);
      // Also store in localStorage directly to ensure it's available on page reload
      localStorage.setItem('user_info', JSON.stringify(userInfo));
    }
    // Create a URLSearchParams object to format the data as form-urlencoded
    const reqBody = new URLSearchParams({
      email: user.email,
      username: user.name,
      password: password,
    });
    // call backend to get token
    
    const response2 = await fetch('http://localhost:8000/token/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: reqBody
    })
    console.log(response2)
    const result2 = await response2.json()
    console.log(result2)
    if (response2.ok) {
      setIsLoggedIn(true)
      setLoading(false)

      if (result2?.access_token) {
        // Store the tokens
        localStorage.setItem('access_token', result2.access_token);
        localStorage.setItem('token_type', result2.token_type || 'bearer');
      }
      router.push('/')
    } else {
      setError(result2.message || 'Token Result not ok')
      setLoading(false)
    }

  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Sign in to your account
          </h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <div className="text-red-600 text-sm text-center">{error}</div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={() => router.push('/auth/register')}
              className="text-indigo-600 hover:text-indigo-500 underline bg-transparent border-none cursor-pointer"
            >
              Don't have an account? Register
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
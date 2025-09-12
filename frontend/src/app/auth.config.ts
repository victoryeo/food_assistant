// server side
import Credentials from "next-auth/providers/credentials"
import type { Provider } from "next-auth/providers"
import type { NextAuthConfig } from 'next-auth';

const providers: Provider[] = [
    Credentials({
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' }
      },
      authorize: async (credentials) => {
        console.log("authorize")

        if (!credentials?.email || !credentials?.password) {
          return null
        }
        console.log(credentials?.email, credentials?.password)
        // Type assertion to ensure credentials are strings
        const email = credentials.email as string
        const password = credentials.password as string

        const baseUrl = process.env.NEXTAUTH_URL || 'http://localhost:3001';
        const response = await fetch(`${baseUrl}/api/auth/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        });
        console.log(response)
        if (response.ok) {
          const user = await response.json();
          console.log('ok', user)
          return user;
        } else {
          console.error("user not found")
          return null
        }
      }
    })
  ]

export const providerMap = providers.map((provider) => {
	if (typeof provider === "function") {
		const providerData = provider()
		console.log("1", providerData.id, providerData.name)
		return { id: providerData.id, name: providerData.name }
	} else {
		console.log("2", provider.id, provider.name)
		return { id: provider.id, name: provider.name }
	}
})

export const authConfig = {
  pages: {
    signIn: '/signin',
  },
  callbacks: {
    authorized({ auth, request: { nextUrl } }) {
      return true;
    },   
  },
  providers: providers,
} satisfies NextAuthConfig;
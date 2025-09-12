'use server'

import { signIn } from "../auth"
import { redirect } from 'next/navigation'

export async function handleSignIn(formData) {
  try {
    const email = formData.get('email')
    const password = formData.get('password')
    console.log("action1", email, password)
    const res = await signIn("credentials", {
      email,
      password,
      redirect: false,
    })
    console.log("action2", res)
    return { success: true }
  } catch (error) {
    return { error: "Invalid email or password" }
  }
  
  //redirect('/')
}
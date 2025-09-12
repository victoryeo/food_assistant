// mock user database
export const users = []

// Simple hash function for client-side (not cryptographically secure)
// Only use this for demo purposes - use proper hashing in production
export async function simpleHash(password) {
  // This is a very basic hash - replace with proper server-side hashing
  let hash = 0
  for (let i = 0; i < password.length; i++) {
    const char = password.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32-bit integer
  }
  return Math.abs(hash).toString()
}

export async function simpleCompare(password, hash) {
  const passwordHash = await simpleHash(password)
  return passwordHash === hash
}

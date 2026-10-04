import { useCallback, useState } from 'react'

// Persisted to this browser's localStorage purely for the user's own convenience (so they don't
// retype their key every reload) — it is never sent anywhere except as the X-LLM-Key/X-LLM-Model
// request headers on each API call (api/client.ts). The backend itself never stores it (PRD
// Section 8): this is client-side-only persistence in the user's own browser.
const KEY_STORAGE = 'pdf-kb-llm-key'
const MODEL_STORAGE = 'pdf-kb-llm-model'

export interface Settings {
  llmKey: string
  setLlmKey: (value: string) => void
  llmModel: string
  setLlmModel: (value: string) => void
  hasKey: boolean
}

export function useSettings(): Settings {
  const [llmKey, setLlmKeyState] = useState(() => localStorage.getItem(KEY_STORAGE) ?? '')
  const [llmModel, setLlmModelState] = useState(() => localStorage.getItem(MODEL_STORAGE) ?? '')

  const setLlmKey = useCallback((value: string) => {
    localStorage.setItem(KEY_STORAGE, value)
    setLlmKeyState(value)
  }, [])

  const setLlmModel = useCallback((value: string) => {
    localStorage.setItem(MODEL_STORAGE, value)
    setLlmModelState(value)
  }, [])

  return { llmKey, setLlmKey, llmModel, setLlmModel, hasKey: llmKey.trim().length > 0 }
}

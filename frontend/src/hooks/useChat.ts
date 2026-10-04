import { useCallback, useState } from 'react'
import { queryStream } from '../api/client'
import type { Citation, DebugInfo } from '../api/types'

// The chat "transcript" here is entirely client-side and ephemeral (cleared on reload) — there is
// no server-side conversation memory to persist it to. PRD Section 0 explicitly excludes a real
// memory store; every turn below is still an independent single-turn call, exactly as the backend
// expects (see app/extensibility/memory_store.py's docstring for why that's deliberate, not a
// gap).
export interface ChatTurn {
  id: string
  question: string
  scopedFilenames: string[]
  isPending: boolean
  // Raw accumulated text while streaming, rendered as plain text (see MessageBubble.tsx) — a
  // citation marker's chunk_id can only be trusted once fully accumulated (a bracket like
  // "[doc-id:1:abc" with no closing "]" yet isn't a match), so nothing here gets citation-chip
  // treatment until `answer`/`citations` are set from the stream's final "done" event.
  streamingText: string
  answer?: string
  citations?: Citation[]
  routedDocs?: string[] | null
  partialAnswer?: boolean
  debug?: DebugInfo | null
  error?: string
}

export function useChat(llmKey: string, llmModel: string) {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [isAsking, setIsAsking] = useState(false)

  const updateTurn = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)))
  }, [])

  const ask = useCallback(
    (question: string, fileIds: string[] | undefined, scopedFilenames: string[]) => {
      const id = crypto.randomUUID()
      setTurns((prev) => [...prev, { id, question, scopedFilenames, isPending: true, streamingText: '' }])
      setIsAsking(true)

      queryStream(
        { question, file_ids: fileIds },
        llmKey,
        llmModel || undefined,
        {
          onToken: (text) => {
            setTurns((prev) =>
              prev.map((turn) => (turn.id === id ? { ...turn, streamingText: turn.streamingText + text } : turn)),
            )
          },
          onDone: (result) => {
            updateTurn(id, {
              isPending: false,
              answer: result.answer,
              citations: result.citations,
              routedDocs: result.routed_docs,
              partialAnswer: result.partial_answer,
              debug: result.debug,
            })
            setIsAsking(false)
          },
          onError: (message) => {
            updateTurn(id, { isPending: false, error: message })
            setIsAsking(false)
          },
        },
      ).catch((err: unknown) => {
        updateTurn(id, { isPending: false, error: err instanceof Error ? err.message : 'Something went wrong asking that question.' })
        setIsAsking(false)
      })
    },
    [llmKey, llmModel, updateTurn],
  )

  const clear = useCallback(() => setTurns([]), [])

  return { turns, ask, clear, isAsking }
}

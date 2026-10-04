import { useEffect, useRef } from 'react'
import type { DebugInfo } from '../api/types'
import type { ChatTurn } from '../hooks/useChat'
import { MessageBubble } from './MessageBubble'

interface ChatPanelProps {
  turns: ChatTurn[]
  hasDocuments: boolean
  onShowDebug: (debug: DebugInfo) => void
}

export function ChatPanel({ turns, hasDocuments, onShowDebug }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns.length, turns[turns.length - 1]?.isPending])

  if (turns.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="max-w-sm text-center">
          <h2 className="mb-1.5 font-serif text-xl text-fg">
            {hasDocuments ? 'Ask something about your documents' : 'Upload a PDF to get started'}
          </h2>
          <p className="text-[13px] text-fg-muted">
            {hasDocuments
              ? 'Questions search across your whole knowledge base unless you scope them to specific files in the sidebar.'
              : 'Once a document finishes processing, you can ask questions about it here.'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-8">
      <div className="mx-auto max-w-2xl">
        {turns.map((turn) => (
          <MessageBubble key={turn.id} turn={turn} onShowDebug={() => turn.debug && onShowDebug(turn.debug)} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

import { Send, X } from 'lucide-react'
import { useState, type FormEvent, type KeyboardEvent } from 'react'
import type { DocumentListItem } from '../api/types'

interface ComposerProps {
  disabled: boolean
  disabledReason?: string
  isAsking: boolean
  scopedDocs: DocumentListItem[]
  onRemoveScope: (docId: string) => void
  onSend: (question: string) => void
}

export function Composer({ disabled, disabledReason, isAsking, scopedDocs, onRemoveScope, onSend }: ComposerProps) {
  const [value, setValue] = useState('')

  function submit() {
    const question = value.trim()
    if (!question || disabled || isAsking) return
    onSend(question)
    setValue('')
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-border bg-bg px-6 py-5">
      <form onSubmit={handleSubmit} className="mx-auto max-w-2xl">
        {scopedDocs.length > 0 && (
          <div className="mb-2.5 flex flex-wrap gap-1.5">
            {scopedDocs.map((doc) => (
              <span
                key={doc.doc_id}
                className="flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-[12px] text-fg"
              >
                {doc.filename}
                <button
                  type="button"
                  onClick={() => onRemoveScope(doc.doc_id)}
                  aria-label={`Remove ${doc.filename} from scope`}
                  className="text-fg-muted hover:text-fg"
                >
                  <X size={11} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 rounded-[22px] border border-border bg-surface px-4 py-2.5 shadow-sm transition-colors focus-within:border-accent">
          <textarea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={disabled ? disabledReason : 'Ask something about your documents…'}
            disabled={disabled}
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-[14px] text-fg outline-none placeholder:text-fg-muted disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={disabled || isAsking || !value.trim()}
            aria-label="Send"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-fg transition-colors hover:bg-accent-hover disabled:bg-border disabled:text-fg-muted"
          >
            <Send size={15} />
          </button>
        </div>
      </form>
    </div>
  )
}

import { KeyRound, X } from 'lucide-react'
import type { Settings } from '../hooks/useSettings'

interface SettingsModalProps {
  open: boolean
  onClose: () => void
  settings: Settings
}

export function SettingsModal({ open, onClose, settings }: SettingsModalProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-fg/20 p-4" onClick={onClose}>
      <div
        className="w-full max-w-sm rounded-2xl border border-border bg-surface shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between px-6 pt-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-accent/10 text-accent">
            <KeyRound size={16} />
          </div>
          <button type="button" onClick={onClose} aria-label="Close settings" className="text-fg-muted hover:text-fg">
            <X size={17} />
          </button>
        </div>

        <div className="px-6 pt-3">
          <h2 className="text-[15px] font-semibold text-fg">Your API key</h2>
          <p className="mt-1 text-[13px] text-fg-muted">Bring your own key — used to ask questions and caption figures.</p>
        </div>

        <div className="space-y-4 px-6 py-5">
          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-fg" htmlFor="llm-key">
              API key
            </label>
            <input
              id="llm-key"
              type="password"
              autoComplete="off"
              placeholder="sk-..."
              value={settings.llmKey}
              onChange={(event) => settings.setLlmKey(event.target.value)}
              className="w-full rounded-lg border border-border bg-bg px-3 py-2.5 text-[13px] text-fg outline-none transition-colors focus:border-accent"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-fg" htmlFor="llm-model">
              Model <span className="text-fg-muted">(optional)</span>
            </label>
            <input
              id="llm-model"
              type="text"
              autoComplete="off"
              placeholder="openai/gpt-4o-mini"
              value={settings.llmModel}
              onChange={(event) => settings.setLlmModel(event.target.value)}
              className="w-full rounded-lg border border-border bg-bg px-3 py-2.5 text-[13px] text-fg outline-none transition-colors focus:border-accent"
            />
          </div>
        </div>

        <div className="border-t border-border px-6 py-4">
          <p className="mb-4 text-[12px] leading-relaxed text-fg-muted">
            Stored only in this browser and sent as a request header on each call — never saved on the server.
          </p>
          <button
            type="button"
            onClick={onClose}
            className="w-full rounded-lg bg-accent py-2.5 text-[13px] font-medium text-accent-fg transition-colors hover:bg-accent-hover"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}

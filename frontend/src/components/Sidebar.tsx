import { Check, Plus, Settings, Trash2 } from 'lucide-react'
import type { DocumentListItem } from '../api/types'
import type { PendingUpload } from '../hooks/useDocuments'
import type { ThemePreference } from '../hooks/useTheme'
import { formatDate, statusDotClasses, statusLabel } from '../lib/format'
import { ThemeToggle } from './ThemeToggle'
import { UploadButton } from './UploadButton'

interface SidebarProps {
  documents: DocumentListItem[]
  pendingUploads: PendingUpload[]
  uploadError: unknown
  hasKey: boolean
  onUpload: (file: File) => void
  onDelete: (docId: string) => void
  deletingDocId?: string
  scopedDocIds: string[]
  onToggleScope: (docId: string) => void
  onNewChat: () => void
  onOpenSettings: () => void
  themePreference: ThemePreference
  onThemeChange: (next: ThemePreference) => void
}

function ScopeCheckbox({ checked, disabled, label, onChange }: { checked: boolean; disabled: boolean; label: string; onChange: () => void }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onChange}
      className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${
        checked ? 'border-accent bg-accent text-accent-fg' : 'border-border bg-surface'
      } disabled:opacity-30`}
    >
      {checked && <Check size={11} strokeWidth={3} />}
    </button>
  )
}

export function Sidebar({
  documents,
  pendingUploads,
  uploadError,
  hasKey,
  onUpload,
  onDelete,
  deletingDocId,
  scopedDocIds,
  onToggleScope,
  onNewChat,
  onOpenSettings,
  themePreference,
  onThemeChange,
}: SidebarProps) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center justify-between px-4 py-4">
        <div className="flex items-center gap-1.5">
          <span className="font-serif text-xl italic text-accent">”</span>
          <span className="font-serif text-lg text-fg">Cite</span>
        </div>
        <button
          type="button"
          onClick={onNewChat}
          title="New chat"
          aria-label="New chat"
          className="flex h-7 w-7 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg"
        >
          <Plus size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pt-1">
        <div className="mb-3">
          <UploadButton disabled={!hasKey} disabledReason="Add your X-LLM-Key in Settings first" onSelect={onUpload} />
          {uploadError instanceof Error && <p className="mt-1.5 text-[12px] text-danger">{uploadError.message}</p>}
        </div>

        <ul className="space-y-0.5">
          {pendingUploads.map((pending) => (
            <li key={pending.jobId} className="rounded-md px-2.5 py-2">
              <p className="truncate text-[13px] text-fg">{pending.filename}</p>
              <p className="text-[11px] text-fg-muted">
                {pending.job ? `${pending.job.stage} · ${pending.job.progress_pct}%` : 'starting…'}
              </p>
            </li>
          ))}

          {documents.length === 0 && pendingUploads.length === 0 && (
            <li className="px-2.5 py-6 text-center text-[13px] text-fg-muted">No documents yet.</li>
          )}

          {documents.map((doc) => {
            const isScopeable = doc.status === 'ready'
            return (
              <li key={doc.doc_id} className="group flex items-start gap-2.5 rounded-md px-2.5 py-2 hover:bg-surface-hover">
                <ScopeCheckbox
                  checked={scopedDocIds.includes(doc.doc_id)}
                  disabled={!isScopeable}
                  label={`Scope questions to ${doc.filename}`}
                  onChange={() => onToggleScope(doc.doc_id)}
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] text-fg" title={doc.filename}>
                    {doc.filename}
                  </p>
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-fg-muted">
                    <span className="flex items-center gap-1">
                      <span className={`inline-block h-1.5 w-1.5 rounded-full ${statusDotClasses(doc.status)}`} />
                      {statusLabel(doc.status)}
                    </span>
                    <span>·</span>
                    <span>{doc.page_count}p</span>
                    <span>·</span>
                    <span>{formatDate(doc.uploaded_at)}</span>
                  </div>
                </div>
                <button
                  type="button"
                  aria-label={`Delete ${doc.filename}`}
                  disabled={deletingDocId === doc.doc_id}
                  onClick={() => onDelete(doc.doc_id)}
                  className="mt-0.5 text-fg-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100 disabled:opacity-40"
                >
                  <Trash2 size={13} />
                </button>
              </li>
            )
          })}
        </ul>
      </div>

      <div className="flex items-center justify-between border-t border-border px-3 py-3">
        <button
          type="button"
          onClick={onOpenSettings}
          className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px] text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg"
        >
          <Settings size={14} />
          Settings
        </button>
        <ThemeToggle preference={themePreference} onChange={onThemeChange} />
      </div>
    </aside>
  )
}

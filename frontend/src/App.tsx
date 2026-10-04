import { useMemo, useState } from 'react'
import type { DebugInfo } from './api/types'
import { ChatPanel } from './components/ChatPanel'
import { Composer } from './components/Composer'
import { DebugSidePanel } from './components/DebugPanel'
import { SettingsModal } from './components/SettingsModal'
import { Sidebar } from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useDocuments } from './hooks/useDocuments'
import { useSettings } from './hooks/useSettings'
import { useTheme } from './hooks/useTheme'

export function App() {
  const settings = useSettings()
  const { preference, setPreference } = useTheme()
  const { documents, pendingUploads, uploadError, upload, deleteDoc, deletingDocId } = useDocuments(
    settings.llmKey,
    settings.llmModel,
  )
  const { turns, ask, clear: clearChat, isAsking } = useChat(settings.llmKey, settings.llmModel)

  const [scopedDocIds, setScopedDocIds] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [debugPanel, setDebugPanel] = useState<DebugInfo | null>(null)

  const readyDocuments = useMemo(() => documents.filter((doc) => doc.status === 'ready'), [documents])
  const scopedDocs = useMemo(
    () => readyDocuments.filter((doc) => scopedDocIds.includes(doc.doc_id)),
    [readyDocuments, scopedDocIds],
  )

  function toggleScope(docId: string) {
    setScopedDocIds((prev) => (prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]))
  }

  function removeScope(docId: string) {
    setScopedDocIds((prev) => prev.filter((id) => id !== docId))
  }

  function handleNewChat() {
    clearChat()
    setScopedDocIds([])
    setDebugPanel(null)
  }

  function handleSend(question: string) {
    const fileIds = scopedDocIds.length > 0 ? scopedDocIds : undefined
    ask(question, fileIds, scopedDocs.map((doc) => doc.filename))
  }

  const composerDisabled = !settings.hasKey || readyDocuments.length === 0
  const composerDisabledReason = !settings.hasKey
    ? 'Add your X-LLM-Key in Settings first'
    : 'Upload a PDF to get started'

  return (
    <div className="flex h-screen w-full overflow-hidden bg-bg">
      <Sidebar
        documents={documents}
        pendingUploads={pendingUploads}
        uploadError={uploadError}
        hasKey={settings.hasKey}
        onUpload={upload}
        onDelete={deleteDoc}
        deletingDocId={deletingDocId}
        scopedDocIds={scopedDocIds}
        onToggleScope={toggleScope}
        onNewChat={handleNewChat}
        onOpenSettings={() => setSettingsOpen(true)}
        themePreference={preference}
        onThemeChange={setPreference}
      />

      <div className="flex flex-1 flex-col">
        <ChatPanel turns={turns} hasDocuments={readyDocuments.length > 0} onShowDebug={setDebugPanel} />

        <Composer
          disabled={composerDisabled}
          disabledReason={composerDisabledReason}
          isAsking={isAsking}
          scopedDocs={scopedDocs}
          onRemoveScope={removeScope}
          onSend={handleSend}
        />
      </div>

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} settings={settings} />
      <DebugSidePanel debug={debugPanel} onClose={() => setDebugPanel(null)} />
    </div>
  )
}

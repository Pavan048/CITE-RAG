import { X } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ChunkTrace, DebugInfo } from '../api/types'
import { shortId } from '../lib/format'

function ChunkTraceRow({ chunk }: { chunk: ChunkTrace }) {
  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-1.5 pr-3 font-mono text-[11px] text-fg-muted">{shortId(chunk.chunk_id)}</td>
      <td className="py-1.5 pr-3 font-mono text-[11px] text-fg-muted">{shortId(chunk.doc_id)}</td>
      <td className="py-1.5 pr-3 text-fg-muted">{chunk.page_number}</td>
      <td className="py-1.5 pr-3 text-fg-muted">{chunk.content_type}</td>
      <td className="py-1.5 pr-3 tabular-nums text-fg-muted">{chunk.score.toFixed(3)}</td>
      <td className="py-1.5 text-fg-muted" title={chunk.text_preview}>
        {chunk.text_preview.slice(0, 50)}
        {chunk.text_preview.length > 50 ? '…' : ''}
      </td>
    </tr>
  )
}

function ChunkTable({ chunks }: { chunks: ChunkTrace[] }) {
  if (chunks.length === 0) return <p className="text-[13px] text-fg-muted">None.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[12px]">
        <thead>
          <tr className="text-fg-muted">
            <th className="pb-1 pr-3 font-medium">Chunk</th>
            <th className="pb-1 pr-3 font-medium">Doc</th>
            <th className="pb-1 pr-3 font-medium">Page</th>
            <th className="pb-1 pr-3 font-medium">Type</th>
            <th className="pb-1 pr-3 font-medium">Score</th>
            <th className="pb-1 font-medium">Preview</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((chunk) => (
            <ChunkTraceRow key={chunk.chunk_id} chunk={chunk} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-b border-border px-5 py-4 last:border-0">
      <h4 className="mb-2.5 text-[13px] font-semibold text-fg">{title}</h4>
      {children}
    </div>
  )
}

interface DebugSidePanelProps {
  debug: DebugInfo | null
  onClose: () => void
}

export function DebugSidePanel({ debug, onClose }: DebugSidePanelProps) {
  if (!debug) return null

  return (
    // No dimming backdrop and no click-outside-to-close — this is a docked inspector meant to sit
    // alongside the conversation while you keep reading it, not a modal that blocks it.
    <aside
      className="fixed inset-y-0 right-0 z-30 flex w-full max-w-md flex-col border-l border-border bg-surface font-sans shadow-xl"
      style={{ animation: 'panel-in 0.18s ease-out' }}
    >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h3 className="text-sm font-semibold text-fg">Debug trace</h3>
          <button type="button" onClick={onClose} aria-label="Close debug trace" className="text-fg-muted hover:text-fg">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <Section title="Graph path">
            <div className="flex flex-wrap items-center gap-1.5">
              {debug.graph_path.map((node, index) => (
                <span key={index} className="flex items-center gap-1.5">
                  <span className="rounded-md bg-surface-hover px-2 py-1 font-mono text-[11px] text-fg">{node}</span>
                  {index < debug.graph_path.length - 1 && <span className="text-fg-muted">→</span>}
                </span>
              ))}
            </div>
          </Section>

          <Section title="Summary">
            <dl className="grid grid-cols-2 gap-y-2 text-[13px]">
              <div>
                <dt className="text-fg-muted">Multi-hop</dt>
                <dd className="text-fg">{debug.is_multi_hop ? 'Yes' : 'No'}</dd>
              </div>
              <div>
                <dt className="text-fg-muted">Hop count</dt>
                <dd className="text-fg">{debug.hop_count}</dd>
              </div>
              <div>
                <dt className="text-fg-muted">Not found</dt>
                <dd className="text-fg">{debug.not_found ? 'Yes' : 'No'}</dd>
              </div>
              <div>
                <dt className="text-fg-muted">Context blocks</dt>
                <dd className="text-fg">{debug.context_block_count}</dd>
              </div>
            </dl>
          </Section>

          {debug.hops.length > 0 && (
            <Section title={`Hops (${debug.hops.length})`}>
              <div className="space-y-4">
                {debug.hops.map((hop) => (
                  <div key={hop.hop_number}>
                    <p className="mb-1.5 text-[13px] font-medium text-fg">
                      Hop {hop.hop_number} <span className="font-normal text-fg-muted">— {hop.sub_question}</span>
                    </p>
                    <ChunkTable chunks={hop.chunks_retrieved} />
                  </div>
                ))}
              </div>
            </Section>
          )}

          <Section title={`Reranked chunks (${debug.reranked_chunks.length})`}>
            <ChunkTable chunks={debug.reranked_chunks} />
          </Section>

          <Section title={`LLM calls (${debug.llm_calls.length})`}>
            <div className="space-y-2">
              {debug.llm_calls.map((call, index) => (
                <details key={index} className="rounded-md border border-border">
                  <summary className="cursor-pointer px-3 py-2 text-[13px] font-medium text-fg">{call.node}</summary>
                  <div className="border-t border-border px-3 py-2.5">
                    <p className="mb-1 text-[11px] font-medium text-fg-muted">Prompt</p>
                    <pre className="mb-3 max-h-48 overflow-y-auto whitespace-pre-wrap text-[11px] text-fg-muted">{call.prompt}</pre>
                    <p className="mb-1 text-[11px] font-medium text-fg-muted">Response</p>
                    <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap text-[11px] text-fg">{call.response}</pre>
                  </div>
                </details>
              ))}
            </div>
          </Section>
        </div>
      </aside>
  )
}

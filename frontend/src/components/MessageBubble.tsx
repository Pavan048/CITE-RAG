import { Bug } from 'lucide-react'
import type { ChatTurn } from '../hooks/useChat'
import { parseAnswerLines, parseLiveSegments } from '../lib/citations'
import { shortId } from '../lib/format'
import { CitationChip } from './CitationChip'
import { PartialAnswerBanner } from './PartialAnswerBanner'

function StreamingText({ text }: { text: string }) {
  const segments = parseLiveSegments(text)
  return (
    <p className="whitespace-pre-wrap">
      {segments.map((segment, index) =>
        segment.type === 'text' ? (
          <span key={index}>{segment.text}</span>
        ) : (
          <span
            key={index}
            aria-label="Resolving citation"
            className="mx-0.5 inline-block h-3 w-3 rounded-full bg-fg-muted/30 align-middle"
            style={{ animation: 'pulse-dot 1.4s infinite ease-in-out' }}
          />
        ),
      )}
    </p>
  )
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-fg-muted"
          style={{ animation: 'pulse-dot 1.4s infinite ease-in-out', animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  )
}

interface MessageBubbleProps {
  turn: ChatTurn
  onShowDebug: () => void
}

export function MessageBubble({ turn, onShowDebug }: MessageBubbleProps) {
  // The grounded_answer prompt only ever asks for plain prose + citation markers, one per
  // sentence — a multi-point answer (a summary, several facts) needs each point on its own line,
  // or it reads as one dense, unscannable block regardless of how good the sentences themselves
  // are. Rendered in the serif face: this is the conversation's actual *content*, set the way
  // something worth reading is set — UI chrome around it stays in the sans face.
  const lines = turn.answer ? parseAnswerLines(turn.answer, turn.citations ?? []) : []

  return (
    <div className="mb-8">
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl bg-user-bubble px-4 py-2.5">
          {turn.scopedFilenames.length > 0 && (
            <p className="mb-1 font-sans text-[11px] font-medium text-fg-muted">
              Scoped to {turn.scopedFilenames.join(', ')}
            </p>
          )}
          <p className="font-serif text-[15px] leading-snug text-fg">{turn.question}</p>
        </div>
      </div>

      <div className="mt-4 flex justify-start">
        <div className="max-w-[80ch] font-serif text-[15px] leading-[1.7] text-fg">
          {turn.isPending && turn.streamingText.length === 0 && <ThinkingIndicator />}

          {turn.isPending && turn.streamingText.length > 0 && <StreamingText text={turn.streamingText} />}

          {turn.error && (
            <p className="rounded-md bg-danger-bg px-3 py-2 font-sans text-[13px] text-danger">{turn.error}</p>
          )}

          {!turn.isPending && turn.partialAnswer && <PartialAnswerBanner />}

          {!turn.isPending &&
            lines.map((segments, lineIndex) => (
              <p key={lineIndex} className="mb-3 last:mb-0">
                {segments.map((segment, segIndex) =>
                  segment.type === 'text' ? (
                    <span key={segIndex}>{segment.text}</span>
                  ) : (
                    <CitationChip key={segIndex} number={segment.number} citation={segment.citation} />
                  ),
                )}
              </p>
            ))}

          {!turn.isPending && (turn.routedDocs?.length || turn.debug) && (
            <div className="mt-3 flex items-center gap-3 font-sans text-[11px] text-fg-muted">
              {turn.routedDocs && turn.routedDocs.length > 0 && (
                <span>Routed to {turn.routedDocs.map(shortId).join(', ')}</span>
              )}
              {turn.debug && (
                <button type="button" onClick={onShowDebug} className="flex items-center gap-1 hover:text-fg">
                  <Bug size={12} />
                  Debug trace
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

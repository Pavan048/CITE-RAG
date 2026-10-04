import { useEffect, useRef, useState } from 'react'
import type { Citation } from '../api/types'
import { shortId } from '../lib/format'

interface CitationChipProps {
  number: number
  citation: Citation | undefined
}

export function CitationChip({ number, citation }: CitationChipProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (!open) return
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  // A citation number with no matching entry means the marker's chunk_id wasn't found in
  // `citations[]` — shouldn't happen (generate_cite_node only ever emits markers it already
  // validated), but render plainly rather than crash if it ever does.
  if (!citation) return <sup className="font-sans text-cite">{number}</sup>

  return (
    <span ref={containerRef} className="relative">
      {/* A plain superscript numeral, not a filled pill — citations happen once per sentence
          throughout an answer, and a badge that heavy repeated that often reads as noise rather
          than a citation mark. The amber ("cite") color is spent nowhere else on the page, so the
          mark alone is enough to say "this is a reference." */}
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="rounded-sm font-sans text-cite underline decoration-cite/40 decoration-1 underline-offset-2 hover:decoration-cite"
      >
        <sup>{number}</sup>
      </button>
      {open && (
        <div className="absolute bottom-full left-1/2 z-20 mb-2 w-72 -translate-x-1/2 rounded-lg border border-border bg-surface p-3.5 text-left font-sans shadow-lg">
          <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-cite">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-cite" />
            {shortId(citation.doc_id)} · page {citation.page_number}
          </p>
          <p className="text-[13px] leading-snug text-fg-muted">{citation.snippet}</p>
        </div>
      )}
    </span>
  )
}

// generate_cite_node (app/retrieval/nodes/generate_cite.py) returns `answer` as plain text that
// still contains inline [doc_id:page_number:chunk_id] markers for every citation it validated —
// see that file's docstring. This splits the answer into individual sentence-sized *lines*, each
// rendered as its own paragraph (MessageBubble.tsx) — a multi-point answer (a summary, a list of
// facts) needs one visually separated line per point, or it reads as an unbroken wall of text
// regardless of how good the prose itself is. Each line is further split into text/citation
// segments, numbered in order of first appearance across the *whole* answer (repeats of the same
// chunk_id keep the same number, like footnotes), never per line.

import type { Citation } from '../api/types'

// Mirrors the server's own citation-marker regex exactly (generate_cite.py's
// _CITATION_MARKER_RE) — this must stay in sync with that pattern, not with any other guess at
// what a "citation-looking" bracket might be.
const CITATION_MARKER_RE = /\[([^[\]:]+):(\d+):([^[\]:]+)\]/g

// A sentence boundary is "]" or [.!?] followed by whitespace and a capital letter — not just
// [.!?], because a citation marker (ending in "]") almost always follows a sentence's punctuation
// directly ("fact stated. [doc:1:abc]"), and splitting there would strand the citation on the
// wrong line. Splitting after the marker's "]" instead keeps a sentence and its own citation(s)
// together as one line.
const SENTENCE_BOUNDARY_RE = /(?<=[\].!?])\s+(?=[A-Z])/g

export type AnswerSegment =
  | { type: 'text'; text: string }
  | { type: 'citation'; number: number; citation: Citation | undefined }

export type AnswerLine = AnswerSegment[]

export function parseAnswerLines(answer: string, citations: Citation[]): AnswerLine[] {
  const citationByChunkId = new Map(citations.map((c) => [c.chunk_id, c]))
  const numberByChunkId = new Map<string, number>()

  const sentences = answer.trim().split(SENTENCE_BOUNDARY_RE).filter((s) => s.length > 0)

  return sentences.map((sentence) => {
    const segments: AnswerLine = []
    let cursor = 0
    let match: RegExpExecArray | null

    CITATION_MARKER_RE.lastIndex = 0
    while ((match = CITATION_MARKER_RE.exec(sentence)) !== null) {
      const [fullMatch, , , chunkId] = match

      if (match.index > cursor) {
        segments.push({ type: 'text', text: sentence.slice(cursor, match.index) })
      }

      if (!numberByChunkId.has(chunkId)) {
        numberByChunkId.set(chunkId, numberByChunkId.size + 1)
      }
      segments.push({ type: 'citation', number: numberByChunkId.get(chunkId)!, citation: citationByChunkId.get(chunkId) })

      cursor = match.index + fullMatch.length
    }

    if (cursor < sentence.length) {
      segments.push({ type: 'text', text: sentence.slice(cursor) })
    }

    return segments
  })
}

export type LiveSegment = { type: 'text'; text: string } | { type: 'citation-pending' }

// Used only while a response is still streaming (MessageBubble.tsx). The final `citations[]`
// array — needed to turn a marker into a real, trusted chip — doesn't exist until the stream's
// "done" event, so this can't resolve anything yet. What it *can* do: never show the raw
// "[doc_id:page:chunk_id]" text. Any complete bracket becomes a neutral placeholder immediately,
// and a bracket that's still arriving (the buffer ends with an unclosed "[", possibly mid-UUID) is
// held back from rendering entirely rather than shown as a half-formed fragment — it renders once
// it either completes or turns out not to be a bracket after all.
export function parseLiveSegments(text: string): LiveSegment[] {
  const segments: LiveSegment[] = []
  const completeBracketRe = /\[[^[\]]*\]/g
  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = completeBracketRe.exec(text)) !== null) {
    if (match.index > cursor) segments.push({ type: 'text', text: text.slice(cursor, match.index) })
    segments.push({ type: 'citation-pending' })
    cursor = match.index + match[0].length
  }

  const tail = text.slice(cursor)
  const openBracketIndex = tail.lastIndexOf('[')
  if (openBracketIndex !== -1) {
    if (openBracketIndex > 0) segments.push({ type: 'text', text: tail.slice(0, openBracketIndex) })
    // tail.slice(openBracketIndex) is held back — it may still be an in-progress citation marker.
  } else if (tail.length > 0) {
    segments.push({ type: 'text', text: tail })
  }

  return segments
}

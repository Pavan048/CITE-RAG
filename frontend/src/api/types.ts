// Mirrors app/models/schemas.py exactly (PRD Section 4). Keep these in lockstep with that file —
// this is the API's public contract, not a place to add UI-only fields (put those in hook/
// component-local types instead).

export type DocumentStatus = 'queued' | 'processing' | 'ready' | 'failed'

export type PipelineStage = 'validate' | 'parse' | 'chunk' | 'embed' | 'index' | 'ready'

export interface JobError {
  stage: PipelineStage
  message: string
}

export interface UploadDocumentResponse {
  doc_id: string
  job_id: string
}

export interface JobStatusResponse {
  status: DocumentStatus
  stage: PipelineStage
  progress_pct: number
  error: JobError | null
  pages_failed: number[] | null
}

export interface DocumentListItem {
  doc_id: string
  filename: string
  page_count: number
  status: DocumentStatus
  uploaded_at: string
}

export interface QueryRequest {
  question: string
  file_ids?: string[]
  top_k?: number
}

export interface Citation {
  doc_id: string
  page_number: number
  chunk_id: string
  snippet: string
}

// Debug trace — not part of the PRD's documented contract (Section 4), added on top of it so a
// debug panel can show which nodes ran, what was retrieved per hop, and every prompt/response,
// instead of that only being visible in server logs. See app/retrieval/debug_trace.py.
export interface ChunkTrace {
  chunk_id: string
  doc_id: string
  page_number: number
  content_type: string
  score: number
  text_preview: string
}

export interface HopTrace {
  hop_number: number
  sub_question: string
  chunks_retrieved: ChunkTrace[]
}

export interface LLMCallTrace {
  node: string
  prompt: string
  response: string
}

export interface DebugInfo {
  graph_path: string[]
  is_multi_hop: boolean
  hop_count: number
  not_found: boolean
  sub_question_history: string[]
  hops: HopTrace[]
  reranked_chunks: ChunkTrace[]
  context_block_count: number
  llm_calls: LLMCallTrace[]
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  routed_docs: string[] | null
  partial_answer: boolean
  debug: DebugInfo | null
}

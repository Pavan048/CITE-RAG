// Typed fetch wrappers for app/api/documents.py, jobs.py, query.py. Every call that touches the
// user's BYOK key builds its headers here and nowhere else, so there is exactly one place that
// ever reads it out of state.

import type {
  DocumentListItem,
  JobStatusResponse,
  QueryRequest,
  QueryResponse,
  UploadDocumentResponse,
} from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
    if (body?.detail) return JSON.stringify(body.detail)
    return response.statusText
  } catch {
    return response.statusText
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response))
  }
  return response.json() as Promise<T>
}

function authHeaders(llmKey: string, llmModel?: string): HeadersInit {
  const headers: Record<string, string> = { 'X-LLM-Key': llmKey }
  if (llmModel) headers['X-LLM-Model'] = llmModel
  return headers
}

export async function listDocuments(): Promise<DocumentListItem[]> {
  const response = await fetch('/v1/documents')
  return handle(response)
}

export async function uploadDocument(
  file: File,
  llmKey: string,
  llmModel?: string,
): Promise<UploadDocumentResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch('/v1/documents', {
    method: 'POST',
    headers: authHeaders(llmKey, llmModel),
    body: formData,
  })
  return handle(response)
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(`/v1/jobs/${jobId}`)
  return handle(response)
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`/v1/documents/${docId}`, { method: 'DELETE' })
  if (!response.ok && response.status !== 204) {
    throw new ApiError(response.status, await parseErrorDetail(response))
  }
}

export async function query(
  request: QueryRequest,
  llmKey: string,
  llmModel?: string,
): Promise<QueryResponse> {
  const response = await fetch('/v1/query', {
    method: 'POST',
    headers: { ...authHeaders(llmKey, llmModel), 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return handle(response)
}

export interface QueryStreamCallbacks {
  onToken: (text: string) => void
  onDone: (result: QueryResponse) => void
  onError: (message: string) => void
}

// The backend's /v1/query/stream (app/api/query.py) sends Server-Sent Events. The native
// `EventSource` can't be used here — it only supports GET with no custom headers, and this needs
// POST plus X-LLM-Key — so this reads the fetch response body as a stream and parses the SSE
// framing ("event: <name>\ndata: <json>\n\n") by hand instead.
export async function queryStream(
  request: QueryRequest,
  llmKey: string,
  llmModel: string | undefined,
  callbacks: QueryStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/v1/query/stream', {
    method: 'POST',
    headers: { ...authHeaders(llmKey, llmModel), 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })

  if (!response.ok || !response.body) {
    callbacks.onError(await parseErrorDetail(response))
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let sawTerminalEvent = false

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      const eventLine = rawEvent.split('\n').find((line) => line.startsWith('event:'))
      const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data:'))
      if (eventLine && dataLine) {
        const eventName = eventLine.slice('event:'.length).trim()
        const data = JSON.parse(dataLine.slice('data:'.length).trim())

        if (eventName === 'token') callbacks.onToken(data.text)
        else if (eventName === 'done') {
          sawTerminalEvent = true
          callbacks.onDone(data as QueryResponse)
        } else if (eventName === 'error') {
          sawTerminalEvent = true
          callbacks.onError(data.message)
        }
      }

      boundary = buffer.indexOf('\n\n')
    }
  }

  // The connection closed without ever sending a "done" or "error" frame — the backend always
  // sends one now (see api/query.py's event_stream, which wraps its entire body in a try/except
  // for exactly this reason), so reaching this point means something outside the application
  // itself cut the connection (e.g. the server process died). Without this, the caller would be
  // left showing "thinking" forever, with no way to know the request will never resolve.
  if (!sawTerminalEvent) {
    callbacks.onError('Connection to the server was lost before a response arrived. Please try again.')
  }
}

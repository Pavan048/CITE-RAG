import type { DocumentStatus } from '../api/types'

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

export function shortId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id
}

const STATUS_LABEL: Record<DocumentStatus, string> = {
  queued: 'Queued',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
}

const STATUS_DOT_CLASSES: Record<DocumentStatus, string> = {
  queued: 'bg-warn',
  processing: 'bg-warn',
  ready: 'bg-success',
  failed: 'bg-danger',
}

export function statusLabel(status: DocumentStatus): string {
  return STATUS_LABEL[status]
}

export function statusDotClasses(status: DocumentStatus): string {
  return STATUS_DOT_CLASSES[status]
}

import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useState } from 'react'
import { deleteDocument, getJob, listDocuments, uploadDocument } from '../api/client'
import type { DocumentListItem, JobStatusResponse } from '../api/types'

const DOCUMENTS_KEY = ['documents']

export interface PendingUpload {
  jobId: string
  filename: string
  job: JobStatusResponse | undefined
}

function isTerminal(status: JobStatusResponse['status'] | undefined): boolean {
  return status === 'ready' || status === 'failed'
}

export function useDocuments(llmKey: string, llmModel: string) {
  const queryClient = useQueryClient()

  const documentsQuery = useQuery({
    queryKey: DOCUMENTS_KEY,
    queryFn: listDocuments,
    // Poll a little faster while anything is still in flight, otherwise there's no need to keep
    // hammering the endpoint just to notice nothing has changed.
    refetchInterval: (query) => {
      const docs = query.state.data ?? []
      const anyInFlight = docs.some((d) => d.status === 'queued' || d.status === 'processing')
      return anyInFlight ? 2000 : 10000
    },
  })

  // jobId -> filename, set at upload time (JobStatusResponse itself carries no filename/doc_id).
  const [pendingFilenames, setPendingFilenames] = useState<Record<string, string>>({})
  const pendingJobIds = Object.keys(pendingFilenames)

  const jobQueries = useQueries({
    queries: pendingJobIds.map((jobId) => ({
      queryKey: ['job', jobId],
      queryFn: () => getJob(jobId),
      refetchInterval: (query: { state: { data?: JobStatusResponse } }) =>
        isTerminal(query.state.data?.status) ? false : 1000,
    })),
  })

  useEffect(() => {
    const finishedJobIds = pendingJobIds.filter((_, index) => isTerminal(jobQueries[index]?.data?.status))
    if (finishedJobIds.length === 0) return
    queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY })
    setPendingFilenames((prev) => {
      const next = { ...prev }
      for (const jobId of finishedJobIds) delete next[jobId]
      return next
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobQueries.map((q) => q.data?.status).join(',')])

  const pendingUploads: PendingUpload[] = pendingJobIds.map((jobId, index) => ({
    jobId,
    filename: pendingFilenames[jobId],
    job: jobQueries[index]?.data,
  }))

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadDocument(file, llmKey, llmModel || undefined),
    onSuccess: ({ job_id }, file) => {
      setPendingFilenames((prev) => ({ ...prev, [job_id]: file.name }))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY }),
  })

  const upload = useCallback((file: File) => uploadMutation.mutate(file), [uploadMutation])

  return {
    documents: (documentsQuery.data ?? []) as DocumentListItem[],
    isLoading: documentsQuery.isLoading,
    upload,
    uploadError: uploadMutation.error,
    pendingUploads,
    deleteDoc: deleteMutation.mutate,
    deletingDocId: deleteMutation.isPending ? (deleteMutation.variables as string) : undefined,
  }
}

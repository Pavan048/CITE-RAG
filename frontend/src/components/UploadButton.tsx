import { Upload } from 'lucide-react'
import { useRef } from 'react'

interface UploadButtonProps {
  disabled: boolean
  disabledReason?: string
  onSelect: (file: File) => void
}

export function UploadButton({ disabled, disabledReason, onSelect }: UploadButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <>
      <button
        type="button"
        title={disabled ? disabledReason : undefined}
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2.5 text-[13px] font-medium text-accent-fg transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Upload size={15} />
        Upload PDF
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onSelect(file)
          event.target.value = ''
        }}
      />
    </>
  )
}

import { TriangleAlert } from 'lucide-react'

export function PartialAnswerBanner() {
  return (
    <div className="mb-3 flex items-start gap-2 rounded-lg bg-warn-bg px-3 py-2.5 font-sans text-[12.5px] text-warn">
      <TriangleAlert size={14} className="mt-0.5 shrink-0" />
      <span>Partial answer — ran out of reasoning hops before finding enough evidence to fully answer this.</span>
    </div>
  )
}

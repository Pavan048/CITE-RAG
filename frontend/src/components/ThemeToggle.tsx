import { Monitor, Moon, Sun } from 'lucide-react'
import type { ThemePreference } from '../hooks/useTheme'

const OPTIONS: { value: ThemePreference; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
]

interface ThemeToggleProps {
  preference: ThemePreference
  onChange: (next: ThemePreference) => void
}

export function ThemeToggle({ preference, onChange }: ThemeToggleProps) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface p-0.5">
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          type="button"
          aria-label={label}
          title={label}
          onClick={() => onChange(value)}
          className={`flex h-6 w-6 items-center justify-center rounded-md transition-colors ${
            preference === value ? 'bg-accent text-accent-fg' : 'text-fg-muted hover:bg-surface-hover'
          }`}
        >
          <Icon size={13} />
        </button>
      ))}
    </div>
  )
}

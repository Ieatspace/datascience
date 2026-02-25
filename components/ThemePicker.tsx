"use client";

import { Palette, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useThemeManager } from "@/components/ThemeProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type ThemePickerProps = {
  compact?: boolean;
};

export function ThemePicker({ compact = false }: ThemePickerProps) {
  const { theme, themes, setThemeId } = useThemeManager();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return themes;
    return themes.filter((t) => t.name.toLowerCase().includes(q));
  }, [query, themes]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (rootRef.current && target && !rootRef.current.contains(target)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative z-[70]">
      <Button
        type="button"
        variant="outline"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="h-10 gap-2 rounded-xl border-[var(--border)] bg-[var(--panel)] px-3 text-[var(--text)] shadow-sm shadow-[var(--shadow)] hover:bg-[var(--panel-strong)]"
      >
        <Palette className="h-4 w-4" />
        <span className={cn(compact ? "hidden sm:inline" : "inline")}>{theme.name}</span>
      </Button>

      {open ? (
        <div className="absolute right-0 top-12 z-[90] w-[min(22rem,calc(100vw-2rem))] rounded-2xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--panel-strong)_90%,transparent)] p-3 shadow-2xl shadow-[var(--shadow)] ring-1 ring-[color-mix(in_srgb,var(--accent)_18%,transparent)] backdrop-blur-xl">
          <div className="mb-3 flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--panel)] px-2">
            <Search className="h-4 w-4 text-[var(--muted)]" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search themes..."
              className="border-0 bg-transparent px-0 text-[var(--text)] shadow-none focus-visible:ring-0"
            />
          </div>
          <div className="grid max-h-72 grid-cols-2 gap-2 overflow-auto pr-1 scrollbar-thin">
            {filtered.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  setThemeId(t.id);
                  setOpen(false);
                }}
                className={cn(
                  "group rounded-xl border p-2 text-left transition duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-[var(--shadow)]",
                  t.id === theme.id
                    ? "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_10%,var(--panel))]"
                    : "border-[var(--border)] bg-[var(--panel)] hover:border-[color-mix(in_srgb,var(--accent)_35%,var(--border))]"
                )}
              >
                <div
                  className="mb-2 h-8 rounded-lg border"
                  style={{
                    background: t.cssVars["--app-gradient"],
                    borderColor: t.cssVars["--border"]
                  }}
                />
                <div className="text-xs font-semibold text-[var(--text)]">{t.name}</div>
                <div className="mt-1 flex gap-1">
                  <span className="h-2 w-2 rounded-full" style={{ background: t.cssVars["--accent"] }} />
                  <span className="h-2 w-2 rounded-full" style={{ background: t.cssVars["--accent2"] }} />
                  <span className="h-2 w-2 rounded-full border" style={{ background: t.cssVars["--panel"], borderColor: t.cssVars["--border"] }} />
                </div>
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

"use client";

import { Download, Maximize2, Minus, Plus, ScanLine } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { DebugOverlay } from "@/components/DebugOverlay";
import { Button } from "@/components/ui/button";
import type { GenerateMeta } from "@/lib/types";
import { cn } from "@/lib/utils";

type PreviewCanvasProps = {
  imageDataUrl: string | null;
  meta: GenerateMeta | null;
  isGenerating: boolean;
  error: string | null;
  showDebugOverlay: boolean;
  showBoxes: boolean;
  showLabels: boolean;
  showFallbackMarkers: boolean;
  fitToWidth: boolean;
  zoom: number;
  reducedMotion: boolean;
  onZoomChange: (value: number) => void;
  onFitToWidth: () => void;
  onToggleFit: (value: boolean) => void;
};

export function PreviewCanvas(props: PreviewCanvasProps) {
  const {
    imageDataUrl,
    meta,
    isGenerating,
    error,
    showDebugOverlay,
    showBoxes,
    showLabels,
    showFallbackMarkers,
    fitToWidth,
    zoom,
    reducedMotion,
    onZoomChange,
    onFitToWidth,
    onToggleFit
  } = props;

  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number; left: number; top: number } | null>(null);
  const [imageSize, setImageSize] = useState({ width: 800, height: 480 });
  const [revealKey, setRevealKey] = useState(0);

  useEffect(() => {
    if (imageDataUrl) setRevealKey((k) => k + 1);
  }, [imageDataUrl]);

  const scale = fitToWidth ? 1 : zoom;
  const imageStyle = useMemo(
    () => ({
      transform: `scale(${scale})`,
      transformOrigin: "top left"
    }),
    [scale]
  );

  return (
    <div className="studio-panel-strong panel-glow accent-ring relative overflow-hidden p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Live Preview</p>
          <p className="text-sm font-medium text-[var(--text)]">
            {isGenerating ? "Generating..." : imageDataUrl ? "Rendered output" : "No image yet"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="studio-button-soft h-8 rounded-lg"
            onClick={() => onZoomChange(Math.max(0.4, +(zoom - 0.1).toFixed(2)))}
            disabled={fitToWidth}
          >
            <Minus className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="studio-button-soft h-8 rounded-lg"
            onClick={() => onZoomChange(Math.min(3, +(zoom + 0.1).toFixed(2)))}
            disabled={fitToWidth}
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            size="sm"
            variant={fitToWidth ? "default" : "outline"}
            className={cn("h-8 rounded-lg", fitToWidth ? "studio-button-accent" : "studio-button-soft")}
            onClick={() => {
              onToggleFit(!fitToWidth);
              if (!fitToWidth) onFitToWidth();
            }}
          >
            <Maximize2 className="mr-1 h-3.5 w-3.5" />
            Fit
          </Button>
        </div>
      </div>

      <div
        ref={scrollerRef}
        className={cn(
          "relative h-[24rem] overflow-auto rounded-xl border border-[var(--border)] bg-[var(--panel)] p-3 scrollbar-thin md:h-[32rem]",
          isGenerating && !reducedMotion ? "preview-shimmer" : ""
        )}
        style={{
          cursor: isDragging ? "grabbing" : "grab",
          backgroundImage:
            "radial-gradient(circle at 12% 10%, color-mix(in srgb, var(--accent) 7%, transparent), transparent 42%), radial-gradient(circle at 88% 14%, color-mix(in srgb, var(--accent2) 7%, transparent), transparent 46%)"
        }}
        onPointerDown={(e) => {
          const target = e.target as HTMLElement;
          if (target.closest("button")) return;
          const el = scrollerRef.current;
          if (!el) return;
          setIsDragging(true);
          setDragStart({ x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop });
        }}
        onPointerMove={(e) => {
          if (!isDragging || !dragStart || !scrollerRef.current) return;
          scrollerRef.current.scrollLeft = dragStart.left - (e.clientX - dragStart.x);
          scrollerRef.current.scrollTop = dragStart.top - (e.clientY - dragStart.y);
        }}
        onPointerUp={() => {
          setIsDragging(false);
          setDragStart(null);
        }}
        onPointerLeave={() => {
          setIsDragging(false);
          setDragStart(null);
        }}
      >
        {imageDataUrl ? (
          <div
            className={cn(
              "relative inline-block rounded-lg shadow-lg transition-transform duration-200",
              reducedMotion ? "" : "reveal-wipe"
            )}
            key={revealKey}
            style={imageStyle}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imageRef}
              src={imageDataUrl}
              alt="Generated handwriting preview"
              className="block rounded-lg border border-[var(--border)] bg-white/80 shadow-[0_14px_40px_-24px_rgba(0,0,0,0.45)]"
              onLoad={(e) => {
                const img = e.currentTarget;
                setImageSize({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height });
              }}
            />
            {showDebugOverlay ? (
              <DebugOverlay
                meta={meta}
                imageRect={imageSize}
                showBoxes={showBoxes}
                showLabels={showLabels}
                showFallbackMarkers={showFallbackMarkers}
              />
            ) : null}
          </div>
        ) : (
          <div className="grid h-full place-items-center rounded-xl border border-dashed border-[var(--border)] bg-[color-mix(in_srgb,var(--panel)_70%,transparent)]">
            <div className="text-center">
              <ScanLine className="mx-auto h-8 w-8 text-[var(--muted)]" />
              <p className="mt-3 text-sm font-medium text-[var(--text)]">Preview will appear here</p>
              <p className="text-xs text-[var(--muted)]">Generate from the left panel to render handwriting.</p>
            </div>
          </div>
        )}

        {isGenerating ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-3 px-3">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--panel-strong)]/90 p-2 text-xs text-[var(--text)] shadow">
              Sampling letters, composing lines, and applying paper styling...
            </div>
          </div>
        ) : null}
      </div>

      {error ? (
        <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--muted)]">
        <div>
          {meta?.stats ? (
            <span>
              model: {meta.stats.modelChars ?? 0} | fallback: {meta.stats.fallbackChars ?? 0} | joins:{" "}
              {meta.stats.joins ?? 0}
            </span>
          ) : (
            <span>fallback info unavailable</span>
          )}
        </div>
        <div className="inline-flex items-center gap-1">
          <Download className="h-3.5 w-3.5" />
          Drag to pan | {fitToWidth ? "Fit" : `${Math.round(zoom * 100)}%`}
        </div>
      </div>
    </div>
  );
}

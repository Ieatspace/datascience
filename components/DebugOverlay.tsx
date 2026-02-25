"use client";

import type { GenerateMeta } from "@/lib/types";

type DebugOverlayProps = {
  meta: GenerateMeta | null;
  imageRect: { width: number; height: number };
  showBoxes: boolean;
  showLabels: boolean;
  showFallbackMarkers: boolean;
};

export function DebugOverlay({
  meta,
  imageRect,
  showBoxes,
  showLabels,
  showFallbackMarkers
}: DebugOverlayProps) {
  if (!meta || !meta.letters.length || (!showBoxes && !showLabels && !showFallbackMarkers)) {
    return null;
  }

  const maxX = Math.max(...meta.letters.map((l) => l.bbox.x + l.bbox.w), 1);
  const maxY = Math.max(...meta.letters.map((l) => l.bbox.y + l.bbox.h), 1);
  const sx = imageRect.width / maxX;
  const sy = imageRect.height / maxY;

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${imageRect.width} ${imageRect.height}`} preserveAspectRatio="none">
      {meta.letters.map((letter, idx) => {
        const x = letter.bbox.x * sx;
        const y = letter.bbox.y * sy;
        const w = letter.bbox.w * sx;
        const h = letter.bbox.h * sy;
        const fallback = letter.source === "fallback";
        const stroke = fallback ? "#f97316" : "#22c55e";

        return (
          <g key={`${letter.char}-${idx}`}>
            {showBoxes ? (
              <rect
                x={x}
                y={y}
                width={Math.max(1, w)}
                height={Math.max(1, h)}
                fill={fallback ? "rgba(249,115,22,0.08)" : "rgba(34,197,94,0.05)"}
                stroke={stroke}
                strokeDasharray={fallback ? "4 3" : "0"}
                strokeWidth={1}
                rx={4}
              />
            ) : null}
            {showFallbackMarkers && fallback ? (
              <circle cx={x + Math.max(8, w - 6)} cy={y + 6} r={4.5} fill="#f97316" opacity={0.9} />
            ) : null}
            {showLabels ? (
              <text
                x={x + 2}
                y={Math.max(10, y - 2)}
                fontSize={10}
                fill={stroke}
                fontWeight={700}
                stroke="rgba(0,0,0,0.25)"
                strokeWidth={0.2}
              >
                {letter.char}
                {typeof letter.confidence === "number" ? ` ${letter.confidence.toFixed(2)}` : ""}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}


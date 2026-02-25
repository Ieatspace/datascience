import type { GenerateMeta } from "@/lib/types";

type SynthesizeOptions = {
  text: string;
  width: number;
  height: number;
  lineSpacing: number;
};

export function synthesizeDebugMeta({
  text,
  width,
  height,
  lineSpacing
}: SynthesizeOptions): GenerateMeta {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const letters: GenerateMeta["letters"] = [];
  const padX = Math.max(32, Math.round(width * 0.05));
  const padY = Math.max(28, Math.round(height * 0.08));
  const fontPx = Math.max(18, Math.min(52, Math.round(width * 0.03)));
  const rowStep = Math.max(fontPx * lineSpacing, fontPx * 1.12);
  const avgWidth = fontPx * 0.52;

  lines.forEach((line, rowIndex) => {
    let x = padX;
    const y = padY + rowIndex * rowStep;
    for (const ch of line) {
      if (ch === " ") {
        x += avgWidth * 0.7;
        continue;
      }
      const w = /[iljt]/i.test(ch) ? avgWidth * 0.38 : /[mw]/i.test(ch) ? avgWidth * 0.9 : avgWidth * 0.62;
      const h = /[gjpqy]/i.test(ch) ? fontPx * 1.05 : fontPx * 0.85;
      letters.push({
        char: ch,
        bbox: {
          x: Math.round(x),
          y: Math.round(y),
          w: Math.max(4, Math.round(w)),
          h: Math.max(8, Math.round(h))
        },
        source: /[a-z]/.test(ch) ? "generated" : "fallback"
      });
      x += w + avgWidth * 0.18;
    }
  });

  return {
    letters,
    warnings: ["fallback info unavailable"],
    fallbackInfoAvailable: false
  };
}


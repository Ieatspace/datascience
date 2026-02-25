import sharp from "sharp";

import { type GenerateRequest, type HandwritingStyle } from "@/lib/types";
import { mulberry32, randomBetween } from "@/lib/rng";

type StyleProfile = {
  fill: string;
  stroke: string;
  strokeWidth: number;
  opacity: number;
  fontWeight: number;
  jitterX: number;
  jitterY: number;
  rotation: number;
};

const STYLE_PROFILES: Record<HandwritingStyle, StyleProfile> = {
  pencil: {
    fill: "#4b5563",
    stroke: "#94a3b8",
    strokeWidth: 0.35,
    opacity: 0.88,
    fontWeight: 450,
    jitterX: 1.4,
    jitterY: 1.7,
    rotation: 3.6
  },
  ink: {
    fill: "#111827",
    stroke: "#1f2937",
    strokeWidth: 0.16,
    opacity: 0.95,
    fontWeight: 500,
    jitterX: 1.1,
    jitterY: 1.2,
    rotation: 2.4
  },
  marker: {
    fill: "#1f2937",
    stroke: "#374151",
    strokeWidth: 0.42,
    opacity: 0.9,
    fontWeight: 600,
    jitterX: 1.6,
    jitterY: 1.9,
    rotation: 3.1
  }
};

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function glyphWidth(fontSize: number, char: string): number {
  if (char === " ") {
    return fontSize * 0.34;
  }
  if (/[ilI'`.,;:!|]/.test(char)) {
    return fontSize * 0.26;
  }
  if (/[mwMW@#%&]/.test(char)) {
    return fontSize * 0.78;
  }
  if (/[A-Z]/.test(char)) {
    return fontSize * 0.61;
  }
  if (/[0-9]/.test(char)) {
    return fontSize * 0.53;
  }
  return fontSize * 0.5;
}

function measureTextWidth(fontSize: number, line: string): number {
  let width = 0;
  for (const char of line) {
    width += glyphWidth(fontSize, char);
  }
  return width;
}

function wrapParagraph(paragraph: string, maxWidth: number, fontSize: number): string[] {
  if (paragraph.trim().length === 0) {
    return [""];
  }

  const words = paragraph.trim().split(/\s+/);
  const rows: string[] = [];
  let currentRow = "";

  for (const word of words) {
    const tentative = currentRow ? `${currentRow} ${word}` : word;
    if (measureTextWidth(fontSize, tentative) <= maxWidth) {
      currentRow = tentative;
      continue;
    }

    if (currentRow) {
      rows.push(currentRow);
      currentRow = "";
    }

    if (measureTextWidth(fontSize, word) <= maxWidth) {
      currentRow = word;
      continue;
    }

    let chunk = "";
    for (const char of word) {
      const candidate = `${chunk}${char}`;
      if (measureTextWidth(fontSize, candidate) > maxWidth && chunk.length > 0) {
        rows.push(chunk);
        chunk = char;
      } else {
        chunk = candidate;
      }
    }

    currentRow = chunk;
  }

  if (currentRow) {
    rows.push(currentRow);
  }

  return rows;
}

function wrapTextToRows(
  text: string,
  maxWidth: number,
  fontSize: number,
  maxRows: number
): { rows: string[]; truncated: boolean } {
  const normalized = text.replace(/\r\n/g, "\n");
  const paragraphs = normalized.split("\n");
  const rows: string[] = [];

  for (const paragraph of paragraphs) {
    const wrapped = wrapParagraph(paragraph, maxWidth, fontSize);
    for (const row of wrapped) {
      if (rows.length >= maxRows) {
        return { rows, truncated: true };
      }
      rows.push(row);
    }
  }

  return { rows, truncated: false };
}

function buildPaperNoise(
  width: number,
  height: number,
  rng: () => number,
  intensity: "off" | "subtle" | "med" = "subtle"
): string {
  if (intensity === "off") {
    return "";
  }
  const speckleCount = Math.max(
    intensity === "med" ? 65 : 40,
    Math.round((width * height) / (intensity === "med" ? 24000 : 38000))
  );
  const nodes: string[] = [];

  for (let index = 0; index < speckleCount; index += 1) {
    const x = randomBetween(rng, 0, width);
    const y = randomBetween(rng, 0, height);
    const r = randomBetween(rng, 0.25, intensity === "med" ? 1.1 : 0.9);
    const opacity = randomBetween(rng, intensity === "med" ? 0.02 : 0.015, intensity === "med" ? 0.07 : 0.05);
    nodes.push(
      `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r.toFixed(2)}" fill="#111827" opacity="${opacity.toFixed(3)}" />`
    );
  }

  return nodes.join("");
}

function buildRuledLines(
  width: number,
  height: number,
  baselineStep: number,
  topPadding: number
): string {
  const lines: string[] = [];
  let y = topPadding + baselineStep * 0.6;

  while (y < height - 24) {
    lines.push(
      `<line x1="28" y1="${y.toFixed(2)}" x2="${(width - 28).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#cdd8ea" stroke-width="1" opacity="0.55" />`
    );
    y += baselineStep;
  }

  return lines.join("");
}

function buildGridLines(width: number, height: number, baselineStep: number, topPadding: number): string {
  const horizontal: string[] = [];
  let y = topPadding + baselineStep * 0.6;
  while (y < height - 24) {
    horizontal.push(
      `<line x1="28" y1="${y.toFixed(2)}" x2="${(width - 28).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#d8e0ee" stroke-width="1" opacity="0.52" />`
    );
    y += baselineStep;
  }

  const vertical: string[] = [];
  const gridStep = Math.max(20, Math.round(baselineStep));
  for (let x = 28; x < width - 28; x += gridStep) {
    vertical.push(
      `<line x1="${x}" y1="20" x2="${x}" y2="${height - 20}" stroke="#e0e7f2" stroke-width="1" opacity="0.5" />`
    );
  }

  return `${horizontal.join("")}${vertical.join("")}`;
}

function buildDotGrid(width: number, height: number, baselineStep: number, topPadding: number): string {
  const nodes: string[] = [];
  const step = Math.max(20, Math.round(baselineStep));
  for (let y = topPadding + baselineStep * 0.6; y < height - 24; y += step) {
    for (let x = 28; x < width - 28; x += step) {
      nodes.push(
        `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.1" fill="#c9d3e4" opacity="0.75" />`
      );
    }
  }
  return nodes.join("");
}

function buildTextGlyphs(
  rows: string[],
  options: {
    width: number;
    height: number;
    fontSize: number;
    lineSpacing: number;
    style: HandwritingStyle;
    paddingX: number;
    paddingTop: number;
    rng: () => number;
  }
): string {
  const {
    width,
    fontSize,
    lineSpacing,
    style,
    paddingX,
    paddingTop,
    rng
  } = options;
  const profile = STYLE_PROFILES[style];
  const baselineStep = fontSize * lineSpacing;
  const nodes: string[] = [];
  const maxX = width - paddingX;

  rows.forEach((row, rowIndex) => {
    const baselineY =
      paddingTop +
      fontSize +
      rowIndex * baselineStep +
      randomBetween(rng, -0.5, 0.7);

    let cursorX = paddingX + randomBetween(rng, -2.4, 2.4);

    for (const char of row) {
      const advance = glyphWidth(fontSize, char);
      if (char === " ") {
        cursorX += advance + randomBetween(rng, -0.2, 0.45);
        continue;
      }

      if (cursorX > maxX - advance) {
        break;
      }

      const x = cursorX + randomBetween(rng, -profile.jitterX, profile.jitterX);
      const y = baselineY + randomBetween(rng, -profile.jitterY, profile.jitterY);
      const rotation = randomBetween(rng, -profile.rotation, profile.rotation);
      const opacity = Math.max(
        0.72,
        Math.min(1, profile.opacity + randomBetween(rng, -0.06, 0.05))
      );
      const skew = randomBetween(rng, -3, 3);

      nodes.push(
        `<text x="${x.toFixed(2)}" y="${y.toFixed(2)}" transform="rotate(${rotation.toFixed(2)} ${x.toFixed(2)} ${y.toFixed(2)}) skewX(${skew.toFixed(2)})" fill="${profile.fill}" stroke="${profile.stroke}" stroke-width="${profile.strokeWidth}" paint-order="stroke fill" opacity="${opacity.toFixed(3)}" font-size="${fontSize}" font-weight="${profile.fontWeight}" letter-spacing="${randomBetween(rng, -0.15, 0.25).toFixed(2)}" font-family="Segoe Print, Bradley Hand, Comic Sans MS, Chalkboard SE, cursive, sans-serif">${escapeXml(char)}</text>`
      );

      if (style === "pencil" && rng() > 0.7) {
        nodes.push(
          `<text x="${(x + randomBetween(rng, -0.6, 0.6)).toFixed(2)}" y="${(y + randomBetween(rng, -0.4, 0.5)).toFixed(2)}" fill="#111827" opacity="0.05" font-size="${fontSize}" font-weight="${profile.fontWeight}" font-family="Segoe Print, Bradley Hand, Comic Sans MS, Chalkboard SE, cursive, sans-serif">${escapeXml(char)}</text>`
        );
      }

      cursorX += advance + randomBetween(rng, -0.4, 0.7);
    }
  });

  return nodes.join("");
}

function truncateRowsWithEllipsis(rows: string[]): string[] {
  if (rows.length === 0) {
    return rows;
  }

  const next = [...rows];
  const last = next[next.length - 1];
  next[next.length - 1] = `${last.replace(/\s+$/, "").slice(0, Math.max(0, last.length - 1))}...`;
  return next;
}

function buildPlaceholderSvg(request: GenerateRequest): string {
  const { width, height, lineSpacing, text, style } = request;
  const pageStyle = request.pageStyle ?? "lined";
  const paperTexture = request.paperTexture ?? "subtle";
  const paddingX = Math.max(36, Math.round(width * 0.05));
  const paddingTop = Math.max(28, Math.round(height * 0.045));
  const paddingBottom = Math.max(24, Math.round(height * 0.04));
  const fontSize = Math.max(18, Math.min(42, Math.round(width * 0.031)));
  const baselineStep = Math.max(fontSize * 1.1, fontSize * lineSpacing);
  const maxRows = Math.max(
    1,
    Math.floor((height - paddingTop - paddingBottom - fontSize * 0.4) / baselineStep)
  );
  const innerWidth = width - paddingX * 2 - 8;

  const seed = request.seed ?? Math.floor(Math.random() * 0xffffffff);
  const rng = mulberry32(seed);

  const wrapped = wrapTextToRows(text, innerWidth, fontSize, maxRows);
  const rows = wrapped.truncated ? truncateRowsWithEllipsis(wrapped.rows) : wrapped.rows;

  const paperNoise = buildPaperNoise(width, height, rng, paperTexture);
  const guides =
    pageStyle === "blank"
      ? ""
      : pageStyle === "grid"
        ? buildGridLines(width, height, baselineStep, paddingTop)
        : pageStyle === "dot"
          ? buildDotGrid(width, height, baselineStep, paddingTop)
          : buildRuledLines(width, height, baselineStep, paddingTop);
  const leftMargin =
    pageStyle === "lined" || pageStyle === "grid"
      ? `<line x1="56" y1="20" x2="56" y2="${(height - 20).toFixed(2)}" stroke="#f3a3a3" stroke-width="1" opacity="0.45" />`
      : "";
  const glyphs = buildTextGlyphs(rows, {
    width,
    height,
    fontSize,
    lineSpacing,
    style,
    paddingX,
    paddingTop,
    rng
  });
  const headerNote = `Style: ${style.toUpperCase()} | ${pageStyle}${request.seed != null ? ` | Seed: ${request.seed}` : " | Seed: auto"}`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <linearGradient id="paperShade" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#fffefa" />
      <stop offset="100%" stop-color="#fbfaf6" />
    </linearGradient>
    <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="1" stdDeviation="1.4" flood-color="#0f172a" flood-opacity="0.12" />
    </filter>
  </defs>
  <rect x="0" y="0" width="${width}" height="${height}" rx="16" fill="#f3f0e8" />
  <rect x="10" y="10" width="${width - 20}" height="${height - 20}" rx="12" fill="url(#paperShade)" stroke="#e7e2d7" filter="url(#softShadow)" />
  <g>${paperNoise}</g>
  <g>${guides}${leftMargin}</g>
  <text x="${paddingX}" y="${Math.max(22, paddingTop - 8)}" fill="#64748b" opacity="0.75" font-size="${Math.max(11, Math.round(fontSize * 0.38))}" font-family="ui-sans-serif, system-ui, sans-serif">${escapeXml(headerNote)}</text>
  <g>${glyphs}</g>
</svg>`;
}

export async function renderHandwritingPlaceholderPng(
  request: GenerateRequest
): Promise<string> {
  const svg = buildPlaceholderSvg(request);
  const pngBuffer = await sharp(Buffer.from(svg, "utf-8"))
    .png({
      compressionLevel: 9,
      adaptiveFiltering: true
    })
    .toBuffer();

  return `data:image/png;base64,${pngBuffer.toString("base64")}`;
}

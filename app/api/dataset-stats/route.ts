import { NextResponse } from "next/server";
import { access, readFile } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

const LETTERS = "abcdefghijklmnopqrstuvwxyz".split("");

async function fileExists(p: string) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function readMockDatasetStats(root: string) {
  const mockPath = path.join(root, "public", "mock", "dataset_stats.json");
  if (!(await fileExists(mockPath))) {
    return null;
  }
  try {
    return JSON.parse(await readFile(mockPath, "utf-8"));
  } catch {
    return null;
  }
}

export async function GET() {
  const root = process.cwd();
  const labelsPath = path.join(root, "out", "labels.csv");
  if (!(await fileExists(labelsPath))) {
    const mock = await readMockDatasetStats(root);
    return NextResponse.json(
      mock ?? {
        datasetSize: 0,
        undertrainedThreshold: 40,
        generatedAt: new Date().toISOString(),
        letters: LETTERS.map((letter) => ({ letter, count: 0 })),
        confusingPairs: [
          ["i", "l"],
          ["u", "v"],
          ["c", "e"]
        ]
      }
    );
  }

  const counts = Object.fromEntries(LETTERS.map((l) => [l, 0])) as Record<string, number>;
  try {
    const raw = await readFile(labelsPath, "utf-8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const parts = trimmed.split(",");
      const label = parts[1]?.trim()?.toLowerCase();
      if (label && label.length === 1 && counts[label] !== undefined) {
        counts[label] += 1;
      }
    }
  } catch {
    // fallback to zeros
  }

  const datasetSize = Object.values(counts).reduce((sum, n) => sum + n, 0);
  return NextResponse.json({
    datasetSize,
    undertrainedThreshold: 40,
    generatedAt: new Date().toISOString(),
    letters: LETTERS.map((letter) => ({ letter, count: counts[letter] })),
    confusingPairs: [
      ["i", "l"],
      ["u", "v"],
      ["a", "o"],
      ["c", "e"],
      ["m", "n"]
    ]
  });
}


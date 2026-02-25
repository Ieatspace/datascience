import { NextResponse } from "next/server";
import { access, readFile, stat } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

async function exists(p: string) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

function parseLatestEpoch(logText: string) {
  const lines = logText.split(/\r?\n/).filter(Boolean).slice(-400);
  let epoch = 0;
  let loss: number | null = null;
  let valLoss: number | null = null;
  for (const line of lines) {
    const m = line.match(/\[epoch\s+(\d+)\].*train\(loss=([0-9.]+).*val\(loss=([0-9.]+)/);
    if (m) {
      epoch = Number(m[1]);
      loss = Number(m[2]);
      valLoss = Number(m[3]);
    }
  }
  return { epoch, loss, valLoss };
}

async function countDatasetSize(labelsPath: string): Promise<number> {
  try {
    const raw = await readFile(labelsPath, "utf-8");
    return raw
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean).length;
  } catch {
    return 0;
  }
}

export async function GET() {
  const root = process.cwd();
  const trainLog = path.join(root, "out", "train.log");
  const weights = path.join(root, "out", "letter_gen.pt");
  const labels = path.join(root, "out", "labels.csv");
  const config = path.join(root, "out", "letter_gen_config.json");

  const [datasetSize, hasLog, hasWeights, hasConfig] = await Promise.all([
    countDatasetSize(labels),
    exists(trainLog),
    exists(weights),
    exists(config)
  ]);

  let status: "idle" | "training" | "error" = "idle";
  let lastTrained: string | null = null;
  let progress:
    | {
        epoch?: number;
        totalEpochs?: number | null;
        etaSeconds?: number | null;
        percent?: number | null;
        loss?: number | null;
        valLoss?: number | null;
      }
    | undefined;

  if (hasWeights) {
    try {
      lastTrained = (await stat(weights)).mtime.toISOString();
    } catch {
      lastTrained = null;
    }
  }

  if (hasLog) {
    try {
      const s = await stat(trainLog);
      const ageMs = Date.now() - s.mtimeMs;
      const logText = await readFile(trainLog, "utf-8");
      const latest = parseLatestEpoch(logText);
      progress = {
        epoch: latest.epoch || 0,
        totalEpochs: null,
        etaSeconds: null,
        percent: null,
        loss: latest.loss,
        valLoss: latest.valLoss
      };
      status = ageMs < 120_000 ? "training" : "idle";
    } catch {
      status = "error";
    }
  }

  let version = "letter-gen-v1";
  if (hasConfig) {
    try {
      const raw = JSON.parse(await readFile(config, "utf-8")) as { train?: { best_epoch?: number } };
      version = raw?.train?.best_epoch ? `letter-gen-v1:e${raw.train.best_epoch}` : version;
    } catch {
      // ignore
    }
  }

  return NextResponse.json({
    version,
    datasetSize,
    lastTrained,
    status,
    ...(progress ? { progress } : {})
  });
}


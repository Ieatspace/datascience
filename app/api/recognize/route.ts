import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";

import { NextResponse } from "next/server";

import {
  recognizePageRequestSchema,
  type RecognizePageRequest,
  type RecognizeResponse
} from "@/lib/types";

export const runtime = "nodejs";

const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp"]);

type PredictorJsonResult = {
  text: string;
  lines: Array<{
    index: number;
    text: string;
    chars: Array<{
      x: number;
      y: number;
      w: number;
      h: number;
      pred: string;
      conf: number;
      topk?: Array<{ label: string; conf: number }>;
    }>;
  }>;
  detectedCharacters: number;
  averageConfidence?: number | null;
  extractor: {
    gentle: boolean;
    dotted: boolean;
    softDotMerge: boolean;
  };
  debugImagePath?: string | null;
};

function parseBooleanField(value: FormDataEntryValue | null, defaultValue: boolean) {
  if (typeof value !== "string") {
    return defaultValue;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "1" || normalized === "yes") {
    return true;
  }
  if (normalized === "false" || normalized === "0" || normalized === "no") {
    return false;
  }
  return defaultValue;
}

function parseIntegerField(value: FormDataEntryValue | null, defaultValue: number) {
  if (typeof value !== "string") {
    return defaultValue;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return defaultValue;
  }
  return Math.floor(parsed);
}

function getPythonExecutable(): string {
  const configured = process.env.PYTHON_EXECUTABLE?.trim();
  return configured && configured.length > 0 ? configured : "python";
}

function getSafeExtension(fileName: string): string {
  const ext = path.extname(fileName).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext) ? ext : ".png";
}

async function runCommand(
  command: string,
  args: string[],
  cwd: string
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ stdout, stderr, code: code ?? -1 });
    });
  });
}

function buildPredictorArgs(
  imagePath: string,
  outDir: string,
  options: RecognizePageRequest
): string[] {
  const args = [
    "predict_page.py",
    imagePath,
    "--json",
    "--out-dir",
    outDir,
    "--topk",
    String(options.topk)
  ];

  if (!options.includeDebugImage) {
    args.push("--no-debug");
  }
  if (options.dotted) {
    args.push("--dotted");
  }
  if (options.strict) {
    args.push("--strict");
  }
  if (options.softDotMerge) {
    args.push("--soft-dot-merge");
  }
  if (options.noSpaces) {
    args.push("--no-spaces");
  }

  return args;
}

export async function POST(request: Request) {
  let tempRoot: string | null = null;

  try {
    const form = await request.formData();
    const fileEntry = form.get("file");

    if (!(fileEntry instanceof File)) {
      return NextResponse.json(
        {
          error: {
            message: "Missing image file"
          }
        },
        { status: 400 }
      );
    }

    if (fileEntry.size <= 0) {
      return NextResponse.json(
        {
          error: {
            message: "Uploaded file is empty"
          }
        },
        { status: 400 }
      );
    }

    if (fileEntry.size > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        {
          error: {
            message: "Uploaded file is too large",
            details: { maxBytes: MAX_UPLOAD_BYTES }
          }
        },
        { status: 413 }
      );
    }

    const parsedOptions = recognizePageRequestSchema.safeParse({
      dotted: parseBooleanField(form.get("dotted"), false),
      strict: parseBooleanField(form.get("strict"), false),
      softDotMerge: parseBooleanField(form.get("softDotMerge"), false),
      noSpaces: parseBooleanField(form.get("noSpaces"), false),
      includeDebugImage: parseBooleanField(form.get("includeDebugImage"), true),
      topk: parseIntegerField(form.get("topk"), 1)
    });

    if (!parsedOptions.success) {
      return NextResponse.json(
        {
          error: {
            message: "Invalid recognition options",
            details: parsedOptions.error.flatten()
          }
        },
        { status: 400 }
      );
    }

    const options = parsedOptions.data;
    tempRoot = await fs.mkdtemp(path.join(tmpdir(), "handwrite-recognize-"));
    const inputExt = getSafeExtension(fileEntry.name || "upload.png");
    const inputPath = path.join(tempRoot, `upload-${randomUUID()}${inputExt}`);
    const outDir = path.join(tempRoot, "predict-out");

    const bytes = Buffer.from(await fileEntry.arrayBuffer());
    await fs.writeFile(inputPath, bytes);
    await fs.mkdir(outDir, { recursive: true });

    const { stdout, stderr, code } = await runCommand(
      getPythonExecutable(),
      buildPredictorArgs(inputPath, outDir, options),
      process.cwd()
    );

    if (code !== 0) {
      return NextResponse.json(
        {
          error: {
            message: "Python recognizer failed",
            details: {
              code,
              stderr: stderr.trim() || null,
              stdout: stdout.trim() || null
            }
          }
        },
        { status: 500 }
      );
    }

    let predictor: PredictorJsonResult;
    try {
      predictor = JSON.parse(stdout) as PredictorJsonResult;
    } catch (error) {
      return NextResponse.json(
        {
          error: {
            message: "Recognizer returned invalid JSON",
            details: {
              parseError: error instanceof Error ? error.message : String(error),
              stdout: stdout.slice(0, 4000),
              stderr: stderr.slice(0, 4000)
            }
          }
        },
        { status: 502 }
      );
    }

    let debugImageDataUrl: string | null = null;
    if (options.includeDebugImage && predictor.debugImagePath) {
      try {
        const debugBytes = await fs.readFile(predictor.debugImagePath);
        debugImageDataUrl = `data:image/png;base64,${debugBytes.toString("base64")}`;
      } catch {
        debugImageDataUrl = null;
      }
    }

    const response: RecognizeResponse = {
      text: predictor.text,
      lines: predictor.lines,
      detectedCharacters: predictor.detectedCharacters,
      averageConfidence: predictor.averageConfidence ?? null,
      extractor: predictor.extractor,
      debugImageDataUrl,
      imageName: fileEntry.name || "upload",
      topk: options.topk
    };

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error("Failed to recognize handwriting page", error);
    return NextResponse.json(
      {
        error: {
          message: "Failed to recognize handwriting page"
        }
      },
      { status: 500 }
    );
  } finally {
    if (tempRoot) {
      await fs.rm(tempRoot, { recursive: true, force: true }).catch(() => {});
    }
  }
}

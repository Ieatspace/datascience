import {
  generateRequestSchema,
  generateResponseSchema,
  type GenerateRequest,
  type GenerateResponse
} from "@/lib/types";
import { renderHandwritingPlaceholderPng } from "@/lib/handwriting-stub";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { z } from "zod";

export type GeneratorBackend = "stub" | "fastapi" | "local";

export class UpstreamGenerateError extends Error {
  code: "UPSTREAM_HTTP" | "UPSTREAM_TIMEOUT" | "UPSTREAM_INVALID";
  status?: number;
  details?: unknown;

  constructor(
    message: string,
    options: {
      code: "UPSTREAM_HTTP" | "UPSTREAM_TIMEOUT" | "UPSTREAM_INVALID";
      status?: number;
      details?: unknown;
    }
  ) {
    super(message);
    this.name = "UpstreamGenerateError";
    this.code = options.code;
    this.status = options.status;
    this.details = options.details;
  }
}

const upstreamLooseGenerateResponseSchema = z.object({
  id: z.string().min(1).optional(),
  createdAt: z.string().min(1).optional(),
  request: generateRequestSchema.optional(),
  imageDataUrl: z.string().startsWith("data:image/png;base64,")
});

function envString(name: string): string | undefined {
  const value = process.env[name];
  if (!value) {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseBackend(value: string | undefined): GeneratorBackend | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === "stub") {
    return "stub";
  }
  if (normalized === "fastapi") {
    return "fastapi";
  }
  if (normalized === "local" || normalized === "python" || normalized === "local-python") {
    return "local";
  }
  return undefined;
}

export function getConfiguredGeneratorBackend(): GeneratorBackend {
  const explicit = parseBackend(envString("HANDWRITE_PROVIDER"));
  if (explicit) {
    return explicit;
  }

  return envString("FASTAPI_GENERATE_URL") ? "fastapi" : "local";
}

export function getFastApiGenerateUrl(): string {
  return envString("FASTAPI_GENERATE_URL") ?? "http://localhost:8000/generate";
}

export function getFastApiTimeoutMs(): number {
  const raw = envString("FASTAPI_TIMEOUT_MS");
  if (!raw) {
    return 20000;
  }

  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return 20000;
  }

  return Math.floor(value);
}

export function getLocalPythonExecutable(): string {
  return envString("PYTHON_EXECUTABLE") ?? "python";
}

export function getLocalPythonGeneratorScript(): string {
  return envString("HANDWRITE_LOCAL_SCRIPT") ?? "generate_handwriting_page.py";
}

export function getLocalLetterModelWeightsPath(): string {
  return envString("HANDWRITE_LOCAL_LETTER_MODEL_PATH") ?? path.join("out", "letter_gen.pt");
}

export function getLocalLetterModelAutoEnable(): boolean {
  const raw = envString("HANDWRITE_LOCAL_USE_LETTER_MODEL");
  if (!raw) {
    return true;
  }
  const normalized = raw.toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}

export function getLocalPythonTimeoutMs(): number {
  const raw = envString("HANDWRITE_LOCAL_TIMEOUT_MS");
  if (!raw) {
    return 45000;
  }

  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return 45000;
  }
  return Math.floor(value);
}

export function getLocalPythonFallbackToStub(): boolean {
  const raw = envString("HANDWRITE_LOCAL_FALLBACK_TO_STUB");
  if (!raw) {
    return false;
  }
  const normalized = raw.toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}

export function normalizeUpstreamGenerateResponse(
  payload: unknown,
  request: GenerateRequest
): GenerateResponse {
  const parsed = upstreamLooseGenerateResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new UpstreamGenerateError("Python backend returned an invalid response", {
      code: "UPSTREAM_INVALID",
      details: parsed.error.flatten()
    });
  }

  const normalized: GenerateResponse = {
    id: parsed.data.id ?? crypto.randomUUID(),
    createdAt: parsed.data.createdAt ?? new Date().toISOString(),
    request,
    imageDataUrl: parsed.data.imageDataUrl
  };

  const validated = generateResponseSchema.safeParse(normalized);
  if (!validated.success) {
    throw new UpstreamGenerateError("Normalized backend response failed validation", {
      code: "UPSTREAM_INVALID",
      details: validated.error.flatten()
    });
  }

  return validated.data;
}

async function safeParseJson(response: Response): Promise<unknown | null> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function runLocalProcess(
  command: string,
  args: string[],
  options?: { timeoutMs?: number; cwd?: string }
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options?.cwd ?? process.cwd(),
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";
    let settled = false;
    const timeoutMs = options?.timeoutMs ?? 45000;

    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill("SIGTERM");
      reject(
        new UpstreamGenerateError("Local Python handwriting generator timed out", {
          code: "UPSTREAM_TIMEOUT",
          details: { timeoutMs }
        })
      );
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve({ code: code ?? -1, stdout, stderr });
    });
  });
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function generateHandwritingViaFastApi(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const controller = new AbortController();
  const timeoutMs = getFastApiTimeoutMs();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(getFastApiGenerateUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(request),
      signal: controller.signal,
      cache: "no-store"
    });

    const payload = await safeParseJson(response);

    if (!response.ok) {
      let message = "Python backend request failed";
      if (
        payload &&
        typeof payload === "object" &&
        "detail" in payload &&
        typeof (payload as { detail?: unknown }).detail === "string"
      ) {
        message = (payload as { detail: string }).detail;
      } else if (
        payload &&
        typeof payload === "object" &&
        "error" in payload &&
        typeof (payload as { error?: unknown }).error === "string"
      ) {
        message = (payload as { error: string }).error;
      }

      throw new UpstreamGenerateError(message, {
        code: "UPSTREAM_HTTP",
        status: response.status,
        details: payload
      });
    }

    return normalizeUpstreamGenerateResponse(payload, request);
  } catch (error) {
    if (error instanceof UpstreamGenerateError) {
      throw error;
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new UpstreamGenerateError("Python backend request timed out", {
        code: "UPSTREAM_TIMEOUT",
        details: { timeoutMs }
      });
    }

    throw new UpstreamGenerateError("Failed to reach Python backend", {
      code: "UPSTREAM_HTTP",
      details: error instanceof Error ? error.message : error
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function generateHandwritingViaLocalPython(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const tempDir = await mkdtemp(path.join(tmpdir(), "handwrite-generate-"));
  const outPath = path.join(tempDir, `${randomUUID()}.png`);
  const debugJsonPath = path.join(tempDir, `${randomUUID()}.json`);
  const args = [
    getLocalPythonGeneratorScript(),
    "--text",
    request.text,
    "--style",
    request.style,
    "--width",
    String(request.width),
    "--height",
    String(request.height),
    "--line-spacing",
    String(request.lineSpacing),
    "--out",
    outPath,
    "--json",
    "--debug-json",
    debugJsonPath
  ];
  if (request.seed !== null) {
    args.push("--seed", String(request.seed));
  }
  // Use classifier scoring by default when the model exists. Allow opt-out for faster local iteration.
  if (envString("HANDWRITE_LOCAL_USE_CLASSIFIER") !== "0") {
    args.push("--use-classifier");
  }

  const letterModelEnabledByRequest = request.letterModelEnabled !== false;
  const letterModelWeightsPath = getLocalLetterModelWeightsPath();
  const shouldUseLetterModel =
    letterModelEnabledByRequest &&
    getLocalLetterModelAutoEnable() &&
    (await pathExists(letterModelWeightsPath));

  if (shouldUseLetterModel) {
    args.push("--use-letter-model", "--letter-model-weights", letterModelWeightsPath);

    if (typeof request.letterModelStyleStrength === "number") {
      args.push("--letter-style-strength", String(request.letterModelStyleStrength));
    }
    if (typeof request.letterModelBaselineJitter === "number") {
      args.push("--baseline-jitter", String(request.letterModelBaselineJitter));
    }
    if (typeof request.letterModelWordSlant === "number") {
      args.push("--word-slant", String(request.letterModelWordSlant));
    }
    if (typeof request.letterModelRotationJitter === "number") {
      args.push("--letter-rot-jitter", String(request.letterModelRotationJitter));
    }
    if (typeof request.letterModelInkVariation === "number") {
      args.push("--ink-variation", String(request.letterModelInkVariation));
    }
  }

  try {
    const result = await runLocalProcess(getLocalPythonExecutable(), args, {
      timeoutMs: getLocalPythonTimeoutMs(),
      cwd: process.cwd()
    });

    if (result.code !== 0) {
      throw new UpstreamGenerateError("Local Python handwriting generator failed", {
        code: "UPSTREAM_HTTP",
        details: {
          code: result.code,
          stdout: result.stdout.trim() || null,
          stderr: result.stderr.trim() || null
        }
      });
    }

    const png = await readFile(outPath);
    const imageDataUrl = `data:image/png;base64,${png.toString("base64")}`;

    const response: GenerateResponse = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      request: { ...request },
      imageDataUrl
    };

    return generateResponseSchema.parse(response);
  } catch (error) {
    if (error instanceof UpstreamGenerateError) {
      throw error;
    }
    throw new UpstreamGenerateError("Failed to reach local Python handwriting generator", {
      code: "UPSTREAM_HTTP",
      details: error instanceof Error ? error.message : error
    });
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

export async function generateHandwritingStub(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const imageDataUrl = await renderHandwritingPlaceholderPng(request);

  const response: GenerateResponse = {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    request: { ...request },
    imageDataUrl
  };

  return generateResponseSchema.parse(response);
}

export async function generateHandwriting(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const backend = getConfiguredGeneratorBackend();
  if (backend === "local") {
    try {
      return await generateHandwritingViaLocalPython(request);
    } catch (error) {
      if (getLocalPythonFallbackToStub()) {
        return generateHandwritingStub(request);
      }
      throw error;
    }
  }
  if (backend === "fastapi") {
    return generateHandwritingViaFastApi(request);
  }
  return generateHandwritingStub(request);
}

import {
  generateRequestSchema,
  generateResponseSchema,
  type GenerateRequest,
  type GenerateResponse
} from "@/lib/types";
import { renderHandwritingPlaceholderPng } from "@/lib/handwriting-stub";
import { z } from "zod";

export type GeneratorBackend = "stub" | "fastapi";

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
  return undefined;
}

export function getConfiguredGeneratorBackend(): GeneratorBackend {
  const explicit = parseBackend(envString("HANDWRITE_PROVIDER"));
  if (explicit) {
    return explicit;
  }

  return envString("FASTAPI_GENERATE_URL") ? "fastapi" : "stub";
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
  if (backend === "fastapi") {
    return generateHandwritingViaFastApi(request);
  }
  return generateHandwritingStub(request);
}

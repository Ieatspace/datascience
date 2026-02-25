import {
  apiErrorResponseSchema,
  datasetStatsSchema,
  generateResponseSchema,
  recognizeResponseSchema,
  trainingStatusSchema,
  type DatasetStats,
  type GenerateRequest,
  type GenerateResponse,
  type RecognizePageRequest,
  type RecognizeResponse,
  type TrainingStatus
} from "@/lib/types";

export class ApiClientError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.details = details;
  }
}

export async function postGenerate(
  request: GenerateRequest,
  options?: { signal?: AbortSignal }
): Promise<GenerateResponse> {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    cache: "no-store",
    signal: options?.signal,
    body: JSON.stringify(request)
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const parsedError = apiErrorResponseSchema.safeParse(payload);
    const message = parsedError.success
      ? parsedError.data.error.message
      : "Generation failed";

    throw new ApiClientError(message, response.status, parsedError.success ? parsedError.data.error.details : payload);
  }

  const parsed = generateResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiClientError(
      "API returned an unexpected response",
      response.status,
      parsed.error.flatten()
    );
  }

  return parsed.data;
}

export async function postRecognizePage(
  file: File,
  options: RecognizePageRequest
): Promise<RecognizeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("dotted", String(options.dotted));
  formData.append("strict", String(options.strict));
  formData.append("softDotMerge", String(options.softDotMerge));
  formData.append("noSpaces", String(options.noSpaces));
  formData.append("includeDebugImage", String(options.includeDebugImage));
  formData.append("topk", String(options.topk));

  const response = await fetch("/api/recognize", {
    method: "POST",
    cache: "no-store",
    body: formData
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const parsedError = apiErrorResponseSchema.safeParse(payload);
    const message = parsedError.success
      ? parsedError.data.error.message
      : "Recognition failed";

    throw new ApiClientError(
      message,
      response.status,
      parsedError.success ? parsedError.data.error.details : payload
    );
  }

  const parsed = recognizeResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiClientError(
      "API returned an unexpected recognition response",
      response.status,
      parsed.error.flatten()
    );
  }

  return parsed.data;
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const response = await fetch("/api/status", { cache: "no-store" });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiClientError("Failed to load training status", response.status, payload);
  }
  const parsed = trainingStatusSchema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiClientError("Invalid training status response", response.status, parsed.error.flatten());
  }
  return parsed.data;
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const response = await fetch("/api/dataset-stats", { cache: "no-store" });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiClientError("Failed to load dataset stats", response.status, payload);
  }
  const parsed = datasetStatsSchema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiClientError("Invalid dataset stats response", response.status, parsed.error.flatten());
  }
  return parsed.data;
}

export async function postTrainControl(action: "train" | "stop"): Promise<{ ok: boolean; message?: string }> {
  const response = await fetch(action === "train" ? "/api/train" : "/api/stop", {
    method: "POST",
    cache: "no-store"
  });
  const payload = await response.json().catch(() => null);
  return {
    ok: response.ok,
    message:
      payload && typeof payload === "object" && "message" in payload
        ? String((payload as { message?: unknown }).message)
        : response.ok
          ? "ok"
          : "backend required"
  };
}

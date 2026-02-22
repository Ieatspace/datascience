import {
  apiErrorResponseSchema,
  generateResponseSchema,
  type GenerateRequest,
  type GenerateResponse
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
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    cache: "no-store",
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

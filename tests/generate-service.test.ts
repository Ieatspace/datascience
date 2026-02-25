import { describe, expect, it } from "vitest";

import {
  generateHandwritingStub,
  getConfiguredGeneratorBackend,
  normalizeUpstreamGenerateResponse,
  UpstreamGenerateError
} from "@/lib/generate-service";
import {
  DEFAULT_GENERATE_REQUEST,
  generateRequestSchema,
  generateResponseSchema
} from "@/lib/types";

describe("generateRequestSchema", () => {
  it("rejects empty text after trimming", () => {
    const parsed = generateRequestSchema.safeParse({
      ...DEFAULT_GENERATE_REQUEST,
      text: "   "
    });

    expect(parsed.success).toBe(false);
  });

  it("rejects invalid dimensions", () => {
    const parsed = generateRequestSchema.safeParse({
      ...DEFAULT_GENERATE_REQUEST,
      width: 200,
      height: 9999
    });

    expect(parsed.success).toBe(false);
  });

  it("accepts optional letter-model tuning fields without breaking the base contract", () => {
    const parsed = generateRequestSchema.safeParse({
      ...DEFAULT_GENERATE_REQUEST,
      text: "hello world",
      letterModelEnabled: true,
      letterModelStyleStrength: 1.2,
      letterModelBaselineJitter: 0.9,
      letterModelWordSlant: 1.1,
      letterModelRotationJitter: 0.8,
      letterModelInkVariation: 0.2
    });

    expect(parsed.success).toBe(true);
  });
});

describe("generateHandwritingStub", () => {
  it("returns a response that matches the API schema", async () => {
    const response = await generateHandwritingStub({
      ...DEFAULT_GENERATE_REQUEST,
      text: "Hello from the API stub"
    });

    expect(generateResponseSchema.safeParse(response).success).toBe(true);
    expect(response.imageDataUrl.startsWith("data:image/png;base64,")).toBe(true);
  });

  it("produces a randomized image for repeated requests", async () => {
    const request = {
      ...DEFAULT_GENERATE_REQUEST,
      text: "Deterministic test",
      style: "pencil" as const
    };

    const first = await generateHandwritingStub(request);
    const second = await generateHandwritingStub(request);

    expect(first.id).not.toEqual(second.id);
    expect(first.imageDataUrl).not.toEqual(second.imageDataUrl);
  });
});

describe("FastAPI integration helpers", () => {
  it("normalizes a minimal upstream payload to the public response contract", () => {
    const request = {
      ...DEFAULT_GENERATE_REQUEST,
      text: "Proxy mode"
    };

    const response = normalizeUpstreamGenerateResponse(
      {
        imageDataUrl: "data:image/png;base64,AAAA"
      },
      request
    );

    expect(response.request).toEqual(request);
    expect(response.imageDataUrl).toBe("data:image/png;base64,AAAA");
    expect(typeof response.id).toBe("string");
    expect(Number.isNaN(Date.parse(response.createdAt))).toBe(false);
  });

  it("throws when upstream payload does not match the required image shape", () => {
    expect(() =>
      normalizeUpstreamGenerateResponse(
        {
          imageDataUrl: "data:image/svg+xml;base64,AAAA"
        },
        DEFAULT_GENERATE_REQUEST
      )
    ).toThrow(UpstreamGenerateError);
  });

  it("defaults to local backend when no proxy env vars are set", () => {
    const previousProvider = process.env.HANDWRITE_PROVIDER;
    const previousUrl = process.env.FASTAPI_GENERATE_URL;

    delete process.env.HANDWRITE_PROVIDER;
    delete process.env.FASTAPI_GENERATE_URL;

    expect(getConfiguredGeneratorBackend()).toBe("local");

    if (previousProvider === undefined) {
      delete process.env.HANDWRITE_PROVIDER;
    } else {
      process.env.HANDWRITE_PROVIDER = previousProvider;
    }

    if (previousUrl === undefined) {
      delete process.env.FASTAPI_GENERATE_URL;
    } else {
      process.env.FASTAPI_GENERATE_URL = previousUrl;
    }
  });

  it("selects fastapi backend when FASTAPI_GENERATE_URL is configured", () => {
    const previousProvider = process.env.HANDWRITE_PROVIDER;
    const previousUrl = process.env.FASTAPI_GENERATE_URL;

    delete process.env.HANDWRITE_PROVIDER;
    process.env.FASTAPI_GENERATE_URL = "http://localhost:8000/generate";

    expect(getConfiguredGeneratorBackend()).toBe("fastapi");

    if (previousProvider === undefined) {
      delete process.env.HANDWRITE_PROVIDER;
    } else {
      process.env.HANDWRITE_PROVIDER = previousProvider;
    }

    if (previousUrl === undefined) {
      delete process.env.FASTAPI_GENERATE_URL;
    } else {
      process.env.FASTAPI_GENERATE_URL = previousUrl;
    }
  });
});

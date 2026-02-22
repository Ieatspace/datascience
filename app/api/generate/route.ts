import { NextResponse } from "next/server";

import {
  generateHandwriting,
  getConfiguredGeneratorBackend,
  UpstreamGenerateError
} from "@/lib/generate-service";
import { generateRequestSchema } from "@/lib/types";

export const runtime = "nodejs";

export async function POST(request: Request) {
  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      {
        error: {
          message: "Invalid JSON payload"
        }
      },
      { status: 400 }
    );
  }

  const parsed = generateRequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: {
          message: "Invalid generation request",
          details: parsed.error.flatten()
        }
      },
      { status: 400 }
    );
  }

  try {
    const result = await generateHandwriting(parsed.data);
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    if (error instanceof UpstreamGenerateError) {
      const isTimeout = error.code === "UPSTREAM_TIMEOUT";
      const status = isTimeout ? 504 : 502;
      const message = isTimeout
        ? "Python backend timed out while generating image"
        : "Python backend generation failed";

      console.error("Failed to generate handwriting via Python backend", {
        code: error.code,
        status: error.status,
        details: error.details
      });

      return NextResponse.json(
        {
          error: {
            message,
            details: {
              upstreamStatus: error.status,
              code: error.code,
              ...(error.details !== undefined ? { upstream: error.details } : {})
            }
          }
        },
        { status }
      );
    }

    console.error("Failed to generate handwriting image", {
      backend: getConfiguredGeneratorBackend(),
      error
    });

    return NextResponse.json(
      {
        error: {
          message: "Failed to generate handwriting image"
        }
      },
      { status: 500 }
    );
  }
}

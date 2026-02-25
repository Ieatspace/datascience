import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST() {
  return NextResponse.json(
    {
      ok: false,
      status: "disabled",
      message: "Backend stop control is not connected in this build. Stop training manually with Ctrl+C or Stop-Process."
    },
    { status: 501 }
  );
}


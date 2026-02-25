import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST() {
  return NextResponse.json(
    {
      ok: false,
      status: "disabled",
      message: "Backend training control is not connected in this build. Use scripts/train_infinite.ps1 or python -m python_ai.lettergen.train."
    },
    { status: 501 }
  );
}


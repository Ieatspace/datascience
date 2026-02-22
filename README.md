# Handwrite Studio (Web Shell)

Production-ready Next.js UI shell for a handwriting-generation project. The current `/api/generate` implementation is a deterministic placeholder stub that renders a handwritten-ish PNG server-side (SVG -> PNG via `sharp`), but the API contract and component structure are designed so you can swap in a real Python model later with minimal changes.

## Features

- Next.js App Router + TypeScript + Tailwind CSS
- shadcn/ui-style components (`button`, `card`, `input`, `textarea`, `tabs`, `select`, `slider`, `toast`)
- Typed request/response contract with `zod`
- Server stub image generation (`POST /api/generate`) returning PNG data URLs
- Responsive two-column UI (editor/settings + preview/history)
- Loading skeleton, inline error state, toast errors
- Local history (last 10 generations) stored in `localStorage`
- Keyboard shortcut: `Ctrl/Cmd + Enter`
- Unit tests for validation + response shape (Vitest)

## Project Structure

- `app/page.tsx` - main page / shell layout entry
- `app/api/generate/route.ts` - API endpoint with request validation
- `components/TopNav.tsx`
- `components/EditorPanel.tsx`
- `components/SettingsPanel.tsx`
- `components/PreviewPanel.tsx`
- `components/HistoryStrip.tsx`
- `components/StudioClient.tsx` - client state orchestration
- `lib/types.ts` - shared API types + zod schemas
- `lib/api.ts` - typed client wrapper for `/api/generate`
- `lib/storage.ts` - localStorage history helpers
- `lib/generate-service.ts` - server-side generation service wrapper
- `lib/handwriting-stub.ts` - deterministic placeholder renderer

## Requirements

- Node.js `18.17+` (Node 20+ recommended)
- npm `9+`

## Run Locally

1. Install dependencies:

```bash
npm install
```

2. Start the dev server:

```bash
npm run dev
```

3. Open:

```text
http://localhost:3000
```

### Optional: Enable FastAPI Proxy Mode

The app now supports an env-driven Python backend proxy while keeping the same
`POST /api/generate` contract for the frontend.

1. Copy `.env.example` to `.env.local`
2. Set one of these:
   - `HANDWRITE_PROVIDER=fastapi`
   - or just set `FASTAPI_GENERATE_URL` (auto-enables fastapi mode)
3. Start your FastAPI server (example: `http://localhost:8000/generate`)
4. Run `npm run dev`

If the proxy env vars are not configured, the app uses the built-in stub renderer.

## Quality Checks

Run lint:

```bash
npm run lint
```

Run tests:

```bash
npm run test
```

Format code:

```bash
npm run format
```

## API Contract

### Request

`POST /api/generate`

```json
{
  "text": "Hello world",
  "style": "ink",
  "width": 1024,
  "height": 512,
  "lineSpacing": 1.35,
  "seed": 42
}
```

### Response

```json
{
  "id": "uuid",
  "createdAt": "2026-02-22T12:00:00.000Z",
  "request": {
    "text": "Hello world",
    "style": "ink",
    "width": 1024,
    "height": 512,
    "lineSpacing": 1.35,
    "seed": 42
  },
  "imageDataUrl": "data:image/png;base64,..."
}
```

## Example curl Request

```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from curl","style":"ink","width":1024,"height":512,"lineSpacing":1.35,"seed":42}'
```

PowerShell-friendly alternative:

```powershell
$body = @{
  text = "Hello from PowerShell"
  style = "ink"
  width = 1024
  height = 512
  lineSpacing = 1.35
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:3000/api/generate -Method Post -ContentType "application/json" -Body $body
```

## How To Replace The Stub With A Python Model Later

The UI and API contract should stay the same. Replace only the internals of the server-side generation path.

### Option A: Proxy to a FastAPI server (`localhost:8000`)

Recommended for most teams.

This is now implemented behind env flags.

Current behavior:

1. `app/api/generate/route.ts` validates the request with `zod`.
2. `lib/generate-service.ts` chooses backend:
   - `stub` (default)
   - `fastapi` (when `HANDWRITE_PROVIDER=fastapi` or `FASTAPI_GENERATE_URL` is set)
3. In `fastapi` mode, Next.js forwards the request to `FASTAPI_GENERATE_URL`.
4. The upstream response is normalized + validated before returning to the client.
5. If upstream omits metadata, Next.js fills `id`, `createdAt`, and echoes `request`.

Why this is good:

- Python model runtime stays isolated from the web runtime
- Easy to scale/replace independently
- Clear logging and failure boundaries

High-level route change:

- `route.ts` already acts as a pass-through adapter (validation + error translation + response normalization)

Expected FastAPI endpoint behavior (recommended):

- Endpoint: `POST /generate`
- Accepts the same JSON request schema used by the Next.js client
- Returns either:
  - full public response contract, or
  - at minimum `{ "imageDataUrl": "data:image/png;base64,..." }`

Minimal FastAPI response example:

```json
{
  "imageDataUrl": "data:image/png;base64,..."
}
```

### Option B: Run Python from a worker/process in the Next.js backend

Use when you want a single service boundary initially and can tolerate more ops complexity inside one app.

High-level approach:

1. Keep `route.ts` validation unchanged.
2. Move generation implementation from `lib/handwriting-stub.ts` to a worker-backed service.
3. Use a queue + child process (or worker thread wrapper that calls Python) to execute model inference.
4. Return PNG bytes -> convert to base64 data URL in `lib/generate-service.ts`.

Important notes:

- Avoid spawning a fresh Python process per request in production if latency matters.
- Prefer a long-lived worker pool and backpressure.
- Keep request/response schema validation in Next.js even when Python is local.

## Notes

- Current stub is intentionally deterministic-ish:
  - same request + seed -> same image output
  - different seed/settings -> slightly different jittered handwriting placeholder
- The placeholder is clearly a stub and is not ML-generated handwriting.

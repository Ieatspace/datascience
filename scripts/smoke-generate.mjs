const endpoint = process.argv[2] ?? "http://localhost:3000/api/generate";

const payload = {
  text: "hello world",
  style: "ink",
  width: 960,
  height: 360,
  lineSpacing: 1.25
};

const response = await fetch(endpoint, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});

let body = null;
try {
  body = await response.json();
} catch {
  body = null;
}

if (!response.ok) {
  console.error("[smoke] /api/generate failed", response.status, body);
  process.exit(1);
}

if (!body || typeof body.imageDataUrl !== "string") {
  console.error("[smoke] response missing imageDataUrl", body);
  process.exit(1);
}

if (!body.imageDataUrl.startsWith("data:image/png;base64,")) {
  console.error("[smoke] response is not a PNG data URL", body.imageDataUrl.slice(0, 64));
  process.exit(1);
}

console.log("[smoke] ok:", endpoint);
console.log("[smoke] id:", body.id);
console.log("[smoke] imageDataUrl prefix:", body.imageDataUrl.slice(0, 32));

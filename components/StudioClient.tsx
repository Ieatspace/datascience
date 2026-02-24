"use client";

import { useEffect, useState } from "react";

import { EditorPanel } from "@/components/EditorPanel";
import { HistoryStrip } from "@/components/HistoryStrip";
import { PreviewPanel } from "@/components/PreviewPanel";
import { RecognizePanel } from "@/components/RecognizePanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { TopNav } from "@/components/TopNav";
import { toast } from "@/components/ui/use-toast";
import { postGenerate } from "@/lib/api";
import {
  DEFAULT_GENERATE_REQUEST,
  MAX_TEXT_LENGTH,
  type GenerateRequest,
  type GenerateResponse,
  type HandwritingStyle
} from "@/lib/types";
import {
  clearGenerationHistory,
  loadGenerationHistory,
  pushGenerationHistoryItem,
  saveGenerationHistory
} from "@/lib/storage";

function isApiLikeError(error: unknown): error is { message: string } {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message: string }).message === "string"
  );
}

async function copyDataUrlToClipboard(dataUrl: string): Promise<"image" | "text"> {
  if (typeof navigator === "undefined" || !navigator.clipboard) {
    throw new Error("Clipboard access is not available in this browser");
  }

  const response = await fetch(dataUrl);
  const blob = await response.blob();

  const clipboardItemCtor = (window as Window & { ClipboardItem?: typeof ClipboardItem })
    .ClipboardItem;

  if (clipboardItemCtor && navigator.clipboard.write) {
    await navigator.clipboard.write([
      new clipboardItemCtor({
        [blob.type]: blob
      })
    ]);
    return "image";
  }

  await navigator.clipboard.writeText(dataUrl);
  return "text";
}

function downloadDataUrl(dataUrl: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function StudioClient() {
  const [request, setRequest] = useState<GenerateRequest>(DEFAULT_GENERATE_REQUEST);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [history, setHistory] = useState<GenerateResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wordCount = request.text.trim() ? request.text.trim().split(/\s+/).length : 0;
  const lineCount = request.text.length === 0 ? 0 : request.text.split(/\r?\n/).length;

  useEffect(() => {
    const stored = loadGenerationHistory();
    setHistory(stored);
    if (stored[0]) {
      setResult(stored[0]);
      setRequest(stored[0].request);
    }
  }, []);

  function updateRequest<K extends keyof GenerateRequest>(
    key: K,
    value: GenerateRequest[K]
  ) {
    setRequest((prev) => ({ ...prev, [key]: value }));
  }

  async function handleGenerate() {
    if (isLoading) {
      return;
    }

    if (request.text.trim().length === 0) {
      const message = "Please enter text before generating.";
      setError(message);
      toast({
        title: "Missing text",
        description: message,
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const nextResult = await postGenerate(request);
      setResult(nextResult);
      setHistory((prev) => {
        const next = pushGenerationHistoryItem(prev, nextResult);
        saveGenerationHistory(next);
        return next;
      });
    } catch (caught) {
      const message = isApiLikeError(caught)
        ? caught.message
        : "Unexpected error while generating preview";
      setError(message);
      toast({
        title: "Generation failed",
        description: message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  }

  function handleHistorySelect(item: GenerateResponse) {
    setResult(item);
    setRequest(item.request);
    setError(null);
  }

  function handleDownload() {
    if (!result) {
      return;
    }

    downloadDataUrl(result.imageDataUrl, `handwrite-${result.id}.png`);
  }

  async function handleCopy() {
    if (!result) {
      return;
    }

    try {
      const copied = await copyDataUrlToClipboard(result.imageDataUrl);
      toast({
        title: copied === "image" ? "Copied image" : "Copied data URL",
        description:
          copied === "image"
            ? "Preview image copied to your clipboard."
            : "Clipboard image write is unavailable, so the data URL was copied instead."
      });
    } catch (caught) {
      const message = isApiLikeError(caught)
        ? caught.message
        : "Unable to copy preview";
      toast({
        title: "Copy failed",
        description: message,
        variant: "destructive"
      });
    }
  }

  function handleClearHistory() {
    setHistory([]);
    clearGenerationHistory();
    toast({
      title: "History cleared",
      description: "Local generation history has been removed from this browser."
    });
  }

  return (
    <div className="space-y-6">
      <TopNav
        isLoading={isLoading}
        historyCount={history.length}
        hasResult={Boolean(result)}
      />

      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-white/70 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
          <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
            Draft Length
          </p>
          <p className="mt-1 text-lg font-semibold text-foreground tabular-nums">
            {request.text.length}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {wordCount} words | {lineCount} line{lineCount === 1 ? "" : "s"}
          </p>
        </div>

        <div className="rounded-xl border border-white/70 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
          <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
            Canvas
          </p>
          <p className="mt-1 text-lg font-semibold text-foreground tabular-nums">
            {request.width} x {request.height}
          </p>
          <p className="mt-1 text-xs text-muted-foreground capitalize">
            {request.style} | line spacing {request.lineSpacing.toFixed(2)}
          </p>
        </div>

        <div className="rounded-xl border border-white/70 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
          <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
            Session
          </p>
          <p className="mt-1 text-lg font-semibold text-foreground">
            {result ? "Active preview" : "Create your first preview"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {history.length > 0
              ? `${history.length} saved locally in browser history`
              : "Generate to start your local history strip"}
          </p>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="space-y-6">
          <EditorPanel
            text={request.text}
            maxLength={MAX_TEXT_LENGTH}
            isLoading={isLoading}
            error={error}
            onTextChange={(value) => updateRequest("text", value)}
            onGenerate={handleGenerate}
          />

          <SettingsPanel
            request={request}
            disabled={isLoading}
            onStyleChange={(value: HandwritingStyle) => updateRequest("style", value)}
            onSizeChange={(width, height) => {
              updateRequest("width", width);
              updateRequest("height", height);
            }}
            onLineSpacingChange={(value) => updateRequest("lineSpacing", value)}
            onSeedChange={(value) => updateRequest("seed", value)}
          />
        </div>

        <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
          <PreviewPanel
            result={result}
            draftRequest={request}
            isLoading={isLoading}
            error={error}
            onDownload={handleDownload}
            onCopy={handleCopy}
          />

          <HistoryStrip
            history={history}
            selectedId={result?.id}
            onSelect={handleHistorySelect}
            onClear={handleClearHistory}
          />
        </div>
      </div>

      <RecognizePanel />
    </div>
  );
}

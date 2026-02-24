"use client";

import Image from "next/image";
import { Copy, ImageIcon, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/use-toast";
import { postRecognizePage } from "@/lib/api";
import {
  recognizePageRequestSchema,
  type RecognizePageRequest,
  type RecognizeResponse
} from "@/lib/types";

const DEFAULT_RECOGNIZE_OPTIONS: RecognizePageRequest = recognizePageRequestSchema.parse({
  dotted: false,
  strict: false,
  softDotMerge: false,
  noSpaces: false,
  includeDebugImage: true,
  topk: 1
});

export function RecognizePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [options, setOptions] = useState<RecognizePageRequest>(DEFAULT_RECOGNIZE_OPTIONS);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecognizeResponse | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return null;
      });
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }
      return nextUrl;
    });

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [file]);

  function updateOption<K extends keyof RecognizePageRequest>(
    key: K,
    value: RecognizePageRequest[K]
  ) {
    setOptions((prev) => ({ ...prev, [key]: value }));
  }

  async function handleRecognize() {
    if (!file || isLoading) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const nextResult = await postRecognizePage(file, options);
      setResult(nextResult);
      toast({
        title: "Recognition complete",
        description: `Detected ${nextResult.detectedCharacters} characters across ${nextResult.lines.length} lines.`
      });
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : "Unexpected error while recognizing page";
      setError(message);
      toast({
        title: "Recognition failed",
        description: message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCopyText() {
    if (!result?.text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(result.text);
      toast({
        title: "Copied text",
        description: "Recognized text copied to clipboard."
      });
    } catch (caught) {
      toast({
        title: "Copy failed",
        description: caught instanceof Error ? caught.message : "Clipboard unavailable",
        variant: "destructive"
      });
    }
  }

  return (
    <Card className="card-accent overflow-hidden bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4 text-muted-foreground" />
              Page Recognition (OCR)
            </CardTitle>
            <CardDescription>
              Upload a handwritten practice page and run your local Python model +
              extractor to recover line-by-line text.
            </CardDescription>
          </div>
          <span className="section-chip bg-white/80">
            {isLoading ? "Recognizing..." : result ? "Ready" : "Local model"}
          </span>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="recognize-file">Handwriting Page Image</Label>
              <Input
                id="recognize-file"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp,image/*"
                onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null);
                  setError(null);
                }}
                disabled={isLoading}
              />
              <p className="text-xs text-muted-foreground">
                JPEG and PNG both work. Higher quality scans usually segment better.
              </p>
            </div>

            <div className="grid gap-3 rounded-xl border border-border/70 bg-background/50 p-4 sm:grid-cols-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border"
                  checked={!options.strict}
                  onChange={(event) => updateOption("strict", !event.target.checked)}
                  disabled={isLoading}
                />
                Gentle extraction
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border"
                  checked={options.dotted}
                  onChange={(event) => updateOption("dotted", event.target.checked)}
                  disabled={isLoading}
                />
                Dotted letters (i/j)
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border"
                  checked={options.softDotMerge}
                  onChange={(event) => updateOption("softDotMerge", event.target.checked)}
                  disabled={isLoading}
                />
                Soft dot merge
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border"
                  checked={!options.noSpaces}
                  onChange={(event) => updateOption("noSpaces", !event.target.checked)}
                  disabled={isLoading}
                />
                Insert spaces
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border"
                  checked={options.includeDebugImage}
                  onChange={(event) =>
                    updateOption("includeDebugImage", event.target.checked)
                  }
                  disabled={isLoading}
                />
                Debug overlay image
              </label>
              <div className="flex items-center gap-2 text-sm">
                <Label htmlFor="recognize-topk" className="min-w-12 text-sm">
                  Top-K
                </Label>
                <Input
                  id="recognize-topk"
                  type="number"
                  min={1}
                  max={5}
                  step={1}
                  value={options.topk}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    updateOption(
                      "topk",
                      Number.isFinite(next) ? Math.max(1, Math.min(5, Math.floor(next))) : 1
                    );
                  }}
                  disabled={isLoading}
                  className="h-9"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleRecognize} disabled={!file || isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Recognizing...
                  </>
                ) : (
                  "Recognize Page"
                )}
              </Button>
              {result ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleCopyText}
                  disabled={isLoading || result.text.length === 0}
                >
                  <Copy className="mr-2 h-4 w-4" />
                  Copy Text
                </Button>
              ) : null}
            </div>

            <div
              className={`rounded-lg border px-3 py-2 text-sm ${
                error
                  ? "border-destructive/30 bg-destructive/5 text-destructive"
                  : "border-border/60 bg-secondary/40 text-muted-foreground"
              }`}
              role={error ? "alert" : "status"}
            >
              {error
                ? error
                : result
                  ? `Recognized ${result.detectedCharacters} chars across ${result.lines.length} lines.`
                  : "Upload a page image, then run local recognition through the trained classifier."}
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-border/70 bg-background/50 p-3">
                <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                  Input Preview
                </p>
                <div className="mt-2 relative aspect-[4/3] overflow-hidden rounded-lg border border-border/60 bg-white">
                  {previewUrl ? (
                    <Image
                      src={previewUrl}
                      alt="Uploaded page preview"
                      fill
                      unoptimized
                      className="object-contain"
                    />
                  ) : (
                    <div className="absolute inset-0 grid place-items-center text-xs text-muted-foreground">
                      No image selected
                    </div>
                  )}
                </div>
                {file ? (
                  <p className="mt-2 truncate text-xs text-muted-foreground">
                    {file.name} ({Math.round(file.size / 1024)} KB)
                  </p>
                ) : null}
              </div>

              <div className="rounded-xl border border-border/70 bg-background/50 p-3">
                <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                  Recognition Stats
                </p>
                <div className="mt-2 space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Characters</span>
                    <span className="font-medium">
                      {result ? result.detectedCharacters : "—"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Lines</span>
                    <span className="font-medium">{result ? result.lines.length : "—"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Avg confidence</span>
                    <span className="font-medium">
                      {result?.averageConfidence != null
                        ? result.averageConfidence.toFixed(3)
                        : "—"}
                    </span>
                  </div>
                  <div className="pt-1 text-xs text-muted-foreground">
                    Extractor:{" "}
                    {result
                      ? `${result.extractor.gentle ? "gentle" : "strict"} | ${result.extractor.dotted ? "dotted" : "normal"}`
                      : "—"}
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="recognized-text">Recognized text</Label>
              <Textarea
                id="recognized-text"
                readOnly
                value={result?.text ?? ""}
                placeholder="Recognition output will appear here..."
                className="min-h-[180px] resize-y bg-white/90 font-mono text-sm leading-6"
              />
            </div>

            {result?.debugImageDataUrl ? (
              <div className="rounded-xl border border-border/70 bg-background/50 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                    Debug Overlay
                  </p>
                  <span className="text-xs text-muted-foreground">
                    boxes + predictions
                  </span>
                </div>
                <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-border/60 bg-white">
                  <Image
                    src={result.debugImageDataUrl}
                    alt="Recognition debug overlay"
                    fill
                    unoptimized
                    className="object-contain"
                  />
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

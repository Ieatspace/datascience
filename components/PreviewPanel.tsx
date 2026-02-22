"use client";

import Image from "next/image";
import { Copy, Download, FileJson2, ImageIcon, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { GenerateRequest, GenerateResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type PreviewPanelProps = {
  result: GenerateResponse | null;
  draftRequest: GenerateRequest;
  isLoading: boolean;
  error: string | null;
  onDownload: () => void;
  onCopy: () => void | Promise<void>;
};

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value));
}

export function PreviewPanel({
  result,
  draftRequest,
  isLoading,
  error,
  onDownload,
  onCopy
}: PreviewPanelProps) {
  const previewRequest = result?.request ?? draftRequest;
  const debugPayload = result
    ? JSON.stringify(result, null, 2)
    : JSON.stringify({ request: draftRequest }, null, 2);
  const hasImage = Boolean(result?.imageDataUrl);

  return (
    <Card className="card-accent overflow-hidden bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              Preview
            </CardTitle>
            <CardDescription>
              Server-rendered placeholder PNG. Replace the stub in{" "}
              <code>app/api/generate/route.ts</code> later with your model call.
            </CardDescription>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="section-chip bg-white/80">
              {isLoading ? "Rendering..." : hasImage ? "Ready to export" : "Awaiting render"}
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onCopy}
              disabled={!hasImage || isLoading}
            >
              <Copy className="h-4 w-4" />
              Copy
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onDownload}
              disabled={!hasImage || isLoading}
            >
              <Download className="h-4 w-4" />
              Download
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <Tabs defaultValue="preview" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="preview">
              <ImageIcon className="mr-2 h-4 w-4" />
              Preview
            </TabsTrigger>
            <TabsTrigger value="debug">
              <FileJson2 className="mr-2 h-4 w-4" />
              Debug Payload
            </TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span className="rounded-full border border-border/70 bg-background/70 px-2 py-1 tabular-nums">
                {previewRequest.width} x {previewRequest.height}
              </span>
              <span className="rounded-full border border-border/70 bg-background/70 px-2 py-1 capitalize">
                {previewRequest.style}
              </span>
              <span className="rounded-full border border-border/70 bg-background/70 px-2 py-1">
                Line spacing {previewRequest.lineSpacing.toFixed(2)}
              </span>
              {result ? (
                <span className="rounded-full border border-border/70 bg-background/70 px-2 py-1">
                  {formatTimestamp(result.createdAt)}
                </span>
              ) : null}
              {result ? (
                <span className="rounded-full border border-border/70 bg-background/70 px-2 py-1 font-mono">
                  #{result.id.slice(0, 8)}
                </span>
              ) : null}
            </div>

            <div className="relative overflow-hidden rounded-xl border border-border/70 bg-gradient-to-b from-slate-50 to-slate-100/70 p-3">
              <div className="absolute inset-0 subtle-grid opacity-40" />
              <div
                className="relative mx-auto w-full max-w-full overflow-hidden rounded-lg border border-border/70 bg-white shadow-sm"
                style={{
                  aspectRatio: `${previewRequest.width} / ${previewRequest.height}`
                }}
              >
                {result ? (
                  <Image
                    src={result.imageDataUrl}
                    alt="Generated handwritten preview"
                    width={result.request.width}
                    height={result.request.height}
                    unoptimized
                    className={cn(
                      "h-full w-full object-contain transition-opacity",
                      isLoading && "opacity-25"
                    )}
                  />
                ) : null}

                {isLoading ? (
                  <div className="absolute inset-0 p-4">
                    <div className="grid h-full gap-3">
                      <Skeleton className="h-4 w-40" />
                      <Skeleton className="h-4 w-3/4" />
                      <Skeleton className="h-4 w-4/5" />
                      <Skeleton className="mt-4 h-full w-full rounded-lg" />
                    </div>
                  </div>
                ) : null}

                {!result && !isLoading ? (
                  <div className="absolute inset-0 grid place-items-center p-6">
                    <div className="max-w-sm text-center text-sm text-muted-foreground">
                      <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-full border border-border/70 bg-background/80 shadow-sm">
                        <ImageIcon className="h-5 w-5" />
                      </div>
                      <p className="font-medium text-foreground">
                        Generate a sample preview
                      </p>
                      <p className="mt-1">
                        The placeholder renderer simulates paper texture and letter
                        jitter so the layout behaves like a real handwriting model.
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            {error ? (
              <p className="text-sm text-destructive" role="alert">
                {error}
              </p>
            ) : null}
          </TabsContent>

          <TabsContent value="debug">
            <div className="rounded-xl border border-border/70 bg-slate-950 shadow-inner">
              <div className="flex items-center justify-between border-b border-white/10 px-4 py-2 text-xs text-slate-300">
                <span>Response payload</span>
                <span className="font-mono">
                  {result ? `${result.imageDataUrl.length.toLocaleString()} chars` : "draft"}
                </span>
              </div>
              <pre className="max-h-[420px] overflow-auto p-4 text-xs leading-relaxed text-slate-100">
              {debugPayload}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

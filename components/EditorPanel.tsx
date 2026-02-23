"use client";

import { Eraser, Loader2, Sparkles, Type } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const PROMPT_SUGGESTIONS = [
  "Meet me by the library at 5:30 pm.",
  "Sincerely,\nAashif",
  "The rain tapped softly against the window as the tea cooled."
] as const;

type EditorPanelProps = {
  text: string;
  maxLength: number;
  isLoading: boolean;
  error: string | null;
  onTextChange: (value: string) => void;
  onGenerate: () => void;
};

export function EditorPanel({
  text,
  maxLength,
  isLoading,
  error,
  onTextChange,
  onGenerate
}: EditorPanelProps) {
  const trimmedEmpty = text.trim().length === 0;
  const charCount = text.length;
  const usagePct = Math.min(100, (charCount / maxLength) * 100);

  return (
    <Card className="card-accent overflow-hidden bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Type className="h-4 w-4 text-muted-foreground" />
              Text Prompt
            </CardTitle>
          </div>
          <span className="section-chip hidden sm:inline-flex">Input</span>
        </div>
        <CardDescription>
          Type the text to render as handwritten output. Use{" "}
          <span className="font-medium text-foreground">Ctrl + Enter</span> to
          generate.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="handwrite-text">Text</Label>
            <div className="text-xs text-muted-foreground">
              Multiline supported
            </div>
          </div>
          <Textarea
            id="handwrite-text"
            value={text}
            onChange={(event) => onTextChange(event.target.value)}
            maxLength={maxLength}
            placeholder="Write a note, quote, signature line, or anything you want to preview..."
            className="min-h-[240px] resize-y bg-white/90 leading-6"
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                if (!isLoading && !trimmedEmpty) {
                  onGenerate();
                }
              }
            }}
          />
          <div className="grid gap-2 pt-1 sm:grid-cols-3">
            {PROMPT_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => onTextChange(suggestion)}
                className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-left text-xs text-muted-foreground transition hover:bg-background hover:text-foreground"
              >
                <span className="line-clamp-2">{suggestion}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border/70 bg-background/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <div className="text-muted-foreground">
              <span className="font-medium text-foreground">{charCount}</span> /{" "}
              {maxLength} characters
            </div>
            <div className="text-muted-foreground tabular-nums">
              {usagePct.toFixed(0)}%
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className={cn(
                "h-full rounded-full bg-primary transition-all",
                charCount > maxLength * 0.9 && "bg-amber-500"
              )}
              style={{ width: `${usagePct}%` }}
              aria-hidden="true"
            />
          </div>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <span className="section-chip">Ctrl + Enter</span>
            <span className="text-xs text-muted-foreground">
              (Cmd + Enter on Mac)
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => onTextChange("")}
              disabled={isLoading || charCount === 0}
              className="h-8 px-2 text-muted-foreground hover:text-foreground"
            >
              <Eraser className="mr-2 h-4 w-4" />
              Clear
            </Button>
          </div>
          <Button
            onClick={onGenerate}
            disabled={isLoading || trimmedEmpty}
            className="shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Preview
              </>
            )}
          </Button>
        </div>

        <div
          className={cn(
            "rounded-lg border px-3 py-2 text-sm",
            error
              ? "border-destructive/30 bg-destructive/5 text-destructive"
              : "border-border/60 bg-secondary/40 text-muted-foreground"
          )}
          role={error ? "alert" : "status"}
          aria-live="polite"
        >
          {error
            ? error
            : "This preview experience is production-ready. Rendering quality will keep improving as the handwriting engine evolves."}
        </div>
      </CardContent>
    </Card>
  );
}

"use client";

import Image from "next/image";
import { Clock3, History, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import type { GenerateResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type HistoryStripProps = {
  history: GenerateResponse[];
  selectedId?: string;
  onSelect: (item: GenerateResponse) => void;
  onClear?: () => void;
};

function formatTime(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value));
}

export function HistoryStrip({
  history,
  selectedId,
  onSelect,
  onClear
}: HistoryStripProps) {
  return (
    <Card className="card-accent overflow-hidden bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-4 w-4 text-muted-foreground" />
              Recent History
            </CardTitle>
            <CardDescription>
              Stored locally in your browser (last 10 generations).
            </CardDescription>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClear}
            disabled={history.length === 0 || !onClear}
            className="self-start text-muted-foreground hover:text-foreground"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Clear history
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <div className="noise-overlay rounded-xl border border-dashed border-border/70 bg-gradient-to-br from-background/80 to-secondary/40 p-6 text-sm text-muted-foreground">
            <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-full border border-border/70 bg-white/80">
              <History className="h-4 w-4" />
            </div>
            <p className="font-medium text-foreground">No history yet</p>
            <p className="mt-1">
              Generate a preview to populate this strip. Your last 10 results stay
              in local storage for quick comparisons.
            </p>
          </div>
        ) : (
          <div className="relative">
            <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-6 bg-gradient-to-r from-white/85 to-transparent" />
            <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-6 bg-gradient-to-l from-white/85 to-transparent" />
            <div className="-mx-1 flex gap-3 overflow-x-auto px-1 pb-1">
            {history.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect(item)}
                className={cn(
                  "group min-w-[190px] max-w-[190px] rounded-xl border bg-background/70 p-2 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-border hover:bg-background",
                  selectedId === item.id
                    ? "border-primary/50 ring-2 ring-primary/20"
                    : "border-border/60"
                )}
              >
                <div className="overflow-hidden rounded-md border border-border/60 bg-white">
                  <Image
                    src={item.imageDataUrl}
                    alt={`History item ${item.id}`}
                    width={176}
                    height={96}
                    unoptimized
                    className="h-24 w-full object-cover object-top transition-transform duration-200 group-hover:scale-[1.02]"
                  />
                </div>
                <div className="mt-2 space-y-1">
                  <p className="line-clamp-2 text-xs leading-5 text-foreground">
                    {item.request.text}
                  </p>
                  <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                    <span className="capitalize">{item.request.style}</span>
                    <span className="inline-flex items-center gap-1">
                      <Clock3 className="h-3 w-3" />
                      {formatTime(item.createdAt)}
                    </span>
                  </div>
                </div>
              </button>
            ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

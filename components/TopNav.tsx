import { Circle, Command, PenTool, WandSparkles } from "lucide-react";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type TopNavProps = {
  isLoading: boolean;
  historyCount: number;
  hasResult: boolean;
};

export function TopNav({ isLoading, historyCount, hasResult }: TopNavProps) {
  return (
    <Card className="card-accent overflow-hidden border-white/70 bg-white/75 p-4 backdrop-blur-sm sm:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 grid h-10 w-10 place-items-center rounded-xl border border-border/70 bg-gradient-to-br from-white to-slate-100 shadow-sm">
            <PenTool className="h-4 w-4 text-slate-700" />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Handwriting Generator
            </p>
            <h1 className="mt-1 text-2xl font-semibold text-foreground sm:text-3xl">
              Handwrite Studio
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Generate, inspect, and export handwriting previews while the model is
              still in development.
            </p>
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2 lg:flex lg:flex-wrap lg:justify-end">
          <div className="section-chip bg-white/80">
            <Circle
              className={cn(
                "h-2.5 w-2.5 fill-current",
                isLoading ? "text-amber-500" : "text-emerald-500"
              )}
            />
            <span className="font-medium text-foreground">
              {isLoading ? "Generating" : "Ready"}
            </span>
          </div>

          <div className="section-chip bg-white/80">
            <WandSparkles className="h-3.5 w-3.5" />
            <span>{hasResult ? "Preview available" : "No preview yet"}</span>
          </div>

          <div className="section-chip bg-white/80">
            <span className="font-semibold text-foreground tabular-nums">
              {historyCount}
            </span>
            <span>History item{historyCount === 1 ? "" : "s"}</span>
          </div>

          <div className="section-chip bg-white/80">
            <Command className="h-3.5 w-3.5" />
            <span>+ Enter to generate</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

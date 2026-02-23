"use client";

import { SlidersHorizontal } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  SIZE_PRESETS,
  STYLE_OPTIONS,
  type GenerateRequest,
  type HandwritingStyle
} from "@/lib/types";

type SettingsPanelProps = {
  request: GenerateRequest;
  disabled?: boolean;
  onStyleChange: (value: HandwritingStyle) => void;
  onSizeChange: (width: number, height: number) => void;
  onLineSpacingChange: (value: number) => void;
  onSeedChange: (value: number | null) => void;
};

function getSizeValue(width: number, height: number) {
  return `${width}x${height}`;
}

export function SettingsPanel({
  request,
  disabled = false,
  onStyleChange,
  onSizeChange,
  onLineSpacingChange,
  onSeedChange
}: SettingsPanelProps) {
  const currentSizeValue = getSizeValue(request.width, request.height);

  return (
    <Card className="card-accent overflow-hidden bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
              Settings
            </CardTitle>
          </div>
          <span className="section-chip hidden sm:inline-flex">Metadata</span>
        </div>
        <CardDescription>
          Tune style, size, spacing, and seed behavior for each generated preview.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-2 rounded-lg border border-border/70 bg-background/50 p-3 sm:grid-cols-4">
          <div className="rounded-md bg-white/70 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Style
            </p>
            <p className="mt-1 text-sm font-medium capitalize">{request.style}</p>
          </div>
          <div className="rounded-md bg-white/70 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Width
            </p>
            <p className="mt-1 text-sm font-medium tabular-nums">{request.width}</p>
          </div>
          <div className="rounded-md bg-white/70 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Height
            </p>
            <p className="mt-1 text-sm font-medium tabular-nums">{request.height}</p>
          </div>
          <div className="rounded-md bg-white/70 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Seed
            </p>
            <p className="mt-1 text-sm font-medium tabular-nums">
              {request.seed ?? "auto"}
            </p>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="style-select">Style</Label>
            <Select
              value={request.style}
              onValueChange={(value) => onStyleChange(value as HandwritingStyle)}
              disabled={disabled}
            >
              <SelectTrigger id="style-select">
                <SelectValue placeholder="Select style" />
              </SelectTrigger>
              <SelectContent>
                {STYLE_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="size-select">Size</Label>
            <Select
              value={currentSizeValue}
              onValueChange={(value) => {
                const [width, height] = value.split("x").map(Number);
                onSizeChange(width, height);
              }}
              disabled={disabled}
            >
              <SelectTrigger id="size-select">
                <SelectValue placeholder="Select size" />
              </SelectTrigger>
              <SelectContent>
                {SIZE_PRESETS.map((preset) => (
                  <SelectItem
                    key={getSizeValue(preset.width, preset.height)}
                    value={getSizeValue(preset.width, preset.height)}
                  >
                    {preset.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-3 rounded-lg border border-border/70 bg-background/40 p-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="line-spacing">Line spacing</Label>
            <span className="text-sm tabular-nums text-muted-foreground">
              {request.lineSpacing.toFixed(2)}
            </span>
          </div>
          <Slider
            id="line-spacing"
            min={0.9}
            max={2.4}
            step={0.05}
            value={[request.lineSpacing]}
            onValueChange={(values) => {
              if (values[0] !== undefined) {
                onLineSpacingChange(values[0]);
              }
            }}
            disabled={disabled}
          />
        </div>

        <div className="space-y-2 rounded-lg border border-border/70 bg-background/40 p-3">
          <Label htmlFor="seed-input">Seed (optional)</Label>
          <Input
            id="seed-input"
            type="number"
            inputMode="numeric"
            min={0}
            max={2147483647}
            step={1}
            value={request.seed ?? ""}
            disabled={disabled}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") {
                onSeedChange(null);
                return;
              }

              const next = Number(raw);
              if (Number.isInteger(next) && next >= 0) {
                onSeedChange(next);
              }
            }}
            placeholder="Auto (derived from input)"
          />
          <p className="text-xs text-muted-foreground">
            Leave blank to derive a deterministic seed from the text and settings.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

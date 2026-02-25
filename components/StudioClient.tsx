"use client";

import {
  Activity,
  Copy,
  Database,
  Download,
  Loader2,
  RefreshCw,
  Save,
  Settings,
  Sparkles,
  Wand2,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { PreviewCanvas } from "@/components/PreviewCanvas";
import { RecognizePanel } from "@/components/RecognizePanel";
import { ThemePicker } from "@/components/ThemePicker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/use-toast";
import { getDatasetStats, getTrainingStatus, postGenerate, postTrainControl } from "@/lib/api";
import { synthesizeDebugMeta } from "@/lib/debug-meta";
import {
  DEFAULT_GENERATE_REQUEST,
  type DatasetStats,
  type GenerateRequest,
  type GenerateResponse,
  type TrainingStatus
} from "@/lib/types";
import {
  clearGenerationHistory,
  loadGenerationHistory,
  pushGenerationHistoryItem,
  saveGenerationHistory
} from "@/lib/storage";
import { loadPresets, savePresets } from "@/lib/studio-storage";
import type { GeneratePreset } from "@/lib/studio-types";
import { cn } from "@/lib/utils";
import { useThemeManager } from "@/components/ThemeProvider";

type TabKey = "generate" | "ocr" | "dataset" | "training";
type BaselineJitter = "off" | "low" | "med" | "high";
type PageStyle = "blank" | "lined" | "grid" | "dot";
type PaperTexture = "off" | "subtle" | "med";

type UiState = {
  text: string;
  style: "pencil" | "ink" | "marker";
  width: number;
  height: number;
  variation: number;
  styleStrength: number;
  temperature: number;
  seed: number | null;
  seedLock: boolean;
  strokeThickness: number;
  baselineJitter: BaselineJitter;
  letterSpacing: number;
  wordSpacing: number;
  lineSpacing: number;
  pageStyle: PageStyle;
  paperTexture: PaperTexture;
  debug: boolean;
  showFallbackMarkers: boolean;
  showBoxes: boolean;
  showLabels: boolean;
  showTimeline: boolean;
  fitToWidth: boolean;
  zoom: number;
  reducedMotion: boolean;
};

const DEFAULT_UI: UiState = {
  text: "hello world\nthis is a test.",
  style: "ink",
  width: 1024,
  height: 640,
  variation: 62,
  styleStrength: 72,
  temperature: 1.0,
  seed: null,
  seedLock: false,
  strokeThickness: 50,
  baselineJitter: "low",
  letterSpacing: 50,
  wordSpacing: 50,
  lineSpacing: 1.35,
  pageStyle: "lined",
  paperTexture: "subtle",
  debug: false,
  showFallbackMarkers: true,
  showBoxes: true,
  showLabels: false,
  showTimeline: false,
  fitToWidth: true,
  zoom: 1,
  reducedMotion: false
};

function randomSeed() {
  return Math.floor(Math.random() * 2_147_483_647);
}

function pad(n: number) {
  return String(n).padStart(2, "0");
}

function pngFilename() {
  const d = new Date();
  return `handwriting_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(
    d.getHours()
  )}${pad(d.getMinutes())}${pad(d.getSeconds())}.png`;
}

function downloadDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function copyDataUrl(dataUrl: string) {
  const r = await fetch(dataUrl);
  const b = await r.blob();
  const Ctor = (window as Window & { ClipboardItem?: typeof ClipboardItem }).ClipboardItem;
  if (Ctor && navigator.clipboard?.write) {
    await navigator.clipboard.write([new Ctor({ [b.type]: b })]);
    return "image";
  }
  await navigator.clipboard.writeText(dataUrl);
  return "url";
}

async function makeTransparent(dataUrl: string) {
  const img = new Image();
  img.src = dataUrl;
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("image load failed"));
  });
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  if (!ctx) return dataUrl;
  ctx.drawImage(img, 0, 0);
  const frame = ctx.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < frame.data.length; i += 4) {
    if (frame.data[i] > 225 && frame.data[i + 1] > 225 && frame.data[i + 2] > 225) frame.data[i + 3] = 0;
  }
  ctx.putImageData(frame, 0, 0);
  return c.toDataURL("image/png");
}

function baselineMap(level: BaselineJitter) {
  if (level === "off") return 0;
  if (level === "low") return 0.8;
  if (level === "med") return 1.4;
  return 2.2;
}

function toGenerateRequest(ui: UiState): GenerateRequest {
  const req = { ...DEFAULT_GENERATE_REQUEST } as GenerateRequest;
  req.text = ui.text;
  req.style = ui.style;
  req.width = ui.width;
  req.height = ui.height;
  req.lineSpacing = ui.lineSpacing;
  req.seed = ui.seedLock ? ui.seed : null;
  req.pageStyle = ui.pageStyle;
  req.paperTexture = ui.paperTexture;
  req.letterModelEnabled = true;
  req.letterModelStyleStrength = Math.max(0.25, Math.min(3, 0.45 + (ui.styleStrength / 100) * 2.1));
  req.letterModelBaselineJitter = baselineMap(ui.baselineJitter);
  req.letterModelWordSlant = Math.max(0, Math.min(3, 0.15 + (ui.variation / 100) * 1.4));
  req.letterModelRotationJitter = Math.max(0, Math.min(3, 0.25 + ui.temperature * 0.55));
  req.letterModelInkVariation = Math.max(0, Math.min(1, (ui.temperature - 0.1) / 1.9));
  return req;
}

function sliderClass() {
  return "[&_[data-radix-slider-range]]:bg-[var(--accent)] [&_[data-radix-slider-thumb]]:border-[var(--accent)]";
}

function SliderField({
  label,
  value,
  display,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-[var(--text)]">{label}</span>
        <span className="text-[var(--muted)]">{display}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={([v]) => onChange(v)} className={sliderClass()} />
    </div>
  );
}

export function StudioClient() {
  const { theme } = useThemeManager();
  const [tab, setTab] = useState<TabKey>("generate");
  const [ui, setUi] = useState<UiState>(DEFAULT_UI);
  const [history, setHistory] = useState<GenerateResponse[]>([]);
  const [presets, setPresetsState] = useState<GeneratePreset[]>([]);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [previewMeta, setPreviewMeta] = useState<GenerateResponse["meta"] | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDrawer, setShowDrawer] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [dataset, setDataset] = useState<DatasetStats | null>(null);
  const [statusSeries, setStatusSeries] = useState<{ loss: number[]; val: number[] }>({ loss: [], val: [] });
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [activeSeed, setActiveSeed] = useState<number | null>(null);
  const [aborter, setAborter] = useState<AbortController | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(true);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);

  const fallbackCandidateChars = useMemo(
    () =>
      Array.from(new Set(ui.text.replace(/\s/g, "").split("").filter(Boolean))).filter((ch) => !/[a-z]/.test(ch)),
    [ui.text]
  );

  function update<K extends keyof UiState>(key: K, value: UiState[K]) {
    setUi((prev) => ({ ...prev, [key]: value }));
  }

  useEffect(() => {
    setHistory(loadGenerationHistory());
    setPresetsState(loadPresets());
    if (typeof window !== "undefined") {
      update("reducedMotion", window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      if (tab === "generate" || tab === "training") void refreshStatus();
    }, 10000);
    return () => window.clearInterval(id);
  }, [tab]);

  useEffect(() => {
    if (tab === "generate" || tab === "training") void refreshStatus();
    if (tab === "dataset") void refreshDataset();
  }, [tab]);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (showExportMenu && exportMenuRef.current && target && !exportMenuRef.current.contains(target)) {
        setShowExportMenu(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowExportMenu(false);
        setShowDrawer(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [showExportMenu]);

  async function refreshStatus() {
    setIsLoadingStatus(true);
    try {
      const s = await getTrainingStatus();
      setTraining(s);
      setStatusSeries((prev) => ({
        loss: [...prev.loss, s.progress?.loss ?? prev.loss.at(-1) ?? 0].slice(-30),
        val: [...prev.val, s.progress?.valLoss ?? prev.val.at(-1) ?? 0].slice(-30)
      }));
    } finally {
      setIsLoadingStatus(false);
    }
  }

  async function refreshDataset() {
    setIsLoadingDataset(true);
    try {
      setDataset(await getDatasetStats());
    } finally {
      setIsLoadingDataset(false);
    }
  }

  async function runGenerate(regenerate = false) {
    if (isGenerating) return;
    if (!ui.text.trim()) {
      setError("Please enter text before generating.");
      return;
    }
    const nextSeed = ui.seedLock ? ui.seed ?? randomSeed() : randomSeed();
    const nextUi = { ...ui, seed: nextSeed };
    setUi(nextUi);
    const request = toGenerateRequest(nextUi);
    setActiveSeed(request.seed ?? nextSeed);
    const ctrl = new AbortController();
    setAborter(ctrl);
    setIsGenerating(true);
    setError(null);
    try {
      const response = await postGenerate(request, { signal: ctrl.signal });
      setResult(response);
      setPreviewMeta(
        response.meta ??
          synthesizeDebugMeta({
            text: response.request.text,
            width: response.request.width,
            height: response.request.height,
            lineSpacing: response.request.lineSpacing
          })
      );
      setHistory((prev) => {
        const next = pushGenerationHistoryItem(prev, response);
        saveGenerationHistory(next);
        return next;
      });
      if (regenerate) {
        toast({ title: "Regenerated", description: "New sampling applied to the same text." });
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setError("Generation cancelled.");
      } else {
        setError(e instanceof Error ? e.message : "Generation failed");
      }
    } finally {
      setAborter(null);
      setIsGenerating(false);
    }
  }

  function selectHistory(item: GenerateResponse) {
    setResult(item);
    setPreviewMeta(
      item.meta ??
        synthesizeDebugMeta({
          text: item.request.text,
          width: item.request.width,
          height: item.request.height,
          lineSpacing: item.request.lineSpacing
        })
    );
    setActiveSeed(item.meta?.seedUsed ?? item.request.seed ?? null);
  }

  async function exportPng(kind: "png" | "transparent") {
    if (!result) return;
    const source = kind === "transparent" ? await makeTransparent(result.imageDataUrl) : result.imageDataUrl;
    downloadDataUrl(source, pngFilename());
  }

  async function copyPreview() {
    if (!result) return;
    try {
      const mode = await copyDataUrl(result.imageDataUrl);
      toast({ title: mode === "image" ? "Copied image" : "Copied URL" });
    } catch (e) {
      toast({
        title: "Copy failed",
        description: e instanceof Error ? e.message : "Clipboard unavailable",
        variant: "destructive"
      });
    }
  }

  function savePresetNow() {
    const name = window.prompt("Preset name", `Preset ${presets.length + 1}`)?.trim();
    if (!name) return;
    const next = [{ id: crypto.randomUUID(), name, createdAt: new Date().toISOString(), form: ui }, ...presets].slice(0, 30) as GeneratePreset[];
    setPresetsState(next);
    savePresets(next);
  }

  function loadPresetById(id: string) {
    const p = presets.find((it) => it.id === id);
    if (!p) return;
    setUi(p.form as unknown as UiState);
    toast({ title: "Preset loaded", description: p.name });
  }

  const lossSeries = statusSeries.loss.filter((v) => Number.isFinite(v) && v > 0);
  const valSeries = statusSeries.val.filter((v) => Number.isFinite(v) && v > 0);
  const lineChart = (vals: number[], color: string) => {
    if (vals.length < 2) return <div className="h-24 rounded-lg border border-[var(--border)] bg-[var(--panel)]" />;
    const w = 320;
    const h = 88;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const span = max - min || 1;
    const points = vals
      .map((v, i) => `${(i / (vals.length - 1)) * w},${h - ((v - min) / span) * (h - 8) - 4}`)
      .join(" ");
    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="h-24 w-full rounded-lg border border-[var(--border)] bg-[var(--panel)]">
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
      </svg>
    );
  };

  const trainingCard = (
    <div className="studio-panel p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs uppercase tracking-[0.14em] text-[var(--muted)]">Training</p>
        <span className={cn("rounded-full px-2 py-0.5 text-[10px]", training?.status === "training" ? "bg-emerald-500/15 text-emerald-300" : "bg-[var(--panel-strong)] text-[var(--muted)]")}>
          {training?.status ?? "idle"}
        </span>
      </div>
      <p className="text-sm font-semibold text-[var(--text)]">{training?.version ?? "letter-gen-v1"}</p>
      <p className="mt-1 text-xs text-[var(--muted)]">Dataset {training?.datasetSize ?? "--"} | Epoch {training?.progress?.epoch ?? "--"}</p>
      <p className="text-xs text-[var(--muted)]">Last trained {training?.lastTrained ? new Date(training.lastTrained).toLocaleString() : "—"}</p>
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="studio-panel-strong panel-glow relative overflow-visible p-4 sm:p-5">
        <div className="absolute inset-0 opacity-40" aria-hidden>
          <div className="h-full w-full bg-[radial-gradient(circle_at_10%_10%,color-mix(in_srgb,var(--accent)_16%,transparent),transparent_42%),radial-gradient(circle_at_90%_15%,color-mix(in_srgb,var(--accent2)_14%,transparent),transparent_36%)]" />
        </div>
        <div className="relative flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--panel)] px-3 py-1 text-xs text-[var(--muted)]">
              <Sparkles className="h-3.5 w-3.5 text-[var(--accent)]" />
              Theme: {theme.name}
            </div>
            <h2 className="mt-3 text-2xl font-semibold text-[var(--text)] sm:text-3xl">Handwriting Studio</h2>
            <p className="mt-1 max-w-3xl text-sm text-[var(--muted)]">
              Lowercase letters are generated independently from the letter model. Debug overlays can show generated vs fallback glyphs.
            </p>
          </div>
          <div className="relative z-[80] flex flex-wrap items-center gap-2">
            <ThemePicker compact />
            <div ref={exportMenuRef} className="relative">
              <Button type="button" variant="outline" className="studio-button-soft h-10 rounded-xl" disabled={!result} onClick={() => setShowExportMenu((v) => !v)}>
                <Download className="mr-2 h-4 w-4" /> Export
              </Button>
              {showExportMenu && result ? (
                <div className="fade-rise absolute right-0 top-12 z-[90] min-w-52 rounded-xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--panel-strong)_92%,transparent)] p-2 shadow-2xl shadow-[var(--shadow)] backdrop-blur-xl">
                  <button className="block w-full rounded-lg px-3 py-2 text-left text-sm text-[var(--text)] hover:bg-[var(--panel)]" onClick={() => void exportPng("png")}>PNG</button>
                  <button className="block w-full rounded-lg px-3 py-2 text-left text-sm text-[var(--text)] hover:bg-[var(--panel)]" onClick={() => void exportPng("transparent")}>Transparent PNG</button>
                  <button className="block w-full rounded-lg px-3 py-2 text-left text-sm text-[var(--text)] hover:bg-[var(--panel)]" onClick={() => toast({ title: "PDF export", description: "Optional PDF export can be added next." })}>PDF (optional)</button>
                </div>
              ) : null}
            </div>
            <Button type="button" variant="outline" className="studio-button-soft h-10 w-10 rounded-xl p-0" onClick={() => setShowDrawer(true)}>
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)} className="grid gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
        <div className="studio-panel p-3">
          <TabsList className="grid h-auto w-full grid-cols-2 gap-2 bg-transparent p-0 lg:grid-cols-1">
            <TabsTrigger value="generate" className="justify-start gap-2 rounded-xl border border-transparent bg-[var(--panel)] data-[state=active]:border-[var(--border)] data-[state=active]:bg-[var(--panel-strong)]"><Wand2 className="h-4 w-4" />Generate</TabsTrigger>
            <TabsTrigger value="ocr" className="justify-start gap-2 rounded-xl border border-transparent bg-[var(--panel)] data-[state=active]:border-[var(--border)] data-[state=active]:bg-[var(--panel-strong)]"><Sparkles className="h-4 w-4" />OCR</TabsTrigger>
            <TabsTrigger value="dataset" className="justify-start gap-2 rounded-xl border border-transparent bg-[var(--panel)] data-[state=active]:border-[var(--border)] data-[state=active]:bg-[var(--panel-strong)]"><Database className="h-4 w-4" />Dataset</TabsTrigger>
            <TabsTrigger value="training" className="justify-start gap-2 rounded-xl border border-transparent bg-[var(--panel)] data-[state=active]:border-[var(--border)] data-[state=active]:bg-[var(--panel-strong)]"><Activity className="h-4 w-4" />Training</TabsTrigger>
          </TabsList>
          <div className="mt-3 hidden lg:block">{trainingCard}</div>
        </div>

        <div>
          <TabsContent value="generate" className="mt-0 space-y-4">
            <div className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
              <div className="space-y-4">
                <div className="studio-panel-strong p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Input</p>
                      <p className="text-sm font-semibold text-[var(--text)]">Generate handwriting from text</p>
                    </div>
                    <button type="button" className="text-xs text-[var(--muted)] hover:text-[var(--text)]" onClick={() => setShowAdvanced((v) => !v)}>
                      {showAdvanced ? "Hide advanced" : "Show advanced"}
                    </button>
                  </div>
                  <Textarea value={ui.text} onChange={(e) => update("text", e.target.value)} rows={6} className="min-h-[11rem] rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]" />
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--muted)]">
                    <span>{ui.text.length} chars | {fallbackCandidateChars.length ? `typed fallback likely: ${fallbackCandidateChars.join(" ")}` : "all lowercase letters use AI if available"}</span>
                    <span>Seed used: {activeSeed ?? "auto"}</span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button type="button" onClick={() => void runGenerate(false)} disabled={isGenerating} className="studio-button-accent rounded-xl">{isGenerating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Wand2 className="mr-2 h-4 w-4" />}Generate</Button>
                    <Button type="button" variant="secondary" onClick={() => void runGenerate(true)} disabled={isGenerating} className="rounded-xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--panel-strong)_80%,transparent)] text-[var(--text)] hover:bg-[var(--panel)]"><RefreshCw className="mr-2 h-4 w-4" />Regenerate</Button>
                    <Button type="button" variant="outline" onClick={() => update("text", "")} className="studio-button-soft rounded-xl">Clear</Button>
                    <Button type="button" variant={ui.seedLock ? "default" : "outline"} onClick={() => update("seedLock", !ui.seedLock)} className={cn("rounded-xl", ui.seedLock ? "studio-button-accent" : "studio-button-soft")}>Use seed lock</Button>
                    <Button type="button" variant="outline" onClick={() => aborter?.abort()} disabled={!aborter} className="studio-button-soft rounded-xl">Cancel</Button>
                  </div>
                </div>

                <div className="studio-panel p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <p className="text-sm font-semibold text-[var(--text)]">Generation controls</p>
                    <span className="text-xs text-[var(--muted)]">variation | temperature | style</span>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <SliderField label="Variation" value={ui.variation} display={String(ui.variation)} min={0} max={100} step={1} onChange={(v) => update("variation", v)} />
                    <SliderField label="Style strength" value={ui.styleStrength} display={`${ui.styleStrength}%`} min={0} max={100} step={1} onChange={(v) => update("styleStrength", v)} />
                    <SliderField label="Temperature" value={ui.temperature} display={ui.temperature.toFixed(2)} min={0.1} max={2} step={0.05} onChange={(v) => update("temperature", v)} />
                    <SliderField label="Stroke thickness" value={ui.strokeThickness} display={`${ui.strokeThickness}%`} min={0} max={100} step={1} onChange={(v) => update("strokeThickness", v)} />
                  </div>

                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-[var(--text)]">Seed (integer)</label>
                      <div className="flex gap-2">
                        <Input type="number" value={ui.seed ?? ""} onChange={(e) => update("seed", e.target.value ? Number(e.target.value) : null)} placeholder="auto" className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]" />
                        <Button type="button" variant="outline" className="studio-button-soft rounded-xl" onClick={() => update("seed", randomSeed())}>Randomize</Button>
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-[var(--text)]">Canvas / Style</label>
                      <div className="grid grid-cols-2 gap-2">
                        <Select value={ui.style} onValueChange={(v) => update("style", v as UiState["style"])}>
                          <SelectTrigger className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="ink">Ink</SelectItem>
                            <SelectItem value="pencil">Pencil</SelectItem>
                            <SelectItem value="marker">Marker</SelectItem>
                          </SelectContent>
                        </Select>
                        <Select value={ui.baselineJitter} onValueChange={(v) => update("baselineJitter", v as BaselineJitter)}>
                          <SelectTrigger className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="off">Jitter Off</SelectItem>
                            <SelectItem value="low">Jitter Low</SelectItem>
                            <SelectItem value="med">Jitter Med</SelectItem>
                            <SelectItem value="high">Jitter High</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>

                  {showAdvanced ? (
                    <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--panel)] p-3">
                      <div className="grid gap-4 md:grid-cols-2">
                        <SliderField label="Letter spacing" value={ui.letterSpacing} display={String(ui.letterSpacing)} min={0} max={100} step={1} onChange={(v) => update("letterSpacing", v)} />
                        <SliderField label="Word spacing" value={ui.wordSpacing} display={String(ui.wordSpacing)} min={0} max={100} step={1} onChange={(v) => update("wordSpacing", v)} />
                        <SliderField label="Line spacing" value={ui.lineSpacing} display={ui.lineSpacing.toFixed(2)} min={0.8} max={3} step={0.05} onChange={(v) => update("lineSpacing", v)} />
                        <SliderField label="Zoom" value={ui.zoom} display={`${Math.round(ui.zoom * 100)}%`} min={0.4} max={3} step={0.05} onChange={(v) => update("zoom", v)} />
                      </div>
                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        <Select value={ui.pageStyle} onValueChange={(v) => update("pageStyle", v as PageStyle)}>
                          <SelectTrigger className="rounded-xl border-[var(--border)] bg-[var(--panel-strong)] text-[var(--text)]"><SelectValue placeholder="Page style" /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="blank">Blank</SelectItem>
                            <SelectItem value="lined">Notebook lines</SelectItem>
                            <SelectItem value="grid">Grid</SelectItem>
                            <SelectItem value="dot">Dot</SelectItem>
                          </SelectContent>
                        </Select>
                        <Select value={ui.paperTexture} onValueChange={(v) => update("paperTexture", v as PaperTexture)}>
                          <SelectTrigger className="rounded-xl border-[var(--border)] bg-[var(--panel-strong)] text-[var(--text)]"><SelectValue placeholder="Paper texture" /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="off">Texture Off</SelectItem>
                            <SelectItem value="subtle">Texture Subtle</SelectItem>
                            <SelectItem value="med">Texture Medium</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="mt-4 grid gap-2 md:grid-cols-2">
                        {[
                          ["debug", "Show debug overlay"],
                          ["showFallbackMarkers", "Show fallback markers"],
                          ["showBoxes", "Show per-letter boxes"],
                          ["showLabels", "Show per-letter labels"],
                          ["showTimeline", "Show generation timeline"],
                          ["fitToWidth", "Fit to width"],
                          ["reducedMotion", "Reduced motion"]
                        ].map(([key, label]) => (
                          <button
                            key={key}
                            type="button"
                            onClick={() => update(key as keyof UiState, !Boolean(ui[key as keyof UiState]) as never)}
                            className={cn(
                              "rounded-xl border px-3 py-2 text-left text-sm",
                              ui[key as keyof UiState]
                                ? "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_12%,var(--panel))] text-[var(--text)]"
                                : "border-[var(--border)] bg-[var(--panel-strong)] text-[var(--muted)]"
                            )}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        <div>
                          <label className="mb-1 block text-xs font-medium text-[var(--text)]">Width</label>
                          <Input type="number" value={ui.width} onChange={(e) => update("width", Math.max(320, Math.min(2048, Number(e.target.value || 1024))))} className="rounded-xl border-[var(--border)] bg-[var(--panel-strong)] text-[var(--text)]" />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs font-medium text-[var(--text)]">Height</label>
                          <Input type="number" value={ui.height} onChange={(e) => update("height", Math.max(240, Math.min(2048, Number(e.target.value || 640))))} className="rounded-xl border-[var(--border)] bg-[var(--panel-strong)] text-[var(--text)]" />
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="space-y-4 xl:sticky xl:top-4 xl:self-start">
                <PreviewCanvas
                  imageDataUrl={result?.imageDataUrl ?? null}
                  meta={previewMeta ?? null}
                  isGenerating={isGenerating}
                  error={error}
                  showDebugOverlay={ui.debug}
                  showBoxes={ui.showBoxes}
                  showLabels={ui.showLabels}
                  showFallbackMarkers={ui.showFallbackMarkers}
                  fitToWidth={ui.fitToWidth}
                  zoom={ui.zoom}
                  reducedMotion={ui.reducedMotion}
                  onZoomChange={(v) => update("zoom", v)}
                  onFitToWidth={() => update("zoom", 1)}
                  onToggleFit={(v) => update("fitToWidth", v)}
                />

                <div className="studio-panel p-4">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm font-semibold text-[var(--text)]">Output actions</p>
                    <span className="text-xs text-[var(--muted)]">seed {activeSeed ?? "auto"}</span>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <Button type="button" onClick={() => void exportPng("png")} disabled={!result} className="studio-button-accent rounded-xl"><Download className="mr-2 h-4 w-4" />Download PNG</Button>
                    <Button type="button" variant="outline" onClick={() => void copyPreview()} disabled={!result} className="studio-button-soft rounded-xl"><Copy className="mr-2 h-4 w-4" />Copy image</Button>
                    <Button type="button" variant="outline" onClick={() => void exportPng("transparent")} disabled={!result} className="studio-button-soft rounded-xl">Transparent PNG</Button>
                    <Button type="button" variant="outline" onClick={savePresetNow} className="studio-button-soft rounded-xl"><Save className="mr-2 h-4 w-4" />Save preset</Button>
                  </div>
                  <div className="mt-3">
                    <Select onValueChange={loadPresetById}>
                      <SelectTrigger className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]"><SelectValue placeholder="Load preset" /></SelectTrigger>
                      <SelectContent>
                        {presets.length === 0 ? <SelectItem value="none" disabled>No presets saved</SelectItem> : presets.map((p) => <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  {ui.showTimeline ? (
                    <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--panel)] p-3 text-xs text-[var(--muted)]">
                      <div>1. Sample latent vectors per lowercase letter</div>
                      <div>2. Decode glyphs and compose lines</div>
                      <div>3. Apply page styling and export PNG</div>
                    </div>
                  ) : null}
                </div>

                <div className="studio-panel p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <p className="text-sm font-semibold text-[var(--text)]">History</p>
                    <Button type="button" variant="ghost" size="sm" className="h-8 rounded-lg text-[var(--muted)]" onClick={() => { setHistory([]); clearGenerationHistory(); }}>Clear</Button>
                  </div>
                  {history.length === 0 ? (
                    <p className="text-xs text-[var(--muted)]">No renders yet.</p>
                  ) : (
                    <div className="grid gap-2 sm:grid-cols-2">
                      {history.slice(0, 6).map((item) => (
                        <button key={item.id} type="button" onClick={() => selectHistory(item)} className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--panel)] text-left transition duration-200 hover:-translate-y-0.5 hover:border-[color-mix(in_srgb,var(--accent)_40%,var(--border))] hover:shadow-lg hover:shadow-[var(--shadow)]">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={item.imageDataUrl} alt="" className="h-20 w-full object-cover" />
                          <div className="p-2 text-xs text-[var(--muted)]">
                            <div className="line-clamp-1 text-[var(--text)]">{item.request.text}</div>
                            <div>{new Date(item.createdAt).toLocaleTimeString()}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="ocr" className="mt-0">
            <div className="studio-panel-strong p-4">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">OCR</p>
                  <p className="text-sm font-semibold text-[var(--text)]">Optional page recognition panel</p>
                </div>
                <span className="rounded-full border border-[var(--border)] bg-[var(--panel)] px-2 py-1 text-xs text-[var(--muted)]">Secondary tool</span>
              </div>
              <RecognizePanel />
            </div>
          </TabsContent>

          <TabsContent value="dataset" className="mt-0">
            <div className="studio-panel-strong p-4">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Dataset inspector</p>
                  <p className="text-sm font-semibold text-[var(--text)]">Class distribution + undertrained letters</p>
                </div>
                <div className="flex gap-2">
                  <Button type="button" variant="outline" className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]" onClick={() => void refreshDataset()} disabled={isLoadingDataset}>
                    {isLoadingDataset ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}Refresh
                  </Button>
                  <Button type="button" className="rounded-xl" disabled={!dataset} onClick={() => downloadDataUrl(`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(dataset, null, 2))}`, `dataset_stats_${new Date().toISOString().slice(0, 10)}.json`)}>
                    Export dataset stats
                  </Button>
                </div>
              </div>
              {!dataset ? (
                <div className="rounded-xl border border-dashed border-[var(--border)] p-6 text-sm text-[var(--muted)]">
                  {isLoadingDataset ? "Loading dataset stats..." : "Dataset stats unavailable (mock/API endpoint expected)."}
                </div>
              ) : (
                <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
                  <div className="studio-panel p-3">
                    <p className="mb-3 text-sm font-semibold text-[var(--text)]">Class distribution</p>
                    <div className="space-y-2">
                      {dataset.letters.map((entry) => {
                        const max = Math.max(...dataset.letters.map((l) => l.count), 1);
                        const widthPct = (entry.count / max) * 100;
                        const under = entry.count < dataset.undertrainedThreshold;
                        return (
                          <div key={entry.letter} className="grid grid-cols-[18px_minmax(0,1fr)_40px] items-center gap-2 text-xs">
                            <span className="font-semibold text-[var(--text)]">{entry.letter}</span>
                            <div className="h-2 rounded-full bg-[var(--panel-strong)]">
                              <div className={cn("h-2 rounded-full", under ? "bg-amber-400" : "bg-[var(--accent)]")} style={{ width: `${widthPct}%` }} />
                            </div>
                            <span className={cn("tabular-nums", under ? "text-amber-300" : "text-[var(--muted)]")}>{entry.count}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="studio-panel p-3">
                      <p className="mb-2 text-sm font-semibold text-[var(--text)]">Undertrained flags</p>
                      <div className="flex flex-wrap gap-2">
                        {dataset.letters.filter((l) => l.count < dataset.undertrainedThreshold).map((l) => (
                          <span key={l.letter} className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-xs text-amber-300">{l.letter} ({l.count})</span>
                        ))}
                      </div>
                    </div>
                    <div className="studio-panel p-3">
                      <p className="mb-2 text-sm font-semibold text-[var(--text)]">Confusing pairs hints</p>
                      <div className="flex flex-wrap gap-2">
                        {dataset.confusingPairs.map(([a, b]) => (
                          <span key={`${a}${b}`} className="rounded-lg border border-[var(--border)] bg-[var(--panel)] px-2 py-1 text-xs text-[var(--muted)]">{a} vs {b}</span>
                        ))}
                      </div>
                    </div>
                    <div className="studio-panel p-3">
                      <p className="mb-2 text-sm font-semibold text-[var(--text)]">Thumbnail preview grid (stub-ready)</p>
                      <div className="grid grid-cols-6 gap-2">
                        {dataset.letters.slice(0, 24).map((l) => (
                          <div key={l.letter} className="grid aspect-square place-items-center rounded-lg border border-[var(--border)] bg-[var(--panel)] text-sm font-semibold text-[var(--text)]">{l.letter}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="training" className="mt-0">
            <div className="studio-panel-strong p-4">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Training dashboard</p>
                  <p className="text-sm font-semibold text-[var(--text)]">Status, version, and run controls</p>
                </div>
                <div className="flex gap-2">
                  <Button type="button" variant="outline" className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]" onClick={() => void refreshStatus()} disabled={isLoadingStatus}>
                    {isLoadingStatus ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}Refresh
                  </Button>
                  <Button type="button" className="rounded-xl" onClick={async () => { const r = await postTrainControl("train"); toast({ title: r.ok ? "Training start sent" : "Backend required", description: r.message }); }}>
                    Start training
                  </Button>
                  <Button type="button" variant="outline" className="rounded-xl border-[var(--border)] bg-[var(--panel)] text-[var(--text)]" onClick={async () => { const r = await postTrainControl("stop"); toast({ title: r.ok ? "Stop sent" : "Backend required", description: r.message }); }}>
                    Stop training
                  </Button>
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-4">
                <div className="studio-panel p-3"><div className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">Model</div><div className="mt-1 text-lg font-semibold text-[var(--text)]">{training?.version ?? "letter-gen-v1"}</div></div>
                <div className="studio-panel p-3"><div className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">Dataset</div><div className="mt-1 text-lg font-semibold text-[var(--text)]">{training?.datasetSize ?? "--"}</div></div>
                <div className="studio-panel p-3"><div className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">Status</div><div className="mt-1 text-lg font-semibold text-[var(--text)]">{training?.status ?? "idle"}</div></div>
                <div className="studio-panel p-3"><div className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">Epoch</div><div className="mt-1 text-lg font-semibold text-[var(--text)]">{training?.progress?.epoch ?? "--"}</div></div>
              </div>

              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <div className="studio-panel p-3"><p className="mb-2 text-sm font-semibold text-[var(--text)]">Loss curve</p>{lineChart(lossSeries, "var(--accent)")}</div>
                <div className="studio-panel p-3"><p className="mb-2 text-sm font-semibold text-[var(--text)]">Val loss curve</p>{lineChart(valSeries, "var(--accent2)")}</div>
              </div>
              <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--panel)] p-3 text-xs text-[var(--muted)]">
                {training?.status === "training" ? `Training active. Epoch ${training.progress?.epoch ?? "?"}. For unattended training use scripts/train_infinite.ps1.` : "Training control API routes are stubbed by default. Connect backend process control to enable buttons."}
              </div>
            </div>
          </TabsContent>
        </div>
      </Tabs>

      {showDrawer ? (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/35 backdrop-blur-[2px]" onClick={() => setShowDrawer(false)} />
          <aside className="absolute right-0 top-0 h-full w-full max-w-md border-l border-[var(--border)] bg-[color-mix(in_srgb,var(--panel-strong)_92%,transparent)] p-5 shadow-2xl backdrop-blur-xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Settings</p>
                <p className="text-sm font-semibold text-[var(--text)]">Workspace options</p>
              </div>
              <Button type="button" size="icon" variant="ghost" onClick={() => setShowDrawer(false)} className="rounded-xl"><X className="h-4 w-4" /></Button>
            </div>
            <div className="space-y-4">
              <div className="studio-panel p-3">
                <p className="text-sm font-semibold text-[var(--text)]">Reduced Motion</p>
                <p className="mt-1 text-xs text-[var(--muted)]">Respects system preference and disables heavy preview effects.</p>
                <Button type="button" variant={ui.reducedMotion ? "default" : "outline"} className={cn("mt-3 rounded-xl", ui.reducedMotion ? "studio-button-accent" : "studio-button-soft")} onClick={() => update("reducedMotion", !ui.reducedMotion)}>
                  {ui.reducedMotion ? "Enabled" : "Disabled"}
                </Button>
              </div>
              <div className="studio-panel p-3">
                <p className="text-sm font-semibold text-[var(--text)]">Theme manager</p>
                <p className="mt-1 text-xs text-[var(--muted)]">Themes persist in localStorage key `handwriting_theme`.</p>
                <div className="mt-3"><ThemePicker /></div>
              </div>
              <div className="studio-panel p-3 text-xs text-[var(--muted)]">
                <div>/api/generate is connected to local Python generation.</div>
                <div>/api/status and /api/dataset-stats are available with stub fallback.</div>
                <div>/api/train and /api/stop are UI stubs until backend process control is wired.</div>
              </div>
            </div>
          </aside>
        </div>
      ) : null}
    </div>
  );
}


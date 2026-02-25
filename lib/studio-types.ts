import type { DatasetStats, GenerateMeta, GenerateResponse, HandwritingStyle, TrainingStatus } from "@/lib/types";

export type BaselineJitterLevel = "off" | "low" | "med" | "high";
export type PageStyle = "blank" | "lined" | "grid" | "dot";
export type PaperTexture = "off" | "subtle" | "med";
export type ExportFormat = "png" | "transparent-png" | "pdf";

export type GenerateUiSettings = {
  variation: number;
  styleStrength: number;
  temperature: number;
  seed: number | null;
  useSeedLock: boolean;
  strokeThickness: number;
  baselineJitter: BaselineJitterLevel;
  letterSpacing: number;
  wordSpacing: number;
  lineSpacing: number;
  pageStyle: PageStyle;
  paperTexture: PaperTexture;
  debugOverlay: boolean;
  showFallbackMarkers: boolean;
  showLetterBoxes: boolean;
  showLetterLabels: boolean;
  showTimeline: boolean;
  fitToWidth: boolean;
  zoom: number;
  reducedMotion: boolean;
};

export type GenerateFormState = {
  text: string;
  style: HandwritingStyle;
  width: number;
  height: number;
  settings: GenerateUiSettings;
};

export type GeneratePreset = {
  id: string;
  name: string;
  createdAt: string;
  form: GenerateFormState;
};

export type PreviewResult = {
  imageDataUrl: string;
  response: GenerateResponse | null;
  meta: GenerateMeta | null;
  generatedAt: string;
  usedSeed: number | null;
};

export type StudioStatusState = {
  training: TrainingStatus | null;
  dataset: DatasetStats | null;
  isLoadingTraining: boolean;
  isLoadingDataset: boolean;
};

export const DEFAULT_UI_SETTINGS: GenerateUiSettings = {
  variation: 62,
  styleStrength: 72,
  temperature: 1.0,
  seed: null,
  useSeedLock: false,
  strokeThickness: 50,
  baselineJitter: "low",
  letterSpacing: 50,
  wordSpacing: 50,
  lineSpacing: 1.35,
  pageStyle: "lined",
  paperTexture: "subtle",
  debugOverlay: false,
  showFallbackMarkers: true,
  showLetterBoxes: true,
  showLetterLabels: false,
  showTimeline: false,
  fitToWidth: true,
  zoom: 1,
  reducedMotion: false
};

export const DEFAULT_FORM_STATE: GenerateFormState = {
  text: "hello world\nthis is a test",
  style: "ink",
  width: 1024,
  height: 640,
  settings: DEFAULT_UI_SETTINGS
};


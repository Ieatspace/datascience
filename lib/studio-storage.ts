import type { GeneratePreset } from "@/lib/studio-types";

const PRESET_KEY = "handwriting_presets";

export function loadPresets(): GeneratePreset[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(PRESET_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as GeneratePreset[]) : [];
  } catch {
    return [];
  }
}

export function savePresets(presets: GeneratePreset[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PRESET_KEY, JSON.stringify(presets.slice(0, 30)));
  } catch {
    // ignore
  }
}


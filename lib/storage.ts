import { z } from "zod";

import { generateResponseSchema, type GenerateResponse } from "@/lib/types";

const HISTORY_STORAGE_KEY = "handwrite-studio.history.v1";
const MAX_HISTORY_ITEMS = 10;

const generationHistorySchema = z.array(generateResponseSchema);

export function getHistoryStorageKey() {
  return HISTORY_STORAGE_KEY;
}

export function loadGenerationHistory(): GenerateResponse[] {
  if (typeof window === "undefined") {
    return [];
  }

  const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsedJson: unknown = JSON.parse(raw);
    const parsed = generationHistorySchema.safeParse(parsedJson);
    if (!parsed.success) {
      return [];
    }
    return parsed.data;
  } catch {
    return [];
  }
}

export function saveGenerationHistory(items: GenerateResponse[]): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
  } catch {
    // Ignore quota/storage errors to keep the UI functional.
  }
}

export function clearGenerationHistory(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch {
    // Ignore storage errors to keep the UI functional.
  }
}

export function pushGenerationHistoryItem(
  items: GenerateResponse[],
  item: GenerateResponse
): GenerateResponse[] {
  const deduped = items.filter((existing) => existing.id !== item.id);
  return [item, ...deduped].slice(0, MAX_HISTORY_ITEMS);
}

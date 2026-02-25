export type ThemeDefinition = {
  id: string;
  name: string;
  cssVars: Record<string, string>;
};

export const THEME_STORAGE_KEY = "handwriting_theme";

export const THEMES: ThemeDefinition[] = [
  {
    id: "light-classic",
    name: "Light Classic",
    cssVars: {
      "--bg": "#f5f7fb",
      "--panel": "rgba(255,255,255,0.78)",
      "--panel-strong": "#ffffff",
      "--text": "#0f172a",
      "--muted": "#64748b",
      "--border": "rgba(148,163,184,0.22)",
      "--accent": "#2563eb",
      "--accent2": "#0ea5e9",
      "--shadow": "rgba(15,23,42,0.08)",
      "--app-gradient":
        "radial-gradient(circle at 10% 8%, rgba(37,99,235,0.16), transparent 42%), radial-gradient(circle at 90% 14%, rgba(14,165,233,0.12), transparent 38%), linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)"
    }
  },
  {
    id: "dark-carbon",
    name: "Dark Carbon",
    cssVars: {
      "--bg": "#090b0f",
      "--panel": "rgba(20,24,31,0.76)",
      "--panel-strong": "#12161d",
      "--text": "#e5e7eb",
      "--muted": "#94a3b8",
      "--border": "rgba(148,163,184,0.18)",
      "--accent": "#60a5fa",
      "--accent2": "#34d399",
      "--shadow": "rgba(0,0,0,0.35)",
      "--app-gradient":
        "radial-gradient(circle at 15% 10%, rgba(96,165,250,0.12), transparent 40%), radial-gradient(circle at 85% 18%, rgba(52,211,153,0.10), transparent 34%), linear-gradient(180deg, #090b0f 0%, #0c1016 100%)"
    }
  },
  {
    id: "midnight-purple",
    name: "Midnight Purple",
    cssVars: {
      "--bg": "#0f1020",
      "--panel": "rgba(27,23,50,0.78)",
      "--panel-strong": "#1b1732",
      "--text": "#efecff",
      "--muted": "#b5afd9",
      "--border": "rgba(167,139,250,0.18)",
      "--accent": "#a78bfa",
      "--accent2": "#22d3ee",
      "--shadow": "rgba(7,5,20,0.45)",
      "--app-gradient":
        "radial-gradient(circle at 12% 12%, rgba(167,139,250,0.22), transparent 42%), radial-gradient(circle at 90% 8%, rgba(34,211,238,0.15), transparent 40%), linear-gradient(180deg, #0b0d1a 0%, #12122a 100%)"
    }
  },
  {
    id: "space-blue",
    name: "Space Blue",
    cssVars: {
      "--bg": "#071423",
      "--panel": "rgba(10,23,40,0.76)",
      "--panel-strong": "#0b1a2c",
      "--text": "#e7f0ff",
      "--muted": "#9fb4d7",
      "--border": "rgba(96,165,250,0.18)",
      "--accent": "#38bdf8",
      "--accent2": "#818cf8",
      "--shadow": "rgba(2,6,23,0.45)",
      "--app-gradient":
        "radial-gradient(circle at 18% 6%, rgba(56,189,248,0.18), transparent 36%), radial-gradient(circle at 82% 10%, rgba(129,140,248,0.15), transparent 40%), linear-gradient(180deg, #071423 0%, #0a1b31 100%)"
    }
  },
  {
    id: "nebula-violet",
    name: "Nebula Violet",
    cssVars: {
      "--bg": "#140d1d",
      "--panel": "rgba(31,17,44,0.78)",
      "--panel-strong": "#241433",
      "--text": "#f3ebff",
      "--muted": "#c4afd7",
      "--border": "rgba(232,121,249,0.17)",
      "--accent": "#e879f9",
      "--accent2": "#8b5cf6",
      "--shadow": "rgba(10,5,15,0.5)",
      "--app-gradient":
        "radial-gradient(circle at 12% 14%, rgba(232,121,249,0.18), transparent 42%), radial-gradient(circle at 88% 16%, rgba(139,92,246,0.20), transparent 40%), linear-gradient(180deg, #120b1b 0%, #1c1129 100%)"
    }
  },
  {
    id: "cyber-neon",
    name: "Cyber Neon",
    cssVars: {
      "--bg": "#05070a",
      "--panel": "rgba(11,17,20,0.78)",
      "--panel-strong": "#0a1013",
      "--text": "#ecfeff",
      "--muted": "#99aab5",
      "--border": "rgba(45,212,191,0.18)",
      "--accent": "#2dd4bf",
      "--accent2": "#a3e635",
      "--shadow": "rgba(0,0,0,0.44)",
      "--app-gradient":
        "radial-gradient(circle at 10% 8%, rgba(45,212,191,0.16), transparent 36%), radial-gradient(circle at 85% 12%, rgba(163,230,53,0.10), transparent 32%), linear-gradient(180deg, #05070a 0%, #090d11 100%)"
    }
  },
  {
    id: "solar-flare",
    name: "Solar Flare",
    cssVars: {
      "--bg": "#1b0f09",
      "--panel": "rgba(39,20,11,0.76)",
      "--panel-strong": "#29160d",
      "--text": "#fff1e6",
      "--muted": "#f1b897",
      "--border": "rgba(251,146,60,0.20)",
      "--accent": "#fb923c",
      "--accent2": "#facc15",
      "--shadow": "rgba(17,5,1,0.44)",
      "--app-gradient":
        "radial-gradient(circle at 14% 8%, rgba(251,146,60,0.22), transparent 42%), radial-gradient(circle at 82% 12%, rgba(250,204,21,0.14), transparent 40%), linear-gradient(180deg, #180d07 0%, #22120a 100%)"
    }
  },
  {
    id: "ocean-depths",
    name: "Ocean Depths",
    cssVars: {
      "--bg": "#06161a",
      "--panel": "rgba(8,32,37,0.78)",
      "--panel-strong": "#0a2329",
      "--text": "#e6fffb",
      "--muted": "#95c5c0",
      "--border": "rgba(20,184,166,0.18)",
      "--accent": "#14b8a6",
      "--accent2": "#0ea5e9",
      "--shadow": "rgba(1,10,12,0.46)",
      "--app-gradient":
        "radial-gradient(circle at 14% 10%, rgba(20,184,166,0.18), transparent 40%), radial-gradient(circle at 84% 14%, rgba(14,165,233,0.14), transparent 36%), linear-gradient(180deg, #06161a 0%, #082026 100%)"
    }
  },
  {
    id: "forest-night",
    name: "Forest Night",
    cssVars: {
      "--bg": "#0a140f",
      "--panel": "rgba(18,31,22,0.78)",
      "--panel-strong": "#15231a",
      "--text": "#eefcf1",
      "--muted": "#a6c7af",
      "--border": "rgba(74,222,128,0.18)",
      "--accent": "#4ade80",
      "--accent2": "#34d399",
      "--shadow": "rgba(3,8,5,0.5)",
      "--app-gradient":
        "radial-gradient(circle at 12% 10%, rgba(74,222,128,0.15), transparent 36%), radial-gradient(circle at 86% 18%, rgba(52,211,153,0.12), transparent 34%), linear-gradient(180deg, #09130e 0%, #0e1a13 100%)"
    }
  },
  {
    id: "rose-quartz",
    name: "Rose Quartz",
    cssVars: {
      "--bg": "#fff5fb",
      "--panel": "rgba(255,255,255,0.8)",
      "--panel-strong": "#ffffff",
      "--text": "#3a1f34",
      "--muted": "#8f6782",
      "--border": "rgba(244,114,182,0.18)",
      "--accent": "#f472b6",
      "--accent2": "#fb7185",
      "--shadow": "rgba(76,29,64,0.08)",
      "--app-gradient":
        "radial-gradient(circle at 14% 10%, rgba(244,114,182,0.12), transparent 42%), radial-gradient(circle at 88% 14%, rgba(251,113,133,0.10), transparent 38%), linear-gradient(180deg, #fff7fb 0%, #fff1f2 100%)"
    }
  },
  {
    id: "sunset-gradient",
    name: "Sunset Gradient",
    cssVars: {
      "--bg": "#1a1220",
      "--panel": "rgba(34,23,43,0.74)",
      "--panel-strong": "#271b33",
      "--text": "#fff5f2",
      "--muted": "#f1bca9",
      "--border": "rgba(251,146,60,0.20)",
      "--accent": "#fb7185",
      "--accent2": "#fb923c",
      "--shadow": "rgba(15,5,20,0.4)",
      "--app-gradient":
        "radial-gradient(circle at 10% 5%, rgba(251,113,133,0.16), transparent 35%), radial-gradient(circle at 90% 12%, rgba(251,146,60,0.18), transparent 40%), linear-gradient(180deg, #1a1220 0%, #312038 60%, #3b241d 100%)"
    }
  },
  {
    id: "arctic-frost",
    name: "Arctic Frost",
    cssVars: {
      "--bg": "#edf8ff",
      "--panel": "rgba(255,255,255,0.78)",
      "--panel-strong": "#ffffff",
      "--text": "#08263d",
      "--muted": "#5a7f95",
      "--border": "rgba(56,189,248,0.18)",
      "--accent": "#0ea5e9",
      "--accent2": "#22c55e",
      "--shadow": "rgba(8,38,61,0.07)",
      "--app-gradient":
        "radial-gradient(circle at 16% 10%, rgba(56,189,248,0.10), transparent 40%), radial-gradient(circle at 88% 18%, rgba(14,165,233,0.08), transparent 34%), linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%)"
    }
  },
  {
    id: "mocha-ink",
    name: "Mocha Ink",
    cssVars: {
      "--bg": "#151110",
      "--panel": "rgba(31,24,22,0.78)",
      "--panel-strong": "#221a18",
      "--text": "#f8eee8",
      "--muted": "#c0a89a",
      "--border": "rgba(180,83,9,0.18)",
      "--accent": "#f59e0b",
      "--accent2": "#f97316",
      "--shadow": "rgba(8,4,3,0.46)",
      "--app-gradient":
        "radial-gradient(circle at 12% 10%, rgba(245,158,11,0.12), transparent 34%), radial-gradient(circle at 87% 12%, rgba(249,115,22,0.12), transparent 34%), linear-gradient(180deg, #151110 0%, #1b1413 100%)"
    }
  },
  {
    id: "terminal-green",
    name: "Terminal Green",
    cssVars: {
      "--bg": "#08110a",
      "--panel": "rgba(10,22,12,0.8)",
      "--panel-strong": "#0d1b10",
      "--text": "#eaffef",
      "--muted": "#97bd9f",
      "--border": "rgba(34,197,94,0.2)",
      "--accent": "#22c55e",
      "--accent2": "#84cc16",
      "--shadow": "rgba(2,8,3,0.52)",
      "--app-gradient":
        "radial-gradient(circle at 12% 8%, rgba(34,197,94,0.14), transparent 34%), radial-gradient(circle at 85% 18%, rgba(132,204,22,0.12), transparent 32%), linear-gradient(180deg, #08110a 0%, #0b160d 100%)"
    }
  },
  {
    id: "lavender-mist",
    name: "Lavender Mist",
    cssVars: {
      "--bg": "#f6f2ff",
      "--panel": "rgba(255,255,255,0.8)",
      "--panel-strong": "#ffffff",
      "--text": "#261b42",
      "--muted": "#7e72a5",
      "--border": "rgba(167,139,250,0.18)",
      "--accent": "#8b5cf6",
      "--accent2": "#c084fc",
      "--shadow": "rgba(38,27,66,0.08)",
      "--app-gradient":
        "radial-gradient(circle at 16% 10%, rgba(139,92,246,0.10), transparent 42%), radial-gradient(circle at 88% 16%, rgba(192,132,252,0.09), transparent 36%), linear-gradient(180deg, #faf5ff 0%, #f3e8ff 100%)"
    }
  },
  {
    id: "crimson-noir",
    name: "Crimson Noir",
    cssVars: {
      "--bg": "#12080b",
      "--panel": "rgba(28,13,17,0.8)",
      "--panel-strong": "#1c0d11",
      "--text": "#fff0f2",
      "--muted": "#d6a5ab",
      "--border": "rgba(244,63,94,0.22)",
      "--accent": "#f43f5e",
      "--accent2": "#fb7185",
      "--shadow": "rgba(8,2,4,0.55)",
      "--app-gradient":
        "radial-gradient(circle at 12% 10%, rgba(244,63,94,0.16), transparent 36%), radial-gradient(circle at 88% 18%, rgba(251,113,133,0.12), transparent 34%), linear-gradient(180deg, #12080b 0%, #1a0c11 100%)"
    }
  }
];

export const DEFAULT_THEME_ID = THEMES[0].id;

export function getThemeById(id: string | null | undefined): ThemeDefinition {
  return THEMES.find((t) => t.id === id) ?? THEMES[0];
}


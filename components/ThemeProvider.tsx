"use client";

import {
  createContext,
  type PropsWithChildren,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";

import {
  DEFAULT_THEME_ID,
  getThemeById,
  THEME_STORAGE_KEY,
  THEMES,
  type ThemeDefinition
} from "@/lib/themes";

type ThemeContextValue = {
  theme: ThemeDefinition;
  themes: ThemeDefinition[];
  setThemeId: (id: string) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

function applyThemeVars(theme: ThemeDefinition) {
  if (typeof document === "undefined") {
    return;
  }
  const root = document.documentElement;
  Object.entries(theme.cssVars).forEach(([key, value]) => {
    root.style.setProperty(key, value);
  });
}

export function ThemeProvider({ children }: PropsWithChildren) {
  const [themeId, setThemeIdState] = useState(DEFAULT_THEME_ID);

  useEffect(() => {
    const stored =
      typeof window !== "undefined"
        ? window.localStorage.getItem(THEME_STORAGE_KEY)
        : null;
    const resolved = getThemeById(stored);
    setThemeIdState(resolved.id);
    applyThemeVars(resolved);
  }, []);

  const setThemeId = (id: string) => {
    const next = getThemeById(id);
    setThemeIdState(next.id);
    applyThemeVars(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, next.id);
    }
  };

  const value = useMemo<ThemeContextValue>(() => {
    const theme = getThemeById(themeId);
    return { theme, themes: THEMES, setThemeId };
  }, [themeId]);

  return (
    <ThemeContext.Provider value={value}>
      <div className="theme-shell min-h-screen transition-colors duration-500">
        {children}
      </div>
    </ThemeContext.Provider>
  );
}

export function useThemeManager() {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useThemeManager must be used within ThemeProvider");
  }
  return ctx;
}


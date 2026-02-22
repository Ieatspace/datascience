import { z } from "zod";

export const MAX_TEXT_LENGTH = 2000;

export const handwritingStyleSchema = z.enum(["pencil", "ink", "marker"]);

export const generateRequestSchema = z.object({
  text: z
    .string()
    .max(MAX_TEXT_LENGTH)
    .refine((value) => value.trim().length > 0, "Text is required"),
  style: handwritingStyleSchema,
  width: z.number().int().min(320).max(2048),
  height: z.number().int().min(240).max(2048),
  lineSpacing: z.number().min(0.8).max(3),
  seed: z
    .number()
    .int()
    .min(0)
    .max(2_147_483_647)
    .nullable()
});

export const generateResponseSchema = z.object({
  id: z.string().min(1),
  createdAt: z
    .string()
    .min(1)
    .refine((value) => !Number.isNaN(Date.parse(value)), {
      message: "createdAt must be a valid ISO date string"
    }),
  request: generateRequestSchema,
  imageDataUrl: z.string().startsWith("data:image/png;base64,")
});

export const apiErrorResponseSchema = z.object({
  error: z.object({
    message: z.string(),
    details: z.unknown().optional()
  })
});

export type HandwritingStyle = z.infer<typeof handwritingStyleSchema>;
export type GenerateRequest = z.infer<typeof generateRequestSchema>;
export type GenerateResponse = z.infer<typeof generateResponseSchema>;
export type ApiErrorResponse = z.infer<typeof apiErrorResponseSchema>;

export const STYLE_OPTIONS: Array<{ label: string; value: HandwritingStyle }> = [
  { label: "Pencil", value: "pencil" },
  { label: "Ink", value: "ink" },
  { label: "Marker", value: "marker" }
];

export const SIZE_PRESETS = [
  { label: "Notebook (1024 x 512)", width: 1024, height: 512 },
  { label: "Letter Wide (1280 x 720)", width: 1280, height: 720 },
  { label: "Tall Note (960 x 1280)", width: 960, height: 1280 }
] as const;

export const DEFAULT_GENERATE_REQUEST: GenerateRequest = {
  text: "The quick brown fox jumps over the lazy dog.",
  style: "ink",
  width: SIZE_PRESETS[0].width,
  height: SIZE_PRESETS[0].height,
  lineSpacing: 1.35,
  seed: null
};

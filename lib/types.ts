import { z } from "zod";

export const MAX_TEXT_LENGTH = 2000;

export const handwritingStyleSchema = z.enum(["pencil", "ink", "marker"]);
export const pageStyleSchema = z.enum(["blank", "lined", "grid", "dot"]);
export const paperTextureSchema = z.enum(["off", "subtle", "med"]);

export const generateRequestSchema = z.object({
  text: z
    .string()
    .max(MAX_TEXT_LENGTH)
    .refine((value) => value.trim().length > 0, "Text is required"),
  style: handwritingStyleSchema,
  width: z.number().int().min(320).max(2048),
  height: z.number().int().min(240).max(2048),
  lineSpacing: z.number().min(0.8).max(3),
  seed: z.number().int().nullable().optional(),
  pageStyle: pageStyleSchema.optional(),
  paperTexture: paperTextureSchema.optional(),
  letterModelEnabled: z.boolean().optional(),
  letterModelStyleStrength: z.number().min(0.25).max(3).optional(),
  letterModelBaselineJitter: z.number().min(0).max(3).optional(),
  letterModelWordSlant: z.number().min(0).max(3).optional(),
  letterModelRotationJitter: z.number().min(0).max(3).optional(),
  letterModelInkVariation: z.number().min(0).max(1).optional()
});

export const letterDebugMetaSchema = z.object({
  char: z.string().min(1),
  bbox: z.object({
    x: z.number().int(),
    y: z.number().int(),
    w: z.number().int().positive(),
    h: z.number().int().positive()
  }),
  source: z.enum(["generated", "fallback"]),
  seedUsed: z.number().int().optional(),
  confidence: z.number().min(0).max(1).optional()
});

export const generateMetaSchema = z.object({
  letters: z.array(letterDebugMetaSchema),
  stats: z
    .object({
      handChars: z.number().int().min(0).optional(),
      modelChars: z.number().int().min(0).optional(),
      cropChars: z.number().int().min(0).optional(),
      fallbackChars: z.number().int().min(0).optional(),
      joins: z.number().int().min(0).optional()
    })
    .partial()
    .optional(),
  warnings: z.array(z.string()).optional(),
  seedUsed: z.number().int().optional(),
  fallbackInfoAvailable: z.boolean().optional()
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
  imageDataUrl: z.string().startsWith("data:image/png;base64,"),
  meta: generateMetaSchema.optional()
});

export const recognizePageRequestSchema = z.object({
  dotted: z.boolean().default(false),
  strict: z.boolean().default(false),
  softDotMerge: z.boolean().default(false),
  noSpaces: z.boolean().default(false),
  includeDebugImage: z.boolean().default(true),
  topk: z.number().int().min(1).max(5).default(1)
});

export const recognizeTopKPredictionSchema = z.object({
  label: z.string().min(1),
  conf: z.number().min(0).max(1)
});

export const recognizeCharPredictionSchema = z.object({
  x: z.number().int(),
  y: z.number().int(),
  w: z.number().int().positive(),
  h: z.number().int().positive(),
  pred: z.string().min(1),
  conf: z.number().min(0).max(1),
  topk: z.array(recognizeTopKPredictionSchema).optional()
});

export const recognizeLineSchema = z.object({
  index: z.number().int().min(0),
  text: z.string(),
  chars: z.array(recognizeCharPredictionSchema)
});

export const recognizeResponseSchema = z.object({
  text: z.string(),
  lines: z.array(recognizeLineSchema),
  detectedCharacters: z.number().int().min(0),
  averageConfidence: z.number().min(0).max(1).nullable().optional(),
  extractor: z.object({
    gentle: z.boolean(),
    dotted: z.boolean(),
    softDotMerge: z.boolean()
  }),
  debugImageDataUrl: z.string().startsWith("data:image/png;base64,").nullable().optional(),
  imageName: z.string().min(1),
  topk: z.number().int().min(1).max(5)
});

export const apiErrorResponseSchema = z.object({
  error: z.object({
    message: z.string(),
    details: z.unknown().optional()
  })
});

export const trainingStatusSchema = z.object({
  version: z.string(),
  datasetSize: z.number().int().min(0),
  lastTrained: z.string().nullable(),
  status: z.enum(["idle", "training", "error"]),
  progress: z
    .object({
      epoch: z.number().int().min(0).optional(),
      totalEpochs: z.number().int().min(0).nullable().optional(),
      etaSeconds: z.number().int().min(0).nullable().optional(),
      percent: z.number().min(0).max(100).nullable().optional(),
      loss: z.number().nullable().optional(),
      valLoss: z.number().nullable().optional()
    })
    .optional()
});

export const datasetLetterStatSchema = z.object({
  letter: z.string().length(1),
  count: z.number().int().min(0),
  thumbnails: z.array(z.string()).optional()
});

export const datasetStatsSchema = z.object({
  datasetSize: z.number().int().min(0),
  undertrainedThreshold: z.number().int().min(1),
  generatedAt: z.string(),
  letters: z.array(datasetLetterStatSchema),
  confusingPairs: z.array(z.tuple([z.string(), z.string()])).default([])
});

export type HandwritingStyle = z.infer<typeof handwritingStyleSchema>;
export type PageStyle = z.infer<typeof pageStyleSchema>;
export type PaperTexture = z.infer<typeof paperTextureSchema>;
export type GenerateRequest = z.infer<typeof generateRequestSchema>;
export type GenerateResponse = z.infer<typeof generateResponseSchema>;
export type ApiErrorResponse = z.infer<typeof apiErrorResponseSchema>;
export type RecognizePageRequest = z.infer<typeof recognizePageRequestSchema>;
export type RecognizeResponse = z.infer<typeof recognizeResponseSchema>;
export type GenerateMeta = z.infer<typeof generateMetaSchema>;
export type LetterDebugMeta = z.infer<typeof letterDebugMetaSchema>;
export type TrainingStatus = z.infer<typeof trainingStatusSchema>;
export type DatasetStats = z.infer<typeof datasetStatsSchema>;

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

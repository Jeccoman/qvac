import { z } from 'zod'

/** Each candidate fetches a registry fit stub, so the list size bounds network work. */
export const MAX_FIT_MODELS = 16

const fitCandidate = z.strictObject({
  model: z
    .string()
    .min(1)
    .describe(
      'A `serve.models` alias, or an SDK model constant name such as `QWEN3_600M_INST_Q4`.'
    ),
  modelConfig: z
    .record(z.string(), z.unknown())
    .optional()
    .describe(
      "Config to assess the load with. Merged over an alias's own `config`, nested objects key by key. Companion model sources (`*ModelSrc`) must name SDK model constants."
    )
})

/** Mirrors the SDK's `assessModelFit()` input, with models named rather than sourced. */
export const modelFitBody = z.strictObject({
  models: z
    .array(fitCandidate)
    .min(1)
    .max(MAX_FIT_MODELS)
    .describe('Models to assess as if loaded at the same time.')
})

const verdict = z.enum(['fits', 'does-not-fit', 'unknown'])

// Loose, so a field the SDK adds later still reaches the client through the
// response serializer.
const modelResult = z.looseObject({
  model: z.string().describe('The alias or constant name as requested.'),
  verdict: verdict.describe("The engine fitter's verdict for this model alone."),
  stats: z
    .record(z.string(), z.unknown())
    .describe("Figures specific to the engine, as its addon's `FitStats` type declares them.")
})

export const modelFitResult = z.looseObject({
  object: z.literal('model_fit'),
  verdict: verdict.describe('The verdict for every model loaded together.'),
  models: z.array(modelResult).describe('One entry per model, in request order.')
})

export const MODEL_FIT_FIELD_CODES: Record<string, string> = {
  models: 'invalid_models',
  'models:too_big': 'too_many_models'
}

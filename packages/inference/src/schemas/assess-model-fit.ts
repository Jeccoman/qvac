import { z } from 'zod'
import { modelSrcInputSchema } from '@/schemas/model-src-utils'
import { modelTypeInputSchema } from '@/schemas/model-types'
import { transcribeParamsSchema } from '@/schemas/transcription'

export const transcriptionFitWorkloadSchema = transcribeParamsSchema
  .pick({ prompt: true, hotwords: true, maxNewTokens: true })
  .extend({ audioSeconds: z.number().positive().describe('Longest whole recording to project.') })
  .strict()
  .refine((value) => value.prompt === undefined || !value.hotwords?.length, {
    message: 'MOSS prompt cannot be combined with hotwords'
  })

export type TranscriptionFitWorkload = z.infer<typeof transcriptionFitWorkloadSchema>

/**
 * `unknown` is the answer for a load that would start but cannot be sized: the
 * engine has no fitter, cannot project this load, has no registry description
 * of a model that is not on disk, or a device reported no free memory.
 */
export const modelFitVerdictSchema = z.enum(['fits', 'does-not-fit', 'unknown'])

/** Fit-only workload used to estimate MOSS text-to-speech memory requirements. */
export const ttsFitWorkloadSchema = z.union([
  z
    .strictObject({
      promptRows: z
        .number()
        .int()
        .positive()
        .max(2147483647)
        .describe('Complete native prompt rows, including dialogue and references.'),
      referenceSamples: z
        .number()
        .int()
        .nonnegative()
        .max(2147483647)
        .describe('Total mono reference audio samples.'),
      streaming: z.boolean().describe('Whether the MOSS workload uses native streaming.')
    })
    .meta({ title: 'MossTtsFitWorkload' }),
  z
    .strictObject({
      prompt: z.string().min(1).describe('Sound-effect description.'),
      seconds: z.number().positive().max(30).describe('Sound-effect duration in seconds, up to 30.')
    })
    .meta({ title: 'MossSoundEffectFitWorkload' })
])

export type TtsFitWorkload = z.infer<typeof ttsFitWorkloadSchema>

/**
 * A load to assess, in `loadModel`'s own parameters. The engine's plugin
 * resolves it, so the config and the artifact keys the fitter reads are the
 * ones the load would run with.
 *
 * Each source resolves to the registry's weightless description of the
 * artifact, or to the file itself where it is already on disk.
 */
export const modelFitCandidateSchema = z.object({
  transcriptionWorkload: transcriptionFitWorkloadSchema
    .optional()
    .describe('MOSS fit-only workload. Required for a MOSS verdict; does not configure loading.'),
  modelSrc: modelSrcInputSchema
    .optional()
    .describe(
      'The model to assess, as `loadModel` takes it. Optional for the loads `loadModel` also takes without one, where every source is a config field.'
    ),
  modelType: modelTypeInputSchema
    .optional()
    .describe(
      'Engine that would run the load. Inferred from `modelSrc` when omitted, as `loadModel` infers it; required only where the source does not name one.'
    ),
  ttsWorkload: ttsFitWorkloadSchema
    .optional()
    .describe(
      'Fit-only MOSS-TTS/TTSD or MOSS-SoundEffect workload; never passed to model loading.'
    ),
  modelConfig: z
    .record(z.string(), z.unknown())
    .optional()
    .describe('The config `loadModel` would be given, including any companion model sources.')
})

export const assessModelFitInputSchema = z.object({
  models: z
    .array(modelFitCandidateSchema)
    .min(1)
    .describe('Models to assess as if loaded at the same time.')
})

export const modelFitStatsSchema = z
  .record(z.string(), z.unknown())
  .describe("Figures specific to the engine, as its addon's `FitStats` type declares them.")

export const modelFitModelResultSchema = z
  .object({
    verdict: modelFitVerdictSchema.describe("The engine fitter's verdict for this model alone."),
    stats: modelFitStatsSchema
  })
  .meta({ title: 'ModelFitModelResult' })

export const assessModelFitResultSchema = z.object({
  verdict: modelFitVerdictSchema.describe('The verdict for every model loaded together.'),
  models: z.array(modelFitModelResultSchema).describe('One entry per model, in input order.')
})

/** What `getLoadedModelInfo` reports of the fit check that ran ahead of a load. */
export const modelFitProbeSchema = modelFitModelResultSchema
  .extend({
    verdict: modelFitVerdictSchema.describe('The verdict for this load against free memory.')
  })
  .meta({ title: 'ModelFitProbe' })

export const assessModelFitRequestSchema = assessModelFitInputSchema.extend({
  type: z.literal('assessModelFit')
})

export const assessModelFitResponseSchema = assessModelFitResultSchema.extend({
  type: z.literal('assessModelFit')
})

export type ModelFitVerdict = z.infer<typeof modelFitVerdictSchema>
export type ModelFitStats = z.infer<typeof modelFitStatsSchema>
export type ModelFitCandidate = z.infer<typeof modelFitCandidateSchema>
export type ModelFitModelResult = z.infer<typeof modelFitModelResultSchema>
export type ModelFitProbe = z.infer<typeof modelFitProbeSchema>
export type AssessModelFitInput = z.input<typeof assessModelFitInputSchema>
export type AssessModelFitResult = z.infer<typeof assessModelFitResultSchema>
export type AssessModelFitRequest = z.infer<typeof assessModelFitRequestSchema>
export type AssessModelFitResponse = z.infer<typeof assessModelFitResponseSchema>

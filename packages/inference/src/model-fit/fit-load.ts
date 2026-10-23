import type { AbortSignal } from 'bare-abort-controller'

import type { AddonFitDevice } from '@/model-fit/run/engine'
import { runFit as runFitDefault } from '@/model-fit/run/run'
import type { ModelFitStats, ModelFitVerdict } from '@/schemas/assess-model-fit'
import { ModelType } from '@/schemas/index'
import type { FitLoad } from '@/schemas/plugin'

export interface LoadFit {
  verdict: ModelFitVerdict
  devices: AddonFitDevice[]
  stats: ModelFitStats
  /** Why the verdict is `unknown`, for the log. */
  reason?: string
}

export interface FitLoadOptions {
  signal?: AbortSignal
  runFit?: typeof runFitDefault
}

function unknown(reason: string): LoadFit {
  return { verdict: 'unknown', devices: [], stats: {}, reason }
}

/**
 * The fitter's answer for one load, through the plugin of the engine that
 * would run it.
 *
 * @throws The fitter's own error when it rejects the load, such as a model it
 *   cannot read or a request it refuses.
 */
export async function fitLoad(
  modelType: string,
  load: FitLoad,
  options: FitLoadOptions = {}
): Promise<LoadFit> {
  const { getPlugin } = await import('@/plugins/registry')
  const plugin = getPlugin(modelType)
  if (plugin?.fitRequest === undefined) return unknown(`no fitter for model type ${modelType}`)
  if (load.transcriptionWorkload !== undefined && plugin.modelType !== ModelType.mossTranscribe) {
    return unknown('a transcriptionWorkload sizes MOSS transcription only')
  }

  const plan = plugin.fitRequest(load)
  if ('unknown' in plan) return unknown(plan.unknown)

  const result = await (options.runFit ?? runFitDefault)(
    { addonPackage: plugin.addonPackage, request: plan.request },
    options.signal !== undefined ? { signal: options.signal } : {}
  )
  if (result.status === 'unknown') return unknown(`${result.reason}: ${result.message}`)

  return { verdict: result.fit.status, devices: result.fit.devices, stats: result.fit.stats }
}

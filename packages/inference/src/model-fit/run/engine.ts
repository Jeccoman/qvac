import type { QvacPlugin } from '@/schemas/plugin'

/** One device a load uses, as every addon's `assessFit` reports it. */
export interface AddonFitDevice {
  /** The PCI bus ID when the backend reports one, the device name otherwise. */
  id: string
  name: string
  /** Memory the host owns: the host CPU, and a GPU that shares the host's memory. */
  unified: boolean
  usedBytes: number
  freeBytes?: number
  totalBytes?: number
}

/** The result every addon's `assessFit` returns. */
export interface AddonFit {
  status: 'fits' | 'does-not-fit' | 'unknown'
  devices: AddonFitDevice[]
  stats: Record<string, unknown>
}

/** A request for the fitter of the plugin that owns `addonPackage`. */
export interface FitProbeRequest {
  addonPackage: string
  request: unknown
}

/**
 * Calls the request's addon fitter. Several plugins can share one addon, and
 * only those that expose its `assessFit` carry it.
 */
export async function callEngineFit(probe: FitProbeRequest): Promise<AddonFit> {
  const { getAllPlugins } = await import('@/plugins/registry')
  const plugin: QvacPlugin | undefined = getAllPlugins().find(
    (candidate) =>
      candidate.addonPackage === probe.addonPackage && candidate.assessFit !== undefined
  )

  if (plugin?.assessFit === undefined) {
    throw new TypeError(`no registered plugin exposes assessFit for ${probe.addonPackage}`)
  }

  return (await plugin.assessFit(probe.request as never)) as AddonFit
}

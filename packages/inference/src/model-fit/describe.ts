import type { AddonFitDevice } from '@/model-fit/run/engine'
import type { PoolUse } from '@/model-fit/sum'

const BYTES_PER_MIB = 1024 * 1024

export function mib(bytes: number): string {
  return `${Math.round(bytes / BYTES_PER_MIB)} MiB`
}

export function describePool(pool: PoolUse): string {
  return pool.freeBytes === undefined
    ? `${pool.pool} ${mib(pool.usedBytes)}`
    : `${pool.pool} ${mib(pool.usedBytes)} of ${mib(pool.freeBytes)} free`
}

function describeDevice(device: AddonFitDevice): string {
  const figures = [
    `used ${mib(device.usedBytes)}`,
    ...(device.freeBytes !== undefined ? [`free ${mib(device.freeBytes)}`] : []),
    ...(device.totalBytes !== undefined ? [`total ${mib(device.totalBytes)}`] : [])
  ]
  const kind = device.unified ? 'unified' : 'own memory'
  return `${device.name} [${device.id}, ${kind}] ${figures.join(', ')}`
}

/**
 * Every pool and every row the addons reported, so a log shows which device
 * joined which pool and which figure bound it.
 */
export function describeAccounting(
  pools: readonly PoolUse[],
  rows: readonly { label: string; devices: readonly AddonFitDevice[] }[]
): string {
  const poolText = pools.length > 0 ? pools.map(describePool).join('; ') : 'none'
  const rowText = rows
    .map(({ label, devices }) =>
      devices.length > 0 ? `${label}: ${devices.map(describeDevice).join('; ')}` : `${label}: none`
    )
    .join(' | ')
  return `pools: ${poolText} — rows: ${rowText}`
}

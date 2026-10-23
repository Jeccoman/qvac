import test from 'brittle'

import { combineFits, HOST_POOL, type ModelAnswer } from '@/model-fit/sum'
import type { AddonFitDevice } from '@/model-fit/run/engine'

const GIB = 1024 ** 3

function host(usedGiB: number, freeGiB?: number): AddonFitDevice {
  return {
    id: 'CPU',
    name: 'CPU',
    unified: true,
    usedBytes: usedGiB * GIB,
    ...(freeGiB !== undefined && { freeBytes: freeGiB * GIB })
  }
}

function gpu(id: string, usedGiB: number, freeGiB?: number, unified = false): AddonFitDevice {
  return {
    id,
    name: id,
    unified,
    usedBytes: usedGiB * GIB,
    ...(freeGiB !== undefined && { freeBytes: freeGiB * GIB })
  }
}

function fits(...devices: AddonFitDevice[]): ModelAnswer {
  return { verdict: 'fits', devices }
}

test('a set that stays within 90% of every pool fits', (t) => {
  const result = combineFits([fits(gpu('0000:01:00.0', 4, 16), host(1, 32))])
  t.is(result.verdict, 'fits')
})

test('models on one device add up against its free memory', (t) => {
  const alone = fits(gpu('0000:01:00.0', 8, 16), host(1, 32))
  t.is(combineFits([alone]).verdict, 'fits')
  t.is(combineFits([alone, alone]).verdict, 'does-not-fit', '16 GiB is over 90% of 16')
})

test('a set over 90% of a pool does not fit, even when every model fits alone', (t) => {
  const result = combineFits([fits(gpu('0000:01:00.0', 15, 16), host(1, 32))])
  t.is(result.verdict, 'does-not-fit')
})

test('a unified GPU and the host CPU share the host pool', (t) => {
  const result = combineFits([fits(gpu('MTL0', 10, 19, true), host(10, 21))])
  const pool = result.pools.find((entry) => entry.pool === HOST_POOL)
  t.is(result.pools.length, 1, 'one pool')
  t.is(pool?.usedBytes, 20 * GIB)
  t.is(pool?.freeBytes, 19 * GIB, 'the lowest free figure')
  t.is(result.verdict, 'does-not-fit')
})

test('the same GPU from two addons is one pool', (t) => {
  const result = combineFits([
    fits(gpu('0000:01:00.0', 5, 16), host(1, 32)),
    fits(gpu('0000:01:00.0', 5, 15), host(1, 32))
  ])
  const device = result.pools.find((entry) => entry.pool === '0000:01:00.0')
  t.is(device?.usedBytes, 10 * GIB)
  t.is(device?.freeBytes, 15 * GIB)
})

test('a pool with no free figure is unknown', (t) => {
  t.is(combineFits([fits(gpu('0000:01:00.0', 1))]).verdict, 'unknown')
})

test('an unknown model makes the set unknown', (t) => {
  const result = combineFits([fits(host(1, 32)), { verdict: 'unknown', devices: [] }])
  t.is(result.verdict, 'unknown')
})

test('a model that does not fit alone makes the set not fit', (t) => {
  const result = combineFits([
    fits(host(1, 32)),
    { verdict: 'does-not-fit', devices: [host(1, 32)] }
  ])
  t.is(result.verdict, 'does-not-fit')
})

test('an overflow stands even beside an unknown model', (t) => {
  const result = combineFits([
    fits(gpu('0000:01:00.0', 15, 16), host(1, 32)),
    { verdict: 'unknown', devices: [] }
  ])
  t.is(result.verdict, 'does-not-fit', 'the unknown model only adds to the sum')
})

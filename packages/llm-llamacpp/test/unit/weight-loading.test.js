'use strict'

const test = require('brittle')
const LlmLlamacpp = require('../../index.js')
const { LlamaInterface } = require('../../addon.js')

const FIRST = '/models/model-00001-of-00003.gguf'

test('constructor rejects the removed streaming forms', async function (t) {
  for (const files of [[FIRST, '/models/model-00002-of-00003.gguf'], [new Uint8Array(8)]]) {
    await t.exception.all(() => new LlmLlamacpp({ files: { model: files } }), /streaming|buffers/)
  }
})

test('disk loading passes only the stored path and never weight buffers', async function (t) {
  const files = [FIRST]
  const model = new LlmLlamacpp({ files: { model: files } })
  files.push('/models/model-00002-of-00003.gguf')
  model._createAddon = function (params) {
    t.is(params.path, FIRST, 'the first path is retained independently of the input array')
    return {
      activate() {
        return Promise.resolve()
      },
      cancel() {
        return Promise.resolve()
      },
      unload() {
        return Promise.resolve()
      },
      loadWeights() {
        t.fail('disk loading must not submit weight buffers')
      }
    }
  }
  await model.load()
  t.ok(model.getState().configLoaded)
  await model.unload()
})

test('legacy loadWeights rejects single files, shards and tensor-name data', async function (t) {
  let submitted = 0
  const binding = {
    createInstance() {
      return {}
    },
    loadWeights() {
      submitted++
    }
  }
  const addon = new LlamaInterface(binding, { path: FIRST, config: {} }, () => {})
  for (const filename of ['model.gguf', 'model-00001-of-00003.gguf', 'model.tensors.txt']) {
    for (const completed of [false, true]) {
      await t.exception(
        () =>
          addon.loadWeights({ filename, chunk: completed ? null : new Uint8Array(8), completed }),
        {
          code: 'ERR_WEIGHT_STREAMING_UNSUPPORTED',
          message: /Pass a disk path/
        }
      )
    }
  }
  t.is(submitted, 0, 'no buffer reaches the native binding')
})

'use strict'

const test = require('brittle')
const path = require('bare-path')
const { GGMLBert, LayaDecisions, Reranker } = require('../../index.js')
const binding = require('../../binding.js')
const { ensureModel } = require('./utils')

test('native weight streaming rejects before inspecting handles or buffers', async function (t) {
  const data = {
    get chunk() {
      throw new Error('removed loader must not inspect buffers')
    }
  }
  await t.exception(() => binding.loadWeights(null, data), {
    code: 'ERR_WEIGHT_STREAMING_UNSUPPORTED',
    message: /first shard path/
  })
})

// prestage-ignore: missing-model-00001-of-10000000.gguf — intentionally missing path for the allocation regression
test('all model classes reject missing disk filenames without expanding oversized shard counts', async function (t) {
  for (const Model of [GGMLBert, LayaDecisions, Reranker]) {
    const model = new Model({
      files: { model: [path.resolve(__dirname, 'missing-model-00001-of-10000000.gguf')] },
      config: { device: 'cpu', gpu_layers: '0' }
    })
    try {
      await t.exception(() => model.load(), /Failed to read GGUF metadata from disk/)
      t.is(model.getState().configLoaded, false)
    } finally {
      await model.unload()
    }
  }
})

test(
  'rejecting weight buffers leaves a loaded disk embedding model usable',
  { timeout: 600_000 },
  async function (t) {
    const [modelName, modelDir] = await ensureModel({ modelName: 'embeddinggemma-300M-Q8_0.gguf' })
    const modelPath = path.join(modelDir, modelName)
    const model = new GGMLBert({
      files: { model: [modelPath] },
      config: { device: 'cpu', gpu_layers: '0', ctx_size: '512' }
    })
    try {
      await model.load()
      await t.exception(
        () =>
          binding.loadWeights(model.addon._handle, {
            filename: path.basename(modelPath),
            chunk: new Uint8Array(8),
            completed: true
          }),
        { code: 'ERR_WEIGHT_STREAMING_UNSUPPORTED' }
      )
      const [vectors] = await (await model.run(['A disk-loaded embedding model.'])).await()
      t.is(vectors.length, 1)
      t.is(vectors[0].length, 768, 'the disk model still produces a complete embedding')
    } finally {
      await model.unload()
    }
  }
)

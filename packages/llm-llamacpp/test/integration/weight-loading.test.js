'use strict'

const test = require('brittle')
const path = require('bare-path')
const LlmLlamacpp = require('../../index.js')
const binding = require('../../binding.js')
const { ensureModelPath } = require('./utils')

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
test('oversized shard count in a missing disk filename does not allocate shard names', async function (t) {
  const model = new LlmLlamacpp({
    files: { model: [path.resolve(__dirname, 'missing-model-00001-of-10000000.gguf')] },
    config: { device: 'cpu', gpu_layers: '0', ctx_size: '256' }
  })
  try {
    await t.exception(() => model.load(), /Failed to read GGUF metadata from disk/)
    t.is(model.getState().configLoaded, false)
  } finally {
    await model.unload()
  }
})

test(
  'rejecting weight buffers leaves a loaded disk model usable',
  { timeout: 600_000 },
  async function (t) {
    const modelPath = await ensureModelPath({ modelName: 'Llama-3.2-1B-Instruct-Q4_0.gguf' })
    const model = new LlmLlamacpp({
      files: { model: [modelPath] },
      config: { device: 'cpu', gpu_layers: '0', ctx_size: '256', n_predict: '16' }
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
      const response = await model.run([{ role: 'user', content: 'Say hello.' }])
      let output = ''
      await response
        .onUpdate((chunk) => {
          output += chunk
        })
        .await()
      t.ok(output.trim().length > 0, 'the disk model still generates output')
    } finally {
      await model.unload()
    }
  }
)

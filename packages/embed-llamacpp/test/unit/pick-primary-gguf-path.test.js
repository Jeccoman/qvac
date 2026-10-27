'use strict'

const test = require('brittle')
const { pickPrimaryGgufPath } = require('../../index.js')

test('single non-sharded file returns that file', function (t) {
  const files = ['/models/bge-small-en-v1.5-q4_0.gguf']
  t.is(pickPrimaryGgufPath(files), '/models/bge-small-en-v1.5-q4_0.gguf')
})

test('removed streaming list with tensors.txt is rejected', async function (t) {
  const files = [
    '/models/big-embed.tensors.txt',
    '/models/big-embed-00001-of-00003.gguf',
    '/models/big-embed-00002-of-00003.gguf',
    '/models/big-embed-00003-of-00003.gguf'
  ]
  await t.exception.all(() => pickPrimaryGgufPath(files), /Weight streaming is no longer supported/)
})

test('removed streaming list without tensors.txt is rejected', async function (t) {
  const files = ['/models/gte-large-00001-of-00002.gguf', '/models/gte-large-00002-of-00002.gguf']
  await t.exception.all(() => pickPrimaryGgufPath(files), /Weight streaming is no longer supported/)
})

test('non-gguf file falls back to first entry', function (t) {
  const files = ['/models/some-model.bin']
  t.is(pickPrimaryGgufPath(files), '/models/some-model.bin')
})

test('first disk shard alone is accepted', function (t) {
  const first = '/models/model-00001-of-00003.gguf'
  t.is(pickPrimaryGgufPath([first]), first)
})

test('weight buffers are rejected as model paths', async function (t) {
  await t.exception.all(
    () => pickPrimaryGgufPath([new Uint8Array(8)]),
    /weight buffers are not supported/
  )
})

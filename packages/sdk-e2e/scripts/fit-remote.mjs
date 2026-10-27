#!/usr/bin/env node
// Builds the nine fit addons from this checkout, installs them into inference,
// the SDK and e2e, and runs the e2e `model-fit` cases on this machine.
//
// Needs a C++ toolchain, the Vulkan SDK, Node with npm, bare-make on PATH and
// VCPKG_ROOT pointing at a vcpkg checkout. The overlays in
// packages/sdk-e2e/fit-overlays build stable-diffusion.cpp and fabric-speech
// from their fit branches.
//
//   node packages/sdk-e2e/scripts/fit-remote.mjs [--skip-build] [--skip-run] [--no-gpu]
//
// --no-gpu runs the cases with every Vulkan device hidden, so each addon sees
// only the CPU. Metal has no such switch.
//
// It rewrites the dependency specs in packages/inference, packages/sdk and
// packages/sdk-e2e package.json to the packed tarballs; `git checkout` restores
// them.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import {
  expectedAddonFile,
  hostToSliceSuffix,
  mangledAddonName,
  slicePackageName
} from '../../../scripts/ci/slice-platform-packages.mjs'

const E2E_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const PACKAGES_DIR = path.resolve(E2E_DIR, '..')
const INFERENCE_DIR = path.join(PACKAGES_DIR, 'inference')
const SDK_DIR = path.join(PACKAGES_DIR, 'sdk')
const ADDONS_OUT = path.join(SDK_DIR, 'addons')
const OVERLAYS_DIR = path.join(E2E_DIR, 'fit-overlays')

// The addon directories, and whether each loads its binary from a platform package.
const ADDONS = [
  { dir: 'llm-llamacpp', platformPackage: false },
  { dir: 'embed-llamacpp', platformPackage: false },
  { dir: 'asr-ggml', platformPackage: true },
  { dir: 'tts-ggml', platformPackage: true },
  { dir: 'bci-whispercpp', platformPackage: false },
  { dir: 'audiogen-ggml', platformPackage: true },
  { dir: 'ocr-ggml', platformPackage: false },
  { dir: 'translation-nmtcpp', platformPackage: false },
  { dir: 'diffusion-cpp', platformPackage: true }
]

const HOST = `${process.platform}-${process.arch}`
const args = new Set(process.argv.slice(2))

// `npm run` exports its own config as npm_config_*, which a child npm would
// honour. npm is a .cmd shim on Windows, which Node only spawns through a shell.
function spawnOptions(cwd, env, stdout) {
  const childEnv = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith('npm_config_'))
  )
  return {
    cwd,
    env: { ...childEnv, ...env },
    shell: process.platform === 'win32',
    stdio: ['inherit', stdout, 'inherit'],
    encoding: 'utf8'
  }
}

/** Runs a command and returns what it printed. */
function run(command, commandArgs, cwd, env = {}) {
  return execFileSync(command, commandArgs, spawnOptions(cwd, env, 'pipe'))
}

/** Runs a command with its output streaming to this terminal. */
function runShown(command, commandArgs, cwd, env = {}) {
  execFileSync(command, commandArgs, spawnOptions(cwd, env, 'inherit'))
}

function step(message) {
  console.log(`\n▶ ${message}`)
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function writeJson(file, value) {
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`)
}

function checkEnvironment() {
  if (!process.env.VCPKG_ROOT) throw new Error('VCPKG_ROOT is not set')
  if (!fs.existsSync(OVERLAYS_DIR)) throw new Error(`no overlays at ${OVERLAYS_DIR}`)
  if (!hostToSliceSuffix(HOST)) throw new Error(`no platform package for host ${HOST}`)
  run('bare-make', ['--version'], PACKAGES_DIR)
}

function buildAddon(addon) {
  step(`Building ${addon.dir}`)
  const cwd = path.join(PACKAGES_DIR, addon.dir)
  runShown('npm', ['install'], cwd)
  runShown('npm', ['run', 'build'], cwd, { VCPKG_OVERLAY_PORTS: OVERLAYS_DIR })
}

function packJson(target) {
  const output = run(
    'npm',
    ['pack', '--ignore-scripts', '--json', '--pack-destination', ADDONS_OUT, target],
    PACKAGES_DIR
  )
  return path.join(ADDONS_OUT, JSON.parse(output)[0].filename)
}

// The platform package build:native stages under node_modules links its
// prebuilds, so it is copied out with real files before packing. The host dir
// also holds the addon's pre-split binary, which is left out.
function packPlatformPackage(addonDir, metaName) {
  const definition = { suffix: hostToSliceSuffix(HOST) }
  const name = slicePackageName(metaName, definition)
  const linked = path.join(addonDir, 'node_modules', ...name.split('/'))
  const staged = fs.mkdtempSync(path.join(os.tmpdir(), 'fit-remote-'))
  const stagedHost = path.join(staged, 'prebuilds', HOST)
  fs.mkdirSync(stagedHost, { recursive: true })
  fs.copyFileSync(path.join(linked, 'package.json'), path.join(staged, 'package.json'))
  fs.copyFileSync(path.join(linked, 'index.js'), path.join(staged, 'index.js'))

  const builtHost = path.join(addonDir, 'prebuilds', HOST)
  const expected = expectedAddonFile(metaName, HOST)
  if (!fs.existsSync(path.join(builtHost, expected))) {
    throw new Error(`${expected} was not built under ${builtHost}`)
  }
  const preSplit = mangledAddonName(metaName)
  for (const entry of fs.readdirSync(builtHost)) {
    if (entry.startsWith(`${preSplit}.bare`)) continue
    fs.cpSync(path.join(builtHost, entry), path.join(stagedHost, entry), {
      recursive: true,
      dereference: true
    })
  }
  const tarball = packJson(staged)
  fs.rmSync(staged, { recursive: true, force: true })
  return { name, tarball }
}

function packAddons() {
  step(`Packing the addons into ${path.relative(PACKAGES_DIR, ADDONS_OUT)}`)
  fs.mkdirSync(ADDONS_OUT, { recursive: true })
  const addons = []
  const platformPackages = []
  for (const addon of ADDONS) {
    const addonDir = path.join(PACKAGES_DIR, addon.dir)
    const { name } = readJson(path.join(addonDir, 'package.json'))
    addons.push({ name, tarball: packJson(addonDir) })
    if (addon.platformPackage) platformPackages.push(packPlatformPackage(addonDir, name))
  }
  for (const { name, tarball } of [...addons, ...platformPackages]) {
    console.log(`  ${name} -> ${path.basename(tarball)}`)
  }
  return { addons, platformPackages }
}

function fileSpec(fromDir, file) {
  return `file:${path.relative(fromDir, file).split(path.sep).join('/')}`
}

function pointAtTarballs({ addons, platformPackages }) {
  step('Pointing inference, the SDK and e2e at the packed addons')

  const inferenceManifest = path.join(INFERENCE_DIR, 'package.json')
  const inference = readJson(inferenceManifest)
  for (const { name, tarball } of addons)
    inference.devDependencies[name] = fileSpec(INFERENCE_DIR, tarball)
  inference.overrides = { ...inference.overrides }
  for (const { name, tarball } of platformPackages)
    inference.overrides[name] = fileSpec(INFERENCE_DIR, tarball)
  writeJson(inferenceManifest, inference)

  const sdkManifest = path.join(SDK_DIR, 'package.json')
  const sdk = readJson(sdkManifest)
  for (const { name, tarball } of addons) sdk.dependencies[name] = fileSpec(SDK_DIR, tarball)
  sdk.dependencies['@qvac/inference'] = fileSpec(SDK_DIR, INFERENCE_DIR)
  sdk.overrides = { ...sdk.overrides }
  for (const { name, tarball } of platformPackages) sdk.overrides[name] = fileSpec(SDK_DIR, tarball)
  writeJson(sdkManifest, sdk)

  // The SDK's overrides apply only at its own root, so e2e names every package.
  const e2eManifest = path.join(E2E_DIR, 'package.json')
  const e2e = readJson(e2eManifest)
  e2e.overrides = { ...e2e.overrides, '@qvac/inference': fileSpec(E2E_DIR, INFERENCE_DIR) }
  for (const { name, tarball } of [...addons, ...platformPackages])
    e2e.overrides[name] = fileSpec(E2E_DIR, tarball)
  writeJson(e2eManifest, e2e)
}

function installAndBuild() {
  step('Installing and building packages/inference')
  runShown('npm', ['install', '--ignore-scripts'], INFERENCE_DIR)
  runShown('npm', ['run', 'build'], INFERENCE_DIR)

  // The SDK's own `build` runs lint first, which refuses the local specs.
  step('Installing and building packages/sdk')
  runShown('npm', ['install', '--ignore-scripts'], SDK_DIR)
  runShown(
    'node',
    [path.join('node_modules', 'typescript', 'bin', 'tsc'), '--project', 'tsconfig.json'],
    SDK_DIR
  )
  runShown('npm', ['run', 'postcompile:aliases'], SDK_DIR)

  // --install-links packs the SDK rather than copying the directory, and
  // bundle:sdk regenerates qvac/worker.entry.mjs, which the runner prefers over
  // the SDK's own worker.
  step('Installing and building e2e')
  runShown('npm', ['run', 'clean:sdk-snapshot'], E2E_DIR)
  fs.rmSync(path.join(E2E_DIR, 'node_modules'), { recursive: true, force: true })
  fs.rmSync(path.join(E2E_DIR, 'package-lock.json'), { force: true })
  runShown('npm', ['install', '--install-links'], E2E_DIR)
  runShown('npm', ['run', 'build'], E2E_DIR)
  runShown('npm', ['run', 'bundle:sdk'], E2E_DIR)
}

// ggml's Vulkan backend keeps only the indices this lists, and a value with no
// index lists none.
const NO_GPU_ENV = { GGML_VK_VISIBLE_DEVICES: 'none' }

function runModelFit() {
  const noGpu = args.has('--no-gpu')
  step(`Running the model-fit e2e cases${noGpu ? ' with the GPUs hidden' : ''}`)
  const reportDir = path.join(E2E_DIR, noGpu ? 'fit-remote-report-no-gpu' : 'fit-remote-report')
  runShown(
    path.join('node_modules', '.bin', 'qvac-test'),
    ['run:local:desktop', '--config', E2E_DIR, '--filter', 'model-fit', '--report-dir', reportDir],
    E2E_DIR,
    noGpu ? NO_GPU_ENV : {}
  )
  console.log(`\nReport: ${reportDir}`)
}

checkEnvironment()
if (!args.has('--skip-build')) for (const addon of ADDONS) buildAddon(addon)
pointAtTarballs(packAddons())
installAndBuild()
if (!args.has('--skip-run')) runModelFit()

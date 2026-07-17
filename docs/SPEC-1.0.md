# Speak 1.0 — spec (drafted 2026-07-17 from live debugging with Stephen)

## 0. The bug/confusion that triggers this release

v0.3.0's "Use Incoming Matte" with NO key wired falls back to the incoming
image alpha. On the color page that alpha is 1.0 (opaque) — which reads as
"matte = fully clean" — so grain lands at FULL strength everywhere, floor
ignored. The user's correct expectation: no matte arriving = no (or floor)
grain. The silent fallback is a Fusion-ism and must never be a default.

## 1. Matte source becomes explicit (the fix)

Replace the `grainMatte` bool with a CHOICE param `matteSource`:
- **Off** (default) — grain uniform, alpha untouched.
- **Key input (blue wire)** — the Matte mask clip. If NOT connected: grain
  drops to **Matte Floor** (treat matte as 0 — absence means absence), and
  the on-image status line says "matte: key not connected".
- **Incoming alpha (Fusion / in-band)** — the old fallback, now opt-in. If
  the incoming alpha is ~1.0 everywhere (no matte data), say so on-screen:
  "matte: incoming alpha is opaque — is Hush's export on?"
Consume-on-use stays: any matte source forces output alpha opaque.
Migration: old projects with grainMatte=1 load as "Key input" if the key is
connected, else "Incoming alpha" (preserve renders), with the status line
explaining. Keep the OFX param for backward compat or map via param
versioning.

## 2. In-panel guidance (tooltips are not enough)

Speak already draws scopes on the image — use that surface:
- A one-line **status strip** (toggleable, on by default, like Hush's
  covenant surfaces): active stages + matte source + its live state
  ("matte: key ✓ (mean 0.31)" / "key not connected — grain at floor").
- A **"Setup Guide" view** in the View dropdown: renders the three-step
  Hush→Speak recipe as text/diagram ON the viewer itself. Zero-cost
  documentation that cannot drift from the plugin that ships it.
- Group labels stay numbered steps (1·Color … 7·Inspect) — extend the idiom:
  rename group 6 to "6 · Grain — where film lives".

## 3. Ship the rest of Speak for 1.0

Audit each phase against speak_core.h and finish to covenant standard
(measured, scoped, honest limitation notes):
- **Tone spine** (H&D negative→printer→print): calibration presets for 2–3
  published stocks; keep the live H&D scope as proof.
- **Dye/subtractive + split toning**: verify against the Macbeth test
  (test_speak_macbeth.cpp); expose only controls that survive it.
- **Optics**: halation/bloom/veil/vignette are in; add honest defaults per
  format preset (S16, 35mm 2-perf/4-perf) that set grain size + halation
  radius together.
- **Grain**: format presets; keep matte pipeline per §1; document that the
  grain is display-referred unless grainRef says otherwise.
- **Gate weave**: needs a deterministic per-frame seed contract (frameIndex
  already plumbed) — verify render vs playback consistency in Resolve.
- **Parity**: Metal/OpenCL/CUDA gates green; SpeakParams was 384B at
  drafting (504B once §1/§2 landed: matteKeyMissing + the status text) — the
  shader-side structs re-declare it (Metal/OpenCL sources) and test_speak.cpp
  pins sizeof + parFields; every field addition re-pins BOTH.
- **Perf**: one full-res timing pass on 4K; scopes off = no measurable cost.

## 4. The visual guide (downloadable + web)

One guide, two outputs, real pictures:
- Build `docs/guide/` — an HTML page ("Hush & Speak — clean early,
  reconstruct late") with annotated screenshots: the node graph, OFX Alpha →
  Enable, the blue wire, before/after crops, the grain-isolated view, the
  matte views; plus the three setups (matte-shaped grain / uniform grain /
  no grain, image untouched) and the one rule (export off when unconsumed).
- Export the same content to PDF (attach to both repos' releases and the
  control-z releases).
- Publish on the site: the control-z repo's site/ generator gets a
  guide page linked from both tool pages (pattern: the existing node-tree
  and template-pack guide pages).
- Screenshots: drive Resolve via the scripting API where possible (Fusion
  page + renders); color-page shots need Stephen once — list the exact
  shots needed and batch them in one sitting.

## 5. Release

v1.0.0 on Speak + mirror on control-z (ci-unsigned zips + Windows CI zip;
signed .pkg from the signing Mac). Hush gets a matching point release only
if §1 requires a hint change. CHANGELOG in this repo's voice; the release
notes tell the matte story once, plainly.

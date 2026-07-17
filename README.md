# Speak — film emulation for DaVinci Resolve, free

**Hush quiets the noise. Speak gives the image its voice.**

Speak is a free, MIT-licensed OpenFX film-emulation plugin for the **last node**
of your grade — the finishing counterpart to
[Hush](https://github.com/amateurmenace/Hush-OpenNR), the free denoiser that
opens the node tree. It works in the **free edition** of DaVinci Resolve, on
the Color, Edit and Fusion pages, rendering via Metal on macOS and OpenCL on
Windows/Linux, with a bit-verified CPU reference behind every kernel.

> **Status: early beta (v0.2).** The color engine, halation and grain are
> built and machine-verified (every module is gated by measurement and all
> four compute backends agree to ~2e-5), but Speak has not yet had a full
> field test inside Resolve. Use it, enjoy it, and file issues — just don't
> cut a feature film on it this week.

## What it does today

- **Film tone scale** — real Hurter–Driffield characteristic curves: a negative
  stage, printer-light color timing in genuine printer points, and a print
  stage, all closed-form and all plotted live on screen (the H&D scope samples
  the exact curve the pixels use — the plot cannot lie).
- **Subtractive color** — saturation in the density domain, where dye density
  adds and light multiplies: highlight chroma self-compresses toward paper
  white and hues bend toward the dye axes, which a 3×3 matrix cannot fake.
  With an inter-image coupler for the classic dye cross-talk.
- **Split toning** — per-channel density offsets by tonal zone, anchored to the
  working curve: mids stay neutral *by construction*, and tints hold hue where
  lift/gamma/gain swings it.
- **Halation** — light that passed through the emulsion, bounced off the base,
  and re-exposed the negative. Injected as *exposure before the curve* (never a
  red overlay on the output), on a multi-scale scatter field with a real
  power-law skirt. Physically self-limiting: a blown highlight's halo goes
  white-hot in the core, red at the edges.
- **Grain** — density noise, per RGB dye layer, rising with density (shadows
  loud, paper white grainless, values can never go negative), bandpass at a
  physical pitch, boiling independently every frame.
- **The Hush handoff** — enable *Export Clean Matte to Alpha* in Hush ≥ 3.7 and
  *Use Incoming Matte* in Speak's grain: grain lands exactly where the denoiser
  cleaned deepest and backs off where real noise survives. "Clean early,
  reconstruct late," as one chain.
- **Scopes and honest views** — H&D curves with the frame's exposure histogram,
  a Status-M density parade measured from the true result, and isolated
  halation/grain views that are never auto-gained (a subtle effect correctly
  looks subtle).

Identity at default; every stage has its own enable; alpha always passes
through untouched.

## What it doesn't do yet

Bloom / veiling glare, vignette, gate weave, stock families and the Stock Book,
Shoot-a-Chart calibration, Match Shot, `.cube` export. They're designed (see
the [design study](https://amateurmenace.github.io/Hush-OpenNR/whitepaper.html))
and land behind the same measurement gates as everything above.

## Install

Grab the latest release from [Releases](../../releases) or the shared
[control-z downloads hub](https://github.com/amateurmenace/control-z/releases)
(tags `speak-vX.Y.Z`):

- **macOS** — run the `.pkg`, or unzip and copy `Speak.ofx.bundle` to
  `/Library/OFX/Plugins`. If Resolve doesn't see it, delete
  `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/OFXPluginCacheV2.xml`
  and relaunch.
- **Windows** — CI now builds a prebuilt zip (`Speak-<version>-Windows.zip`,
  OpenCL render path) on every push; it attaches to releases from the next
  one on. Until then, or on Linux: build from source (`plugin/Makefile`,
  Windows: `CMakeLists.txt`).

In Resolve: last node → OpenFX → **Hush → Speak Film**. On a DaVinci Wide
Gamut / Intermediate managed timeline, leave Output on *Working space* and let
Resolve Color Management deliver.

## The point

Paid film tools are black boxes. Speak is built from published sensitometry —
H&D curves, printer points, subtractive dye behaviour, Selwyn-family
granularity — with **original constants stated as modelled defaults, never
cloned profiles or trademarked stock names**, and every claim about the image
verifiable on screen. The source of truth is one readable header
([`plugin/speak_core.h`](plugin/speak_core.h)); the GPU kernels are
line-by-line ports of it, held to agreement by parity tests.

MIT License. Sibling projects: [Hush](https://github.com/amateurmenace/Hush-OpenNR)
· [control-z downloads hub](https://github.com/amateurmenace/control-z).

# Speak changelog

## 1.0.0 — 2026-07-17

First stable release. The film-reconstruction counterpart to Hush is complete:
tone, colour, optics and grain, every claim showable on screen, held to
CPU/Metal/OpenCL parity (and a syntax-checked CUDA port) by a gate suite that
grew to 160-plus checks.

**The matte story, once and plainly.** In v0.3, *Use Incoming Matte* with no
key wired quietly fell back to the picture's alpha. On the color page that
alpha is opaque (1.0) — which read as "the whole frame is clean" — so grain
landed at full strength everywhere and the Matte Floor was ignored. The silent
fallback was the bug. So the matte source is now an explicit **choice**, not a
guess:

- **Matte Source → Off** — grain is uniform, alpha passes through untouched.
- **Matte Source → Key input (blue wire)** — grain is keyed on Hush's
  clean-confidence matte, wired straight into Speak's key input. If no key is
  connected, the matte is treated as **absent**: grain holds at the Matte
  Floor. Absence means absence — nothing substitutes silently.
- **Matte Source → Incoming alpha** — the old in-band path (Fusion page), now
  opt-in by explicit choice only.

Consume-on-use stays and now covers every backend: whenever a matte source is
active, Speak forces its own output alpha opaque, so the matte is spent at its
consumer and never rides downstream where a host unpremultiply could lift it
into the picture. Old projects migrate without changing a pixel — the v0.3
toggle survives as a hidden param and resolves exactly as it used to render
until you make an explicit choice, and the on-image status strip says which.

**Guidance moved onto the image (tooltips are not enough).** A one-line
**status strip** (on by default) reports the active stages, the matte source
and its live state — `matte: key ✓ mean 0.31`, or `key not connected — grain at
floor`, or `incoming alpha is opaque — is Hush's export on?`. A new **View →
Setup Guide** draws the whole Hush→Speak recipe on the viewer itself: the node
boxes, the blue key wire, the three setups and the one rule. Both are drawn by
the render kernel, so the documentation ships inside the plugin and cannot
drift from it. The strip and card share a 5×7 font whose table lives in all
four backends; the plugin composes the text, the kernels only rasterize it.

**Stock and format presets, both measured.** Three tone families beside Neutral
— Long latitude (soft, system gamma ~1.15), Punchy print (~1.7), Chrome
(reversal-like, ~1.8) — every one a family SHAPE from published sensitometry,
gray-balanced, no commercial stock cloned or named. Selecting one moves the
Contrast/Shoulder/Toe handles to its values, so the preset is transparent. A
gate pins each family's measured system gamma against the number the hint
states. Format presets — Super 35, 35mm 2-perf, Super 16 — set Grain Size and
Halation Radius **together** from one physical model, because grain and
halation are physical sizes on the film: a smaller format carries them larger
relative to the frame.

**Other v1 work.** Group 6 is now "Grain — where film lives"; grain is
documented display-referred. The gate weave gained a render-vs-playback gate
(scrubbed order equals sequential order, bit-for-bit). A phase-by-phase
completion audit closed the last verification gap — split toning's pivot
control is now exercised at a non-zero pivot, not just the degenerate one. The
visual guide (docs/guide/, plus a PDF) tells the Hush + Speak story with real
renders from the reference implementation.

Bit-exact identity at default, alpha honoured, deterministic, real-time UHD via
Metal — unchanged and still gated.

## 0.3.0 — 2026-07-17

- **The color-page-native Hush handoff: a real KEY input.** Speak now defines a
  "Matte" mask clip, which Resolve feeds from the node's **blue KEY input**.
  Wire Hush's node key output (blue) → Speak's key input (blue) and the
  clean-confidence matte travels its **own wire** — it never rides the RGB
  path's alpha, so nothing unpremultiplies (blows out) the picture on the
  color page. *Use Incoming Matte* keys grain on the key when wired, and falls
  back to the incoming RGBA alpha (Hush 3.7.2's in-band export) when it isn't.
- With a key wired, Speak forces its **output alpha opaque**, so the matte is
  consumed at Speak and can't blow up Speak's own node downstream.
- The mask value is read robustly as max(luma, alpha) — correct whether Resolve
  delivers the key in RGB or in alpha. Verified: output alpha opaque and grain
  ~290× stronger where the key is 1 vs 0; Metal/OpenCL parity re-pinned and green.

## 0.2.0 — 2026-07-16 (early beta; first public release)

First release, in Speak's own repository (the code began as a second plugin in
the Hush/OpenNR bundle; the plugin identifier `org.opennr.Speak` is kept so any
dev-build project stays valid). Everything below is gated by measurement and
verified across CPU/Metal/OpenCL (+ a syntax-checked CUDA port), but v0.2 has
not yet had a full field test inside Resolve — hence *beta*.

- **Film tone scale** (Phase 1): closed-form monotone H&D negative → printer
  lights (0.025 logE/point) → print cascade in a color-managed Log-Exposure
  Spine (DWG/DI, Rec.709 g2.4, ACEScct, linear); per-channel gray pivot makes
  neutral-in → neutral-out exact by construction. Live H&D scope with the
  frame's exposure histogram; honest Bake-to-Rec.709 output mode.
- **Subtractive color** (Phase 2): density-domain saturation + inter-image
  coupler; highlight chroma self-compresses; neutral invariant by construction.
  Gated against a matched 3×3 saturation on a Macbeth chart in CIELAB.
- **Split toning** (Phase 3): per-channel density offsets over a 3-zone
  partition anchored to the working curve; mids untouched by construction.
  Gated against matched lift/gamma/gain.
- **Halation** (Phase 4): per-channel highlight scatter re-injected as exposure
  *before* the curve (a blue neon halates blue, never red), on an
  energy-conserving coarse-to-fine scatter pyramid with a cubic-B-spline
  accumulate (radially symmetric — measured, gate G18) and an ~r⁻³ skirt a
  single Gaussian cannot produce. Radius is % of frame height, so proxies and
  full res match (~1%, gate G16). Isolated-scatter view, never auto-gained.
- **Grain** (Phase 4): density noise per RGB dye layer, σ_D ∝ √D (shadows loud,
  paper white grainless, never negative), bandpass at a physical pitch (% of
  frame height), temporally independent. Isolated-grain view on gray.
- **The Hush handoff**: *Use Incoming Matte (Alpha)* keys grain on Hush ≥ 3.7's
  clean-confidence matte (`clamp((effN−1)/6)` in alpha) — full grain where the
  denoiser averaged deep, the Matte Floor where motion was protected. Alpha
  passes through untouched, so the matte survives Speak.
- Scopes: H&D curves + exposure histogram (top-left), Status-M density parade
  of the true result (top-right).
- 44 parameters, identity at default, per-stage enables, deterministic.

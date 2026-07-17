# Speak changelog

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

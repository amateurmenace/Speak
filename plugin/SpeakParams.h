// Speak — shared parameter block passed from the OFX plugin to the GPU kernels.
//
// Speak is the film-reconstruction counterpart to Hush (see OpenNRPlugin.cpp):
// the LAST node in a grade, where Hush is the first. It lives in the SAME .ofx
// bundle as a second plugin (org.opennr.Speak) and inherits Hush's cardinal
// rule verbatim — speak_core.h is the single source of truth, and the three GPU
// kernels (SpeakMetalKernel.mm, SpeakCudaKernel.cu, SpeakOpenCLKernel.cpp) are
// line-by-line ports verified by a parity test (~2e-5 mean).
//
// EVERY member of BOTH structs below is 4 bytes, so the layout is identical in
// C++, Metal, CUDA and OpenCL with no padding on any of them. SpeakProfile is
// the engine's POD "look" struct — built-in stock families and user-calibrated
// profiles emit the SAME struct and hit exactly ONE kernel path. A field added
// here MUST also be added to the struct declarations inside SpeakMetalKernel.mm
// and SpeakOpenCLKernel.cpp kernel sources (CUDA includes this header directly),
// and the layout parity check in test/test_speak.cpp must be updated.
//
// MIT License.

#ifndef OPENNR_SPEAKPARAMS_H
#define OPENNR_SPEAKPARAMS_H

// ---------------------------------------------------------------------------
// SpeakProfile — the film "look", one versioned POD of 4-byte fields.
//
// The density spine is a cascaded two-stage tone scale:
//   scene logE  →  Negative H&D  →  Printer Lights  →  Print H&D  →  positive
// so there are TWO sets of per-channel Hurter-Driffield characteristic-curve
// handles (negative and print), with the printer lights injected as per-channel
// log-exposure offsets in the density gap between them (real lab points, where
// 1 point = 0.025 logE). Every handle is published-sensitometry-shaped; nothing
// clones a commercial profile.
//
// Curve handle semantics (per channel, both stages), all in log10 optical
// density on a log10-exposure axis (the physical sensitometric convention;
// the canonical cross-module datum is log2 stops and is converted internally):
//   Dmin      base/fog density (density at zero exposure)
//   Dmax      maximum density (the emulsion/paper ceiling)
//   gamma     straight-line contrast index dD/d(log10 E)
//   toe       toe knee sharpness  (larger = sharper toe,  ~0.1..40)
//   shoulder  shoulder knee sharpness (larger = sharper shoulder, ~0.1..40)
//   speed     speed point: log10-exposure (rel. 18% gray) where the
//             extrapolated straight line meets Dmin
// ---------------------------------------------------------------------------
typedef struct SpeakProfile
{
    // ---- Negative characteristic curves (R,G,B) ----
    float negDmin[3];
    float negDmax[3];
    float negGamma[3];
    float negToe[3];
    float negShoulder[3];
    float negSpeed[3];

    // ---- Printer lights: R/G/B color timing + master, in printer points ----
    // (1 point = 0.025 log10-exposure), injected in the negative→print gap.
    float printerLights[3];
    float printerMaster;

    // ---- Print characteristic curves (R,G,B) ----
    float prnDmin[3];
    float prnDmax[3];
    float prnGamma[3];
    float prnToe[3];
    float prnShoulder[3];
    float prnSpeed[3];

    // ---- Subtractive CMY dye coupling (Phase 2) ----
    float dyeCouple[9];   // inter-image coupler c_kj (row-major 3x3)
    float subSat[3];      // per-dye subtractive-saturation crosstalk strength
    float subSatKnee[3];  // per-dye Dmax knee in log-density

    // ---- Split toning / chromogenic crossover anchors (Phase 3) ----
    float splitShadow[3]; // CMY density offsets applied in the shadows
    float splitHigh[3];   // CMY density offsets applied in the highlights
    float splitPivot;     // tonal pivot (stops, rel. 18% gray) for the split
    float splitBalance;   // shadow/highlight weighting knob

    // ---- Halation (Phase 4 — the first spatial module) ----
    // Scattered light that re-exposes the NEGATIVE, so it is injected as
    // exposure (additive in scene-linear) ahead of the whole density spine.
    // See speak_core.h for the physics and for why there is no injection in
    // the negative->print gap.
    float halAmount;      // 0 = off (bit-exact identity); scales the re-exposure
    float halRadius;      // scatter radius as % of FRAME HEIGHT (format-relative)
    float halThresh;      // scene-linear highlight threshold that feeds the scatter

    // ---- Grain (Phase 4) ----
    // Film grain is DENSITY noise: the dye clouds ARE the image, so the noise
    // multiplies light rather than adding to it. Applied in the density domain
    // of the finished look (print-referred), per RGB dye layer, independent
    // every frame. See speak_core.h applyGrain for the physics and the gates.
    float grainAmount;    // 0 = off (bit-exact identity)
    float grainSize;      // grain pitch as % of FRAME HEIGHT (format-relative)

    // ---- Bloom / veiling glare (Phase 4, spec 1B.5) ----
    // Optical glare on the VIEWED image, sited in linear after the print: an
    // energy-conserving convex mix toward a normalized multi-scale scatter of
    // the finished look's linear light (a lens transmits — it creates
    // nothing), plus a veiling floor carried as a frame-MEAN term in the same
    // normalized mixture, so the black-lift scales with frame luminance
    // exactly. See speak_core.h for the physics and the gates.
    float bloomAmount;    // 0 = off (bit-exact identity); the scattered fraction
    float bloomRadius;    // scatter radius as % of FRAME HEIGHT (format-relative)
    float bloomVeil;      // veiling floor's share of the scattered light [0..0.9]

    // ---- Vignette (Phase 4) ----
    // cos^4 natural illumination falloff of the TAKING lens (the published
    // cosine-fourth law), applied to scene-linear BEFORE the negative — so
    // corners ride down the H&D toe the way film corners actually do, and
    // light that never arrived cannot halate. See speak_core.h vignGain.
    float vignAmount;     // 0 = off (bit-exact identity); mix toward cos^4
    float vignField;      // half-DIAGONAL field angle, degrees (lens FOV knob)

    // ---- Gate weave (Phase 4) ----
    // The transport's picture-position noise: a deterministic closed form of
    // frameIndex (hash-lattice octaves, ~1/f spectrum), resampled sub-pixel
    // with Catmull-Rom. Same frame in, same frame out — always, everywhere.
    float weaveAmount;    // 0 = off (bit-exact identity); amplitude, % of height
    float weaveSpeed;     // temporal scale of the weave pattern (1 = default)

    // ---- meta ----
    float systemGamma;    // target overall system gamma (~1.6), for the scope
    int   residualLUT;    // 0 = separable model only, 1 = 3D residual cube on
    int   profileVersion; // struct version for save/load compatibility
    int   _pad0;          // reserved (keeps the field count explicit)
} SpeakProfile;

// ---------------------------------------------------------------------------
// SpeakParams — the full per-render block: color-management + runtime state +
// the embedded look profile. Passed by value to the kernels exactly like
// Hush's NRParams. SpeakProfile is embedded LAST so its offsets stay stable as
// runtime fields are added ahead of it.
// ---------------------------------------------------------------------------

// inputColorSpace values
#define SPEAK_CS_DWG_INTERMEDIATE 0   // DaVinci Wide Gamut / Intermediate (default)
#define SPEAK_CS_REC709_G24       1   // Rec.709, gamma 2.4
#define SPEAK_CS_DWG_LINEAR       2   // DaVinci Wide Gamut, linear
#define SPEAK_CS_ACESCCT          3   // ACEScct
#define SPEAK_CS_LINEAR           4   // scene-linear passthrough (identity gamut)

// outputMode values
#define SPEAK_OUT_WORKING         0   // return DWG/DI, let RCM deliver (default)
#define SPEAK_OUT_BAKE_REC709     1   // apply DWG/DI -> Rec.709 as the last node

// viewMode values
#define SPEAK_VIEW_RESULT         0
#define SPEAK_VIEW_SPLIT          1   // input | result
#define SPEAK_VIEW_INPUT          2
#define SPEAK_VIEW_SCATTER        3   // the isolated halation scatter field
#define SPEAK_VIEW_GRAIN          4   // gray + the grain increment, isolated
#define SPEAK_VIEW_BLOOM          5   // gray + the SIGNED bloom delta, isolated
#define SPEAK_VIEW_SETUP          6   // the Setup Guide card (v1.0) — the
                                      // Hush->Speak recipe drawn by the kernel

// matteSource values (v1.0 — replaces the v0.3 grainMatte bool; see §1 of
// docs/SPEC-1.0.md). The value here is the EFFECTIVE source: the plugin
// resolves the legacy v0.3 bool into one of these before the kernel runs.
#define SPEAK_MATTE_OFF           0   // grain uniform, alpha untouched
#define SPEAK_MATTE_KEY           1   // the Matte clip (blue key wire)
#define SPEAK_MATTE_ALPHA         2   // the incoming image alpha (in-band)

typedef struct SpeakParams
{
    int   inputColorSpace; // SPEAK_CS_*
    int   outputMode;      // SPEAK_OUT_*
    int   grainRef;        // 0 display-referred, 1 scene-referred (Phase 4)
    float strength;        // 0..1 global look mix; 0 = bit-exact identity
    int   frameIndex;      // for deterministic grain / gate weave (Phase 4)
    int   viewMode;        // SPEAK_VIEW_*

    // ---- module enables (each stage's contribution is toggleable) ----
    int   enableTone;      // the density spine (negative -> printer -> print)
    int   enableDye;       // subtractive color (Phase 2)
    int   enableSplit;     // split toning (Phase 3)
    int   enableOptics;    // halation/bloom/grain/vignette (Phase 4)

    // ---- scopes (read-only, rendered INTO the image like Hush's) ----
    int   scopeHD;         // live H&D characteristic-curve scope
    int   scopeDensity;    // Status-M density waveform scope
    int   scopeVector;     // subtractive-sat vector field (Phase 2)

    // ---- grain pipeline controls (NOT stock properties, so params-level) ----
    int   enableGrain;     // the grain stage's own toggle
    int   matteSource;     // SPEAK_MATTE_* — the EFFECTIVE matte source (v1.0).
                           // With KEY the plugin packs the key clip into the
                           // src alpha the kernels read; with ALPHA the source
                           // buffer's own alpha is the matte (Hush's clean-
                           // confidence matte: clamp((effN-1)/6), high = cleaned)
    float grainMatteFloor; // grain amount where the matte is 0 (motion) — the
                           // real noise is still there, so add less on top
    int   matteKeyMissing; // 1 = KEY selected but no key wired: the matte reads
                           // as 0 everywhere (absence means absence — grain sits
                           // at the Floor; the silent fall-through to incoming
                           // alpha was v0.3's color-page trap, see SPEC-1.0 §0)

    // ---- on-image guidance (v1.0, SPEC-1.0 §2) ----
    // The one-line status strip is drawn by the kernels from TEXT COMPOSED BY
    // THE PLUGIN — message logic lives once in C++; the kernels only share a
    // rasterizer (and its font table, re-declared per backend like everything
    // else). Packed 4 chars per int, little-endian byte order, NUL-terminated.
    // Codes 1..15 are the font's specials (check, middot, emdash, arrow, and
    // the box-drawing set); see kSpeakFontBits in speak_core.h.
    int   statusStrip;     // 1 = draw the strip (bottom-left, panel chrome)
    int   statusText[28];  // up to 111 chars + NUL, packed 4-per-int

    SpeakProfile profile;
    int   maskExternal;    // consume-on-use: 1 = a matte source is active, so
                           // Speak is the matte's consumer and the OUTPUT alpha
                           // is forced opaque (a matte that keeps riding past
                           // its consumer is what host alpha math feeds on)
} SpeakParams;

// ---------------------------------------------------------------------------
// Scope statistics buffer (uint32 slots). The scopes are measured FROM the
// frame by the same kernels that filter it, exactly like Hush's — so the panel
// can never disagree with the pixels. Populated only when a scope is on.
//
// The frame is sampled on a stride-2 grid (identically on CPU and GPU) and
// binned with integer atomics, which are order-independent — so every backend
// lands on bit-identical counts and the parity test stays meaningful.
//
// This table is re-declared inside the Metal and OpenCL kernel sources (CUDA
// includes this header). Bump carefully.
// ---------------------------------------------------------------------------
#define SPEAK_EXP_BINS       128  // scene log2-exposure bins over [-6,+6] stops
#define SPEAK_STATS_HIST_EXP 0    // 128 bins: the frame's exposure distribution
#define SPEAK_STATS_HIST_MAX 128  // the largest bin count (bar normalisation)

// Status-M density waveform/parade. A FIXED grid (independent of the panel's
// scale, so this buffer's size never depends on the image height): per channel,
// 128 image-column buckets x 96 density buckets over [0, SPEAK_WF_DMAX] film
// density. Channel-major: ch*(COLS*ROWS) + col*ROWS + row.
#define SPEAK_WF_COLS        128
#define SPEAK_WF_ROWS        96
#define SPEAK_WF_DMAX        3.0f
#define SPEAK_STATS_WF       129                       // 128*96*3 = 36864 cells
#define SPEAK_STATS_WF_MAX   (129 + 36864)             // largest cell count
#define SPEAK_STATS_UINTS    (129 + 36864 + 1)

// ---------------------------------------------------------------------------
// The shared separable scatter pyramid (Phase 4). An energy-normalized pyramid
// read back at full resolution as a weighted mixture of octave levels. The
// same machinery runs twice when both optics modules are live: halation
// builds it on the SCENE-linear highlight excess (pre-curve — the light that
// re-exposes the negative), bloom on the LOOK's linear output (post-print —
// the light the viewer's optics scatter). Spec 1B.5's "read the same pyramid
// twice" could not survive the halation injection-site physics: the two
// fields live on opposite sides of the tone scale, and an energy-conserving
// bloom must subtract light in the same domain where it re-adds it. What is
// shared is the machinery (decimate/accumulate/sample), textually.
//
// WHY A PYRAMID, MEASURED (not assumed — this repo builds no architecture
// without a measurement). Metal, UHD 3840x2160, 3 channels, per frame:
//     radius   sigma   taps   direct separable   this pyramid
//      0.5%     19.2    117        45.1 ms          5.0 ms
//      1.0%     38.4    233        85.2 ms          9.9 ms
//      2.0%     76.8    463       170.7 ms          6.8 ms
//      4.09%   157.1    945       333.3 ms          9.5 ms
// The direct blur is O(radius) and misses real-time at EVERY setting (Hush
// renders its whole pipeline at 38.7 ms UHD). The pyramid is ~flat in radius.
// That flatness is the architectural property; it is why the radius control can
// be a real slider instead of a performance cliff.
//
// Levels are 4x4-binomial-decimated ([1,3,3,1]/8 separable, 16 taps), which is
// mean-preserving, deterministic and needs no atomics — so it is immune to the
// Apple-OpenCL atomics bug that forces the stats pass to be skipped there.
// The level count must be governed by MINDIM, never by MAXLEV. The mixture's
// skirt spans (nLev - L_target) octaves; nLev ~ log2(H/MINDIM) and L_target ~
// log2(radius% * H / C), so their DIFFERENCE is independent of frame height —
// which is exactly what makes the halo resolution-independent (G16). A MAXLEV
// that binds breaks that: at 9 it truncated UHD's skirt to 12.4% missing weight
// against 6.3% at 1080p and below, i.e. UHD rendered a different halo from the
// proxy the shot was graded on. MAXLEV is now only a runaway guard.
#define SPEAK_HAL_MAXLEV     14   // guard only — MINDIM must be what binds
#define SPEAK_HAL_MINDIM     8    // stop decimating below this level dimension
#define SPEAK_HAL_SIGMA_MIN  0.05f // sigma floor: halRadius=0 would divide by zero

#endif // OPENNR_SPEAKPARAMS_H

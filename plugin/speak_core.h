// Speak — film-reconstruction core (the CPU reference and single source of
// truth for the algorithm). The three GPU kernels (SpeakMetalKernel.mm,
// SpeakCudaKernel.cu, SpeakOpenCLKernel.cpp) are line-by-line ports of this
// file; ANY change to the math must be applied to all four and verified with
// test/test_speak_metal (parity ~2e-5 mean). Keep constants, curve formulas
// and loop order textually parallel — same discipline as Hush's nr_core.h.
//
// Phase 1 — the density spine:
//   color-manage in  ->  Log-Exposure Spine (log2, 18% gray = 0)
//                    ->  per-channel Negative H&D characteristic curves
//                    ->  Printer Lights (per-channel logE offset, in the gap)
//                    ->  per-channel Print H&D characteristic curves
//                    ->  positive transmittance  ->  color-manage out
//   + a live, deterministic H&D curve scope rendered INTO the image.
//
// Every look is gated behind `strength`: at strength 0 the node is a bit-exact
// pass-through (identity), and for a gray-balanced profile a neutral input maps
// to a neutral output exactly, by construction (see processPixel).
//
// MIT License.

#ifndef OPENNR_SPEAK_CORE_H
#define OPENNR_SPEAK_CORE_H

#include <cmath>
#include <cstdint>
#include <vector>

#include "SpeakParams.h"

namespace speakcore {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const float kLog10_2    = 0.301029996f; // log10(2): stops -> log10 exposure
static const float k18Gray     = 0.18f;        // scene-linear middle gray datum
static const float kPrinterPt  = 0.025f;       // 1 printer point = 0.025 log10 E
static const float kLinTiny    = 1e-8f;        // floor before log2 (avoids -inf)
static const float kKneeMin    = 0.05f;        // min toe/shoulder sharpness

// DaVinci Intermediate transfer (verified against the colour-science reference
// and Blackmagic's DWG/DI white paper). Encode/decode are exact inverses, so
// the working round-trip is lossless by construction.
static const float kDI_A       = 0.0075f;
static const float kDI_B       = 7.0f;
static const float kDI_C       = 0.07329248f;
static const float kDI_M       = 10.44426855f;
static const float kDI_LIN_CUT = 0.00262409f;
static const float kDI_LOG_CUT = 0.02740668f;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float pow10f(float x) { return std::exp2(x * 3.32192809f); } // 10^x via exp2

// Numerically stable softplus: log(1 + e^z) = max(z,0) + log(1 + e^-|z|).
// Written with primitives every backend has (no log1p) so the four ports are
// textually identical. Monotone increasing, C-infinity; the H&D building block.
static inline float softplusf(float z)
{
    const float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + std::log(1.0f + std::exp(-az));
}

// ---------------------------------------------------------------------------
// Color management — per-channel transfer only (gamut is preserved: the tone
// spine operates per channel, so no gamut matrix is needed here). The declared
// working space is DaVinci Wide Gamut / Intermediate.
// ---------------------------------------------------------------------------
static inline float diDecode(float v)   // DI -> scene-linear
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M)
                              : (std::exp2(v / kDI_C - kDI_B) - kDI_A);
}
static inline float diEncode(float L)   // scene-linear -> DI
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M)
                              : ((std::log2(L + kDI_A) + kDI_B) * kDI_C);
}

// ACEScct (used only when the user declares an ACES timeline).
static inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return std::exp2(v * 17.52f - 9.72f);
}
static inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (std::log2(L) + 9.72f) / 17.52f;
}

static inline float decodeToLinear(int cs, float v)
{
    switch (cs) {
        case SPEAK_CS_DWG_INTERMEDIATE: return diDecode(v);
        case SPEAK_CS_REC709_G24:       return v <= 0.0f ? 0.0f : std::pow(v, 2.4f);
        case SPEAK_CS_ACESCCT:          return acesCctDecode(v);
        case SPEAK_CS_DWG_LINEAR:
        case SPEAK_CS_LINEAR:
        default:                        return v;
    }
}
static inline float encodeFromLinear(int cs, float L)
{
    switch (cs) {
        case SPEAK_CS_DWG_INTERMEDIATE: return diEncode(L);
        case SPEAK_CS_REC709_G24:       return L <= 0.0f ? 0.0f : std::pow(L, 1.0f / 2.4f);
        case SPEAK_CS_ACESCCT:          return acesCctEncode(L);
        case SPEAK_CS_DWG_LINEAR:
        case SPEAK_CS_LINEAR:
        default:                        return L;
    }
}

// ---------------------------------------------------------------------------
// Gamut colorimetry (for the Bake-to-Rec.709 output mode). DaVinci Wide Gamut
// -> CIE XYZ (D65) is the published white-paper / colour-science matrix; XYZ ->
// linear Rec.709 is the standard D65 matrix. Both share D65 so a neutral maps
// to a neutral (the tiny residual is the published matrix's own rounding). The
// inverse matrices used by the round-trip CST gate live in the test.
// ---------------------------------------------------------------------------
static const float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
static const float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };

static inline void mul3(const float* m, float r, float g, float b,
                        float& oR, float& oG, float& oB)
{
    oR = m[0] * r + m[1] * g + m[2] * b;
    oG = m[3] * r + m[4] * g + m[5] * b;
    oB = m[6] * r + m[7] * g + m[8] * b;
}

// Working-space linear RGB -> linear Rec.709. Bake targets the DaVinci Wide
// Gamut working space (the documented use); Rec.709 in is a gamut-identity, and
// other declared spaces get a transfer-only bake (gamut left as-is) — stated in
// the UI hint so the mode never claims a conversion it does not perform.
static inline void gamutToRec709Lin(int cs, float r, float g, float b,
                                    float& oR, float& oG, float& oB)
{
    if (cs == SPEAK_CS_DWG_INTERMEDIATE || cs == SPEAK_CS_DWG_LINEAR) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        oR = r; oG = g; oB = b;
    }
}

// CIE L*a*b* (D65) — the perceptual metric the Phase-2 control arm scores in.
// Used by the Macbeth gate and (later) the subtractive-sat vector scope.
static inline float labF(float t)
{
    const float d = 6.0f / 29.0f;
    return (t > d * d * d) ? std::cbrt(t) : (t / (3.0f * d * d) + 4.0f / 29.0f);
}
static inline void xyzToLab(float X, float Y, float Z, float& L, float& a, float& b)
{
    const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
    const float fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}
static inline void dwgLinToLab(float r, float g, float b, float& L, float& aa, float& bb)
{
    float X, Y, Z;
    mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
    xyzToLab(X, Y, Z, L, aa, bb);
}

// ---------------------------------------------------------------------------
// The closed-form Hurter-Driffield characteristic curve  D(logH).
//
// Two stable softplus segments give independently tunable toe and shoulder
// while staying strictly monotone increasing for all inputs:
//   d1 = Dmin + (gamma/toe) * softplus( toe * (logH - speed) )   -- toe -> straight
//   D  = Dmax - (1/shoulder) * softplus( shoulder * (Dmax - d1) ) -- straight -> shoulder
// As toe,shoulder -> large this approaches a hard-clipped straight line of
// slope gamma between Dmin and Dmax; as they shrink the knees lengthen. The
// params ARE the plotted curve — no LUT, no interpolation drift.
// ---------------------------------------------------------------------------
static inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                            float toe, float shoulder, float speed)
{
    const float t = toe      < kKneeMin ? kKneeMin : toe;
    const float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    const float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}

// The full negative -> printer-light -> print cascade for one channel, in
// density. `stops` is the scene log2-exposure relative to 18% gray (the
// canonical Log-Exposure Spine datum). Returns the PRINT density D_prn.
static inline float chainDensity(float stops, int ch, const SpeakProfile& p)
{
    const float logH = stops * kLog10_2;
    // Negative characteristic curve.
    const float Dneg = hdCurve(logH, p.negDmin[ch], p.negDmax[ch], p.negGamma[ch],
                               p.negToe[ch], p.negShoulder[ch], p.negSpeed[ch]);
    // Print exposure = light through the negative (transmittance 10^-Dneg, i.e.
    // log10 exposure -Dneg) plus the printer-light timing offset, in the gap.
    const float printerOff = (p.printerMaster + p.printerLights[ch]) * kPrinterPt;
    const float logHprn = -Dneg + printerOff;
    // Print characteristic curve.
    return hdCurve(logHprn, p.prnDmin[ch], p.prnDmax[ch], p.prnGamma[ch],
                   p.prnToe[ch], p.prnShoulder[ch], p.prnSpeed[ch]);
}

// One channel of the density spine: scene-linear in -> scene-linear out.
// The positive is pivoted at 18% gray so the reference gray maps to itself
// (each channel divides out its own gray-reference density) — this is what
// makes a gray-balanced profile neutral-preserving by construction.
static inline float toneChannel(float lin, int ch, const SpeakProfile& p)
{
    const float stops = std::log2((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    const float Dprn  = chainDensity(stops, ch, p);
    const float Dref  = chainDensity(0.0f, ch, p);   // print density at 18% gray
    return k18Gray * pow10f(-(Dprn - Dref));
}

// ---------------------------------------------------------------------------
// Density-Space Subtractive Saturation + inter-image coupler (Phase 2). This
// is *why film looks like film*: it works in log-density (where dye density
// adds and transmittance multiplies), not in linear or HSL. Converting to
// density, amplifying each dye's deviation from the neutral (gray) density, and
// viewing back through 10^-D produces two structural signatures a linear 3x3
// saturation cannot fake — highlight chroma self-compresses toward base white,
// and hues skew toward the dye axes (from the asymmetric coupler). Neutral is
// invariant by construction: the transform acts on deviations from D-bar, which
// are zero on the gray axis. Standalone (usable on any grade).
// ---------------------------------------------------------------------------
static inline float density10(float lin)   // linear -> optical density (log10)
{
    return -std::log2(lin < 1e-6f ? 1e-6f : lin) * kLog10_2;  // log10 via log2 (parity-safe)
}
static inline float softCapKnee(float d, float cap)  // soft cap density at `cap`
{
    if (cap <= 0.0f) return d;                          // disabled
    return cap - (1.0f / 8.0f) * softplusf(8.0f * (cap - d));
}
static inline void subtractiveColor(float r, float g, float b, const SpeakProfile& p,
                                    float& oR, float& oG, float& oB)
{
    const float DR = density10(r), DG = density10(g), DB = density10(b);
    const float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    const float devR = DR - Dbar, devG = DG - Dbar, devB = DB - Dbar;
    // diagonal = per-dye subtractive saturation; off-diagonals = inter-image
    // coupler (unwanted-absorption cross terms). dyeCouple diagonal is unused.
    const float cR = (1.0f + p.subSat[0]) * devR - (p.dyeCouple[1] * devG + p.dyeCouple[2] * devB);
    const float cG = (1.0f + p.subSat[1]) * devG - (p.dyeCouple[3] * devR + p.dyeCouple[5] * devB);
    const float cB = (1.0f + p.subSat[2]) * devB - (p.dyeCouple[6] * devR + p.dyeCouple[7] * devG);
    const float DpR = softCapKnee(Dbar + cR, p.subSatKnee[0]);
    const float DpG = softCapKnee(Dbar + cG, p.subSatKnee[1]);
    const float DpB = softCapKnee(Dbar + cB, p.subSatKnee[2]);
    oR = pow10f(-DpR); oG = pow10f(-DpG); oB = pow10f(-DpB);
}
static inline bool dyeActive(const SpeakProfile& p)
{
    return p.subSat[0] != 0.0f || p.subSat[1] != 0.0f || p.subSat[2] != 0.0f ||
           p.dyeCouple[1] != 0.0f || p.dyeCouple[2] != 0.0f || p.dyeCouple[3] != 0.0f ||
           p.dyeCouple[5] != 0.0f || p.dyeCouple[6] != 0.0f || p.dyeCouple[7] != 0.0f;
}

// ---------------------------------------------------------------------------
// Split toning / film-referred tonal-zone balance (Phase 3) — the lift-gamma-
// gain replacement, done in the density domain.
//
// Per-channel density offsets weighted by a 3-zone partition of unity (toe /
// mid / shoulder) anchored to the WORKING H&D CURVE rather than fixed luma
// cuts: the tone position is the pixel's own neutral density, and the spine
// pivots 18% gray to D = -log10(0.18) = 0.745, so that IS the mid anchor.
//
// Two structural properties fall out, both of which LGG cannot offer:
//  - MIDS STAY NEUTRAL BY CONSTRUCTION: at the pivot both zone weights are
//    exactly 0, so the mid tone receives no offset at all — you cannot tint
//    shadows and highlights and accidentally drag mid-gray.
//  - HUE-STABLE: additive in density == MULTIPLICATIVE in linear, so a tint
//    scales a channel rather than offsetting it the way LGG's lift does (which
//    is what desaturates and swings hue in the shadows).
// Chromogenic crossover (cool shadows / warm highlights) is simply opposite
// shadow and highlight offsets; cross-process is a more extreme pair.
// ---------------------------------------------------------------------------
static inline float smooth01(float t)
{
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
static inline void splitWeights(float Dbar, const SpeakProfile& p, float& wShadow, float& wHigh)
{
    const float grayD  = 0.744727f;                          // D of 18% gray (the spine's pivot)
    const float pivotD = grayD - p.splitPivot * kLog10_2;    // pivot: stops -> density
    const float halfW  = 0.25f + 1.5f * clampf(p.splitBalance, 0.0f, 1.0f);
    const float x = (Dbar - pivotD) / halfW;                 // signed: + = darker, - = brighter
    wShadow = smooth01(x);        // -> 1 into the toe (shadows)
    wHigh   = smooth01(-x);       // -> 1 into the shoulder (highlights)
}                                 // at the pivot both are 0 => the mid zone is untouched
static inline void splitTone(float r, float g, float b, const SpeakProfile& p,
                             float& oR, float& oG, float& oB)
{
    const float DR = density10(r), DG = density10(g), DB = density10(b);
    const float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, wS, wH);
    oR = pow10f(-(DR + wS * p.splitShadow[0] + wH * p.splitHigh[0]));
    oG = pow10f(-(DG + wS * p.splitShadow[1] + wH * p.splitHigh[1]));
    oB = pow10f(-(DB + wS * p.splitShadow[2] + wH * p.splitHigh[2]));
}
static inline bool splitActive(const SpeakProfile& p)
{
    return p.splitShadow[0] != 0.0f || p.splitShadow[1] != 0.0f || p.splitShadow[2] != 0.0f ||
           p.splitHigh[0] != 0.0f || p.splitHigh[1] != 0.0f || p.splitHigh[2] != 0.0f;
}

// The generic dye cross-absorption pattern. The hue skew lives ENTIRELY in the
// within-row asymmetry (the density deviations sum to zero, so equal cross terms
// in a row would collapse to a plain saturation boost). Follows the classic
// published unwanted absorptions — cyan absorbs mostly green, magenta mostly
// blue, yellow mostly green — behavior-named, cloning no stock. Verified by the
// Macbeth CIELAB control arm (test/test_speak_macbeth.cpp).
static const float kCouplerRG = 0.28f, kCouplerRB = 0.06f;
static const float kCouplerGR = 0.08f, kCouplerGB = 0.30f;
static const float kCouplerBR = 0.04f, kCouplerBG = 0.22f;
static inline void setDyeCoupler(SpeakProfile& p, float amount)
{
    p.dyeCouple[1] = kCouplerRG * amount; p.dyeCouple[2] = kCouplerRB * amount;
    p.dyeCouple[3] = kCouplerGR * amount; p.dyeCouple[5] = kCouplerGB * amount;
    p.dyeCouple[6] = kCouplerBR * amount; p.dyeCouple[7] = kCouplerBG * amount;
}

// ---------------------------------------------------------------------------
// HALATION (Phase 4) — Speak's first spatial module.
//
// THE PHYSICS, AND WHY THE INJECTION POINT IS WHERE IT IS.
// Light focused by the lens passes THROUGH the emulsion, reflects off the
// base/air interface, and re-enters the emulsion displaced sideways. Therefore:
//   * the SOURCE of the scattering light is the SCENE image on the emulsion;
//   * what the scattered light ADDS TO is the negative's EXPOSURE H;
//   * exposure is additive in linear light (photons add).
// So H_c'(x) = H_c(x) + amount * w_c * (PSF * excess(H_c))(x), and the WHOLE
// negative -> printer -> print cascade then sees H'. The print's shoulder is
// what compresses the halo, which is why the halo self-limits and goes
// white-hot in its core instead of pumping unbounded red (test/proto_halation).
//
// There is NO halation injection between the negative and the print. The print's
// exposure is 10^-Dneg (chainDensity, above): it is BRIGHT where the SCENE was
// DARK, so a highlight-driven injection there DARKENS highlight edges instead of
// haloing them — measured, a bright disc inverts 8.056 -> 0.007. (Scatter in the
// optical printer, and the print stock's own halation, ARE gap-sited; they are
// real but different phenomena with the opposite polarity, and are not this
// module.) toneChannel/chainDensity/hdCurve are therefore untouched by halation.
//
// WHY IT IS RED, AND WHAT THAT CLAIM IS WORTH. The light that reaches the base
// has already been depleted of blue and green — by the yellow filter layer and
// the upper emulsion layers — and the red-sensitive layer sits closest to the
// base. That MECHANISM is published sensitometry. The specific {1, 0.30, 0.10}
// ratio is NOT: it is our MODELLED DEFAULT for it, gated by H3/H4 in
// test/proto_halation.cpp rather than cited to anyone. It is not a user control,
// and no shipping hint attributes it to a stock or to a measurement.
// (An earlier revision called these "Beer-Lambert-shaped, published AH
// behaviour" with no citation — that fitted a made-up ratio and then cited
// Beer-Lambert for it, and is withdrawn. Nor is this the eye's glare-spread
// function: Vos & van den Berg / CIE 135 model corneal and lens scatter in
// degrees of visual angle for a human observer. Citing them for base reflection
// in an emulsion would be laundering a different physical system's authority.)
//
// The weight is applied PER CHANNEL to that channel's OWN scattered light.
// Collapsing RGB to a luminous mean and tinting it red — what the prototype
// originally did, and what the arm could not see — lets BLUE photons manufacture
// a RED halo (measured: a (0,0,2.0) source got a red scatter 10x its blue).
// Gated by H3/H4 in test/proto_halation.cpp.
// ---------------------------------------------------------------------------
static const float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };

// The scatter source in scene-linear light, one channel. Pure, so the pyramid
// builder and the tests call the exact same function.
static inline float halExcess(float lin, float thresh)
{
    const float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}

// Effective halation amount: 0 unless the optics stage is on. Callers ALSO gate
// on the tone spine — halation re-exposes the NEGATIVE, so with no spine there
// is no negative, and injecting scatter into linear with no curve downstream is
// EXACTLY the end-chain overlay the control arm rejected. The UI disables the
// controls when Film Tone is off and says why.
static inline float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}

// Halation spreads a fixed distance in mm on the film, so its radius is
// FORMAT-relative: sigma scales with frame height, never with pixel count. This
// is what makes the look survive a proxy/full-res switch (measured: the halo
// profile matches to ~1% across a 4x resolution change, G16).
//
// KNOWN AND MEASURED LIMIT, stated rather than papered over: the halo's actual
// half-width tracks this sigma closely once sigma is well resolved (HWHM/sigma
// = 0.93..1.00 above ~20 px) but UNDERSIZES below that, down to ~0.6x at a
// sigma of ~1 px — a pyramid cannot represent a halo a few pixels wide. There is
// also a ~8% ripple as the target level slides between octaves. Neither is
// visible in use, and the Radius hint promises a percentage of frame height, not
// an exact half-width. Pinned by G13.
static inline float halSigmaPx(int H, const SpeakParams& pr)
{
    const float s = pr.profile.halRadius * 0.01f * static_cast<float>(H);
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
static inline bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- pyramid geometry (identical arithmetic in all four backends) ----
static inline int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
// Dimensions and the packed arena offset (in PIXELS, x3 for RGB) of level L.
static inline void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
static inline int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    const int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);   // offset just past the last level
    return off;
}

// Level L's effective full-res sigma. Two terms, and BOTH are load-bearing:
//   1. DECIMATION. The [1,3,3,1]/8 kernel has variance 0.75 in its own level's
//      pixel units, so going from level j to j+1 adds 0.75*(2^j)^2 full-res
//      px^2:   sum_{j<L} 0.75*4^j = 0.25*(4^L - 1).
//   2. UPSAMPLE. Reading level L back at full res bilinearly interpolates
//      samples 2^L px apart, and a tent filter of spacing h has variance h^2/6,
//      adding 4^L/6. (Level 0 needs no interpolation, so it takes no such term.)
//   var_L = 0.25*(4^L - 1) + [L>0] * 4^L/6      =>   sigma_L -> 0.6455 * 2^L
//
// Term 2 was missing from the first draft and the ladder came out 1.29x low at
// EVERY level — a constant scale error, so the octave spacing still looked
// perfect and only an absolute measurement could catch it. G13 caught it:
// predicted 15.99 vs 20.65 measured at level 5. It is PINNED THERE against the
// real impulse response, because a wrong ladder silently resizes the shipped
// halo and parity would never see it.
static const float kHalSigmaC = 0.645497f;   // sqrt(0.25 + 1/6): sigma_L / 2^L
static inline float halLevelSigma(int L)
{
    const float q = std::exp2(2.0f * static_cast<float>(L));   // 4^L
    const float up = (L > 0) ? (q * (1.0f / 6.0f)) : 0.0f;
    return std::sqrt(0.25f * (q - 1.0f) + up);
}

// The mixture weights. Level L contributes a Gaussian of sigma_L ~ 2^(L-1); a
// weighted sum over octaves is a MULTI-SCALE scatter — a bright core plus a wide
// faint skirt — which a single Gaussian cannot produce at any sigma. Levels
// tighter than the target fall off fast (kHalCoreFall); broader levels decay
// slowly (kHalSkirtFall = 1 => weight halves per octave), which is the skirt.
//
// The tail this produces is a power law, by construction: at radius r the
// dominant term is the level with sigma_L ~ r, contributing w_L/sigma_L^2 (a 2D
// Gaussian's peak amplitude goes as 1/sigma^2). With w_L ~ 2^-d and sigma_L ~
// 2^d that is 2^-d/4^d = 8^-d, and r ~ 2^d, so the contribution falls as r^-3 —
// versus a Gaussian's exp(-r^2/2sigma^2). Measured against a matched-core
// Gaussian in G14; kHalSkirtFall is a MODELLED DEFAULT, not a cited constant.
static const float kHalCoreFall  = 3.0f;   // octaves tighter than target: fast cut
static const float kHalSkirtFall = 1.0f;   // octaves broader: halve per octave

// The fall rates are ARGUMENTS because two optics modules share this pyramid
// machinery with different profiles: halation reads (kHalCoreFall,
// kHalSkirtFall); bloom reads a broader skirt (its veiling floor is a frame-
// mean term handled at normalization, not a level weight — see
// buildBloomScatter). Halation's call sites pass its old constants verbatim,
// so its arithmetic is bit-identical to the pre-bloom build (pinned by the
// halation gates).
static inline float halLevelWeight(int L, float sigmaTarget,
                                   float coreFall, float skirtFall)
{
    // Target level from sigma_L -> kHalSigmaC * 2^L  =>  L_t = log2(sigma/C).
    // Continuous in L_t, so the radius slider is smooth: it does NOT snap to
    // octaves as the level bracket moves (gated by G16 across a 4x resolution
    // change, which is the same machinery).
    const float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    const float Lt = std::log2(s / kHalSigmaC);
    const float d = static_cast<float>(L) - Lt;
    return (d <= 0.0f) ? std::exp2(coreFall * d) : std::exp2(-skirtFall * d);
}

// ---- pyramid taps (per-pixel, so the GPU kernels are textual ports) ----
// Clamp-to-edge fetch. NOTE the consequence, which is stated in the UI rather
// than hidden: a highlight AT the frame edge scatters against a clamped border,
// so its halo carries slightly less energy than one in frame centre. There is
// no closed form to compensate; the energy gate (G15) therefore measures the
// INTERIOR, at least 2 sigma from every border, where the identity is exact.
static inline float halFetch(const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    const int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    const int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[(static_cast<size_t>(off) + static_cast<size_t>(yy) * lw + xx) * 3 + c];
}

// The [1,3,3,1]/8 separable binomial decimation. Its taps sum to 1 on each axis,
// so it is MEAN-PRESERVING: every level carries the same total energy as level 0
// once upsampled, which is what makes the mixture energy-normalized by Sum(w)=1
// alone. Deterministic and atomics-free (lesson: Apple's OpenCL miscompiles
// global int32 atomics, so no spatial pass may depend on them).
static const float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
static inline void halDecimatePixel(const float* arena, int sOff, int sW, int sH,
                                    int dx, int dy, float* out)
{
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                acc += kHalDec[i] * kHalDec[j] *
                       halFetch(arena, sOff, sW, sH, 2 * dx - 1 + i, 2 * dy - 1 + j, c);
        out[c] = acc;
    }
}

// Cubic B-spline read of one level onto a (W,H) grid.
//
// B-spline, NOT bilinear and NOT Catmull-Rom, for two reasons that both matter
// here: it is C2 (bilinear is only C0, and its derivative discontinuity at every
// texel boundary is what made the skirt read as hard rectangular BLOCKS), and
// its weights are NON-NEGATIVE and sum to 1 (Catmull-Rom's overshoot would
// undershoot a scatter field below zero and break energy conservation). It is
// approximating rather than interpolating — it blurs slightly — which for a
// light-scatter field is exactly what is wanted.
static inline void halBSpline(float t, float* w)
{
    const float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
static inline float halSampleLevel(const float* arena, int off, int lw, int lh,
                                   int W, int H, int x, int y, int c)
{
    const float fx = (static_cast<float>(x) + 0.5f) * static_cast<float>(lw) / static_cast<float>(W) - 0.5f;
    const float fy = (static_cast<float>(y) + 0.5f) * static_cast<float>(lh) / static_cast<float>(H) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
    float wx[4], wy[4];
    halBSpline(fx - static_cast<float>(x0), wx);
    halBSpline(fy - static_cast<float>(y0), wy);
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        float row = 0.0f;
        for (int i = 0; i < 4; ++i)
            row += wx[i] * halFetch(arena, off, lw, lh, x0 - 1 + i, y0 - 1 + j, c);
        acc += wy[j] * row;
    }
    return acc;
}

// The mixture's total weight — the normalizer that makes the pyramid
// energy-preserving (each level is mean-preserving, so a convex combination of
// them carries exactly the source's energy). Same loop on every backend.
static inline float halWeightSum(int nLev, float sigmaTarget,
                                 float coreFall, float skirtFall)
{
    float wsum = 0.0f;
    for (int L = 0; L < nLev; ++L)
        wsum += halLevelWeight(L, sigmaTarget, coreFall, skirtFall);
    return wsum;
}

// COARSE-TO-FINE ACCUMULATE — one pixel of level L.
//     acc_L = w_L * level_L + upsample_2x(acc_{L+1})
// running L = nLev-1 down to 0, in place in the arena. acc_0 is then the whole
// (unnormalized) mixture at full resolution.
//
// WHY NOT SAMPLE EVERY LEVEL DIRECTLY AT FULL RES (what this replaced): reading
// a coarse level at full res interpolates between texels 2^L px apart, and
// bilinear is only C0 — the derivative discontinuity at every texel boundary
// reads as VISIBLE RECTANGULAR BLOCKS, worst on exactly the coarse levels that
// carry the skirt. It shipped past every numeric gate (energy, ladder, tail,
// resolution) because none of them measured isotropy, and it was plainly visible
// the moment the scatter field was rendered: the PSF had ~9% angular variation
// at every radius and 372% at r=128 (G18 now measures this; it read 3.72 before
// this change and ~0.02 after).
//
// Going one octave at a time fixes it because each step interpolates only
// between ADJACENT samples and is then re-filtered by every step below it. It is
// also strictly CHEAPER: the total work is sum_L (level L's pixels) = 4/3 N,
// against nLev*N for the full-res reads.
//
// In-place is safe: a thread reads its OWN (x,y) at level L and a neighbourhood
// at level L+1 (a disjoint region), and writes only its own (x,y) at level L.
static inline void halAccumPixel(float* arena, int W, int H, int L, int nLev,
                                 float sigmaTarget, float coreFall, float skirtFall,
                                 int x, int y, float* out)
{
    int lw, lh, off;
    halLevelInfo(W, H, L, lw, lh, off);
    const float wl = halLevelWeight(L, sigmaTarget, coreFall, skirtFall);
    if (L >= nLev - 1) {                       // the coarsest level: nothing above it
        for (int c = 0; c < 3; ++c) out[c] = wl * halFetch(arena, off, lw, lh, x, y, c);
        return;
    }
    int cw, ch, coff;
    halLevelInfo(W, H, L + 1, cw, ch, coff);
    for (int c = 0; c < 3; ++c)
        out[c] = wl * halFetch(arena, off, lw, lh, x, y, c)
               + halSampleLevel(arena, coff, cw, ch, lw, lh, x, y, c);
}

// ---------------------------------------------------------------------------
// GRAIN (Phase 4) — the emulsion's own noise, and the Hush handoff.
//
// THE PHYSICS. Film grain is not additive video noise: the developed dye clouds
// ARE the image, so their randomness is a fluctuation of DENSITY — which
// multiplies the transmitted light (additive in density == multiplicative in
// linear, the same identity the split-toning module rests on). And granularity
// GROWS with density: more dye, more clouds, more fluctuation — which on a
// POSITIVE print means the shadows are the loud end, exactly the "shadow-loud"
// behaviour the design study measured on real footage. So:
//     D_c' = D_c + amount * kGrainScale * sqrt(min(D_c, cap)) * n_c(x, y, frame)
// applied per RGB DYE LAYER (independent noise per channel — film's layers are
// separate emulsions) in the density domain of the FINISHED look
// (print-referred), then viewed back through 10^-D.
//
// The sqrt(D) shape is the published granularity-vs-density behaviour family
// (Selwyn-style RMS granularity rising with density); the constant kGrainScale
// and the cap are OUR MODELLED DEFAULTS, stated as such — no stock is named and
// none was measured. Consequences that fall out of the physics rather than
// being programmed in: paper white is grainless (D -> 0 => sigma -> 0), and the
// multiplicative form means grain can never push a value negative.
//
// HONESTY NOTE, so nobody "fixes" this: multiplicative density noise has a
// small POSITIVE exposure bias (Jensen: E[10^-(D+s*n)] > 10^-D), ~0.5*(s*ln10)^2
// relative — ~0.2% at mid-gray at amount 0.5. Real film has the same bias for
// the same reason. It is measured and bounded by gate G22, not removed.
//
// TEMPORAL BEHAVIOUR: independent every frame (the hash includes frameIndex).
// Real grain boils; correlating it across frames reads as a dirty lens and was
// explicitly killed in the design study. Gated (G21).
//
// THE HUSH HANDOFF (matteSource, v1.0 — SPEC-1.0 §1): when a matte source is
// active, the matte keys the grain. Hush >= 3.7 with "Export Clean Matte to
// Alpha" writes its clean-confidence matte — clamp((effN-1)/6, 0, 1), high
// where the denoiser averaged deep (texture was flattened: put grain back),
// low on the motion the gate protected (the real noise survives there: add
// less on top). Speak reads the [0,1] value as-is; the calibration lives in
// Hush. The source is EXPLICIT now:
//   SPEAK_MATTE_OFF    grain uniform; conf never read (gated: G23).
//   SPEAK_MATTE_KEY    the blue key wire. The plugin packs the key clip into
//                      the src alpha this function receives as `conf`. If the
//                      key is NOT wired, matteKeyMissing=1 and conf reads as 0
//                      everywhere — absence means absence, grain sits at the
//                      Floor. v0.3's silent fall-through to the incoming alpha
//                      (opaque on the color page => full grain, floor ignored)
//                      is the bug this replaced; never restore it.
//   SPEAK_MATTE_ALPHA  the incoming image alpha, opt-in (Fusion / in-band).
// grainMatteFloor is the amount kept where the matte is 0. Consume-on-use:
// any active source forces the OUTPUT alpha opaque (maskExternal) — the matte
// never rides past its consumer.
// ---------------------------------------------------------------------------
static const float kGrainScale = 0.045f;  // sigma_D at D=1, amount=1 (modelled default)
static const float kGrainDCap  = 4.0f;    // density cap feeding sqrt (deep-shadow safety)

// Integer hash noise in [-1,1] — TEXTUALLY Hush's hashNoise (nr_core.h): pure
// uint32 math, identical on CPU/Metal/CUDA/OpenCL by construction.
static inline float grainHash(uint32_t ix, uint32_t iy, uint32_t f, uint32_t ch)
{
    uint32_t h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (static_cast<float>(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}

// Variance-normalized bilinear value noise at a lattice of pitch `size` px.
// Plain bilinear value noise has a 4:1 VARIANCE ripple between lattice nodes
// and cell centres (sum of squared weights: 1.0 at a node, 0.25 mid-cell),
// which reads as a periodic grid stamped on the grain — the same class of
// artefact G18 caught on the halation pyramid. Dividing by sqrt(sum w^2) makes
// the variance constant everywhere; the field is still continuous and still
// lowpass at the lattice scale.
static inline float grainLattice(float x, float y, float size, uint32_t f, uint32_t ch)
{
    const float gx = x / size, gy = y / size;
    const int ix = static_cast<int>(std::floor(gx)), iy = static_cast<int>(std::floor(gy));
    const float tx = gx - static_cast<float>(ix), ty = gy - static_cast<float>(iy);
    const float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty,          w11 = tx * ty;
    const float n = w00 * grainHash(static_cast<uint32_t>(ix),     static_cast<uint32_t>(iy),     f, ch)
                  + w10 * grainHash(static_cast<uint32_t>(ix + 1), static_cast<uint32_t>(iy),     f, ch)
                  + w01 * grainHash(static_cast<uint32_t>(ix),     static_cast<uint32_t>(iy + 1), f, ch)
                  + w11 * grainHash(static_cast<uint32_t>(ix + 1), static_cast<uint32_t>(iy + 1), f, ch);
    const float ww = w00 * w00 + w10 * w10 + w01 * w01 + w11 * w11;
    return n / std::sqrt(ww);
}

// The grain field: an octave DIFFERENCE, n_size - n_2size. The subtraction
// kills the DC and the low-frequency blotch a plain lowpass noise carries (the
// "grain has a size" claim is a BANDPASS claim: energy at the grain pitch,
// rolled off both above and below). Different salt per octave so the two are
// independent; 1/sqrt(2) restores unit-ish sd after the difference.
static inline float grainBand(float x, float y, float sizePx, uint32_t f, uint32_t ch)
{
    return (grainLattice(x, y, sizePx, f, ch) -
            grainLattice(x, y, sizePx * 2.0f, f, ch + 8u)) * 0.70710678f;
}

// Grain pitch in px: % of frame height (a physical size on the film scales with
// the frame, never with pixel count — same format-relative logic as halation),
// floored at 1 px (a sub-pixel lattice cannot be represented and aliases).
static inline float grainSizePx(int H, const SpeakParams& pr)
{
    const float s = pr.profile.grainSize * 0.01f * static_cast<float>(H);
    return s < 1.0f ? 1.0f : s;
}
static inline bool grainActive(const SpeakParams& pr)
{
    return (pr.enableGrain != 0) && (pr.profile.grainAmount > 0.0f);
}

// Apply grain to the LOOK's working-linear output. `conf` is the pixel's INPUT
// ALPHA, forwarded by the caller (the pixel path and the scope's measurement
// pass must forward the SAME value, which is why this stays a pure function of
// its arguments). It is only read when a matte source is active, and clamped
// here; a selected-but-unwired key forces it to 0 (the Floor).
static inline void applyGrain(float& r, float& g, float& b, float conf,
                              int x, int y, int H, const SpeakParams& pr)
{
    if (!grainActive(pr)) return;
    const float sz = grainSizePx(H, pr);
    const float confK = (pr.matteKeyMissing != 0) ? 0.0f : conf;
    const float m  = (pr.matteSource != SPEAK_MATTE_OFF)
                   ? lerpf(clampf(pr.grainMatteFloor, 0.0f, 1.0f), 1.0f, clampf(confK, 0.0f, 1.0f))
                   : 1.0f;
    const float a  = pr.profile.grainAmount * m * kGrainScale;
    if (a <= 0.0f) return;
    const uint32_t fr = static_cast<uint32_t>(pr.frameIndex);
    const float fx = static_cast<float>(x), fy = static_cast<float>(y);
    float* ch3[3] = { &r, &g, &b };
    for (int c = 0; c < 3; ++c) {
        const float D = density10(*ch3[c]);
        const float Dc = D < 0.0f ? 0.0f : (D > kGrainDCap ? kGrainDCap : D);
        const float sigmaD = a * std::sqrt(Dc);
        const float n = grainBand(fx, fy, sz, fr, static_cast<uint32_t>(c));
        *ch3[c] = pow10f(-(D + sigmaD * n));
    }
}

// ---------------------------------------------------------------------------
// BLOOM / VEILING GLARE (Phase 4, spec 1B.5) — the second optics module.
//
// THE PHYSICS. Glare in the viewing/printing optics scatters a fraction of
// the light of the FINISHED image: some of every ray is redistributed by the
// glass, the aperture and internal reflections. Two consequences the model
// keeps, because they are the phenomenon:
//   1. A lens TRANSMITS — it creates no light and (to first order) loses
//      none. So bloom is a CONVEX MIX toward a mean-preserving scatter of
//      the image itself:  out = (1-a)*L + a*S[L],  with S built from
//      mean-preserving decimation and normalized weights (Sum w = 1). Total
//      linear energy is conserved BY CONSTRUCTION (gate G26 measures it, and
//      measures that the additive strawman FAILS it). Highlights dim
//      slightly as their light leaves; surroundings lift as it arrives —
//      spec 1B.5's "borrows highlight energy and subtracts it from the
//      source" falls out rather than being programmed in.
//   2. There is NO THRESHOLD. A PSF applies to every ray, not only to
//      values above a knee; the visible effect concentrates at highlights
//      only because they dominate the scattered sum. (bloomThresh is
//      deliberately absent — a thresholded bloom is the video-game screen
//      effect, and it cannot conserve energy across the knee.)
//
// SPECTRALLY NEUTRAL, deliberately: glass scatter has no antihalation layer,
// so per-channel weights are all 1. A blue neon blooms blue (G28) — where
// halation, one module up the light path, halates red (H3a). The two views
// side by side are the honest demonstration that these are different physics.
//
// THE VEILING FLOOR. The far-field term of the same glare: stray light that
// arrives UNIFORMLY, lifting blacks in proportion to how much light the
// frame carries (the veiling-glare of ISO 9358 — that standard describes
// CAMERA LENSES, so citing it here passes the "is the cited work about THIS
// phenomenon?" test; the eye's glare-spread function would not). It is the
// pyramid taken to its logical end — the frame MEAN — mixed into the same
// normalized sum:
//     S = [ Sum_L w_L * up(level_L)  +  w_veil * mean(look) ] / (Sum w + w_veil)
//   - the black-lift scales with FRAME luminance exactly (G29c), which a
//     coarsest-LEVEL weight would not give: an 8x5 level's corner texels
//     never see a centre highlight, and the first draft's floor measurably
//     failed to rise at the far corner (the gate caught it — G29b);
//   - normalization keeps the total weight 1, so conservation survives any
//     veil setting (a hazier element trades local contrast for flare);
//   - the mean is computed by finishing the decimation chain (a ~96-texel
//     sum over the coarsest level), so it stays atomics-free (Apple's OpenCL
//     atomics bug can never touch it).
// bloomVeil is the veil's SHARE of the scattered light: w_veil = v/(1-v) *
// (sum of the profile weights), which makes the share exactly v after
// normalization — a claim the hint can state and G29a pins as arithmetic.
// kBloomSkirtFall and the defaults are MODELLED, stated as such; the
// MECHANISM (near-field halo + luminance-proportional veil) is the published
// part.
//
// SITED after the print, in the look's working linear (spec 1B.5: "added in
// linear AFTER the print"), grain included — the projector sees the print's
// grain through the same glass. Physically this is the OTHER side of the
// tone scale from halation (which re-exposes the negative), which is why
// bloom builds its own pyramid on the LOOK field rather than literally
// re-reading halation's scene-referred excess pyramid: an energy-conserving
// mix must subtract light in the same domain where it re-adds it, and the
// spec's "read the same pyramid twice" predates the injection-site
// measurement that fixed halation's build order. The MACHINERY is shared
// textually (decimate / B-spline accumulate / weight profile with arguments).
// ---------------------------------------------------------------------------
static const float kBloomSkirtFall = 0.5f;  // broader skirt than halation's 1.0
                                            // (modelled default, stated as such)

static inline float bloomAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? clampf(pr.profile.bloomAmount, 0.0f, 1.0f) : 0.0f;
}
// Unlike halation, bloom does NOT require enableTone: it acts on the look's
// output whatever the look is (a dye-only grade still projects through glass).
static inline bool bloomActive(const SpeakParams& pr)
{
    return (pr.strength > 0.0f) && (bloomAmountOf(pr) > 0.0f);
}
static inline float bloomSigmaPx(int H, const SpeakParams& pr)
{
    const float s = pr.profile.bloomRadius * 0.01f * static_cast<float>(H);
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
// The veil's mixture weight, from its target share v (see the header block
// above). Frame-uniform, tiny loop, identical on every backend.
static inline float bloomVeilAdd(int nLev, float sigmaTarget, const SpeakParams& pr)
{
    const float v = clampf(pr.profile.bloomVeil, 0.0f, 0.9f);
    if (v <= 0.0f) return 0.0f;
    const float base = halWeightSum(nLev, sigmaTarget, kHalCoreFall, kBloomSkirtFall);
    return v / (1.0f - v) * base;
}

// The energy-conserving mix at one pixel. `s*` is the normalized bloom
// scatter sampled at this pixel; the amount rides the global Strength like
// every other look stage (the scope-defect-(a) lesson: nothing in the look
// may ignore the mix the pixels use).
static inline void bloomApplyPixel(float& r, float& g, float& b,
                                   float sR, float sG, float sB,
                                   const SpeakParams& pr)
{
    if (!bloomActive(pr)) return;
    const float a = bloomAmountOf(pr) * clampf(pr.strength, 0.0f, 1.0f);
    r = lerpf(r, sR, a);
    g = lerpf(g, sG, a);
    b = lerpf(b, sB, a);
}

// ---------------------------------------------------------------------------
// VIGNETTE (Phase 4) — the cosine-fourth law.
//
// THE PHYSICS. Off-axis image points receive less light for four compounding
// geometric reasons (inverse-square to a tilted exit pupil, twice-obliquity,
// pupil foreshortening); their product is the classic cos^4(theta) natural
// illumination falloff of photographic lenses — published lens photometry
// about exactly this phenomenon, so citing it passes the project's own test.
// Mechanical vignetting (hoods, stacked filters) is steeper and is NOT
// modelled; `amount` mixes toward the natural cos^4 floor, no further.
//
// SITED AT CAPTURE, deliberately: the taking lens attenuates the light that
// EXPOSES the negative, so a vignetted corner rides DOWN the H&D toe — its
// contrast and color respond like film, not like a post-look dim. The gain
// therefore multiplies scene-linear ahead of everything, including the
// halation excess extraction (light that never arrived cannot scatter), and
// G33 pins the injection site as arithmetic: a corner pixel must equal a
// centre pixel whose scene light was pre-scaled by the same gain.
//
// Geometry: tan(theta) at a pixel scales linearly with its distance from
// centre; vignField names the HALF-DIAGONAL field angle in degrees (a lens
// FOV knob — 27 degrees ~ a normal lens; MODELLED default, the hint says
// so). cos^2 = 1/(1+tan^2) keeps it trig-free per pixel.
// ---------------------------------------------------------------------------
static inline bool vignActive(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) && (pr.strength > 0.0f)
        && (pr.profile.vignAmount > 0.0f);
}
static inline float vignGain(int x, int y, int W, int H, const SpeakParams& pr)
{
    if (!vignActive(pr)) return 1.0f;
    const float a = clampf(pr.profile.vignAmount, 0.0f, 1.0f)
                  * clampf(pr.strength, 0.0f, 1.0f);
    const float cx = 0.5f * static_cast<float>(W - 1);
    const float cy = 0.5f * static_cast<float>(H - 1);
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    const float rhd2 = cx * cx + cy * cy;            // half-diagonal, squared
    const float r2 = (dx * dx + dy * dy) / (rhd2 > 0.0f ? rhd2 : 1.0f);
    const float tanm = std::tan(pr.profile.vignField * 0.017453293f);
    const float c2 = 1.0f / (1.0f + r2 * tanm * tanm);   // cos^2(theta)
    return lerpf(1.0f, c2 * c2, a);
}

// ---------------------------------------------------------------------------
// GATE WEAVE (Phase 4) — the transport's picture-position noise.
//
// Film in a camera or projector gate is registered by perforations with real
// mechanical slop: the picture wanders, slowly and slightly, mostly along
// the transport axis. The MODEL: a per-frame global sub-pixel displacement
// (dx, dy), built as an octave stack of hash-lattice value noise over the
// FRAME INDEX — amplitude proportional to period, i.e. a ~1/f spectrum: the
// top octave stands in for per-frame pulldown jitter, the bottom for the
// long wander. All constants are MODELLED defaults, stated as such.
//
// Deterministic closed form of frameIndex, deliberately: scrubbing is
// repeatable, a re-render is identical, and all four backends agree bit-for-
// bit because the source of randomness is the SAME uint32 hash grain uses —
// integer math plus lerp, no trig. (A sine stack was rejected: sin() of a
// six-figure frame index diverges across backend math libraries, and its
// octave periods share a short LCM that visibly loops.)
//
// The resample is Catmull-Rom, per spec 1B: a PICTURE resample must
// interpolate — it reproduces a linear ramp exactly and does not soften —
// where the scatter pyramid wanted B-spline's non-negative smoothing. Alpha
// is displaced WITH the color it describes: Hush's matte must keep pointing
// at the pixels it measured; "alpha passes through" means un-invented, not
// un-moved. Clamp-to-edge exposes no invented content at the frame edge.
//
// Weave applies to the RESULT view only (and is the LAST spatial operation,
// before the scopes are overlaid): the isolated diagnostic views hold still
// so they can be read, and the scopes are panel chrome, not picture.
// ---------------------------------------------------------------------------
static const int kWeaveOctaves = 6;   // periods 2..64 frames (~1/f stack)

static inline bool weaveActive(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) && (pr.strength > 0.0f)
        && (pr.profile.weaveAmount > 0.0f)
        && (pr.viewMode == SPEAK_VIEW_RESULT);
}

// 1-D value noise on the frame-index axis: hash at integer frames, C1
// smoothstep between them. Bounded arguments at any frame index.
static inline float weaveSmooth1D(float t, uint32_t salt)
{
    const float tf = std::floor(t);
    const int i0 = static_cast<int>(tf);
    const float fr = t - tf;
    const float s = fr * fr * (3.0f - 2.0f * fr);
    const float n0 = grainHash(static_cast<uint32_t>(i0),     0x5EA7u, 0u, salt);
    const float n1 = grainHash(static_cast<uint32_t>(i0 + 1), 0x5EA7u, 0u, salt);
    return n0 + (n1 - n0) * s;
}

// The frame's displacement in pixels. Frame-uniform (the whole picture moves
// as one, which is what a gate does), amplitude format-relative (% of frame
// height), transport axis (y) wandering more than lateral (modelled 1.4x).
static inline void weaveDisp(const SpeakParams& pr, int H, float& dx, float& dy)
{
    dx = 0.0f; dy = 0.0f;
    if (!weaveActive(pr)) return;
    const float amp = pr.profile.weaveAmount * 0.01f * static_cast<float>(H)
                    * clampf(pr.strength, 0.0f, 1.0f);
    const float speed = pr.profile.weaveSpeed > 0.0f ? pr.profile.weaveSpeed : 1.0f;
    const float t = static_cast<float>(pr.frameIndex) * speed;
    float sx = 0.0f, sy = 0.0f, norm = 0.0f;
    for (int o = 0; o < kWeaveOctaves; ++o) {
        const float period = static_cast<float>(1 << (o + 1));   // 2..64 frames
        const float a = period;                                   // amp ~ 1/f
        sx += a * weaveSmooth1D(t / period, static_cast<uint32_t>(2 * o));
        sy += a * weaveSmooth1D(t / period, static_cast<uint32_t>(2 * o + 1));
        norm += a;
    }
    dx = amp * sx / norm;
    dy = amp * 1.4f * sy / norm;
}

// Catmull-Rom weights: interpolating (w sums to 1, reproduces linears
// exactly); the slight over/undershoot at hard edges is the sharpness.
static inline void weaveCRw(float t, float* w)
{
    const float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] =  1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] =  0.5f * t3 - 0.5f * t2;
}

// One output pixel of the displaced picture: 4x4 Catmull-Rom at (sx, sy) in
// the source, clamp-to-edge, all FOUR channels together (see header).
static inline void weaveSamplePixel(const float* img, int W, int H,
                                    float sx, float sy, float* out4)
{
    const float fx = std::floor(sx), fy = std::floor(sy);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    float wx[4], wy[4];
    weaveCRw(sx - fx, wx);
    weaveCRw(sy - fy, wy);
    for (int c = 0; c < 4; ++c) out4[c] = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const int yy = y0 - 1 + j;
        const int yc = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
        for (int i = 0; i < 4; ++i) {
            const int xx = x0 - 1 + i;
            const int xc = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
            const float w = wx[i] * wy[j];
            const size_t o = (static_cast<size_t>(yc) * W + xc) * 4;
            out4[0] += w * img[o + 0];
            out4[1] += w * img[o + 1];
            out4[2] += w * img[o + 2];
            out4[3] += w * img[o + 3];
        }
    }
    // Alpha is the matte and its domain is [0,1]; Catmull-Rom's negative
    // lobes invent values outside it at hard matte edges (measured: -0.079
    // to +1.153). RGB keeps the overshoot — that is the sharpness — but a
    // confidence no input pixel ever had is invented, and "alpha passes
    // through" means un-invented.
    out4[3] = out4[3] < 0.0f ? 0.0f : (out4[3] > 1.0f ? 1.0f : out4[3]);
}

// ---------------------------------------------------------------------------
// Live H&D curve scope (deterministic — a pure function of the params, so it is
// parity-trivial). The curves are drawn by evaluating the SAME production
// hdCurve()/chainDensity() the pixels use, so the plot can never disagree with
// the filter (the "measured-transfer renderer" discipline). Panel anchors in
// DISPLAY space (yd = H-1-y) exactly like Hush's scopes.
//
// Slice 1 draws the applied per-channel system curve (input stops -> output
// stops, i.e. the whole negative->printer->print tone scale) on a stops grid,
// with an 18% gray crosshair and R/G/B legend swatches. Text labels and the
// live exposure histogram land in the next increment.
// ---------------------------------------------------------------------------
// The LOOK in working-space linear: decode, halation re-exposure, tone spine,
// subtractive color. Shared by the pixel path and the density scope's
// measurement pass, so the scope measures exactly what the pixels became — it
// cannot drift from them.
//
// `scat*` are the energy-normalized, blurred, PER-CHANNEL scene-linear highlight
// excesses at this pixel (0 when halation is off). They are deliberately NOT
// defaulted: the density scope calls this function, and a scope that silently
// passed 0 would measure an image the pixels never became — an L3-class bug that
// parity CANNOT catch, because all four backends would agree on the same wrong
// parade. Forcing every call site to pass scatter is the gate. The failure would
// even be directional: halation raises E, so toneChannel(E) rises and density
// falls, and a scatter-blind parade would draw the halo DARKER than it is —
// exactly where the user opened the scope to look.
// `vgain` is this pixel's vignette gain (1 when vignette is off) — the
// taking-lens attenuation multiplies scene light AHEAD of the whole spine,
// so it is an argument for the same reason scatter is: the density scope
// calls this function too, and a vignette-blind parade would draw corner
// columns brighter than the pixels (the same L3 class as a scatter-blind
// scope). The H&D curve scope stays position-free by stated claim: it plots
// the CENTRE curve (a transfer curve has no single field angle).
static inline void lookLinear(float r, float g, float b,
                              float scatR, float scatG, float scatB,
                              float vgain,
                              const SpeakParams& pr,
                              float& oR, float& oG, float& oB)
{
    const SpeakProfile& p = pr.profile;
    const int cs = pr.inputColorSpace;
    const float lr = vgain * decodeToLinear(cs, r);
    const float lg = vgain * decodeToLinear(cs, g);
    const float lb = vgain * decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        const float s = clampf(pr.strength, 0.0f, 1.0f);
        const float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light adds to the scene light ENTERING the
        // negative, so it goes in the CURVE'S ARGUMENT. The dry side of the mix
        // stays lr, not the re-exposed value — otherwise strength 0 would leave
        // the raw scatter added to linear with no curve downstream, which is
        // precisely the end-chain overlay the arm rejected (and would break the
        // identity gate G3). Branch rather than multiply by zero so amount 0 is
        // provably bit-exact with the pre-halation build; `a` is frame-uniform,
        // so the branch costs no GPU divergence.
        float er = lr, eg = lg, eb = lb;
        if (a > 0.0f) {
            er = lr + a * kHalWeight[0] * scatR;
            eg = lg + a * kHalWeight[1] * scatG;
            eb = lb + a * kHalWeight[2] * scatB;
        }
        mr = lerpf(lr, toneChannel(er, 0, p), s);
        mg = lerpf(lg, toneChannel(eg, 1, p), s);
        mb = lerpf(lb, toneChannel(eb, 2, p), s);
    }
    if ((pr.enableDye != 0) && dyeActive(p)) subtractiveColor(mr, mg, mb, p, mr, mg, mb);
    if ((pr.enableSplit != 0) && splitActive(p)) splitTone(mr, mg, mb, p, mr, mg, mb);
    oR = mr; oG = mg; oB = mb;
}

// ---------------------------------------------------------------------------
// Scope statistics — the frame's own exposure distribution, and the Status-M
// density waveform of the RESULT, measured on a stride-2 grid and binned with
// integer counts (order-independent, so all four backends land on identical
// bins).
// ---------------------------------------------------------------------------
static inline int wfColOf(int x, int W)
{
    const int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
static inline int wfRowOf(float D)
{
    const int r = static_cast<int>(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
static inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}
static inline int expBinOf(float stops)
{
    const int b = static_cast<int>((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
    return b < 0 ? 0 : (b >= SPEAK_EXP_BINS ? SPEAK_EXP_BINS - 1 : b);
}
// A pixel's scene exposure in stops (mean of the linear channels — color-space
// agnostic, so the histogram means the same thing in any declared input space).
static inline float pixelStops(int cs, float r, float g, float b)
{
    const float m = (decodeToLinear(cs, r) + decodeToLinear(cs, g) + decodeToLinear(cs, b)) * (1.0f / 3.0f);
    return std::log2((m < kLinTiny ? kLinTiny : m) / k18Gray);
}
// `scat` is the W*H*3 interleaved halation scatter field (null when halation
// is off) and `bscat` the bloom scatter field (null when bloom is off). The
// density parade MUST see BOTH — see the lookLinear header for why a
// scatter-blind scope is an L3 bug parity cannot catch; a bloom-blind parade
// would draw highlight columns taller and shadow floors deeper than the
// pixels the user is looking at.
inline void computeStats(const float* src, const float* scat, const float* bscat,
                         int W, int H, const SpeakParams& pr, uint32_t* stats)
{
    for (int i = 0; i < SPEAK_STATS_UINTS; ++i) stats[i] = 0u;
    if (pr.scopeHD == 0 && pr.scopeDensity == 0) return;   // only measured when shown
    const int cs = pr.inputColorSpace;
    for (int y = 0; y < H; y += 2)
        for (int x = 0; x < W; x += 2) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t j = (static_cast<size_t>(y) * W + x) * 3;
            if (pr.scopeHD != 0)
                stats[SPEAK_STATS_HIST_EXP + expBinOf(pixelStops(cs, src[i], src[i + 1], src[i + 2]))]++;
            if (pr.scopeDensity != 0) {
                float mr, mg, mb;
                lookLinear(src[i], src[i + 1], src[i + 2],
                           scat ? scat[j + 0] : 0.0f, scat ? scat[j + 1] : 0.0f,
                           scat ? scat[j + 2] : 0.0f,
                           vignGain(x, y, W, H, pr), pr, mr, mg, mb);
                // Grain and bloom are part of the result, so the parade must
                // measure them — the same L3 rule the halation scatter
                // established (G17). The matte value comes from the SAME input
                // alpha the pixels use, the bloom sample from the SAME field.
                applyGrain(mr, mg, mb, src[i + 3], x, y, H, pr);
                if (bscat)
                    bloomApplyPixel(mr, mg, mb,
                                    bscat[j + 0], bscat[j + 1], bscat[j + 2], pr);
                const int col = wfColOf(x, W);
                stats[wfIdx(0, col, wfRowOf(density10(mr)))]++;
                stats[wfIdx(1, col, wfRowOf(density10(mg)))]++;
                stats[wfIdx(2, col, wfRowOf(density10(mb)))]++;
            }
        }
    uint32_t mx = 0u;
    for (int b = 0; b < SPEAK_EXP_BINS; ++b)
        if (stats[SPEAK_STATS_HIST_EXP + b] > mx) mx = stats[SPEAK_STATS_HIST_EXP + b];
    stats[SPEAK_STATS_HIST_MAX] = mx;
    uint32_t wmx = 0u;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (stats[SPEAK_STATS_WF + k] > wmx) wmx = stats[SPEAK_STATS_WF + k];
    stats[SPEAK_STATS_WF_MAX] = wmx;
}

// The APPLIED TONE SCALE for one input exposure, in stops — the Strength mix and
// the enable toggle included, so the plot cannot drift from the curve the pixels
// use (at strength 0 it collapses to the y=x diagonal, matching the identity
// pass-through). Output encode is a display transform applied equally to curve
// and diagonal, so it cancels in stops.
//
// WHAT THIS PLOT DOES NOT SHOW, deliberately: halation. This is a function of
// EXPOSURE ALONE, and halation is SPATIAL — a pixel's re-exposure depends on its
// neighbours, so there is no single scatter value for an exposure and no honest
// way to draw it as one curve (threading a scatter value in here would turn one
// curve into a per-pixel family, and the per-channel AH weight into three). The
// curve remains exactly true of the tone scale; it is not the whole transform
// once Light is on. The two scopes that DO see halation are the Status-M density
// parade (which measures the halated result — G17) and the isolated-scatter
// view. Do not restore an "it can never disagree with the pixels" claim here.
static inline float scopeYStops(float inStops, int ch, const SpeakParams& pr)
{
    const float lin = k18Gray * std::exp2(inStops);
    float outLin = lin;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        const float s = clampf(pr.strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, pr.profile), s);
    }
    return std::log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

// Returns true and writes an (r,g,b) display-space color if (x,y) is a scope
// pixel. `out*` are only touched when it returns true.
static inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                const uint32_t* stats,
                                float& outR, float& outG, float& outB)
{
    if (pr.scopeHD == 0) return false;

    const int sc = (H / 540) > 1 ? (H / 540) : 1;      // panel scale (as Hush)
    const int panelW = 220 * sc, panelH = 150 * sc;
    const int margin = 12 * sc;
    const int px0 = margin, py0 = margin;              // top-left in DISPLAY space
    const int yd = H - 1 - y;                          // display row
    const int lx = x - px0, ly = yd - py0;             // panel-local coords
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    // Plot area inside a small inset.
    const int pad = 6 * sc;
    const int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    const int gx = lx - pad, gy = ly - pad;            // plot-local (gy down)

    // Opaque dark panel with a 1px border.
    outR = outG = outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = outG = outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    // gy runs DOWN the plot; convert to an output-stops value at this row.
    const float rowStops = 6.0f - 12.0f * (static_cast<float>(gy) / (plotH - 1));

    // Grid every 2 stops + the 18% gray crosshair (input & output = 0 stops).
    const int gcol0 = static_cast<int>((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    const int grow0 = static_cast<int>((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { outR = outG = outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { outR = outG = outB = 0.13f; }

    // The frame's own exposure histogram, projected onto the logE axis under
    // the curves — so you can see where THIS shot's tones sit on the curve.
    const uint32_t hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        const int hb = expBinOf(-6.0f + 12.0f * (static_cast<float>(gx) / (plotW - 1)));
        const float f = static_cast<float>(stats[SPEAK_STATS_HIST_EXP + hb]) / static_cast<float>(hmax);
        const int barH = static_cast<int>(std::sqrt(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { outR = 0.16f; outG = 0.19f; outB = 0.24f; }
    }

    // The three applied per-channel curves. A column is "on" a curve when the
    // row's output-stops straddles the curve value between this and the next
    // column (so the trace stays connected on steep segments).
    const int chR[3] = { 1, 0, 0 }, chG[3] = { 0, 1, 0 }, chB[3] = { 0, 0, 1 };
    for (int ch = 0; ch < 3; ++ch) {
        const float inS  = -6.0f + 12.0f * (static_cast<float>(gx)     / (plotW - 1));
        const float inS2 = -6.0f + 12.0f * (static_cast<float>(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { const float t = y0; y0 = y1; y1 = t; }
        const float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            outR = 0.10f + 0.85f * chR[ch];
            outG = 0.10f + 0.85f * chG[ch];
            outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }

    // Legend swatches, bottom-left of the plot.
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        const int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { outR = 0.95f; outG = 0.10f; outB = 0.10f; return true; }
            if (sw == 1) { outR = 0.10f; outG = 0.95f; outB = 0.10f; return true; }
            if (sw == 2) { outR = 0.10f; outG = 0.10f; outB = 0.95f; return true; }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Status-M density scope: an RGB parade of the RESULT's film density, measured
// from the frame by the same look the pixels get. Vertical axis is optical
// density (0 = paper white at the top, SPEAK_WF_DMAX at the bottom) with
// markers at paper white and 18% gray (D = -log10(0.18) = 0.745). Anchored
// top-RIGHT in display space, so it sits beside the H&D scope.
// ---------------------------------------------------------------------------
static inline bool densityScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                     const uint32_t* stats,
                                     float& outR, float& outG, float& outB)
{
    if (pr.scopeDensity == 0) return false;

    const int sc = (H / 540) > 1 ? (H / 540) : 1;
    const int panelW = 220 * sc, panelH = 150 * sc;
    const int margin = 12 * sc;
    const int px0 = W - margin - panelW, py0 = margin;   // top-right, display space
    const int yd = H - 1 - y;
    const int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    const int pad = 6 * sc;
    const int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    const int gx = lx - pad, gy = ly - pad;

    outR = outG = outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = outG = outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    // Density markers: paper white (D=0, the top) and 18% gray.
    const int rowGray = static_cast<int>(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { outR = 0.35f; outG = 0.35f; outB = 0.35f; return true; }
    if (gy == rowGray) { outR = 0.30f; outG = 0.26f; outB = 0.12f; return true; }

    // The parade: three channel panes side by side.
    const int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { outR = outG = outB = 0.16f; return true; }  // pane divider

    const uint32_t wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        const int within = gx - ch * chW;
        const int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        const int wrow = gy * SPEAK_WF_ROWS / plotH;
        const uint32_t c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                                       wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            const float inten = std::sqrt(static_cast<float>(c) / static_cast<float>(wmax));
            const float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            outR = (ch == 0) ? v : 0.05f;
            outG = (ch == 1) ? v : 0.05f;
            outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ON-IMAGE TEXT (v1.0, SPEC-1.0 §2) — the status strip and the Setup Guide.
//
// The kernels RASTERIZE, they never compose: the strip's text is written by
// the plugin into SpeakParams.statusText (packed 4 chars/int), so message
// logic lives exactly once, in C++. The guide card is fixed documentation —
// a character grid burned into each backend at build time, so the recipe
// ships inside the plugin and cannot drift from it.
//
// 5x7 glyphs on a 6x9 cell, bit = row*5+col (row 0 top, col 0 left). ASCII
// 32..126 at [0..94]; specials at [95..106] for text codes 1..8, 11, 12, 14,
// 15 = check, middot, emdash, arrow, hline, vline, TL, TR, teeDown, teeUp,
// BL, BR. Line-drawing glyphs (index >= 99) replicate their edge row/col
// into the cell's spacing gap so wires CONNECT; text keeps its kerning.
// Tables generated by the scratch gen_font.py/gen_guide.py masters.
// ---------------------------------------------------------------------------
#define SPEAK_FONT_GLYPHS 107
#define SPEAK_GUIDE_ROWS 27
#define SPEAK_GUIDE_COLS 66
static const uint64_t kSpeakFontBits[SPEAK_FONT_GLYPHS] = {
    0x000000000ull, 0x100421084ull, 0x00000294aull, 0x295f57d4aull,
    0x11f0707c4ull, 0x632222263ull, 0x593511526ull, 0x000001084ull,
    0x208210888ull, 0x088842082ull, 0x009575480ull, 0x0084f9080ull,
    0x088c00000ull, 0x0000f8000ull, 0x318000000ull, 0x002222200ull,
    0x3a33ae62eull, 0x3884210c4ull, 0x7c444422eull, 0x3a304111full,
    0x211f4a988ull, 0x3a3083c3full, 0x3a317844cull, 0x08422221full,
    0x3a317462eull, 0x1910f462eull, 0x018c03180ull, 0x088c03180ull,
    0x208208888ull, 0x001f07c00ull, 0x088882082ull, 0x10044422eull,
    0x3ab5b422eull, 0x4631fc62eull, 0x3e317c62full, 0x3a210862eull,
    0x1d318c527ull, 0x7c217843full, 0x04217843full, 0x7a31e862eull,
    0x4631fc631ull, 0x38842108eull, 0x19284211cull, 0x452519531ull,
    0x7c2108421ull, 0x4631ad771ull, 0x4631cd671ull, 0x3a318c62eull,
    0x04217c62full, 0x59358c62eull, 0x45257c62full, 0x3e107043eull,
    0x10842109full, 0x3a318c631ull, 0x11518c631ull, 0x4775ac631ull,
    0x462a22a31ull, 0x108422a31ull, 0x7c222221full, 0x38421084eull,
    0x020820820ull, 0x39084210eull, 0x000004544ull, 0x7c0000000ull,
    0x000000082ull, 0x7a3e83800ull, 0x3e318bc21ull, 0x3a210b800ull,
    0x7a318fa10ull, 0x383f8b800ull, 0x084238a4cull, 0x3a1e8c7c0ull,
    0x46318bc21ull, 0x388421804ull, 0x192843008ull, 0x24a32a421ull,
    0x388421086ull, 0x56b5aac00ull, 0x46318bc00ull, 0x3a318b800ull,
    0x042f8c5e0ull, 0x421e8c7c0ull, 0x04219b400ull, 0x3e0e0f800ull,
    0x324211c42ull, 0x5b318c400ull, 0x11518c400ull, 0x2ab58c400ull,
    0x454454400ull, 0x3a1e8c400ull, 0x7c4447c00ull, 0x608411098ull,
    0x108421084ull, 0x0c8441083ull, 0x0008a8800ull, 0x10c944200ull,
    0x000c60000ull, 0x0000f8000ull, 0x0088fa080ull, 0x0000f8000ull,
    0x108421084ull, 0x1084e0000ull, 0x108438000ull, 0x1084f8000ull,
    0x0000f9084ull, 0x0000e1084ull, 0x000039084ull,
};
// Per-row style of the guide card: 0 body, 1 title, 2 dim, 3 the blue
// wire, 4 numbered step.
static const int kSpeakGuideStyle[SPEAK_GUIDE_ROWS] = { 1, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 2, 2, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2 };
static const unsigned char kSpeakGuideText[SPEAK_GUIDE_ROWS * SPEAK_GUIDE_COLS] = {
    83,80,69,65,75,32,2,32,83,69,84,85,80,32,71,85,73,68,69,32,3,32,
    116,104,101,32,72,117,115,104,4,83,112,101,97,107,32,104,97,110,100,111,102,102,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    72,117,115,104,32,113,117,105,101,116,115,32,116,104,101,32,110,111,105,115,101,32,
    97,116,32,116,104,101,32,70,73,82,83,84,32,110,111,100,101,59,32,83,112,101,
    97,107,32,103,105,118,101,115,32,116,104,101,32,105,109,97,103,101,32,32,32,32,
    105,116,115,32,118,111,105,99,101,32,97,116,32,116,104,101,32,76,65,83,84,46,
    32,67,108,101,97,110,32,101,97,114,108,121,44,32,114,101,99,111,110,115,116,114,
    117,99,116,32,108,97,116,101,46,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    49,32,2,32,79,110,32,72,117,115,104,58,32,116,117,114,110,32,79,78,32,34,
    69,120,112,111,114,116,32,67,108,101,97,110,32,77,97,116,116,101,32,116,111,32,
    65,108,112,104,97,34,44,32,116,104,101,110,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,114,105,103,104,116,45,99,108,105,99,107,32,116,104,101,32,72,117,
    115,104,32,110,111,100,101,32,4,32,79,70,88,32,65,108,112,104,97,32,4,32,
    69,110,97,98,108,101,46,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    50,32,2,32,68,114,97,103,32,116,104,101,32,66,76,85,69,32,107,101,121,32,
    119,105,114,101,32,102,114,111,109,32,72,117,115,104,32,105,110,116,111,32,83,112,
    101,97,107,39,115,32,107,101,121,32,105,110,112,117,116,46,32,32,32,32,32,32,
    51,32,2,32,79,110,32,83,112,101,97,107,58,32,77,97,116,116,101,32,83,111,
    117,114,99,101,32,4,32,34,75,101,121,32,105,110,112,117,116,34,44,32,69,110,
    97,98,108,101,32,71,114,97,105,110,32,79,78,46,32,32,32,32,32,32,32,32,
    32,32,32,32,70,117,108,108,32,103,114,97,105,110,32,119,104,101,114,101,32,72,
    117,115,104,32,99,108,101,97,110,101,100,32,100,101,101,112,32,40,116,101,120,116,
    117,114,101,32,119,97,115,32,102,108,97,116,116,101,110,101,100,41,59,32,32,32,
    32,32,32,32,77,97,116,116,101,32,70,108,111,111,114,32,119,104,101,114,101,32,
    109,111,116,105,111,110,32,119,97,115,32,112,114,111,116,101,99,116,101,100,32,40,
    114,101,97,108,32,110,111,105,115,101,32,115,117,114,118,105,118,101,100,41,46,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    7,5,5,5,5,5,5,8,32,32,32,32,32,32,32,32,32,32,7,5,5,5,
    5,5,5,5,5,5,5,5,5,8,32,32,32,32,32,32,32,32,32,32,7,5,
    5,5,5,5,5,5,8,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    6,32,72,117,115,104,32,6,5,5,5,5,5,5,5,5,5,5,6,32,121,111,
    117,114,32,103,114,97,100,101,32,6,5,5,5,5,5,5,5,5,5,5,6,32,
    83,112,101,97,107,32,6,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    14,5,5,11,5,5,5,15,32,32,32,32,32,32,32,32,32,32,14,5,5,5,
    5,5,5,5,5,5,5,5,5,15,32,32,32,32,32,32,32,32,32,32,14,5,
    5,11,5,5,5,5,15,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,14,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,32,75,69,
    89,32,40,98,108,117,101,32,119,105,114,101,41,32,5,5,5,5,5,5,5,5,
    5,15,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    84,104,101,32,116,104,114,101,101,32,115,101,116,117,112,115,58,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,103,114,97,105,110,32,115,104,97,112,101,100,32,98,121,32,116,104,101,32,
    109,97,116,116,101,32,32,32,32,77,97,116,116,101,32,83,111,117,114,99,101,58,
    32,75,101,121,32,105,110,112,117,116,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,117,110,105,102,111,114,109,32,103,114,97,105,110,44,32,110,111,32,119,105,
    114,101,115,32,32,32,32,32,32,77,97,116,116,101,32,83,111,117,114,99,101,58,
    32,79,102,102,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,110,111,32,103,114,97,105,110,44,32,105,109,97,103,101,32,117,110,116,111,
    117,99,104,101,100,32,32,32,32,69,110,97,98,108,101,32,71,114,97,105,110,32,
    111,102,102,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    84,104,101,32,111,110,101,32,114,117,108,101,58,32,108,101,97,118,101,32,72,117,
    115,104,39,115,32,101,120,112,111,114,116,32,79,70,70,32,117,110,108,101,115,115,
    32,83,112,101,97,107,32,99,111,110,115,117,109,101,115,32,105,116,46,32,32,32,
    65,110,121,32,97,99,116,105,118,101,32,109,97,116,116,101,32,115,111,117,114,99,
    101,32,105,115,32,67,79,78,83,85,77,69,68,32,104,101,114,101,58,32,83,112,
    101,97,107,39,115,32,111,117,116,112,117,116,32,97,108,112,104,97,32,32,32,32,
    103,111,101,115,32,111,112,97,113,117,101,44,32,115,111,32,116,104,101,32,109,97,
    116,116,101,32,110,101,118,101,114,32,114,105,100,101,115,32,112,97,115,116,32,116,
    104,105,115,32,110,111,100,101,46,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    84,104,105,115,32,99,97,114,100,32,105,115,32,100,114,97,119,110,32,98,121,32,
    116,104,101,32,114,101,110,100,101,114,32,107,101,114,110,101,108,32,105,116,115,101,
    108,102,44,32,115,111,32,105,116,32,99,97,110,110,111,116,32,32,32,32,32,32,
    100,114,105,102,116,32,102,114,111,109,32,116,104,101,32,112,108,117,103,105,110,32,
    105,116,32,115,104,105,112,115,32,119,105,116,104,46,32,86,105,101,119,32,4,32,
    82,101,115,117,108,116,32,116,111,32,103,114,97,100,101,46,32,32,32,32,32,32,
};

static inline int fontIndexOf(int c)
{
    if (c >= 32 && c < 127) return c - 32;
    if (c >= 1 && c <= 8)   return 94 + c;   // check middot emdash arrow hline vline TL TR
    if (c == 11 || c == 12) return 92 + c;   // teeDown teeUp
    if (c == 14 || c == 15) return 91 + c;   // BL BR
    return 0;                                // anything else draws as space
}

// One glyph-cell sample: is (cellRow 0..8, cellCol 0..5) of char c lit?
static inline bool fontBit(int c, int cellRow, int cellCol)
{
    const int gi = fontIndexOf(c);
    int r = cellRow, cc = cellCol;
    if (gi >= 99) {                          // line-drawing: bridge the gaps
        if (cc == 5) cc = 4;
        if (r >= 7) r = 6;
    }
    if (r > 6 || cc > 4 || r < 0 || cc < 0) return false;
    return ((kSpeakFontBits[gi] >> (r * 5 + cc)) & 1) != 0;
}

static inline int statusCharAt(const SpeakParams& pr, int k)
{
    return (pr.statusText[k >> 2] >> ((k & 3) * 8)) & 0xFF;
}
static inline int statusLen(const SpeakParams& pr)
{
    for (int k = 0; k < 112; ++k) if (statusCharAt(pr, k) == 0) return k;
    return 112;
}

// The one-line status strip (SPEC-1.0 §2): bottom-left, panel chrome like
// the scopes — same scale rule, same palette, pinned through the weave by
// the same overlay pass. Returns true when (x,y) belongs to the strip.
static inline bool statusStripPixel(int x, int y, int W, int H, const SpeakParams& pr,
                             float& outR, float& outG, float& outB)
{
    if (pr.statusStrip == 0) return false;
    const int n = statusLen(pr);
    if (n <= 0) return false;
    const int sc = (H / 540) > 1 ? (H / 540) : 1;
    const int pad = 4 * sc, margin = 12 * sc;
    const int stripH = 7 * sc + 2 * pad;
    int stripW = n * 6 * sc + 2 * pad;
    if (stripW > W - 2 * margin) stripW = W - 2 * margin;   // clip, never wrap
    const int yd = H - 1 - y;                               // display row
    const int lx = x - margin, ly = yd - (H - margin - stripH);
    if (lx < 0 || ly < 0 || lx >= stripW || ly >= stripH) return false;
    outR = outG = outB = 0.06f;
    if (lx < sc || ly < sc || lx >= stripW - sc || ly >= stripH - sc) {
        outR = outG = outB = 0.30f; return true;
    }
    const int tx = lx - pad, ty = ly - pad;
    if (tx < 0 || ty < 0) return true;
    const int k = tx / (6 * sc);
    if (k >= n) return true;
    const int c = statusCharAt(pr, k);
    if (fontBit(c, ty / sc, (tx - k * 6 * sc) / sc)) {
        if (c == 1) { outR = 0.25f; outG = 0.90f; outB = 0.40f; }  // the check earns green
        else        { outR = outG = outB = 0.82f; }
    }
    return true;
}

// The Setup Guide view (SPEC-1.0 §2): the Hush->Speak recipe rendered by the
// render kernel itself. Pure function of (x,y,W,H) — trivially parity-safe.
static inline void guidePixel(int x, int y, int W, int H,
                     float& outR, float& outG, float& outB)
{
    const int yd = H - 1 - y;
    const int gw = SPEAK_GUIDE_COLS * 6, gh = SPEAK_GUIDE_ROWS * 9;
    int gsc = (W * 92 / 100) / gw;                 // integer scale, ~92% fit
    const int vs = (H * 92 / 100) / gh;
    if (vs < gsc) gsc = vs;
    if (gsc < 1) gsc = 1;
    const int x0 = (W - gw * gsc) / 2, y0 = (H - gh * gsc) / 2;
    outR = outG = outB = 0.045f;                   // the card
    const int lx = x - x0, ly = yd - y0;
    if (lx < 0 || ly < 0 || lx >= gw * gsc || ly >= gh * gsc) return;
    const int col = lx / (6 * gsc), row = ly / (9 * gsc);
    if (col >= SPEAK_GUIDE_COLS || row >= SPEAK_GUIDE_ROWS) return;
    const int c = (int)kSpeakGuideText[row * SPEAK_GUIDE_COLS + col];
    if (!fontBit(c, (ly - row * 9 * gsc) / gsc, (lx - col * 6 * gsc) / gsc)) return;
    const int st = kSpeakGuideStyle[row];
    if      (st == 1) { outR = outG = outB = 0.95f; }                 // title
    else if (st == 2) { outR = outG = outB = 0.55f; }                 // dim
    else if (st == 3) { outR = 0.30f; outG = 0.55f; outB = 0.98f; }   // the blue wire
    else if (st == 4) { outR = 0.95f; outG = 0.82f; outB = 0.55f; }   // steps
    else              { outR = outG = outB = 0.78f; }                 // body
}

// Host-side helper: pack a composed status line into params (the kernels are
// pure consumers). ASCII plus the font's special codes; truncates at 111.
static inline void packStatusText(SpeakParams& pr, const char* s)
{
    for (int i = 0; i < 28; ++i) pr.statusText[i] = 0;
    for (int k = 0; k < 111 && s[k] != 0; ++k) {
        const uint32_t b = static_cast<uint32_t>(static_cast<unsigned char>(s[k]));
        pr.statusText[k >> 2] = static_cast<int>(
            static_cast<uint32_t>(pr.statusText[k >> 2]) | (b << ((k & 3) * 8)));
    }
}

// The INPUT pixel delivered through the same output CST as the result (no look
// applied). Split/Input views use this so both halves of a comparison are in
// the SAME space: in Bake mode both are Rec.709 (a valid look A/B), in Working
// mode both are the untouched working space (bit-identical to the raw input).
static inline void deliverInput(const SpeakParams& pr, float r, float g, float b,
                                float& oR, float& oG, float& oB)
{
    if (pr.outputMode == SPEAK_OUT_BAKE_REC709) {
        const int cs = pr.inputColorSpace;
        const float lr = decodeToLinear(cs, r);
        const float lg = decodeToLinear(cs, g);
        const float lb = decodeToLinear(cs, b);
        float rr, rg, rb;
        gamutToRec709Lin(cs, lr, lg, lb, rr, rg, rb);
        rr = rr < 0.0f ? 0.0f : rr;
        rg = rg < 0.0f ? 0.0f : rg;
        rb = rb < 0.0f ? 0.0f : rb;
        oR = encodeFromLinear(SPEAK_CS_REC709_G24, rr);
        oG = encodeFromLinear(SPEAK_CS_REC709_G24, rg);
        oB = encodeFromLinear(SPEAK_CS_REC709_G24, rb);
    } else {
        oR = r; oG = g; oB = b;
    }
}

// ---------------------------------------------------------------------------
// The whole per-pixel operation. `x,y` are buffer coords (y up, OFX-native);
// the scope converts to display space internally. RGBA in -> RGBA out (alpha
// passes through). This is the ONE function the four backends must agree on.
// ---------------------------------------------------------------------------
static inline void processPixel(float r, float g, float b, float srcA,
                                float scatR, float scatG, float scatB,
                                float bloomR, float bloomG, float bloomB,
                                int x, int y, int W, int H,
                                const SpeakParams& pr, const uint32_t* stats,
                                int drawScopes,
                                float& outR, float& outG, float& outB)
{
    const SpeakProfile& p = pr.profile;
    const int cs = pr.inputColorSpace;
    const bool toneOn  = (pr.enableTone != 0) && (pr.strength > 0.0f);
    const bool dyeOn   = (pr.enableDye != 0) && dyeActive(p);
    const bool splitOn = (pr.enableSplit != 0) && splitActive(p);
    const bool grainOn = grainActive(pr);
    const bool bloomOn = bloomActive(pr);
    const bool vignOn  = vignActive(pr);
    const bool bake    = (pr.outputMode == SPEAK_OUT_BAKE_REC709);

    if (!toneOn && !dyeOn && !splitOn && !grainOn && !bloomOn && !vignOn && !bake) {
        // Working space + no look: bit-exact pass-through (identity). Scopes
        // may still overwrite below. Halation cannot reach here: halActive()
        // requires toneOn, so this branch already excludes it.
        outR = r; outG = g; outB = b;
    } else {
        // The look in working-space linear (halation + tone spine + subtractive
        // color + grain + bloom) — the SAME functions the density scope measures.
        float mr, mg, mb;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, mr, mg, mb);
        applyGrain(mr, mg, mb, srcA, x, y, H, pr);
        bloomApplyPixel(mr, mg, mb, bloomR, bloomG, bloomB, pr);
        if (bake) {
            // Output CST: gamut-convert to Rec.709 and encode gamma 2.4. Applies
            // regardless of the look (it is delivery, not a look) — a hard gamut
            // clip for now (the constant-hue soft guardrail is a later module).
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, rr, rg, rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, rr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, rg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, rb);
        } else {
            // Working space: re-encode to the input space and let RCM deliver.
            outR = encodeFromLinear(cs, mr);
            outG = encodeFromLinear(cs, mg);
            outB = encodeFromLinear(cs, mb);
        }
    }

    // View modes (Split / Input): show the input delivered through the same
    // output CST, so a Split isolates the LOOK, not the color space.
    if (pr.viewMode == SPEAK_VIEW_INPUT ||
        (pr.viewMode == SPEAK_VIEW_SPLIT && x < W / 2))
        deliverInput(pr, r, g, b, outR, outG, outB);

    // Isolated-scatter view (spec 1B.5: "an isolated-scatter scope"; 1B "show
    // halation-only"). It shows the ACTUAL injected re-exposure, a * w_c * S_c,
    // delivered through the SAME output transform as the picture — so it is in
    // the same units as the image and cannot overstate itself. Deliberately NOT
    // auto-normalized or gained up: a per-frame normalisation would make a
    // negligible halo look enormous, which is the exact dishonesty this view
    // exists to prevent. If it reads dark, that IS the measurement — halation is
    // a small addition everywhere except beside a real highlight.
    if (pr.viewMode == SPEAK_VIEW_SCATTER) {
        const float a = halAmountOf(pr);
        const float on = ((pr.enableTone != 0) && (pr.strength > 0.0f)) ? 1.0f : 0.0f;
        float sr = on * a * kHalWeight[0] * scatR;
        float sg = on * a * kHalWeight[1] * scatG;
        float sb = on * a * kHalWeight[2] * scatB;
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, sr, sg, sb, rr, rg, rb);
            sr = rr < 0.0f ? 0.0f : rr;
            sg = rg < 0.0f ? 0.0f : rg;
            sb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, sr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, sg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, sb);
        } else {
            outR = encodeFromLinear(cs, sr);
            outG = encodeFromLinear(cs, sg);
            outB = encodeFromLinear(cs, sb);
        }
    }

    // Isolated-grain view (spec 1B: "show grain-only" — the transparency ethic).
    // 18% gray plus the EXACT grain increment this pixel received: the look is
    // run with and without applyGrain and the linear difference sits on gray,
    // delivered through the same output transform as the picture. Same honesty
    // stance as the scatter view — never auto-gained, so subtle grain correctly
    // looks subtle, and the matte's placement (matteSource) is directly visible.
    if (pr.viewMode == SPEAK_VIEW_GRAIN) {
        float pr0, pg0, pb0, pr1, pg1, pb1;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, pr0, pg0, pb0);
        pr1 = pr0; pg1 = pg0; pb1 = pb0;
        applyGrain(pr1, pg1, pb1, srcA, x, y, H, pr);
        float gr = k18Gray + (pr1 - pr0);
        float gg = k18Gray + (pg1 - pg0);
        float gb = k18Gray + (pb1 - pb0);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, gr, gg, gb, rr, rg, rb);
            gr = rr < 0.0f ? 0.0f : rr;
            gg = rg < 0.0f ? 0.0f : rg;
            gb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, gr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, gg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, gb);
        } else {
            gr = gr < 0.0f ? 0.0f : gr;
            gg = gg < 0.0f ? 0.0f : gg;
            gb = gb < 0.0f ? 0.0f : gb;
            outR = encodeFromLinear(cs, gr);
            outG = encodeFromLinear(cs, gg);
            outB = encodeFromLinear(cs, gb);
        }
    }

    // Isolated-bloom view: gray + the SIGNED bloom delta this pixel received
    // (out - look). Bloom both gives and takes — the halo's lift is positive,
    // the source's borrowed light is NEGATIVE — and showing the signed field
    // on gray is what makes the conservation visible: the darkening at the
    // source is the same light as the glow around it. Same honesty stance as
    // the scatter and grain views: never auto-gained.
    if (pr.viewMode == SPEAK_VIEW_BLOOM) {
        float lr0, lg0, lb0, lr1, lg1, lb1;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, lr0, lg0, lb0);
        applyGrain(lr0, lg0, lb0, srcA, x, y, H, pr);
        lr1 = lr0; lg1 = lg0; lb1 = lb0;
        bloomApplyPixel(lr1, lg1, lb1, bloomR, bloomG, bloomB, pr);
        float gr = k18Gray + (lr1 - lr0);
        float gg = k18Gray + (lg1 - lg0);
        float gb = k18Gray + (lb1 - lb0);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, gr, gg, gb, rr, rg, rb);
            gr = rr < 0.0f ? 0.0f : rr;
            gg = rg < 0.0f ? 0.0f : rg;
            gb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, gr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, gg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, gb);
        } else {
            gr = gr < 0.0f ? 0.0f : gr;
            gg = gg < 0.0f ? 0.0f : gg;
            gb = gb < 0.0f ? 0.0f : gb;
            outR = encodeFromLinear(cs, gr);
            outG = encodeFromLinear(cs, gg);
            outB = encodeFromLinear(cs, gb);
        }
    }

    // The Setup Guide view (v1.0, SPEC-1.0 §2): the recipe card, drawn by the
    // kernel itself. It replaces the picture; the strip and scopes still
    // overlay it below (they are chrome, and the strip's live matte state is
    // exactly what the recipe asks the user to check).
    if (pr.viewMode == SPEAK_VIEW_SETUP)
        guidePixel(x, y, W, H, outR, outG, outB);

    // Scopes and the status strip render last, over any view (each owns its
    // own corner). When the gate-weave pass is live, speakFrame suppresses
    // them here and overlays them AFTER the displacement — panel chrome does
    // not weave.
    if (drawScopes != 0) {
        float sr, sg, sb;
        if (hdScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
        if (densityScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
        if (statusStripPixel(x, y, W, H, pr, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
    }
}

// ---------------------------------------------------------------------------
// Whole-frame CPU entry point (the reference the GPU kernels are ported from).
// Interleaved RGBA float, row-major, y up (OFX-native buffer order).
// ---------------------------------------------------------------------------
// Builds the scatter pyramid for a frame. `arena` must hold
// halArenaPixels(W,H)*3 floats and `scat` W*H*3. Only called when halation is
// live — see speakFrame.
inline void buildHalScatter(const float* src, int W, int H, const SpeakParams& pr,
                            float* arena, float* scat)
{
    const int cs = pr.inputColorSpace;
    const float th = pr.profile.halThresh;
    // Level 0: the per-channel scene-linear highlight excess.
    // THRESHOLD BEFORE DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t), so
    // thresholding a downsampled level would scatter a different field than the
    // one the physics names (and would leak sub-threshold light into the halo).
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t o = (static_cast<size_t>(y) * W + x) * 3;
            // vignette first: light the lens never delivered cannot halate
            const float vg = vignGain(x, y, W, H, pr);
            arena[o + 0] = halExcess(vg * decodeToLinear(cs, src[i + 0]), th);
            arena[o + 1] = halExcess(vg * decodeToLinear(cs, src[i + 1]), th);
            arena[o + 2] = halExcess(vg * decodeToLinear(cs, src[i + 2]), th);
        }
    const int nLev = halLevelCount(W, H);
    for (int L = 1; L < nLev; ++L) {
        int sw, sh, so, dw, dh, doff;
        halLevelInfo(W, H, L - 1, sw, sh, so);
        halLevelInfo(W, H, L,     dw, dh, doff);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                float v[3];
                halDecimatePixel(arena, so, sw, sh, x, y, v);
                const size_t o = (static_cast<size_t>(doff) + static_cast<size_t>(y) * dw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // Coarse-to-fine accumulate, in place, one octave at a time.
    const float sig = halSigmaPx(H, pr);
    for (int L = nLev - 1; L >= 0; --L) {
        int lw, lh, off;
        halLevelInfo(W, H, L, lw, lh, off);
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x) {
                float v[3];
                halAccumPixel(arena, W, H, L, nLev, sig,
                              kHalCoreFall, kHalSkirtFall, x, y, v);
                const size_t o = (static_cast<size_t>(off) + static_cast<size_t>(y) * lw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // Normalize level 0 into the scatter plane: sum(w) = 1 => energy preserved.
    const float inv = 1.0f / halWeightSum(nLev, sig, kHalCoreFall, kHalSkirtFall);
    for (size_t k = 0; k < static_cast<size_t>(W) * H * 3; ++k) scat[k] = arena[k] * inv;
}

// Builds the BLOOM pyramid for a frame, on the LOOK's working-linear output
// (`lookBuf`, W*H*3 — the image the projector sees, grain included). No
// threshold, broader skirt, veil weight on the coarsest level; see the bloom
// header block for why this is its own build rather than a re-read of
// halation's scene-referred excess pyramid. Same arena contract as
// buildHalScatter; only called when bloom is live — see speakFrame.
inline float bloomFinite(float v)
{
    // One EXR hole or upstream divide must stay ONE bad pixel. Without this,
    // the pyramid's mean-preserving decimation carries a NaN into every
    // level, the frame-mean veil adds it to every pixel (0*NaN is still
    // NaN, so even veil 0 doesn't opt out), and bloomApplyPixel's lerp
    // blanks the whole frame — measured, one texel to 24576/24576. Halation
    // contains the same pixel by accident of halExcess's comparison; bloom
    // contains it here, on purpose.
    return (v == v && v <= 3.402823e38f && v >= -3.402823e38f) ? v : 0.0f;
}

inline void buildBloomScatter(const float* lookBuf, int W, int H, const SpeakParams& pr,
                              float* arena, float* scat)
{
    const size_t n0 = static_cast<size_t>(W) * H * 3;
    for (size_t k = 0; k < n0; ++k) arena[k] = bloomFinite(lookBuf[k]);
    const int nLev = halLevelCount(W, H);
    for (int L = 1; L < nLev; ++L) {
        int sw, sh, so, dw, dh, doff;
        halLevelInfo(W, H, L - 1, sw, sh, so);
        halLevelInfo(W, H, L,     dw, dh, doff);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                float v[3];
                halDecimatePixel(arena, so, sw, sh, x, y, v);
                const size_t o = (static_cast<size_t>(doff) + static_cast<size_t>(y) * dw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // The veil's source: the frame mean, i.e. the decimation chain taken to
    // its logical end. Computed over the coarsest level BEFORE the in-place
    // accumulate overwrites it (each level is mean-preserving, so this is the
    // frame mean up to the border-clamp reweighting every level shares).
    const float sig  = bloomSigmaPx(H, pr);
    const float veil = bloomVeilAdd(nLev, sig, pr);
    float meanC[3] = { 0.0f, 0.0f, 0.0f };
    {
        int cw, ch, coff;
        halLevelInfo(W, H, nLev - 1, cw, ch, coff);
        for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x) {
                const size_t o = (static_cast<size_t>(coff) + static_cast<size_t>(y) * cw + x) * 3;
                meanC[0] += arena[o + 0]; meanC[1] += arena[o + 1]; meanC[2] += arena[o + 2];
            }
        const float invN = 1.0f / (static_cast<float>(cw) * static_cast<float>(ch));
        meanC[0] *= invN; meanC[1] *= invN; meanC[2] *= invN;
    }
    for (int L = nLev - 1; L >= 0; --L) {
        int lw, lh, off;
        halLevelInfo(W, H, L, lw, lh, off);
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x) {
                float v[3];
                halAccumPixel(arena, W, H, L, nLev, sig,
                              kHalCoreFall, kBloomSkirtFall, x, y, v);
                const size_t o = (static_cast<size_t>(off) + static_cast<size_t>(y) * lw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // Normalize profile + veil together: S = (mixture + w_veil*mean) / (Sum w + w_veil).
    const float base = halWeightSum(nLev, sig, kHalCoreFall, kBloomSkirtFall);
    const float inv  = 1.0f / (base + veil);
    for (size_t k = 0; k < n0; ++k)
        scat[k] = (arena[k] + veil * meanC[k % 3]) * inv;
}

inline void speakFrame(const float* src, int W, int H, const SpeakParams& pr, float* dst)
{
    // Heap, not stack: SPEAK_STATS_UINTS is 36994 uints = ~148 KB, which is at
    // or past the default stack limit on some hosts' worker threads.
    std::vector<uint32_t> stats(SPEAK_STATS_UINTS, 0u);
    std::vector<float> arena, scat;
    const bool hal = halActive(pr) || (pr.viewMode == SPEAK_VIEW_SCATTER);
    if (hal) {
        arena.assign(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
        scat.assign(static_cast<size_t>(W) * H * 3, 0.0f);
        buildHalScatter(src, W, H, pr, arena.data(), scat.data());
    }
    const float* sc = hal ? scat.data() : nullptr;

    // Bloom scatters the LOOK's output (grain included), so its pyramid needs
    // the look field first. The per-pixel look is recomputed in processPixel
    // rather than read back from this buffer — a pure function of the same
    // inputs, so the two cannot disagree, and the GPU ports keep the same
    // shape (their main pass stays a pure per-pixel kernel).
    std::vector<float> lookBuf, bloomScat;
    const bool blm = bloomActive(pr);
    if (blm) {
        lookBuf.assign(static_cast<size_t>(W) * H * 3, 0.0f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                const size_t j = (static_cast<size_t>(y) * W + x) * 3;
                float lr, lg, lb;
                lookLinear(src[i + 0], src[i + 1], src[i + 2],
                           sc ? sc[j + 0] : 0.0f, sc ? sc[j + 1] : 0.0f,
                           sc ? sc[j + 2] : 0.0f,
                           vignGain(x, y, W, H, pr), pr, lr, lg, lb);
                applyGrain(lr, lg, lb, src[i + 3], x, y, H, pr);
                lookBuf[j + 0] = lr; lookBuf[j + 1] = lg; lookBuf[j + 2] = lb;
            }
        if (arena.empty())
            arena.assign(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
        bloomScat.assign(static_cast<size_t>(W) * H * 3, 0.0f);
        buildBloomScatter(lookBuf.data(), W, H, pr, arena.data(), bloomScat.data());
    }
    const float* bs = blm ? bloomScat.data() : nullptr;

    // The parade measures the PRE-weave picture: a rigid sub-pixel shift
    // does not change which densities exist, only where they sit. Measured
    // bound: at UI-max weave the registration error is ~0.32 of a parade
    // column bucket on 16:9 and can reach ~1.02 buckets on a portrait
    // frame's worst wander — visible never, arguable once. Stated, not
    // hidden.
    computeStats(src, sc, bs, W, H, pr, stats.data());   // measure the frame, then render
    const bool weave = weaveActive(pr);
    const int drawScopes = weave ? 0 : 1;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t j = (static_cast<size_t>(y) * W + x) * 3;
            float oR, oG, oB;
            processPixel(src[i + 0], src[i + 1], src[i + 2], src[i + 3],
                         sc ? sc[j + 0] : 0.0f, sc ? sc[j + 1] : 0.0f, sc ? sc[j + 2] : 0.0f,
                         bs ? bs[j + 0] : 0.0f, bs ? bs[j + 1] : 0.0f, bs ? bs[j + 2] : 0.0f,
                         x, y, W, H, pr, stats.data(), drawScopes, oR, oG, oB);
            dst[i + 0] = oR;
            dst[i + 1] = oG;
            dst[i + 2] = oB;
            dst[i + 3] = pr.maskExternal ? 1.0f : src[i + 3];   // opaque when an external key drives the matte; else passthrough
        }
    }
    if (weave) {
        // The gate displaces the finished picture — grain, bloom and all —
        // as one rigid sub-pixel move; then the scopes are drawn on top,
        // pinned (they are panel chrome, not picture).
        float wdx, wdy;
        weaveDisp(pr, H, wdx, wdy);
        std::vector<float> pre(dst, dst + static_cast<size_t>(W) * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                weaveSamplePixel(pre.data(), W, H,
                                 static_cast<float>(x) - wdx,
                                 static_cast<float>(y) - wdy, &dst[i]);
            }
        if (pr.scopeHD != 0 || pr.scopeDensity != 0 || pr.statusStrip != 0) {
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                    float sr, sg, sb;
                    if (hdScopePixel(x, y, W, H, pr, stats.data(), sr, sg, sb)) {
                        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
                    }
                    if (densityScopePixel(x, y, W, H, pr, stats.data(), sr, sg, sb)) {
                        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
                    }
                    if (statusStripPixel(x, y, W, H, pr, sr, sg, sb)) {
                        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
                    }
                }
        }
    }
}

// A canonical "Neutral" gray-balanced profile: three identical channels, a mild
// negative + a stronger print, zero printer lights. Neutral-preserving by
// construction (identical channels => equal outputs for equal inputs). Built-in
// stock families and Shoot-a-Chart calibration emit the SAME struct.
static inline SpeakProfile neutralProfile()
{
    // The negative and print stages are registered so that 18% gray (stops=0)
    // lands in the STRAIGHT region of BOTH curves: the negative maps gray to
    // D_neg ~= 1.05, and the print's speed point is set below that print
    // exposure so gray sits mid-straight on the print (full print gamma). The
    // product neg_gamma * print_gamma gives the ~1.5 system gamma of a film
    // print. Printer lights slide this operating point at use time.
    SpeakProfile p;
    for (int c = 0; c < 3; ++c) {
        p.negDmin[c] = 0.15f; p.negDmax[c] = 2.90f; p.negGamma[c] = 0.62f;
        p.negToe[c]  = 2.5f;  p.negShoulder[c] = 2.5f; p.negSpeed[c] = -1.5f;
        p.prnDmin[c] = 0.10f; p.prnDmax[c] = 3.30f; p.prnGamma[c] = 2.60f;
        p.prnToe[c]  = 3.5f;  p.prnShoulder[c] = 2.2f; p.prnSpeed[c] = -1.75f;
        p.printerLights[c] = 0.0f;
        p.subSat[c] = 0.0f; p.subSatKnee[c] = 0.0f;   // 0 = subtractive color off / knee disabled
        p.splitShadow[c] = 0.0f; p.splitHigh[c] = 0.0f;
    }
    for (int k = 0; k < 9; ++k) p.dyeCouple[k] = 0.0f;
    p.printerMaster = 0.0f;
    p.splitPivot = 0.0f; p.splitBalance = 0.5f;
    // Halation: OFF by default (amount 0 => bit-exact identity, and the whole
    // multi-pass scatter chain is skipped). The radius default is a modelled
    // characteristic scale, not a measured stock: base-reflection geometry puts
    // the TIR onset around 2*t*tan(asin(1/n)) ~ 0.23 mm for a 0.13 mm base,
    // ~1% of a Super-35 frame. That sets the ORDER of the default; it is not a
    // precision claim, and the hint does not make one.
    p.halAmount = 0.0f; p.halRadius = 1.0f; p.halThresh = 0.6f;
    // Grain: OFF by default (amount 0 => bit-exact identity). The size default
    // is a fine-grain pitch of 0.10% of frame height (~1.1 px at 1080p, ~2.2 px
    // at UHD — the same physical size on the frame at both, which is the point).
    p.grainAmount = 0.0f; p.grainSize = 0.10f;
    // Bloom: OFF by default (amount 0 => bit-exact identity). Radius and veil
    // are MODELLED defaults for a clean projection path — a wider spread than
    // halation's base-reflection ring (glare comes from the whole optical
    // train) and a small veil share. No lens was measured; the hints say so.
    p.bloomAmount = 0.0f; p.bloomRadius = 4.0f; p.bloomVeil = 0.10f;
    // Vignette: OFF by default. The field default is a normal lens's
    // half-diagonal (~27 deg); MODELLED, the hint says so.
    p.vignAmount = 0.0f; p.vignField = 27.0f;
    // Gate weave: OFF by default. Amplitude default when enabled is small on
    // purpose (0.05% of height ~ 0.5px at 1080p); MODELLED.
    p.weaveAmount = 0.0f; p.weaveSpeed = 1.0f;
    p.systemGamma = 1.6f; p.residualLUT = 0; p.profileVersion = 1; p._pad0 = 0;
    return p;
}

// ---------------------------------------------------------------------------
// STOCK PRESETS (v1.0, SPEC-1.0 §3) — tone-spine starting points.
//
// Three behavioural families shaped from PUBLISHED sensitometry (data-sheet
// H&D curve families: modern camera negatives publish gamma ~0.5-0.6 with
// long straight-line latitude; classic release-print curves publish gamma
// ~2.6-3.1 with hard toes; reversal stocks publish overall gamma ~1.7-1.9
// with deep Dmax and abrupt shoulders). No commercial profile is cloned and
// no trademarked stock is named — these are FAMILY SHAPES, stated as such,
// gray-balanced by construction (identical channels). The UI macro handles
// (contrast / shoulder / toe / printer lights) are written by the plugin on
// preset selection, so the preset is transparent: the user SEES the handles
// move and can trim from there. Gated: G46 (neutrality, monotonicity, gray
// operating point, measured system gamma per family).
// ---------------------------------------------------------------------------
#define SPEAK_STOCK_NEUTRAL   0
#define SPEAK_STOCK_LATITUDE  1   // modern negative: long latitude, soft knee
#define SPEAK_STOCK_PUNCH     2   // classic punchy print: hard toe, fast climb
#define SPEAK_STOCK_CHROME    3   // reversal-like: steep, deep, unforgiving

static inline SpeakProfile stockProfile(int preset)
{
    SpeakProfile p = neutralProfile();
    switch (preset) {
    default:
        break;
    case SPEAK_STOCK_LATITUDE:
        // Modern camera negative + standard print: the negative carries the
        // latitude (low gamma, gentle shoulder far out), the print carries
        // the contrast. The SOFTEST family — measured system gamma 1.15 at
        // gray (the whole point of latitude is a flatter, wider curve). Gray
        // sits at the local-gamma peak (G46, gray/peak 0.99).
        for (int c = 0; c < 3; ++c) {
            p.negGamma[c] = 0.58f; p.negShoulder[c] = 1.6f; p.negToe[c] = 2.0f;
            p.negDmax[c] = 3.10f;  p.negSpeed[c] = -1.8f;
            p.prnGamma[c] = 2.65f; p.prnShoulder[c] = 1.9f; p.prnToe[c] = 3.0f;
        }
        p.systemGamma = 1.15f;
        break;
    case SPEAK_STOCK_PUNCH:
        // Classic negative through a punchy release print: harder print toe
        // (published classic print curves), quicker shoulder. Measured system
        // gamma 1.69 at gray, gray at the peak (G46, gray/peak 0.99).
        for (int c = 0; c < 3; ++c) {
            p.negGamma[c] = 0.69f; p.negShoulder[c] = 2.6f; p.negToe[c] = 2.8f;
            p.prnGamma[c] = 2.70f; p.prnShoulder[c] = 2.6f; p.prnToe[c] = 5.0f;
            p.prnDmax[c] = 3.40f;
        }
        p.systemGamma = 1.69f;
        break;
    case SPEAK_STOCK_CHROME:
        // Reversal-like: one unforgiving stage in spirit — here the mild
        // negative leaves the print to do the work. The STEEPEST family —
        // measured system gamma 1.79 at gray, deep Dmax, abrupt shoulder, the
        // classic chrome shadow drop. Gray sits just below the local-gamma
        // peak (G46, gray/peak 0.97 — chrome IS shoulder-heavy at gray).
        for (int c = 0; c < 3; ++c) {
            p.negGamma[c] = 0.70f; p.negShoulder[c] = 3.2f; p.negToe[c] = 3.0f;
            p.prnGamma[c] = 2.78f; p.prnShoulder[c] = 3.4f; p.prnToe[c] = 6.0f;
            p.prnDmax[c] = 3.60f;  p.prnSpeed[c] = -1.65f;
        }
        p.systemGamma = 1.79f;
        break;
    }
    return p;
}

// The UI macro-handle values each preset writes (plugin-side, on selection):
// contrast is a trim on the preset's own print gamma, so it resets to 1;
// shoulder and toe SET the print curve absolutely, so they carry the
// preset's values. Kept beside stockProfile so the two cannot drift.
static inline void stockHandleDefaults(int preset, float& contrast,
                                       float& shoulder, float& toe)
{
    const SpeakProfile p = stockProfile(preset);
    contrast = 1.0f;
    shoulder = p.prnShoulder[0];
    toe      = p.prnToe[0];
}

// ---------------------------------------------------------------------------
// FORMAT PRESETS (v1.0, SPEC-1.0 §3) — one physical size, three frames.
//
// Grain pitch and halation radius are PHYSICAL sizes on the film, so their
// %-of-frame-height values scale inversely with the format's frame height:
// shooting Super 16 does not shrink the grain, it shrinks the frame around
// it. One modelled pair — a ~10 um effective grain cluster and the ~0.23 mm
// base-reflection (TIR) halation ring already modelled in neutralProfile —
// divided by each format's camera-aperture height:
//     Super 35, 4-perf   18.66 mm
//     35mm 2-perf         9.35 mm
//     Super 16            7.49 mm
// MODELLED defaults, stated as such (no stock was measured); the point is
// the RATIOS, which are geometry, not opinion. Gated: G47 pins
// size% x height == const across the table, both columns.
// ---------------------------------------------------------------------------
#define SPEAK_FMT_LEAVE   0   // "leave as set" — the preset applier's rest state
#define SPEAK_FMT_S35     1
#define SPEAK_FMT_2PERF   2
#define SPEAK_FMT_S16     3

static inline float formatHeightMm(int fmt)
{
    if (fmt == SPEAK_FMT_S35)   return 18.66f;
    if (fmt == SPEAK_FMT_2PERF) return 9.35f;
    if (fmt == SPEAK_FMT_S16)   return 7.49f;
    return 0.0f;
}
static const float kFmtGrainUm  = 10.0f;   // effective grain-cluster size, um
static const float kFmtHalMm    = 0.23f;   // TIR onset ring (see neutralProfile)

static inline void formatDefaults(int fmt, float& grainSizePct, float& halRadiusPct)
{
    const float h = formatHeightMm(fmt);
    if (h <= 0.0f) { grainSizePct = 0.0f; halRadiusPct = 0.0f; return; }
    grainSizePct = kFmtGrainUm * 0.001f / h * 100.0f;   // um -> mm -> % of height
    halRadiusPct = kFmtHalMm / h * 100.0f;
}

} // namespace speakcore

#endif // OPENNR_SPEAK_CORE_H

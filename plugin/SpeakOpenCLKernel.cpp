// Speak — OpenCL implementation of the film-reconstruction pipeline.
// Line-by-line port of plugin/speak_core.h; keep the two in sync. This is the
// render path Resolve uses on NVIDIA/AMD/Intel when CUDA isn't advertised, and
// the primary path on Windows/Linux.

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "SpeakParams.h"

static const char* kSpeakKernelSource = R"CLC(

typedef struct SpeakProfile
{
    float negDmin[3];   float negDmax[3];   float negGamma[3];
    float negToe[3];    float negShoulder[3]; float negSpeed[3];
    float printerLights[3]; float printerMaster;
    float prnDmin[3];   float prnDmax[3];   float prnGamma[3];
    float prnToe[3];    float prnShoulder[3]; float prnSpeed[3];
    float dyeCouple[9]; float subSat[3];    float subSatKnee[3];
    float splitShadow[3]; float splitHigh[3]; float splitPivot; float splitBalance;
    float halAmount;    float halRadius;   float halThresh;
    float grainAmount;  float grainSize;
    float bloomAmount;  float bloomRadius; float bloomVeil;
    float vignAmount;   float vignField;
    float weaveAmount;  float weaveSpeed;
    float systemGamma;  int residualLUT;    int profileVersion; int _pad0;
} SpeakProfile;

typedef struct SpeakParams
{
    int inputColorSpace; int outputMode; int grainRef; float strength;
    int frameIndex; int viewMode;
    int enableTone; int enableDye; int enableSplit; int enableOptics;
    int scopeHD; int scopeDensity; int scopeVector;
    int enableGrain; int matteSource; float grainMatteFloor;
    int matteKeyMissing;
    int statusStrip; int statusText[28];
    SpeakProfile profile;
    int maskExternal;
} SpeakParams;

#define kLog10_2 0.301029996f
#define k18Gray  0.18f
#define kPrinterPt 0.025f
#define kLinTiny 1e-8f
#define kKneeMin 0.05f
#define kDI_A 0.0075f
#define kDI_B 7.0f
#define kDI_C 0.07329248f
#define kDI_M 10.44426855f
#define kDI_LIN_CUT 0.00262409f
#define kDI_LOG_CUT 0.02740668f

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float pow10f(float x) { return exp2(x * 3.32192809f); }

inline float softplusf(float z)
{
    float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + log(1.0f + exp(-az));
}

inline float diDecode(float v)
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M) : (exp2(v / kDI_C - kDI_B) - kDI_A);
}
inline float diEncode(float L)
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M) : ((log2(L + kDI_A) + kDI_B) * kDI_C);
}
inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return exp2(v * 17.52f - 9.72f);
}
inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (log2(L) + 9.72f) / 17.52f;
}
inline float decodeToLinear(int cs, float v)
{
    if (cs == 0) return diDecode(v);
    if (cs == 1) return v <= 0.0f ? 0.0f : pow(v, 2.4f);
    if (cs == 3) return acesCctDecode(v);
    return v;
}
inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : pow(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

__constant float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
__constant float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };
inline void mul3(__constant float* m, float r, float g, float b,
                 float* oR, float* oG, float* oB)
{
    *oR = m[0] * r + m[1] * g + m[2] * b;
    *oG = m[3] * r + m[4] * g + m[5] * b;
    *oB = m[6] * r + m[7] * g + m[8] * b;
}
inline void gamutToRec709Lin(int cs, float r, float g, float b,
                             float* oR, float* oG, float* oB)
{
    if (cs == 0 || cs == 2) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, &X, &Y, &Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        *oR = r; *oG = g; *oB = b;
    }
}

inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                     float toe, float shoulder, float speed)
{
    float t = toe < kKneeMin ? kKneeMin : toe;
    float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}
inline float chainDensity(float stops, int ch, const SpeakProfile* p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p->negDmin[ch], p->negDmax[ch], p->negGamma[ch],
                         p->negToe[ch], p->negShoulder[ch], p->negSpeed[ch]);
    float printerOff = (p->printerMaster + p->printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p->prnDmin[ch], p->prnDmax[ch], p->prnGamma[ch],
                   p->prnToe[ch], p->prnShoulder[ch], p->prnSpeed[ch]);
}
inline float toneChannel(float lin, int ch, const SpeakProfile* p)
{
    float stops = log2((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    float Dprn = chainDensity(stops, ch, p);
    float Dref = chainDensity(0.0f, ch, p);
    return k18Gray * pow10f(-(Dprn - Dref));
}
#define SPEAK_EXP_BINS       128
#define SPEAK_STATS_HIST_EXP 0
#define SPEAK_STATS_HIST_MAX 128
#define SPEAK_WF_COLS        128
#define SPEAK_WF_ROWS        96
#define SPEAK_WF_DMAX        3.0f
#define SPEAK_STATS_WF       129
#define SPEAK_STATS_WF_MAX   (129 + 36864)

inline int wfColOf(int x, int W)
{
    int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
inline int wfRowOf(float D)
{
    int r = (int)(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}

inline int expBinOf(float stops)
{
    int b = (int)((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
    return b < 0 ? 0 : (b >= SPEAK_EXP_BINS ? SPEAK_EXP_BINS - 1 : b);
}
inline float pixelStops(int cs, float r, float g, float b)
{
    float m = (decodeToLinear(cs, r) + decodeToLinear(cs, g) + decodeToLinear(cs, b)) * (1.0f / 3.0f);
    return log2((m < kLinTiny ? kLinTiny : m) / k18Gray);
}

inline float density10(float lin)
{
    return -log2(lin < 1e-6f ? 1e-6f : lin) * kLog10_2;
}
inline float softCapKnee(float d, float cap)
{
    if (cap <= 0.0f) return d;
    return cap - (1.0f / 8.0f) * softplusf(8.0f * (cap - d));
}
inline void subtractiveColor(float r, float g, float b, const SpeakProfile* p,
                             float* oR, float* oG, float* oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float devR = DR - Dbar, devG = DG - Dbar, devB = DB - Dbar;
    float cR = (1.0f + p->subSat[0]) * devR - (p->dyeCouple[1] * devG + p->dyeCouple[2] * devB);
    float cG = (1.0f + p->subSat[1]) * devG - (p->dyeCouple[3] * devR + p->dyeCouple[5] * devB);
    float cB = (1.0f + p->subSat[2]) * devB - (p->dyeCouple[6] * devR + p->dyeCouple[7] * devG);
    float DpR = softCapKnee(Dbar + cR, p->subSatKnee[0]);
    float DpG = softCapKnee(Dbar + cG, p->subSatKnee[1]);
    float DpB = softCapKnee(Dbar + cB, p->subSatKnee[2]);
    *oR = pow10f(-DpR); *oG = pow10f(-DpG); *oB = pow10f(-DpB);
}
inline bool dyeActive(const SpeakProfile* p)
{
    return p->subSat[0] != 0.0f || p->subSat[1] != 0.0f || p->subSat[2] != 0.0f ||
           p->dyeCouple[1] != 0.0f || p->dyeCouple[2] != 0.0f || p->dyeCouple[3] != 0.0f ||
           p->dyeCouple[5] != 0.0f || p->dyeCouple[6] != 0.0f || p->dyeCouple[7] != 0.0f;
}

inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline void splitWeights(float Dbar, const SpeakProfile* p, float* wShadow, float* wHigh)
{
    float grayD  = 0.744727f;
    float pivotD = grayD - p->splitPivot * kLog10_2;
    float halfW  = 0.25f + 1.5f * clampf(p->splitBalance, 0.0f, 1.0f);
    float x = (Dbar - pivotD) / halfW;
    *wShadow = smooth01(x);
    *wHigh   = smooth01(-x);
}
inline void splitTone(float r, float g, float b, const SpeakProfile* p,
                      float* oR, float* oG, float* oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, &wS, &wH);
    *oR = pow10f(-(DR + wS * p->splitShadow[0] + wH * p->splitHigh[0]));
    *oG = pow10f(-(DG + wS * p->splitShadow[1] + wH * p->splitHigh[1]));
    *oB = pow10f(-(DB + wS * p->splitShadow[2] + wH * p->splitHigh[2]));
}
inline bool splitActive(const SpeakProfile* p)
{
    return p->splitShadow[0] != 0.0f || p->splitShadow[1] != 0.0f || p->splitShadow[2] != 0.0f ||
           p->splitHigh[0] != 0.0f || p->splitHigh[1] != 0.0f || p->splitHigh[2] != 0.0f;
}

// ---------------------------------------------------------------------------
// HALATION (Phase 4) — the scatter pyramid. Port of speak_core.h's halation
// block; see there for the physics and for why the injection is a re-exposure
// ahead of the negative rather than an end-chain overlay. Atomics-free by
// construction (box decimation is order-independent), which also keeps it clear
// of Apple's OpenCL global-int32-atomics miscompile.
// ---------------------------------------------------------------------------
#define SPEAK_HAL_MAXLEV     14
#define SPEAK_HAL_MINDIM     8
#define SPEAK_HAL_SIGMA_MIN  0.05f

__constant float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };

inline float halExcess(float lin, float thresh)
{
    float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}
inline float halAmountOf(const SpeakParams* pr)
{
    return (pr->enableOptics != 0) ? pr->profile.halAmount : 0.0f;
}
inline float halSigmaPx(int H, const SpeakParams* pr)
{
    float s = pr->profile.halRadius * 0.01f * (float)H;
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
inline bool halActive(const SpeakParams* pr)
{
    return (pr->enableTone != 0) && (pr->strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- pyramid geometry (identical arithmetic in all four backends) ----
inline int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
inline void halLevelInfo(int W, int H, int L, int* lw, int* lh, int* off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    *lw = w; *lh = h; *off = o;
}

#define kHalSigmaC   0.645497f
// Fall rates are ARGUMENTS, passed from the host (kHostHalCoreFall etc.):
// halation and bloom share the pyramid machinery with different profiles
// (speak_core.h). No in-kernel fall constants exist on purpose — a constant
// here that nothing reads would be a knob that cannot fire.
inline float halLevelWeight(int L, float sigmaTarget, float coreFall, float skirtFall)
{
    float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    float Lt = log2(s / kHalSigmaC);
    float d = (float)L - Lt;
    return (d <= 0.0f) ? exp2(coreFall * d) : exp2(-skirtFall * d);
}

// ---- pyramid taps (per-pixel, so the GPU kernels are textual ports) ----
inline float halFetch(__global const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[((size_t)off + (size_t)yy * lw + xx) * 3 + c];
}

__constant float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
inline void halDecimatePixel(__global const float* arena, int sOff, int sW, int sH,
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
inline void halBSpline(float t, float* w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
inline float halSampleLevel(__global const float* arena, int off, int lw, int lh,
                            int W, int H, int x, int y, int c)
{
    float fx = ((float)x + 0.5f) * (float)lw / (float)W - 0.5f;
    float fy = ((float)y + 0.5f) * (float)lh / (float)H - 0.5f;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float wx[4], wy[4];
    halBSpline(fx - (float)x0, wx);
    halBSpline(fy - (float)y0, wy);
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
inline float halWeightSum(int nLev, float sigmaTarget, float coreFall, float skirtFall)
{
    float wsum = 0.0f;
    for (int L = 0; L < nLev; ++L) wsum += halLevelWeight(L, sigmaTarget, coreFall, skirtFall);
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
// resolution) because none of them measured isotropy. Going one octave at a time
// fixes it because each step interpolates only between ADJACENT samples and is
// then re-filtered by every step below it. It is also strictly CHEAPER: the
// total work is sum_L (level L's pixels) = 4/3 N, against nLev*N.
//
// IN-PLACE IS SAFE: a thread reads its OWN (x,y) at level L and a neighbourhood
// at level L+1 (a disjoint arena region), and writes only its own (x,y) at
// level L. Ordering BETWEEN levels is the only requirement (see the host's
// per-level barriers).
//
// The level geometry arrives as ARGUMENTS rather than being derived here from L,
// for the same reason SpeakDecimateKernel takes it — see that kernel's header:
// Apple's OpenCL optimizer miscompiles in-kernel halLevelInfo feeding halFetch's
// clamped tap addressing. The host computes it with its own textual mirror of
// halLevelInfo, so the body below stays a line-for-line port of the core's.
inline void halAccumPixel(__global float* arena, int L, int nLev, float sigmaTarget,
                          float coreFall, float skirtFall,
                          int lw, int lh, int off, int cw, int ch, int coff,
                          int x, int y, float* out)
{
    float wl = halLevelWeight(L, sigmaTarget, coreFall, skirtFall);
    if (L >= nLev - 1) {                       // the coarsest level: nothing above it
        for (int c = 0; c < 3; ++c) out[c] = wl * halFetch(arena, off, lw, lh, x, y, c);
        return;
    }
    for (int c = 0; c < 3; ++c)
        out[c] = wl * halFetch(arena, off, lw, lh, x, y, c)
               + halSampleLevel(arena, coff, cw, ch, lw, lh, x, y, c);
}

inline void lookLinear(float r, float g, float b,
                       float scatR, float scatG, float scatB,
                       float vgain,
                       const SpeakParams* pr,
                       float* oR, float* oG, float* oB)
{
    int cs = pr->inputColorSpace;
    float lr = vgain * decodeToLinear(cs, r);
    float lg = vgain * decodeToLinear(cs, g);
    float lb = vgain * decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr->enableTone != 0) && (pr->strength > 0.0f)) {
        float s = clampf(pr->strength, 0.0f, 1.0f);
        float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light adds to the scene light ENTERING the
        // negative, so it goes in the CURVE'S ARGUMENT. The dry side of the mix
        // stays lr, not the re-exposed value — otherwise strength 0 would leave
        // the raw scatter added to linear with no curve downstream, which is
        // precisely the end-chain overlay the arm rejected (and would break the
        // identity gate G3).
        float er = lr, eg = lg, eb = lb;
        if (a > 0.0f) {
            er = lr + a * kHalWeight[0] * scatR;
            eg = lg + a * kHalWeight[1] * scatG;
            eb = lb + a * kHalWeight[2] * scatB;
        }
        mr = lerpf(lr, toneChannel(er, 0, &pr->profile), s);
        mg = lerpf(lg, toneChannel(eg, 1, &pr->profile), s);
        mb = lerpf(lb, toneChannel(eb, 2, &pr->profile), s);
    }
    if ((pr->enableDye != 0) && dyeActive(&pr->profile)) subtractiveColor(mr, mg, mb, &pr->profile, &mr, &mg, &mb);
    if ((pr->enableSplit != 0) && splitActive(&pr->profile)) splitTone(mr, mg, mb, &pr->profile, &mr, &mg, &mb);
    *oR = mr; *oG = mg; *oB = mb;
}

// ---- Grain (Phase 4) — see speak_core.h for the physics and the gates ----
#define kGrainScale 0.045f
#define kGrainDCap  4.0f

inline float grainHash(uint ix, uint iy, uint f, uint ch)
{
    uint h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return ((float)(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}
/* Variance-normalized bilinear value noise (constant variance everywhere —
   plain bilinear has a 4:1 variance ripple that reads as a lattice grid). */
inline float grainLattice(float x, float y, float size, uint f, uint ch)
{
    float gx = x / size, gy = y / size;
    int ix = (int)floor(gx), iy = (int)floor(gy);
    float tx = gx - (float)ix, ty = gy - (float)iy;
    float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty,          w11 = tx * ty;
    float n = w00 * grainHash((uint)ix,       (uint)iy,       f, ch)
            + w10 * grainHash((uint)(ix + 1), (uint)iy,       f, ch)
            + w01 * grainHash((uint)ix,       (uint)(iy + 1), f, ch)
            + w11 * grainHash((uint)(ix + 1), (uint)(iy + 1), f, ch);
    float ww = w00 * w00 + w10 * w10 + w01 * w01 + w11 * w11;
    return n / sqrt(ww);
}
/* Octave-difference bandpass: kills DC/low-freq blotch; "grain has a size". */
inline float grainBand(float x, float y, float sizePx, uint f, uint ch)
{
    return (grainLattice(x, y, sizePx, f, ch) -
            grainLattice(x, y, sizePx * 2.0f, f, ch + 8u)) * 0.70710678f;
}
inline float grainSizePx(int H, const SpeakParams* pr)
{
    float s = pr->profile.grainSize * 0.01f * (float)H;
    return s < 1.0f ? 1.0f : s;
}
inline bool grainActive(const SpeakParams* pr)
{
    return (pr->enableGrain != 0) && (pr->profile.grainAmount > 0.0f);
}
/* Density-domain grain on the look's working-linear output. `conf` = the
   pixel's INPUT ALPHA (only read when a matte source is active; clamped
   here; a selected-but-unwired key forces it to 0 - the Floor). */
inline void applyGrain(float* r, float* g, float* b, float conf,
                       int x, int y, int H, const SpeakParams* pr)
{
    if (!grainActive(pr)) return;
    float sz = grainSizePx(H, pr);
    float confK = (pr->matteKeyMissing != 0) ? 0.0f : conf;
    float m = (pr->matteSource != 0)
            ? lerpf(clampf(pr->grainMatteFloor, 0.0f, 1.0f), 1.0f, clampf(confK, 0.0f, 1.0f))
            : 1.0f;
    float a = pr->profile.grainAmount * m * kGrainScale;
    if (a <= 0.0f) return;
    uint fr = (uint)pr->frameIndex;
    float fx = (float)x, fy = (float)y;
    for (int c = 0; c < 3; ++c) {
        float* ch = (c == 0) ? r : ((c == 1) ? g : b);
        float D = density10(*ch);
        float Dc = D < 0.0f ? 0.0f : (D > kGrainDCap ? kGrainDCap : D);
        float sigmaD = a * sqrt(Dc);
        float n = grainBand(fx, fy, sz, fr, (uint)c);
        *ch = pow10f(-(D + sigmaD * n));
    }
}

/* ---- VIGNETTE (Phase 4) — cos^4, pre-curve; see speak_core.h ---- */
inline bool vignActive(const SpeakParams* pr)
{
    return (pr->enableOptics != 0) && (pr->strength > 0.0f)
        && (pr->profile.vignAmount > 0.0f);
}
inline float vignGain(int x, int y, int W, int H, const SpeakParams* pr)
{
    if (!vignActive(pr)) return 1.0f;
    float a = clampf(pr->profile.vignAmount, 0.0f, 1.0f)
            * clampf(pr->strength, 0.0f, 1.0f);
    float cx = 0.5f * (float)(W - 1);
    float cy = 0.5f * (float)(H - 1);
    float dx = (float)x - cx, dy = (float)y - cy;
    float rhd2 = cx * cx + cy * cy;
    float r2 = (dx * dx + dy * dy) / (rhd2 > 0.0f ? rhd2 : 1.0f);
    float tanm = tan(pr->profile.vignField * 0.017453293f);
    float c2 = 1.0f / (1.0f + r2 * tanm * tanm);
    return lerpf(1.0f, c2 * c2, a);
}

/* ---- BLOOM (Phase 4) — energy-conserving glare; see speak_core.h ---- */
inline float bloomAmountOf(const SpeakParams* pr)
{
    return (pr->enableOptics != 0) ? clampf(pr->profile.bloomAmount, 0.0f, 1.0f) : 0.0f;
}
inline bool bloomActive(const SpeakParams* pr)
{
    return (pr->strength > 0.0f) && (bloomAmountOf(pr) > 0.0f);
}
inline float bloomSigmaPx(int H, const SpeakParams* pr)
{
    float s = pr->profile.bloomRadius * 0.01f * (float)H;
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
inline float bloomFinite(float v)
{
    /* see speak_core.h: contain EXR holes / upstream divides to one pixel */
    return (v == v && v <= 3.402823e38f && v >= -3.402823e38f) ? v : 0.0f;
}
inline void bloomApplyPixel(float* r, float* g, float* b,
                            float sR, float sG, float sB,
                            const SpeakParams* pr)
{
    if (!bloomActive(pr)) return;
    float a = bloomAmountOf(pr) * clampf(pr->strength, 0.0f, 1.0f);
    *r = lerpf(*r, sR, a);
    *g = lerpf(*g, sG, a);
    *b = lerpf(*b, sB, a);
}

/* ---- GATE WEAVE sampling (Phase 4) — Catmull-Rom, see speak_core.h.
   (The frame's displacement is computed HOST-side by the same speak_core
   closed form and passed in — it is frame-uniform.) */
inline void weaveCRw(float t, float* w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] =  1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] =  0.5f * t3 - 0.5f * t2;
}
inline void weaveSamplePixel(__global const float* img, int W, int H,
                             float sx, float sy, float* out4)
{
    float fx = floor(sx), fy = floor(sy);
    int x0 = (int)fx, y0 = (int)fy;
    float wx[4], wy[4];
    weaveCRw(sx - fx, wx);
    weaveCRw(sy - fy, wy);
    for (int c = 0; c < 4; ++c) out4[c] = 0.0f;
    for (int j = 0; j < 4; ++j) {
        int yy = y0 - 1 + j;
        int yc = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
        for (int i = 0; i < 4; ++i) {
            int xx = x0 - 1 + i;
            int xc = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
            float w = wx[i] * wy[j];
            size_t o = ((size_t)yc * W + xc) * 4;
            out4[0] += w * img[o + 0];
            out4[1] += w * img[o + 1];
            out4[2] += w * img[o + 2];
            out4[3] += w * img[o + 3];
        }
    }
    /* alpha is the matte: clamp CR's invented out-of-[0,1] values (RGB keeps
       the overshoot). See speak_core.h. */
    out4[3] = out4[3] < 0.0f ? 0.0f : (out4[3] > 1.0f ? 1.0f : out4[3]);
}

inline float scopeYStops(float inStops, int ch, const SpeakParams* pr)
{
    float lin = k18Gray * exp2(inStops);
    float outLin = lin;
    if ((pr->enableTone != 0) && (pr->strength > 0.0f)) {
        float s = clampf(pr->strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, &pr->profile), s);
    }
    return log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams* pr,
                         __global const uint* stats,
                         float* outR, float* outG, float* outB)
{
    if (pr->scopeHD == 0) return false;
    int sc = (H / 540) > 1 ? (H / 540) : 1;
    int panelW = 220 * sc, panelH = 150 * sc;
    int margin = 12 * sc;
    int px0 = margin, py0 = margin;
    int yd = H - 1 - y;
    int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    int pad = 6 * sc;
    int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    int gx = lx - pad, gy = ly - pad;

    *outR = 0.06f; *outG = 0.06f; *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        *outR = 0.30f; *outG = 0.30f; *outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    float rowStops = 6.0f - 12.0f * ((float)gy / (plotH - 1));
    int gcol0 = (int)((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = (int)((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { *outR = 0.24f; *outG = 0.24f; *outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { *outR = 0.13f; *outG = 0.13f; *outB = 0.13f; }

    uint hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        int hb = expBinOf(-6.0f + 12.0f * ((float)gx / (plotW - 1)));
        float f = (float)(stats[SPEAK_STATS_HIST_EXP + hb]) / (float)hmax;
        int barH = (int)(sqrt(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { *outR = 0.16f; *outG = 0.19f; *outB = 0.24f; }
    }

    int chR[3]; chR[0]=1; chR[1]=0; chR[2]=0;
    int chG[3]; chG[0]=0; chG[1]=1; chG[2]=0;
    int chB[3]; chB[0]=0; chB[1]=0; chB[2]=1;
    for (int ch = 0; ch < 3; ++ch) {
        float inS  = -6.0f + 12.0f * ((float)gx       / (plotW - 1));
        float inS2 = -6.0f + 12.0f * ((float)(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { float tt = y0; y0 = y1; y1 = tt; }
        float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            *outR = 0.10f + 0.85f * chR[ch];
            *outG = 0.10f + 0.85f * chG[ch];
            *outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { *outR = 0.95f; *outG = 0.10f; *outB = 0.10f; return true; }
            if (sw == 1) { *outR = 0.10f; *outG = 0.95f; *outB = 0.10f; return true; }
            if (sw == 2) { *outR = 0.10f; *outG = 0.10f; *outB = 0.95f; return true; }
        }
    }
    return true;
}

inline bool densityScopePixel(int x, int y, int W, int H, const SpeakParams* pr,
                              __global const uint* stats,
                              float* outR, float* outG, float* outB)
{
    if (pr->scopeDensity == 0) return false;

    int sc = (H / 540) > 1 ? (H / 540) : 1;
    int panelW = 220 * sc, panelH = 150 * sc;
    int margin = 12 * sc;
    int px0 = W - margin - panelW, py0 = margin;
    int yd = H - 1 - y;
    int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    int pad = 6 * sc;
    int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    int gx = lx - pad, gy = ly - pad;

    *outR = 0.06f; *outG = 0.06f; *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        *outR = 0.30f; *outG = 0.30f; *outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    int rowGray = (int)(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { *outR = 0.35f; *outG = 0.35f; *outB = 0.35f; return true; }
    if (gy == rowGray) { *outR = 0.30f; *outG = 0.26f; *outB = 0.12f; return true; }

    int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { *outR = 0.16f; *outG = 0.16f; *outB = 0.16f; return true; }

    uint wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        int within = gx - ch * chW;
        int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        int wrow = gy * SPEAK_WF_ROWS / plotH;
        uint c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                             wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            float inten = sqrt((float)c / (float)wmax);
            float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            *outR = (ch == 0) ? v : 0.05f;
            *outG = (ch == 1) ? v : 0.05f;
            *outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

inline void deliverInput(const SpeakParams* pr, float r, float g, float b,
                         float* oR, float* oG, float* oB)
{
    if (pr->outputMode == 1) {
        int cs = pr->inputColorSpace;
        float lr = decodeToLinear(cs, r);
        float lg = decodeToLinear(cs, g);
        float lb = decodeToLinear(cs, b);
        float rr, rg, rb;
        gamutToRec709Lin(cs, lr, lg, lb, &rr, &rg, &rb);
        rr = rr < 0.0f ? 0.0f : rr;
        rg = rg < 0.0f ? 0.0f : rg;
        rb = rb < 0.0f ? 0.0f : rb;
        *oR = encodeFromLinear(1, rr);
        *oG = encodeFromLinear(1, rg);
        *oB = encodeFromLinear(1, rb);
    } else {
        *oR = r; *oG = g; *oB = b;
    }
}

// ---------------------------------------------------------------------------
// ON-IMAGE TEXT (v1.0, SPEC-1.0 S2) — the status strip and the Setup Guide.
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
__constant ulong kSpeakFontBits[SPEAK_FONT_GLYPHS] = {
    0x000000000ul, 0x100421084ul, 0x00000294aul, 0x295f57d4aul,
    0x11f0707c4ul, 0x632222263ul, 0x593511526ul, 0x000001084ul,
    0x208210888ul, 0x088842082ul, 0x009575480ul, 0x0084f9080ul,
    0x088c00000ul, 0x0000f8000ul, 0x318000000ul, 0x002222200ul,
    0x3a33ae62eul, 0x3884210c4ul, 0x7c444422eul, 0x3a304111ful,
    0x211f4a988ul, 0x3a3083c3ful, 0x3a317844cul, 0x08422221ful,
    0x3a317462eul, 0x1910f462eul, 0x018c03180ul, 0x088c03180ul,
    0x208208888ul, 0x001f07c00ul, 0x088882082ul, 0x10044422eul,
    0x3ab5b422eul, 0x4631fc62eul, 0x3e317c62ful, 0x3a210862eul,
    0x1d318c527ul, 0x7c217843ful, 0x04217843ful, 0x7a31e862eul,
    0x4631fc631ul, 0x38842108eul, 0x19284211cul, 0x452519531ul,
    0x7c2108421ul, 0x4631ad771ul, 0x4631cd671ul, 0x3a318c62eul,
    0x04217c62ful, 0x59358c62eul, 0x45257c62ful, 0x3e107043eul,
    0x10842109ful, 0x3a318c631ul, 0x11518c631ul, 0x4775ac631ul,
    0x462a22a31ul, 0x108422a31ul, 0x7c222221ful, 0x38421084eul,
    0x020820820ul, 0x39084210eul, 0x000004544ul, 0x7c0000000ul,
    0x000000082ul, 0x7a3e83800ul, 0x3e318bc21ul, 0x3a210b800ul,
    0x7a318fa10ul, 0x383f8b800ul, 0x084238a4cul, 0x3a1e8c7c0ul,
    0x46318bc21ul, 0x388421804ul, 0x192843008ul, 0x24a32a421ul,
    0x388421086ul, 0x56b5aac00ul, 0x46318bc00ul, 0x3a318b800ul,
    0x042f8c5e0ul, 0x421e8c7c0ul, 0x04219b400ul, 0x3e0e0f800ul,
    0x324211c42ul, 0x5b318c400ul, 0x11518c400ul, 0x2ab58c400ul,
    0x454454400ul, 0x3a1e8c400ul, 0x7c4447c00ul, 0x608411098ul,
    0x108421084ul, 0x0c8441083ul, 0x0008a8800ul, 0x10c944200ul,
    0x000c60000ul, 0x0000f8000ul, 0x0088fa080ul, 0x0000f8000ul,
    0x108421084ul, 0x1084e0000ul, 0x108438000ul, 0x1084f8000ul,
    0x0000f9084ul, 0x0000e1084ul, 0x000039084ul,
};
// Per-row style of the guide card: 0 body, 1 title, 2 dim, 3 the blue
// wire, 4 numbered step.
__constant int kSpeakGuideStyle[SPEAK_GUIDE_ROWS] = { 1, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 2, 2, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2 };
__constant uchar kSpeakGuideText[SPEAK_GUIDE_ROWS * SPEAK_GUIDE_COLS] = {
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

inline int fontIndexOf(int c)
{
    if (c >= 32 && c < 127) return c - 32;
    if (c >= 1 && c <= 8)   return 94 + c;   /* check middot emdash arrow hline vline TL TR */
    if (c == 11 || c == 12) return 92 + c;   /* teeDown teeUp */
    if (c == 14 || c == 15) return 91 + c;   /* BL BR */
    return 0;                                /* anything else draws as space */
}

/* One glyph-cell sample: is (cellRow 0..8, cellCol 0..5) of char c lit? */
inline bool fontBit(int c, int cellRow, int cellCol)
{
    const int gi = fontIndexOf(c);
    int r = cellRow, cc = cellCol;
    if (gi >= 99) {                          /* line-drawing: bridge the gaps */
        if (cc == 5) cc = 4;
        if (r >= 7) r = 6;
    }
    if (r > 6 || cc > 4 || r < 0 || cc < 0) return false;
    return ((kSpeakFontBits[gi] >> (r * 5 + cc)) & 1) != 0;
}

inline int statusCharAt(const SpeakParams* pr, int k)
{
    return (pr->statusText[k >> 2] >> ((k & 3) * 8)) & 0xFF;
}
inline int statusLen(const SpeakParams* pr)
{
    for (int k = 0; k < 112; ++k) if (statusCharAt(pr, k) == 0) return k;
    return 112;
}

/* The one-line status strip (SPEC-1.0 §2): bottom-left, panel chrome like
   the scopes — same scale rule, same palette, pinned through the weave by
   the same overlay pass. Returns true when (x,y) belongs to the strip. */
inline bool statusStripPixel(int x, int y, int W, int H, const SpeakParams* pr,
                             float* outR, float* outG, float* outB)
{
    if (pr->statusStrip == 0) return false;
    const int n = statusLen(pr);
    if (n <= 0) return false;
    const int sc = (H / 540) > 1 ? (H / 540) : 1;
    const int pad = 4 * sc, margin = 12 * sc;
    const int stripH = 7 * sc + 2 * pad;
    int stripW = n * 6 * sc + 2 * pad;
    if (stripW > W - 2 * margin) stripW = W - 2 * margin;   /* clip, never wrap */
    const int yd = H - 1 - y;                               /* display row */
    const int lx = x - margin, ly = yd - (H - margin - stripH);
    if (lx < 0 || ly < 0 || lx >= stripW || ly >= stripH) return false;
    *outR = *outG = *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= stripW - sc || ly >= stripH - sc) {
        *outR = *outG = *outB = 0.30f; return true;
    }
    const int tx = lx - pad, ty = ly - pad;
    if (tx < 0 || ty < 0) return true;
    const int k = tx / (6 * sc);
    if (k >= n) return true;
    const int c = statusCharAt(pr, k);
    if (fontBit(c, ty / sc, (tx - k * 6 * sc) / sc)) {
        if (c == 1) { *outR = 0.25f; *outG = 0.90f; *outB = 0.40f; }  /* the check earns green */
        else        { *outR = *outG = *outB = 0.82f; }
    }
    return true;
}

/* The Setup Guide view (SPEC-1.0 §2): the Hush->Speak recipe rendered by the
   render kernel itself. Pure function of (x,y,W,H) — trivially parity-safe. */
inline void guidePixel(int x, int y, int W, int H,
                       float* outR, float* outG, float* outB)
{
    const int yd = H - 1 - y;
    const int gw = SPEAK_GUIDE_COLS * 6, gh = SPEAK_GUIDE_ROWS * 9;
    int gsc = (W * 92 / 100) / gw;                 /* integer scale, ~92% fit */
    const int vs = (H * 92 / 100) / gh;
    if (vs < gsc) gsc = vs;
    if (gsc < 1) gsc = 1;
    const int x0 = (W - gw * gsc) / 2, y0 = (H - gh * gsc) / 2;
    *outR = *outG = *outB = 0.045f;                /* the card */
    const int lx = x - x0, ly = yd - y0;
    if (lx < 0 || ly < 0 || lx >= gw * gsc || ly >= gh * gsc) return;
    const int col = lx / (6 * gsc), row = ly / (9 * gsc);
    if (col >= SPEAK_GUIDE_COLS || row >= SPEAK_GUIDE_ROWS) return;
    const int c = (int)kSpeakGuideText[row * SPEAK_GUIDE_COLS + col];
    if (!fontBit(c, (ly - row * 9 * gsc) / gsc, (lx - col * 6 * gsc) / gsc)) return;
    const int st = kSpeakGuideStyle[row];
    if      (st == 1) { *outR = *outG = *outB = 0.95f; }                 /* title */
    else if (st == 2) { *outR = *outG = *outB = 0.55f; }                 /* dim */
    else if (st == 3) { *outR = 0.30f; *outG = 0.55f; *outB = 0.98f; }   /* the blue wire */
    else if (st == 4) { *outR = 0.95f; *outG = 0.82f; *outB = 0.55f; }   /* steps */
    else              { *outR = *outG = *outB = 0.78f; }                 /* body */
}

inline void processPixel(float r, float g, float b, float srcA,
                         float scatR, float scatG, float scatB,
                         float bloomR, float bloomG, float bloomB,
                         int x, int y, int W, int H,
                         const SpeakParams* pr, __global const uint* stats,
                         int drawScopes,
                         float* outR, float* outG, float* outB)
{
    int cs = pr->inputColorSpace;
    bool toneOn = (pr->enableTone != 0) && (pr->strength > 0.0f);
    bool dyeOn = (pr->enableDye != 0) && dyeActive(&pr->profile);
    bool splitOn = (pr->enableSplit != 0) && splitActive(&pr->profile);
    bool grainOn = grainActive(pr);
    bool bloomOn = bloomActive(pr);
    bool vignOn = vignActive(pr);
    bool bake = (pr->outputMode == 1);
    if (!toneOn && !dyeOn && !splitOn && !grainOn && !bloomOn && !vignOn && !bake) {
        *outR = r; *outG = g; *outB = b;
    } else {
        float mr, mg, mb;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, &mr, &mg, &mb);
        applyGrain(&mr, &mg, &mb, srcA, x, y, H, pr);
        bloomApplyPixel(&mr, &mg, &mb, bloomR, bloomG, bloomB, pr);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, &rr, &rg, &rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, rr);
            *outG = encodeFromLinear(1, rg);
            *outB = encodeFromLinear(1, rb);
        } else {
            *outR = encodeFromLinear(cs, mr);
            *outG = encodeFromLinear(cs, mg);
            *outB = encodeFromLinear(cs, mb);
        }
    }
    if (pr->viewMode == 2 || (pr->viewMode == 1 && x < W / 2))
        deliverInput(pr, r, g, b, outR, outG, outB);

    // Isolated-scatter view: the ACTUAL injected re-exposure a * w_c * S_c,
    // delivered through the SAME output transform as the picture. Deliberately
    // NOT auto-normalized — see speak_core.h.
    if (pr->viewMode == 3) {
        float a = halAmountOf(pr);
        float on = ((pr->enableTone != 0) && (pr->strength > 0.0f)) ? 1.0f : 0.0f;
        float sr = on * a * kHalWeight[0] * scatR;
        float sg = on * a * kHalWeight[1] * scatG;
        float sb = on * a * kHalWeight[2] * scatB;
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, sr, sg, sb, &rr, &rg, &rb);
            sr = rr < 0.0f ? 0.0f : rr;
            sg = rg < 0.0f ? 0.0f : rg;
            sb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, sr);
            *outG = encodeFromLinear(1, sg);
            *outB = encodeFromLinear(1, sb);
        } else {
            *outR = encodeFromLinear(cs, sr);
            *outG = encodeFromLinear(cs, sg);
            *outB = encodeFromLinear(cs, sb);
        }
    }

    /* Isolated-grain view: 18% gray + the EXACT grain increment this pixel
       received, through the same output transform. Never auto-gained. */
    if (pr->viewMode == 4) {
        float pr0, pg0, pb0, pr1, pg1, pb1;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, &pr0, &pg0, &pb0);
        pr1 = pr0; pg1 = pg0; pb1 = pb0;
        applyGrain(&pr1, &pg1, &pb1, srcA, x, y, H, pr);
        float gr = k18Gray + (pr1 - pr0);
        float gg = k18Gray + (pg1 - pg0);
        float gb = k18Gray + (pb1 - pb0);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, gr, gg, gb, &rr, &rg, &rb);
            gr = rr < 0.0f ? 0.0f : rr;
            gg = rg < 0.0f ? 0.0f : rg;
            gb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, gr);
            *outG = encodeFromLinear(1, gg);
            *outB = encodeFromLinear(1, gb);
        } else {
            gr = gr < 0.0f ? 0.0f : gr;
            gg = gg < 0.0f ? 0.0f : gg;
            gb = gb < 0.0f ? 0.0f : gb;
            *outR = encodeFromLinear(cs, gr);
            *outG = encodeFromLinear(cs, gg);
            *outB = encodeFromLinear(cs, gb);
        }
    }

    /* Isolated-bloom view: gray + the SIGNED delta (out - look); the borrow
       at sources is negative, the halo positive — see speak_core.h. */
    if (pr->viewMode == 5) {
        float lr0, lg0, lb0, lr1, lg1, lb1;
        lookLinear(r, g, b, scatR, scatG, scatB, vignGain(x, y, W, H, pr),
                   pr, &lr0, &lg0, &lb0);
        applyGrain(&lr0, &lg0, &lb0, srcA, x, y, H, pr);
        lr1 = lr0; lg1 = lg0; lb1 = lb0;
        bloomApplyPixel(&lr1, &lg1, &lb1, bloomR, bloomG, bloomB, pr);
        float gr = k18Gray + (lr1 - lr0);
        float gg = k18Gray + (lg1 - lg0);
        float gb = k18Gray + (lb1 - lb0);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, gr, gg, gb, &rr, &rg, &rb);
            gr = rr < 0.0f ? 0.0f : rr;
            gg = rg < 0.0f ? 0.0f : rg;
            gb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, gr);
            *outG = encodeFromLinear(1, gg);
            *outB = encodeFromLinear(1, gb);
        } else {
            gr = gr < 0.0f ? 0.0f : gr;
            gg = gg < 0.0f ? 0.0f : gg;
            gb = gb < 0.0f ? 0.0f : gb;
            *outR = encodeFromLinear(cs, gr);
            *outG = encodeFromLinear(cs, gg);
            *outB = encodeFromLinear(cs, gb);
        }
    }

    /* The Setup Guide view (v1.0, SPEC-1.0 S2): the recipe card, drawn by
       the kernel itself; the strip and scopes overlay it below (chrome). */
    if (pr->viewMode == 6)
        guidePixel(x, y, W, H, outR, outG, outB);

    if (drawScopes != 0) {
        float sr, sg, sb;
        if (hdScopePixel(x, y, W, H, pr, stats, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
        if (densityScopePixel(x, y, W, H, pr, stats, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
        if (statusStripPixel(x, y, W, H, pr, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
    }
}

// ---------------------------------------------------------------------------
// The scatter pyramid, mirroring buildHalScatter in speak_core.h. Four passes:
//   excess (full res, level 0) -> decimate (once per level, fine to coarse) ->
//   accum (once per level, COARSE TO FINE, in place) -> normalize (full res).
// The host skips them all when halation is inactive, and the readers below gate
// on the SAME condition, so nothing ever touches the placeholder binding that
// stands in for the buffers in that case.
// ---------------------------------------------------------------------------

// Level 0: the per-channel scene-linear highlight excess. THRESHOLD BEFORE
// DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t).
__kernel void SpeakExcessKernel(SpeakParams p, int W, int H,
                                __global const float* src, __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int cs = p.inputColorSpace;
    float th = p.profile.halThresh;
    size_t i = ((size_t)y * W + x) * 4;
    size_t o = ((size_t)y * W + x) * 3;
    /* vignette first: light the lens never delivered cannot halate */
    float vg = vignGain(x, y, W, H, &p);
    arena[o + 0] = halExcess(vg * decodeToLinear(cs, src[i + 0]), th);
    arena[o + 1] = halExcess(vg * decodeToLinear(cs, src[i + 1]), th);
    arena[o + 2] = halExcess(vg * decodeToLinear(cs, src[i + 2]), th);
}

// One dispatch per level L = 1..nLev-1: build level L from level L-1. Reads and
// writes disjoint regions of the same arena, so the only ordering requirement is
// between dispatches (see the host's barriers).
//
// The level geometry arrives as ARGUMENTS rather than being derived here from L.
// The host computes it with its own textual mirror of halLevelInfo, so this
// kernel body stays a line-for-line port of buildHalScatter's inner loop — the
// host loop above it now carries the two halLevelInfo(L-1)/halLevelInfo(L) calls
// that the core's loop body has.
//
// This is NOT a style choice. Apple's OpenCL optimizer MISCOMPILES the in-kernel
// form: with `halLevelInfo(W,H,L-1,&sw,&sh,&so)` feeding halFetch's clamped tap
// addressing, levels >= 3 come out as ~0.5 * source[x + dw] instead of the 4x4
// binomial mean (levels 0-2 are unaffected, which is why a small frame looks
// fine). Verified: -cl-opt-disable fixes it; passing the SOURCE geometry (sw,sh,
// so) as args fixes it; passing only the DEST geometry does not. This is the
// same class of defect as the global-int32-atomics miscompile that already
// forces the stats pass to be skipped on this runtime. The arithmetic is
// unchanged — it just happens on the host now.
__kernel void SpeakDecimateKernel(int sw, int sh, int so, int dw, int dh, int doff,
                                  __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    float v[3];
    halDecimatePixel(arena, so, sw, sh, x, y, v);
    size_t o = ((size_t)doff + (size_t)y * dw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// One dispatch per level L = nLev-1 down to 0, COARSE TO FINE, with the grid set
// to LEVEL L's dims: acc_L = w_L * level_L + upsample_2x(acc_{L+1}), written IN
// PLACE. Level L reads level L+1, which the PREVIOUS dispatch wrote, so the
// host's barrier between levels is required. Geometry comes in as args (see
// halAccumPixel's header). `cw/ch/coff` are level L+1's and are unread at
// L == nLev-1.
__kernel void SpeakAccumKernel(SpeakParams p, int H, int L, int nLev,
                               float coreFall, float skirtFall, int isBloom,
                               int lw, int lh, int off, int cw, int ch, int coff,
                               __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= lw || y >= lh) return;
    float sig = (isBloom != 0) ? bloomSigmaPx(H, &p) : halSigmaPx(H, &p);
    float v[3];
    halAccumPixel(arena, L, nLev, sig, coreFall, skirtFall,
                  lw, lh, off, cw, ch, coff, x, y, v);
    size_t o = ((size_t)off + (size_t)y * lw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// Normalize level 0 into the scatter plane: sum(w) = 1 => energy preserved.
// `inv` is recomputed here rather than passed in from the host so it uses the
// very same in-kernel halLevelWeight the accumulate used. `isBloom` selects
// the sigma; the veil share and its weight are computed IN-KERNEL from the
// same halWeightSum, so no host-side exp2 can drift the scale. Halation runs
// this with isBloom = 0 (veil 0) and a placeholder mean binding whose value
// is multiplied by that zero — the binding just has to be valid and finite.
__kernel void SpeakNormalizeKernel(SpeakParams p, int W, int H,
                                   float coreFall, float skirtFall, int isBloom,
                                   __global const float* arena, __global float* scat,
                                   __global const float* meanC)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int nLev = halLevelCount(W, H);
    float sig = (isBloom != 0) ? bloomSigmaPx(H, &p) : halSigmaPx(H, &p);
    float base = halWeightSum(nLev, sig, coreFall, skirtFall);
    float veil = 0.0f;
    if (isBloom != 0) {
        float v = clampf(p.profile.bloomVeil, 0.0f, 0.9f);
        if (v > 0.0f) veil = v / (1.0f - v) * base;
    }
    float inv = 1.0f / (base + veil);
    size_t k = ((size_t)y * W + x) * 3;
    /* meanC is only READ under isBloom: halation binds a placeholder whose
       contents are undefined, and 0 * NaN is NaN — the guard is the fix. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    if (isBloom != 0) { a0 = veil * meanC[0]; a1 = veil * meanC[1]; a2 = veil * meanC[2]; }
    scat[k + 0] = (arena[k + 0] + a0) * inv;
    scat[k + 1] = (arena[k + 1] + a1) * inv;
    scat[k + 2] = (arena[k + 2] + a2) * inv;
}

// Scope measurement pass: bin the frame on a stride-2 grid. Integer atomics are
// order-independent, so the counts are identical on every backend.
//
// It reads the SCATTER plane: the density parade must measure the HALATED
// result. A scatter-blind scope is a bug parity CANNOT catch — every backend
// would agree on the same wrong parade (see the lookLinear header in the core).
__kernel void SpeakStatsKernel(SpeakParams p, int W, int H,
                               __global const float* src, __global const float* scat,
                               __global const float* bscat,
                               volatile __global uint* stats)
{
    int x = get_global_id(0) * 2, y = get_global_id(1) * 2;
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    if (p.scopeHD != 0) {
        int bin = expBinOf(pixelStops(p.inputColorSpace, src[i + 0], src[i + 1], src[i + 2]));
        atomic_inc(&stats[SPEAK_STATS_HIST_EXP + bin]);
    }
    if (p.scopeDensity != 0) {
        // Mirrors the core's `scat ? scat[j] : 0.0f`: the host builds the plane
        // under exactly this condition, so a false `hal` means the binding is a
        // placeholder and must not be dereferenced.
        bool hal = halActive(&p) || (p.viewMode == 3);
        float sR = 0.0f, sG = 0.0f, sB = 0.0f;
        if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
        float mr, mg, mb;
        lookLinear(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB,
                   vignGain(x, y, W, H, &p), &p, &mr, &mg, &mb);
        /* Grain and bloom are part of the result, so the parade measures
           them (G17's rule; bscat is only dereferenced when the host built it). */
        applyGrain(&mr, &mg, &mb, src[i + 3], x, y, H, &p);
        if (bloomActive(&p))
            bloomApplyPixel(&mr, &mg, &mb, bscat[j + 0], bscat[j + 1], bscat[j + 2], &p);
        int col = wfColOf(x, W);
        atomic_inc(&stats[wfIdx(0, col, wfRowOf(density10(mr)))]);
        atomic_inc(&stats[wfIdx(1, col, wfRowOf(density10(mg)))]);
        atomic_inc(&stats[wfIdx(2, col, wfRowOf(density10(mb)))]);
    }
}

__kernel void SpeakStatsMaxKernel(__global uint* stats)
{
    uint mx = 0u;
    for (int b = 0; b < SPEAK_EXP_BINS; ++b)
        if (stats[SPEAK_STATS_HIST_EXP + b] > mx) mx = stats[SPEAK_STATS_HIST_EXP + b];
    stats[SPEAK_STATS_HIST_MAX] = mx;
    uint wmx = 0u;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (stats[SPEAK_STATS_WF + k] > wmx) wmx = stats[SPEAK_STATS_WF + k];
    stats[SPEAK_STATS_WF_MAX] = wmx;
}

__kernel void SpeakKernel(SpeakParams p, int W, int H,
                          __global const float* src, __global const float* scat,
                          __global const float* bscat,
                          __global float* dst, __global const uint* stats,
                          int drawScopes)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    // Mirrors speakFrame's `sc ? sc[j] : 0.0f` — same gate as the host's, so the
    // placeholder binding used when halation is off is never dereferenced.
    bool hal = halActive(&p) || (p.viewMode == 3);
    float sR = 0.0f, sG = 0.0f, sB = 0.0f;
    if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
    // Same rule for the bloom plane: the host builds it only when bloom is
    // active (a bloom view with bloom off shows a zero delta on gray).
    bool blm = bloomActive(&p) || (p.viewMode == 5);
    float bR = 0.0f, bG = 0.0f, bB = 0.0f;
    if (blm && bloomActive(&p)) { bR = bscat[j + 0]; bG = bscat[j + 1]; bB = bscat[j + 2]; }
    float oR, oG, oB;
    processPixel(src[i + 0], src[i + 1], src[i + 2], src[i + 3], sR, sG, sB,
                 bR, bG, bB, x, y, W, H, &p, stats, drawScopes, &oR, &oG, &oB);
    dst[i + 0] = oR; dst[i + 1] = oG; dst[i + 2] = oB; dst[i + 3] = p.maskExternal ? 1.0f : src[i + 3];
}

// The LOOK's working-linear output (grain included) into arena level 0 — the
// field bloom scatters. Mirrors speakFrame's lookBuf pass; the main kernel
// RECOMPUTES the same values per pixel, so the two cannot disagree.
__kernel void SpeakLookKernel(SpeakParams p, int W, int H,
                              __global const float* src, __global const float* scat,
                              __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = ((size_t)y * W + x) * 4;
    size_t j = ((size_t)y * W + x) * 3;
    bool hal = halActive(&p) || (p.viewMode == 3);
    float sR = 0.0f, sG = 0.0f, sB = 0.0f;
    if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
    float lr, lg, lb;
    lookLinear(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB,
               vignGain(x, y, W, H, &p), &p, &lr, &lg, &lb);
    applyGrain(&lr, &lg, &lb, src[i + 3], x, y, H, &p);
    /* contain non-finite inputs to ONE pixel — see speak_core.h bloomFinite */
    arena[j + 0] = bloomFinite(lr); arena[j + 1] = bloomFinite(lg); arena[j + 2] = bloomFinite(lb);
}

// The veil's source: the frame mean, computed by ONE work-item over the
// coarsest level (geometry passed in as scalars, like the other per-level
// kernels). ~96 reads; DELIBERATELY atomics-free — Apple's OpenCL runtime
// miscompiles global int32 atomics, so no new pass may lean on them.
__kernel void SpeakBloomMeanKernel(int cw, int ch, int coff,
                                   __global const float* arena, __global float* meanC)
{
    if (get_global_id(0) != 0 || get_global_id(1) != 0) return;
    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
            size_t o = ((size_t)coff + (size_t)y * cw + x) * 3;
            m0 += arena[o + 0]; m1 += arena[o + 1]; m2 += arena[o + 2];
        }
    float invN = 1.0f / ((float)cw * (float)ch);
    meanC[0] = m0 * invN; meanC[1] = m1 * invN; meanC[2] = m2 * invN;
}

// The gate displaces the finished picture as one rigid sub-pixel move (all
// four channels: the matte rides with the pixels it describes). dx/dy are
// computed HOST-side by the same speak_core closed form — frame-uniform.
__kernel void SpeakWeaveKernel(int W, int H, float dx, float dy,
                               __global const float* pre, __global float* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float out4[4];
    weaveSamplePixel(pre, W, H, (float)x - dx, (float)y - dy, out4);
    size_t i = ((size_t)y * W + x) * 4;
    dst[i + 0] = out4[0]; dst[i + 1] = out4[1]; dst[i + 2] = out4[2]; dst[i + 3] = out4[3];
}

// Scopes on top of the displaced picture — panel chrome does not weave.
__kernel void SpeakScopeOverlayKernel(SpeakParams p, int W, int H,
                                      __global const uint* stats, __global float* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float sr, sg, sb;
    size_t i = ((size_t)y * W + x) * 4;
    if (hdScopePixel(x, y, W, H, &p, stats, &sr, &sg, &sb)) {
        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
    }
    if (densityScopePixel(x, y, W, H, &p, stats, &sr, &sg, &sb)) {
        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
    }
    if (statusStripPixel(x, y, W, H, &p, &sr, &sg, &sb)) {
        dst[i + 0] = sr; dst[i + 1] = sg; dst[i + 2] = sb;
    }
}
)CLC";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

void SpeakCheck(cl_int e, const char* m) { if (e != CL_SUCCESS) fprintf(stderr, "Speak: %s [%d]\n", m, e); }

class SpeakLocker
{
public:
#ifdef _WIN64
    SpeakLocker() { InitializeCriticalSection(&m); }
    ~SpeakLocker() { DeleteCriticalSection(&m); }
    void Lock() { EnterCriticalSection(&m); }
    void Unlock() { LeaveCriticalSection(&m); }
    CRITICAL_SECTION m;
#else
    SpeakLocker() { pthread_mutex_init(&m, NULL); }
    ~SpeakLocker() { pthread_mutex_destroy(&m); }
    void Lock() { pthread_mutex_lock(&m); }
    void Unlock() { pthread_mutex_unlock(&m); }
    pthread_mutex_t m;
#endif
};

// Host-side mirrors of speak_core.h's pyramid geometry — the dispatch sizes and
// the buffer sizes MUST be the same arithmetic the kernels use, so these are
// textual ports of halLevelCount/halLevelInfo/halArenaPixels/halAmountOf/
// halActive, exactly like the CLC copies above.
int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    const int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);   // offset just past the last level
    return off;
}
float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}
bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- bloom / weave host mirrors of speak_core.h (kept textually parallel,
// same set as the Metal host's). The fall rates ride to the kernels as
// ARGUMENTS — the in-string constants were deleted on purpose (a constant
// nothing reads is a knob that cannot fire).
static const float kHostHalCoreFall    = 3.0f;   // == kHalCoreFall in the core
static const float kHostHalSkirtFall   = 1.0f;
static const float kHostBloomSkirtFall = 0.5f;

float hostClampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
float bloomAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? hostClampf(pr.profile.bloomAmount, 0.0f, 1.0f) : 0.0f;
}
bool bloomActive(const SpeakParams& pr)
{
    return (pr.strength > 0.0f) && (bloomAmountOf(pr) > 0.0f);
}
// ---- gate weave, host side: the displacement is frame-uniform, so it is
// computed here with speak_core.h's exact closed form and passed to the
// kernel as scalars (textually parallel copies of weaveDisp and its parts).
bool weaveActive(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) && (pr.strength > 0.0f)
        && (pr.profile.weaveAmount > 0.0f)
        && (pr.viewMode == SPEAK_VIEW_RESULT);
}
float hostGrainHash(uint32_t ix, uint32_t iy, uint32_t f, uint32_t ch)
{
    uint32_t h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (static_cast<float>(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}
float hostWeaveSmooth1D(float t, uint32_t salt)
{
    const float tf = std::floor(t);
    const int i0 = static_cast<int>(tf);
    const float fr = t - tf;
    const float sm = fr * fr * (3.0f - 2.0f * fr);
    const float n0 = hostGrainHash(static_cast<uint32_t>(i0),     0x5EA7u, 0u, salt);
    const float n1 = hostGrainHash(static_cast<uint32_t>(i0 + 1), 0x5EA7u, 0u, salt);
    return n0 + (n1 - n0) * sm;
}
void hostWeaveDisp(const SpeakParams& pr, int H, float& dx, float& dy)
{
    dx = 0.0f; dy = 0.0f;
    if (!weaveActive(pr)) return;
    const float amp = pr.profile.weaveAmount * 0.01f * static_cast<float>(H)
                    * hostClampf(pr.strength, 0.0f, 1.0f);
    const float speed = pr.profile.weaveSpeed > 0.0f ? pr.profile.weaveSpeed : 1.0f;
    const float t = static_cast<float>(pr.frameIndex) * speed;
    float sx = 0.0f, sy = 0.0f, norm = 0.0f;
    for (int o = 0; o < 6; ++o) {                       // kWeaveOctaves
        const float period = static_cast<float>(1 << (o + 1));
        const float a = period;
        sx += a * hostWeaveSmooth1D(t / period, static_cast<uint32_t>(2 * o));
        sy += a * hostWeaveSmooth1D(t / period, static_cast<uint32_t>(2 * o + 1));
        norm += a;
    }
    dx = amp * sx / norm;
    dy = amp * 1.4f * sy / norm;
}

struct SpeakRes {
    cl_kernel k = NULL;         // main
    cl_kernel kStats = NULL;    // scope measurement pass
    cl_kernel kMax = NULL;      // bin-max finalize
    cl_kernel kExcess = NULL;   // scatter pyramid: level 0 highlight excess
    cl_kernel kDecimate = NULL; // scatter pyramid: level L from level L-1
    cl_kernel kAccum = NULL;    // scatter pyramid: coarse-to-fine accumulate, in place
    cl_kernel kNorm = NULL;     // scatter pyramid: normalize level 0 into the plane
    cl_kernel kLook = NULL;     // bloom: the look field (grain included) into arena level 0
    cl_kernel kBloomMean = NULL;// bloom: coarsest-level mean (the veil's source)
    cl_kernel kWeave = NULL;    // gate weave: Catmull-Rom resample of the pre-copy
    cl_kernel kScopeOverlay = NULL; // scopes after the weave, pinned
    cl_mem stats = NULL;
    // The veil mean: 4 floats, fixed size like `stats`, allocated once and
    // zero-filled — the halation normalize binds it with veil = 0, and
    // 0 * (uninitialized NaN) would poison the whole scatter plane.
    cl_mem mean = NULL;
    // Unlike `stats` (fixed size, allocated once), these two are SIZE-DEPENDENT:
    // the OFX host hands us any image size and it changes between renders (proxy
    // vs full res, different clips). Cache the allocated size and grow when it is
    // no longer enough — otherwise the first proxy->full-res switch overruns.
    // Growth-only: an oversized buffer is harmless (every offset is recomputed
    // from W,H on both sides), and it avoids reallocating on every size wobble.
    cl_mem arena = NULL; size_t arenaBytes = 0;
    cl_mem scat = NULL;  size_t scatBytes = 0;
    cl_mem bscat = NULL; size_t bscatBytes = 0;  // bloom scatter (LOOK-referred)
    cl_mem pre = NULL;   size_t preBytes = 0;    // weave: the pre-displacement picture
};

} // namespace

void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                    const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    cl_int error;
    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    static std::map<void*, SpeakRes> s_res;
    static SpeakLocker s_lock;

    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    SpeakCheck(error, "get context");

    SpeakParams params = p_Params;
    int W = p_Width, H = p_Height;

    // Mirror speak_core.h's speakFrame gate. When halation is inactive the whole
    // three-pass scatter chain is skipped and the buffers stay at their minimum
    // placeholder size — the identity path stays bit-exact AND free. The kernels
    // gate their scatter reads on this same condition, so the placeholder
    // binding is never dereferenced (a NULL binding would crash, so we always
    // bind something valid).
    const bool wantHal = halActive(params) || params.viewMode == SPEAK_VIEW_SCATTER;
    const bool wantBloom = bloomActive(params);
    const bool wantWeave = weaveActive(params);
    // bloom reuses the ARENA sequentially (halation's content is dead after
    // its normalize), so one arena serves both pyramids.
    const size_t arenaNeed = (wantHal || wantBloom)
        ? static_cast<size_t>(halArenaPixels(W, H)) * 3 * sizeof(cl_float) : sizeof(cl_float);
    const size_t scatNeed = wantHal
        ? static_cast<size_t>(W) * H * 3 * sizeof(cl_float) : sizeof(cl_float);
    const size_t bscatNeed = wantBloom
        ? static_cast<size_t>(W) * H * 3 * sizeof(cl_float) : sizeof(cl_float);
    const size_t preNeed = wantWeave
        ? static_cast<size_t>(W) * H * 4 * sizeof(cl_float) : sizeof(cl_float);

    SpeakRes res;
    s_lock.Lock();
    {
        SpeakRes& r = s_res[p_CmdQ];
        if (!r.k) {
            cl_program program = clCreateProgramWithSource(clContext, 1, &kSpeakKernelSource, NULL, &error);
            SpeakCheck(error, "create program");
            error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (error != CL_SUCCESS) {
                cl_device_id dev = NULL;
                clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &dev, NULL);
                char log[16384] = { 0 };
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
                fprintf(stderr, "Speak OpenCL build log:\n%s\n", log);
                s_lock.Unlock();
                return;
            }
            r.k = clCreateKernel(program, "SpeakKernel", &error);
            SpeakCheck(error, "create kernel");
            r.kStats = clCreateKernel(program, "SpeakStatsKernel", &error);
            SpeakCheck(error, "create stats kernel");
            r.kMax = clCreateKernel(program, "SpeakStatsMaxKernel", &error);
            SpeakCheck(error, "create stats-max kernel");
            r.kExcess = clCreateKernel(program, "SpeakExcessKernel", &error);
            SpeakCheck(error, "create excess kernel");
            r.kDecimate = clCreateKernel(program, "SpeakDecimateKernel", &error);
            SpeakCheck(error, "create decimate kernel");
            r.kAccum = clCreateKernel(program, "SpeakAccumKernel", &error);
            SpeakCheck(error, "create accum kernel");
            r.kNorm = clCreateKernel(program, "SpeakNormalizeKernel", &error);
            SpeakCheck(error, "create normalize kernel");
            r.kLook = clCreateKernel(program, "SpeakLookKernel", &error);
            SpeakCheck(error, "create look kernel");
            r.kBloomMean = clCreateKernel(program, "SpeakBloomMeanKernel", &error);
            SpeakCheck(error, "create bloom-mean kernel");
            r.kWeave = clCreateKernel(program, "SpeakWeaveKernel", &error);
            SpeakCheck(error, "create weave kernel");
            r.kScopeOverlay = clCreateKernel(program, "SpeakScopeOverlayKernel", &error);
            SpeakCheck(error, "create scope-overlay kernel");
        }
        if (!r.stats)
            r.stats = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                     SPEAK_STATS_UINTS * sizeof(cl_uint), NULL, &error);
        if (!r.mean) {
            r.mean = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                    4 * sizeof(cl_float), NULL, &error);
            SpeakCheck(error, "create mean buffer");
            // Zero once at birth: the halation normalize reads it under a
            // veil of 0, and 0 * (uninitialized NaN) would still be NaN.
            const cl_float fz = 0.0f;
            clEnqueueFillBuffer(cmdQ, r.mean, &fz, sizeof(cl_float), 0,
                                4 * sizeof(cl_float), 0, NULL, NULL);
        }
        if (!r.arena || r.arenaBytes < arenaNeed) {
            if (r.arena) clReleaseMemObject(r.arena);
            r.arena = clCreateBuffer(clContext, CL_MEM_READ_WRITE, arenaNeed, NULL, &error);
            SpeakCheck(error, "create arena buffer");
            r.arenaBytes = (error == CL_SUCCESS) ? arenaNeed : 0;
        }
        if (!r.scat || r.scatBytes < scatNeed) {
            if (r.scat) clReleaseMemObject(r.scat);
            r.scat = clCreateBuffer(clContext, CL_MEM_READ_WRITE, scatNeed, NULL, &error);
            SpeakCheck(error, "create scatter buffer");
            r.scatBytes = (error == CL_SUCCESS) ? scatNeed : 0;
        }
        if (!r.bscat || r.bscatBytes < bscatNeed) {
            if (r.bscat) clReleaseMemObject(r.bscat);
            r.bscat = clCreateBuffer(clContext, CL_MEM_READ_WRITE, bscatNeed, NULL, &error);
            SpeakCheck(error, "create bloom scatter buffer");
            r.bscatBytes = (error == CL_SUCCESS) ? bscatNeed : 0;
        }
        if (!r.pre || r.preBytes < preNeed) {
            if (r.pre) clReleaseMemObject(r.pre);
            r.pre = clCreateBuffer(clContext, CL_MEM_READ_WRITE, preNeed, NULL, &error);
            SpeakCheck(error, "create weave pre buffer");
            r.preBytes = (error == CL_SUCCESS) ? preNeed : 0;
        }
        res = r;
    }
    s_lock.Unlock();

    cl_mem src = reinterpret_cast<cl_mem>(const_cast<float*>(p_Src));
    cl_mem dst = reinterpret_cast<cl_mem>(p_Dst);

    const size_t local[2] = { 16, 16 };
    const size_t globalF[2] = { static_cast<size_t>((W + 15) / 16) * 16,
                                static_cast<size_t>((H + 15) / 16) * 16 };

    // ---- the scatter pyramid ----
    //   excess -> decimate(L=1..nLev-1) -> accum(L=nLev-1..0) -> normalize
    // Every pass depends on the previous one, and the queue is NOT ours (Resolve
    // hands it in), so we cannot assume it was created in-order. A barrier costs
    // nothing on an in-order queue and is what makes an out-of-order one correct.
    if (wantHal) {
        const int nLev = halLevelCount(W, H);
        int c = 0;
        error  = clSetKernelArg(res.kExcess, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(cl_mem), &src);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(cl_mem), &res.arena);
        SpeakCheck(error, "set excess args");
        error = clEnqueueNDRangeKernel(cmdQ, res.kExcess, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue excess");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        for (int L = 1; L < nLev; ++L) {
            int sw, sh, so, dw, dh, doff;
            halLevelInfo(W, H, L - 1, sw, sh, so);
            halLevelInfo(W, H, L,     dw, dh, doff);
            int c2 = 0;
            error  = clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &so);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &doff);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set decimate args");
            const size_t globalL[2] = { static_cast<size_t>((dw + 15) / 16) * 16,
                                        static_cast<size_t>((dh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kDecimate, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue decimate");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        // Coarse-to-fine accumulate, in place, one octave at a time — the same
        // loop buildHalScatter runs, with level L's pixel loop become the grid.
        // The barrier after each level is REQUIRED, not defensive: level L reads
        // level L+1, which the previous dispatch wrote. Halation passes its own
        // fall rates verbatim (isBloom 0), so its arithmetic is bit-identical to
        // the pre-bloom build.
        const int isBloom0 = 0;
        for (int L = nLev - 1; L >= 0; --L) {
            int lw, lh, off, cw, ch, coff;
            halLevelInfo(W, H, L,     lw, lh, off);
            halLevelInfo(W, H, L + 1, cw, ch, coff);
            int c2 = 0;
            error  = clSetKernelArg(res.kAccum, c2++, sizeof(SpeakParams), &params);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &H);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &L);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &nLev);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(float), &kHostHalCoreFall);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(float), &kHostHalSkirtFall);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &isBloom0);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lh);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &off);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &cw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &ch);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &coff);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set accum args");
            const size_t globalL[2] = { static_cast<size_t>((lw + 15) / 16) * 16,
                                        static_cast<size_t>((lh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kAccum, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue accum");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        c = 0;
        error  = clSetKernelArg(res.kNorm, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(float), &kHostHalCoreFall);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(float), &kHostHalSkirtFall);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &isBloom0);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.arena);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.scat);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.mean);   // veil 0: never used
        SpeakCheck(error, "set normalize args");
        error = clEnqueueNDRangeKernel(cmdQ, res.kNorm, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue normalize");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    }

    // ---- bloom chain: look field -> pyramid on the arena -> mean -> normalize.
    // The parade measures bloom, so this must complete BEFORE the stats pass
    // (the same ordering rule the halation chain established). The arena is
    // reused: halation's content is dead once its normalize has run.
    if (wantBloom) {
        const int nLev = halLevelCount(W, H);
        int c = 0;
        error  = clSetKernelArg(res.kLook, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kLook, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kLook, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kLook, c++, sizeof(cl_mem), &src);
        error |= clSetKernelArg(res.kLook, c++, sizeof(cl_mem), &res.scat);
        error |= clSetKernelArg(res.kLook, c++, sizeof(cl_mem), &res.arena);
        SpeakCheck(error, "set look args");
        error = clEnqueueNDRangeKernel(cmdQ, res.kLook, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue look");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        for (int L = 1; L < nLev; ++L) {
            int sw, sh, so, dw, dh, doff;
            halLevelInfo(W, H, L - 1, sw, sh, so);
            halLevelInfo(W, H, L,     dw, dh, doff);
            int c2 = 0;
            error  = clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &so);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &doff);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set decimate args (bloom)");
            const size_t globalL[2] = { static_cast<size_t>((dw + 15) / 16) * 16,
                                        static_cast<size_t>((dh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kDecimate, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue decimate (bloom)");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        // The frame mean BEFORE the in-place accumulate overwrites the levels
        // (ONE work-item, ~96 reads, no atomics anywhere near it).
        {
            int cw, ch, coff;
            halLevelInfo(W, H, nLev - 1, cw, ch, coff);
            int c2 = 0;
            error  = clSetKernelArg(res.kBloomMean, c2++, sizeof(int), &cw);
            error |= clSetKernelArg(res.kBloomMean, c2++, sizeof(int), &ch);
            error |= clSetKernelArg(res.kBloomMean, c2++, sizeof(int), &coff);
            error |= clSetKernelArg(res.kBloomMean, c2++, sizeof(cl_mem), &res.arena);
            error |= clSetKernelArg(res.kBloomMean, c2++, sizeof(cl_mem), &res.mean);
            SpeakCheck(error, "set bloom-mean args");
            const size_t one[2] = { 1, 1 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kBloomMean, 2, NULL, one, NULL, 0, NULL, NULL);
            SpeakCheck(error, "enqueue bloom-mean");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        const int isBloom1 = 1;
        for (int L = nLev - 1; L >= 0; --L) {
            int lw, lh, off, cw, ch, coff;
            halLevelInfo(W, H, L,     lw, lh, off);
            halLevelInfo(W, H, L + 1, cw, ch, coff);
            int c2 = 0;
            error  = clSetKernelArg(res.kAccum, c2++, sizeof(SpeakParams), &params);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &H);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &L);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &nLev);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(float), &kHostHalCoreFall);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(float), &kHostBloomSkirtFall);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &isBloom1);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lh);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &off);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &cw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &ch);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &coff);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set accum args (bloom)");
            const size_t globalL[2] = { static_cast<size_t>((lw + 15) / 16) * 16,
                                        static_cast<size_t>((lh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kAccum, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue accum (bloom)");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        c = 0;
        error  = clSetKernelArg(res.kNorm, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(float), &kHostHalCoreFall);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(float), &kHostBloomSkirtFall);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &isBloom1);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.arena);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.bscat);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.mean);
        SpeakCheck(error, "set normalize args (bloom)");
        error = clEnqueueNDRangeKernel(cmdQ, res.kNorm, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue normalize (bloom)");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    }

    // Zero the stats, then measure the frame only when a scope is showing it.
    // This runs AFTER the scatter pass because the density parade measures it.
    const cl_uint zero = 0;
    clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint), 0,
                        SPEAK_STATS_UINTS * sizeof(cl_uint), 0, NULL, NULL);
    clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    if (params.scopeHD != 0 || params.scopeDensity != 0) {
        int c = 0;
        error  = clSetKernelArg(res.kStats, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kStats, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kStats, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &src);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &res.scat);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &res.bscat);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &res.stats);
        SpeakCheck(error, "set stats args");
        const size_t localH[2]  = { 16, 16 };
        // ceil(W/2) sample threads, not floor(W/2): the kernel indexes x = gid*2
        // and the CPU reference loops x = 0,2,..<W, so an odd W needs (W+1)/2
        // columns. With floor(W/2) this fell exactly one sample column short
        // whenever floor(W/2) landed on a multiple of the 16-wide group — i.e.
        // W = 1 (mod 32), which includes 1921 and 3841. Scope-only, but it made
        // the parade and histogram silently miss the frame's last sample column
        // at those sizes. (Metal's dispatch over-covers and was never affected.)
        const size_t globalH[2] = { static_cast<size_t>(((p_Width  + 1) / 2 + 15) / 16) * 16,
                                    static_cast<size_t>(((p_Height + 1) / 2 + 15) / 16) * 16 };
        error = clEnqueueNDRangeKernel(cmdQ, res.kStats, 2, NULL, globalH, localH, 0, NULL, NULL);
        SpeakCheck(error, "enqueue stats");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        error = clSetKernelArg(res.kMax, 0, sizeof(cl_mem), &res.stats);
        const size_t one[2] = { 1, 1 };
        error |= clEnqueueNDRangeKernel(cmdQ, res.kMax, 2, NULL, one, NULL, 0, NULL, NULL);
        SpeakCheck(error, "enqueue stats-max");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    }

    // When the gate-weave pass is live the main kernel suppresses the scopes;
    // they are overlaid AFTER the displacement (panel chrome does not weave).
    const bool weave = weaveActive(params);
    const int drawScopes = weave ? 0 : 1;

    int c = 0;
    error  = clSetKernelArg(res.k, c++, sizeof(SpeakParams), &params);
    error |= clSetKernelArg(res.k, c++, sizeof(int), &W);
    error |= clSetKernelArg(res.k, c++, sizeof(int), &H);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &src);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &res.scat);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &res.bscat);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &dst);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &res.stats);
    error |= clSetKernelArg(res.k, c++, sizeof(int), &drawScopes);
    SpeakCheck(error, "set args");

    const size_t global[2] = { static_cast<size_t>((p_Width + 15) / 16) * 16,
                               static_cast<size_t>((p_Height + 15) / 16) * 16 };
    error = clEnqueueNDRangeKernel(cmdQ, res.k, 2, NULL, global, local, 0, NULL, NULL);
    SpeakCheck(error, "enqueue");

    if (weave) {
        // The gate displaces the finished picture — grain, bloom and all — as
        // one rigid sub-pixel move: copy dst aside, resample it back through
        // Catmull-Rom, then pin the scopes on top (mirrors speakFrame).
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        error = clEnqueueCopyBuffer(cmdQ, dst, res.pre, 0, 0,
                                    static_cast<size_t>(W) * H * 4 * sizeof(cl_float),
                                    0, NULL, NULL);
        SpeakCheck(error, "copy pre-weave");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        float wdx, wdy;
        hostWeaveDisp(params, H, wdx, wdy);
        int c2 = 0;
        error  = clSetKernelArg(res.kWeave, c2++, sizeof(int), &W);
        error |= clSetKernelArg(res.kWeave, c2++, sizeof(int), &H);
        error |= clSetKernelArg(res.kWeave, c2++, sizeof(float), &wdx);
        error |= clSetKernelArg(res.kWeave, c2++, sizeof(float), &wdy);
        error |= clSetKernelArg(res.kWeave, c2++, sizeof(cl_mem), &res.pre);
        error |= clSetKernelArg(res.kWeave, c2++, sizeof(cl_mem), &dst);
        SpeakCheck(error, "set weave args");
        error = clEnqueueNDRangeKernel(cmdQ, res.kWeave, 2, NULL, global, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue weave");

        if (params.scopeHD != 0 || params.scopeDensity != 0 ||
            params.statusStrip != 0) {
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
            int c3 = 0;
            error  = clSetKernelArg(res.kScopeOverlay, c3++, sizeof(SpeakParams), &params);
            error |= clSetKernelArg(res.kScopeOverlay, c3++, sizeof(int), &W);
            error |= clSetKernelArg(res.kScopeOverlay, c3++, sizeof(int), &H);
            error |= clSetKernelArg(res.kScopeOverlay, c3++, sizeof(cl_mem), &res.stats);
            error |= clSetKernelArg(res.kScopeOverlay, c3++, sizeof(cl_mem), &dst);
            SpeakCheck(error, "set scope-overlay args");
            error = clEnqueueNDRangeKernel(cmdQ, res.kScopeOverlay, 2, NULL, global, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue scope overlay");
        }
    }
}

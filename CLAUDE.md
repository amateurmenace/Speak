# Speak — project guide

Free, MIT-licensed film-emulation OpenFX plugin for DaVinci Resolve (works in
the free edition). The film-reconstruction counterpart to Hush
(github.com/amateurmenace/Hush-OpenNR): Hush quiets the noise at the FIRST node;
Speak gives the image its voice at the LAST. Original published color science —
H&D sensitometry, printer points, subtractive dye behaviour, Selwyn-family
granularity — never cloned profiles or trademarked stock names.

## The cardinal rule

`plugin/speak_core.h` is the **single source of truth** for the algorithm. The
three GPU kernels (`SpeakMetalKernel.mm`, `SpeakCudaKernel.cu`,
`SpeakOpenCLKernel.cpp`) are line-by-line ports of it. **Any change to the math
must be applied to all four files** and verified with the parity tests (~2e-5
mean). Constants, formulas, loop order — keep them textually parallel.

Shared plumbing lives in `plugin/SpeakParams.h`: SpeakProfile + SpeakParams
(ALL fields 4 bytes; re-declared inside the Metal and OpenCL kernel strings —
a field added in one place must be added in all declarations AND the layout
gate G1 in test/test_speak.cpp), and the scope-stats slot layout.

## House discipline (inherited from Hush, hard-won here)

- **Gate first, build second.** Every look module declares a control arm — the
  cheap baseline it must beat and the measurement — and is prototyped CPU-first.
  The arm has overturned design reasoning six times in this repo's history.
- **A gate must be able to FAIL.** If a gate is credited with catching X,
  sabotage X and watch it go red — otherwise the credit is fiction.
- **Render the thing and look at it.** The halation pyramid once shipped a
  rectangular skirt past every numeric gate; one render caught it (G18 now
  measures isotropy).
- **Transparency is the product.** Every claim about the image must be showable
  on screen (H&D scope, density parade, isolated scatter/grain views).
- **Honesty in the UI.** Never a control that doesn't do what its hint says;
  modelled defaults are labelled as modelled defaults, never as measurements.
- **Originality is a hard gate.** Published sensitometry or user calibration
  only. Never cite a different physical system's literature for authority
  (e.g. the eye's glare-spread function is NOT film halation).
- Deterministic, real-time UHD, **identity at default**, alpha passes through.

## Build & test (run all of this before calling any change done)

```sh
cd test
# CPU gates: layout, CST round-trip, identity, monotone, gray pivot, scope==kernel,
# halation (pyramid ladder/energy/skirt/isotropy/resolution), grain (G19-G24)
c++ -O2 -std=c++14 -I../plugin test_speak.cpp -o test_speak && ./test_speak
# Control arms
c++ -O2 -std=c++14 -I../plugin test_speak_macbeth.cpp -o test_speak_macbeth && ./test_speak_macbeth
c++ -O2 -std=c++14 -I../plugin test_speak_split.cpp -o test_speak_split && ./test_speak_split
c++ -O2 -std=c++14 -I../plugin proto_halation.cpp -o proto_halation && ./proto_halation
# GPU parity — the REAL entry points vs the CPU reference
c++ -O2 -std=c++14 -I../plugin test_speak_metal.mm ../plugin/SpeakMetalKernel.mm \
    -framework Metal -framework Foundation -o test_speak_metal && ./test_speak_metal
c++ -O2 -std=c++14 -I../plugin test_speak_opencl.cpp ../plugin/SpeakOpenCLKernel.cpp \
    -framework OpenCL -o test_speak_opencl && ./test_speak_opencl
# Offline kernel compiles (Resolve compiles them at load)
awk '/^static const char\* kSpeakKernelSource = R"MSL\(/{flag=1;next} /^\)MSL";/{flag=0} flag' \
    ../plugin/SpeakMetalKernel.mm > /tmp/speak.metal && xcrun -sdk macosx metal -c /tmp/speak.metal -o /tmp/speak.air
c++ -O2 -std=c++14 compile_opencl.cpp -framework OpenCL -o compile_opencl
awk '/R"CLC\(/{flag=1;next} /^\)CLC";/{flag=0} flag' ../plugin/SpeakOpenCLKernel.cpp > /tmp/speak.cl \
    && ./compile_opencl /tmp/speak.cl
# CUDA syntax (no toolkit needed; CUDA is hardware-unverified like Hush's)
./check_cuda_syntax.sh
# Describe smoke test — a describe crash is INVISIBLE in Resolve
c++ -O2 -std=c++14 -I../ofx/include test_describe.cpp -o test_describe
./test_describe ../plugin/Speak.ofx.bundle/Contents/MacOS/Speak.ofx   # expect 1 plugin
# Build the bundle
cd ../plugin && make
```

Known platform trap: Apple's DEPRECATED OpenCL runtime miscompiles global int32
atomics — test_speak_opencl probes it and loudly SKIPS the scope cases on such
devices. The scatter/grain paths are atomics-free by design. macOS production
renders via Metal; OpenCL's real targets are Win/Linux NVIDIA/AMD/Intel.

## Release

`./build_release.sh` builds, signs (Developer ID auto-detected), and packages
`release/Speak-<ver>-macOS.{pkg,zip}`. Version bump checklist (all four):
`plugin/SpeakPlugin.cpp` (kSpeakVersionMinor + description), `plugin/Info.plist`
(both strings), `build_release.sh` (VERSION=), `CHANGELOG.md`.

Releases are published BOTH here and on the shared hub
`amateurmenace/control-z` with tags `speak-vX.Y.Z` (Hush uses `hush-vX.Y.Z`).

## The Hush handoff

Hush ≥ 3.7's "Export Clean Matte to Alpha" writes `clamp((effN−1)/6, 0, 1)`
into alpha in its Result view. Speak's grain reads it via "Use Incoming Matte
(Alpha)": full grain where the matte is 1 (denoiser averaged deep — texture was
flattened), the Matte Floor where 0 (protected motion — real noise survives).
The calibration lives in Hush; Speak consumes [0,1] as-is and passes alpha
through untouched. Speak stays fully useful standalone (Tier 0).

## Repo layout

- `plugin/` — speak_core.h (reference), SpeakParams.h, SpeakPlugin.{h,cpp},
  SpeakOfxMain.cpp, 3 GPU kernels, Makefile, Info.plist
- `ofx/` — vendored OpenFX 1.4 headers + Support wrapper (BSD, don't edit)
- `test/` — CPU gates, control arms, GPU parity, describe harness, dev renderer
- `installer/` — double-click install scripts

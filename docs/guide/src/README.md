# Visual guide — how it's built

`../index.html` is the self-contained Hush + Speak visual guide. It is authored
here, baked to one HTML file (data-URI images inline), then:

- committed at `docs/guide/index.html` (the repo copy);
- exported to `docs/guide/hush-speak-guide.pdf` (Chrome headless
  `--print-to-pdf`, attached to releases);
- carried verbatim into the control-z site by `site/build.py` (→ `guide.html`),
  linked from both the Hush and Speak tool cards.

## Regenerate

```sh
cd docs/guide/src
# the conceptual illustrations are rendered from the CPU reference, so they
# can never drift from the shipping look:
c++ -O2 -std=c++14 -I../../../plugin guide_assets.cpp -o guide_assets && ./guide_assets
c++ -O2 -std=c++14 -I../../../plugin matte_demo.cpp   -o matte_demo   && ./matte_demo
# (matte_demo's PNGs get a labeled contrast stretch — see the one-liner in the
#  guide caption; the grain-isolated view is subtle by design.)
python3 build_guide.py                       # -> ../index.html
# PDF:
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="../hush-speak-guide.pdf" "file://$PWD/../index.html"
```

## The two remaining screenshots — SHOT LIST for Stephen

Everything conceptual is rendered from the plugin itself. Two captures live in
Resolve's own UI and can't be scripted; the guide has labeled slots for them.
Take these once, on the color page, with a Hush→Speak node graph set up:

1. **Color page — the blue key wire.** The node editor showing a Hush node and
   a Speak node with the **blue key wire** dragged from Hush's key output to
   Speak's key input. Frame both nodes and the wire. (Guide §4, left slot.)

2. **OFX Alpha → Enable.** The right-click context menu on the **Hush** node,
   open, with **OFX Alpha → Enable** visible/highlighted. (Guide §4, right slot.)

Optional, nice-to-have (would replace the synthetic renders with real footage):

3. Speak's **View → Setup Guide** on a real clip (proves the in-plugin card).
4. A real before/after on graded footage with grain + halation, if you have a
   shot with a practical/highlight in it.

Drop the PNGs in this folder, swap the two `SHOT(...)` slots in `build_guide.py`
for `FIG(datauri("..."), "...")`, and rebuild.

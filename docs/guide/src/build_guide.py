#!/usr/bin/env python3
"""Bake the Hush & Speak visual guide -> Speak/docs/guide/index.html.

Self-contained (data-URI images, inline CSS in the control-z house palette),
so the same file serves the repo, the control-z site (carried like the Hush
whitepaper), and the PDF export (Chrome headless --print-to-pdf).
"""
import base64, pathlib, sys

# Assets live beside this script (docs/guide/src/); the baked page is one level
# up (docs/guide/index.html). Regenerate: rebuild the two .cpp render tools
# against ../../../plugin, then `python3 build_guide.py`.
SCRATCH = pathlib.Path(__file__).parent
OUT = SCRATCH.parent / "index.html"

def datauri(name):
    p = SCRATCH / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

IMG = {k: datauri(k + ".png") for k in
       ["g_input", "g_look", "g_grain", "g_halation", "m_shaped", "m_uniform"]}

# ---- the node-graph diagram, hand-drawn SVG (cannot drift from a screenshot) ----
NODE_SVG = '''
<svg viewBox="0 0 742 220" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Node graph: Hush at the first node, your grade in the middle, Speak at the last, with a blue key wire from Hush to Speak.">
  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8a8a98"/>
    </marker>
    <marker id="arb" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#4d8ce0"/>
    </marker>
  </defs>
  <!-- RGB spine -->
  <line x1="88" y1="70" x2="150" y2="70" stroke="#8a8a98" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="270" y1="70" x2="332" y2="70" stroke="#8a8a98" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="452" y1="70" x2="514" y2="70" stroke="#8a8a98" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="634" y1="70" x2="700" y2="70" stroke="#8a8a98" stroke-width="2" marker-end="url(#ar)"/>
  <!-- IO -->
  <text x="44" y="74" fill="#8a8a98" font="700 11px monospace" font-family="monospace" font-size="11" text-anchor="middle">IN</text>
  <text x="722" y="74" fill="#8a8a98" font-family="monospace" font-size="11" text-anchor="middle">OUT</text>
  <!-- Hush -->
  <rect x="150" y="46" width="120" height="48" rx="8" fill="#20281f" stroke="#5B8C65" stroke-width="1.6"/>
  <text x="210" y="66" fill="#8FBF9A" font-family="'Space Grotesk',sans-serif" font-weight="700" font-size="14" text-anchor="middle">Hush</text>
  <text x="210" y="82" fill="#7f9a86" font-family="monospace" font-size="9" text-anchor="middle">1 · clean early</text>
  <!-- grade -->
  <rect x="332" y="46" width="120" height="48" rx="8" fill="#26262f" stroke="#45454f" stroke-width="1.6"/>
  <text x="392" y="66" fill="#c9c9d4" font-family="'Space Grotesk',sans-serif" font-weight="700" font-size="13" text-anchor="middle">your grade</text>
  <text x="392" y="82" fill="#8a8a98" font-family="monospace" font-size="9" text-anchor="middle">2 · shape it</text>
  <!-- Speak -->
  <rect x="514" y="46" width="120" height="48" rx="8" fill="#2a2417" stroke="#C99A3A" stroke-width="1.6"/>
  <text x="574" y="66" fill="#E4BA6A" font-family="'Space Grotesk',sans-serif" font-weight="700" font-size="14" text-anchor="middle">Speak</text>
  <text x="574" y="82" fill="#b39a63" font-family="monospace" font-size="9" text-anchor="middle">3 · reconstruct late</text>
  <!-- blue key wire: Hush bottom -> down -> across -> up to Speak key -->
  <path d="M210,94 L210,160 L574,160 L574,94" fill="none" stroke="#4d8ce0" stroke-width="2.4" marker-end="url(#arb)"/>
  <rect x="300" y="150" width="184" height="20" rx="10" fill="#12233a" stroke="#4d8ce0" stroke-width="1"/>
  <text x="392" y="164" fill="#7fb0ee" font-family="monospace" font-size="10.5" text-anchor="middle">KEY — the blue wire (clean-confidence matte)</text>
</svg>'''

FIG = lambda src, cap: f'''<figure class="fig">
  <img src="{src}" alt="{cap}" loading="lazy">
  <figcaption>{cap}</figcaption></figure>'''

SHOT = lambda label, what: f'''<figure class="fig shot">
  <div class="shotph"><span>SCREENSHOT</span><b>{label}</b></div>
  <figcaption>{what}</figcaption></figure>'''

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hush &amp; Speak — clean early, reconstruct late</title>
<meta name="description" content="How Hush and Speak work together in DaVinci Resolve: quiet the noise at the first node, give the image its voice at the last. The blue-wire matte handoff, the three setups, and the one rule.">
<style>
  :root{{
    --cream:#F5F3EE;--card:#fff;--border:#E0DED9;--green:#3D5A47;--green-br:#5B8C65;
    --amber:#E5A835;--text-med:#5A6B5E;--text-lt:#8A9A8E;--dark:#20281f;
    --blue:#4d8ce0;--mono:"SF Mono",ui-monospace,Menlo,monospace;--r:14px;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--cream);color:var(--green);font:17px/1.65 "DM Sans",-apple-system,system-ui,sans-serif;}}
  .wrap{{max-width:860px;margin:0 auto;padding:0 26px;}}
  h1,h2,h3{{font-family:"Space Grotesk","DM Sans",sans-serif;letter-spacing:-.02em;}}
  a{{color:var(--green-br);}}
  header{{padding:64px 0 8px;}}
  .kicker{{font:700 12px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--amber);margin-bottom:12px;}}
  h1{{font-size:clamp(34px,6vw,56px);line-height:1.02;color:var(--green);}}
  h1 .amber{{color:var(--amber);}}
  .lead{{font-size:20px;color:var(--text-med);max-width:60ch;margin-top:18px;}}
  .rule{{height:3px;background:var(--amber);width:64px;margin:26px 0 0;border-radius:2px;}}
  section{{padding:40px 0;border-top:1px solid var(--border);}}
  section:first-of-type{{border-top:0;}}
  h2{{font-size:clamp(24px,3.4vw,32px);margin-bottom:8px;}}
  h2 .n{{font:700 14px var(--mono);color:var(--amber);margin-right:10px;}}
  h3{{font-size:19px;margin:22px 0 6px;color:var(--green);}}
  p{{color:var(--text-med);margin:10px 0;max-width:66ch;}}
  strong{{color:var(--green);}}
  .fig{{margin:20px 0;background:var(--card);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;}}
  .fig img{{display:block;width:100%;height:auto;}}
  .fig svg{{display:block;width:100%;height:auto;background:#191921;}}
  figcaption{{font:500 13.5px/1.5 var(--mono);color:var(--text-lt);padding:11px 16px;letter-spacing:.01em;}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}}
  @media(max-width:640px){{.two{{grid-template-columns:1fr;}}}}
  .shotph{{aspect-ratio:16/7;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;
    background:repeating-linear-gradient(45deg,#ecebe4,#ecebe4 12px,#e4e2da 12px,#e4e2da 24px);color:var(--text-lt);}}
  .shotph span{{font:700 10px var(--mono);letter-spacing:.22em;}}
  .shotph b{{font:600 15px "Space Grotesk";color:var(--green);}}
  .steps{{counter-reset:s;list-style:none;margin:14px 0;}}
  .steps li{{position:relative;padding:12px 0 12px 46px;border-bottom:1px solid var(--border);color:var(--text-med);max-width:66ch;}}
  .steps li:last-child{{border-bottom:0;}}
  .steps li::before{{counter-increment:s;content:counter(s);position:absolute;left:0;top:12px;width:30px;height:30px;
    border-radius:50%;background:var(--green);color:var(--cream);font:700 14px var(--mono);display:flex;align-items:center;justify-content:center;}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;margin:16px 0;}}
  .card.rulebox{{border-left:4px solid var(--amber);background:#FEF9EF;}}
  .setups{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:18px 0;}}
  @media(max-width:640px){{.setups{{grid-template-columns:1fr;}}}}
  .setup{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;}}
  .setup h4{{font:700 15px "Space Grotesk";color:var(--green);margin-bottom:6px;}}
  .setup .src{{font:600 12px var(--mono);color:var(--amber);display:block;margin-bottom:8px;}}
  .setup p{{font-size:14.5px;margin:0;}}
  code{{font:500 14px var(--mono);background:#eceae3;padding:1px 6px;border-radius:5px;color:var(--green);}}
  .kv{{font:600 13px var(--mono);color:var(--blue);}}
  footer{{padding:48px 0 72px;color:var(--text-lt);font:500 13px var(--mono);border-top:1px solid var(--border);margin-top:24px;}}
  footer a{{color:var(--green-br);}}
  @media print{{
    body{{background:#fff;}} section{{page-break-inside:avoid;}} .shotph{{background:#f0f0f0;}}
    header{{padding:20px 0 8px;}} a{{color:var(--green);text-decoration:none;}}
  }}
</style>
</head>
<body>
<header class="wrap">
  <div class="kicker">control-z · the Hush + Speak handoff</div>
  <h1>Clean early,<br><span class="amber">reconstruct late.</span></h1>
  <p class="lead">Hush quiets the noise at the <strong>first</strong> node. Speak gives the image
  its voice at the <strong>last</strong>. In between is your grade. This is how the two hand off —
  and the one rule that keeps the handoff honest.</p>
  <div class="rule"></div>
</header>

<main class="wrap">

<section>
  <h2><span class="n">01</span>The shape of it</h2>
  <p>Two nodes, one grade between them. Hush measures the clip and removes noise where it can prove
  there is noise — and exports a <strong>clean-confidence matte</strong>: high where it averaged deep and
  flattened real texture, low where it protected moving detail. Speak reads that matte at the end of the
  chain and puts film grain back <em>exactly where the texture was lost</em>. The matte rides its own
  wire — the blue key — never the picture's alpha.</p>
  <figure class="fig">{NODE_SVG}<figcaption>The tree: Hush (green, first) → your grade → Speak (amber, last). The blue key wire carries the clean-confidence matte straight from Hush to Speak, bypassing the grade.</figcaption></figure>
</section>

<section>
  <h2><span class="n">02</span>What Speak does to the picture</h2>
  <p>Speak is not a filter over your image — it models the photochemical chain: an H&amp;D negative and
  print tone scale, subtractive dye colour, optical halation and bloom, and density-domain grain. Here is
  a synthetic test frame before and after the full look (punchy-print family, halation + bloom + grain on).</p>
  <div class="two">
    {FIG(IMG["g_input"], "Before — the untouched working-space image (Speak at identity).")}
    {FIG(IMG["g_look"], "After — film tone, subtractive colour, halation glow around the practical, and grain. Note the warm halo: halation halates red.")}
  </div>
  <p>Every claim Speak makes about the image is <strong>showable on screen</strong>. The isolated views
  put a single stage on 18% gray so you can see exactly what it adds — never auto-gained, so a subtle
  effect correctly looks subtle.</p>
  <div class="two">
    {FIG(IMG["g_grain"], "View → Grain: the grain increment alone, on gray. Loud in the shadows, silent at paper white — density noise, the way film behaves.")}
    {FIG(IMG["g_halation"], "View → Halation Scatter: the re-exposure field alone, in the same units as the picture. Small everywhere except beside a real highlight.")}
  </div>
</section>

<section>
  <h2><span class="n">03</span>The matte handoff, seen</h2>
  <p>This is the whole idea in one image. A flat shadow field, grain isolated on gray, with the matte
  wired: <strong>left half cleaned</strong> (matte = 1, full grain goes back), <strong>right half protected
  motion</strong> (matte = 0, grain sits at the Floor because the real noise survived there). Contrast
  boosted for the page — the grain view is subtle by design.</p>
  <div class="two">
    {FIG(IMG["m_shaped"], "Matte Source: Key input — grain lands where Hush cleaned, and backs off where it protected. The seam is the matte.")}
    {FIG(IMG["m_uniform"], "Matte Source: Off — uniform grain everywhere, ignoring where the noise actually went. Fine when you have no matte; wrong when you do.")}
  </div>
</section>

<section>
  <h2><span class="n">04</span>Wiring it on the color page</h2>
  <p>Three steps, once per shot. The two screenshots below are the parts that live in Resolve's own UI
  (the node editor and the OFX menu) — capture them from your project.</p>
  <ol class="steps">
    <li>On the <strong>Hush</strong> node: turn on <span class="kv">Export Clean Matte to Alpha</span>,
      then right-click the node → <span class="kv">OFX Alpha → Enable</span>. That publishes the matte on
      Hush's key output.</li>
    <li>Drag the <strong>blue key wire</strong> from Hush's key output to <strong>Speak's</strong> key input.
      (Not the RGB wire — the key. It runs underneath your grade.)</li>
    <li>On the <strong>Speak</strong> node: set <span class="kv">Matte Source → Key input</span> and turn
      <span class="kv">Enable Grain</span> on. The status strip confirms it:
      <code>matte: key ✓ mean 0.31</code>.</li>
  </ol>
  <div class="two">
    {SHOT("Color page — the blue key wire", "The node editor with Hush → Speak and the blue key wire dragged between them. (Capture from your project.)")}
    {SHOT("OFX Alpha → Enable", "The right-click menu on the Hush node showing OFX Alpha → Enable. (Capture from your project.)")}
  </div>
  <p>No key wired? Speak does <strong>not</strong> silently fall back to the picture's alpha (which on the
  color page is opaque — that read as "everything is clean" and blew grain to full strength everywhere in
  v0.3). With <em>Key input</em> selected and nothing wired, the matte is treated as <strong>absent</strong>:
  grain holds at the Matte Floor, and the strip says <code>key not connected — grain at floor</code>.</p>
</section>

<section>
  <h2><span class="n">05</span>The three setups</h2>
  <p>Pick the one that matches what you have. Speak stays fully useful with no matte at all.</p>
  <div class="setups">
    <div class="setup"><span class="src">Matte Source: Key input</span>
      <h4>Grain shaped by the matte</h4>
      <p>The full handoff. Grain returns where Hush cleaned, floors where it protected. Needs the blue wire.</p></div>
    <div class="setup"><span class="src">Matte Source: Off</span>
      <h4>Uniform grain</h4>
      <p>Speak standalone. Even grain over the whole frame — the classic film-grain move, no Hush required.</p></div>
    <div class="setup"><span class="src">Enable Grain: off</span>
      <h4>No grain, image untouched</h4>
      <p>Tone, colour and optics only — or bypass entirely. Alpha passes through untouched.</p></div>
  </div>
</section>

<section>
  <h2><span class="n">06</span>The one rule</h2>
  <div class="card rulebox">
    <p style="margin:0"><strong>Leave Hush's "Export Clean Matte to Alpha" OFF unless Speak consumes it.</strong>
    An exported matte that nothing reads keeps riding your alpha channel, where a downstream unpremultiply
    or composite can lift it into the picture. When Speak <em>does</em> consume a matte (any active source),
    it forces its own output alpha opaque — the matte is spent at its consumer and never rides further.
    That is consume-on-use, and it is why the handoff can't quietly corrupt a delivery.</p>
  </div>
  <p>Speak draws this whole recipe on the viewer itself — <span class="kv">View → Setup Guide</span> — so the
  instructions ship inside the plugin and can never drift from it. This page is the same story, with room
  for the pictures.</p>
</section>

</main>

<footer class="wrap">
  Hush &amp; Speak are free, MIT-licensed OpenFX plugins for DaVinci Resolve (they work in the free
  edition). &nbsp;·&nbsp;
  <a href="https://github.com/amateurmenace/Hush-OpenNR">Hush</a> &nbsp;·&nbsp;
  <a href="https://github.com/amateurmenace/Speak">Speak</a> &nbsp;·&nbsp;
  <a href="https://control-z.org">control-z.org</a><br>
  Original published color science — H&amp;D sensitometry, printer points, subtractive dye behaviour,
  Selwyn-family granularity. Never cloned profiles or trademarked stock names.
</footer>
</body>
</html>'''

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"wrote {OUT}  ({len(HTML)//1024} KB)")

# Changelog — DK2 Remix Predetermined Hashing

A day-by-day journal of the Dungeon Keeper 2 → RTX Remix work. Dates are real (drawn from
project notes); for the 2026-06 entries the commits were reconstructed at publish time rather
than committed live (see the README note). Read top-to-bottom, it's the story of the project.

---

## 2026-01 — First light

- **2026-01-19** — First real-time **path tracing on Dungeon Keeper 2** (DirectX 6, 1999) via dxwrapper + RTX Remix. Believed to be the first DX6 title on Remix.

## 2026-05 — Torches, textures, and the cache

- Diagnosed DK2's **runtime atlas composition**: the engine bin-packs sprites into shared pages with its own LRU cache, so Remix's texture hashes **drift** between runs. Confirmed empirically — 41 hand-tagged hashes were all orphaned the next day.
- **Torch flicker solved**: universal UV-region decompose + power-of-two sub-textures; flame regions split out of co-atlased pages.
- Built a 7-layer decompose cache (LRU + skip-splice + streaming blacklist + per-Blt rect overlap + a global crops/sec scarcity gate with hysteresis — "Phase 8").
- Cracked the `EngineTextures.dat/.dir` format and its YCbCr 8×8 DCT codec; built a near-bit-exact encoder.
- Verified our XXH3 hashing is **byte-identical** to Remix's — the key that makes hashes computable offline.
- Texture-replacement pipelines established (PBRify, then CHORD joint-PBR).

## 2026-06-09 to 06-11 — The drift war

- **Canonical Identity Layer**: at whole-surface bind, resolve drifted content back to a frozen canonical texture, so Remix sees one stable hash per content regardless of atlas churn.
- **Name-key resolution**: a read-only detour at the dominant upload blit (`DKII.exe 0x58E53B`) reads the game's descriptor name index — stable source identity — and records sprite placements `{name, mip, x, y, w, h}`.
- **V2 placement-keyed crops**: bind-time resolution serves payload-built crops keyed by `(name, mip)`, pixel-verified against the offline-known payload, failing closed to the original on mismatch. **~200k draws/session served by predetermined identity.** The drift war, won.
- Full clean PBRify art corpus authored against the predetermined hashes; UI made readable; the Hand of Evil cursor made visible.

## 2026-06-11 to 06-12 — Water, metal, lighting

- **Translucent water**: 32 animation frames collapsed to one stable hash and swapped for an AperturePBR translucent material. The per-frame sine-wave geometry is flattened wrapper-side — DK2's pre-transformed screen-space verts are inverse-projected to world space, then a per-frame least-squares quadratic surface is fit and the verts snapped to it (camera-independent, stateless).
- **Metal "red glow" solved**: DK2's drawn-scene reflection map (`DefaultEnvMap`) was being read as self-lit under the path tracer; neutralized through the replacement pipeline.
- **Dynamic-UI emissive**: pointer/tooltips/minimap forced self-lit so the HUD reads in the dark.

## 2026-06-12 — Lighting, menu, heap, camera

- **Lighting responsiveness won**: `rtx.suppressLightKeeping=True` — Remix's default keeps moved/dead game lights ~100 frames (the ~2s lag); found via a source dive of the Remix runtime.
- **Front-end menu at 2560×1600+**: a 2-instruction runtime patch at `DKII.exe 0x52BB66` (port of the DiaLight/Flame `menu-res` idea); the exe's own GUI scaler handles the rest.
- **CRT heap replacement**: detour DK2's five VC6 CRT allocators to a private low-fragmentation heap, bypassing the 1999 small-block-heap global lock that serialized alloc/free across the sound and main threads.
- **Camera pitch** tuned via `GlobalVariables.kwd` record edits (steeper top-down angle, free frustum-shrink perf win).

## 2026-06-16 — Clear water, and the next mountain

- Water material tuned to a clear translucent blue (found that with `thin_walled` the transmittance distance is ignored — the tint was driven purely by the transmittance color).
- **Geometry pillar opened.** Meshes live in `Data/Meshes.WAD` (DWFB → KMF). Added `DdrawGeomProbe`, a read-only probe that hashes each draw's topology + UVs (position-independent) to ask whether DK2 geometry has a stable identity a Remix mesh replacement could bind to.

## 2026-06-17 — What the geometry actually is

- Probe results: a bindable *core* of stable-topology meshes exists, but it's drowned by a flood of per-frame one-shot geometry. Cracked `Meshes.WAD` (custom Bullfrog LZ, not RNC) and extracted the full mesh inventory — every creature named, cross-checked against the live draw stream.
- **Trass3r's expert reality check**: creatures are **vertex-animated** (each frame a different hash), models are **split** into named submeshes, and **LOD** is a per-submesh runtime mystery. Confirmed: Remix mesh replacement works for **static props**, not animated creatures or batched terrain.

## 2026-06-18 to 06-19 — Reverse-engineering the render path

- Added `DdrawDrawCallerLog` — a read-only stack scan that finds the game's own draw call sites in `DKII.exe`. Located DK2's **render loop** (`0x4150EE`, the COM draw `call [vtable+0x48]`), the per-submesh state setup, and the split-model "recipe" char-array — the first step toward forcing a single LOD so static meshes become reliably replaceable. Ongoing.

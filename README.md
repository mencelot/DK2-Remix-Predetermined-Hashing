# DK2 — Remix Predetermined Hashing

A modified [`dxwrapper`](https://github.com/elishacloud/dxwrapper) that bridges **Dungeon Keeper 2** (1999, DirectX 6) into **[NVIDIA RTX Remix](https://github.com/NVIDIAGameWorks/rtx-remix)** path tracing — and, more specifically, makes DK2's runtime-composed textures *replaceable* by giving them a **predetermined, stable hash** that can be authored against offline.

As far as we know, DK2 is the only DirectX 6 title running on Remix. This repo is the wrapper that makes it possible. It's a hobby project and very much an experiment — please read it in that spirit.

> First real-time path tracing on DK2: **2026-01-19**.

---

## Why this exists

RTX Remix replaces a game's textures by **hashing** each texture the game uploads and letting you swap art keyed to that hash. That works beautifully when a game uploads stable, identifiable textures.

DK2 does not. It's a 1999 engine that **composes textures at runtime** — it bin-packs sprites (wall tiles, torch flames, HUD bits, water frames) into shared atlas pages on the fly, using its own LRU cache. The same wall can land in a different page, at a different offset, on a different run. To Remix, the hash of that page therefore **drifts**: the texture you tagged today is orphaned tomorrow. Tagging hashes by hand is a losing game — we verified empirically that 41 hashes saved one day were all gone the next.

**The core idea here — "predetermined hashing" — is to stop chasing runtime hashes and instead make them knowable in advance.**

1. **Recover stable source identity.** A read-only detour at DK2's dominant texture-upload blit reads the game's *own* descriptor, which carries a stable `EngineTextures` name index. So we know *what* a page is made of, by name, regardless of where the atlas packer put it.
2. **Predetermine the hash.** Because we can decode the source bytes and we replicate Remix's exact hashing (XXH3 of the tightly-packed mip-0 bytes), we can compute — **offline, before the game even runs** — the precise hash Remix will see for any given sprite or canonical page. Art can be authored against those hashes ahead of time.
3. **Resolve by name at runtime.** At bind time, the wrapper resolves the drifted runtime page back to its canonical, predetermined identity by name (pixel-verified against the expected payload, failing closed to the original on any mismatch), and serves payload-built crops keyed by `(name, mip)`.

The result: ~200k draws per session served by stable, offline-known identity — the texture "drift war," won. The deep-dive write-up is in [`DK2_RTX_REMIX_DOCUMENTATION.md`](DK2_RTX_REMIX_DOCUMENTATION.md).

## What works today

- **Path-traced Dungeon Keeper 2** — real ray-traced global illumination, reflections, and emissive torch light on the original game.
- **PBR texture replacement** at scale — every `EngineTextures` image re-authored (PBRify/CHORD pipelines), bound through the predetermined-hash layer so it actually sticks.
- **Translucent, still water** — DK2's 32-frame animated water is frame-collapsed to one stable hash and swapped for a path-traced translucent material; the per-frame sine-wave geometry is flattened wrapper-side via a per-frame least-squares surface fit (DK2's screen-space verts are inverse-projected back to world space first).
- **Responsive lighting** — moved/extinguished lights update near-instantly (Remix's default keeps stale lights ~100 frames).
- **Readable UI in the dark**, a visible Hand of Evil cursor, and an optional **high-resolution front-end menu** (2560×1600+).
- A handful of **surgical `DKII.exe` patches** (zoom-out, camera pitch, menu resolution) and a **private LFH heap** detour that bypasses the 1999 CRT small-block-heap lock.

## What does *not* work (the honest part)

- **Geometry / meshes are the open frontier.** Remix mesh *replacement* binds by geometry hash, and DK2 fights it the same way it fought textures, only harder:
  - **LOD** is selected at runtime, per-submesh, by a calculation even the experts call a mystery — so a mesh's hash changes with camera distance.
  - **Creatures are vertex-animated** (morph/shape-keys, not skeletal), so the submitted geometry — and thus its hash — changes *every frame*. A static replacement can't follow it.
  - **Models are split** into named submeshes (e.g. an imp is back + front + pickaxe) for texture resolution, multiplying the hash count further.
  - Net: **static props are replaceable; animated creatures and per-frame-batched terrain are not** — at least not without a wrapper-side geometry-identity layer or forcing the engine to a single LOD. That work is in progress and documented honestly in the notes, dead-ends included.

## Where the DK2-specific work lives

This is a **fork of dxwrapper** (via [dkollmann's RTX-Remix fork](https://github.com/dkollmann/dxwrapper)), which is a large, general DirectX-compatibility toolkit. The vast majority of files here are upstream and untouched. Our contribution is concentrated in:

- `ddraw/IDirectDrawSurfaceX.cpp` — the predetermined-hash / canonical identity / name-key layer, the universal UV-decompose cache, and the offline-hashing machinery.
- `ddraw/IDirect3DDeviceX.cpp` — the XYZRHW→world inverse projection, the water surface-fit, and various read-only RE probes.
- `ddraw/HeapReplace.cpp`, `ddraw/MenuRes.cpp` — the heap detour and the menu-resolution exe patch.
- `Settings/Settings.h` — the `Ddraw*` flags that gate all of the above.

## Building

Visual Studio 2022, MSBuild, **Win32 / Release**, toolset v143:

```
msbuild dxwrapper.sln /t:dxwrapper /p:Configuration=Release /p:Platform=Win32 /p:PlatformToolset=v143
```

The output `dxwrapper.dll` (plus a `dxwrapper.ini`) is dropped into the DK2 game folder, alongside an RTX Remix runtime. This wrapper alone does not ship the Remix runtime, the mods, or the re-authored art.

## Credits & lineage

Standing entirely on other people's shoulders:

- **[elishacloud/dxwrapper](https://github.com/elishacloud/dxwrapper)** — the DirectDraw/Direct3D 1–7 → D3D9 foundation everything here is built on. Upstream README preserved as [`README_dxwrapper.md`](README_dxwrapper.md).
- **[dkollmann/dxwrapper](https://github.com/dkollmann/dxwrapper)** — the RTX-Remix-oriented fork this branched from.
- **[NVIDIA RTX Remix](https://github.com/NVIDIAGameWorks/rtx-remix)** — the path tracer doing the actual heavy lifting.
- **[Trass3r's GameTools](https://github.com/Trass3r/GameTools)** and **[OpenKeeper](https://github.com/tonihele/OpenKeeper)** — for reverse-engineered knowledge of DK2's formats (WAD, KMF, the codec, and the hard truths about LOD and vertex animation). Trass3r in particular has been generous with straight, expert answers.
- The **PBRify** and **CHORD** communities for the texture-upscaling pipelines.

## License

zlib, inherited from dxwrapper — see [`License.txt`](License.txt). Per-file copyright headers are intact; nothing here claims authorship of upstream code.

---

### A note on this repo's history

We built most of this by iterating live — editing, rebuilding, testing in-game — without committing along the way. So the commit dates for the 2026-06 work are a **faithful reconstruction** assembled at publish time from dated project notes, not the original moment-by-moment history. The dates and descriptions are real; they're simply grouped after the fact. The earlier history (the January/May torch, texture, and XYZRHW work) is genuine. See [`CHANGELOG.md`](CHANGELOG.md) for the day-by-day story.

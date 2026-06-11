/**
* Copyright (C) 2025 Elisha Riedlinger
*
* This software is  provided 'as-is', without any express  or implied  warranty. In no event will the
* authors be held liable for any damages arising from the use of this software.
* Permission  is granted  to anyone  to use  this software  for  any  purpose,  including  commercial
* applications, and to alter it and redistribute it freely, subject to the following restrictions:
*
*   1. The origin of this software must not be misrepresented; you must not claim that you  wrote the
*      original  software. If you use this  software  in a product, an  acknowledgment in the product
*      documentation would be appreciated but is not required.
*   2. Altered source versions must  be plainly  marked as such, and  must not be  misrepresented  as
*      being the original software.
*   3. This notice may not be removed or altered from any source distribution.
*/

#include "winmm.h"
#include "ddraw.h"
#include <fstream>
#include <intrin.h>
#include <unordered_set>
#include <list>
#include <filesystem>
#include "Utils\Utils.h"
// xxHash single-header (dxvk-remix's exact copy). XXH_INLINE_ALL pulls in the
// full implementation as static functions in this translation unit. We use
// XXH3_64bits over tightly-packed mip-0 bytes to reproduce Remix's texture
// content hash byte-exactly -- see dxvk-remix d3d9_common_texture.cpp
// SetupForRtxFrom(): imageHash = XXH3_64bits(buffer->mapPtr(0), buffer->info().size).
// Matching it means our captured content hashes equal Remix capture filenames
// + community PBR mod filenames, so replacements drop in by hash name.
#define XXH_INLINE_ALL
#include "External\xxhash\xxhash.h"

namespace {
	constexpr DWORD ExtraHeightPadding = 4;
	constexpr DWORD SurfaceWaitTimeoutMS = 4;

	// Used for dummy mipmaps
	std::vector<BYTE, aligned_allocator<BYTE, 4>> dummySurface;

	// Used to allow presenting non-primary surfaces in case the primary surface present fails
	bool dirtyFlag = false;
	bool SceneReady = false;
	bool IsPresentRunning = false;

	// Used for sharing emulated memory
	bool ShareEmulatedMemory = false;
	std::vector<EMUSURFACE*> memorySurfaces;

	// Lock/Unlock tracking for RTX Remix texture atlas analysis
	struct LockInfo {
		void* surface;
		DWORD width;
		DWORD height;
		DWORD flags;
		bool isWriteLock;
	};

	struct SurfaceLockStats {
		DWORD lockCount = 0;
		DWORD writeLockCount = 0;
		DWORD width = 0;
		DWORD height = 0;
		std::vector<LockInfo> recentLocks;
	};

	std::unordered_map<void*, SurfaceLockStats> lockTrackingMap;
	DWORD lockTrackingFrame = 0;

	// --- Surface-pool tracking (Phase A: torch-flicker investigation, 2026-05-15) ---
	// Tests whether DK2 thrashes a fixed-size texture-surface pool as visible
	// content (torches/water) rises. Gated on Config.DdrawLogTextureAtlas.
	// Single render thread => no locking, matching the lockTrackingMap style above.
	struct SurfacePoolStats {
		LONG  liveTextureSurfaces = 0;     // current count of live DDSCAPS_TEXTURE surfaces
		LONG  peakTextureSurfaces = 0;     // session high-water mark
		DWORD createdThisWindow = 0;       // creations since last 60-frame dump
		DWORD releasedThisWindow = 0;      // destructions since last 60-frame dump
		DWORD bltToTextureThisWindow = 0;  // Blts whose destination is a texture surface
	};
	SurfacePoolStats poolStats;
	std::unordered_map<void*, char> trackedTextureSurfaces;  // surfaces counted as textures
	std::unordered_map<DWORD, DWORD> createCallerHist;       // DKII.exe return addr -> count

	// Phase A.6 (2026-05-18): per-destination Blt attribution. Tests whether the
	// baseline 4-26 bltToTex/window in steady state is concentrated on a small
	// set of "live" surfaces (= candidate torch/animated frames) or spread thin.
	struct BltDestInfo {
		DWORD count = 0;
		DWORD width = 0;
		DWORD height = 0;
		void* lastSrc = nullptr;
	};
	std::unordered_map<void*, BltDestInfo> bltTargetsThisWindow;

	// Path B (2026-05-18 evening): animation-pool collapse.
	// Phase A.7 determinism test proved one logical torch produces ~11 distinct
	// hashes per session and zero overlap across sessions. The mechanism is:
	// the engine maintains a pool of per-instance surfaces all blt'd from one
	// shared source. We detect a pool when one source's distinct-destination
	// count >= POOL_DETECT_THRESHOLD, then redirect SetTexture binds of the
	// non-canonical members to the first-seen (canonical) member.
	constexpr size_t POOL_DETECT_THRESHOLD = 5;  // tune: 11 torches => 5 is well below
	struct AnimPoolInfo {
		void* canonical = nullptr;                       // first dest seen (typed as m_IDirectDrawSurfaceX*)
		std::unordered_set<void*> members;               // all dests fed from this src
	};
	std::unordered_map<LPDIRECTDRAWSURFACE7, AnimPoolInfo> poolBySource;
	std::unordered_map<void*, void*> memberToCanonical; // dest_wrapper -> canonical_wrapper for fast SetTexture lookup
	// Per-dest blt count -- gate redirection to dests that have been re-written
	// >= 2 times (= dynamic content) so we don't collapse static terrain pools
	// where each tile is written once at load.
	std::unordered_map<void*, DWORD> destBltCount;
	DWORD poolRedirectsTotal = 0;
	DWORD poolsActiveTotal = 0;

	// --- Phase A.10 (2026-05-18): atlas decomposition ---
	// For known REAL_K_IN_1 atlases (identified offline by L2-fingerprint clustering
	// of per-drawcall UV regions, per dk2_phase_a8_atlas_decomp memory), synthesize
	// one d3d9 sub-texture per region and bind it instead of the source atlas. The
	// drawcall's vertex UVs are rewritten to [0,1] over the sub-texture so Remix
	// observes a distinct content hash per region, enabling per-region replacements.
	//
	// Atlas identity is keyed by content hash (via firstHashByTex), NOT wrapper
	// pointer -- pointers don't survive session restarts.
	struct UVRect { float u0, v0, u1, v1; };
	// Content-hash-keyed decomposition (fast path -- O(1) lookup, exact match).
	// Works only for atlases whose surface bytes are deterministic across
	// sessions. Pool atlases (one source Blt'd into N per-instance surfaces)
	// drift byte-wise per session and need visual-fingerprint matching
	// (see kAtlasFamilies below) instead.
	const std::unordered_map<uint64_t, std::vector<UVRect>> kAtlasRegions = {
		// Wrapper 0EAC13C8 in original session, 0EC25428/0EC0BFD0/0EC2D180/0EC35978
		// in later runs (pointer shifts, content hash is stable). Torch flame +
		// cyan shapes + stone wall + HUD ("1" badge + segmented ring).
		// Cross-session stability: 8/13 manifests on disk.
		// Verified end-to-end across 3 sessions 2026-05-18 (3 sub-textures
		// synthesized each time, cropped content matches expected quadrants).
		{ 0x6FBA7910BBB20D7DULL, {
			{ 0.50f, 0.75f, 0.75f, 1.00f },   // bottom-left of BR quadrant: "1" badge
			{ 0.75f, 0.75f, 1.00f, 1.00f },   // bottom-right of BR quadrant: segmented ring
			{ 0.50f, 0.00f, 1.00f, 0.50f },   // TR half-quadrant: cyan/white shapes
		} },
		// Character portrait atlas (2 dwarf-like heads visible, plus top strip).
		// Cross-session stability: 12/13 manifests on disk -- the MOST stable
		// REAL_K_IN_1 atlas. Fresh-session classifier output 2026-05-18 21:46
		// produced these two non-nested regions (R0 has unique [0.65,1.0]
		// u-range, R1 has unique [0,0.05] u-range -- engine samples both
		// distinctly).
		{ 0x6E3141D82ABA1AEDULL, {
			{ 0.05f, 0.00f, 1.00f, 1.00f },   // whole atlas minus left sliver (428 draws)
			{ 0.00f, 0.25f, 0.65f, 0.95f },   // mid-left rect (191 draws)
		} },
	};

	// Synthesized sub-textures cached per (atlas wrapper ptr, region idx). Keyed by
	// wrapper because two surfaces sharing a content hash still have independent
	// d3d9 source textures and we want to crop from the actual bound one.
	struct SubTexKey {
		void* atlasWrapper;
		int regionIdx;
		bool operator==(const SubTexKey& o) const { return atlasWrapper == o.atlasWrapper && regionIdx == o.regionIdx; }
	};
	struct SubTexKeyHash { size_t operator()(const SubTexKey& k) const { return std::hash<void*>()(k.atlasWrapper) ^ ((size_t)k.regionIdx * 0x9E3779B97F4A7C15ULL); } };
	std::unordered_map<SubTexKey, IDirect3DTexture9*, SubTexKeyHash> subTextureCache;
	DWORD atlasDecomposeBindsTotal = 0;

	// --- Visual fingerprint matching (fallback for pool atlases) ---
	// Same UVRect type as above. Family table includes a 32x32 grayscale
	// fingerprint + sorted-smallest-first region list. At first bind of an
	// unknown atlas, runtime computes the bound surface's fingerprint via
	// D3DXLoadSurfaceFromSurface scale-to-32x32 + LockRect readback, then
	// L2-distance-matches against family fingerprints. Match decision is
	// cached per wrapper so subsequent binds are free.
#include "AtlasFingerprints.inc"

	// Per-wrapper fingerprint-match cache:
	//   -2 = never computed
	//   -1 = computed, no family matched (do not decompose)
	//  >=0 = matched family index in kAtlasFamilies
	constexpr int FP_NEVER_COMPUTED = -2;
	constexpr int FP_NO_MATCH = -1;
	std::unordered_map<void*, int> wrapperToFamilyIdx;
	DWORD fingerprintComputeTotal = 0;
	DWORD fingerprintMatchTotal = 0;

	// --- Path C: canonical sub-texture by (family, region) ---
	// Once a fingerprint-matched atlas synthesizes a sub-texture for region R,
	// store that d3d9 texture as the canonical for (family_idx, R). Subsequent
	// pool variants matching the same family+region bind THE SAME canonical
	// d3d9 texture instead of synthesizing their own copies. Remix then sees
	// ONE hash per (family, region) regardless of how many pool variants the
	// game spawned, so a single mod replacement covers them all.
	//
	// Key packs family_idx (int32) high, region_idx (int32) low. Negative
	// family_idx is impossible here since we only insert on positive match.
	std::unordered_map<uint64_t, IDirect3DTexture9*> canonicalByFamilyRegion;
	DWORD pathCRedirectsTotal = 0;
	DWORD pathCCanonicalsCreated = 0;

	inline uint64_t MakeFamilyRegionKey(int familyIdx, int regionIdx)
	{
		return ((uint64_t)(uint32_t)familyIdx << 32) | (uint64_t)(uint32_t)regionIdx;
	}

	// --- Universal UV-region decomposition (2026-05-21) ---
	// No fingerprints, no hardcoded hashes, no source PNGs. For ANY drawcall whose
	// stage-0 UV bbox is a proper sub-rect of its bound texture (area gate applied
	// caller-side), crop that exact texel rect into a sub-texture and bind it.
	//
	// One file-scope cache (shared across all surfaces): univDrawCache, keyed by
	// (atlas wrapper, quantized UV rect) -> sub-texture + the snapped region transform.
	// PERF-CRITICAL: first sight of a (surface, rect) does the crop; every later frame
	// is an O(1) hit unless the temporal-validation gate (Phase 2, 2026-05-24) decides
	// the surface was repurposed -- in which case the entry is invalidated and re-cropped
	// from current content. The temporal gate replaces the old "freeze forever" policy,
	// which fixed torch flicker but caused torches to bleed into imp footprints / flies
	// when DK2 reused source surfaces for different sprites over a session.
	// (A second cache `univContentCache` was previously declared for cross-surface content
	// dedup, but the dedup path was removed because the post-stretch sysmem readback
	// deadlocked against Remix. The declaration is gone now.)
	struct UnivDrawKey {
		void* wrapper;
		uint16_t qu0, qv0, qu1, qv1;   // UV quantized to 1/4096 of the texture
		bool operator==(const UnivDrawKey& o) const {
			return wrapper == o.wrapper && qu0 == o.qu0 && qv0 == o.qv0 && qu1 == o.qu1 && qv1 == o.qv1;
		}
	};
	struct UnivDrawKeyHash {
		size_t operator()(const UnivDrawKey& k) const {
			size_t h = std::hash<void*>()(k.wrapper);
			h ^= ((size_t)k.qu0 << 1) ^ ((size_t)k.qv0 << 13) ^ ((size_t)k.qu1 << 27) ^ ((size_t)k.qv1 << 41);
			return h;
		}
	};
	struct UnivSubTex {
		IDirect3DTexture9* tex = nullptr;
		float u0 = 0, v0 = 0, u1 = 1, v1 = 1;   // snapped region in atlas UV space (transform source)
		std::list<UnivDrawKey>::iterator lruIt; // position in univLru (for O(1) LRU touch/evict)
		DWORD cachedAtRepurposeGen = 0;         // value of surface.RepurposeGen at cache insert; mismatch at hit time -> invalidate (Phase 5: per-surface idle-then-Blt gate)
		LONG px0 = 0, py0 = 0, px1 = 0, py1 = 0; // pixel-space rect this sub-texture covers, for per-Blt rect overlap test (Phase 7)
		DWORD cachedAtTimeMs = 0;               // GetTickCount() at cache insert; per-Blt rect overlap gate compares this to BltHistory entry timestamps (Phase 7)
	};
	std::unordered_map<UnivDrawKey, UnivSubTex, UnivDrawKeyHash> univDrawCache;
	// LRU eviction (2026-05-21): the crop cache USED to be frozen forever -> in a long
	// session the map's constantly-shifting sub-rects spawned 10k+ MANAGED textures that
	// were never freed, blowing the 32-bit DKII.exe ~2GB address space -> OOM (confirmed:
	// canonical without LRU died at 86k crops, ~11min). Now bound: front = LRU, back = MRU.
	// On a cache hit we splice to back (skipped if already MRU). When over cap we evict +
	// Release() the front's texture, ONE per insert -- batch eviction (tried earlier)
	// caused periodic 4k-Release stutters. Torches are drawn every frame -> always MRU.
	//
	// Cap retuned to 16384 (2026-05-24 evening) after a 20-min Phase 2 session log showed
	// 255k crops + 248k evicts on cap=4096 (98% hit ratio but constant LRU churn). Each
	// evict = a fresh texture appearing/disappearing for Remix to hash, which thrashes
	// its denoiser caches -> visible as "every texture flickers between usual and black"
	// in busy scenes. Bigger cap with one-per-insert eviction (NOT batch) should let the
	// working set fit, evicts -> ~0, Remix sees stable textures. Worst case at 50KB avg
	// per entry: 16384 * 50KB = ~800MB -- tight for 32-bit but within budget.
	std::list<UnivDrawKey> univLru;
	static const size_t kUnivCacheCap = 16384;
	DWORD univDecomposeBindsTotal = 0;
	DWORD univCropsTotal = 0;
	DWORD univEvictTotal = 0;
	DWORD univInvalidatesTotal = 0;	// times the gen-mismatch gate fired (entry's gen != current surface.RepurposeGen)

	// --- Global rate scarcity gate (Phase 8, 2026-05-24) ---
	// During particle/effect bursts (e.g. Horny+pool steam: peak 1232 crops/sec across
	// 141 surfaces, 22442 unique (atlas, rect) keys in 60s, none individually exceeding
	// the per-surface 30/s blacklist threshold), the per-surface blacklist can't fire
	// fast enough -- the work fans out across too many surfaces, each just below trip.
	// Solution: a 1-second sliding window counting crops globally. When rate exceeds
	// kGlobalRateScarcityEnterPerSec, NEW synthesis returns false (caller falls back to
	// whole-surface texture). Cache hits are NOT gated -- torches with cached entries
	// keep their per-region replacements during the storm. Hysteresis (lower exit
	// threshold) prevents oscillation. Verified normal background is ~50 crops/sec;
	// burst peak ~1230/s -- threshold 200/s sits comfortably between.
	static const DWORD kGlobalRateScarcityEnterPerSec = 200;
	static const DWORD kGlobalRateScarcityExitPerSec = 100;
	DWORD g_globalRateWindowStartMs = 0;
	DWORD g_globalRateCropsInWindow = 0;
	bool  g_globalScarcityMode = false;
	DWORD univScarcitySkippedTotal = 0;
	DWORD univScarcityEntersTotal = 0;

	// --- Stage 3 measurement (2026-05-27) ---
	// Goal: measure mod utilization rather than asserting it. Two aggregates:
	//   stage3PerSurface[wrapperPtr] -> per-branch counts (hits/crops/invals/scarcity/blacklist)
	//   stage3BindByHash[xxh3]       -> per-Remix-bind counts (whole vs crop)
	// Dumped every kStage3DumpIntervalMs from a hot path. Cross-ref hashes vs mod
	// folders offline to get true substitution coverage. See dk2_mission_measure_before_claiming.
	struct Stage3SurfaceStat { DWORD hits=0, crops=0, invals=0, scarcitySkips=0, blacklistSkips=0; };
	std::unordered_map<void*, Stage3SurfaceStat> stage3PerSurface;
	struct Stage3BindStat { DWORD wholeBinds=0, cropBinds=0; };
	std::unordered_map<uint64_t, Stage3BindStat> stage3BindByHash;
	DWORD stage3WholeBindsUnhashed = 0;     // bind whose level-0 couldn't be locked/hashed (rare)
	DWORD stage3CropBindsUnhashed = 0;
	// Stage 3 fix (2026-05-29): self-contained content-hash cache keyed by the EXACT
	// d3d9 texture Remix binds. The original design looked the bound texture up in
	// capState.firstHashByTex, but that map is (a) only populated when
	// DdrawContentCapture=1 and (b) keyed by the WRAPPER pointer while bind sites pass
	// the d3d9 IDirect3DTexture9* -> every lookup missed, all binds counted unhashed.
	// Hashing the bound texture directly (once per pointer) removes both the capture-flag
	// dependency and the wrapper-key mismatch.
	std::unordered_map<IDirect3DTexture9*, uint64_t> stage3HashByD3d9Tex;
	// Stage 3 corpus dump (2026-05-29): when DdrawContentCapture=1, dump ONE PNG per
	// distinct WHOLE-surface bound hash, named by the exact Remix key + a manifest row
	// (hash,width,height,format). This is the runtime-keyed authoring corpus: the OLD
	// first-bind capture missed the heavily-bound whole surfaces (12/13 head gaps absent
	// from it). We hash exactly what Remix binds, so <hash>.png IS the correct DDS key.
	// Crops are skipped (session-specific noise). Dedup is by HASH (the pointer-keyed
	// cache above can present the same hash via multiple surfaces).
	std::unordered_set<uint64_t> stage3DumpedHashes;
	std::ofstream stage3BoundManifest;
	std::string stage3BoundDir;
	bool stage3BoundDirReady = false;
	DWORD stage3LastDumpMs = 0;
	static const DWORD kStage3DumpIntervalMs = 60000;

	// --- [RECIPE] composition-recipe instrumentation (2026-06-10) ---
	// Decisive test for write-side canonicalization: DK2 composes atlases via Blt,
	// which flows through us. If atlas hashes that DRIFT across sessions share an
	// IDENTICAL recipe -- ordered (srcContentHash, srcRect, dstRect) component list --
	// then deterministic shadow recomposition can collapse all per-instance/per-session
	// variants onto one stable hash, removing the per-hash authoring ceiling (216
	// stable canonicals as of v4). One [RECIPE] line per distinct bound whole-surface
	// hash; offline cross-session join answers the question with real counts.
	// Source identity = XXH3 of the source surface's mip-0 (same domain as Remix keys),
	// cached per wrapper ptr and invalidated via UniquenessValue (bumped on every write).
	struct RecipeEntry { uint64_t srcHash; uint16_t sx, sy, sw, sh, dx, dy, dw, dh; };
	struct RecipeState { std::vector<RecipeEntry> entries; DWORD overflows = 0; DWORD directWrites = 0; };
	std::unordered_map<IDirect3DTexture9*, RecipeState> recipeByTex;
	std::unordered_map<void*, std::pair<DWORD, uint64_t>> recipeSrcHashCache;	// wrapper ptr -> (UniquenessValue, mip0 hash)
	std::unordered_set<uint64_t> recipeEmitted;	// content hashes already logged
	DWORD recipeSurfacesDropped = 0;			// dest surfaces not tracked (map cap hit)
	IDirect3DTexture9* recipeLastBltTex = nullptr;	// suppresses the directWrites false positive from Blt's own
												// SetDirtyFlag call, which runs AFTER ScopedFlagSet(IsInBlt) closes
	// [LOCKW] (2026-06-10, recipe step 3): runs A/B proved ALL whole-surface Blts are
	// verbatim full-copies from staging surfaces -- composition happens in the game's
	// Lock writes (or earlier, in game memory). For every surface that has ever served
	// as a Blt SOURCE, capture each game Lock's rect + game-EXE caller address (via
	// FindGameCaller stack walk), and flush at SetDirtyFlag with the resulting content
	// hash. Tells us (a) whether composition is visible at Lock-rect granularity and
	// (b) the DKII.exe call sites that produce staged content -- the reverse-engineering
	// entry points for true recipes either way.
	std::unordered_set<void*> recipeBltSources;		// wrapper ptrs that have fed a recorded Blt
	struct LockWPend { DWORD eip; RECT rect; bool hasRect; };
	std::unordered_map<void*, std::vector<LockWPend>> lockWPending;
	static const size_t kRecipeMaxEntries = 64;		// per-surface component cap (composites observed are ~4-16 Blts)
	static const size_t kRecipeMaxSurfaces = 8192;	// 32-bit OOM guard: ~8k * ~0.3KB typical = ~2.5MB

	// --- Canonical Identity Layer (2026-06-09) ---
	// DK2 re-composites pool/atlas surfaces with per-session byte drift (A4R4G4B4
	// alpha-bit noise + Blt ordering), so the same visual content presents different
	// Remix hashes across sessions and per-hash mod replacements rot. Instead of
	// stabilizing the GAME, we stabilize what Remix SEES: at whole-surface bind time,
	// resolve the bound content to a frozen canonical texture (raw A4R4G4B4 bytes
	// from _canonical\tex\<HASH>.a4r4, captured 2026-05-29) and bind THAT. The
	// canonical's Remix hash equals its filename by construction (the file holds the
	// exact packed mip-0 bytes), so every drifted variant converges onto one stable,
	// offline-known hash -- mods authored against that hash hit every session.
	// Resolution order: exact-hash identity (already canonical -> no rebind) ->
	// warm map (canon_map.csv, harvested from prior sessions' FPMATCH log lines) ->
	// 32x32-L2 fingerprint match vs canon_fps.bin (same metric as A.10, threshold
	// kFingerprintL2Threshold). Non-matching content (minimap, live composites) is
	// bound untouched. SetDirtyFlag invalidates per-texture resolutions on content
	// writes; a churn cap permanently exempts high-frequency writers (animations)
	// so they never pay per-write rehashing (mirrors Phase 8 scarcity philosophy).
	struct CanonFpEntry { uint64_t hash; uint16_t w, h; uint8_t fp[1024]; };
	std::vector<CanonFpEntry> canonFps;                                  // fingerprint table of canonical contents
	std::unordered_map<uint64_t, uint64_t> canonMapRuntimeToCanon;       // warm map: runtime hash -> canonical hash
	std::unordered_set<uint64_t> canonHashSet;                           // all canonical hashes (identity short-circuit)
	std::unordered_map<uint64_t, IDirect3DTexture9*> canonTexByHash;     // lazy-created canonical textures (nullptr = load failed, don't retry)
	std::unordered_map<IDirect3DTexture9*, IDirect3DTexture9*> canonResolveByTex; // bind-time resolution cache (nullptr = resolved, no rebind)
	std::unordered_map<IDirect3DTexture9*, DWORD> canonChurnByTex;       // SetDirtyFlag invalidation count per texture
	bool canonSidecarsLoaded = false;
	bool canonDisabled = false;        // sidecars missing/corrupt -> layer inert this session
	DWORD canonRebindBindsTotal = 0;   // binds redirected to a canonical (per-bind)
	DWORD canonFpMatchTotal = 0;       // unique contents resolved via fingerprint
	DWORD canonFpNoMatchTotal = 0;     // unique contents with no canonical (expected: minimap/dynamic)
	DWORD canonIdentityTotal = 0;      // unique contents already presenting a canonical hash
	DWORD canonTexCreatedTotal = 0;    // canonical textures synthesized from .a4r4 files
	DWORD canonChurnExemptTotal = 0;   // textures permanently exempted by the churn cap
	DWORD canonFpRejectTotal = 0;      // fingerprint matches REJECTED by the pixel verifier (would-be false positives)
	static const size_t kCanonTexCap = 1500;   // hard cap on synthesized canonicals (32-bit address space discipline)
	static const DWORD kCanonChurnCap = 32;    // SetDirtyFlag invalidations before permanent exemption
	// Pixel verifier (added after the 2026-06-09 session audit): the 32x32
	// fingerprint cannot see quadrant-level content swaps (co-atlased sprite
	// surfaces, text composites), so a fingerprint match alone may pair contents
	// that share layout but differ in a region. Before accepting, compare the
	// runtime bytes to the canonical payload: a pixel is a "big diff" when any
	// RGB nibble differs by more than kCanonVerifyBigDiffNibbles; reject when
	// more than 1% of pixels are big-diffs. Mirrors the offline gate that
	// derives canon_map.csv (canon_assign_map.py), so runtime acceptance and
	// offline assignment have identical semantics.
	static const int kCanonVerifyBigDiffNibbles = 2;

	inline uint16_t QuantizeUV(float u)
	{
		if (u < 0.0f) u = 0.0f;
		if (u > 1.0f) u = 1.0f;
		return (uint16_t)(u * 4096.0f + 0.5f);
	}

	// Compute 32x32 grayscale fingerprint by scaling source surface to a
	// 32x32 A8R8G8B8 system-memory target, then luma-converting per pixel.
	// Returns false on any d3d9 error.
	bool ComputeFingerprintFromTexture(IDirect3DTexture9* srcTexture, LPDIRECT3DDEVICE9* d3d9Device, uint8_t outFp[1024])
	{
		if (!srcTexture || !d3d9Device || !*d3d9Device) return false;
		IDirect3DSurface9* srcLevel = nullptr;
		if (FAILED(srcTexture->GetSurfaceLevel(0, &srcLevel)) || !srcLevel) return false;
		IDirect3DSurface9* tmp = nullptr;
		HRESULT hr = (*d3d9Device)->CreateOffscreenPlainSurface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &tmp, nullptr);
		if (FAILED(hr) || !tmp) { srcLevel->Release(); return false; }
		hr = D3DXLoadSurfaceFromSurface(tmp, nullptr, nullptr, srcLevel, nullptr, nullptr, D3DX_FILTER_LINEAR, 0);
		srcLevel->Release();
		if (FAILED(hr)) { tmp->Release(); return false; }
		D3DLOCKED_RECT lr = {};
		if (FAILED(tmp->LockRect(&lr, nullptr, D3DLOCK_READONLY))) { tmp->Release(); return false; }
		// A8R8G8B8 in memory is byte order BGRA. ITU-R BT.601 luma.
		for (int y = 0; y < 32; y++)
		{
			const uint8_t* row = (const uint8_t*)lr.pBits + y * lr.Pitch;
			for (int x = 0; x < 32; x++)
			{
				uint8_t b = row[x*4 + 0];
				uint8_t g = row[x*4 + 1];
				uint8_t r = row[x*4 + 2];
				outFp[y*32 + x] = (uint8_t)((r*299 + g*587 + b*114) / 1000);
			}
		}
		tmp->UnlockRect();
		tmp->Release();
		fingerprintComputeTotal++;
		return true;
	}

	// L2 distance between two 1024-byte fingerprints (sqrt of sum of squared diffs).
	float FingerprintL2(const uint8_t a[1024], const uint8_t b[1024])
	{
		uint64_t sum = 0;
		for (int i = 0; i < 1024; i++)
		{
			int d = (int)a[i] - (int)b[i];
			sum += (uint64_t)(d * d);
		}
		return sqrtf((float)sum);
	}

	// Find the best family match for a fingerprint. Returns family index or -1.
	int FindFamilyForFingerprint(const uint8_t fp[1024], float* outBestL2 = nullptr)
	{
		const size_t numFamilies = sizeof(kAtlasFamilies) / sizeof(kAtlasFamilies[0]);
		int bestIdx = -1;
		float bestL2 = 1e30f;
		for (size_t i = 0; i < numFamilies; i++)
		{
			float d = FingerprintL2(fp, kAtlasFamilies[i].fp);
			if (d < bestL2) { bestL2 = d; bestIdx = (int)i; }
		}
		if (outBestL2) *outBestL2 = bestL2;
		return (bestL2 < kFingerprintL2Threshold) ? bestIdx : -1;
	}

	// UV match: drawcall UV bounds (u_min,v_min,u_max,v_max) contained within
	// region's rectangle with a small slack (UV jitter from rasterizer rounding).
	bool UVRectContains(const UVRect& r, float u_min, float v_min, float u_max, float v_max)
	{
		constexpr float SLACK = 0.02f;
		return u_min >= r.u0 - SLACK && u_max <= r.u1 + SLACK
			&& v_min >= r.v0 - SLACK && v_max <= r.v1 + SLACK;
	}

	// Phase A.7 (2026-05-18): continuous content capture.
	// Hook fires inside CopyToDrawTexture immediately after D3DXLoadSurfaceFromSurface
	// lands fresh content in surface.DrawTexture (the layer Remix's bridge sees on
	// upload, per dk2_hash_mapping memory). For each new content-hash this session,
	// dump the bytes as a PNG into <gamedir>\_capture_phase_a7\<hash>.png and
	// append a manifest CSV row. Run over a full play session to enumerate every
	// unique texture content the game produces -- the post-processing step then
	// maps each captured PNG to a PBRify replacement via the existing L2 visual
	// fingerprint workflow.
	//
	// Hash is XXH3_64bits over tightly-packed mip-0 bytes (2026-05-20), byte-exact
	// match to dxvk-remix. PNG filenames + TEX_HASH log values therefore EQUAL
	// Remix capture filenames and community PBR mod filenames -- replacements drop
	// in by hash name with no manual correlation. (Was FNV-1a until the xxhash.h
	// integration; FNV was session-dedup only and Remix-incompatible.)
	struct ContentCaptureState {
		bool initialized = false;
		std::string captureDir;
		std::ofstream manifest;
		std::unordered_set<uint64_t> seenHashes;
		// Phase A.7 v2 (2026-05-18): per-surface "first bind seen" dedup so the
		// bind-time hook in IDirect3DDeviceX::SetTexture only captures each
		// surface once -- the SetDirtyFlag hook handles subsequent content
		// changes via content-hash dedup.
		std::unordered_set<void*> firstBindSeen;
		// Phase A.7 v3 (2026-05-18 eve): map from d3d9 texture pointer to its
		// first-observed content hash. Logged once per (d3d9_ptr) so offline
		// analysis can cross-reference draw-log tex= entries to capture PNGs.
		std::unordered_map<void*, uint64_t> firstHashByTex;
		DWORD bltSeenTotal = 0;
		DWORD bindSeenTotal = 0;
		DWORD savedTotal = 0;
	};
	ContentCaptureState capState;

	// Remix-compatible content hash: XXH3_64bits over tightly-packed bytes.
	// Matches dxvk-remix exactly so our hashes equal Remix capture filenames.
	uint64_t ComputeContentHash(const void* data, size_t size)
	{
		return static_cast<uint64_t>(XXH3_64bits(data, size));
	}

	void EnsureContentCaptureInit()
	{
		if (capState.initialized) return;
		capState.initialized = true;
		char modulePath[MAX_PATH] = {};
		GetModuleFileNameA(nullptr, modulePath, MAX_PATH);
		std::string gameDir = modulePath;
		auto slash = gameDir.find_last_of("\\/");
		if (slash != std::string::npos) gameDir = gameDir.substr(0, slash);
		// Phase A.7 v2: per-run timestamped dir so multiple runs don't clobber
		// each other -- enables determinism testing (compare PNG sets across runs).
		SYSTEMTIME st;
		GetLocalTime(&st);
		char ts[32];
		sprintf_s(ts, sizeof(ts), "_v2_%04d%02d%02d_%02d%02d%02d",
			st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
		capState.captureDir = gameDir + "\\_capture_phase_a7" + ts;
		std::error_code ec;
		std::filesystem::create_directories(capState.captureDir, ec);
		// Origin-tagged buckets (2026-05-21): `world` = original game textures
		// (PBRify targets; hashes unchanged by the torch fix). `crops` = textures the
		// universal decompose synthesized (torches + co-atlased sprites) -- leave stock,
		// PBRify last. Lets the PBRify batch skip torches by BUCKET, no visual hunting.
		std::filesystem::create_directories(capState.captureDir + "\\world", ec);
		std::filesystem::create_directories(capState.captureDir + "\\crops", ec);
		std::string manifestPath = capState.captureDir + "\\manifest.csv";
		capState.manifest.open(manifestPath, std::ios::out | std::ios::app);
		if (capState.manifest.is_open())
		{
			capState.manifest << "hash,width,height,format,bltSeenTotal,origin\n";
			capState.manifest.flush();
		}
		Logging::Log() << "[A.7] content capture initialized -> " << capState.captureDir.c_str();
	}

	// Hash dest surface mip-0 bytes (skipping pitch padding so the hash is independent
	// of allocation alignment). If new, dump PNG via D3DXSaveSurfaceToFileInMemory.
	// `originatingTex` (optional): the d3d9 texture this surface came from --
	// when supplied AND we've never seen this texture before, emit a TEX_HASH
	// log line so offline analysis can join draw-log `tex=` entries to PNG files.
	void CaptureSurfaceContent(IDirect3DSurface9* destSurface, void* originatingTex = nullptr, bool isCrop = false)
	{
		if (!destSurface) return;
		EnsureContentCaptureInit();
		capState.bltSeenTotal++;
		D3DSURFACE_DESC desc = {};
		if (FAILED(destSurface->GetDesc(&desc))) return;
		D3DLOCKED_RECT lr = {};
		if (FAILED(destSurface->LockRect(&lr, nullptr, D3DLOCK_READONLY))) return;
		// Hash row-by-row to skip allocator pitch padding. Bytes-per-pixel must
		// come from the surface format -- DK2's per-instance textures land in
		// A4R4G4B4 (2 bpp), NOT A8R8G8B8 (4 bpp). Earlier v1/v2 captures used
		// a hardcoded width*4 which read 16 KB of uninitialized garbage past
		// every A4R4G4B4 texel buffer, producing non-deterministic hashes that
		// varied per run (hashes had zero overlap across two runs of the same
		// scene). With the correct bpp the hash is fully deterministic.
		const DWORD bpp = GetBitCount(desc.Format) / 8;
		if (bpp == 0) { destSurface->UnlockRect(); return; }
		// XXH3_64bits over tightly-packed mip-0 bytes -- byte-exact match to
		// dxvk-remix's imageHash = XXH3_64bits(stagingBuffer, size). DXVK stages
		// the upload tightly packed (no row padding), so we feed each row's
		// width*bpp bytes via the streaming API, skipping the driver's lr.Pitch
		// padding. The streaming digest equals a one-shot over the concatenation.
		// rowBytes uses the real bpp (DK2 per-instance textures are A4R4G4B4 = 2bpp,
		// not 4) so we don't hash uninitialized padding -> deterministic + matches Remix.
		uint64_t h = 0;
		const UINT rowBytes = desc.Width * bpp;
		XXH3_state_t* xstate = XXH3_createState();
		if (xstate)
		{
			XXH3_64bits_reset(xstate);
			const uint8_t* base = static_cast<const uint8_t*>(lr.pBits);
			for (UINT y = 0; y < desc.Height; ++y)
			{
				const uint8_t* row = base + static_cast<size_t>(y) * lr.Pitch;
				UINT bytesThisRow = rowBytes;
				if (bytesThisRow > static_cast<UINT>(lr.Pitch)) bytesThisRow = static_cast<UINT>(lr.Pitch);
				XXH3_64bits_update(xstate, row, bytesThisRow);
			}
			h = static_cast<uint64_t>(XXH3_64bits_digest(xstate));
			XXH3_freeState(xstate);
		}
		destSurface->UnlockRect();
		// Emit TEX_HASH mapping once per (d3d9 texture pointer) so offline
		// tooling can resolve draw-log tex= entries to a representative PNG.
		if (originatingTex && capState.firstHashByTex.find(originatingTex) == capState.firstHashByTex.end())
		{
			capState.firstHashByTex[originatingTex] = h;
			char buf[96];
			sprintf_s(buf, sizeof(buf), "TEX_HASH d3d9=%p hash=%016llX", originatingTex, (unsigned long long)h);
			Logging::Log() << buf;
		}
		if (!capState.seenHashes.insert(h).second) return; // already saved this session
		// New hash -- save PNG
		LPD3DXBUFFER pBuffer = nullptr;
		if (SUCCEEDED(D3DXSaveSurfaceToFileInMemory(&pBuffer, D3DXIFF_PNG, destSurface, nullptr, nullptr)) && pBuffer)
		{
			char pngPath[MAX_PATH];
			sprintf_s(pngPath, sizeof(pngPath), "%s\\%s\\%016llX.png", capState.captureDir.c_str(),
				isCrop ? "crops" : "world", (unsigned long long)h);
			HANDLE fh = CreateFileA(pngPath, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (fh != INVALID_HANDLE_VALUE)
			{
				DWORD written = 0;
				if (WriteFile(fh, pBuffer->GetBufferPointer(), (DWORD)pBuffer->GetBufferSize(), &written, nullptr))
				{
					capState.savedTotal++;
				}
				CloseHandle(fh);
			}
			pBuffer->Release();
		}
		if (capState.manifest.is_open())
		{
			char buf[160];
			sprintf_s(buf, sizeof(buf), "%016llX,%u,%u,%u,%u,%s\n",
				(unsigned long long)h, desc.Width, desc.Height, (UINT)desc.Format, capState.bltSeenTotal,
				isCrop ? "crop" : "world");
			capState.manifest << buf;
			if ((capState.savedTotal & 0x0F) == 0) capState.manifest.flush();
		}
	}

	// Phase A.7 v2 entry point invoked from IDirect3DDeviceX::SetTexture for each
	// texture about to be bound for drawing. Captures content of static textures
	// that never trigger SetDirtyFlag (loaded once at init, never modified).
	// One-shot per surface pointer -- the SetDirtyFlag hook re-captures on changes.
	void CaptureSurfaceForPhaseA7FirstBind(IDirect3DTexture9* texture, void* wrapperPtr)
	{
		if (!Config.DdrawContentCapture || !texture) return;
		if (!capState.firstBindSeen.insert(texture).second) return;
		capState.bindSeenTotal++;
		IDirect3DSurface9* level0 = nullptr;
		if (FAILED(texture->GetSurfaceLevel(0, &level0)) || !level0) return;
		// Pass wrapper pointer so TEX_HASH log uses same key as DRAW_XYZRHW tex=
		CaptureSurfaceContent(level0, wrapperPtr);
		level0->Release();
	}

	// DKII.exe (main module) address range, resolved once.
	uintptr_t gameModBase = 0, gameModEnd = 0;
	void EnsureGameModuleBounds()
	{
		if (gameModEnd)
		{
			return;
		}
		HMODULE h = GetModuleHandleW(nullptr);
		if (!h)
		{
			return;
		}
		IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)h;
		if (dos->e_magic != IMAGE_DOS_SIGNATURE)
		{
			return;
		}
		IMAGE_NT_HEADERS* nt = (IMAGE_NT_HEADERS*)((BYTE*)h + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE)
		{
			return;
		}
		gameModBase = (uintptr_t)h;
		gameModEnd = gameModBase + nt->OptionalHeader.SizeOfImage;
	}

	// Scan the raw stack upward for the first return address that lands inside
	// DKII.exe and is immediately preceded by a relative CALL (0xE8). That is the
	// game-side call site that asked for this surface -- the bridge into Phase B.
	DWORD FindGameCaller()
	{
		EnsureGameModuleBounds();
		if (!gameModEnd)
		{
			return 0;
		}
		uintptr_t* sp = (uintptr_t*)_AddressOfReturnAddress();
		for (int i = 0; i < 256; i++)
		{
			uintptr_t v = sp[i];
			if (v <= gameModBase + 0x1000 || v >= gameModEnd)
			{
				continue;
			}
			if (*(BYTE*)(v - 5) == 0xE8)	// rel32 CALL
			{
				return (DWORD)v;
			}
		}
		return 0;
	}

	// --- [OWNUP] source-identity detour (2026-06-10, recipe step 5) ---------------
	// Static RE (re_owner_trace.py) established that DK2's texture descriptors carry a
	// NAME INDEX at +0x24 that resolves, via the name table at [0x78E564] (3-dword
	// records, record[0] = char* fullname), to an EngineTextures entry name such as
	// "GUI\C-Spells\c-windMM2". The cache loader 0x58C170 is called __thiscall with
	// ecx = the descriptor and reads [this+0x24] for exactly this lookup. The OPEN
	// QUESTION is whether the descriptor present at the dominant upload blit
	// (DKII.exe 0x58E53B, 78% of staging writes; ESI = the size-class pool slot, [ESI]
	// = source content surface, [ESI+0x04] = owner back-link) carries a live, resolvable
	// +0x24 -- i.e. whether STABLE source identity is available right where the atlas is
	// composed. If yes, replacements can be authored against identity, and DK2's
	// per-session hash drift (its in-house LRU atlas cache repacking pages) stops
	// mattering.
	//
	// This installs a read-only call-site detour: it rewrites the single rel32 of the
	// `call 0x58B770` at 0x58E53B to jump through a trampoline that logs the descriptor's
	// identity fields, then tail-jumps to the real blitter (which `ret 0xC`s straight
	// back to 0x58E540). The blit is byte-for-byte unchanged; we only observe. Precedent
	// for patching DKII.exe code: the zoom patch (WriteMemory) and the April pretransform
	// hook (DdrawEnablePreTransformHook). All game-memory reads are SEH-guarded.
	uintptr_t ownupBlitTarget = 0;        // rebased 0x58B770 -- value the trampoline jmps to
	uintptr_t ownupNameTableAddr = 0;     // rebased &[0x78E564] -- holds the name-table base ptr
	volatile LONG ownupInstallTried = 0;
	DWORD ownupCalls = 0, ownupBadReads = 0;
	std::unordered_set<uint32_t> ownupSeen;   // dedup by resolved identity key (bounds log volume)

	struct OwnupSnap {
		bool ok;
		void* content; void* owner;
		uint32_t vtbl, flags, idxA, idxB;
		int sw, sh, pw, ph;
		float uv0, uv1, uv2, uv3;
		char nameA[96]; char nameB[96];
		bool nameAok, nameBok;
	};

	// POD-only reads inside SEH (no unwindable C++ objects -> avoids C2712).
	void OwnupSnapshot(void* slot, OwnupSnap* s)
	{
		s->ok = false; s->nameAok = false; s->nameBok = false;
		s->nameA[0] = 0; s->nameB[0] = 0;
		__try {
			BYTE* d = (BYTE*)slot;
			s->content = *(void**)(d + 0x00);
			s->owner   = *(void**)(d + 0x04);
			s->idxA    = *(uint32_t*)(d + 0x24);
			s->uv0     = *(float*)(d + 0x28);
			s->uv1     = *(float*)(d + 0x2c);
			s->uv2     = *(float*)(d + 0x30);
			s->uv3     = *(float*)(d + 0x34);
			s->flags   = *(uint32_t*)(d + 0x38);
			s->sw      = *(BYTE*)(d + 0x3c);
			s->sh      = *(BYTE*)(d + 0x3d);
			s->pw      = *(BYTE*)(d + 0x3e);
			s->ph      = *(BYTE*)(d + 0x3f);
			s->vtbl    = s->content ? *(uint32_t*)s->content : 0;
			s->idxB    = s->owner ? *(uint32_t*)((BYTE*)s->owner + 0x24) : 0xFFFFFFFFu;
			s->ok = true;
		} __except (EXCEPTION_EXECUTE_HANDLER) { s->ok = false; }
	}

	// Resolve a candidate name index through the runtime-built name table. Returns true
	// and fills out[] (NUL-terminated, printable ASCII) only on a clean resolution.
	bool OwnupResolveName(uint32_t idx, char out[96])
	{
		out[0] = 0;
		if (idx >= 200000u || ownupNameTableAddr == 0) return false;
		bool ok = false;
		__try {
			uintptr_t tableBase = *(uintptr_t*)ownupNameTableAddr;
			if (tableBase) {
				const char* name = *(const char**)(tableBase + (uintptr_t)idx * 12u);
				if (name) {
					int i = 0;
					for (; i < 95; i++) {
						char c = name[i];
						if (c == 0) break;
						if ((unsigned char)c < 32 || (unsigned char)c > 126) { i = 0; break; }
						out[i] = c;
					}
					out[i] = 0;
					ok = (i > 0);
				}
			}
		} __except (EXCEPTION_EXECUTE_HANDLER) { out[0] = 0; ok = false; }
		return ok;
	}

	// __cdecl logger called by the naked trampoline with ESI (the upload descriptor).
	void __cdecl OwnupLogSlot(void* slot)
	{
		ownupCalls++;
		OwnupSnap s;
		OwnupSnapshot(slot, &s);
		if (!s.ok) { ownupBadReads++; return; }
		s.nameAok = OwnupResolveName(s.idxA, s.nameA);
		s.nameBok = OwnupResolveName(s.idxB, s.nameB);

		// Dedup key prefers a resolvable name index (slot first, then owner); falls back
		// to the slot pointer so unresolved variants still log once without flooding.
		uint32_t key;
		if (s.nameAok)      key = s.idxA;
		else if (s.nameBok) key = s.idxB ^ 0x40000000u;
		else                key = 0x80000000u | ((uint32_t)(uintptr_t)slot & 0x3FFFFFFFu);

		if (!ownupSeen.insert(key).second) return;   // identity already logged
		if (ownupSeen.size() > 6000) return;          // 32-bit flood/OOM guard

		char buf[460];
		sprintf_s(buf, sizeof(buf),
			"[OWNUP] slot=%p content=%p vtbl=%08X flags=%08X srcwh=%dx%d padwh=%dx%d "
			"idxA=%u nameA=%s%s%s owner=%p idxB=%u nameB=%s%s%s uv=%.4f,%.4f,%.4f,%.4f",
			slot, s.content, s.vtbl, s.flags, s.sw, s.sh, s.pw, s.ph,
			s.idxA, s.nameAok ? "\"" : "", s.nameAok ? s.nameA : "(none)", s.nameAok ? "\"" : "",
			s.owner, s.idxB, s.nameBok ? "\"" : "", s.nameBok ? s.nameB : "(none)", s.nameBok ? "\"" : "",
			s.uv0, s.uv1, s.uv2, s.uv3);
		Logging::Log() << buf;
	}

	// POD-only SEH helper, kept separate from OwnupInstallHookOnce: that function builds
	// Logging stream temporaries (objects requiring unwinding), which cannot coexist with
	// __try in one function (C2712). Verifies the call-site still holds `E8 ->0x58B770`.
	bool OwnupVerifyCallSite(const BYTE* p, uintptr_t callSite, int32_t* outRel)
	{
		bool sane = false;
		__try {
			if (p[0] == 0xE8) {
				int32_t rel = *(const int32_t*)(p + 1);
				*outRel = rel;
				sane = (callSite + 5 + (intptr_t)rel == ownupBlitTarget);
			}
		} __except (EXCEPTION_EXECUTE_HANDLER) { sane = false; }
		return sane;
	}

	// --- [NAMEKEY] B-v1: name-keyed canonicalization (2026-06-11) -------------------
	// The OWNUP run proved the descriptor at the 0x58E53B upload blit carries stable
	// EngineTextures identity (614/614 names resolved, 0 bad reads). This block turns
	// that fact into a resolver. The SAME read-only call-site detour now RECORDS each
	// placement {nameIdx, mip, destX, destY, w, h} into an accumulator; the
	// full-surface staging->texture Blt (the [RECIPE]-proven upload shape, runs A/B:
	// ~99% of dest content arrives exactly that way) attaches the accumulated
	// placements to the dest texture as its TRUE composition recipe; at bind, a
	// single-name FULL-page recipe resolves through _canonical\name_map.csv
	// (key = fullname + "MM{mip}", what cache loader 0x58C170 sprintf's) to a
	// canonical by EXACT identity -- replacing the fragile 32x32-L2 fingerprint
	// retrieval for everything the detour saw. Game memory is never written (beyond
	// the one rel32 call-site redirect the OWNUP run proved safe); the "write" is the
	// existing d3d9 rebind. Composites (ncomp>1 / sub-page sprites) stay on the
	// fingerprint path until shadow recomposition (v2).
	// DdrawNameKey: 1 = shadow mode (record+resolve+verify but the legacy path stays
	// authoritative; logs agreement stats), 2 = name key authoritative (pixel-verified,
	// legacy fingerprint as fallback). Requires DdrawCanonicalRebind=1 to take effect.
	struct NamekeyPlacement { uint32_t idx; uint32_t mip; int32_t a2, a3; int32_t sw, sh; float u0, v0, u1, v1; bool isWater = false; };
	std::vector<NamekeyPlacement> namekeyAccum;       // placements since the current composition started
	bool namekeyConsumedSinceLast = false;            // a consume happened; next placement starts a new composition
	// V2 (2026-06-11): comps retains the FULL placement list (the accumulator copy) so the
	// crop layer can match draw rects to sprites; the scalar fields stay the comp0 summary
	// the v1 whole-page resolver reads. u0..v1 = the descriptor's four UV floats (+0x28..0x34,
	// 0x58E3E0 computes them with the half-texel 0.5 constant) -- telemetry for now.
	struct NamekeyRecipe { uint32_t idx; uint32_t mip; uint32_t ncomp; bool fullPage; bool dynUiSolo; bool hasWater = false; int32_t x0, y0, sw0, sh0; std::vector<NamekeyPlacement> comps; };
	std::unordered_map<IDirect3DTexture9*, NamekeyRecipe> namekeyRecipeByTex;  // dest tex -> last attached recipe (no erase on release; same precedent as stage3 maps)
	std::unordered_map<uint32_t, std::string> namekeyNameByIdx;                // resolved-name cache (~hundreds of entries)
	std::unordered_map<std::string, uint64_t> namekeyNameToCanon;              // name_map.csv, keys lowercased
	const size_t kNamekeyMaxAccum = 64;
	const size_t kNamekeyMaxRecipes = 8192;
	DWORD namekeyPlacements = 0, namekeyAccumOverflow = 0, namekeyConsumes = 0, namekeyOrphanConsumes = 0,
		namekeyMultiComp = 0, namekeyResolves = 0, namekeyVerifyFails = 0, namekeyNoRecipe = 0,
		namekeyNotSingleMulti = 0, namekeyNotSingleSub = 0, namekeyUnmapped = 0, namekeyShadowAgree = 0,
		namekeyShadowDisagree = 0, namekeyShadowOnly = 0, namekeyRawLogged = 0, namekeyConsumeLogged = 0;
	// Run-2 telemetry (2026-06-11): dedup'd one-line inventories of the three
	// populations the shadow run exposed. UNMAPPED = the v1.5 authoring list;
	// SUBPAGE (single sprite, stale rest-of-page -- why fpReject ran 2943/run and
	// those hashes can never be stable) + MULTI (composite name-sets, recurring or
	// not) = the v2 scoping data.
	std::unordered_set<uint64_t> namekeyMultiSeen;
	std::unordered_set<uint32_t> namekeySubSeen;
	std::unordered_set<std::string> namekeyUnmappedSeen;
	DWORD namekeyMultiLogged = 0, namekeySubLogged = 0, namekeyUnmappedLogged = 0;

	// --- [NAMEKEY] V2: placement-keyed crop resolution (2026-06-11) -------------------
	// The crop counterpart of the v1 whole-page resolver. crop_payloads.bin holds the
	// expected A4R4G4B4 payload for EVERY EngineTextures entry (5767, built offline by
	// crop_payloads_build.py from extracted\ decodes, truncate quantization). At draw
	// time, a snapped UV rect contained in exactly one recipe placement resolves to
	// (nameIdx, mip); the live page rect is verified against the payload with the SAME
	// nibble gate as CanonVerifyAgainstPayload, and the bound sub-texture is built FROM
	// the payload -- mip-0 verbatim, so its Remix hash == XXH3(payload), known offline
	// for the whole corpus (_scratch_discovery\crop_hashes.csv = the authoring table).
	// Verification runs on EVERY univDrawCache insert, not just first synthesis: both
	// global-cache poisoning AND stale-recipe misbinds fail closed to the legacy crop.
	struct NamekeyCropIdxEnt { uint32_t off; uint16_t w, h; };
	std::unordered_map<uint64_t, NamekeyCropIdxEnt> namekeyCropIdx;	// XXH3(lowercased "nameMMn") -> payload location
	// Per-(nameIdx<<3|mip) resolution state. st: 1 = VERIFIED (tex + payload resident,
	// ~7KB avg, working set a few MB), -1 = permanent fail (no sidecar/dims/create),
	// 2 = shadow-logged (mode 1; payload dropped). Verify failures at serve time do NOT
	// poison the state -- they are per-draw (stale recipe) and retried naturally.
	struct NamekeyCropState { int st = 0; bool dynUi = false; bool isPointer = false; bool isWater = false; IDirect3DTexture9* tex = nullptr; uint16_t w = 0, h = 0; std::vector<uint8_t> payload; };
	std::unordered_map<uint64_t, NamekeyCropState> namekeyCropByKey;
	// payload hash -> texture, the OWNER reference (never released; session lifetime,
	// same precedent as canonTexByHash). Byte-identical twins (chicken/Unique_chicken)
	// share one texture. univDrawCache entries AddRef on insert / Release on evict.
	std::unordered_map<uint64_t, IDirect3DTexture9*> namekeyCropTexByHash;
	static const size_t kNamekeyCropTexCap = 2048;	// 32-bit discipline: <=2048 * ~45KB chain = ~90MB worst, working set ~25MB
	DWORD namekeyCropMatch = 0, namekeyCropAmbiguous = 0, namekeyCropSubmiss = 0,
		namekeyCropNoSidecar = 0, namekeyCropDimsMismatch = 0, namekeyCropBadFormat = 0,
		namekeyCropReadFail = 0, namekeyCropVerifyFail = 0, namekeyCropServeVerifyFail = 0,
		namekeyCropSynth = 0, namekeyCropServe = 0, namekeyCropShadowPass = 0,
		namekeyCropShadowFail = 0, namekeyCropTexCreated = 0, namekeyCropMm0Collapse = 0;
	DWORD namekeyCropLogged = 0, namekeyCropShadowLogged = 0;
	DWORD namekeyDynUiDraws = 0, namekeyWaterCollapse = 0;

	// --- Dynamic-UI emissive classification (2026-06-11) ------------------------------
	// The mouse pointer (PointerInfo Page* strips), tooltips (ToolTip Page*), path
	// subtitles, and the minimap (map_texture) are RENDERED AT RUNTIME -- no
	// EngineTextures source, no stable hash, so no mod entry can ever make them
	// visible. But they ARE name-identified by the placement system, and visibility
	// under a path tracer is a render-state question: forcing those draws to
	// additive blending makes Remix treat them as emissive (the torch-flame
	// mechanism), self-lit in the dark. Classification happens here (name level);
	// IDirect3DDeviceX.cpp wraps the actual draw via the extern accessors below.
	bool g_namekeyDrawDynUi = false;	// additive-eligible dynamic UI (tooltips/map/subtitles), set during TryGetUniversalSubTextureForUV
	bool g_namekeyDrawDynUiAny = false;	// ANY dynamic UI incl. pointer strips (drives DYNUI CROP hash logging)
	bool g_namekeyDrawWater = false;	// water-surface draw (WaterN/WaterSOLIDN placement) -> device flattens its world heights
	DWORD namekeyWaterFlatten = 0;		// draws flattened (counted device-side via NamekeyNoteWaterFlatten)
	DWORD namekeyWaterLarge = 0;		// draws classified water by the gate-free recipe check (large bodies the crop layer never sees)
	std::unordered_set<uint64_t> namekeyDynUiCropLogged;
	DWORD namekeyDynUiCropLogLines = 0;

	bool NamekeyNameIsDynamicUi(const char* lowerName)
	{
		return !strncmp(lowerName, "pointerinfo", 11) || !strncmp(lowerName, "tooltip", 7) ||
			!strncmp(lowerName, "map_texture", 11) || !strncmp(lowerName, "followpathsubt", 14);
	}

	// WaterN / WaterSOLIDN sprite names (digit required: bare "Water"/"WaterSOLID"
	// would also match non-frame entries). Works on the bare lowercased name AND on
	// the "nameMMn" crop key -- the char after the prefix is a digit in both.
	bool NamekeyNameIsWater(const char* lowerName)
	{
		return (!strncmp(lowerName, "watersolid", 10) && isdigit((unsigned char)lowerName[10])) ||
			(!strncmp(lowerName, "water", 5) && isdigit((unsigned char)lowerName[5]));
	}

	// The mouse pointer is split out of the ADDITIVE treatment (2026-06-11, after the
	// first dynUi run): additive can only ADD light, so the mostly-dark hand sprite
	// stayed invisible while tooltips/map lit up. The pointer's correct treatment is
	// rtx.uiTextures rasterization (original colors on top) -- the Desktop copy
	// proved it persistently; we tag the CURRENT bound hashes via the DYNUI CROP log.
	bool NamekeyNameIsPointer(const char* lowerName)
	{
		return !strncmp(lowerName, "pointerinfo", 11);
	}

	// Resolve-and-cache a descriptor name index (SEH-guarded game-memory read inside).
	const char* NamekeyNameForIdx(uint32_t idx)
	{
		auto nIt = namekeyNameByIdx.find(idx);
		if (nIt == namekeyNameByIdx.end())
		{
			char nm[96];
			if (!OwnupResolveName(idx, nm)) return nullptr;
			nIt = namekeyNameByIdx.emplace(idx, nm).first;
		}
		return nIt->second.c_str();
	}

	// __cdecl recorder called by the naked trampoline: ESI (the descriptor) plus the
	// real blitter's three stack args (static RE step 4: srcSurface/destX/destY; the
	// exact push order is confirmed empirically via the RAW lines below -- whichever
	// arg equals [slot+0x00] is the source surface). destX/destY only matter for v2
	// composites; v1's single-name-full-page test needs only sw/sh, so a swapped
	// provisional order is harmless.
	void __cdecl NamekeyRecordUpload(void* slot, DWORD a1, DWORD a2, DWORD a3)
	{
		if (Config.DdrawOwnerLog) OwnupLogSlot(slot);   // [OWNUP] diagnostics still available under the unified hook
		if (!Config.DdrawNameKey) return;

		OwnupSnap s;
		OwnupSnapshot(slot, &s);
		if (!s.ok) return;

		namekeyPlacements++;
		if (namekeyConsumedSinceLast) { namekeyAccum.clear(); namekeyConsumedSinceLast = false; }
		if (namekeyAccum.size() >= kNamekeyMaxAccum) { namekeyAccumOverflow++; return; }
		NamekeyPlacement p;
		p.idx = s.idxA; p.mip = (s.flags & 7);
		p.a2 = (int32_t)a2; p.a3 = (int32_t)a3; p.sw = s.sw; p.sh = s.sh;
		p.u0 = s.uv0; p.v0 = s.uv1; p.u1 = s.uv2; p.v1 = s.uv3;
		namekeyAccum.push_back(p);

		if (namekeyRawLogged < 8)	// arg-order self-check: a1 should equal the content ptr
		{
			namekeyRawLogged++;
			char buf[260];
			sprintf_s(buf, sizeof(buf), "[NAMEKEY] RAW slot=%p content=%p a1=%08lX a2=%ld a3=%ld idx=%u mip=%u sw=%d sh=%d uv=%.4f,%.4f,%.4f,%.4f",
				slot, s.content, (unsigned long)a1, (long)a2, (long)a3, s.idxA, (unsigned)(s.flags & 7), s.sw, s.sh,
				s.uv0, s.uv1, s.uv2, s.uv3);
			Logging::Log() << buf;
		}
	}

	// Naked call-site trampoline: save full state, record (and optionally [OWNUP]-log),
	// restore, tail-jump to the real blitter. The original `call` already pushed the
	// return address 0x58E540 and the three blit args below it -- 0x58B770 `ret 0xC`s
	// straight back, so a plain jmp (not call) preserves the stack exactly. Offsets:
	// pushad+pushfd = 36 bytes, so at the first mov the args sit at [esp+40/44/48];
	// each push shifts the next target back to +48.
	__declspec(naked) void NamekeyTrampoline()
	{
		__asm {
			pushad
			pushfd
			mov  eax, dword ptr [esp + 48]	; blit arg3
			push eax
			mov  eax, dword ptr [esp + 48]	; blit arg2
			push eax
			mov  eax, dword ptr [esp + 48]	; blit arg1
			push eax
			push esi
			call NamekeyRecordUpload
			add  esp, 16
			popfd
			popad
			jmp  dword ptr [ownupBlitTarget]
		}
	}

	// [NAMEKEY] consume: a full-surface verbatim copy of the composed staging page
	// into a texture. Attach the accumulated placements to the dest. The accumulator
	// is COPIED, not cleared: the game composes once then blts the same staging
	// content to N per-instance textures (Phase A.6 pattern); the next placement
	// arriving after a consume is what starts a fresh composition.
	void NamekeyConsumeAt(IDirect3DTexture9* destTex, DWORD W, DWORD H)
	{
		namekeyConsumes++;
		if (namekeyAccum.empty())
		{
			namekeyOrphanConsumes++;
			namekeyRecipeByTex.erase(destTex);	// an untracked recompose must not leave a stale recipe
			return;
		}
		if (namekeyRecipeByTex.size() >= kNamekeyMaxRecipes && !namekeyRecipeByTex.count(destTex))
		{
			namekeyConsumedSinceLast = true;
			return;
		}
		NamekeyRecipe r;
		r.comps = namekeyAccum;	// V2: full placement list (the crop layer's match set)
		r.ncomp = (uint32_t)namekeyAccum.size();
		r.idx = namekeyAccum[0].idx;
		r.mip = namekeyAccum[0].mip;
		// Dynamic-UI solo page (map_texture etc.): classify once at consume so even
		// FULL-page draws (which never reach the crop layer) can be identified.
		r.dynUiSolo = false;
		if (r.ncomp == 1)
		{
			const char* nm0 = NamekeyNameForIdx(r.idx);
			if (nm0)
			{
				char ln[96];
				strcpy_s(ln, sizeof(ln), nm0);
				for (char* c = ln; *c; ++c) *c = (char)tolower((unsigned char)*c);
				r.dynUiSolo = NamekeyNameIsDynamicUi(ln);
			}
		}
		// Water placements flagged once per consume so the device's gate-free flatten
		// classification (IsNamekeyWaterDraw) costs only a containment scan per draw.
		// Run-11 measurement: large water bodies are 34-340 vert meshes whose UV bbox
		// sits exactly inside ONE Water placement -- they never pass the decompose
		// caller gates, so consume-time is the only place that sees them coming.
		r.hasWater = false;
		for (NamekeyPlacement& q : r.comps)
		{
			q.isWater = false;
			const char* nmw = NamekeyNameForIdx(q.idx);	// cached by idx after first resolve
			if (nmw)
			{
				char lnw[96];
				strcpy_s(lnw, sizeof(lnw), nmw);
				for (char* c = lnw; *c; ++c) *c = (char)tolower((unsigned char)*c);
				q.isWater = NamekeyNameIsWater(lnw);
				r.hasWater |= q.isWater;
			}
		}
		r.x0 = namekeyAccum[0].a2;	// arg order confirmed run 1: a2=destX, a3=destY (a1==content ptr 8/8)
		r.y0 = namekeyAccum[0].a3;
		r.sw0 = namekeyAccum[0].sw;
		r.sh0 = namekeyAccum[0].sh;
		r.fullPage = (r.ncomp == 1 && namekeyAccum[0].sw == (int32_t)W && namekeyAccum[0].sh == (int32_t)H);
		if (r.ncomp > 1)
		{
			namekeyMultiComp++;
			// [NAMEKEY] MULTI: dedup-log distinct compositions (signature over the
			// full placement list). Answers the v2 question: are composite pages
			// recurring (name,pos) sets, or free-form repacks?
			uint64_t sig = 1469598103934665603ull;
			for (const NamekeyPlacement& q : namekeyAccum)
			{
				const uint32_t parts[4] = { q.idx, (q.mip << 16) ^ (uint32_t)(q.sw << 8) ^ (uint32_t)q.sh,
					(uint32_t)q.a2, (uint32_t)q.a3 };
				for (int i = 0; i < 4; i++) { sig ^= parts[i]; sig *= 1099511628211ull; }
			}
			if (namekeyMultiLogged < 80 && namekeyMultiSeen.insert(sig).second)
			{
				namekeyMultiLogged++;
				char part[160];
				sprintf_s(part, sizeof(part), "[NAMEKEY] MULTI dst=%lux%lu n=%u comps=", (unsigned long)W, (unsigned long)H, r.ncomp);
				std::string line = part;
				for (const NamekeyPlacement& q : namekeyAccum)
				{
					const char* nm = NamekeyNameForIdx(q.idx);
					sprintf_s(part, sizeof(part), "%s:MM%u@%ld,%ld:%dx%d|",
						nm ? nm : "?", q.mip, (long)q.a2, (long)q.a3, q.sw, q.sh);
					line += part;
				}
				Logging::Log() << line.c_str();
			}
		}
		namekeyRecipeByTex[destTex] = r;
		namekeyConsumedSinceLast = true;

		if (namekeyConsumeLogged < 40)
		{
			namekeyConsumeLogged++;
			char buf[180];
			sprintf_s(buf, sizeof(buf), "[NAMEKEY] CONSUME tex=%p dst=%lux%lu ncomp=%u idx0=%u mip=%u sw=%d sh=%d full=%d",
				(void*)destTex, (unsigned long)W, (unsigned long)H, r.ncomp, r.idx, r.mip,
				namekeyAccum[0].sw, namekeyAccum[0].sh, r.fullPage ? 1 : 0);
			Logging::Log() << buf;
		}
	}

	// [NAMEKEY] resolve helper: the bound texture's attached recipe -> canonical hash,
	// or 0 with a counter bump explaining why not. Single-name FULL-page recipes only
	// in v1. nameOut receives the computed "fullnameMM{mip}" key for logging.
	uint64_t NamekeyLookupCanon(IDirect3DTexture9* tex, char* nameOut, size_t nameOutLen)
	{
		if (nameOut && nameOutLen) nameOut[0] = 0;
		auto it = namekeyRecipeByTex.find(tex);
		if (it == namekeyRecipeByTex.end()) { namekeyNoRecipe++; return 0; }
		const NamekeyRecipe& r = it->second;
		if (r.ncomp != 1) { namekeyNotSingleMulti++; return 0; }
		if (!r.fullPage)
		{
			// Single sprite NOT covering the page: the rest of the page is stale LRU
			// residue, so the page hash can never be stable -- the class behind run 1's
			// fpReject=2943. Inventory it for v2 (rebind needs a recomposed canonical
			// at the recorded position, or UV-aware substitution).
			namekeyNotSingleSub++;
			if (namekeySubLogged < 80 && namekeySubSeen.insert(r.idx ^ (r.mip << 28)).second)
			{
				namekeySubLogged++;
				const char* nm = NamekeyNameForIdx(r.idx);
				char buf[220];
				sprintf_s(buf, sizeof(buf), "[NAMEKEY] SUBPAGE name=\"%s\" mip=%u at=%ld,%ld size=%dx%d",
					nm ? nm : "?", r.mip, (long)r.x0, (long)r.y0, r.sw0, r.sh0);
				Logging::Log() << buf;
			}
			return 0;
		}

		const char* nm = NamekeyNameForIdx(r.idx);
		if (!nm) return 0;
		char key[128];
		sprintf_s(key, sizeof(key), "%sMM%u", nm, r.mip);
		if (nameOut && nameOutLen) strcpy_s(nameOut, nameOutLen, key);
		std::string k(key);
		for (auto& c : k) c = (char)tolower((unsigned char)c);
		auto mIt = namekeyNameToCanon.find(k);
		if (mIt == namekeyNameToCanon.end())
		{
			namekeyUnmapped++;
			// The v1.5 authoring list: single-name full-page contents with no
			// canonical yet. Generate from extracted\ sources, add map row + mod entry.
			if (namekeyUnmappedLogged < 400 && namekeyUnmappedSeen.insert(k).second)
			{
				namekeyUnmappedLogged++;
				char buf[220];
				sprintf_s(buf, sizeof(buf), "[NAMEKEY] UNMAPPED key=\"%s\" dims=%dx%d", key, r.sw0, r.sh0);
				Logging::Log() << buf;
			}
			return 0;
		}
		return mIt->second;
	}

	void OwnupInstallHookOnce()
	{
		if (!Config.DdrawOwnerLog && !Config.DdrawNameKey) return;
		if (InterlockedExchange(&ownupInstallTried, 1) != 0) return;

		uintptr_t base = (uintptr_t)GetModuleHandleW(nullptr);
		if (!base) { Logging::Log() << "[OWNUP] install: no module base"; return; }
		const uintptr_t kImgBase = 0x400000;
		uintptr_t callSite   = base + (0x58E53B - kImgBase);
		ownupBlitTarget      = base + (0x58B770 - kImgBase);
		ownupNameTableAddr   = base + (0x78E564 - kImgBase);

		BYTE* p = (BYTE*)callSite;
		DWORD oldProt = 0;
		if (!VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &oldProt))
		{ Logging::Log() << "[OWNUP] install: VirtualProtect failed err=" << GetLastError(); return; }

		int32_t curRel = 0;
		bool sane = OwnupVerifyCallSite(p, callSite, &curRel);

		if (sane) {
			int32_t newRel = (int32_t)((intptr_t)((BYTE*)&NamekeyTrampoline) - (intptr_t)(callSite + 5));
			*(int32_t*)(p + 1) = newRel;   // E8 opcode preserved; only the rel32 changes
			FlushInstructionCache(GetCurrentProcess(), p, 5);
			char b[220];
			sprintf_s(b, sizeof(b), "[NAMEKEY] HOOK INSTALLED base=%p site=%p trampoline=%p blit=%p nametbl=%p ownerlog=%d namekey=%lu",
				(void*)base, (void*)callSite, (void*)&NamekeyTrampoline, (void*)ownupBlitTarget, (void*)ownupNameTableAddr,
				Config.DdrawOwnerLog ? 1 : 0, (unsigned long)Config.DdrawNameKey);
			Logging::Log() << b;
		} else {
			char b[160];
			sprintf_s(b, sizeof(b), "[OWNUP] install ABORTED: unexpected bytes at %p b0=%02X rel=%08X",
				(void*)callSite, (unsigned)p[0], (unsigned)curRel);
			Logging::Log() << b;
		}
		VirtualProtect(p, 5, oldProt, &oldProt);
	}

	// Stage 3 fix (2026-05-29): compute a bound d3d9 texture's content hash the same
	// way CaptureSurfaceContent does -- XXH3_64 over tightly-packed mip-0 rows, skipping
	// allocator pitch padding -- so it equals Remix's substitution key (verified byte-exact
	// for static textures, see dk2_xxh3_remix_hash). Returns 0 on any failure (no surface,
	// unknown bpp, lock failure). Capture-independent: does NOT require DdrawContentCapture.
	uint64_t Stage3HashD3d9Texture(IDirect3DTexture9* tex)
	{
		if (!tex) return 0;
		IDirect3DSurface9* lvl0 = nullptr;
		if (FAILED(tex->GetSurfaceLevel(0, &lvl0)) || !lvl0) return 0;
		uint64_t h = 0;
		D3DSURFACE_DESC desc = {};
		if (SUCCEEDED(lvl0->GetDesc(&desc)))
		{
			const DWORD bpp = GetBitCount(desc.Format) / 8;
			D3DLOCKED_RECT lr = {};
			if (bpp != 0 && SUCCEEDED(lvl0->LockRect(&lr, nullptr, D3DLOCK_READONLY)))
			{
				const UINT rowBytes = desc.Width * bpp;
				XXH3_state_t* xstate = XXH3_createState();
				if (xstate)
				{
					XXH3_64bits_reset(xstate);
					const uint8_t* base = static_cast<const uint8_t*>(lr.pBits);
					for (UINT y = 0; y < desc.Height; ++y)
					{
						const uint8_t* row = base + static_cast<size_t>(y) * lr.Pitch;
						UINT bytesThisRow = rowBytes;
						if (bytesThisRow > static_cast<UINT>(lr.Pitch)) bytesThisRow = static_cast<UINT>(lr.Pitch);
						XXH3_64bits_update(xstate, row, bytesThisRow);
					}
					h = static_cast<uint64_t>(XXH3_64bits_digest(xstate));
					XXH3_freeState(xstate);
				}
				lvl0->UnlockRect();
			}
		}
		lvl0->Release();
		return h;
	}

	// [RECIPE] append one Blt to the destination texture's composition recipe.
	// New-composition detection: a full-surface overwrite OR a re-write of a dst rect
	// already present in the recipe starts a new generation (clear, then append) --
	// per-frame recomposition of the same layout thus yields one frame's recipe, not
	// an unbounded mix of generations.
	void RecipeRecordBlt(IDirect3DTexture9* destTex, uint64_t srcHash, const RECT& s, const RECT& d, DWORD W, DWORD H)
	{
		if (!destTex) return;
		auto it = recipeByTex.find(destTex);
		if (it == recipeByTex.end())
		{
			if (recipeByTex.size() >= kRecipeMaxSurfaces) { recipeSurfacesDropped++; return; }
			it = recipeByTex.emplace(destTex, RecipeState{}).first;
		}
		RecipeState& st = it->second;
		const uint16_t dx = (uint16_t)d.left, dy = (uint16_t)d.top;
		const uint16_t dw = (uint16_t)(d.right - d.left), dh = (uint16_t)(d.bottom - d.top);
		bool resetFirst = (d.left <= 0 && d.top <= 0 && d.right >= (LONG)W && d.bottom >= (LONG)H && !st.entries.empty());
		if (!resetFirst)
		{
			for (const RecipeEntry& e : st.entries)
			{
				if (e.dx == dx && e.dy == dy && e.dw == dw && e.dh == dh) { resetFirst = true; break; }
			}
		}
		if (resetFirst) { st.entries.clear(); st.overflows = 0; st.directWrites = 0; }
		if (st.entries.size() >= kRecipeMaxEntries) { st.overflows++; return; }
		RecipeEntry e;
		e.srcHash = srcHash;
		e.sx = (uint16_t)s.left; e.sy = (uint16_t)s.top;
		e.sw = (uint16_t)(s.right - s.left); e.sh = (uint16_t)(s.bottom - s.top);
		e.dx = dx; e.dy = dy; e.dw = dw; e.dh = dh;
		st.entries.push_back(e);
	}

	// [RECIPE] emit one log line per distinct bound whole-surface content hash:
	// the recipe that produced it (or norecipe=1 for Lock-written / untracked content).
	// rhash = XXH3 over the packed entry array (order-sensitive); comps list lets the
	// offline join also compare order-insensitively.
	void RecipeEmitForHash(IDirect3DTexture9* tex, uint64_t h)
	{
		if (!recipeEmitted.insert(h).second) return;
		auto it = recipeByTex.find(tex);
		char head[160];
		if (it == recipeByTex.end() || it->second.entries.empty())
		{
			sprintf_s(head, sizeof(head), "[RECIPE] hash=%016llX ncomp=0 norecipe=1", (unsigned long long)h);
			Logging::Log() << head;
			return;
		}
		const RecipeState& st = it->second;
		const uint64_t rhash = (uint64_t)XXH3_64bits(st.entries.data(), st.entries.size() * sizeof(RecipeEntry));
		sprintf_s(head, sizeof(head), "[RECIPE] hash=%016llX rhash=%016llX ncomp=%u ov=%u direct=%u comps=",
			(unsigned long long)h, (unsigned long long)rhash, (unsigned)st.entries.size(), st.overflows, st.directWrites);
		std::string line = head;
		char comp[96];
		for (const RecipeEntry& e : st.entries)
		{
			sprintf_s(comp, sizeof(comp), "%016llX:%u,%u,%u,%u->%u,%u,%u,%u|",
				(unsigned long long)e.srcHash, e.sx, e.sy, e.sw, e.sh, e.dx, e.dy, e.dw, e.dh);
			line += comp;
		}
		Logging::Log() << line.c_str();
	}

	// Stage 3 corpus dump (2026-05-29): save the bound whole-surface texture as a PNG
	// keyed by its exact Remix hash + a manifest row. Once per distinct hash. Gated by
	// DdrawContentCapture so normal play has zero overhead. Lets offline tooling visually
	// fingerprint-match each runtime hash to an extracted source PNG, then author an
	// upscaled <hash>_*.dds that Remix will substitute (the hash IS the key).
	void Stage3MaybeDumpBound(IDirect3DTexture9* tex, uint64_t hash, bool isCrop)
	{
		// Flag split 2026-06-09: DdrawStage3BoundDump enables JUST this cheap dump
		// (one PNG per distinct whole hash) without the full A.7 capture's
		// ~60k-files/session overhead. Either flag turns it on.
		if ((!Config.DdrawContentCapture && !Config.DdrawStage3BoundDump) || isCrop || !tex || hash == 0) return;
		if (!stage3DumpedHashes.insert(hash).second) return; // one PNG per distinct hash
		if (!stage3BoundDirReady)
		{
			EnsureContentCaptureInit();
			std::error_code ec;
			stage3BoundDir = capState.captureDir + "\\stage3_bound";
			std::filesystem::create_directories(stage3BoundDir, ec);
			stage3BoundManifest.open(stage3BoundDir + "\\manifest.csv", std::ios::out | std::ios::app);
			if (stage3BoundManifest.is_open()) { stage3BoundManifest << "hash,width,height,format\n"; stage3BoundManifest.flush(); }
			stage3BoundDirReady = true;
		}
		IDirect3DSurface9* lvl0 = nullptr;
		if (FAILED(tex->GetSurfaceLevel(0, &lvl0)) || !lvl0) return;
		D3DSURFACE_DESC desc = {};
		lvl0->GetDesc(&desc);
		LPD3DXBUFFER pBuffer = nullptr;
		if (SUCCEEDED(D3DXSaveSurfaceToFileInMemory(&pBuffer, D3DXIFF_PNG, lvl0, nullptr, nullptr)) && pBuffer)
		{
			char pngPath[MAX_PATH];
			sprintf_s(pngPath, sizeof(pngPath), "%s\\%016llX.png", stage3BoundDir.c_str(), (unsigned long long)hash);
			HANDLE fh = CreateFileA(pngPath, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (fh != INVALID_HANDLE_VALUE)
			{
				DWORD written = 0;
				WriteFile(fh, pBuffer->GetBufferPointer(), (DWORD)pBuffer->GetBufferSize(), &written, nullptr);
				CloseHandle(fh);
			}
			pBuffer->Release();
		}
		if (stage3BoundManifest.is_open())
		{
			char row[96];
			sprintf_s(row, sizeof(row), "%016llX,%u,%u,%u\n", (unsigned long long)hash, desc.Width, desc.Height, (UINT)desc.Format);
			stage3BoundManifest << row; stage3BoundManifest.flush();
		}
		lvl0->Release();
	}

	// --- Canonical Identity Layer helpers (2026-06-09) ---

	std::string canonBaseDir;

	// One-time sidecar load from <gamedir>\_canonical\ (generated offline by
	// _scratch_discovery\canon_build_sidecars.py from a stage3_bound capture run).
	void EnsureCanonSidecarsLoaded()
	{
		if (canonSidecarsLoaded) return;
		canonSidecarsLoaded = true;
		char modulePath[MAX_PATH] = {};
		GetModuleFileNameA(nullptr, modulePath, MAX_PATH);
		std::string gameDir = modulePath;
		auto slash = gameDir.find_last_of("\\/");
		if (slash != std::string::npos) gameDir = gameDir.substr(0, slash);
		canonBaseDir = gameDir + "\\_canonical";

		// canon_fps.bin: 'CFP1' + uint32 count + entries{uint64 hash, uint16 w, uint16 h, uint8 fp[1024]}
		std::ifstream f(canonBaseDir + "\\canon_fps.bin", std::ios::binary);
		char magic[4] = {};
		uint32_t count = 0;
		if (!f.read(magic, 4) || memcmp(magic, "CFP1", 4) != 0 || !f.read((char*)&count, 4) || count == 0 || count > 100000)
		{
			canonDisabled = true;
			Logging::Log() << "[CANON] DISABLED: canon_fps.bin missing/corrupt at " << canonBaseDir.c_str();
			return;
		}
		canonFps.resize(count);
		for (uint32_t i = 0; i < count; i++)
		{
			CanonFpEntry& e = canonFps[i];
			if (!f.read((char*)&e.hash, 8) || !f.read((char*)&e.w, 2) || !f.read((char*)&e.h, 2) || !f.read((char*)e.fp, 1024))
			{
				canonDisabled = true;
				canonFps.clear();
				canonHashSet.clear();
				Logging::Log() << "[CANON] DISABLED: canon_fps.bin truncated at entry " << i;
				return;
			}
			canonHashSet.insert(e.hash);
		}

		// canon_map.csv: "runtime_hash,canon_hash" 16-hex pairs; header/malformed lines skipped.
		std::ifstream m(canonBaseDir + "\\canon_map.csv");
		std::string line;
		size_t mapRows = 0;
		while (std::getline(m, line))
		{
			if (line.size() < 33 || line[16] != ',') continue;
			uint64_t rh = strtoull(line.substr(0, 16).c_str(), nullptr, 16);
			uint64_t ch = strtoull(line.substr(17, 16).c_str(), nullptr, 16);
			if (rh && ch && canonHashSet.count(ch)) { canonMapRuntimeToCanon[rh] = ch; mapRows++; }
		}
		// [NAMEKEY] name_map.csv (2026-06-11): "name,canon_hash,l2,margin,src_png".
		// Key = EngineTextures fullname + "MM{mip}" -- exactly what the upload detour
		// computes at runtime. Names may contain backslashes/spaces; first field up to
		// the comma, hash is the 16 hex chars after it. Lowercased for lookup.
		if (Config.DdrawNameKey)
		{
			std::ifstream nm(canonBaseDir + "\\name_map.csv");
			std::string nline;
			bool nFirst = true;
			while (std::getline(nm, nline))
			{
				if (nFirst) { nFirst = false; continue; }	// header
				auto c1 = nline.find(',');
				if (c1 == std::string::npos || c1 == 0 || nline.size() < c1 + 17) continue;
				std::string key = nline.substr(0, c1);
				uint64_t ch = strtoull(nline.substr(c1 + 1, 16).c_str(), nullptr, 16);
				if (!ch || !canonHashSet.count(ch)) continue;
				for (auto& c : key) c = (char)tolower((unsigned char)c);
				namekeyNameToCanon[key] = ch;
			}
			char nbuf[140];
			sprintf_s(nbuf, sizeof(nbuf), "[NAMEKEY] INIT keys=%zu mode=%lu", namekeyNameToCanon.size(), (unsigned long)Config.DdrawNameKey);
			Logging::Log() << nbuf;

			// V2 crop sidecar index: "CRP1" + u32 count + {u64 keyhash, u32 absOffset,
			// u16 w, u16 h}*count + payload blobs (read lazily). ~90KB resident.
			if (Config.DdrawNameKeyCrop)
			{
				std::ifstream cf(canonBaseDir + "\\crop_payloads.bin", std::ios::binary);
				char cmagic[4] = {};
				uint32_t ccount = 0;
				if (cf.read(cmagic, 4) && memcmp(cmagic, "CRP1", 4) == 0 && cf.read((char*)&ccount, 4) && ccount > 0 && ccount < 100000)
				{
					namekeyCropIdx.reserve(ccount);
					for (uint32_t i = 0; i < ccount; i++)
					{
						uint64_t kh = 0; NamekeyCropIdxEnt e{};
						if (!cf.read((char*)&kh, 8) || !cf.read((char*)&e.off, 4) || !cf.read((char*)&e.w, 2) || !cf.read((char*)&e.h, 2))
						{
							namekeyCropIdx.clear();
							Logging::Log() << "[NAMEKEY] CROPIDX DISABLED: crop_payloads.bin truncated at entry " << i;
							break;
						}
						namekeyCropIdx.emplace(kh, e);
					}
				}
				else
				{
					Logging::Log() << "[NAMEKEY] CROPIDX DISABLED: crop_payloads.bin missing/corrupt at " << canonBaseDir.c_str();
				}
				if (!namekeyCropIdx.empty())
				{
					char cbuf[160];
					sprintf_s(cbuf, sizeof(cbuf), "[NAMEKEY] CROPIDX entries=%zu mode=%lu", namekeyCropIdx.size(), (unsigned long)Config.DdrawNameKeyCrop);
					Logging::Log() << cbuf;
				}
			}
		}

		char buf[300];
		sprintf_s(buf, sizeof(buf), "[CANON] INIT fps=%u map=%zu dir=%s", count, mapRows, canonBaseDir.c_str());
		Logging::Log() << buf;
	}

	// Read a canonical payload file ('A4R4' + w + h + packed 16-bit rows).
	// Returns false (with a log) on any problem.
	bool CanonReadPayload(uint64_t h, uint32_t& w, uint32_t& ht, std::vector<uint8_t>& payload)
	{
		char path[MAX_PATH];
		sprintf_s(path, sizeof(path), "%s\\tex\\%016llX.a4r4", canonBaseDir.c_str(), (unsigned long long)h);
		std::ifstream f(path, std::ios::binary);
		char magic[4] = {};
		w = ht = 0;
		if (!f.read(magic, 4) || memcmp(magic, "A4R4", 4) != 0 ||
			!f.read((char*)&w, 4) || !f.read((char*)&ht, 4) ||
			w == 0 || ht == 0 || w > 4096 || ht > 4096)
		{
			Logging::Log() << "[CANON] tex file missing/corrupt: " << path;
			return false;
		}
		payload.resize((size_t)w * ht * 2);
		if (!f.read((char*)payload.data(), payload.size()))
		{
			Logging::Log() << "[CANON] tex file truncated: " << path;
			return false;
		}
		return true;
	}

	// Pixel verifier: compare a runtime A4R4G4B4 texture's mip 0 against a
	// canonical payload. True = same content (a drift variant), safe to rebind.
	// See the comment at kCanonVerifyBigDiffNibbles for why fingerprints alone
	// are not sufficient. Alpha is ignored (4-bit alpha noise IS the drift).
	bool CanonVerifyAgainstPayload(IDirect3DTexture9* tex, const std::vector<uint8_t>& payload, uint32_t w, uint32_t ht, float* outBigFrac)
	{
		if (outBigFrac) *outBigFrac = 1.0f;
		IDirect3DSurface9* lvl0 = nullptr;
		if (FAILED(tex->GetSurfaceLevel(0, &lvl0)) || !lvl0) return false;
		D3DSURFACE_DESC desc = {};
		bool ok = false;
		if (SUCCEEDED(lvl0->GetDesc(&desc)) && desc.Format == D3DFMT_A4R4G4B4 && desc.Width == w && desc.Height == ht)
		{
			D3DLOCKED_RECT lr = {};
			if (SUCCEEDED(lvl0->LockRect(&lr, nullptr, D3DLOCK_READONLY)))
			{
				DWORD big = 0;
				const uint16_t* canon = (const uint16_t*)payload.data();
				for (uint32_t y = 0; y < ht; y++)
				{
					const uint16_t* row = (const uint16_t*)((const uint8_t*)lr.pBits + (size_t)y * lr.Pitch);
					const uint16_t* crow = canon + (size_t)y * w;
					for (uint32_t x = 0; x < w; x++)
					{
						const int dr = abs((int)((row[x] >> 8) & 0xF) - (int)((crow[x] >> 8) & 0xF));
						const int dg = abs((int)((row[x] >> 4) & 0xF) - (int)((crow[x] >> 4) & 0xF));
						const int db = abs((int)(row[x] & 0xF) - (int)(crow[x] & 0xF));
						if (dr > kCanonVerifyBigDiffNibbles || dg > kCanonVerifyBigDiffNibbles || db > kCanonVerifyBigDiffNibbles) big++;
					}
				}
				lvl0->UnlockRect();
				const float frac = (float)big / (float)(w * ht);
				if (outBigFrac) *outBigFrac = frac;
				// 0.0025 (was 0.01): co-atlased tooltip/icon differences ran
				// 0.003-0.0099 and passed the 1% gate (visible bed mixup,
				// 2026-06-10). Must match the offline gate in canon_assign_map.py.
				ok = (frac < 0.0025f);
			}
		}
		lvl0->Release();
		return ok;
	}

	// --- [NAMEKEY] V2 helpers (2026-06-11) ------------------------------------------

	// Lazy payload read from crop_payloads.bin (index loaded at sidecar init; one
	// small seek+read per first synthesis of a sprite).
	bool NamekeyCropReadPayload(const NamekeyCropIdxEnt& e, std::vector<uint8_t>& payload)
	{
		std::ifstream f(canonBaseDir + "\\crop_payloads.bin", std::ios::binary);
		payload.resize((size_t)e.w * e.h * 2);
		if (!f.seekg(e.off) || !f.read((char*)payload.data(), payload.size()))
		{
			payload.clear();
			return false;
		}
		return true;
	}

	// Rect-scoped pixel verifier: compare the page's mip-0 rect against an expected
	// payload. Mirrors CanonVerifyAgainstPayload (RGB nibbles, big-diff threshold,
	// frac<0.0025; alpha ignored -- 4-bit alpha noise IS the drift). Returns 0=match,
	// 1=content mismatch, 2=bad format/bounds, 3=lock failure.
	int NamekeyCropVerifyRect(IDirect3DTexture9* pageTex, long rx, long ry,
		const std::vector<uint8_t>& payload, uint32_t w, uint32_t h, float* outBigFrac)
	{
		if (outBigFrac) *outBigFrac = 1.0f;
		IDirect3DSurface9* lvl0 = nullptr;
		if (FAILED(pageTex->GetSurfaceLevel(0, &lvl0)) || !lvl0) return 3;
		D3DSURFACE_DESC desc = {};
		int rc = 2;
		if (SUCCEEDED(lvl0->GetDesc(&desc)) && desc.Format == D3DFMT_A4R4G4B4 &&
			rx >= 0 && ry >= 0 && rx + (long)w <= (long)desc.Width && ry + (long)h <= (long)desc.Height)
		{
			rc = 3;
			D3DLOCKED_RECT lr = {};
			if (SUCCEEDED(lvl0->LockRect(&lr, nullptr, D3DLOCK_READONLY)))
			{
				DWORD big = 0;
				const uint16_t* canon = (const uint16_t*)payload.data();
				for (uint32_t y = 0; y < h; y++)
				{
					const uint16_t* row = (const uint16_t*)((const uint8_t*)lr.pBits + (size_t)(ry + y) * lr.Pitch) + rx;
					const uint16_t* crow = canon + (size_t)y * w;
					for (uint32_t x = 0; x < w; x++)
					{
						const int dr = abs((int)((row[x] >> 8) & 0xF) - (int)((crow[x] >> 8) & 0xF));
						const int dg = abs((int)((row[x] >> 4) & 0xF) - (int)((crow[x] >> 4) & 0xF));
						const int db = abs((int)(row[x] & 0xF) - (int)(crow[x] & 0xF));
						if (dr > kCanonVerifyBigDiffNibbles || dg > kCanonVerifyBigDiffNibbles || db > kCanonVerifyBigDiffNibbles) big++;
					}
				}
				lvl0->UnlockRect();
				const float frac = (float)big / (float)(w * h);
				if (outBigFrac) *outBigFrac = frac;
				rc = (frac < 0.0025f) ? 0 : 1;	// gate matches CanonVerifyAgainstPayload + canon_assign_map.py
			}
		}
		lvl0->Release();
		return rc;
	}

	// Build (or fetch) the canonical crop texture for a payload. A4R4G4B4 MANAGED,
	// mip-0 = payload verbatim (Remix hash == payloadHash by construction), full mip
	// chain filtered from it. Pre-seeds the Stage3/canon caches exactly like
	// CanonGetOrCreateTexture so binds count under the known hash and never rebind.
	// The map holds the OWNER reference; univDrawCache entries AddRef/Release on top.
	IDirect3DTexture9* NamekeyCropGetOrCreateTex(uint64_t payloadHash, const std::vector<uint8_t>& payload,
		uint32_t w, uint32_t h, LPDIRECT3DDEVICE9* d3d9Device)
	{
		auto it = namekeyCropTexByHash.find(payloadHash);
		if (it != namekeyCropTexByHash.end()) return it->second;
		if (!d3d9Device || !*d3d9Device) return nullptr;	// don't negative-cache: device may exist next draw
		if (namekeyCropTexCreated >= kNamekeyCropTexCap) return nullptr;
		IDirect3DTexture9* tex = nullptr;
		if (FAILED((*d3d9Device)->CreateTexture(w, h, 0, 0, D3DFMT_A4R4G4B4, D3DPOOL_MANAGED, &tex, nullptr)) || !tex)
		{
			namekeyCropTexByHash[payloadHash] = nullptr;
			return nullptr;
		}
		D3DLOCKED_RECT lr = {};
		if (FAILED(tex->LockRect(0, &lr, nullptr, 0)))
		{
			tex->Release();
			namekeyCropTexByHash[payloadHash] = nullptr;
			return nullptr;
		}
		const size_t rowBytes = (size_t)w * 2;
		for (uint32_t y = 0; y < h; y++)
		{
			memcpy((uint8_t*)lr.pBits + (size_t)y * lr.Pitch, payload.data() + (size_t)y * rowBytes, rowBytes);
		}
		tex->UnlockRect(0);
		if (tex->GetLevelCount() > 1)
		{
			D3DXFilterTexture(tex, nullptr, 0, D3DX_FILTER_LINEAR);
		}
		stage3HashByD3d9Tex[tex] = payloadHash;
		stage3DumpedHashes.insert(payloadHash);
		canonResolveByTex[tex] = nullptr;	// a canonical never rebinds further
		namekeyCropTexByHash[payloadHash] = tex;
		namekeyCropTexCreated++;
		return tex;
	}

	// Load (or return cached) canonical texture for a canonical hash. MANAGED
	// A4R4G4B4 with a full mip chain; mip-0 bytes are the .a4r4 payload verbatim,
	// so Remix's XXH3 over packed mip 0 EQUALS `h` by construction. A null cache
	// entry means a prior load failed -- never retried.
	IDirect3DTexture9* CanonGetOrCreateTexture(uint64_t h, LPDIRECT3DDEVICE9* d3d9Device)
	{
		auto it = canonTexByHash.find(h);
		if (it != canonTexByHash.end()) return it->second;
		if (!d3d9Device || !*d3d9Device) return nullptr;	// don't negative-cache: device may exist next bind
		if (canonTexCreatedTotal >= kCanonTexCap)
		{
			canonTexByHash[h] = nullptr;
			return nullptr;
		}

		uint32_t w = 0, ht = 0;
		std::vector<uint8_t> payload;
		if (!CanonReadPayload(h, w, ht, payload))
		{
			canonTexByHash[h] = nullptr;
			return nullptr;
		}

		IDirect3DTexture9* tex = nullptr;
		// Levels=0 -> full mip chain; a mip-0-only texture renders as a white
		// block at far/small draws under Remix (A.10 finding).
		if (FAILED((*d3d9Device)->CreateTexture(w, ht, 0, 0, D3DFMT_A4R4G4B4, D3DPOOL_MANAGED, &tex, nullptr)) || !tex)
		{
			char buf[120];
			sprintf_s(buf, sizeof(buf), "[CANON] CreateTexture failed for %016llX", (unsigned long long)h);
			Logging::Log() << buf;
			canonTexByHash[h] = nullptr;
			return nullptr;
		}
		D3DLOCKED_RECT lr = {};
		if (FAILED(tex->LockRect(0, &lr, nullptr, 0)))
		{
			tex->Release();
			canonTexByHash[h] = nullptr;
			return nullptr;
		}
		const size_t rowBytes = (size_t)w * 2;
		for (uint32_t y = 0; y < ht; y++)
		{
			memcpy((uint8_t*)lr.pBits + (size_t)y * lr.Pitch, payload.data() + (size_t)y * rowBytes, rowBytes);
		}
		tex->UnlockRect(0);
		// FilterTexture fills levels 1+ FROM level 0; mip-0 bytes (the hash) stay intact.
		if (tex->GetLevelCount() > 1)
		{
			D3DXFilterTexture(tex, nullptr, 0, D3DX_FILTER_LINEAR);
		}

		// Pre-seed Stage 3 so canonical binds count under their known hash without
		// a redundant lock+rehash, and never self-dump into the bound corpus.
		stage3HashByD3d9Tex[tex] = h;
		stage3DumpedHashes.insert(h);
		canonResolveByTex[tex] = nullptr;	// a canonical never rebinds further
		canonTexByHash[h] = tex;
		canonTexCreatedTotal++;
		char buf[160];
		sprintf_s(buf, sizeof(buf), "[CANON] CREATE %016llX %ux%u total=%lu", (unsigned long long)h, w, ht, (unsigned long)canonTexCreatedTotal);
		Logging::Log() << buf;
		return tex;
	}

	// Resolve a bound whole-surface texture to its canonical replacement, or
	// nullptr when none applies. Cached per d3d9 pointer; SetDirtyFlag erases
	// entries when the surface's content changes.
	IDirect3DTexture9* CanonResolveImpl(IDirect3DTexture9* tex, LPDIRECT3DDEVICE9* d3d9Device)
	{
		auto rIt = canonResolveByTex.find(tex);
		if (rIt != canonResolveByTex.end())
		{
			if (rIt->second) canonRebindBindsTotal++;
			return rIt->second;
		}

		EnsureCanonSidecarsLoaded();
		if (canonDisabled) return nullptr;

		// Reuse / seed the Stage 3 content-hash cache (same key domain: the exact
		// bound d3d9 pointer). First sight of a hash also feeds the bound-corpus dump.
		uint64_t h;
		auto hIt = stage3HashByD3d9Tex.find(tex);
		if (hIt != stage3HashByD3d9Tex.end())
		{
			h = hIt->second;
		}
		else
		{
			h = Stage3HashD3d9Texture(tex);
			stage3HashByD3d9Tex[tex] = h;
			Stage3MaybeDumpBound(tex, h, false);
		}
		if (h == 0)
		{
			canonResolveByTex[tex] = nullptr;
			return nullptr;
		}

		uint64_t canonHash = 0;

		// [NAMEKEY] (2026-06-11): exact-identity resolution, attempted before any
		// content-similarity machinery. The pixel verifier stays in the loop as the
		// guard against recipe mis-attribution (wrong page attached) and wrong-twin
		// map rows -- a failed verify falls through to the legacy path.
		uint64_t nameCanon = 0;
		bool nameVerified = false;
		char nameKeyStr[128] = {};
		if (Config.DdrawNameKey && !canonHashSet.count(h))
		{
			nameCanon = NamekeyLookupCanon(tex, nameKeyStr, sizeof(nameKeyStr));
			if (nameCanon)
			{
				uint32_t ncw = 0, nch = 0;
				std::vector<uint8_t> npayload;
				float nBigFrac = 1.0f;
				if (CanonReadPayload(nameCanon, ncw, nch, npayload) &&
					CanonVerifyAgainstPayload(tex, npayload, ncw, nch, &nBigFrac))
				{
					nameVerified = true;
					namekeyResolves++;
				}
				else
				{
					namekeyVerifyFails++;
					char buf[240];
					sprintf_s(buf, sizeof(buf), "[NAMEKEY] VERIFYFAIL runtime=%016llX name=%s canon=%016llX bigfrac=%.4f",
						(unsigned long long)h, nameKeyStr, (unsigned long long)nameCanon, nBigFrac);
					Logging::Log() << buf;
				}
			}
		}

		if (canonHashSet.count(h))
		{
			// Content already presents a canonical hash: Remix sees the right key
			// with no rebind, and mods keyed on it already fire.
			canonIdentityTotal++;
		}
		else if (Config.DdrawNameKey >= 2 && nameVerified)
		{
			// Name key authoritative: exact identity, pixel-verified. Warm the hash
			// map so repeats (and canon_assign_map.py's offline rebuild) see it.
			canonHash = nameCanon;
			canonMapRuntimeToCanon[h] = canonHash;
			char buf[240];
			sprintf_s(buf, sizeof(buf), "[NAMEKEY] MATCH runtime=%016llX -> canon=%016llX name=%s",
				(unsigned long long)h, (unsigned long long)nameCanon, nameKeyStr);
			Logging::Log() << buf;
		}
		else
		{
			auto mIt = canonMapRuntimeToCanon.find(h);
			if (mIt != canonMapRuntimeToCanon.end())
			{
				canonHash = mIt->second;
			}
			else
			{
				// Unknown content: one-time fingerprint match against the canonical
				// corpus. Dims must agree -- never swap detail levels.
				D3DSURFACE_DESC desc = {};
				IDirect3DSurface9* lvl0 = nullptr;
				if (SUCCEEDED(tex->GetSurfaceLevel(0, &lvl0)) && lvl0)
				{
					lvl0->GetDesc(&desc);
					lvl0->Release();
				}
				uint8_t fp[1024];
				if (desc.Width && ComputeFingerprintFromTexture(tex, d3d9Device, fp))
				{
					float best = 1e30f, second = 1e30f;
					uint64_t bestHash = 0;
					for (const CanonFpEntry& e : canonFps)
					{
						if (e.w != desc.Width || e.h != desc.Height) continue;
						float d = FingerprintL2(fp, e.fp);
						if (d < best) { second = best; best = d; bestHash = e.hash; }
						else if (d < second) { second = d; }
					}
					char buf[240];
					if (bestHash && best < kFingerprintL2Threshold)
					{
						// Fingerprint retrieval found a candidate; the PIXEL
						// verifier decides (fingerprints can't see quadrant-level
						// content swaps -- 2026-06-09 session audit).
						uint32_t cw = 0, chh = 0;
						std::vector<uint8_t> cpayload;
						float bigFrac = 1.0f;
						if (CanonReadPayload(bestHash, cw, chh, cpayload) &&
							CanonVerifyAgainstPayload(tex, cpayload, cw, chh, &bigFrac))
						{
							canonHash = bestHash;
							canonMapRuntimeToCanon[h] = canonHash;	// session-warm; canon_assign_map.py rebuilds the durable map offline
							canonFpMatchTotal++;
							sprintf_s(buf, sizeof(buf), "[CANON] FPMATCH runtime=%016llX -> canon=%016llX L2=%.1f second=%.1f bigfrac=%.4f",
								(unsigned long long)h, (unsigned long long)canonHash, best, second, bigFrac);
							Logging::Log() << buf;
						}
						else
						{
							canonFpRejectTotal++;
							sprintf_s(buf, sizeof(buf), "[CANON] FPREJECT runtime=%016llX candidate=%016llX L2=%.1f bigfrac=%.4f",
								(unsigned long long)h, (unsigned long long)bestHash, best, bigFrac);
							Logging::Log() << buf;
						}
					}
					else
					{
						canonFpNoMatchTotal++;
						sprintf_s(buf, sizeof(buf), "[CANON] FPNOMATCH runtime=%016llX bestL2=%.1f (threshold=%.0f)",
							(unsigned long long)h, best, kFingerprintL2Threshold);
						Logging::Log() << buf;
					}
				}
			}

			// [NAMEKEY] shadow mode: the legacy path just decided; compare what the
			// (verified) name key would have done. Disagreements are exactly the
			// mixup class the fingerprint path was historically vulnerable to.
			if (Config.DdrawNameKey == 1 && nameVerified)
			{
				char buf[280];
				if (canonHash && canonHash == nameCanon)
				{
					namekeyShadowAgree++;
				}
				else if (canonHash)
				{
					namekeyShadowDisagree++;
					sprintf_s(buf, sizeof(buf), "[NAMEKEY] SHADOW_DISAGREE runtime=%016llX name=%s nameCanon=%016llX legacyCanon=%016llX",
						(unsigned long long)h, nameKeyStr, (unsigned long long)nameCanon, (unsigned long long)canonHash);
					Logging::Log() << buf;
				}
				else
				{
					namekeyShadowOnly++;
					sprintf_s(buf, sizeof(buf), "[NAMEKEY] SHADOW_ONLY runtime=%016llX name=%s nameCanon=%016llX (legacy path found nothing)",
						(unsigned long long)h, nameKeyStr, (unsigned long long)nameCanon);
					Logging::Log() << buf;
				}
			}
		}

		IDirect3DTexture9* canon = canonHash ? CanonGetOrCreateTexture(canonHash, d3d9Device) : nullptr;
		canonResolveByTex[tex] = canon;
		if (canon) canonRebindBindsTotal++;
		return canon;
	}

	// Stage 3 (2026-05-27): periodic dump of per-surface + per-hash bind stats.
	// Called from Stage3OnRemixBind below (which IDirect3DDeviceX.cpp invokes on
	// every d3d9 SetTexture binding), so cadence is stable as long as the game draws.
	void Stage3DumpIfDue()
	{
		DWORD now = GetTickCount();
		if (stage3LastDumpMs == 0) { stage3LastDumpMs = now; return; }
		if (now - stage3LastDumpMs < kStage3DumpIntervalMs) return;
		stage3LastDumpMs = now;

		struct Row { void* surf; DWORD total; Stage3SurfaceStat s; };
		std::vector<Row> rows; rows.reserve(stage3PerSurface.size());
		DWORD totHits=0, totCrops=0, totInvals=0, totScar=0, totBL=0;
		for (auto& kv : stage3PerSurface) {
			DWORD t = kv.second.hits + kv.second.crops + kv.second.invals + kv.second.scarcitySkips + kv.second.blacklistSkips;
			rows.push_back({kv.first, t, kv.second});
			totHits += kv.second.hits; totCrops += kv.second.crops; totInvals += kv.second.invals;
			totScar += kv.second.scarcitySkips; totBL += kv.second.blacklistSkips;
		}
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.total > b.total; });
		{
			char buf[260];
			sprintf_s(buf, sizeof(buf), "[STAGE3] PER_SURFACE_SUMMARY surfaces=%zu hits=%lu crops=%lu invals=%lu scarcity=%lu blacklist=%lu",
				rows.size(), (unsigned long)totHits, (unsigned long)totCrops,
				(unsigned long)totInvals, (unsigned long)totScar, (unsigned long)totBL);
			Logging::Log() << buf;
		}
		const size_t topN = (rows.size() < 20) ? rows.size() : 20;
		for (size_t i = 0; i < topN; ++i) {
			const Row& r = rows[i];
			char buf[220];
			sprintf_s(buf, sizeof(buf), "[STAGE3] SURF #%zu %p total=%lu hits=%lu crops=%lu invals=%lu scar=%lu bl=%lu",
				i, r.surf, (unsigned long)r.total,
				(unsigned long)r.s.hits, (unsigned long)r.s.crops, (unsigned long)r.s.invals,
				(unsigned long)r.s.scarcitySkips, (unsigned long)r.s.blacklistSkips);
			Logging::Log() << buf;
		}

		struct HRow { uint64_t h; DWORD total; Stage3BindStat s; };
		std::vector<HRow> hr; hr.reserve(stage3BindByHash.size());
		DWORD bWhole=0, bCrop=0;
		for (auto& kv : stage3BindByHash) {
			hr.push_back({kv.first, kv.second.wholeBinds + kv.second.cropBinds, kv.second});
			bWhole += kv.second.wholeBinds; bCrop += kv.second.cropBinds;
		}
		std::sort(hr.begin(), hr.end(), [](const HRow& a, const HRow& b){ return a.total > b.total; });
		{
			char buf[260];
			sprintf_s(buf, sizeof(buf), "[STAGE3] BIND_SUMMARY uniqueHashes=%zu whole=%lu crop=%lu wholeUnhashed=%lu cropUnhashed=%lu",
				hr.size(), (unsigned long)bWhole, (unsigned long)bCrop,
				(unsigned long)stage3WholeBindsUnhashed, (unsigned long)stage3CropBindsUnhashed);
			Logging::Log() << buf;

			// [RECIPE] (2026-06-10): tracked-destination + emission health counters.
			if (Config.DdrawRecipeLog)
			{
				sprintf_s(buf, sizeof(buf), "[RECIPE] SUMMARY tracked=%zu emitted=%zu srcCache=%zu surfacesDropped=%lu",
					recipeByTex.size(), recipeEmitted.size(), recipeSrcHashCache.size(), (unsigned long)recipeSurfacesDropped);
				Logging::Log() << buf;
			}

			// [OWNUP] (2026-06-10): upload-detour health + distinct-identity count.
			if (Config.DdrawOwnerLog)
			{
				sprintf_s(buf, sizeof(buf), "[OWNUP] SUMMARY calls=%lu badReads=%lu distinctIdentities=%zu",
					(unsigned long)ownupCalls, (unsigned long)ownupBadReads, ownupSeen.size());
				Logging::Log() << buf;
			}

			// [NAMEKEY] (2026-06-11): recorder/consume/resolve health.
			if (Config.DdrawNameKey)
			{
				char nbuf[420];
				sprintf_s(nbuf, sizeof(nbuf), "[NAMEKEY] SUMMARY placements=%lu overflow=%lu consumes=%lu orphan=%lu multi=%lu recipes=%zu resolves=%lu verifyFail=%lu noRecipe=%lu multiName=%lu subPage=%lu unmapped=%lu distinctUnmapped=%zu shadowAgree=%lu shadowDisagree=%lu shadowOnly=%lu",
					(unsigned long)namekeyPlacements, (unsigned long)namekeyAccumOverflow, (unsigned long)namekeyConsumes,
					(unsigned long)namekeyOrphanConsumes, (unsigned long)namekeyMultiComp, namekeyRecipeByTex.size(),
					(unsigned long)namekeyResolves, (unsigned long)namekeyVerifyFails, (unsigned long)namekeyNoRecipe,
					(unsigned long)namekeyNotSingleMulti, (unsigned long)namekeyNotSingleSub, (unsigned long)namekeyUnmapped,
					namekeyUnmappedSeen.size(), (unsigned long)namekeyShadowAgree,
					(unsigned long)namekeyShadowDisagree, (unsigned long)namekeyShadowOnly);
				Logging::Log() << nbuf;

				// [NAMEKEY] V2 crop-layer health (2026-06-11).
				if (Config.DdrawNameKeyCrop)
				{
					char cbuf[420];
					sprintf_s(cbuf, sizeof(cbuf), "[NAMEKEY] CROPSUMMARY match=%lu ambig=%lu submiss=%lu noSidecar=%lu dims=%lu fmt=%lu readFail=%lu verifyFail=%lu serveVerifyFail=%lu synth=%lu serve=%lu mm0=%lu waterCol=%lu waterFlat=%lu waterFlatL=%lu dynUi=%lu shadowPass=%lu shadowFail=%lu states=%zu texes=%lu idx=%zu",
						(unsigned long)namekeyCropMatch, (unsigned long)namekeyCropAmbiguous, (unsigned long)namekeyCropSubmiss,
						(unsigned long)namekeyCropNoSidecar, (unsigned long)namekeyCropDimsMismatch, (unsigned long)namekeyCropBadFormat,
						(unsigned long)namekeyCropReadFail, (unsigned long)namekeyCropVerifyFail, (unsigned long)namekeyCropServeVerifyFail,
						(unsigned long)namekeyCropSynth, (unsigned long)namekeyCropServe, (unsigned long)namekeyCropMm0Collapse,
						(unsigned long)namekeyWaterCollapse, (unsigned long)namekeyWaterFlatten, (unsigned long)namekeyWaterLarge, (unsigned long)namekeyDynUiDraws,
						(unsigned long)namekeyCropShadowPass, (unsigned long)namekeyCropShadowFail, namekeyCropByKey.size(),
						(unsigned long)namekeyCropTexCreated, namekeyCropIdx.size());
					Logging::Log() << cbuf;
				}
			}
		}
		const size_t topH = (hr.size() < 30) ? hr.size() : 30;
		for (size_t i = 0; i < topH; ++i) {
			char buf[160];
			sprintf_s(buf, sizeof(buf), "[STAGE3] BIND #%zu %016llX total=%lu whole=%lu crop=%lu",
				i, (unsigned long long)hr[i].h, (unsigned long)hr[i].total,
				(unsigned long)hr[i].s.wholeBinds, (unsigned long)hr[i].s.cropBinds);
			Logging::Log() << buf;
		}

		if (Config.DdrawCanonicalRebind)
		{
			char buf[300];
			sprintf_s(buf, sizeof(buf), "[CANON] SUMMARY rebindBinds=%lu fpMatch=%lu fpReject=%lu fpNoMatch=%lu identity=%lu texCreated=%lu churnExempt=%lu resolveCache=%zu mapSize=%zu",
				(unsigned long)canonRebindBindsTotal, (unsigned long)canonFpMatchTotal, (unsigned long)canonFpRejectTotal, (unsigned long)canonFpNoMatchTotal,
				(unsigned long)canonIdentityTotal, (unsigned long)canonTexCreatedTotal, (unsigned long)canonChurnExemptTotal,
				canonResolveByTex.size(), canonMapRuntimeToCanon.size());
			Logging::Log() << buf;
		}
	}
}

// Stage 3 (2026-05-27, fixed 2026-05-29): external-linkage shim called from
// IDirect3DDeviceX.cpp at every d3d9 SetTexture binding. Hashes the EXACT bound
// d3d9 texture (cached once per pointer) and increments the per-hash bind counter,
// then triggers the periodic dump. d3d9Tex may be a whole-surface texture or a
// synthesized POT crop. Forward-declared in IDirect3DDeviceX.cpp (no header touch).
//
// FIX (2026-05-29): the original body looked d3d9Tex up in capState.firstHashByTex,
// which (a) is only populated when DdrawContentCapture=1 and (b) is keyed by the
// WRAPPER pointer, not the d3d9 texture pointer the bind sites pass -- so every
// lookup missed and all binds counted as unhashed. We now hash the bound texture
// directly and cache by its d3d9 pointer, which both removes the capture-flag
// dependency and keys on exactly the object Remix receives.
void Stage3OnRemixBind(IDirect3DTexture9* d3d9Tex, bool isCrop)
{
	OwnupInstallHookOnce();   // one-shot, flag-gated (DdrawOwnerLog or DdrawNameKey): arms the 0x58E53B call-site detour once DKII.exe is live
	if (!d3d9Tex) return;
	uint64_t h;
	auto it = stage3HashByD3d9Tex.find(d3d9Tex);
	if (it != stage3HashByD3d9Tex.end()) {
		h = it->second;
	} else {
		h = Stage3HashD3d9Texture(d3d9Tex);
		// Cache 0 too: a texture that can't be locked won't become lockable on the
		// next bind, so don't re-attempt the lock millions of times per session.
		stage3HashByD3d9Tex[d3d9Tex] = h;
		// When capture is on, save the runtime-keyed corpus PNG (whole-surface only).
		Stage3MaybeDumpBound(d3d9Tex, h, isCrop);
	}
	if (h == 0) {
		if (isCrop) stage3CropBindsUnhashed++; else stage3WholeBindsUnhashed++;
	} else {
		auto& s = stage3BindByHash[h];
		if (isCrop) s.cropBinds++; else s.wholeBinds++;
		// [RECIPE] (2026-06-10): first bind of each distinct whole-surface content hash
		// emits the composition recipe that produced it (set-guarded inside).
		if (Config.DdrawRecipeLog && !isCrop)
		{
			RecipeEmitForHash(d3d9Tex, h);
		}
	}
	Stage3DumpIfDue();
}

// Canonical Identity Layer (2026-06-09): external-linkage shim called from
// IDirect3DDeviceX.cpp at the whole-surface bind site BEFORE SetTexture. Returns
// the canonical texture to bind instead of `d3d9Tex`, or nullptr to bind the
// original. See the block comment at the canon globals for the architecture.
IDirect3DTexture9* Stage3CanonicalResolve(IDirect3DTexture9* d3d9Tex, LPDIRECT3DDEVICE9* d3d9Device)
{
	if (!Config.DdrawCanonicalRebind || !d3d9Tex) return nullptr;
	return CanonResolveImpl(d3d9Tex, d3d9Device);
}

// Dynamic-UI emissive (2026-06-11): external-linkage accessors for the per-draw
// classification. IDirect3DDeviceX.cpp resets the flag before its decompose probe
// (TryGetUniversalSubTextureForUV sets it when the draw's placement resolves to a
// runtime-rendered UI name), reads it after, and counts applied overrides.
void NamekeyResetDrawDynUi() { g_namekeyDrawDynUi = false; g_namekeyDrawDynUiAny = false; g_namekeyDrawWater = false; }
bool NamekeyGetDrawDynUi() { return g_namekeyDrawDynUi; }
void NamekeyNoteDynUiDraw() { namekeyDynUiDraws++; }
bool NamekeyGetDrawWater() { return g_namekeyDrawWater; }
void NamekeyNoteWaterFlatten() { namekeyWaterFlatten++; }

// ******************************
// IUnknown functions
// ******************************

HRESULT m_IDirectDrawSurfaceX::QueryInterface(REFIID riid, LPVOID FAR* ppvObj, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") " << riid;

	if (!ppvObj)
	{
		return E_POINTER;
	}
	*ppvObj = nullptr;

	if (riid == IID_GetRealInterface)
	{
		*ppvObj = ProxyInterface;
		return DD_OK;
	}
	if (riid == IID_GetInterfaceX)
	{
		*ppvObj = this;
		return DD_OK;
	}
	if (riid == IID_GetMipMapLevel)
	{
		*ppvObj = 0;
		return DD_OK;
	}

	bool IsD3DDevice = (riid == IID_IDirect3DHALDevice || riid == IID_IDirect3DTnLHalDevice ||
		riid == IID_IDirect3DRGBDevice || riid == IID_IDirect3DRampDevice || riid == IID_IDirect3DMMXDevice ||
		riid == IID_IDirect3DRefDevice || riid == IID_IDirect3DNullDevice);

	DWORD DxVersion = (Config.Dd7to9 && CheckWrapperType(riid)) ? GetGUIDVersion(riid) : DirectXVersion;

	if (riid == GetWrapperType(DxVersion) || riid == IID_IUnknown)
	{
		*ppvObj = GetWrapperInterfaceX(DxVersion);

		AddRef(DxVersion);

		return DD_OK;
	}

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, false, false, false)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Query failed for " << riid << " from " << GetWrapperType(DirectXVersion));

			return E_NOINTERFACE;
		}

		if (IsD3DDevice)
		{
			// Check for Direct3D surface
			if (!IsSurface3D())
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not a Direct3D surface for " << riid << " from " << GetWrapperType(DirectXVersion));

				return E_NOINTERFACE;
			}

			m_IDirect3DX* D3DX = *ddrawParent->GetCurrentD3D();

			if (!D3DX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: Direct3D not setup when creating Direct3DDevice.");
			}

			DxVersion = (DxVersion == 4) ? 3 : DxVersion;

			if (!attached3DDevice)
			{
				attached3DDevice = new m_IDirect3DDeviceX(ddrawParent, D3DX, (LPDIRECTDRAWSURFACE7)GetWrapperInterfaceX(DirectXVersion), riid, DirectXVersion);

				attached3DDevice->SetParent3DSurface(this, DirectXVersion);
			}
			else
			{
				attached3DDevice->AddRef(DxVersion);	// No need to add a ref when creating a device because it is already added when creating the device
			}

			*ppvObj = (LPDIRECT3DDEVICE7)attached3DDevice->GetWrapperInterfaceX(DxVersion);

			return D3D_OK;
		}
		// ColorControl doesn't work on native ddraw
		/*if (riid == IID_IDirectDrawColorControl)
		{
			m_IDirectDrawColorControl* lpColorControl = ddrawParent->GetColorControlInterface();

			if (lpColorControl)
			{
				*ppvObj = lpColorControl;

				lpColorControl->AddRef();
			}
			else
			{
				if (FAILED(ddrawParent->CreateColorControl(reinterpret_cast<m_IDirectDrawColorControl**>(ppvObj))))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to create ColorControl for " << riid << " from " << GetWrapperType(DirectXVersion));

					return E_NOINTERFACE;
				}
			}

			return DD_OK;
		}*/
		if (riid == IID_IDirectDrawGammaControl)
		{
			m_IDirectDrawGammaControl* lpGammaControl = ddrawParent->GetGammaControlInterface();

			if (lpGammaControl)
			{
				*ppvObj = lpGammaControl;

				lpGammaControl->AddRef();
			}
			else
			{
				if (FAILED(ddrawParent->CreateGammaControl(reinterpret_cast<m_IDirectDrawGammaControl**>(ppvObj))))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to create Gamma for " << riid << " from " << GetWrapperType(DirectXVersion));

					return E_NOINTERFACE;
				}
			}

			return DD_OK;
		}
		if (riid == IID_IDirect3DTexture || riid == IID_IDirect3DTexture2)
		{
			if (ddrawParent->IsCreatedEx())
			{
				LOG_LIMIT(100, __FUNCTION__ << " Query Not Implemented for " << riid << " from " << GetWrapperType(DirectXVersion));

				return E_NOINTERFACE;
			}

			DxVersion = GetGUIDVersion(riid);

			m_IDirect3DTextureX* InterfaceX = nullptr;

			if (!attached3DTexture)
			{
				attached3DTexture = new m_IDirect3DTextureX(DxVersion, this, DirectXVersion);
			}
			else
			{
				attached3DTexture->AddRef(DxVersion);	// No need to add a ref when creating a texture because it is already added when creating the texture
			}

			InterfaceX = attached3DTexture;

			*ppvObj = InterfaceX->GetWrapperInterfaceX(DxVersion);

			return DD_OK;
		}
	}

	HRESULT hr = ProxyQueryInterface(ProxyInterface, riid, ppvObj, GetWrapperType(DirectXVersion));

	if (IsD3DDevice && SUCCEEDED(hr))
	{
		if (DirectXVersion == 1)
		{
			*ppvObj = ProxyAddressLookupTable.FindAddress<m_IDirect3DDevice>(*ppvObj);
		}
		else if (DirectXVersion == 2)
		{
			*ppvObj = ProxyAddressLookupTable.FindAddress<m_IDirect3DDevice2>(*ppvObj);
		}
		else if (DirectXVersion == 3 || DirectXVersion == 4)
		{
			*ppvObj = ProxyAddressLookupTable.FindAddress<m_IDirect3DDevice3>(*ppvObj);
		}
		else
		{
			*ppvObj = ProxyAddressLookupTable.FindAddress<m_IDirect3DDevice7>(*ppvObj);
		}
	}

	return hr;
}

ULONG m_IDirectDrawSurfaceX::AddRef(DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") v" << DirectXVersion;

	if (Config.Dd7to9)
	{
		switch (DirectXVersion)
		{
		case 1:
			return InterlockedIncrement(&RefCount1);
		case 2:
			return InterlockedIncrement(&RefCount2);
		case 3:
			return InterlockedIncrement(&RefCount3);
		case 4:
			return InterlockedIncrement(&RefCount4);
		case 7:
			return InterlockedIncrement(&RefCount7);
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
			return 0;
		}
	}

	return ProxyInterface->AddRef();
}

ULONG m_IDirectDrawSurfaceX::Release(DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") v" << DirectXVersion;

	if (Config.Dd7to9)
	{
		ULONG ref;

		switch (DirectXVersion)
		{
		case 1:
			ref = (InterlockedCompareExchange(&RefCount1, 0, 0)) ? InterlockedDecrement(&RefCount1) : 0;
			break;
		case 2:
			ref = (InterlockedCompareExchange(&RefCount2, 0, 0)) ? InterlockedDecrement(&RefCount2) : 0;
			break;
		case 3:
			ref = (InterlockedCompareExchange(&RefCount3, 0, 0)) ? InterlockedDecrement(&RefCount3) : 0;
			break;
		case 4:
			ref = (InterlockedCompareExchange(&RefCount4, 0, 0)) ? InterlockedDecrement(&RefCount4) : 0;
			break;
		case 7:
			ref = (InterlockedCompareExchange(&RefCount7, 0, 0)) ? InterlockedDecrement(&RefCount7) : 0;
			break;
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
			ref = 0;
		}

		if (InterlockedCompareExchange(&RefCount1, 0, 0) + InterlockedCompareExchange(&RefCount2, 0, 0) +
			InterlockedCompareExchange(&RefCount3, 0, 0) + InterlockedCompareExchange(&RefCount4, 0, 0) +
			InterlockedCompareExchange(&RefCount7, 0, 0) == 0)
		{
			if (CanSurfaceBeDeleted())
			{
				// Handle cases where games use surface addresses after the surface is released (Final Liberation: Warhammer Epic 40,000)
				if (IsSurfaceBusy())
				{
					Logging::Log() << __FUNCTION__ << " Warning: surface still in use! Locked: " << IsSurfaceLocked() << " DC: " << IsSurfaceInDC() << " Blt: " << IsSurfaceBlitting();
					if (ddrawParent)
					{
						ddrawParent->AddReleasedSurface(this);
					}
					ReleaseD9AuxiliarySurfaces();
					ReleaseDirectDrawResources();
				}
				else
				{
					delete this;
				}
			}
		}

		return ref;
	}

	ULONG ref = ProxyInterface->Release();

	if (ref == 0)
	{
		delete this;
	}

	return ref;
}

// ******************************
// IDirectDrawSurface v1 functions
// ******************************

HRESULT m_IDirectDrawSurfaceX::AddAttachedSurface(LPDIRECTDRAWSURFACE7 lpDDSurface, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSurface)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		m_IDirectDrawSurfaceX *lpAttachedSurfaceX = nullptr;

		lpDDSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpAttachedSurfaceX);

		if (lpAttachedSurfaceX == this)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: cannot attach self");
			return DDERR_CANNOTATTACHSURFACE;
		}

		if (!ddrawParent->DoesSurfaceExist(lpAttachedSurfaceX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid surface!");
			return DDERR_INVALIDPARAMS;
		}

		if (DoesAttachedSurfaceExist(lpAttachedSurfaceX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: attached surface already exists");
			return DDERR_SURFACEALREADYATTACHED;
		}

		if (lpAttachedSurfaceX->IsDepthStencil() && GetAttachedDepthStencil())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: zbuffer surface already exists");
			return DDERR_CANNOTATTACHSURFACE;
		}

		DWORD AttachedSurfaceCaps = lpAttachedSurfaceX->GetSurfaceCaps().dwCaps;
		if (!(((AttachedSurfaceCaps & DDSCAPS_BACKBUFFER) && (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_FRONTBUFFER)) ||
			((AttachedSurfaceCaps & DDSCAPS_FRONTBUFFER) && (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_BACKBUFFER)) ||
			((AttachedSurfaceCaps & DDSCAPS_MIPMAP) && (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_MIPMAP)) ||
			(AttachedSurfaceCaps & DDSCAPS_ZBUFFER)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: cannot attach surface with this method. dwCaps: " << lpAttachedSurfaceX->GetSurfaceCaps());
			return DDERR_CANNOTATTACHSURFACE;
		}

		// Check for MipMaps
		if (AttachedSurfaceCaps & DDSCAPS_MIPMAP)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Attaching a MipMap not Implemented.");
			return DDERR_CANNOTATTACHSURFACE;
		}

		// Update attached stencil surface
		if (lpAttachedSurfaceX->IsDepthStencil())
		{
			UpdateAttachedDepthStencil(lpAttachedSurfaceX);
		}

		AddAttachedSurfaceToMap(lpAttachedSurfaceX, true, DirectXVersion, 1);

		return DD_OK;
	}

	if (lpDDSurface)
	{
		lpDDSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSurface);
	}

	return ProxyInterface->AddAttachedSurface(lpDDSurface);
}

HRESULT m_IDirectDrawSurfaceX::AddOverlayDirtyRect(LPRECT lpRect)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	return ProxyInterface->AddOverlayDirtyRect(lpRect);
}

void MenuBlitOverlayEraseRegion(LPRECT lpRect, LONG surfW, LONG surfH);  // [BLITQUAD] v10.3, defined with the overlay block below

HRESULT m_IDirectDrawSurfaceX::Blt(LPRECT lpDestRect, LPDIRECTDRAWSURFACE7 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx, DWORD MipMapLevel, bool PresentBlt)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" DestRect = " << lpDestRect <<
		" SrcSurface = " << lpDDSrcSurface <<
		" SrcRect = " << lpSrcRect <<
		" Flags = " << Logging::hex(dwFlags) <<
		" BltFX = " << lpDDBltFx <<
		" MipMapLevel = " << MipMapLevel <<
		" PresentBlt = " << PresentBlt;

	// Surface-pool tracking (Phase A): count Blts whose destination is a texture
	// surface -- tests the "content churned into persistent surfaces" hypothesis.
	if (Config.DdrawLogTextureAtlas && IsSurfaceTexture())
	{
		poolStats.bltToTextureThisWindow++;
		// Phase A.6: per-destination attribution
		auto& info = bltTargetsThisWindow[this];
		info.count++;
		info.width = GetWidth();
		info.height = GetHeight();
		info.lastSrc = lpDDSrcSurface;
	}

	// [MENUDIAG] (2026-06-12): menu text appears in NO geometry path (UP/VB/strided all
	// covered + telemetried) -- find the CPU channel. Log every Blt whose DEST is the
	// primary/backbuffer/render target: glyphs composited straight onto the frame are
	// invisible under the path tracer but present under ortho=True raster passthrough.
	if (Config.DdrawOrphanOverlayLift && (IsPrimaryOrBackBuffer() || IsRenderTarget()))
	{
		LOG_LIMIT(20000, "[MENUDIAG] Blt dest=" << this <<
			(IsPrimarySurface() ? " PRIMARY" : IsBackBuffer() ? " BACKBUFFER" : " RENDERTARGET") <<
			" src=" << lpDDSrcSurface <<
			" destRect=" << lpDestRect << " srcRect=" << lpSrcRect <<
			" flags=" << Logging::hex(dwFlags));
	}

	// Path B: animation-pool collapse detection + cycling canonical.
	// Detect: once a source has fed >=POOL_DETECT_THRESHOLD distinct dests,
	// it is a pool. Dynamic gate: only redirect dests with bltCount >= 2 so
	// static terrain pools (write-once-at-load) are excluded. Cycling canonical:
	// the just-written dest becomes the canonical; previous canonical joins
	// the redirect map. Net effect: animation continues at engine rate but
	// all visible pool instances bind the SAME current frame each draw call,
	// producing one stable hash sequence Remix can map to a replacement.
	if (Config.DdrawCollapseAnimationPools && IsSurfaceTexture() && lpDDSrcSurface)
	{
		const DWORD newCount = ++destBltCount[this];
		auto& pool = poolBySource[lpDDSrcSurface];
		const bool wasUnderThreshold = pool.members.size() < POOL_DETECT_THRESHOLD;
		pool.members.insert(this);

		if (pool.members.size() >= POOL_DETECT_THRESHOLD && newCount >= 2)
		{
			// This is a re-write to an animated dest in an active pool.
			// Promote `this` to canonical; redirect all other re-written members.
			void* prevCanonical = pool.canonical;
			pool.canonical = this;
			memberToCanonical.erase(this);  // new canonical doesn't redirect
			for (void* m : pool.members)
			{
				if (m != pool.canonical && destBltCount[m] >= 2)
				{
					memberToCanonical[m] = pool.canonical;
				}
			}
			if (prevCanonical == nullptr)
			{
				poolsActiveTotal++;
				Logging::Log() << "[PathB] pool ACTIVATED src=" << lpDDSrcSurface
					<< " canonical=" << pool.canonical
					<< " poolSize=" << pool.members.size();
			}
		}
		else if (wasUnderThreshold && pool.members.size() >= POOL_DETECT_THRESHOLD)
		{
			// Pool just hit threshold but this dest is still a singleton write
			// (count==1, so likely static-load member of a pool that's about to
			// also see dynamic members). Don't activate yet; wait for a dynamic
			// member to write twice.
		}
	}

	// Check if source Surface exists
	if (lpDDSrcSurface && !ProxyAddressLookupTable.CheckSurfaceExists(lpDDSrcSurface))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not find source surface! " << lpDDSrcSurface);
		return DDERR_INVALIDPARAMS;
	}

	if (Config.Dd7to9)
	{
		// All DDBLT_ALPHA flag values, Not currently implemented in DirectDraw.
		DWORD AlphaFlags = dwFlags & (DDBLT_ALPHADEST | DDBLT_ALPHADESTCONSTOVERRIDE | DDBLT_ALPHADESTNEG | DDBLT_ALPHADESTSURFACEOVERRIDE |
			DDBLT_ALPHASRC | DDBLT_ALPHASRCCONSTOVERRIDE | DDBLT_ALPHASRCNEG | DDBLT_ALPHASRCSURFACEOVERRIDE);
		if (AlphaFlags)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: alpha flags not implemented: " << AlphaFlags);
			return DDERR_NOALPHAHW;
		}

		// All DDBLT_ZBUFFER flag values: This method does not currently support z-aware bitblt operations. None of the flags beginning with "DDBLT_ZBUFFER" are supported in DirectDraw.
		if (dwFlags & (DDBLT_ZBUFFER | DDBLT_ZBUFFERDESTCONSTOVERRIDE | DDBLT_ZBUFFERDESTOVERRIDE | DDBLT_ZBUFFERSRCCONSTOVERRIDE | DDBLT_ZBUFFERSRCOVERRIDE))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: zbuffer values not implemented!");
			return DDERR_UNSUPPORTED;
		}

		// DDBLT_DDROPS - dwDDROP is ignored as "no such ROPs are currently defined" in DirectDraw
		if (dwFlags & DDBLT_DDROPS)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DDROP values not implemented!");
			return DDERR_NODDROPSHW;
		}

		// Check for required DDBLTFX structure
		bool RequiresFxStruct = (dwFlags & (DDBLT_DDFX | DDBLT_COLORFILL | DDBLT_DEPTHFILL | DDBLT_KEYDESTOVERRIDE | DDBLT_KEYSRCOVERRIDE | DDBLT_ROP | DDBLT_ROTATIONANGLE));
		if (RequiresFxStruct && !lpDDBltFx)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DDBLTFX structure not found!");
			return DDERR_INVALIDPARAMS;
		}

		// Check for DDBLTFX structure size
		if (RequiresFxStruct && lpDDBltFx->dwSize != sizeof(DDBLTFX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DDBLTFX structure is not initialized to the correct size: " << lpDDBltFx->dwSize);
			return DDERR_INVALIDPARAMS;
		}

		// Check for rotation flags
		// ToDo: add support for other rotation flags (90,180, 270).  Not sure if any game uses these other flags.
		if ((dwFlags & DDBLT_ROTATIONANGLE) || ((dwFlags & DDBLT_DDFX) && (lpDDBltFx->dwDDFX & (DDBLTFX_ROTATE90 | DDBLTFX_ROTATE180 | DDBLTFX_ROTATE270))))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Rotation operations Not Implemented: " << Logging::hex(lpDDBltFx->dwDDFX & (DDBLTFX_ROTATE90 | DDBLTFX_ROTATE180 | DDBLTFX_ROTATE270)));
			return DDERR_NOROTATIONHW;
		}

		// Check supported raster operations
		if ((dwFlags & DDBLT_ROP) && (lpDDBltFx->dwROP != SRCCOPY && lpDDBltFx->dwROP != BLACKNESS && lpDDBltFx->dwROP != WHITENESS))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Raster operation Not Implemented " << Logging::hex(lpDDBltFx->dwROP));
			return DDERR_UNSUPPORTED;
		}

		// Get source mipmap level
		DWORD SrcMipMapLevel = 0;
		if (lpDDSrcSurface)
		{
			lpDDSrcSurface->QueryInterface(IID_GetMipMapLevel, (LPVOID*)&SrcMipMapLevel);
		}

		// Typically, Blt returns immediately with an error if the bitbltter is busy and the bitblt could not be set up. Specify the DDBLT_WAIT flag to request a synchronous bitblt.
		const bool BltWait = ((dwFlags & DDBLT_WAIT) && (dwFlags & DDBLT_DONOTWAIT) == 0);

		// Check if the scene needs to be presented
		const bool IsSkipScene = (lpDestRect) ? CheckRectforSkipScene(*lpDestRect) : false;

		// Other flags, not yet implemented in dxwrapper
		// DDBLT_ASYNC - Current dxwrapper implementation never does async if calling from multiple threads

		// Get source surface
		m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = nullptr;
		if (lpDDSrcSurface)
		{
			lpDDSrcSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDSrcSurfaceX);
			if (!lpDDSrcSurfaceX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surfaceX!");
				return DDERR_INVALIDPARAMS;
			}
		}
		else
		{
			lpDDSrcSurfaceX = this;
		}

		// Fix empty rects
		RECT SrcRect = {}, DestRect = {};
		if (lpSrcRect && lpSrcRect->left == 0 && lpSrcRect->right == 0 && lpSrcRect->top == 0 && lpSrcRect->bottom == 0 && lpDestRect)
		{
			SrcRect = { 0, 0, lpDestRect->right - lpDestRect->left, lpDestRect->bottom - lpDestRect->top };
			lpSrcRect = &SrcRect;
		}
		if (lpDestRect && lpDestRect->left == 0 && lpDestRect->right == 0 && lpDestRect->top == 0 && lpDestRect->bottom == 0 && lpSrcRect)
		{
			DestRect = { 0, 0, lpSrcRect->right - lpSrcRect->left, lpSrcRect->bottom - lpSrcRect->top };
			lpDestRect = &DestRect;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		HRESULT s_hr = (lpDDSrcSurfaceX == this) ? c_hr : lpDDSrcSurfaceX->CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr) || FAILED(s_hr))
		{
			return (c_hr == DDERR_SURFACELOST || s_hr == DDERR_SURFACELOST) ? DDERR_SURFACELOST : FAILED(c_hr) ? c_hr : s_hr;
		}

		// Handle depth stencil surface
		if (IsDepthStencil())
		{
			return CopyZBuffer(lpDDSrcSurfaceX, lpSrcRect, lpDestRect, (dwFlags & DDBLT_DEPTHFILL), lpDDBltFx ? lpDDBltFx->dwFillDepth : 0);
		}

		// Set critical section
		ScopedCriticalSection ThreadLock(GetCriticalSection());
		ScopedCriticalSection ThreadLockSrc(lpDDSrcSurfaceX->GetCriticalSection());

		// Present before write if needed
		if (PresentBlt)
		{
			BeginWritePresent(IsSkipScene);
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
		bool CopySurfaceFlag = false;
#endif

		HRESULT hr = DD_OK;

		do {
			// Check if locked from other thread
			if (BltWait)
			{
				// Wait for lock from other thread
				DWORD beginTime = timeGetTime();
				while (IsLockedFromOtherThread(MipMapLevel) || lpDDSrcSurfaceX->IsLockedFromOtherThread(MipMapLevel))
				{
					Utils::BusyWaitYield((DWORD)-1);

					// Break once timeout has passed
					if ((timeGetTime() - beginTime) >= SurfaceWaitTimeoutMS)
					{
						break;
					}
				}
			}

			do {
				// Set blt flag
				ScopedFlagSet AutoSet(IsInBlt);
				ScopedFlagSet AutoSetSrc(lpDDSrcSurfaceX->IsInBlt);

				// Do color fill
				if (dwFlags & DDBLT_COLORFILL)
				{
					// [BLITQUAD] v11: fills on either screen surface erase that canvas
					// region (mid-cycle fill+rewrite resolves before the EndScene drain,
					// so per-cycle scratch churn no longer flashes).
					if (Config.DdrawMenuBlitOverlay && (IsPrimaryOrBackBuffer() || IsRenderTarget()))
					{
						MenuBlitOverlayEraseRegion(lpDestRect, (LONG)surface.Width, (LONG)surface.Height);
					}
					hr = ColorFill(lpDestRect, lpDDBltFx->dwFillColor, MipMapLevel);
					break;
				}

				// Do supported raster operations
				if (dwFlags & DDBLT_ROP)
				{
					if (lpDDBltFx->dwROP == SRCCOPY)
					{
						// Do nothing
					}
					else if (lpDDBltFx->dwROP == BLACKNESS)
					{
						hr = ColorFill(lpDestRect, 0x00000000, MipMapLevel);
						break;
					}
					else if (lpDDBltFx->dwROP == WHITENESS)
					{
						hr = ColorFill(lpDestRect, 0xFFFFFFFF, MipMapLevel);
						break;
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: Unknown ROP: " << Logging::hex(lpDDBltFx->dwROP));
					}
				}

				// Get surface copy flags
				DWORD Flags =
					(dwFlags & (DDBLT_KEYDESTOVERRIDE | DDBLT_KEYSRCOVERRIDE | DDBLT_KEYDEST | DDBLT_KEYSRC) ? BLT_COLORKEY : 0) |
					((dwFlags & DDBLT_DDFX) && (lpDDBltFx->dwDDFX & DDBLTFX_MIRRORLEFTRIGHT) ? BLT_MIRRORLEFTRIGHT : 0) |
					((dwFlags & DDBLT_DDFX) && (lpDDBltFx->dwDDFX & DDBLTFX_MIRRORUPDOWN) ? BLT_MIRRORUPDOWN : 0);

				// Get color key
				DDCOLORKEY ColorKey = {};
				if (dwFlags & DDBLT_KEYDESTOVERRIDE)
				{
					ColorKey = lpDDBltFx->ddckDestColorkey;
				}
				else if (dwFlags & DDBLT_KEYSRCOVERRIDE)
				{
					ColorKey = lpDDBltFx->ddckSrcColorkey;
				}
				else if ((dwFlags & DDBLT_KEYDEST) && (surfaceDesc2.dwFlags & DDSD_CKDESTBLT))
				{
					ColorKey = surfaceDesc2.ddckCKDestBlt;
				}
				else if ((dwFlags & DDBLT_KEYSRC) && (lpDDSrcSurfaceX->surfaceDesc2.dwFlags & DDSD_CKSRCBLT))
				{
					ColorKey = lpDDSrcSurfaceX->surfaceDesc2.ddckCKSrcBlt;
				}
				else if (dwFlags & (DDBLT_KEYDEST | DDBLT_KEYSRC))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: color key not found!");
					hr = DDERR_INVALIDPARAMS;
					break;
				}

				D3DTEXTUREFILTERTYPE Filter = ((dwFlags & DDBLT_DDFX) && (lpDDBltFx->dwDDFX & DDBLTFX_ARITHSTRETCHY)) ? D3DTEXF_LINEAR : D3DTEXF_NONE;

				hr = CopySurface(lpDDSrcSurfaceX, lpSrcRect, lpDestRect, Filter, ColorKey.dwColorSpaceLowValue, Flags, SrcMipMapLevel, MipMapLevel);
#ifdef ENABLE_PROFILING
				CopySurfaceFlag = true;
#endif

			} while (false);

#ifdef ENABLE_PROFILING
			Logging::Log() << __FUNCTION__ << " (" << lpDDSrcSurfaceX << ") -> (" << this << ")" <<
				(CopySurfaceFlag ? " CopySurface()" : " ColorFill()") <<
				" Type = " << lpDDSrcSurfaceX->surface.Type << " " << lpDDSrcSurfaceX->surface.Pool << " -> " << surface.Type << " " << surface.Pool <<
				" hr = " << (D3DERR)hr <<
				" Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

			// If successful
			if (SUCCEEDED(hr))
			{
				// [BLITQUAD] (2026-06-12 menu-text fix): the front end composites its UI
				// (text!) via 2D Blts straight onto the backbuffer, which the path tracer
				// discards. Queue a copy of each such blit; the device re-emits them as
				// lifted self-lit quads at EndScene. Fills excluded (fullscreen clears).
				if (Config.DdrawMenuBlitOverlay && lpDDSrcSurfaceX && !(dwFlags & DDBLT_COLORFILL) &&
					(IsPrimaryOrBackBuffer() || IsRenderTarget()))
				{
					DWORD nativeKey = 0;
					bool hasKey = false;
					if (dwFlags & DDBLT_KEYSRCOVERRIDE)
					{
						nativeKey = lpDDBltFx->ddckSrcColorkey.dwColorSpaceLowValue;
						hasKey = true;
					}
					else if ((dwFlags & DDBLT_KEYSRC) && (lpDDSrcSurfaceX->surfaceDesc2.dwFlags & DDSD_CKSRCBLT))
					{
						nativeKey = lpDDSrcSurfaceX->surfaceDesc2.ddckCKSrcBlt.dwColorSpaceLowValue;
						hasKey = true;
					}
					MenuBlitOverlayQueue(lpDDSrcSurfaceX, lpSrcRect, lpDestRect, hasKey, nativeKey);
				}

				// [RECIPE] (2026-06-10): record this Blt into the dest texture's composition
				// recipe. Source identity = mip-0 content hash, cached per wrapper ptr and
				// invalidated by UniquenessValue (bumped by SetDirtyFlag on every write).
				// Fills/self-blts get sentinel markers instead of a content hash.
				if (Config.DdrawRecipeLog && IsSurfaceTexture() && surface.Texture)
				{
					uint64_t srcH = 0;
					const bool isFill = (dwFlags & DDBLT_COLORFILL) ||
						((dwFlags & DDBLT_ROP) && lpDDBltFx && (lpDDBltFx->dwROP == BLACKNESS || lpDDBltFx->dwROP == WHITENESS));
					if (isFill)
					{
						const DWORD fill = (dwFlags & DDBLT_COLORFILL) ? lpDDBltFx->dwFillColor :
							(lpDDBltFx->dwROP == WHITENESS ? 0xFFFFFFFF : 0x00000000);
						srcH = 0xF111000000000000ull | fill;	// fill marker
					}
					else if (lpDDSrcSurfaceX == this)
					{
						srcH = 0x5E1F000000000000ull;			// self-blt marker (pre-write hash would be stale)
					}
					else if (lpDDSrcSurfaceX->IsSurfaceTexture() && lpDDSrcSurfaceX->surface.Texture)
					{
						recipeBltSources.insert((void*)lpDDSrcSurfaceX);	// [LOCKW] track staging surfaces
						auto& ce = recipeSrcHashCache[(void*)lpDDSrcSurfaceX];
						if (ce.second == 0 || ce.first != lpDDSrcSurfaceX->UniquenessValue)
						{
							ce.first = lpDDSrcSurfaceX->UniquenessValue;
							ce.second = Stage3HashD3d9Texture(lpDDSrcSurfaceX->surface.Texture);
						}
						srcH = ce.second;
					}
					const RECT sR = lpSrcRect ? *lpSrcRect :
						RECT{ 0, 0, (LONG)lpDDSrcSurfaceX->surface.Width, (LONG)lpDDSrcSurfaceX->surface.Height };
					const RECT dR = lpDestRect ? *lpDestRect :
						RECT{ 0, 0, (LONG)surface.Width, (LONG)surface.Height };
					RecipeRecordBlt(surface.Texture, srcH, sR, dR, surface.Width, surface.Height);
					recipeLastBltTex = surface.Texture;
				}

				// [NAMEKEY] (2026-06-11): a full-surface verbatim texture->texture copy
				// is the staging-page upload shape ([RECIPE] runs A/B: ~99% of dest
				// content arrives exactly this way). Attach the placements the upload
				// detour accumulated for this composition to the dest texture.
				// (BltFast funnels through Blt, so this one site covers both.)
				if (Config.DdrawNameKey && IsSurfaceTexture() && surface.Texture &&
					lpDDSrcSurfaceX && lpDDSrcSurfaceX != this &&
					!(dwFlags & DDBLT_COLORFILL) && !(dwFlags & DDBLT_ROP) &&
					lpDDSrcSurfaceX->surface.Width == surface.Width &&
					lpDDSrcSurfaceX->surface.Height == surface.Height)
				{
					const RECT nsR = lpSrcRect ? *lpSrcRect :
						RECT{ 0, 0, (LONG)lpDDSrcSurfaceX->surface.Width, (LONG)lpDDSrcSurfaceX->surface.Height };
					const RECT ndR = lpDestRect ? *lpDestRect :
						RECT{ 0, 0, (LONG)surface.Width, (LONG)surface.Height };
					if (nsR.left == 0 && nsR.top == 0 && ndR.left == 0 && ndR.top == 0 &&
						nsR.right == (LONG)surface.Width && nsR.bottom == (LONG)surface.Height &&
						ndR.right == (LONG)surface.Width && ndR.bottom == (LONG)surface.Height)
					{
						NamekeyConsumeAt(surface.Texture, surface.Width, surface.Height);
					}
				}

				// Set vertical sync wait timer
				if (SUCCEEDED(c_hr) && (dwFlags & DDBLT_DDFX) && (lpDDBltFx->dwDDFX & DDBLTFX_NOTEARING))
				{
					ddrawParent->SetVsync();
				}

				if (PresentBlt)
				{
					// Set dirty flag (with the specific Blt dest rect for per-rect cache invalidation)
					SetDirtyFlag(MipMapLevel, lpDestRect);

					// Present surface
					EndWritePresent(lpDestRect, IsSkipScene);
				}
			}

		} while (false);

		// Check invalid rect
		if (hr == DDERR_INVALIDRECT)
		{
			if (ShouldPresentToWindow(true) &&
				(!lpSrcRect || (lpSrcRect->left < lpSrcRect->right && lpSrcRect->top < lpSrcRect->bottom)) &&
				(!lpDestRect || (lpDestRect->left < lpDestRect->right && lpDestRect->top < lpDestRect->bottom)))
			{
				hr = DD_OK;
			}
			else
			{
				if (lpDDSrcSurface)
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpSrcRect << " -> " << lpDestRect);
				}
				else
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpSrcRect);
				}
			}
		}

		// Check if surface was busy
		if (!BltWait && (hr == DDERR_SURFACEBUSY || IsLockedFromOtherThread(MipMapLevel) || lpDDSrcSurfaceX->IsLockedFromOtherThread(MipMapLevel)))
		{
			hr = DDERR_WASSTILLDRAWING;
		}
		else if (FAILED(hr) && (IsLost() == DDERR_SURFACELOST || lpDDSrcSurfaceX->IsLost() == DDERR_SURFACELOST))
		{
			hr = DDERR_SURFACELOST;
		}

		// Return
		return hr;
	}

	if (lpDDSrcSurface)
	{
		lpDDSrcSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSrcSurface);
	}

	HRESULT hr = ProxyInterface->Blt(lpDestRect, lpDDSrcSurface, lpSrcRect, dwFlags, lpDDBltFx);

	// Fix for some games that calculate the rect incorrectly
	if (hr == DDERR_INVALIDRECT)
	{
		RECT SrcRect, DestRect;
		if (lpSrcRect)
		{
			SrcRect = *lpSrcRect;
			SrcRect.left -= 1;
			SrcRect.bottom -= 1;
			lpSrcRect = &SrcRect;
		}
		if (lpDestRect)
		{
			DestRect = *lpDestRect;
			DestRect.left -= 1;
			DestRect.bottom -= 1;
			lpDestRect = &DestRect;
		}
		hr = ProxyInterface->Blt(lpDestRect, lpDDSrcSurface, lpSrcRect, dwFlags, lpDDBltFx);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::BltBatch(LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags, DWORD MipMapLevel)
{
	UNREFERENCED_PARAMETER(dwFlags);

	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lpDDBltBatch)
	{
		return DDERR_INVALIDPARAMS;
	}

	// Check for device interface before doing batch
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr))
	{
		return c_hr;
	}

	HRESULT hr = DD_OK;

	bool IsSkipScene = false;

	ScopedCriticalSection ThreadLock(GetCriticalSection());

	// Present before write if needed
	BeginWritePresent(IsSkipScene);

	{
		// Set blt flag
		ScopedFlagSet AutoSet(IsInBltBatch);

		for (DWORD x = 0; x < dwCount; x++)
		{
			IsSkipScene |= (lpDDBltBatch[x].lprDest) ? CheckRectforSkipScene(*lpDDBltBatch[x].lprDest) : false;

			hr = Blt(lpDDBltBatch[x].lprDest, (LPDIRECTDRAWSURFACE7)lpDDBltBatch[x].lpDDSSrc, lpDDBltBatch[x].lprSrc, lpDDBltBatch[x].dwFlags | DDBLT_DONOTWAIT, lpDDBltBatch[x].lpDDBltFx, MipMapLevel, false);
			if (FAILED(hr))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: BltBatch failed before the end! " << x << " of " << dwCount << " " << (DDERR)hr);
				break;
			}
		}
	}

	if (SUCCEEDED(hr))
	{
		// Set dirty flag
		SetDirtyFlag(MipMapLevel);

		// Present surface
		EndWritePresent(nullptr, IsSkipScene);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::BltFast(DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE7 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	// Check if source Surface exists
	if (lpDDSrcSurface && !ProxyAddressLookupTable.CheckSurfaceExists(lpDDSrcSurface))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not find source surface! " << lpDDSrcSurface);
		return DDERR_INVALIDPARAMS;
	}

	if (Config.Dd7to9)
	{
		// NOTE: If you call IDirectDrawSurface7::BltFast on a surface with an attached clipper, it returns DDERR_UNSUPPORTED.
		if (attachedClipper)
		{
			return DDERR_UNSUPPORTED;
		}

		// Convert BltFast flags into Blt flags
		DWORD Flags = DDBLT_ASYNC;
		if (dwFlags & DDBLTFAST_SRCCOLORKEY)
		{
			Flags |= DDBLT_KEYSRC;
		}
		if (dwFlags & DDBLTFAST_DESTCOLORKEY)
		{
			Flags |= DDBLT_KEYDEST;
		}
		if (dwFlags & DDBLTFAST_WAIT)
		{
			Flags |= DDBLT_WAIT;
		}

		// Get source surface
		m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = nullptr;
		if (lpDDSrcSurface)
		{
			lpDDSrcSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDSrcSurfaceX);
			if (!lpDDSrcSurfaceX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surfaceX!");
				return DDERR_GENERIC;
			}
		}
		else
		{
			lpDDSrcSurfaceX = this;
		}

		// Get SrcRect
		RECT SrcRect = {};
		if (!lpDDSrcSurfaceX->CheckCoordinates(SrcRect, lpSrcRect, nullptr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Invalid rect: " << lpSrcRect);
		}

		// Create DestRect
		RECT DestRect = { (LONG)dwX, (LONG)dwY, SrcRect.right - SrcRect.left + (LONG)dwX , SrcRect.bottom - SrcRect.top + (LONG)dwY };
		LPRECT pDestRect = &DestRect;
		if (!lpSrcRect && !dwX && !dwY)
		{
			pDestRect = nullptr;
		}

		// Call Blt
		return Blt(pDestRect, lpDDSrcSurface, lpSrcRect, Flags, nullptr, MipMapLevel);
	}

	if (lpDDSrcSurface)
	{
		lpDDSrcSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSrcSurface);
	}

	HRESULT hr = ProxyInterface->BltFast(dwX, dwY, lpDDSrcSurface, lpSrcRect, dwFlags);

	// Fix for some games that calculate the rect incorrectly
	if (lpSrcRect && hr == DDERR_INVALIDRECT)
	{
		RECT SrcRect = *lpSrcRect;
		SrcRect.left -= 1;
		SrcRect.bottom -= 1;
		hr = ProxyInterface->BltFast(dwX, dwY, lpDDSrcSurface, &SrcRect, dwFlags);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::DeleteAttachedSurface(DWORD dwFlags, LPDIRECTDRAWSURFACE7 lpDDSAttachedSurface)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// dwFlags: Reserved. Must be zero.
		if (!lpDDSAttachedSurface || dwFlags)
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirectDrawSurfaceX *lpAttachedSurfaceX = nullptr;

		lpDDSAttachedSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpAttachedSurfaceX);

		if (!DoesAttachedSurfaceExist(lpAttachedSurfaceX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find attached surface");
			return DDERR_SURFACENOTATTACHED;
		}

		if (!WasAttachedSurfaceAdded(lpAttachedSurfaceX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: only surfaces added with AddAttachedSurface can be deleted with this method");
			return DDERR_CANNOTDETACHSURFACE;
		}

		// clear zbuffer
		if (lpAttachedSurfaceX->IsDepthStencil() &&
			(ddrawParent->GetDepthStencilSurface() == lpAttachedSurfaceX || ddrawParent->GetRenderTargetSurface() == this))
		{
			ddrawParent->SetDepthStencilSurface(nullptr);
		}

		RemoveAttachedSurfaceFromMap(lpAttachedSurfaceX);

		lpDDSAttachedSurface->Release();

		return DD_OK;
	}

	if (lpDDSAttachedSurface)
	{
		lpDDSAttachedSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSAttachedSurface);
	}

	return ProxyInterface->DeleteAttachedSurface(dwFlags, lpDDSAttachedSurface);
}

HRESULT m_IDirectDrawSurfaceX::EnumAttachedSurfaces(LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		return EnumAttachedSurfaces2(lpContext, nullptr, lpEnumSurfacesCallback, MipMapLevel, DirectXVersion);
	}

	if (!lpEnumSurfacesCallback)
	{
		return DDERR_INVALIDPARAMS;
	}

	struct EnumSurface
	{
		LPVOID lpContext;
		LPDDENUMSURFACESCALLBACK lpCallback;
		DWORD DirectXVersion;

		static HRESULT CALLBACK ConvertCallback(LPDIRECTDRAWSURFACE lpDDSurface, LPDDSURFACEDESC lpDDSurfaceDesc, LPVOID lpContext)
		{
			EnumSurface *self = (EnumSurface*)lpContext;

			if (lpDDSurface)
			{
				lpDDSurface = (LPDIRECTDRAWSURFACE)ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(lpDDSurface, self->DirectXVersion);
			}

			return self->lpCallback(lpDDSurface, lpDDSurfaceDesc, self->lpContext);
		}
	} CallbackContext = {};
	CallbackContext.lpContext = lpContext;
	CallbackContext.lpCallback = lpEnumSurfacesCallback;
	CallbackContext.DirectXVersion = DirectXVersion;

	return GetProxyInterfaceV3()->EnumAttachedSurfaces(&CallbackContext, EnumSurface::ConvertCallback);
}

HRESULT m_IDirectDrawSurfaceX::EnumAttachedSurfaces2(LPVOID lpContext, LPDDENUMSURFACESCALLBACK7 lpEnumSurfacesCallback7, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lpEnumSurfacesCallback7 && !lpEnumSurfacesCallback)
	{
		return DDERR_INVALIDPARAMS;
	}

	struct EnumSurface
	{
		LPVOID lpContext;
		LPDDENUMSURFACESCALLBACK7 lpCallback7;
		LPDDENUMSURFACESCALLBACK lpCallback;
		DWORD DirectXVersion;

		static HRESULT CALLBACK ConvertCallback(LPDIRECTDRAWSURFACE7 lpDDSurface, LPDDSURFACEDESC2 lpDDSurfaceDesc2, LPVOID lpContext)
		{
			EnumSurface* self = (EnumSurface*)lpContext;

			if (!Config.Dd7to9)
			{
				lpDDSurface = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(lpDDSurface, self->DirectXVersion);
			}

			if (self->lpCallback7)
			{
				return self->lpCallback7(lpDDSurface, lpDDSurfaceDesc2, self->lpContext);
			}
			else if (self->lpCallback)
			{
				DDSURFACEDESC Desc = {};
				Desc.dwSize = sizeof(DDSURFACEDESC);
				ConvertSurfaceDesc(Desc, *lpDDSurfaceDesc2);

				return self->lpCallback((LPDIRECTDRAWSURFACE)lpDDSurface, &Desc, self->lpContext);
			}

			return DDENUMRET_OK;
		}
	} CallbackContext = {};
	CallbackContext.lpContext = lpContext;
	CallbackContext.lpCallback7 = lpEnumSurfacesCallback7;
	CallbackContext.lpCallback = lpEnumSurfacesCallback;
	CallbackContext.DirectXVersion = DirectXVersion;

	if (Config.Dd7to9)
	{
		// Handle mipmaps
		if (!MipMaps.empty())
		{
			LPDIRECTDRAWSURFACE7 lpDDAttachedSurface = nullptr;
			if (SUCCEEDED(GetMipMapSubLevel(&lpDDAttachedSurface, MipMapLevel, DirectXVersion)))
			{
				DDSURFACEDESC2 Desc2 = {};
				Desc2.dwSize = sizeof(DDSURFACEDESC2);
				GetSurfaceDesc2(&Desc2, MipMapLevel + 1, DirectXVersion);
				if (EnumSurface::ConvertCallback(lpDDAttachedSurface, &Desc2, &CallbackContext) == DDENUMRET_CANCEL)
				{
					return DD_OK;
				}
			}
		}
		for (auto& it : AttachedSurfaceMap)
		{
			// This method enumerates all the surfaces attached to a given surface.
			// In a flipping chain of three or more surfaces, only one surface is enumerated because each surface is attached only to the next surface in the flipping chain.
			// In such a configuration, you can call EnumAttachedSurfaces on each successive surface to walk the entire flipping chain.
			// The front buffer should not be returned as attached.
			if (!(it.second.pSurface->surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_FRONTBUFFER))
			{
				DDSURFACEDESC2 Desc2 = {};
				Desc2.dwSize = sizeof(DDSURFACEDESC2);
				it.second.pSurface->GetSurfaceDesc2(&Desc2, 0, DirectXVersion);
				LPDIRECTDRAWSURFACE7 lpSurface = (LPDIRECTDRAWSURFACE7)it.second.pSurface->GetWrapperInterfaceX(DirectXVersion);
				if (EnumSurface::ConvertCallback(lpSurface, &Desc2, &CallbackContext) == DDENUMRET_CANCEL)
				{
					return DD_OK;
				}
			}
		}

		return DD_OK;
	}

	return ProxyInterface->EnumAttachedSurfaces(&CallbackContext, EnumSurface::ConvertCallback);
}

HRESULT m_IDirectDrawSurfaceX::EnumOverlayZOrders(DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpfnCallback, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		return EnumOverlayZOrders2(dwFlags, lpContext, nullptr, lpfnCallback, DirectXVersion);
	}

	if (!lpfnCallback)
	{
		return DDERR_INVALIDPARAMS;
	}

	struct EnumSurface
	{
		LPVOID lpContext;
		LPDDENUMSURFACESCALLBACK lpCallback;
		DWORD DirectXVersion;

		static HRESULT CALLBACK ConvertCallback(LPDIRECTDRAWSURFACE lpDDSurface, LPDDSURFACEDESC lpDDSurfaceDesc, LPVOID lpContext)
		{
			EnumSurface *self = (EnumSurface*)lpContext;

			if (lpDDSurface)
			{
				lpDDSurface = (LPDIRECTDRAWSURFACE)ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(lpDDSurface, self->DirectXVersion);
			}

			return self->lpCallback(lpDDSurface, lpDDSurfaceDesc, self->lpContext);
		}
	} CallbackContext = {};
	CallbackContext.lpContext = lpContext;
	CallbackContext.lpCallback = lpfnCallback;
	CallbackContext.DirectXVersion = DirectXVersion;

	return GetProxyInterfaceV3()->EnumOverlayZOrders(dwFlags, &CallbackContext, EnumSurface::ConvertCallback);
}

HRESULT m_IDirectDrawSurfaceX::EnumOverlayZOrders2(DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK7 lpfnCallback7, LPDDENUMSURFACESCALLBACK lpfnCallback, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lpfnCallback7 && !lpfnCallback)
	{
		return DDERR_INVALIDPARAMS;
	}

	struct EnumSurface
	{
		LPVOID lpContext;
		LPDDENUMSURFACESCALLBACK7 lpCallback7;
		LPDDENUMSURFACESCALLBACK lpCallback;
		DWORD DirectXVersion;

		static HRESULT CALLBACK ConvertCallback(LPDIRECTDRAWSURFACE7 lpDDSurface, LPDDSURFACEDESC2 lpDDSurfaceDesc2, LPVOID lpContext)
		{
			EnumSurface *self = (EnumSurface*)lpContext;

			if (!Config.Dd7to9)
			{
				lpDDSurface = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(lpDDSurface, self->DirectXVersion);
			}

			if (self->lpCallback7)
			{
				return self->lpCallback7(lpDDSurface, lpDDSurfaceDesc2, self->lpContext);
			}
			else if (self->lpCallback)
			{
				DDSURFACEDESC Desc = {};
				Desc.dwSize = sizeof(DDSURFACEDESC);
				ConvertSurfaceDesc(Desc, *lpDDSurfaceDesc2);

				return self->lpCallback((LPDIRECTDRAWSURFACE)lpDDSurface, &Desc, self->lpContext);
			}

			return DDENUMRET_OK;
		}
	} CallbackContext = {};
	CallbackContext.lpContext = lpContext;
	CallbackContext.lpCallback7 = lpfnCallback7;
	CallbackContext.lpCallback = lpfnCallback;
	CallbackContext.DirectXVersion = DirectXVersion;

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	return ProxyInterface->EnumOverlayZOrders(dwFlags, &CallbackContext, EnumSurface::ConvertCallback);
}

HRESULT m_IDirectDrawSurfaceX::Flip(LPDIRECTDRAWSURFACE7 lpDDSurfaceTargetOverride, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" SrcSurface = " << lpDDSurfaceTargetOverride <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Version = " << DirectXVersion;

	if (Config.Dd7to9)
	{
		if ((dwFlags & (DDFLIP_EVEN | DDFLIP_ODD)) == (DDFLIP_EVEN | DDFLIP_ODD))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid flags!");
			return DDERR_INVALIDPARAMS;
		}

		// Flip can be called only for a surface that has the DDSCAPS_FLIP and DDSCAPS_FRONTBUFFER capabilities
		if (!IsFlipSurface())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: This surface cannot be flipped");
			return DDERR_NOTFLIPPABLE;
		}

		if (dwFlags & DDFLIP_STEREO)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Stereo flipping not implemented");
			return DDERR_NOSTEREOHARDWARE;
		}

		if ((dwFlags & (DDFLIP_INTERVAL2 | DDFLIP_INTERVAL3 | DDFLIP_INTERVAL4)) && (surfaceDesc2.ddsCaps.dwCaps2 & DDCAPS2_FLIPINTERVAL))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Interval flipping not implemented");
			return DDERR_UNSUPPORTED;
		}

		if (dwFlags & (DDFLIP_ODD | DDFLIP_EVEN))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Even and odd flipping not implemented");
			return DDERR_UNSUPPORTED;
		}

		const bool FlipWait = (((dwFlags & DDFLIP_WAIT) || DirectXVersion == 7) && (dwFlags & DDFLIP_DONOTWAIT) == 0);

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Check if is in scene
		if (Using3D)
		{
			m_IDirect3DX* D3DX = *ddrawParent->GetCurrentD3D();

			DWORD x = 0;
			while (D3DX)
			{
				m_IDirect3DDeviceX* D3DDeviceX = D3DX->GetNextD3DDevice(x++);

				if (!D3DDeviceX)
				{
					break;
				}
				if (D3DDeviceX->IsDeviceInScene())
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Device is in scene when attempting to flip!");
					return DDERR_GENERIC;
				}
			}
		}

		// Create flip list
		std::vector<m_IDirectDrawSurfaceX*> FlipList;
		c_hr = GetFlipList(FlipList, lpDDSurfaceTargetOverride);
		if (FAILED(c_hr))
		{
			return c_hr;
		}
		if (FlipList.size() < 2)
		{
			return DDERR_NOTFLIPPABLE;
		}

		// Lambda function to check if any surface is busy
		auto FlipSurfacesAreLockedFromOtherThread = [&FlipList]() {
			for (m_IDirectDrawSurfaceX*& pSurfaceX : FlipList)
			{
				if (pSurfaceX->IsLockedFromOtherThread(0))
				{
					return true;
				}
			}
			return false; };

		// Prepare critical sections
		std::vector<ScopedCriticalSection> ThreadLocks;
		ThreadLocks.reserve(FlipList.size() + 1);
		ThreadLocks.emplace_back(GetCriticalSection());

		// Set critical section for each surface
		for (auto& pSurfaceX : FlipList)
		{
			// Constructs AUTOCRITICALLOCK and locks the section
			ThreadLocks.emplace_back(pSurfaceX->GetCriticalSection());
		}

		// Present before write if needed
		BeginWritePresent(false);

		HRESULT hr = DD_OK;

		do {
			// Check if locked from other thread
			if (FlipWait)
			{
				// Wait for locks from other threads
				DWORD beginTime = timeGetTime();
				while (FlipSurfacesAreLockedFromOtherThread())
				{
					Utils::BusyWaitYield((DWORD)-1);

					// Break once timeout has passed
					if ((timeGetTime() - beginTime) >= SurfaceWaitTimeoutMS)
					{
						break;
					}
				}
			}

			// Check if any surface is busy
			for (m_IDirectDrawSurfaceX*& pSurfaceX : FlipList)
			{
				if (pSurfaceX->IsSurfaceBusy())
				{
					if (FlipWait)
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: surface is busy: " <<
							pSurfaceX->IsSurfaceLocked() << " DC: " << pSurfaceX->IsSurfaceInDC() << " Blt: " << pSurfaceX->IsSurfaceBlitting());
						hr = DDERR_WASSTILLDRAWING;
						break;
					}
					hr = IsLost() == DDERR_SURFACELOST ? DDERR_SURFACELOST : DDERR_GENERIC;
					break;
				}
			}

			{
				// Set flip flag
				ScopedFlagSet AutoSet(IsInFlip);

				// Clear surface before flip if system memory
				if (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY)
				{
					if (FAILED(ColorFill(nullptr, Config.DdrawFlipFillColor, 0)))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: could not color fill surface.");
					}
					ClearDirtyFlags();
				}

				// Execute flip
				for (size_t x = 0; x < FlipList.size() - 1; x++)
				{
					SwapAddresses(&FlipList[x]->surface, &FlipList[x + 1]->surface);
				}
			}

#ifdef ENABLE_PROFILING
			Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr;
#endif

			// If texture is not dirty then mark it as dirty in case the game wrote to the memory directly (Nox does this)
			if (!surface.IsDirtyFlag)
			{
				// Set dirty flag
				SetDirtyFlag(0);

				// Keep surface insync
				EndWriteSyncSurfaces(nullptr);

				// Add dirty rect
				LPDIRECT3DTEXTURE9 displayTexture = Get3DTexture();
				if (displayTexture)
				{
					displayTexture->AddDirtyRect(nullptr);
				}
			}

			// Set vertical sync wait timer
			if ((dwFlags & DDFLIP_NOVSYNC) == 0)
			{
				ddrawParent->SetVsync();
			}

			// Present surface
			EndWritePresent(nullptr, false);

			if (IsRenderTarget())
			{
				ddrawParent->SetCurrentRenderTarget();
			}

		} while (false);

		return hr;
	}

	if (lpDDSurfaceTargetOverride)
	{
		lpDDSurfaceTargetOverride->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSurfaceTargetOverride);
	}

	return ProxyInterface->Flip(lpDDSurfaceTargetOverride, dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::GetAttachedSurface(LPDDSCAPS lpDDSCaps, LPDIRECTDRAWSURFACE7 FAR * lplpDDAttachedSurface, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDDAttachedSurface)
		{
			return DDERR_INVALIDPARAMS;
		}

		DDSCAPS2 Caps2;
		ConvertCaps(Caps2, *lpDDSCaps);

		return GetAttachedSurface2((lpDDSCaps ? &Caps2 : nullptr), lplpDDAttachedSurface, MipMapLevel, DirectXVersion);
	}

	HRESULT hr = GetProxyInterfaceV3()->GetAttachedSurface(lpDDSCaps, (LPDIRECTDRAWSURFACE3*)lplpDDAttachedSurface);

	if (SUCCEEDED(hr) && lplpDDAttachedSurface)
	{
		*lplpDDAttachedSurface = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(*lplpDDAttachedSurface, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::GetAttachedSurface2(LPDDSCAPS2 lpDDSCaps2, LPDIRECTDRAWSURFACE7 FAR * lplpDDAttachedSurface, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDDAttachedSurface || !lpDDSCaps2)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpDDAttachedSurface = nullptr;

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		m_IDirectDrawSurfaceX *lpFoundSurface = nullptr;

		// Check if attached surface exists
		for (auto& it : AttachedSurfaceMap)
		{
			m_IDirectDrawSurfaceX *lpSurface = it.second.pSurface;

			if ((lpSurface->GetSurfaceCaps().dwCaps & lpDDSCaps2->dwCaps) == lpDDSCaps2->dwCaps)
			{
				if (lpFoundSurface)
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: more than one surface is attached that matches the capabilities requested.");
					return DDERR_GENERIC;
				}

				lpFoundSurface = lpSurface;
			}
		}

		// No attached surface found
		if (!lpFoundSurface)
		{
			// Handle mipmaps
			if ((lpDDSCaps2->dwCaps & DDSCAPS_MIPMAP) && (GetSurfaceCaps().dwCaps & lpDDSCaps2->dwCaps) == lpDDSCaps2->dwCaps)
			{
				// Normal MipMaps
				if (SUCCEEDED(GetMipMapSubLevel(lplpDDAttachedSurface, MipMapLevel, DirectXVersion)))
				{
					(*lplpDDAttachedSurface)->AddRef();

					return DD_OK;
				}
				// Use dummy mipmap surface to prevent some games from crashing
				DWORD Level = (MipMapLevel & ~DXW_IS_MIPMAP_DUMMY);
				if (Level < GetMaxMipMapLevel(surfaceDesc2.dwWidth, surfaceDesc2.dwHeight) - 1)
				{
					while (MipMaps.size() < Level + 1)
					{
						MIPMAP MipMap;
						MipMaps.push_back(MipMap);
					}

					if (SUCCEEDED(GetMipMapLevelAddr(lplpDDAttachedSurface, MipMaps[Level], DXW_IS_MIPMAP_DUMMY + Level + 1, DirectXVersion)))
					{
						MipMaps[Level].IsDummy = true;

						(*lplpDDAttachedSurface)->AddRef();

						return DD_OK;
					}
				}
				return DDERR_NOTFOUND;
			}

			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to find attached surface that matches the capabilities requested: " << *lpDDSCaps2 <<
				" Attached number of surfaces: " << AttachedSurfaceMap.size() << " MaxMipMapLevel: " << MaxMipMapLevel << " Caps: " << surfaceDesc2.ddsCaps);
			return DDERR_NOTFOUND;
		}

		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)lpFoundSurface->GetWrapperInterfaceX(DirectXVersion);

		(*lplpDDAttachedSurface)->AddRef();

		return DD_OK;
	}

	DDSCAPS2 DDSCaps2;
	
	if (lpDDSCaps2)
	{
		DDSCaps2 = *lpDDSCaps2;

		lpDDSCaps2 = &DDSCaps2;

		if (ProxyDirectXVersion != DirectXVersion)
		{
			DDSCaps2.dwCaps2 = 0;
			DDSCaps2.dwCaps3 = 0;
			DDSCaps2.dwCaps4 = 0;
		}
	}

	HRESULT hr = ProxyInterface->GetAttachedSurface(lpDDSCaps2, lplpDDAttachedSurface);

	if (SUCCEEDED(hr) && lplpDDAttachedSurface)
	{
		*lplpDDAttachedSurface = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(*lplpDDAttachedSurface, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::GetBltStatus(DWORD dwFlags, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check if device interface is lost
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Inquires whether a blit involving this surface can occur immediately, and returns DD_OK if the blit can be completed.
		if (dwFlags == DDGBS_CANBLT)
		{
			if (IsSurfaceBlitting())
			{
				return DDERR_WASSTILLDRAWING;
			}
			if (IsSurfaceBusy(MipMapLevel))
			{
				return DDERR_SURFACEBUSY;
			}
			return DD_OK;
		}
		// Inquires whether the blit is done, and returns DD_OK if the last blit on this surface has completed.
		else if (dwFlags == DDGBS_ISBLTDONE)
		{
			if (IsSurfaceBlitting())
			{
				return DDERR_WASSTILLDRAWING;
			}
			return DD_OK;
		}

		return DDERR_INVALIDPARAMS;
	}

	return ProxyInterface->GetBltStatus(dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::GetCaps(LPDDSCAPS lpDDSCaps)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSCaps)
		{
			return DDERR_INVALIDPARAMS;
		}

		DDSCAPS2 Caps2;

		HRESULT hr = GetCaps2(&Caps2);

		// Convert back to DDSCAPS
		if (SUCCEEDED(hr))
		{
			ConvertCaps(*lpDDSCaps, Caps2);
		}

		return hr;
	}

	return GetProxyInterfaceV3()->GetCaps(lpDDSCaps);
}

HRESULT m_IDirectDrawSurfaceX::GetCaps2(LPDDSCAPS2 lpDDSCaps2)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSCaps2)
		{
			return DDERR_INVALIDPARAMS;
		}

		*lpDDSCaps2 = surfaceDesc2.ddsCaps;

		return DD_OK;
	}

	return ProxyInterface->GetCaps(lpDDSCaps2);
}

HRESULT m_IDirectDrawSurfaceX::GetClipper(LPDIRECTDRAWCLIPPER FAR * lplpDDClipper)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDDClipper)
		{
			return DDERR_INVALIDPARAMS;
		}

		// No clipper attached
		if (!attachedClipper)
		{
			*lplpDDClipper = nullptr;
			return DDERR_NOCLIPPERATTACHED;
		}

		// Return attached clipper
		*lplpDDClipper = (LPDIRECTDRAWCLIPPER)attachedClipper;

		// Increase ref counter
		(*lplpDDClipper)->AddRef();

		// Success
		return DD_OK;
	}

	HRESULT hr = ProxyInterface->GetClipper(lplpDDClipper);

	if (SUCCEEDED(hr) && lplpDDClipper)
	{
		*lplpDDClipper = ProxyAddressLookupTable.FindAddress<m_IDirectDrawClipper>(*lplpDDClipper);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::GetColorKey(DWORD dwFlags, LPDDCOLORKEY lpDDColorKey)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check index
		if (!lpDDColorKey)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Get color key index
		DWORD dds = 0;
		switch (dwFlags)
		{
		case DDCKEY_DESTBLT:
			dds = DDSD_CKDESTBLT;
			break;
		case DDCKEY_DESTOVERLAY:
			dds = DDSD_CKDESTOVERLAY;
			break;
		case DDCKEY_SRCBLT:
			dds = DDSD_CKSRCBLT;
			break;
		case DDCKEY_SRCOVERLAY:
			dds = DDSD_CKSRCOVERLAY;
			break;
		default:
			return DDERR_INVALIDPARAMS;
		}

		// Check if color key is set
		if (!(surfaceDesc2.dwFlags & dds))
		{
			return DDERR_NOCOLORKEY;
		}

		// Set color key
		switch (dds)
		{
		case DDSD_CKDESTBLT:
			*lpDDColorKey = surfaceDesc2.ddckCKDestBlt;
			break;
		case DDSD_CKDESTOVERLAY:
			*lpDDColorKey = surfaceDesc2.ddckCKDestOverlay;
			break;
		case DDSD_CKSRCBLT:
			*lpDDColorKey = surfaceDesc2.ddckCKSrcBlt;
			break;
		case DDSD_CKSRCOVERLAY:
			*lpDDColorKey = surfaceDesc2.ddckCKSrcOverlay;
			break;
		}

		// Return
		return DD_OK;
	}

	return ProxyInterface->GetColorKey(dwFlags, lpDDColorKey);
}

HRESULT m_IDirectDrawSurfaceX::GetDC(HDC FAR* lphDC, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" lpDC = " << (void*)lphDC <<
		" MipMapLevel = " << MipMapLevel;

	// [MENUDIAG]: GDI text (TextOut on a surface DC) is invisible to all draw/Blt paths.
	if (Config.DdrawOrphanOverlayLift)
	{
		LOG_LIMIT(1000, "[MENUDIAG] GetDC surf=" << this <<
			(IsPrimarySurface() ? " PRIMARY" : IsBackBuffer() ? " BACKBUFFER" : IsSurfaceTexture() ? " TEXTURE" : " OTHER"));
	}

	if (Config.Dd7to9)
	{
		if (!lphDC)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lphDC = nullptr;

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		ScopedCriticalSection ThreadLock(GetCriticalSection());

		// MipMap level support
		if (MipMapLevel && (IsUsingEmulation() || DCRequiresEmulation || IsDummyMipMap(MipMapLevel)))
		{
			// Prepare surfaceDesc
			DDSURFACEDESC2 Desc2 = {};
			Desc2.dwSize = sizeof(Desc2);
			GetSurfaceDesc2(&Desc2, MipMapLevel, 7);

			LOG_LIMIT(100, __FUNCTION__ << " Error: Emulated DC not supported from MipMap level: " << MipMapLevel
				<< " UsingEmulation: " << IsUsingEmulation()
				<< " RequiresEmulation: " << DCRequiresEmulation
				<< " DummyMipMap: " << IsDummyMipMap(MipMapLevel)
				<< " surface: " << Desc2
			);
			return DDERR_UNSUPPORTED;
		}

		if (GetDCLevel[MipMapLevel])
		{
			*lphDC = GetDCLevel[MipMapLevel];
			return DD_OK;
		}

		// Present before write if needed
		BeginWritePresent(false);

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		// Check if render target should use shadow
		if (MipMapLevel == 0 && (surface.Usage & D3DUSAGE_RENDERTARGET) && !IsUsingShadowSurface())
		{
			SetRenderTargetShadow();
		}

		HRESULT hr = DD_OK;

		do {

			if (IsUsingEmulation() || DCRequiresEmulation)
			{
				if (!IsUsingEmulation())
				{
					if (FAILED(CreateDCSurface()))
					{
						hr = DDERR_GENERIC;
						break;
					}

					CopyToEmulatedSurface(nullptr);
				}

				// Set new palette data
				UpdatePaletteData();

				// Read surface from GDI
				if (ShouldReadFromGDI())
				{
					CopyEmulatedSurfaceFromGDI(nullptr);
				}

				// Prepare GameDC
				SetEmulationGameDC();

				*lphDC = surface.emu->GameDC;
			}
			else
			{
				// Get surface
				ScopedGetMipMapContext Dest(this, MipMapLevel);
				if (!Dest.GetSurface())
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
					hr = DDERR_GENERIC;
					break;
				}

				// Get device context
				hr = Dest.GetSurface()->GetDC(lphDC);
				if (FAILED(hr))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not get device context!");
					hr = (hr == D3DERR_DEVICELOST || IsLost() == DDERR_SURFACELOST) ? DDERR_SURFACELOST :
						(hr == DDERR_WASSTILLDRAWING || IsSurfaceBusy(MipMapLevel)) ? DDERR_SURFACEBUSY : DDERR_GENERIC;
					break;
				}
			}

			// Handle clipper
			if (attachedClipper && attachedClipper->HasClipList())
			{
				HRGN hClipRgn = CreateRectRgn(0, 0, 0, 0);
				if (SUCCEEDED(attachedClipper->GetClipRegion(hClipRgn)))
				{
					SelectClipRgn(*lphDC, hClipRgn);
				}
				DeleteObject(hClipRgn);
			}

			// Set DC level
			GetDCLevel[MipMapLevel] = *lphDC;

		} while (false);

		if (FAILED(hr))
		{
			hr = IsSurfaceBusy(MipMapLevel) ? DDERR_SURFACEBUSY : IsLost() == DDERR_SURFACELOST ? DDERR_SURFACELOST : DDERR_GENERIC;
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ")" <<
			" Type = " << surface.Type << " " << surface.Pool <<
			" hr = " << (D3DERR)hr <<
			" Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	return ProxyInterface->GetDC(lphDC);
}

HRESULT m_IDirectDrawSurfaceX::GetFlipStatus(DWORD dwFlags, bool CheckOnly)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Flip can be called only for a surface that has the DDSCAPS_FLIP and DDSCAPS_FRONTBUFFER capabilities
		if (!IsFlipSurface())
		{
			if (!CheckOnly)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: This surface cannot be flipped");
			}
			return DDERR_INVALIDSURFACETYPE;
		}

		// Check if device interface is lost
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Queries whether the surface can flip now. The method returns DD_OK if the flip can be completed.
		if ((dwFlags == DDGFS_CANFLIP))
		{
			// Check if flip is still happening
			if (IsInFlip)
			{
				return DDERR_WASSTILLDRAWING;
			}

			// Create flip list
			std::vector<m_IDirectDrawSurfaceX*> FlipList;
			HRESULT hr = GetFlipList(FlipList, nullptr);

			// Check if there is a backbuffer
			if (FlipList.size() < 2)
			{
				return DDERR_INVALIDSURFACETYPE;
			}

			// Check if surface is busy
			if (IsSurfaceBusy())
			{
				return DDERR_SURFACEBUSY;
			}
			if (FAILED(hr))
			{
				for (auto& entry : FlipList)
				{
					if (entry->IsSurfaceBusy())
					{
						return DDERR_SURFACEBUSY;
					}
				}
			}
			return DD_OK;
		}
		// Queries whether the flip is done. The method returns DD_OK if the last flip on this surface has completed.
		else if (dwFlags == DDGFS_ISFLIPDONE)
		{
			if (IsInFlip)
			{
				return DDERR_WASSTILLDRAWING;
			}
			return DD_OK;
		}

		return DDERR_INVALIDPARAMS;
	}

	return ProxyInterface->GetFlipStatus(dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::GetOverlayPosition(LPLONG lplX, LPLONG lplY)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Not fully Implemented.");

		if (!lplX || !lplY)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Set lplX and lplY to X, Y of this overlay surface
		*lplX = overlayX;
		*lplY = overlayY;

		return DD_OK;
	}

	return ProxyInterface->GetOverlayPosition(lplX, lplY);
}

HRESULT m_IDirectDrawSurfaceX::GetPalette(LPDIRECTDRAWPALETTE FAR * lplpDDPalette)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDDPalette)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpDDPalette = nullptr;

		// No palette attached
		if (!attachedPalette)
		{
			return DDERR_NOPALETTEATTACHED;
		}

		// Return attached palette
		*lplpDDPalette = (LPDIRECTDRAWPALETTE)attachedPalette;

		// Increase ref counter
		(*lplpDDPalette)->AddRef();

		// Success
		return DD_OK;
	}

	HRESULT hr = ProxyInterface->GetPalette(lplpDDPalette);

	if (SUCCEEDED(hr) && lplpDDPalette)
	{
		*lplpDDPalette = ProxyAddressLookupTable.FindAddress<m_IDirectDrawPalette>(*lplpDDPalette);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::GetPixelFormat(LPDDPIXELFORMAT lpDDPixelFormat)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDPixelFormat || lpDDPixelFormat->dwSize != sizeof(DDPIXELFORMAT))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid parameters. dwSize: " << ((lpDDPixelFormat) ? lpDDPixelFormat->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		// Update surface description
		UpdateSurfaceDesc();

		// Copy pixel format to lpDDPixelFormat
		*lpDDPixelFormat = surfaceDesc2.ddpfPixelFormat;

		return DD_OK;
	}

	return ProxyInterface->GetPixelFormat(lpDDPixelFormat);
}

HRESULT m_IDirectDrawSurfaceX::GetSurfaceDesc(LPDDSURFACEDESC lpDDSurfaceDesc, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (lpDDSurfaceDesc && lpDDSurfaceDesc->dwSize == sizeof(DDSURFACEDESC2))
		{
			return GetSurfaceDesc2((LPDDSURFACEDESC2)lpDDSurfaceDesc, MipMapLevel, DirectXVersion);
		}

		if (!lpDDSurfaceDesc || lpDDSurfaceDesc->dwSize != sizeof(DDSURFACEDESC))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid parameters. dwSize: " << (lpDDSurfaceDesc ? lpDDSurfaceDesc->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		DDSURFACEDESC2 Desc2 = {};
		Desc2.dwSize = sizeof(DDSURFACEDESC2);

		HRESULT hr = GetSurfaceDesc2(&Desc2, MipMapLevel, DirectXVersion);

		// Convert back to LPDDSURFACEDESC
		if (SUCCEEDED(hr))
		{
			ConvertSurfaceDesc(*lpDDSurfaceDesc, Desc2);
		}

		return hr;
	}

	return GetProxyInterfaceV3()->GetSurfaceDesc(lpDDSurfaceDesc);
}

HRESULT m_IDirectDrawSurfaceX::GetSurfaceDesc2(LPDDSURFACEDESC2 lpDDSurfaceDesc2, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSurfaceDesc2 || lpDDSurfaceDesc2->dwSize != sizeof(DDSURFACEDESC2))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid parameters. dwSize: " << (lpDDSurfaceDesc2 ? lpDDSurfaceDesc2->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		// Update surfacedesc
		UpdateSurfaceDesc();

		// Copy surfacedesc to lpDDSurfaceDesc2
		*lpDDSurfaceDesc2 = surfaceDesc2;

		// Handle mipmaps
		if (MipMapLevel && MipMaps.size())
		{
			// Remove a couple of flags
			lpDDSurfaceDesc2->dwFlags &= ~(DDSD_LPSURFACE | DDSD_PITCH | DDSD_LINEARSIZE);
			lpDDSurfaceDesc2->lpSurface = nullptr;
			lpDDSurfaceDesc2->lPitch = 0;

			// Handle new v7 flag
			if (DirectXVersion == 7)
			{
				lpDDSurfaceDesc2->ddsCaps.dwCaps2 |= DDSCAPS2_MIPMAPSUBLEVEL;
			}

			// Handle dummy mipmaps
			if (IsDummyMipMap(MipMapLevel))
			{
				DWORD Level = (MipMapLevel & ~DXW_IS_MIPMAP_DUMMY);

				// Get width and height
				DWORD BitCount = surface.BitCount ? surface.BitCount : GetBitCount(lpDDSurfaceDesc2->ddpfPixelFormat);
				DWORD Width = surface.Width ? surface.Width : GetByteAlignedWidth(lpDDSurfaceDesc2->dwWidth, BitCount);
				DWORD Height = surface.Height ? surface.Height : GetByteAlignedWidth(lpDDSurfaceDesc2->dwHeight, BitCount);
				lpDDSurfaceDesc2->dwWidth = max(1, Width >> Level);
				lpDDSurfaceDesc2->dwHeight = max(1, Height >> Level);

				// Mipmap count
				lpDDSurfaceDesc2->dwMipMapCount = 1;
			}
			// Handle normal mipmaps
			else
			{
				// Check for device interface to ensure correct max MipMap level
				CheckInterface(__FUNCTION__, true, true, false);

				// Get width and height
				DWORD Level = min(MipMaps.size(), MipMapLevel) - 1;
				if ((!MipMaps[Level].dwWidth || !MipMaps[Level].dwHeight) && surface.Texture)
				{
					D3DSURFACE_DESC Desc = {};
					surface.Texture->GetLevelDesc(GetD3d9MipMapLevel(MipMapLevel), &Desc);
					MipMaps[Level].dwWidth = Desc.Width;
					MipMaps[Level].dwHeight = Desc.Height;
				}
				lpDDSurfaceDesc2->dwWidth = MipMaps[Level].dwWidth;
				lpDDSurfaceDesc2->dwHeight = MipMaps[Level].dwHeight;

				// Set pitch
				if (MipMaps[Level].lPitch)
				{
					lpDDSurfaceDesc2->dwFlags |= DDSD_PITCH;
					lpDDSurfaceDesc2->lPitch = MipMaps[Level].lPitch;
				}

				// Mipmap count
				lpDDSurfaceDesc2->dwMipMapCount = MaxMipMapLevel + 1 > MipMapLevel ? MaxMipMapLevel + 1 - MipMapLevel : 1;
			}

			// Set pitch
			if (!(lpDDSurfaceDesc2->dwFlags & DDSD_PITCH))
			{
				DWORD Pitch = ComputePitch(GetDisplayFormat(lpDDSurfaceDesc2->ddpfPixelFormat), lpDDSurfaceDesc2->dwWidth, lpDDSurfaceDesc2->dwHeight);
				if (Pitch)
				{
					lpDDSurfaceDesc2->dwFlags |= DDSD_PITCH;
					lpDDSurfaceDesc2->lPitch = Pitch;
				}
			}
		}
		else if (MipMapLevel)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: MipMap found with no MipMap list!");
		}

		// Set correct flags for textures
		if (ISDXTEX(surface.Format))
		{
			FixTextureFlags(lpDDSurfaceDesc2);
		}

		// Handle managed texture memory type
		if ((lpDDSurfaceDesc2->ddsCaps.dwCaps & DDSCAPS_TEXTURE) && (lpDDSurfaceDesc2->ddsCaps.dwCaps2 & DDSCAPS2_TEXTUREMANAGE))
		{
			lpDDSurfaceDesc2->ddsCaps.dwCaps = (lpDDSurfaceDesc2->ddsCaps.dwCaps & ~(DDSCAPS_LOCALVIDMEM | DDSCAPS_VIDEOMEMORY)) | DDSCAPS_SYSTEMMEMORY;
		}

		// Return
		return DD_OK;
	}

	return ProxyInterface->GetSurfaceDesc(lpDDSurfaceDesc2);
}

HRESULT m_IDirectDrawSurfaceX::Initialize(LPDIRECTDRAW lpDD, LPDDSURFACEDESC lpDDSurfaceDesc)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (lpDDSurfaceDesc && lpDDSurfaceDesc->dwSize != sizeof(DDSURFACEDESC))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Invalid parameters. dwSize: " << ((lpDDSurfaceDesc) ? lpDDSurfaceDesc->dwSize : -1));
		}

		DDSURFACEDESC2 Desc2 = {};
		Desc2.dwSize = sizeof(DDSURFACEDESC2);
		if (lpDDSurfaceDesc)
		{
			ConvertSurfaceDesc(Desc2, *lpDDSurfaceDesc);
		}

		return Initialize2(lpDD, (lpDDSurfaceDesc) ? &Desc2 : nullptr);
	}

	if (lpDD)
	{
		lpDD->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDD);
	}

	return GetProxyInterfaceV3()->Initialize(lpDD, lpDDSurfaceDesc);
}

HRESULT m_IDirectDrawSurfaceX::Initialize2(LPDIRECTDRAW lpDD, LPDDSURFACEDESC2 lpDDSurfaceDesc2)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Because the DirectDrawSurface object is initialized when it is created, this method always returns DDERR_ALREADYINITIALIZED.
		return DDERR_ALREADYINITIALIZED;
	}

	if (lpDD)
	{
		lpDD->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDD);
	}

	return ProxyInterface->Initialize(lpDD, lpDDSurfaceDesc2);
}

HRESULT m_IDirectDrawSurfaceX::IsLost()
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check device interface
		if ((surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_VIDEOMEMORY) || IsD9UsingVideoMemory())
		{
			// Check for device interface
			HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
			if (FAILED(c_hr))
			{
				return c_hr;
			}

			switch (ddrawParent->TestD3D9CooperativeLevel())
			{
			case D3D_OK:
			case DDERR_NOEXCLUSIVEMODE:
				if (IsSurfaceLost)
				{
					return DDERR_SURFACELOST;
				}
				return DD_OK;
			case D3DERR_DEVICELOST:
				MarkSurfaceLost();
				return DD_OK;		// Native DriectDraw returns ok here, until surface is ready to be reset
			case D3DERR_DEVICENOTRESET:
				MarkSurfaceLost();
				if (IsSurfaceLost)
				{
					return DDERR_SURFACELOST;
				}
				[[fallthrough]];
			default:
				return DDERR_WRONGMODE;
			}
		}

		return DD_OK;
	}

	return ProxyInterface->IsLost();
}

void MenuBlitOverlayMarkLockDirty(m_IDirectDrawSurfaceX* pSurf);  // defined with the [BLITQUAD] block below

HRESULT m_IDirectDrawSurfaceX::Lock(LPRECT lpDestRect, LPDDSURFACEDESC lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	// [MENUDIAG]: CPU writes into the frame would explain text invisible to all draw paths.
	if (Config.DdrawOrphanOverlayLift && (IsPrimaryOrBackBuffer() || IsRenderTarget()) && !(dwFlags & DDLOCK_READONLY))
	{
		LOG_LIMIT(20000, "[MENUDIAG] Lock-write surf=" << this <<
			(IsPrimarySurface() ? " PRIMARY" : IsBackBuffer() ? " BACKBUFFER" : " RENDERTARGET") <<
			" rect=" << lpDestRect << " flags=" << Logging::hex(dwFlags));

		// [BLITQUAD] v2: software-rendered content (menu TEXT) lands here -- mark the
		// frame dirty so EndScene captures the emulated backbuffer as an overlay.
		if (Config.DdrawMenuBlitOverlay)
		{
			MenuBlitOverlayMarkLockDirty(this);
		}
	}

	// Game using old DirectX, Convert to LPDDSURFACEDESC2
	if (Config.Dd7to9)
	{
		if (!lpDDSurfaceDesc || lpDDSurfaceDesc->dwSize != sizeof(DDSURFACEDESC))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid parameters. dwSize: " << (lpDDSurfaceDesc ? lpDDSurfaceDesc->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		DDSURFACEDESC2 Desc2 = {};
		Desc2.dwSize = sizeof(DDSURFACEDESC2);

		HRESULT hr = Lock2(lpDestRect, &Desc2, dwFlags, hEvent, MipMapLevel, DirectXVersion);

		// Convert back to LPDDSURFACEDESC
		ConvertSurfaceDesc(*lpDDSurfaceDesc, Desc2);

		return hr;
	}

	return GetProxyInterfaceV3()->Lock(lpDestRect, lpDDSurfaceDesc, dwFlags, hEvent);
}

HRESULT m_IDirectDrawSurfaceX::Lock2(LPRECT lpDestRect, LPDDSURFACEDESC2 lpDDSurfaceDesc2, DWORD dwFlags, HANDLE hEvent, DWORD MipMapLevel, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" Rect = " << lpDestRect <<
		" Desc = " << lpDDSurfaceDesc2 <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Event = " << hEvent <<
		" MipMapLevel = " << MipMapLevel <<
		" Version = " << DirectXVersion;

	// [LOCKW] (2026-06-10): a write-lock on a known Blt-source (staging) surface.
	// Capture the lock rect + the DKII.exe caller now (the writer's identity);
	// flushed with the resulting content hash at SetDirtyFlag.
	if (Config.DdrawRecipeLog && MipMapLevel == 0 && !(dwFlags & DDLOCK_READONLY) &&
		recipeBltSources.count((void*)this))
	{
		auto& pend = lockWPending[(void*)this];
		if (pend.size() < 16)
		{
			LockWPend p;
			p.eip = FindGameCaller();
			p.hasRect = (lpDestRect != nullptr);
			p.rect = lpDestRect ? *lpDestRect : RECT{};
			pend.push_back(p);
		}
	}

	if (Config.Dd7to9)
	{
		// Check surfaceDesc size
		if (lpDDSurfaceDesc2 && lpDDSurfaceDesc2->dwSize == sizeof(DDSURFACEDESC))
		{
			return Lock(lpDestRect, (LPDDSURFACEDESC)lpDDSurfaceDesc2, dwFlags, hEvent, MipMapLevel, DirectXVersion);
		}

		// Check surfaceDesc
		if (!lpDDSurfaceDesc2 || lpDDSurfaceDesc2->dwSize != sizeof(DDSURFACEDESC2))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid parameters. dwSize: " << (lpDDSurfaceDesc2 ? lpDDSurfaceDesc2->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		// If primary surface and palette surface and created via Lock() then mark as created by lock to emulate surface (eg. Diablo, Wizardry 8, Wizards and Warriors)
		if (!IsUsingEmulation() && !IsSurfaceTexture() && surfaceDesc2.dwBackBufferCount == 0 && (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_FLIP) == 0 && ddrawParent &&
			(((surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY) && !IsPrimaryOrBackBuffer() && ddrawParent->IsExclusiveMode() && IsDisplayResolution(surfaceDesc2.dwWidth, surfaceDesc2.dwHeight))) ||
			(ShouldEmulate == SC_NOT_CREATED && IsPrimarySurface() && surfaceDesc2.dwFlags == DDSD_CAPS && ddrawParent->GetDisplayBPP() == 8))
		{
			ShouldEmulate = SC_FORCE_EMULATED;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, false);

		// Prepare surfaceDesc
		GetSurfaceDesc2(lpDDSurfaceDesc2, MipMapLevel, DirectXVersion);
		if (!surface.UsingSurfaceMemory && !IsUsingEmulation())
		{
			lpDDSurfaceDesc2->dwFlags |= DDSD_LPSURFACE | DDSD_PITCH;
			lpDDSurfaceDesc2->lpSurface = dummySurface.data();
			lpDDSurfaceDesc2->lPitch = ComputePitch(surface.Format, lpDDSurfaceDesc2->dwWidth, lpDDSurfaceDesc2->dwHeight);
		}
		if (IsUsingEmulation())
		{
			D3DLOCKED_RECT LockedRect = {};
			if (lpDestRect && SUCCEEDED(LockEmulatedSurface(&LockedRect, lpDestRect)))
			{
				lpDDSurfaceDesc2->dwFlags |= DDSD_LPSURFACE | DDSD_PITCH;
				lpDDSurfaceDesc2->lpSurface = LockedRect.pBits;
				lpDDSurfaceDesc2->lPitch = LockedRect.Pitch;
			}
			else
			{
				lpDDSurfaceDesc2->dwFlags |= DDSD_LPSURFACE | DDSD_PITCH;
				lpDDSurfaceDesc2->lpSurface = surface.emu->pBits;
				lpDDSurfaceDesc2->lPitch = surface.emu->Pitch;
			}
		}
		else if (lpDDSurfaceDesc2->dwFlags & DDSD_LINEARSIZE)
		{
			surfaceDesc2.dwFlags &= ~(DDSD_PITCH | DDSD_LINEARSIZE);
			surfaceDesc2.dwLinearSize = 0;
		}

		// Clear lpSurface
		if (!surface.UsingSurfaceMemory &&
			(!(lpDDSurfaceDesc2->dwFlags & DDSD_LPSURFACE) || !(lpDDSurfaceDesc2->dwFlags & DDSD_PITCH) ||
			!lpDDSurfaceDesc2->lpSurface || !lpDDSurfaceDesc2->lPitch ||
			(DWORD)lpDDSurfaceDesc2->lPitch > lpDDSurfaceDesc2->dwWidth * 4 + 128))
		{
			lpDDSurfaceDesc2->dwFlags &= ~DDSD_LPSURFACE;
			lpDDSurfaceDesc2->lpSurface = nullptr;
		}

		// Return error for CheckInterface after preparing surfaceDesc
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Check for video memory zbuffers
		if ((IsDepthStencil() || (surface.Usage & D3DUSAGE_DEPTHSTENCIL)) && IsD9UsingVideoMemory())
		{
			return DDERR_UNSUPPORTED;
		}

		ScopedCriticalSection ThreadLock(GetCriticalSection());

		LASTLOCK& LastLock = LockedLevel[MipMapLevel];

		// Check for already locked state
		if (!lpDestRect && !LastLock.LockRectList.empty())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: locking surface with NULL rect when surface is already locked!");
			return DDERR_INVALIDRECT;
		}

		// Update rect
		RECT DestRect = {};
		if (!CheckCoordinates(DestRect, lpDestRect, lpDDSurfaceDesc2) || (lpDestRect && (lpDestRect->left < 0 || lpDestRect->top < 0 ||
			lpDestRect->right <= lpDestRect->left || lpDestRect->bottom <= lpDestRect->top ||
			lpDestRect->right > (LONG)lpDDSurfaceDesc2->dwWidth || lpDestRect->bottom > (LONG)lpDDSurfaceDesc2->dwHeight)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
			return DDERR_INVALIDRECT;
		}

		// Handle dummy mipmaps
		if (IsDummyMipMap(MipMapLevel))
		{
			lpDDSurfaceDesc2->dwFlags |= DDSD_LPSURFACE;
			// Add surface size to dummy data address to ensure that each mipmap gets a unique address
			lpDDSurfaceDesc2->lpSurface = dummySurface.data() + (lpDDSurfaceDesc2->dwWidth * lpDDSurfaceDesc2->dwHeight * surface.BitCount);
			if (!(lpDDSurfaceDesc2->dwFlags & DDSD_PITCH))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: no pitch found!");
				return DDERR_GENERIC;
			}
			return DD_OK;
		}

		// Set to indicate that Lock should wait until it can obtain a valid memory pointer before returning.
		const bool LockWait = (((dwFlags & DDLOCK_WAIT) || DirectXVersion == 7) && (dwFlags & DDLOCK_DONOTWAIT) == 0);

		// Convert flags to d3d9
		DWORD Flags = (dwFlags & (D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE)) |
			((dwFlags & D3DLOCK_NOSYSLOCK) ? D3DLOCK_NOSYSLOCK : 0) |
			(!LockWait && !surface.Texture ? D3DLOCK_DONOTWAIT : 0) |
			((dwFlags & DDLOCK_NODIRTYUPDATE) ? D3DLOCK_NO_DIRTY_UPDATE : 0);

		// Check if the scene needs to be presented
		const bool IsSkipScene = (CheckRectforSkipScene(DestRect) || (Flags & D3DLOCK_READONLY));

		// Present before write if needed
		BeginWritePresent(IsSkipScene);

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		// Check if render target should use shadow
		if (MipMapLevel == 0 && (surface.Usage & D3DUSAGE_RENDERTARGET))
		{
			if (surface.IsLockable)
			{
				// Don't use shadow for Lock()
				// Some games write to surface without locking so we don't want to give them a shadow surface or it could make the shadow surface out of sync
				PrepareRenderTarget();
			}
			else if (!IsUsingShadowSurface())
			{
				SetRenderTargetShadow();
			}
		}

		HRESULT hr = DD_OK;

		do {
			// Check if locked from other thread
			if (LockWait)
			{
				// Wait for lock from other thread
				DWORD beginTime = timeGetTime();
				while (IsLockedFromOtherThread(MipMapLevel))
				{
					Utils::BusyWaitYield((DWORD)-1);

					// Break once timeout has passed
					if ((timeGetTime() - beginTime) >= SurfaceWaitTimeoutMS)
					{
						break;
					}
				}
			}

			// Emulated surface
			D3DLOCKED_RECT LockedRect = {};
			if (IsUsingEmulation())
			{
				// Set locked rect
				if (FAILED(LockEmulatedSurface(&LockedRect, &DestRect)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock emulated surface!");
					hr = DDERR_GENERIC;
					break;
				}

				// Read surface from GDI
				if (ShouldReadFromGDI())
				{
					CopyEmulatedSurfaceFromGDI(&DestRect);
				}
			}
			// Lock surface
			else if (surface.Surface || surface.Texture)
			{
				// Lock surface
				HRESULT ret = LockD3d9Surface(&LockedRect, &DestRect, Flags, MipMapLevel);
				if (FAILED(ret))
				{
					if (IsSurfaceLocked(MipMapLevel))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: attempting to lock surface twice!");
						UnLockD3d9Surface(MipMapLevel);
					}
					ret = LockD3d9Surface(&LockedRect, &DestRect, Flags, MipMapLevel);
				}
				if (FAILED(ret))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock surface." << (surface.Surface ? " Is Surface." : " Is Texture.") <<
						" Size: " << lpDDSurfaceDesc2->dwWidth << "x" << lpDDSurfaceDesc2->dwHeight << " Format: " << surface.Format << " Flags: " << Logging::hex(Flags) <<
						" HasData: " << surface.HasData << " Locked: " << IsSurfaceLocked(MipMapLevel) << " DC: " << IsSurfaceInDC(MipMapLevel) << " Blt: " << IsSurfaceBlitting() << " hr: " << (D3DERR)ret);
					hr = (ret == D3DERR_DEVICELOST || IsLost() == DDERR_SURFACELOST) ? DDERR_SURFACELOST :
						(IsSurfaceBusy(MipMapLevel)) ? DDERR_SURFACEBUSY :
						(ret == DDERR_WASSTILLDRAWING || (!LockWait && IsPresentRunning)) ? DDERR_WASSTILLDRAWING : DDERR_GENERIC;
					break;
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
				hr = DDERR_GENERIC;
				break;
			}

			// Check pointer and pitch
			if (!LockedRect.pBits || !LockedRect.Pitch)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface address or pitch!");
				hr = DDERR_GENERIC;
				break;
			}

			// Set thread ID
			if (LastLock.LockedWithID && LastLock.LockedWithID != GetCurrentThreadId())
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: surface locked thread ID set! " << LastLock.LockedWithID);
			}
			LastLock.LockedWithID = GetCurrentThreadId();

			// Store locked rect
			if (lpDestRect)
			{
				RECT lRect = { lpDestRect->left, lpDestRect->top, lpDestRect->right, lpDestRect->bottom };
				LastLock.LockRectList.push_back(lRect);
			}

			// Set surfaceDesc
			lpDDSurfaceDesc2->lpSurface = LockedRect.pBits;
			lpDDSurfaceDesc2->dwFlags |= DDSD_LPSURFACE;

			// Pitch for DXT surfaces in DirectDraw is the full surface byte size
			LockedRect.Pitch =
				(ISDXTEX(surface.Format) || surface.Format == D3DFMT_YV12 || surface.Format == D3DFMT_NV12) ?
				ComputePitch(surface.Format, lpDDSurfaceDesc2->dwWidth, lpDDSurfaceDesc2->dwHeight) :
				LockedRect.Pitch;
			lpDDSurfaceDesc2->lPitch = LockedRect.Pitch;
			lpDDSurfaceDesc2->dwFlags |= DDSD_PITCH;

			// Set surface pitch
			if (MipMapLevel && MipMaps.size())
			{
				DWORD Level = min(MipMaps.size(), MipMapLevel) - 1;
				if (MipMaps[Level].lPitch && MipMaps[Level].lPitch != LockedRect.Pitch)
				{
					LOG_LIMIT(100, __FUNCTION__ << " (" << this << ")" << " Warning: surface pitch does not match locked pitch! Format: " << surface.Format <<
						" Width: " << lpDDSurfaceDesc2->dwWidth << " Pitch: " << MipMaps[Level].lPitch << "->" << LockedRect.Pitch
						<< " MipMapLevel: " << MipMapLevel);
				}
				MipMaps[Level].lPitch = LockedRect.Pitch;
			}
			else
			{
				if ((surfaceDesc2.dwFlags & DDSD_PITCH) && surfaceDesc2.lPitch != LockedRect.Pitch)
				{
					LOG_LIMIT(100, __FUNCTION__ << " (" << this << ")" << " Warning: surface pitch does not match locked pitch! Format: " << surface.Format <<
						" Width: " << surfaceDesc2.dwWidth << " Pitch: " << surfaceDesc2.lPitch << "->" << LockedRect.Pitch <<
						" Default: " << ComputePitch(surface.Format, surface.Width, surface.BitCount) << " BitCount: " << surface.BitCount);
				}
				surfaceDesc2.lPitch = LockedRect.Pitch;
				surfaceDesc2.dwFlags |= DDSD_PITCH;
			}

			// Set correct flags for textures
			if (ISDXTEX(surface.Format))
			{
				FixTextureFlags(lpDDSurfaceDesc2);
			}

			// Emulate lock
			if (((Config.DdrawEmulateLock && !IsUsingEmulation()) || Config.DdrawFixByteAlignment) && !(Flags & D3DLOCK_READONLY) && MipMapLevel == 0)
			{
				LockEmuLock(lpDestRect, lpDDSurfaceDesc2);
			}

			// Backup last rect before removing scanlines
			LastLock.IsLocked = true;
			LastLock.ReadOnly = (Flags & D3DLOCK_READONLY);
			LastLock.IsSkipScene = IsSkipScene;
			LastLock.Rect = DestRect;
			LastLock.LockedRect.pBits = LockedRect.pBits;
			LastLock.LockedRect.Pitch = LockedRect.Pitch;
			LastLock.MipMapLevel = MipMapLevel;

			// Restore scanlines before returing surface memory
			if (Config.DdrawRemoveScanlines && IsPrimaryOrBackBuffer() && !(Flags & D3DLOCK_READONLY) && MipMapLevel == 0)
			{
				RestoreScanlines(LastLock);
			}

		} while (false);

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ")" <<
			" Type = " << surface.Type << " " << surface.Pool <<
			" hr = " << (D3DERR)hr <<
			" Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		// Track Lock operations for atlas detection
		if (Config.DdrawLogTextureAtlas && SUCCEEDED(hr))
		{
			SurfaceLockStats& stats = lockTrackingMap[this];
			stats.lockCount++;
			stats.width = surfaceDesc2.dwWidth;
			stats.height = surfaceDesc2.dwHeight;

			// Check if this is a write lock (not read-only)
			bool isWriteLock = !(dwFlags & DDLOCK_READONLY);
			if (isWriteLock)
			{
				stats.writeLockCount++;
			}
		}

		return hr;
	}

	return ProxyInterface->Lock(lpDestRect, lpDDSurfaceDesc2, dwFlags, hEvent);
}

HRESULT m_IDirectDrawSurfaceX::ReleaseDC(HDC hDC, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" DC = " << hDC;

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Check struct
		if (GetDCLevel.find(MipMapLevel) == GetDCLevel.end())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DC level struct hasn't been created yet: " << MipMapLevel);
			return DDERR_NOTLOCKED;
		}

		if (!GetDCLevel[MipMapLevel])
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not in DC!");
			return DDERR_GENERIC;
		}

		if (GetDCLevel[MipMapLevel] != hDC)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: HDC doesn't match: " << GetDCLevel[MipMapLevel] << " -> " << hDC);
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		HRESULT hr = DD_OK;

		do {
			if (IsUsingEmulation() || DCRequiresEmulation)
			{
				if (!IsUsingEmulation())
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: surface not using emulated DC!");
					break;
				}

				// Restore DC
				UnsetEmulationGameDC();
			}
			else
			{
				// Get surface
				ScopedGetMipMapContext Dest(this, MipMapLevel);
				if (!Dest.GetSurface())
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
					hr = DDERR_GENERIC;
					break;
				}

				// Release device context
				if (FAILED(Dest.GetSurface()->ReleaseDC(hDC)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to release surface DC!");
					hr = IsSurfaceBusy(MipMapLevel) ? DDERR_SURFACEBUSY : IsLost() == DDERR_SURFACELOST ? DDERR_SURFACELOST : DDERR_GENERIC;
					break;
				}
			}

			// Clear DC level
			GetDCLevel[MipMapLevel] = nullptr;

		} while (false);

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		if (SUCCEEDED(hr))
		{
			// Set dirty flag
			SetDirtyFlag(MipMapLevel);

			if (MipMapLevel == 0)
			{
				// Keep surface insync
				EndWriteSyncSurfaces(nullptr);

				// Present surface
				EndWritePresent(nullptr, false);
			}
		}

		return hr;
	}

	return ProxyInterface->ReleaseDC(hDC);
}

HRESULT m_IDirectDrawSurfaceX::Restore()
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// ToDo: A single call to this method will restore a DirectDrawSurface object's associated implicit surfaces (back buffers, and so on). 

		switch (ddrawParent->TestD3D9CooperativeLevel())
		{
		case D3DERR_DEVICENOTRESET:
			if (FAILED(ddrawParent->ResetD9Device()))
			{
				return DDERR_WRONGMODE;
			}
			[[fallthrough]];
		case D3D_OK:
		case DDERR_NOEXCLUSIVEMODE:
			if (FAILED(CheckInterface(__FUNCTION__, true, true, false)))
			{
				return DDERR_WRONGMODE;
			}
			IsSurfaceLost = false;
			return DD_OK;
		case D3DERR_DEVICELOST:
		default:
			return DDERR_WRONGMODE;
		}
	}

	return ProxyInterface->Restore();
}

HRESULT m_IDirectDrawSurfaceX::SetClipper(LPDIRECTDRAWCLIPPER lpDDClipper)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDClipper && !attachedClipper)
		{
			return DDERR_NOCLIPPERATTACHED;
		}

		if (lpDDClipper == attachedClipper)
		{
			return DD_OK;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// If clipper exists increament ref
		if (lpDDClipper)
		{
			if (!ProxyAddressLookupTable.IsValidWrapperAddress((m_IDirectDrawClipper*)lpDDClipper))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find clipper " << lpDDClipper);
				return DDERR_INVALIDPARAMS;
			}

			lpDDClipper->AddRef();
		}

		// Decrement ref count
		if (attachedClipper && ddrawParent->DoesClipperExist(attachedClipper))
		{
			attachedClipper->Release();
		}

		// Set clipper address
		attachedClipper = (m_IDirectDrawClipper*)lpDDClipper;

		return DD_OK;
	}

	if (lpDDClipper)
	{
		lpDDClipper->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDClipper);
	}

	return ProxyInterface->SetClipper(lpDDClipper);
}

HRESULT m_IDirectDrawSurfaceX::SetColorKey(DWORD dwFlags, LPDDCOLORKEY lpDDColorKey)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Get color key index
		DWORD dds = 0;
		switch (dwFlags & ~DDCKEY_COLORSPACE)
		{
		case DDCKEY_DESTBLT:
			dds = DDSD_CKDESTBLT;
			break;
		case DDCKEY_DESTOVERLAY:
			dds = DDSD_CKDESTOVERLAY;
			break;
		case DDCKEY_SRCBLT:
			dds = DDSD_CKSRCBLT;
			break;
		case DDCKEY_SRCOVERLAY:
			dds = DDSD_CKSRCOVERLAY;
			break;
		default:
			return DDERR_INVALIDPARAMS;
		}

		// Check for color space
		if (lpDDColorKey && (dwFlags & DDCKEY_COLORSPACE) && lpDDColorKey->dwColorSpaceLowValue != lpDDColorKey->dwColorSpaceHighValue)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: color space not supported!");
			return DDERR_NOCOLORKEYHW;
		}

		// Set color key
		if (!lpDDColorKey)
		{
			surfaceDesc2.dwFlags &= ~dds;
		}
		else
		{
			// You must add the flag DDCKEY_COLORSPACE, otherwise DirectDraw will collapse the range to one value
			DDCOLORKEY ColorKey = { lpDDColorKey->dwColorSpaceLowValue, lpDDColorKey->dwColorSpaceLowValue };

			// Set color key
			switch (dds)
			{
			case DDSD_CKDESTBLT:
				surfaceDesc2.ddckCKDestBlt = ColorKey;
				break;
			case DDSD_CKDESTOVERLAY:
				surfaceDesc2.ddckCKDestOverlay = ColorKey;
				break;
			case DDSD_CKSRCBLT:
				if (!(surfaceDesc2.dwFlags & dds) || ColorKey.dwColorSpaceLowValue != surfaceDesc2.ddckCKSrcBlt.dwColorSpaceLowValue)
				{
					ShaderColorKey.IsSet = false;
					surface.IsDrawTextureDirty = true;
				}
				surfaceDesc2.ddckCKSrcBlt = ColorKey;
				break;
			case DDSD_CKSRCOVERLAY:
				surfaceDesc2.ddckCKSrcOverlay = ColorKey;
				break;
			}

			// Set color key flag
			surfaceDesc2.dwFlags |= dds;
		}

		// Return
		return DD_OK;
	}

	return ProxyInterface->SetColorKey(dwFlags, lpDDColorKey);
}

HRESULT m_IDirectDrawSurfaceX::SetOverlayPosition(LONG lX, LONG lY)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Not fully Implemented.");

		// Store the new overlay position
		overlayX = lX;
		overlayY = lY;

		return DD_OK;
	}

	return ProxyInterface->SetOverlayPosition(lX, lY);
}

HRESULT m_IDirectDrawSurfaceX::SetPalette(LPDIRECTDRAWPALETTE lpDDPalette)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDPalette && !attachedPalette)
		{
			return DDERR_NOPALETTEATTACHED;
		}

		if (lpDDPalette == attachedPalette)
		{
			return DD_OK;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

		// If palette exists increament ref
		if (lpDDPalette)
		{
			if (!ddrawParent->DoesPaletteExist((m_IDirectDrawPalette*)lpDDPalette))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find palette " << lpDDPalette);
				return DDERR_INVALIDPARAMS;
			}

			lpDDPalette->AddRef();

			// Set primary flag
			if (IsPrimarySurface())
			{
				((m_IDirectDrawPalette*)lpDDPalette)->SetPrimary();
			}
		}

		// Decrement ref count
		if (attachedPalette && ddrawParent->DoesPaletteExist(attachedPalette))
		{
			// Remove primary flag
			if (IsPrimarySurface() && attachedPalette != lpDDPalette)
			{
				attachedPalette->RemovePrimary();
			}

			attachedPalette->Release();
		}

		// Set palette address
		attachedPalette = (m_IDirectDrawPalette*)lpDDPalette;

		// Reset data for new palette
		surface.LastPaletteUSN = 0;
		surface.PaletteEntryArray = nullptr;

		// Set new palette data
		UpdatePaletteData();

		return DD_OK;
	}

	if (lpDDPalette)
	{
		lpDDPalette->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDPalette);
	}

	return ProxyInterface->SetPalette(lpDDPalette);
}

HRESULT m_IDirectDrawSurfaceX::Unlock(LPRECT lpRect, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" Rect = " << lpRect <<
		" MipMapLevel = " << MipMapLevel;

	if (Config.Dd7to9)
	{
		// Handle dummy mipmaps
		if (IsDummyMipMap(MipMapLevel))
		{
			return DD_OK;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		// Check struct
		if (LockedLevel.find(MipMapLevel) == LockedLevel.end())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: locked level struct hasn't been created yet: " << MipMapLevel);
			return DDERR_NOTLOCKED;
		}

		LASTLOCK& LastLock = LockedLevel[MipMapLevel];

		// [BLITQUAD] v3: the menu TEXT is software-rendered through direct d3d9
		// backbuffer locks (no emulation mirror, v2 found). Grab the written bits NOW,
		// while the mapping is still valid, and queue them as an overlay.
		if (Config.DdrawMenuBlitOverlay && !LastLock.ReadOnly && LastLock.LockedRect.pBits &&
			(IsPrimaryOrBackBuffer() || IsRenderTarget()))
		{
			MenuBlitOverlayCaptureLockedBits(LastLock.LockedRect.pBits, LastLock.LockedRect.Pitch, LastLock.Rect);
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		HRESULT hr = DD_OK;

		do {
			// Check rect
			if (!lpRect && LastLock.LockRectList.size() > 1)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: Rect cannot be NULL when locked with a specific rect!");
				hr = DDERR_INVALIDRECT;
				break;
			}

			// Check stored rect
			if (lpRect && LastLock.LockRectList.size() > 1)
			{
				auto it = std::find_if(LastLock.LockRectList.begin(), LastLock.LockRectList.end(),
					[=](auto Rect) -> bool { return (Rect.left == lpRect->left && Rect.top == lpRect->top && Rect.right == lpRect->right && Rect.bottom == lpRect->bottom); });

				if (it != std::end(LastLock.LockRectList))
				{
					LastLock.LockRectList.erase(it);

					// Unlock once all rects have been unlocked
					if (!LastLock.LockRectList.empty())
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: multiple locked rects found: " << LastLock.LockRectList.size());
						hr = DD_OK;
						break;
					}
				}
				else
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: Rect does not match locked rect: " << lpRect);
					hr = DDERR_INVALIDRECT;
					break;
				}
			}

			// Emulate unlock
			if (EmuLock.Locked && MipMapLevel == 0)
			{
				UnlockEmuLock();
			}

			// Remove scanlines before unlocking surface
			if (Config.DdrawRemoveScanlines && IsPrimaryOrBackBuffer() && MipMapLevel == 0)
			{
				RemoveScanlines(LastLock);
			}

			// Emulated surface
			if (IsUsingEmulation())
			{
				// No need to unlock emulated surface
			}
			// Lock surface
			else if (surface.Surface || surface.Texture)
			{
				HRESULT ret = UnLockD3d9Surface(MipMapLevel);
				if (FAILED(ret))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to unlock surface texture." <<
						" Rect = " << lpRect <<
						" MipMapLevel = " << MipMapLevel <<
						" hr = " << (DDERR)ret);
					hr = (ret == DDERR_WASSTILLDRAWING) ? DDERR_WASSTILLDRAWING :
						IsLost() == DDERR_SURFACELOST ? DDERR_SURFACELOST : DDERR_GENERIC;
					break;
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
				hr = DDERR_GENERIC;
				break;
			}

			// Clear memory pointer
			LastLock.LockedRect.pBits = nullptr;

			// Clear vector
			LastLock.LockRectList.clear();

			// Reset locked flag
			LastLock.IsLocked = false;

			// Reset locked thread ID
			if (!IsSurfaceBlitting() && !IsSurfaceLocked(MipMapLevel))
			{
				LastLock.LockedWithID = 0;
			}

		} while (false);

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		// If surface was changed
		if (SUCCEEDED(hr))
		{
			if (!LastLock.ReadOnly)
			{
				// Set dirty flag (with the specific locked rect for per-rect cache invalidation)
				SetDirtyFlag(LastLock.MipMapLevel, &LastLock.Rect);

				if (LastLock.MipMapLevel == 0)
				{
					// Keep surface insync
					EndWriteSyncSurfaces(&LastLock.Rect);

					// Present surface
					EndWritePresent(&LastLock.Rect, LastLock.IsSkipScene);
				}
			}
		}

		return hr;
	}

	return ProxyInterface->Unlock(lpRect);
}

HRESULT m_IDirectDrawSurfaceX::UpdateOverlay(LPRECT lpSrcRect, LPDIRECTDRAWSURFACE7 lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for overlay flag
		if (!(surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_OVERLAY))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Not an overlay surface!");
			return DDERR_NOTAOVERLAYSURFACE;
		}

		// Turns off this overlay.
		if (dwFlags & DDOVER_HIDE)
		{
			SurfaceOverlay.OverlayEnabled = false;
			return DD_OK;
		}

		// Check for required DDOVERLAYFX structure
		bool RequiresFxStruct = (dwFlags & (DDOVER_DDFX | DDOVER_ALPHADESTCONSTOVERRIDE | DDOVER_ALPHADESTSURFACEOVERRIDE | DDOVER_ALPHAEDGEBLEND | DDOVER_ALPHASRCCONSTOVERRIDE |
			DDOVER_ALPHASRCSURFACEOVERRIDE | DDOVER_ARGBSCALEFACTORS | DDOVER_KEYDESTOVERRIDE | DDOVER_KEYSRCOVERRIDE));
		if (RequiresFxStruct && !lpDDOverlayFx)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DDOVERLAYFX structure not found!");
			return DDERR_INVALIDPARAMS;
		}

		// Check for DDOVERLAYFX structure size
		if (RequiresFxStruct && lpDDOverlayFx->dwSize != sizeof(DDOVERLAYFX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: DDOVERLAYFX structure is not initialized to the correct size: " << lpDDOverlayFx->dwSize);
			return DDERR_INVALIDPARAMS;
		}

		// Cehck for auto flip flag
		if (dwFlags & DDOVER_AUTOFLIP)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: overlay flip not implemented!");
			return DDERR_UNSUPPORTED;
		}

		// Check for alpha flags
		if (dwFlags & (DDOVER_ALPHADEST | DDOVER_ALPHADESTCONSTOVERRIDE | DDOVER_ALPHADESTNEG | DDOVER_ALPHADESTSURFACEOVERRIDE |
			DDOVER_ALPHAEDGEBLEND | DDOVER_ALPHASRC | DDOVER_ALPHASRCCONSTOVERRIDE | DDOVER_ALPHASRCNEG | DDOVER_ALPHASRCSURFACEOVERRIDE))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: overlay alpha not implemented!");
			return DDERR_NOALPHAHW;
		}

		// Check scaling flags
		if (dwFlags & (DDOVER_ARGBSCALEFACTORS | DDOVER_DEGRADEARGBSCALING))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: overlay scale factors not implemented!");
		}

		// Check BOB flags
		if (dwFlags & (DDOVER_BOB | DDOVER_BOBHARDWARE | DDOVER_OVERRIDEBOBWEAVE))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: overlay BOB not implemented!");
		}

		// Check interleave flags
		if (dwFlags & (DDOVER_INTERLEAVED))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: overlay interleave not implemented!");
		}

		// Handle refresh flags
		if (dwFlags & (DDOVER_REFRESHALL | DDOVER_REFRESHDIRTYRECTS))
		{
			// Just refresh whole surface
			return PresentOverlay(nullptr);
		}

		// Check dirty flag
		if (dwFlags & DDOVER_ADDDIRTYRECT)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: overlay dirty rect not implemented!");
			return DD_OK;	// Just return ok for now, nothing else may be needed here
		}

		// Get WrapperX
		if (!lpDDDestSurface)
		{
			return DDERR_INVALIDPARAMS;
		}
		m_IDirectDrawSurfaceX* lpDDDestSurfaceX = nullptr;
		lpDDDestSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDDestSurfaceX);
		if (!lpDDDestSurfaceX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surfaceX!");
			return DDERR_INVALIDPARAMS;
		}
		if (lpDDDestSurfaceX == this)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: cannot overlay surface onto itself!");
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		HRESULT s_hr = lpDDDestSurfaceX->CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr) || FAILED(s_hr))
		{
			return (c_hr == DDERR_SURFACELOST || s_hr == DDERR_SURFACELOST) ? DDERR_SURFACELOST : FAILED(c_hr) ? c_hr : s_hr;
		}

		// Check rect
		if ((lpSrcRect && !CheckCoordinates(lpSrcRect)) ||
			(lpDestRect && !lpDDDestSurfaceX->CheckCoordinates(lpDestRect)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpSrcRect << " -> " << lpDestRect);
			return DDERR_INVALIDRECT;
		}

		// Add entry to overlay vector
		SURFACEOVERLAY Overlay;
		Overlay.OverlayEnabled = (SurfaceOverlay.OverlayEnabled || (dwFlags & DDOVER_SHOW));
		if (lpSrcRect)
		{
			Overlay.isSrcRectNull = false;
			Overlay.SrcRect = *lpSrcRect;
		}
		Overlay.lpDDDestSurface = lpDDDestSurface;
		Overlay.lpDDDestSurfaceX = lpDDDestSurfaceX;
		if (lpDestRect)
		{
			Overlay.isDestRectNull = false;
			Overlay.DestRect = *lpDestRect;
		}
		Overlay.DDOverlayFxFlags = dwFlags;
		Overlay.DDBltFx.dwSize = sizeof(DDBLTFX);
		if (lpDDOverlayFx)
		{
			Overlay.DDOverlayFx = *lpDDOverlayFx;

			// Color keying
			if (dwFlags & DDOVER_KEYDESTOVERRIDE)
			{
				Overlay.DDBltFxFlags |= (DDBLT_DDFX | DDBLT_KEYDESTOVERRIDE);
				Overlay.DDBltFx.ddckDestColorkey = lpDDOverlayFx->dckDestColorkey;
			}
			else if (dwFlags & DDOVER_KEYSRCOVERRIDE)
			{
				Overlay.DDBltFxFlags |= (DDBLT_DDFX | DDBLT_KEYSRCOVERRIDE);
				Overlay.DDBltFx.ddckSrcColorkey = lpDDOverlayFx->dckSrcColorkey;
			}
			// DDOverlayFx flags
			if (dwFlags & DDOVER_DDFX)
			{
				Overlay.DDBltFxFlags |= DDBLT_DDFX;
				Overlay.DDBltFx.dwDDFX = (lpDDOverlayFx->dwFlags & (DDBLTFX_ARITHSTRETCHY | DDBLTFX_MIRRORLEFTRIGHT | DDBLTFX_MIRRORUPDOWN));
			}
		}

		// Update overlay
		SurfaceOverlay = Overlay;

		// Return
		return DD_OK;
	}

	if (lpDDDestSurface)
	{
		lpDDDestSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDDestSurface);
	}

	return ProxyInterface->UpdateOverlay(lpSrcRect, lpDDDestSurface, lpDestRect, dwFlags, lpDDOverlayFx);
}

HRESULT m_IDirectDrawSurfaceX::UpdateOverlayDisplay(DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	return ProxyInterface->UpdateOverlayDisplay(dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::UpdateOverlayZOrder(DWORD dwFlags, LPDIRECTDRAWSURFACE7 lpDDSReference)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	if (lpDDSReference)
	{
		lpDDSReference->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDDSReference);
	}

	return ProxyInterface->UpdateOverlayZOrder(dwFlags, lpDDSReference);
}

// ******************************
// IDirectDrawSurface v2 functions
// ******************************

HRESULT m_IDirectDrawSurfaceX::GetDDInterface(LPVOID FAR * lplpDD, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lplpDD)
	{
		return DDERR_INVALIDPARAMS;
	}
	*lplpDD = nullptr;

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, false, false, false);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		*lplpDD = ddrawParent->GetWrapperInterfaceX(DirectXVersion);

		ddrawParent->AddRef(DirectXVersion);

		return DD_OK;
	}

	LPVOID NewDD = nullptr;
	HRESULT hr = ProxyInterface->GetDDInterface(&NewDD);

	if (SUCCEEDED(hr))
	{
		// Calling the GetDDInterface method from any surface created under DirectDrawEx will return a pointer to the 
		// IUnknown interface instead of a pointer to an IDirectDraw interface. Applications must use the
		// IUnknown::QueryInterface method to retrieve the IDirectDraw, IDirectDraw2, or IDirectDraw3 interfaces.
		IID tmpID = (DirectXVersion == 1) ? IID_IDirectDraw :
			(DirectXVersion == 2) ? IID_IDirectDraw2 :
			(DirectXVersion == 3) ? IID_IDirectDraw3 :
			(DirectXVersion == 4) ? IID_IDirectDraw4 :
			(DirectXVersion == 7) ? IID_IDirectDraw7 : IID_IDirectDraw7;

		hr = ((IUnknown*)NewDD)->QueryInterface(tmpID, lplpDD);
		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get DirectDraw interface: " << (DDERR)hr);
			return hr;
		}

		((IUnknown*)NewDD)->Release();

		*lplpDD = ProxyAddressLookupTable.FindAddress<m_IDirectDraw7>(*lplpDD, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::PageLock(DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// This method was not implemented in the IDirectDraw interface.
		return DD_OK;
	}

	return ProxyInterface->PageLock(dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::PageUnlock(DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// This method was not implemented in the IDirectDraw interface.
		return DD_OK;
	}

	return ProxyInterface->PageUnlock(dwFlags);
}

// ******************************
// IDirectDrawSurface v3 functions
// ******************************

HRESULT m_IDirectDrawSurfaceX::SetSurfaceDesc(LPDDSURFACEDESC lpDDSurfaceDesc, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSurfaceDesc)
		{
			return DDERR_INVALIDPARAMS;
		}

		DDSURFACEDESC2 Desc2 = {};
		Desc2.dwSize = sizeof(DDSURFACEDESC2);
		ConvertSurfaceDesc(Desc2, *lpDDSurfaceDesc);

		return SetSurfaceDesc2(&Desc2, dwFlags);
	}

	return GetProxyInterfaceV3()->SetSurfaceDesc(lpDDSurfaceDesc, dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::SetSurfaceDesc2(LPDDSURFACEDESC2 lpDDSurfaceDesc2, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDDSurfaceDesc2)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check flags
		DWORD SurfaceFlags = lpDDSurfaceDesc2->dwFlags;

		bool RecreateDevice = false;

		// Handle lpSurface flag
		if ((SurfaceFlags & DDSD_LPSURFACE) && lpDDSurfaceDesc2->lpSurface)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: lpSurface not fully Implemented.");

			SurfaceFlags &= ~DDSD_LPSURFACE;
			surfaceDesc2.dwFlags |= DDSD_LPSURFACE;
			surfaceDesc2.lpSurface = lpDDSurfaceDesc2->lpSurface;
			surface.UsingSurfaceMemory = true;
			if (surfaceDesc2.dwMipMapCount > 1)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: lpSurface not supported with MipMaps!");
			}
			if (surface.Surface || surface.Texture)
			{
				RecreateDevice = true;
			}
		}

		// Handle width, height and pitch flags
		if (SurfaceFlags & (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PITCH))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: surfaceDesc flags being updated: " << Logging::hex(SurfaceFlags));
			bool Flag = false;
			if ((SurfaceFlags & DDSD_WIDTH) && lpDDSurfaceDesc2->dwWidth)
			{
				Flag = Flag || (surfaceDesc2.dwWidth != lpDDSurfaceDesc2->dwWidth);
				SurfaceFlags &= ~DDSD_WIDTH;
				ResetDisplayFlags &= ~DDSD_WIDTH;
				surfaceDesc2.dwFlags |= DDSD_WIDTH;
				surfaceDesc2.dwWidth = lpDDSurfaceDesc2->dwWidth;
			}
			if ((SurfaceFlags & DDSD_HEIGHT) && lpDDSurfaceDesc2->dwHeight)
			{
				Flag = Flag || (surfaceDesc2.dwHeight != lpDDSurfaceDesc2->dwHeight);
				SurfaceFlags &= ~DDSD_HEIGHT;
				ResetDisplayFlags &= ~DDSD_HEIGHT;
				surfaceDesc2.dwFlags |= DDSD_HEIGHT;
				surfaceDesc2.dwHeight = lpDDSurfaceDesc2->dwHeight;
			}
			if (SurfaceFlags & DDSD_PITCH)
			{
				SurfaceFlags &= ~DDSD_PITCH;
			}
			if (Flag && (surface.Surface || surface.Texture))
			{
				RecreateDevice = true;
			}
		}

		// Recreate device
		if (RecreateDevice)
		{
			CreateD9Surface();
		}

		// Check for unhandled flags
		if (SurfaceFlags)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: flags not implemented " << Logging::hex(SurfaceFlags));
			return DDERR_UNSUPPORTED;
		}

		return DD_OK;
	}

	return ProxyInterface->SetSurfaceDesc(lpDDSurfaceDesc2, dwFlags);
}

// ******************************
// IDirectDrawSurface v4 functions
// ******************************

HRESULT m_IDirectDrawSurfaceX::SetPrivateData(REFGUID guidTag, LPVOID lpData, DWORD cbSize, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		LOG_LIMIT(100, __FUNCTION__ << " Warning: private data may not be preserved!");

		if (surface.Surface)
		{
			return surface.Surface->SetPrivateData(guidTag, lpData, cbSize, dwFlags);
		}
		else if (surface.Texture)
		{
			return surface.Texture->SetPrivateData(guidTag, lpData, cbSize, dwFlags);
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
			return DDERR_GENERIC;
		}
	}

	return ProxyInterface->SetPrivateData(guidTag, lpData, cbSize, dwFlags);
}

HRESULT m_IDirectDrawSurfaceX::GetPrivateData(REFGUID guidTag, LPVOID lpBuffer, LPDWORD lpcbBufferSize)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		if (surface.Surface)
		{
			return surface.Surface->GetPrivateData(guidTag, lpBuffer, lpcbBufferSize);
		}
		else if (surface.Texture)
		{
			return surface.Texture->GetPrivateData(guidTag, lpBuffer, lpcbBufferSize);
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
			return DDERR_GENERIC;
		}
	}

	return ProxyInterface->GetPrivateData(guidTag, lpBuffer, lpcbBufferSize);
}

HRESULT m_IDirectDrawSurfaceX::FreePrivateData(REFGUID guidTag)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		if (surface.Surface)
		{
			return surface.Surface->FreePrivateData(guidTag);
		}
		else if (surface.Texture)
		{
			return surface.Texture->FreePrivateData(guidTag);
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find surface!");
			return DDERR_GENERIC;
		}
	}

	return ProxyInterface->FreePrivateData(guidTag);
}

HRESULT m_IDirectDrawSurfaceX::GetUniquenessValue(LPDWORD lpValue, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpValue)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (IsSurfaceBusy() || (MipMapLevel && MipMapLevel > MipMaps.size()))
		{
			// The only defined uniqueness value is 0, which indicates that the surface is likely to be changing beyond the control of DirectDraw.
			*lpValue = 0;
		}
		else
		{
			if (MipMapLevel == 0)
			{
				*lpValue = UniquenessValue;
			}
			else
			{
				*lpValue = MipMaps[MipMapLevel - 1].UniquenessValue;
			}
		}
		return DD_OK;
	}

	return ProxyInterface->GetUniquenessValue(lpValue);
}

HRESULT m_IDirectDrawSurfaceX::ChangeUniquenessValue(DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Manually updates the uniqueness value for this surface.
		if (MipMapLevel == 0)
		{
			UniquenessValue++;
		}
		return DD_OK;
	}

	return ProxyInterface->ChangeUniquenessValue();
}

// ******************************
// IDirect3DTexture v7 functions moved here
// ******************************

HRESULT m_IDirectDrawSurfaceX::SetPriority(DWORD dwPriority)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Applications can call this method only for managed textures (those surfaces that were created with the DDSCAPS2_TEXTUREMANAGE flag).
		if ((surfaceDesc2.ddsCaps.dwCaps2 & (DDSCAPS2_TEXTUREMANAGE | DDSCAPS2_D3DTEXTUREMANAGE)) == 0)
		{
			return DDERR_INVALIDOBJECT;
		}

		Priority = dwPriority;

		return DD_OK;
	}

	return ProxyInterface->SetPriority(dwPriority);
}

HRESULT m_IDirectDrawSurfaceX::GetPriority(LPDWORD lpdwPriority)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwPriority)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Applications can call this method only for managed textures (those surfaces that were created with the DDSCAPS2_TEXTUREMANAGE flag).
		if ((surfaceDesc2.ddsCaps.dwCaps2 & (DDSCAPS2_TEXTUREMANAGE | DDSCAPS2_D3DTEXTUREMANAGE)) == 0)
		{
			return DDERR_INVALIDOBJECT;
		}

		*lpdwPriority = Priority;

		return DD_OK;
	}

	return ProxyInterface->GetPriority(lpdwPriority);
}

HRESULT m_IDirectDrawSurfaceX::SetLOD(DWORD dwMaxLOD)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Applications can call this method only for managed textures (those surfaces that were created with the DDSCAPS2_TEXTUREMANAGE flag).
		if ((surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_TEXTUREMANAGE) == 0)
		{
			return DDERR_INVALIDOBJECT;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		if (surface.Texture)
		{
			surface.Texture->SetLOD(dwMaxLOD);
		}

		return DD_OK;
	}

	return ProxyInterface->SetLOD(dwMaxLOD);
}

HRESULT m_IDirectDrawSurfaceX::GetLOD(LPDWORD lpdwMaxLOD)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwMaxLOD)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Applications can call this method only for managed textures (those surfaces that were created with the DDSCAPS2_TEXTUREMANAGE flag).
		if ((surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_TEXTUREMANAGE) == 0)
		{
			return DDERR_INVALIDOBJECT;
		}

		// Check for device interface
		HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
		if (FAILED(c_hr))
		{
			return c_hr;
		}

		*lpdwMaxLOD = 0;
		if (surface.Texture)
		{
			*lpdwMaxLOD = surface.Texture->GetLOD();
		}

		return DD_OK;
	}

	return ProxyInterface->GetLOD(lpdwMaxLOD);
}

// ******************************
// Helper functions
// ******************************

void m_IDirectDrawSurfaceX::InitInterface(DWORD DirectXVersion)
{
	if (ddrawParent)
	{
		ddrawParent->AddSurface(this);
	}

	if (Config.Dd7to9)
	{
		AddRef(DirectXVersion);

		if (!InitializeCriticalSectionAndSpinCount(&ddscs, 4000))
		{
			InitializeCriticalSection(&ddscs);
		}

		if (ddrawParent)
		{
			d3d9Device = ddrawParent->GetDirectD9Device();
		}

		// Set Uniqueness Value
		UniquenessValue = 1;

		// Update surface description and create backbuffers
		InitSurfaceDesc(DirectXVersion);

		// Surface-pool tracking (Phase A): count this as a live texture surface
		// and record which DKII.exe call site requested it.
		if (Config.DdrawLogTextureAtlas && IsSurfaceTexture())
		{
			trackedTextureSurfaces[this] = 1;
			LONG live = ++poolStats.liveTextureSurfaces;
			if (live > poolStats.peakTextureSurfaces)
			{
				poolStats.peakTextureSurfaces = live;
			}
			poolStats.createdThisWindow++;
			DWORD caller = FindGameCaller();
			if (caller)
			{
				createCallerHist[caller]++;
			}
		}
	}
}

void m_IDirectDrawSurfaceX::ReleaseInterface()
{
	// Surface-pool tracking (Phase A): a tracked texture surface is being destroyed.
	if (Config.DdrawLogTextureAtlas)
	{
		auto it = trackedTextureSurfaces.find(this);
		if (it != trackedTextureSurfaces.end())
		{
			trackedTextureSurfaces.erase(it);
			poolStats.liveTextureSurfaces--;
			poolStats.releasedThisWindow++;
		}
	}

	{
		ScopedCriticalSection ThreadLock(GetCriticalSection(), Config.Dd7to9);

		if (Config.Exiting)
		{
			return;
		}

		// Don't delete wrapper interface
		SaveInterfaceAddress(WrapperInterface);
		SaveInterfaceAddress(WrapperInterface2);
		SaveInterfaceAddress(WrapperInterface3);
		SaveInterfaceAddress(WrapperInterface4);
		SaveInterfaceAddress(WrapperInterface7);

		// Clean up mipmaps
		if (!MipMaps.empty())
		{
			for (auto& entry : MipMaps)
			{
				if (entry.Addr) entry.Addr->DeleteMe();
				if (entry.Addr2) entry.Addr2->DeleteMe();
				if (entry.Addr3) entry.Addr3->DeleteMe();
				if (entry.Addr4) entry.Addr4->DeleteMe();
				if (entry.Addr7) entry.Addr7->DeleteMe();
			}
		}

		ReleaseDirectDrawResources();

		if (Config.Dd7to9)
		{
			ReleaseD9Surface(false, false);
		}
	}
	if (Config.Dd7to9)
	{
		// Delete critical section last
		DeleteCriticalSection(&ddscs);
	}
}

HRESULT m_IDirectDrawSurfaceX::CheckInterface(char* FunctionName, bool CheckD3DDevice, bool CheckD3DSurface, bool CheckLostSurface)
{
	// Check ddrawParent device
	if (!ddrawParent)
	{
		m_IDirectDrawX* pInterface = m_IDirectDrawX::GetDirectDrawInterface();
		if (pInterface)
		{
			LOG_LIMIT(100, FunctionName << " Error: no ddraw parent!");
			return DDERR_INVALIDOBJECT;
		}

		SetDdrawParent(pInterface);
		ddrawParent->AddSurface(this);
	}

	// Check d3d9 device
	if (CheckD3DDevice)
	{
		if (!ddrawParent->CheckD9Device(FunctionName) || !d3d9Device || !*d3d9Device)
		{
			LOG_LIMIT(100, FunctionName << " Error: d3d9 device not setup!");
			return DDERR_INVALIDOBJECT;
		}
		if (ShouldPresentToWindow(true))
		{
			HWND CurrentClipperHWnd = ddrawParent->GetClipperHWnd();

			HWND hWnd = nullptr;
			if (attachedClipper)
			{
				attachedClipper->GetHWnd(&hWnd);
				if (IsWindow(hWnd) && hWnd != CurrentClipperHWnd)
				{
					ddrawParent->SetClipperHWnd(hWnd);
				}
			}
			if (!IsWindow(hWnd) && (!IsWindow(CurrentClipperHWnd) || !Utils::IsMainWindow(CurrentClipperHWnd)))
			{
				hWnd = Utils::GetMainWindowForProcess(GetCurrentProcessId());
				if (hWnd != CurrentClipperHWnd)
				{
					ddrawParent->SetClipperHWnd(hWnd);
				}
			}
		}
	}

	// Check if device is lost
	if (CheckLostSurface && CanSurfaceBeLost())
	{
		HRESULT hr = ddrawParent->TestD3D9CooperativeLevel();
		switch (hr)
		{
		case DD_OK:
		case DDERR_NOEXCLUSIVEMODE:
			break;
		case D3DERR_DEVICENOTRESET:
			if (SUCCEEDED(ddrawParent->ResetD9Device()))
			{
				break;
			}
			[[fallthrough]];
		case D3DERR_DEVICELOST:
			return DDERR_SURFACELOST;
		default:
			LOG_LIMIT(100, FunctionName << " Error: TestCooperativeLevel = " << (D3DERR)hr);
			return DDERR_WRONGMODE;
		}

		if (IsSurfaceLost && !LostDeviceBackup.empty())
		{
			LOG_LIMIT(100, FunctionName << " Warning: surface is lost and there is no backup for it!");
			return DDERR_SURFACELOST;
		}
	}

	// Check surface
	if (CheckD3DSurface)
	{
		// Check if using windowed mode
		bool LastWindowedMode = surface.IsUsingWindowedMode;
		surface.IsUsingWindowedMode = !ddrawParent->IsExclusiveMode();

		// Check if using Direct3D
		bool LastUsing3D = Using3D;
		Using3D = ddrawParent->IsUsing3D();

		// Remove emulated surface if not needed
		if (IsUsingEmulation() && !CanSurfaceUseEmulation() && !IsSurfaceBusy())
		{
			ReleaseDCSurface();
		}

		// Clear Using 3D if not needed
		if (!Using3D && LastUsing3D)
		{
			ClearUsing3DFlag();
		}

		// Make sure surface exists, if not then create it
		if ((!surface.Surface && !surface.Texture) ||											// Surface not created yet
			(attached3DTexture && surface.Pool != D3DPOOL_MANAGED) ||							// Surface changed to be a texture but not using the correct memory pool
			(IsPrimaryOrBackBuffer() && LastWindowedMode != surface.IsUsingWindowedMode) ||		// Primary surface and window mode changed
			(PrimaryDisplayTexture && !ShouldPresentToWindow(false)))							// Needs to present to a window but display texture not setup
		{
			if (FAILED(CreateD9Surface()))
			{
				LOG_LIMIT(100, FunctionName << " Error: d3d9 surface texture not setup!");
				return DDERR_WRONGMODE;
			}
		}

		// Check auxiliary surfaces
		if ((RecreateAuxiliarySurfaces || surface.RecreateAuxiliarySurfaces) && FAILED(CreateD9AuxiliarySurfaces()))
		{
			return DDERR_WRONGMODE;
		}
	}

	return DD_OK;
}

void* m_IDirectDrawSurfaceX::GetWrapperInterfaceX(DWORD DirectXVersion)
{
	switch (DirectXVersion)
	{
	case 0:
		if (WrapperInterface7) return WrapperInterface7;
		if (WrapperInterface4) return WrapperInterface4;
		if (WrapperInterface3) return WrapperInterface3;
		if (WrapperInterface2) return WrapperInterface2;
		if (WrapperInterface) return WrapperInterface;
		break;
	case 1:
		return GetInterfaceAddress(WrapperInterface, (LPDIRECTDRAWSURFACE)ProxyInterface, this);
	case 2:
		return GetInterfaceAddress(WrapperInterface2, (LPDIRECTDRAWSURFACE2)ProxyInterface, this);
	case 3:
		return GetInterfaceAddress(WrapperInterface3, (LPDIRECTDRAWSURFACE3)ProxyInterface, this);
	case 4:
		return GetInterfaceAddress(WrapperInterface4, (LPDIRECTDRAWSURFACE4)ProxyInterface, this);
	case 7:
		return GetInterfaceAddress(WrapperInterface7, (LPDIRECTDRAWSURFACE7)ProxyInterface, this);
	}
	LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
	return nullptr;
}

void m_IDirectDrawSurfaceX::SetDdrawParent(m_IDirectDrawX* ddraw)
{
	if (!ddraw)
	{
		return;
	}

	if (ddrawParent && ddrawParent != ddraw)
	{
		Logging::Log() << __FUNCTION__ << " Warning: ddrawParent interface has already been set!";
	}

	ddrawParent = ddraw;

	d3d9Device = ddrawParent->GetDirectD9Device();
}

void m_IDirectDrawSurfaceX::ClearDdraw()
{
	ddrawParent = nullptr;
	d3d9Device = nullptr;
}

void m_IDirectDrawSurfaceX::ReleaseDirectDrawResources()
{
	if (attachedClipper)
	{
		attachedClipper->Release();
		attachedClipper = nullptr;
	}

	if (attachedPalette)
	{
		attachedPalette->Release();
		attachedPalette = nullptr;
	}

	if (attached3DTexture)
	{
		attached3DTexture->DeleteMe();
		attached3DTexture = nullptr;
	}

	if (attached3DDevice)
	{
		attached3DDevice->DeleteMe();
		attached3DDevice = nullptr;
	}

	if (ddrawParent)
	{
		ddrawParent->ClearSurface(this);
	}

	while (!AttachedSurfaceMap.empty())
	{
		auto it = AttachedSurfaceMap.begin();
		DWORD RefCount = it->second.RefCount;
		DWORD DxVersion = it->second.DxVersion;
		m_IDirectDrawSurfaceX* lpSurfaceX = it->second.pSurface;

		AttachedSurfaceMap.erase(it);	// Erase from list before releasing

		if (RefCount == 1)
		{
			lpSurfaceX->Release(DxVersion);
		}
	}
}

LPDIRECT3DSURFACE9 m_IDirectDrawSurfaceX::GetD3d9Surface()
{
	// Check for device interface
	if (FAILED(CheckInterface(__FUNCTION__, true, true, true)))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface not setup!");
		return nullptr;
	}

	return Get3DSurface();
}

LPDIRECT3DSURFACE9 m_IDirectDrawSurfaceX::Get3DSurface()
{
	if (surface.Surface)
	{
		return surface.Surface;
	}
	else if (surface.Texture)
	{
		if (surface.Context || SUCCEEDED(surface.Texture->GetSurfaceLevel(0, &surface.Context)))
		{
			return surface.Context;
		}
	}
	return nullptr;
}

LPDIRECT3DSURFACE9 m_IDirectDrawSurfaceX::Get3DMipMapSurface(DWORD MipMapLevel)
{
	if (IsUsingShadowSurface())
	{
		return surface.Shadow;
	}
	else if (MipMapLevel == 0 || surface.Type != D3DTYPE_TEXTURE)
	{
		return Get3DSurface();
	}
	else if (surface.Texture)
	{
		LPDIRECT3DSURFACE9 pSurfaceD9 = nullptr;
		surface.Texture->GetSurfaceLevel(GetD3d9MipMapLevel(MipMapLevel), &pSurfaceD9);
		return pSurfaceD9;
	}
	return nullptr;
}

void m_IDirectDrawSurfaceX::Release3DMipMapSurface(LPDIRECT3DSURFACE9 pSurfaceD9, DWORD MipMapLevel)
{
	if (pSurfaceD9 && MipMapLevel != 0 && surface.Type == D3DTYPE_TEXTURE && !IsUsingShadowSurface())
	{
		pSurfaceD9->Release();
	}
}

LPDIRECT3DTEXTURE9 m_IDirectDrawSurfaceX::GetD3d9DrawTexture()
{
	// Check if texture already exists
	if (surface.DrawTexture)
	{
		if (surface.IsDrawTextureDirty && FAILED(CopyToDrawTexture(nullptr)))
		{
			return nullptr;
		}
		return surface.DrawTexture;
	}

	// Create texture
	if (surface.Texture)
	{
		DWORD Level = IsMipMapAutogen() ? 0 : MaxMipMapLevel + 1;
		if (FAILED((*d3d9Device)->CreateTexture(surface.Width, surface.Height, Level, surface.Usage, D3DFMT_A8R8G8B8, surface.Pool, &surface.DrawTexture, nullptr)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create surface texture. Size: " << surface.Width << "x" << surface.Height <<
				" Format: " << surface.Format << " dwCaps: " << surfaceDesc2.ddsCaps);
			return nullptr;
		}
		if (FAILED(CopyToDrawTexture(nullptr)))
		{
			return nullptr;
		}
		return surface.DrawTexture;
	}
	return nullptr;
}

LPDIRECT3DTEXTURE9 m_IDirectDrawSurfaceX::GetD3d9Texture(bool InterfaceCheck)
{
	if (InterfaceCheck)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true, true, false)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: texture not setup!");
			return nullptr;
		}

		// Check texture pool
		if ((surface.Pool == D3DPOOL_SYSTEMMEM || surface.Pool == D3DPOOL_SCRATCH) && IsSurfaceTexture())
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: texture pool does not support Driect3D: " << surface.Format << " Pool: " << surface.Pool <<
				" Caps: " << surfaceDesc2.ddsCaps << " Attached: " << attached3DTexture);
			return nullptr;
		}
	}

	return Get3DTexture();
}

LPDIRECT3DTEXTURE9 m_IDirectDrawSurfaceX::Get3DTexture()
{
	// Primary display texture
	if (PrimaryDisplayTexture && ShouldPresentToWindow(true))
	{
		if (IsPalette() && surface.IsUsingWindowedMode && (surface.DisplayTexture || !primary.PaletteTexture))
		{
			Logging::Log() << __FUNCTION__ << " Error: using non-shader palette surface on window mode not supported!";
		}
		return PrimaryDisplayTexture;
	}

	// Prepare paletted surface for display
	if (surface.IsPaletteDirty && IsUsingEmulation() && !primary.PaletteTexture)
	{
		CopyEmulatedPaletteSurface(nullptr);
	}

	// Return palette display texture
	if (surface.DisplayTexture)
	{
		return surface.DisplayTexture;
	}

	// Return surface texture
	return surface.Texture;
}

void m_IDirectDrawSurfaceX::CheckMipMapLevelGen()
{
	if (!IsMipMapReadyToUse)
	{
		for (UINT x = 0; x < min(MaxMipMapLevel, MipMaps.size()); x++)
		{
			if (!MipMaps[x].IsDummy && MipMaps[x].UniquenessValue < UniquenessValue)
			{
				return;
			}
		}
		IsMipMapReadyToUse = true;
	}
}

HRESULT m_IDirectDrawSurfaceX::GenerateMipMapLevels()
{
	IDirect3DSurface9* pSourceSurfaceD9 = Get3DMipMapSurface(0);
	if (!pSourceSurfaceD9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface level!");
		return DDERR_GENERIC;
	}

	for (UINT x = 0; x < min(MaxMipMapLevel, MipMaps.size()); x++)
	{
		if (!MipMaps[x].IsDummy && MipMaps[x].UniquenessValue < UniquenessValue)
		{
			ScopedGetMipMapContext Dest(this, x + 1);
			if (Dest.GetSurface())
			{
				LOG_LIMIT(100, __FUNCTION__ << " (" << this << ") Warning: attempting to add missing data to MipMap surface level: " << (x + 1) <<
					" UniquenessValue: " << MipMaps[x].UniquenessValue << " -> " << UniquenessValue);
				if (SUCCEEDED(D3DXLoadSurfaceFromSurface(Dest.GetSurface(), nullptr, nullptr, pSourceSurfaceD9, nullptr, nullptr, D3DX_FILTER_LINEAR, 0x00000000)))
				{
					MipMaps[x].UniquenessValue = UniquenessValue;
				}
				else
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not copy MipMap surface level!");
				}
			}
		}
	}

	CheckMipMapLevelGen();

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::CreateD9AuxiliarySurfaces()
{
	// Create primary surface texture
	if (!PrimaryDisplayTexture && ShouldPresentToWindow(false))
	{
		D3DSURFACE_DESC Desc;
		if (FAILED(surface.Surface ? surface.Surface->GetDesc(&Desc) : surface.Texture->GetLevelDesc(0, &Desc)) ||
			FAILED((*d3d9Device)->CreateTexture(Desc.Width, Desc.Height, 1, 0, Desc.Format, D3DPOOL_DEFAULT, &PrimaryDisplayTexture, nullptr)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create primary surface texture. Size: " << surface.Width << "x" << surface.Height << " Format: " << surface.Format << " dwCaps: " << surfaceDesc2.ddsCaps);
			return DDERR_GENERIC;
		}
	}

	// Create palette surface
	if (!primary.PaletteTexture && IsPrimarySurface() && surface.Format == D3DFMT_P8)
	{
		if (FAILED((*d3d9Device)->CreateTexture(MaxPaletteSize, MaxPaletteSize, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_MANAGED, &primary.PaletteTexture, nullptr)))
		{
			// Try failover format
			if (FAILED((*d3d9Device)->CreateTexture(MaxPaletteSize, MaxPaletteSize, 1, 0, GetFailoverFormat(D3DFMT_X8R8G8B8), D3DPOOL_MANAGED, &primary.PaletteTexture, nullptr)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create palette surface texture");
				return DDERR_GENERIC;
			}
		}
	}

	// Reset flags
	RecreateAuxiliarySurfaces = false;
	surface.RecreateAuxiliarySurfaces = false;

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::CreateD9Surface()
{
	// Don't recreate surface while it is locked
	if ((surface.Surface || surface.Texture) && IsSurfaceBusy())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: surface is busy! Locked: " << IsSurfaceLocked() << " DC: " << IsSurfaceInDC() << " Blt: " << IsSurfaceBlitting());
	}

	// Check for device interface
	if (FAILED(CheckInterface(__FUNCTION__, true, false, false)))
	{
		return DDERR_GENERIC;
	}

	ScopedCriticalSection ThreadLock(GetCriticalSection());

	// Release existing surface
	ReleaseD9Surface(true, false);

	// Update surface description
	UpdateSurfaceDesc();

	// Get texture format
	surface.Format = GetDisplayFormat(surfaceDesc2.ddpfPixelFormat);
	surface.BitCount = GetBitCount(surface.Format);
	SurfaceRequiresEmulation = (CanSurfaceUseEmulation() && (Config.DdrawEmulateSurface || ShouldEmulate == SC_FORCE_EMULATED ||
		surface.Format == D3DFMT_A8B8G8R8 || surface.Format == D3DFMT_X8B8G8R8 || surface.Format == D3DFMT_B8G8R8 || surface.Format == D3DFMT_R8G8B8));
	const bool CreateSurfaceEmulated = (CanSurfaceUseEmulation() && (SurfaceRequiresEmulation ||
		(IsPrimaryOrBackBuffer() && (Config.DdrawWriteToGDI || Config.DdrawReadFromGDI || Config.DdrawRemoveScanlines))));
	DCRequiresEmulation = (CanSurfaceUseEmulation() &&
		surface.Format != D3DFMT_R5G6B5 && surface.Format != D3DFMT_X1R5G5B5 && surface.Format != D3DFMT_A1R5G5B5 && surface.Format != D3DFMT_R8G8B8 &&
		surface.Format != D3DFMT_X8R8G8B8 && surface.Format != D3DFMT_A8R8G8B8);
	const D3DFORMAT Format = ((surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_NOTUSERLOCKABLE) && surface.Format == D3DFMT_D16_LOCKABLE) ? D3DFMT_D16 : ConvertSurfaceFormat(surface.Format);

	// Check if surface should be a texture
	bool IsTexture = ((IsPrimaryOrBackBuffer() && !ShouldPresentToWindow(false)) || IsPalette() || IsSurfaceTexture());

	// Get memory pool
	bool UseVideoMemory = IsRenderTarget() || IsDepthStencil();
	surface.Pool = (IsPrimaryOrBackBuffer() && ShouldPresentToWindow(false)) ? D3DPOOL_SYSTEMMEM :
		(surfaceDesc2.ddsCaps.dwCaps2 & (DDSCAPS2_TEXTUREMANAGE | DDSCAPS2_D3DTEXTUREMANAGE)) ? D3DPOOL_MANAGED :
		UseVideoMemory ? (((surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_VIDEOMEMORY) || IsPrimaryOrBackBuffer()) ? D3DPOOL_DEFAULT :
			(surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY) ? D3DPOOL_SYSTEMMEM : D3DPOOL_DEFAULT) :
		((IsPrimaryOrBackBuffer() || IsSurfaceTexture()) ? D3DPOOL_MANAGED :									// For now use managed for all textures
			(surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY) ? D3DPOOL_SYSTEMMEM : D3DPOOL_SYSTEMMEM);		// Default to system memory for Lock()

	// Default usage
	surface.Usage = 0;

	// Adjust Width to be byte-aligned
	surface.Width = GetByteAlignedWidth(surfaceDesc2.dwWidth, surface.BitCount);
	surface.Height = surfaceDesc2.dwHeight;

	// Set lockable
	surface.IsLockable = true;

	// Anti-aliasing
	if (IsRenderTarget())
	{
		bool AntiAliasing = (surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_HINTANTIALIASING) || (surfaceDesc2.ddsCaps.dwCaps3 & DDSCAPS3_MULTISAMPLE_MASK);
		if (AntiAliasing && !surface.MultiSampleType)
		{
			DWORD MaxSamples = 0;
			DWORD Mask =
				!(surfaceDesc2.ddsCaps.dwCaps3 & DDSCAPS3_MULTISAMPLE_MASK) ? 0 :
				surfaceDesc2.ddpfPixelFormat.MultiSampleCaps.wBltMSTypes > surfaceDesc2.ddpfPixelFormat.MultiSampleCaps.wFlipMSTypes ?
				surfaceDesc2.ddpfPixelFormat.MultiSampleCaps.wBltMSTypes : surfaceDesc2.ddpfPixelFormat.MultiSampleCaps.wFlipMSTypes;
			if (Mask)
			{
				while ((Mask & 1) == 0)
				{
					Mask >>= 1;
					MaxSamples++;
				}
			}
			// Default to 8 samples as some games have issues with more samples
			surface.MultiSampleType = ddrawParent->GetMultiSampleTypeQuality(Format, MaxSamples ? MaxSamples : D3DMULTISAMPLE_8_SAMPLES, surface.MultiSampleQuality);
		}
	}

	// Set created by
	ShouldEmulate = (ShouldEmulate == SC_NOT_CREATED) ? SC_DONT_FORCE : ShouldEmulate;

	Logging::LogDebug() << __FUNCTION__ " (" << this << ") D3d9 Surface. Size: " << surface.Width << "x" << surface.Height << " Format: " << surface.Format <<
		" Pool: " << surface.Pool << " dwCaps: " << surfaceDesc2.ddsCaps << " " << surfaceDesc2;

	HRESULT hr = DD_OK;

	do {
		// Create depth stencil
		if (IsDepthStencil() && surface.Pool != D3DPOOL_SYSTEMMEM)
		{
			surface.IsLockable = false;
			surface.Type = D3DTYPE_DEPTHSTENCIL;
			surface.Usage = D3DUSAGE_DEPTHSTENCIL;
			surface.Pool = D3DPOOL_DEFAULT;
			if (FAILED((*d3d9Device)->CreateDepthStencilSurface(surface.Width, surface.Height, Format, surface.MultiSampleType, surface.MultiSampleQuality, surface.MultiSampleType ? TRUE : FALSE, &surface.Surface, nullptr)) &&
				FAILED((*d3d9Device)->CreateDepthStencilSurface(surface.Width, surface.Height, GetFailoverFormat(Format), surface.MultiSampleType, surface.MultiSampleQuality, surface.MultiSampleType ? TRUE : FALSE, &surface.Surface, nullptr)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create depth stencil surface. Size: " << surface.Width << "x" << surface.Height << " Format: " << Format << " dwCaps: " << surfaceDesc2.ddsCaps);
				hr = DDERR_GENERIC;
				break;
			}
		}
		// Create render target
		else if (IsRenderTarget())
		{
			// ToDo: if render surface is a texture then create as a texture (MipMaps can be supported on render target textures)
			surface.Usage = D3DUSAGE_RENDERTARGET;
			surface.Pool = D3DPOOL_DEFAULT;
			if (IsSurfaceTexture() || IsPalette())
			{
				surface.IsLockable = false;
				surface.Type = D3DTYPE_TEXTURE;
				if (FAILED((*d3d9Device)->CreateTexture(surface.Width, surface.Height, 1, surface.Usage, Format, surface.Pool, &surface.Texture, nullptr)) &&
					FAILED((*d3d9Device)->CreateTexture(surface.Width, surface.Height, 1, surface.Usage, GetFailoverFormat(Format), surface.Pool, &surface.Texture, nullptr)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create render target texture. Size: " << surface.Width << "x" << surface.Height << " Format: " << Format << " dwCaps: " << surfaceDesc2.ddsCaps);
					hr = DDERR_GENERIC;
					break;
				}
			}
			else
			{
				BOOL IsLockable = (surface.MultiSampleType || Config.AntiAliasing || Config.DdrawUseShadowSurface || (surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_NOTUSERLOCKABLE)) ? FALSE : TRUE;
				surface.IsLockable = (IsLockable == TRUE);
				surface.Type = D3DTYPE_RENDERTARGET;
				if (FAILED((*d3d9Device)->CreateRenderTarget(surface.Width, surface.Height, Format, surface.MultiSampleType, surface.MultiSampleQuality, IsLockable, &surface.Surface, nullptr)) &&
					FAILED((*d3d9Device)->CreateRenderTarget(surface.Width, surface.Height, GetFailoverFormat(Format), surface.MultiSampleType, surface.MultiSampleQuality, IsLockable, &surface.Surface, nullptr)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create render target surface. Size: " << surface.Width << "x" << surface.Height << " Format: " << Format << " dwCaps: " << surfaceDesc2.ddsCaps);
					hr = DDERR_GENERIC;
					break;
				}
			}
			// Update attached stencil surface
			m_IDirectDrawSurfaceX* lpAttachedSurfaceX = GetAttachedDepthStencil();
			if (lpAttachedSurfaceX)
			{
				UpdateAttachedDepthStencil(lpAttachedSurfaceX);
			}
		}
		// Create texture
		else if (IsTexture)
		{
			surface.Type = D3DTYPE_TEXTURE;
			DWORD MipMapCount = (surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) ? surfaceDesc2.dwMipMapCount : 1;
			DWORD MipMapLevel = (CreateSurfaceEmulated || !MipMapCount) ? 1 : MipMapCount;
			HRESULT hr_t;
			do {
				surface.Usage = (Config.DdrawForceMipMapAutoGen && MipMapLevel > 1) ? D3DUSAGE_AUTOGENMIPMAP : 0;
				DWORD Level = ((surface.Usage & D3DUSAGE_AUTOGENMIPMAP) && MipMapLevel == MipMapCount) ? 0 : MipMapLevel;
				// Create texture
				hr_t = (*d3d9Device)->CreateTexture(surface.Width, surface.Height, Level, surface.Usage, Format, surface.Pool, &surface.Texture, nullptr);
				if (FAILED(hr_t))
				{
					hr_t = (*d3d9Device)->CreateTexture(surface.Width, surface.Height, Level, surface.Usage, GetFailoverFormat(Format), surface.Pool, &surface.Texture, nullptr);
				}
			} while (FAILED(hr_t) && ((!MipMapLevel && ++MipMapLevel) || --MipMapLevel > 0));
			if (FAILED(hr_t))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create surface texture. Size: " << surface.Width << "x" << surface.Height << " Format: " << Format << " dwCaps: " << surfaceDesc2.ddsCaps);
				hr = DDERR_GENERIC;
				break;
			}
			MaxMipMapLevel = (MipMapLevel > 1 && !IsMipMapAutogen()) ? MipMapLevel - 1 : 0;
			while (MipMaps.size() < MaxMipMapLevel)
			{
				MIPMAP MipMap;
				MipMaps.push_back(MipMap);
			}
			if ((surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) && !IsMipMapAutogen())
			{
				surfaceDesc2.dwMipMapCount = MipMapLevel;
			}
		}
		else
		{
			const D3DFORMAT NewFormat = IsDepthStencil() ? GetStencilEmulatedFormat(surface.BitCount) : Format;
			surface.Type = D3DTYPE_OFFPLAINSURFACE;
			if (FAILED((*d3d9Device)->CreateOffscreenPlainSurface(surface.Width, surface.Height, NewFormat, surface.Pool, &surface.Surface, nullptr)) &&
				FAILED((*d3d9Device)->CreateOffscreenPlainSurface(surface.Width, surface.Height, GetFailoverFormat(NewFormat), surface.Pool, &surface.Surface, nullptr)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create offplain surface. Size: " << surface.Width << "x" << surface.Height << " Format: " << NewFormat << " dwCaps: " << surfaceDesc2.ddsCaps);
				hr = DDERR_GENERIC;
				break;
			}
		}

		if (FAILED(CreateD9AuxiliarySurfaces()))
		{
			hr = DDERR_GENERIC;
			break;
		}

		surface.IsPaletteDirty = IsPalette();

	} while (false);

	// Create emulated surface using device context for creation
	bool EmuSurfaceCreated = false;
	if ((CreateSurfaceEmulated || IsUsingEmulation()) && !DoesDCMatch(surface.emu))
	{
		EmuSurfaceCreated = true;
		CreateDCSurface();
	}

	// Reset flags
	surface.HasData = false;
	surface.UsingShadowSurface = false;

	// Restore d3d9 surface texture data
	if (surface.Surface || surface.Texture)
	{
		// Fill surface with color
		if (Config.DdrawFillSurfaceColor)
		{
			static DWORD Count = 0;
			struct COLORS {
				DWORD a;
				DWORD r;
				DWORD g;
				DWORD b;
			};
			COLORS Colors[] = {
				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0x00000000, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0x55555555, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0x55555555, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },
				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
				{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
				{ 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF, 0xFFFFFFFF },
				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF },

				{ 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF },
				{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x55555555 },
				{ 0xFFFFFFFF, 0x55555555, 0xFFFFFFFF, 0xFFFFFFFF },
			};

			if (Format == D3DFMT_P8)
			{
				ColorFill(nullptr, 10 * (Count + 1), 0);
			}
			else if (IsPixelFormatRGB(surfaceDesc2.ddpfPixelFormat))
			{
				ColorFill(nullptr,
					(Colors[Count].a & surfaceDesc2.ddpfPixelFormat.dwRGBAlphaBitMask) +
					(Colors[Count].r & surfaceDesc2.ddpfPixelFormat.dwRBitMask) +
					(Colors[Count].g & surfaceDesc2.ddpfPixelFormat.dwGBitMask) +
					(Colors[Count].b & surfaceDesc2.ddpfPixelFormat.dwBBitMask), 0);
			}
			else
			{
				ComPtr<IDirect3DSurface9> SrcSurface;
				DWORD t_Width = 128, t_Height = 128;
				if (SUCCEEDED((*d3d9Device)->CreateOffscreenPlainSurface(t_Width, t_Height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, SrcSurface.GetAddressOf(), nullptr)))
				{
					D3DCOLOR NewColor =
						(Colors[Count].a & 0xFF000000) +
						(Colors[Count].r & 0x00FF0000) +
						(Colors[Count].g & 0x0000FF00) +
						(Colors[Count].b & 0x000000FF);
					D3DLOCKED_RECT LockedRect = {};
					if (SUCCEEDED(SrcSurface->LockRect(&LockedRect, nullptr, 0)))
					{
						BYTE* pBuffer = (BYTE*)LockedRect.pBits;
						for (UINT x = 0; x < t_Width; x++)
						{
							for (UINT y = 0; y < LockedRect.Pitch / sizeof(D3DCOLOR); y++)
							{
								((DWORD*)pBuffer)[y] = NewColor;
							}
							pBuffer += LockedRect.Pitch;
						}
						SrcSurface->UnlockRect();

						LPDIRECT3DSURFACE9 DstSurface = Get3DSurface();
						if (DstSurface)
						{
							D3DXLoadSurfaceFromSurface(DstSurface, nullptr, nullptr, SrcSurface.Get(), nullptr, nullptr, D3DX_FILTER_POINT, 0);
						}
					}
				}
			}
			if (++Count >= 24)
			{
				Count = 0;
			}
		}

		// Restore surface texture data
		bool RestoreData = false;
		if (IsUsingEmulation() && !EmuSurfaceCreated)
		{
			// Copy surface to emulated surface
			CopyFromEmulatedSurface(nullptr);
			RestoreData = true;
			surface.HasData = true;
		}
		else if (!LostDeviceBackup.empty())
		{
			if ((LostDeviceBackup[0].Format == Format || GetFailoverFormat(LostDeviceBackup[0].Format) == Format || LostDeviceBackup[0].Format == GetFailoverFormat(Format)) &&
				LostDeviceBackup[0].Width == surface.Width && LostDeviceBackup[0].Height == surface.Height)
			{
				for (UINT Level = 0; Level < LostDeviceBackup.size(); Level++)
				{
					// Check if render target should use shadow
					if (Level == 0 && (surface.Usage & D3DUSAGE_RENDERTARGET) && !surface.IsLockable && !IsUsingShadowSurface())
					{
						SetRenderTargetShadow();
					}

					D3DLOCKED_RECT LockRect = {};
					if (FAILED(LockD3d9Surface(&LockRect, nullptr, 0, Level)))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to restore surface data!");
						break;
					}

					Logging::LogDebug() << __FUNCTION__ << " Restoring Direct3D9 texture surface data: " << Format;

					D3DSURFACE_DESC Desc = {};
					if (FAILED(surface.Surface ? surface.Surface->GetDesc(&Desc) : surface.Texture->GetLevelDesc(GetD3d9MipMapLevel(Level), &Desc)))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface desc!");
						break;
					}

					if (LostDeviceBackup[Level].Format == Desc.Format && LostDeviceBackup[Level].Width == Desc.Width && LostDeviceBackup[Level].Height == Desc.Height)
					{
						size_t size = GetSurfaceSize(Desc.Format, Desc.Width, Desc.Height, LockRect.Pitch);

						if (size == LostDeviceBackup[Level].Bits.size())
						{
							memcpy(LockRect.pBits, LostDeviceBackup[Level].Bits.data(), size);
						}
						else
						{
							BYTE* pSrcSurface = LostDeviceBackup[Level].Bits.data();
							BYTE* pDestSurface = (BYTE*)LockRect.pBits;
							DWORD MinPitchSize = min((UINT)LockRect.Pitch, LostDeviceBackup[Level].Pitch);

							for (UINT x = 0; x < surface.Height; x++)
							{
								memcpy(pDestSurface, pSrcSurface, MinPitchSize);

								pSrcSurface += LostDeviceBackup[Level].Pitch;
								pDestSurface += LockRect.Pitch;
							}
						}

						RestoreData = true;
						surface.HasData = true;
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: restore backup surface data mismatch! For Level: " << Level << " " <<
							LostDeviceBackup[Level].Format << " -> " << Format << " " << LostDeviceBackup[Level].Width << "x" << LostDeviceBackup[Level].Height << " -> " <<
							surface.Width << "x" << surface.Height << " " << LostDeviceBackup[Level].Pitch << " - > " << LockRect.Pitch);
					}

					UnLockD3d9Surface(Level);

					// Copy surface to emulated surface
					if (IsUsingEmulation() && Level == 0 && surface.HasData)
					{
						CopyToEmulatedSurface(nullptr);
					}
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: restore backup surface data mismatch!: " <<
					LostDeviceBackup[0].Format << " -> " << Format << " " << LostDeviceBackup[0].Width << "x" << LostDeviceBackup[0].Height << " -> " <<
					surface.Width << "x" << surface.Height);
			}
		}

		// Copy surface to display texture
		if (RestoreData && PrimaryDisplayTexture)
		{
			IDirect3DSurface9* pSrcSurfaceD9 = Get3DSurface();
			if (pSrcSurfaceD9)
			{
				ComPtr<IDirect3DSurface9> pPrimaryDisplaySurfaceD9;
				if (SUCCEEDED(PrimaryDisplayTexture->GetSurfaceLevel(0, pPrimaryDisplaySurfaceD9.GetAddressOf())))
				{
					D3DXLoadSurfaceFromSurface(pPrimaryDisplaySurfaceD9.Get(), nullptr, nullptr, pSrcSurfaceD9, nullptr, nullptr, D3DX_FILTER_NONE, 0);
				}
			}
		}

		// Data is no longer needed
		LostDeviceBackup.clear();
	}

	// Delete emulatd surface if not needed
	if (!CreateSurfaceEmulated && IsUsingEmulation())
	{
		ReleaseDCSurface();
	}

	return hr;
}

bool m_IDirectDrawSurfaceX::DoesDCMatch(EMUSURFACE* pEmuSurface) const
{
	if (!pEmuSurface || !pEmuSurface->DC || !pEmuSurface->pBits)
	{
		return false;
	}

	// Adjust Width to be byte-aligned
	DWORD Width = GetByteAlignedWidth(surfaceDesc2.dwWidth, surface.BitCount);
	DWORD Height = surfaceDesc2.dwHeight;
	DWORD Pitch = ComputePitch(surface.Format, Width, surface.BitCount);

	if (pEmuSurface->bmi->bmiHeader.biWidth == (LONG)Width &&
		pEmuSurface->bmi->bmiHeader.biHeight == -(LONG)Height &&
		pEmuSurface->bmi->bmiHeader.biBitCount == (WORD)surface.BitCount &&
		pEmuSurface->Format == surface.Format &&
		pEmuSurface->Pitch == Pitch)
	{
		return true;
	}

	return false;
}

void m_IDirectDrawSurfaceX::SetEmulationGameDC()
{
	if (IsUsingEmulation() && !surface.emu->UsingGameDC)
	{
		// Restore old object into DC
		HGDIOBJ NewObject = SelectObject(surface.emu->DC, surface.emu->OldDCObject);
		if (!NewObject || NewObject == HGDI_ERROR)
		{
			Logging::Log() << __FUNCTION__ << " Error: failed to select old object into DC!";
			return;
		}
		// Select bitmap into GameDC
		surface.emu->OldGameDCObject = SelectObject(surface.emu->GameDC, surface.emu->bitmap);
		if (!surface.emu->OldGameDCObject || surface.emu->OldGameDCObject == HGDI_ERROR)
		{
			Logging::Log() << __FUNCTION__ << " Error: failed to select bitmap into GameDC!";
			return;
		}
		// Set DC flag
		surface.emu->UsingGameDC = true;
	}
}

void m_IDirectDrawSurfaceX::UnsetEmulationGameDC()
{
	if (IsUsingEmulation() && surface.emu->UsingGameDC)
	{
		// Restore old object into GameDC
		HGDIOBJ NewObject = SelectObject(surface.emu->GameDC, surface.emu->OldGameDCObject);
		if (!NewObject || NewObject == HGDI_ERROR)
		{
			Logging::Log() << __FUNCTION__ << " Error: failed to select old object into GameDC!";
			return;
		}
		// Select bitmap into DC
		surface.emu->OldDCObject = SelectObject(surface.emu->DC, surface.emu->bitmap);
		if (!surface.emu->OldDCObject || surface.emu->OldDCObject == HGDI_ERROR)
		{
			Logging::Log() << __FUNCTION__ << " Error: failed to select bitmap into GameDC!";
			return;
		}
		// Unset DC flag
		surface.emu->UsingGameDC = false;
	}
}

HRESULT m_IDirectDrawSurfaceX::CreateDCSurface()
{
	ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

	// Check if color masks are needed
	bool ColorMaskReq = ((surface.BitCount == 16 || surface.BitCount == 24 || surface.BitCount == 32) &&									// Only valid when used with 16 bit, 24 bit and 32 bit surfaces
		(surfaceDesc2.ddpfPixelFormat.dwRBitMask || surfaceDesc2.ddpfPixelFormat.dwGBitMask || surfaceDesc2.ddpfPixelFormat.dwBBitMask));	// Check to make sure the masks actually exist

	// Adjust Width to be byte-aligned
	DWORD Width = GetByteAlignedWidth(surfaceDesc2.dwWidth, surface.BitCount);
	DWORD Height = surfaceDesc2.dwHeight;
	DWORD Pitch = ComputePitch(surface.Format, Width, surface.BitCount);

	// Check if emulated surface already exists
	if (surface.emu)
	{
		// Restore DC
		UnsetEmulationGameDC();

		// Check if emulated memory is good
		if (!IsUsingEmulation())
		{
			DeleteEmulatedMemory(&surface.emu);
		}
		else
		{
			// Check if current emulated surface is still ok
			if (DoesDCMatch(surface.emu))
			{
				return DD_OK;
			}

			// Save current emulated surface and prepare for creating a new one.
			if (ShareEmulatedMemory)
			{
				memorySurfaces.push_back(surface.emu);
				surface.emu = nullptr;
			}
			else
			{
				DeleteEmulatedMemory(&surface.emu);
			}
		}
	}

	// If sharing memory than check the shared memory vector for a surface that matches
	if (ShareEmulatedMemory)
	{
		for (auto it = memorySurfaces.begin(); it != memorySurfaces.end(); it++)
		{
			EMUSURFACE* pEmuSurface = *it;

			if (DoesDCMatch(pEmuSurface))
			{
				surface.emu = pEmuSurface;

				it = memorySurfaces.erase(it);

				break;
			}
		}

		if (surface.emu && surface.emu->pBits)
		{
			ZeroMemory(surface.emu->pBits, surface.emu->Size);

			return DD_OK;
		}
	}

	Logging::LogDebug() << __FUNCTION__ " (" << this << ") creating emulated surface. Size: " << Width << "x" << Height << " Format: " << surface.Format << " dwCaps: " << surfaceDesc2.ddsCaps;

	// Create new emulated surface structure
	surface.emu = new EMUSURFACE;

	// Create device context memory
	ZeroMemory(surface.emu->bmiMemory, sizeof(surface.emu->bmiMemory));
	surface.emu->bmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	surface.emu->bmi->bmiHeader.biWidth = Width;
	surface.emu->bmi->bmiHeader.biHeight = -((LONG)Height + (LONG)ExtraHeightPadding);
	surface.emu->bmi->bmiHeader.biPlanes = 1;
	surface.emu->bmi->bmiHeader.biBitCount = (WORD)surface.BitCount;
	surface.emu->bmi->bmiHeader.biCompression =
		(surface.BitCount == 8 || surface.BitCount == 24) ? BI_RGB :
		(ColorMaskReq) ? BI_BITFIELDS : 0;	// BI_BITFIELDS is only valid for 16-bpp and 32-bpp bitmaps.
	surface.emu->bmi->bmiHeader.biSizeImage = ((Width * surface.BitCount + 31) & ~31) / 8 * Height;

	if (surface.BitCount == 8)
	{
		for (int i = 0; i < 256; i++)
		{
			surface.emu->bmi->bmiColors[i].rgbRed = (byte)i;
			surface.emu->bmi->bmiColors[i].rgbGreen = (byte)i;
			surface.emu->bmi->bmiColors[i].rgbBlue = (byte)i;
			surface.emu->bmi->bmiColors[i].rgbReserved = 0;
		}
	}
	else if (ColorMaskReq)
	{
		((DWORD*)surface.emu->bmi->bmiColors)[0] = surfaceDesc2.ddpfPixelFormat.dwRBitMask;
		((DWORD*)surface.emu->bmi->bmiColors)[1] = surfaceDesc2.ddpfPixelFormat.dwGBitMask;
		((DWORD*)surface.emu->bmi->bmiColors)[2] = surfaceDesc2.ddpfPixelFormat.dwBBitMask;
		((DWORD*)surface.emu->bmi->bmiColors)[3] = surfaceDesc2.ddpfPixelFormat.dwRGBAlphaBitMask;
	}
	else
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to set bmi colors! " << surface.Format << " " << surface.BitCount);
		DeleteEmulatedMemory(&surface.emu);
		return DDERR_GENERIC;
	}
	HDC hDC = ddrawParent->GetDC();
	surface.emu->DC = CreateCompatibleDC(hDC);
	surface.emu->GameDC = CreateCompatibleDC(hDC);
	if (!surface.emu->DC || !surface.emu->GameDC)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create compatible DC: " << hDC << " " << surface.Format);
		DeleteEmulatedMemory(&surface.emu);
		return DDERR_GENERIC;
	}
	surface.emu->bitmap = CreateDIBSection(surface.emu->DC, surface.emu->bmi, (surface.BitCount == 8) ? DIB_PAL_COLORS : DIB_RGB_COLORS, (void**)&surface.emu->pBits, nullptr, 0);
	if (!surface.emu->bitmap)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create bitmap!");
		DeleteEmulatedMemory(&surface.emu);
		return DDERR_GENERIC;
	}
	surface.emu->OldDCObject = SelectObject(surface.emu->DC, surface.emu->bitmap);
	if (!surface.emu->OldDCObject || surface.emu->OldDCObject == HGDI_ERROR)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to replace object in DC!");
		DeleteEmulatedMemory(&surface.emu);
		return DDERR_GENERIC;
	}
	surface.emu->bmi->bmiHeader.biHeight = -(LONG)Height;
	surface.emu->Format = surface.Format;
	surface.emu->Pitch = Pitch;
	surface.emu->Size = Height * Pitch;

	return DD_OK;
}

// Check surface for alpha channel
bool m_IDirectDrawSurfaceX::HasAlphaChannel() const
{
	if (surfaceDesc2.ddpfPixelFormat.dwFlags & DDPF_ALPHAPIXELS)
	{
		return true;
	}
	switch (surfaceDesc2.ddpfPixelFormat.dwFourCC)
	{
	case D3DFMT_DXT1:
		// ToDo: maybe need to check if DXT1 texture has alpha bits
	case D3DFMT_DXT2:
	case D3DFMT_DXT3:
	case D3DFMT_DXT4:
	case D3DFMT_DXT5:
		return true;
	}
	return false;
}

void m_IDirectDrawSurfaceX::UpdateAttachedDepthStencil(m_IDirectDrawSurfaceX* lpAttachedSurfaceX)
{
	bool HasChanged = false;
	// Verify depth stencil's with and height
	if ((surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT)) == (DDSD_WIDTH | DDSD_HEIGHT) &&
		(surfaceDesc2.dwWidth != lpAttachedSurfaceX->surfaceDesc2.dwWidth || surfaceDesc2.dwHeight != lpAttachedSurfaceX->surfaceDesc2.dwHeight))
	{
		HasChanged = true;
		lpAttachedSurfaceX->surfaceDesc2.dwWidth = surfaceDesc2.dwWidth;
		lpAttachedSurfaceX->surfaceDesc2.dwHeight = surfaceDesc2.dwHeight;
	}
	// Set depth stencil multisampling
	if (surface.MultiSampleType != lpAttachedSurfaceX->surface.MultiSampleType || surface.MultiSampleQuality != lpAttachedSurfaceX->surface.MultiSampleQuality)
	{
		HasChanged = true;
		lpAttachedSurfaceX->surface.MultiSampleType = surface.MultiSampleType;
		lpAttachedSurfaceX->surface.MultiSampleQuality = surface.MultiSampleQuality;
	}
	// If depth stencil changed
	if (HasChanged)
	{
		lpAttachedSurfaceX->ReleaseD9Surface(false, false);
	}
	// Set depth stencil
	if (ddrawParent->GetRenderTargetSurface() == this)
	{
		ddrawParent->SetDepthStencilSurface(lpAttachedSurfaceX);
	}
}

void m_IDirectDrawSurfaceX::UpdateSurfaceDesc()
{
	bool IsChanged = false;
	if (SUCCEEDED(CheckInterface(__FUNCTION__, false, false, false)) &&
		((surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT)) != (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT) ||
		((surfaceDesc2.dwFlags & DDSD_REFRESHRATE) && !surfaceDesc2.dwRefreshRate)))
	{
		// Get resolution
		DWORD Width, Height, RefreshRate, BPP;
		ddrawParent->GetSurfaceDisplay(Width, Height, BPP, RefreshRate);

		// Set Height and Width
		if (Width && Height &&
			(surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT)) != (DDSD_WIDTH | DDSD_HEIGHT))
		{
			ResetDisplayFlags |= DDSD_WIDTH | DDSD_HEIGHT;
			surfaceDesc2.dwFlags |= DDSD_WIDTH | DDSD_HEIGHT;
			surfaceDesc2.dwWidth = Width;
			surfaceDesc2.dwHeight = Height;
			surfaceDesc2.lPitch = 0;
			IsChanged = true;
		}
		// Set Refresh Rate
		if (RefreshRate && ((surfaceDesc2.dwFlags & DDSD_REFRESHRATE) || IsPrimaryOrBackBuffer()))
		{
			surfaceDesc2.dwFlags |= DDSD_REFRESHRATE;
			surfaceDesc2.dwRefreshRate = RefreshRate;
		}
		// Set PixelFormat
		if (BPP && !(surfaceDesc2.dwFlags & DDSD_PIXELFORMAT))
		{
			ResetDisplayFlags |= DDSD_PIXELFORMAT;
			surfaceDesc2.dwFlags |= DDSD_PIXELFORMAT;
			ddrawParent->GetDisplayPixelFormat(surfaceDesc2.ddpfPixelFormat, BPP);
			surfaceDesc2.lPitch = 0;
			IsChanged = true;
		}
		// Reset MipMap level pitch
		if (IsChanged && MipMaps.size())
		{
			for (auto& entry : MipMaps)
			{
				entry.dwWidth = 0;
				entry.dwHeight = 0;
				entry.lPitch = 0;
			}
		}
	}
	// Remove surface memory pointer
	if (!surface.UsingSurfaceMemory)
	{
		surfaceDesc2.dwFlags &= ~DDSD_LPSURFACE;
		surfaceDesc2.lpSurface = nullptr;
	}
	// Unset lPitch
	if ((((surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT)) != (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT) ||
		!(surfaceDesc2.dwFlags & DDSD_PITCH)) && !(surfaceDesc2.dwFlags & DDSD_LINEARSIZE)) || !surfaceDesc2.lPitch)
	{
		surfaceDesc2.dwFlags &= ~(DDSD_PITCH | DDSD_LINEARSIZE);
		surfaceDesc2.lPitch = 0;
	}
	// Set lPitch
	if ((surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT)) == (DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT) &&
		!(surfaceDesc2.dwFlags & DDSD_LINEARSIZE) && !(surfaceDesc2.dwFlags & DDSD_PITCH))
	{
		DWORD Pitch = ComputePitch(GetDisplayFormat(surfaceDesc2.ddpfPixelFormat), surfaceDesc2.dwWidth, surfaceDesc2.dwHeight);
		if (Pitch)
		{
			surfaceDesc2.dwFlags |= DDSD_PITCH;
			surfaceDesc2.lPitch = Pitch;
		}
	}
	// Set surface format
	if (surface.Format == D3DFMT_UNKNOWN && (surfaceDesc2.dwFlags & DDSD_PIXELFORMAT))
	{
		surface.Format = GetDisplayFormat(surfaceDesc2.ddpfPixelFormat);
	}
	// Set attached stencil surface size
	if (IsChanged && (surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT)) == (DDSD_WIDTH | DDSD_HEIGHT))
	{
		m_IDirectDrawSurfaceX* lpAttachedSurfaceX = GetAttachedDepthStencil();
		if (lpAttachedSurfaceX && (surfaceDesc2.dwWidth != lpAttachedSurfaceX->surfaceDesc2.dwWidth ||
			surfaceDesc2.dwHeight != lpAttachedSurfaceX->surfaceDesc2.dwHeight ||
			(lpAttachedSurfaceX->surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY)))
		{
			lpAttachedSurfaceX->ReleaseD9Surface(false, false);
			lpAttachedSurfaceX->surfaceDesc2.dwWidth = surfaceDesc2.dwWidth;
			lpAttachedSurfaceX->surfaceDesc2.dwHeight = surfaceDesc2.dwHeight;
			lpAttachedSurfaceX->surfaceDesc2.ddsCaps.dwCaps = (lpAttachedSurfaceX->surfaceDesc2.ddsCaps.dwCaps & ~DDSCAPS_SYSTEMMEMORY) | DDSCAPS_VIDEOMEMORY | DDSCAPS_LOCALVIDMEM;
			if (ddrawParent->GetRenderTargetSurface() == this)
			{
				ddrawParent->SetDepthStencilSurface(lpAttachedSurfaceX);
			}
		}
	}
}

void m_IDirectDrawSurfaceX::SetAsRenderTarget()
{
	if (!surface.CanBeRenderTarget)
	{
		surface.CanBeRenderTarget = true;
		if (surface.Surface || surface.Texture)
		{
			CreateD9Surface();
		}
		if (!AttachedSurfaceMap.empty())
		{
			for (auto& entry : AttachedSurfaceMap)
			{
				if (entry.second.pSurface->IsPrimaryOrBackBuffer() && entry.second.pSurface->IsSurface3D() && !entry.second.pSurface->IsRenderTarget())
				{
					entry.second.pSurface->SetAsRenderTarget();
				}
			}
		}
	}
}

DWORD m_IDirectDrawSurfaceX::GetAttachedStencilSurfaceZBits()
{
	m_IDirectDrawSurfaceX* lpAttachedSurfaceX = GetAttachedDepthStencil();
	if (lpAttachedSurfaceX)
	{
		DWORD ZMask = lpAttachedSurfaceX->surfaceDesc2.ddpfPixelFormat.dwZBitMask;
		DWORD ZBits = 0;
		while (ZMask)
		{
			ZBits += (ZMask & 1);
			ZMask >>= 1;
		}
		return max(ZBits, 15);
	}
	return 16; // safe default
}

void m_IDirectDrawSurfaceX::ClearUsing3DFlag()
{
	Using3D = false;

	if (surface.CanBeRenderTarget)
	{
		surface.CanBeRenderTarget = false;
		if (surface.Surface || surface.Texture)
		{
			CreateD9Surface();
		}
		if (!AttachedSurfaceMap.empty())
		{
			for (auto& entry : AttachedSurfaceMap)
			{
				if (entry.second.pSurface->IsPrimaryOrBackBuffer() && entry.second.pSurface->IsRenderTarget())
				{
					entry.second.pSurface->ClearUsing3DFlag();
				}
			}
		}
	}
}

void m_IDirectDrawSurfaceX::ReleaseD9AuxiliarySurfaces()
{
	// Release d3d9 shadow surface when surface is released
	if (surface.Shadow)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 surface";
		ULONG ref = surface.Shadow->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'surface.Shadow' " << ref;
		}
		surface.Shadow = nullptr;
		surface.UsingShadowSurface = false;
	}

	// Release d3d9 tmp shadow surface when surface is released
	if (tmpVideo.Texture)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 surface";
		ULONG ref = tmpVideo.Texture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'tmpVideo.Texture' " << ref;
		}
		tmpVideo.Texture = nullptr;
	}
	if (tmpVideo.Surface)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 surface";
		ULONG ref = tmpVideo.Surface->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'tmpVideo.Surface' " << ref;
		}
		tmpVideo.Surface = nullptr;
	}

	// Release primary display texture
	if (PrimaryDisplayTexture)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 primary display texture";
		ULONG ref = PrimaryDisplayTexture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'PrimaryDisplayTexture' " << ref;
		}
		PrimaryDisplayTexture = nullptr;
	}

	// Release d3d9 palette surface texture
	if (primary.PaletteTexture)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 palette texture surface";
		ULONG ref = primary.PaletteTexture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'paletteTexture' " << ref;
		}
		primary.PaletteTexture = nullptr;
	}

	// Release d3d9 color keyed surface texture
	if (surface.DrawTexture)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 DrawTexture surface";
		ULONG ref = surface.DrawTexture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'DrawTexture' " << ref;
		}
		surface.DrawTexture = nullptr;
	}

	// Release d3d9 context surface
	if (surface.Context)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 context surface";
		ULONG ref = surface.Context->Release();
		if (ref > 1)	// Ref count is higher becasue it is a surface of 'surfaceTexture'
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'contextSurface' " << ref;
		}
		surface.Context = nullptr;
	}

	// Release d3d9 palette context texture
	if (surface.DisplayContext)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 palette context texture";
		ULONG ref = surface.DisplayContext->Release();
		if (ref > 1)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'paletteDisplaySurface' " << ref;
		}
		surface.DisplayContext = nullptr;
	}

	// Release d3d9 palette display texture
	if (surface.DisplayTexture)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 palette display texture";
		ULONG ref = surface.DisplayTexture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'paletteDisplayTexture' " << ref;
		}
		surface.DisplayTexture = nullptr;
	}

	// Set flags
	RecreateAuxiliarySurfaces = true;
	surface.RecreateAuxiliarySurfaces = true;
}

void m_IDirectDrawSurfaceX::ReleaseD9Surface(bool BackupData, bool ResetSurface)
{
	// Check if surface is busy
	if (IsSurfaceBusy())
	{
		Logging::Log() << __FUNCTION__ << " Warning: surface still in use! Locked: " << IsSurfaceLocked() << " DC: " << IsSurfaceInDC() << " Blt: " << IsSurfaceBlitting();
	}

	// Release DC levels (before releasing surface)
	for (auto& entry : GetDCLevel)
	{
		if (entry.second)
		{
			ReleaseDC(entry.second, entry.first);
			entry.second = nullptr;
		}
	}

	// Restore DC
	UnsetEmulationGameDC();

	// Unlock surface (before releasing)
	for (auto& entry : LockedLevel)
	{
		if (entry.second.IsLocked)
		{
			UnLockD3d9Surface(entry.first);
			entry.second.IsLocked = false;
			entry.second.LockedWithID = 0;
			entry.second.LockRectList.clear();
		}
	}

	IsInBlt = false;
	IsInBltBatch = false;
	LockedLevel[0].IsLocked = false;
	GetDCLevel[0] = nullptr;
	IsInFlip = false;

	const bool ShouldReleaseMainSurface = (!ResetSurface || IsD9UsingVideoMemory());

	// Backup d3d9 surface texture
	if (BackupData)
	{
		if ((surface.Surface || surface.Texture) &&
			surface.HasData && ShouldReleaseMainSurface &&
			!(IsDepthStencil() || (surface.Usage & D3DUSAGE_DEPTHSTENCIL)))
		{
			if (!IsUsingEmulation() && LostDeviceBackup.empty())
			{
				for (UINT Level = 0; Level < ((IsMipMapAutogen() || !MaxMipMapLevel) ? 1 : MaxMipMapLevel); Level++)
				{
					// Check if render target should use shadow
					if (Level == 0 && (surface.Usage & D3DUSAGE_RENDERTARGET) && !surface.IsLockable && !IsUsingShadowSurface())
					{
						SetRenderTargetShadow();
					}

					D3DLOCKED_RECT LockRect = {};
					if (FAILED(LockD3d9Surface(&LockRect, nullptr, D3DLOCK_READONLY, Level)))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to backup surface data!");
						break;
					}

					D3DSURFACE_DESC Desc = {};
					if (FAILED(surface.Surface ? surface.Surface->GetDesc(&Desc) : surface.Texture->GetLevelDesc(GetD3d9MipMapLevel(Level), &Desc)))
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface desc!");
						break;
					}

					Logging::LogDebug() << __FUNCTION__ << " Storing Direct3D9 texture surface data: " << Desc.Format;

					size_t size = GetSurfaceSize(Desc.Format, Desc.Width, Desc.Height, LockRect.Pitch);

					DDBACKUP entry;
					LostDeviceBackup.push_back(entry);
					if (size && LockRect.pBits && LostDeviceBackup.size() > Level)
					{
						LostDeviceBackup[Level].Format = Desc.Format;
						LostDeviceBackup[Level].Width = Desc.Width;
						LostDeviceBackup[Level].Height = Desc.Height;
						LostDeviceBackup[Level].Pitch = LockRect.Pitch;
						LostDeviceBackup[Level].Bits.resize(size);

						memcpy(LostDeviceBackup[Level].Bits.data(), LockRect.pBits, size);
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: mismatch in LostDeviceBackup data structure!");
					}

					UnLockD3d9Surface(Level);
				}
			}
		}
	}
	// Release emulated surface if not backing up surface
	else if (IsUsingEmulation())
	{
		ReleaseDCSurface();
	}

	ReleaseD9AuxiliarySurfaces();

	// Release d3d9 3D surface
	if (surface.Surface && ShouldReleaseMainSurface)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 surface";
		ULONG ref = surface.Surface->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'surface3D' " << ref;
		}
		surface.Surface = nullptr;
	}

	// Release d3d9 surface texture
	if (surface.Texture && ShouldReleaseMainSurface)
	{
		Logging::LogDebug() << __FUNCTION__ << " Releasing Direct3D9 texture surface";
		ULONG ref = surface.Texture->Release();
		if (ref)
		{
			Logging::Log() << __FUNCTION__ << " Error: there is still a reference to 'surfaceTexture' " << ref;
		}
		surface.Texture = nullptr;
	}

	// Reset display flags
	if (ResetDisplayFlags && !ResetSurface)
	{
		surfaceDesc2.dwFlags &= ~ResetDisplayFlags;
		ClearUnusedValues(surfaceDesc2);
	}

	if (surfaceDesc2.dwFlags & DDSD_REFRESHRATE)
	{
		surfaceDesc2.dwRefreshRate = 0;
	}
}

void m_IDirectDrawSurfaceX::ReleaseDCSurface()
{
	if (surface.emu)
	{
		ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

		if (!ShareEmulatedMemory || !IsUsingEmulation())
		{
			DeleteEmulatedMemory(&surface.emu);
		}
		else
		{
			memorySurfaces.push_back(surface.emu);
			surface.emu = nullptr;
		}
	}
}

HRESULT m_IDirectDrawSurfaceX::PresentSurface(LPRECT lpDestRect, bool IsSkipScene)
{
	// Check for device interface
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr))
	{
		return c_hr;
	}

	bool ShouldSkipScene = ((IsSkipScene && !SceneReady) || IsPresentRunning);

	// Check if is not primary surface or if scene should be skipped
	if (ShouldWriteToGDI() || ddrawParent->IsInScene())
	{
		// Never present when writing to GDI, presenting to a window or using Direct3D and InScene
		if (IsPrimarySurface() && !ShouldSkipScene)
		{
			SceneReady = true;
		}
		return DD_OK;
	}
	else if (!IsPrimarySurface())
	{
		if (SceneReady && !IsPresentRunning)
		{
			m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = ddrawParent->GetPrimarySurface();
			if (lpDDSrcSurfaceX)
			{
				return lpDDSrcSurfaceX->PresentSurface(lpDestRect, IsSkipScene);
			}
		}
		return DDERR_GENERIC;
	}
	else if (ShouldSkipScene)
	{
		Logging::LogDebug() << __FUNCTION__ << " Skipping scene!";
		return DDERR_GENERIC;
	}

	// Set scene ready
	SceneReady = true;

	// Check if surface is locked or has an open DC
	if (IsSurfaceBusy())
	{
		Logging::LogDebug() << __FUNCTION__ << " Surface is busy!";
		return DDERR_SURFACEBUSY;
	}

	// Set present flag
	ScopedFlagSet ScopedFlag(IsPresentRunning);

	ScopedCriticalSection ThreadLock(GetCriticalSection());

	// Present to d3d9
	HRESULT hr = ddrawParent->PresentScene(lpDestRect);
	if (FAILED(hr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to present 2D scene!");
	}

	return hr;
}

void m_IDirectDrawSurfaceX::ResetSurfaceDisplay()
{
	if (ResetDisplayFlags)
	{
		ReleaseD9Surface(true, false);
	}
}

bool m_IDirectDrawSurfaceX::CheckCoordinates(RECT& OutRect, LPRECT lpInRect, LPDDSURFACEDESC2 lpDDSurfaceDesc2)
{
	if (!lpDDSurfaceDesc2)
	{
		lpDDSurfaceDesc2 = &surfaceDesc2;
	}

	// Check device coordinates
	if (!lpDDSurfaceDesc2->dwWidth || !lpDDSurfaceDesc2->dwHeight)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface has no size!");
		return false;
	}

	if (lpInRect)
	{
		OutRect.left = lpInRect->left;
		OutRect.top = lpInRect->top;
		OutRect.right = lpInRect->right;
		OutRect.bottom = lpInRect->bottom;
	}
	else
	{
		OutRect.left = 0;
		OutRect.top = 0;
		OutRect.right = lpDDSurfaceDesc2->dwWidth;
		OutRect.bottom = lpDDSurfaceDesc2->dwHeight;
	}

	if (OutRect.left < 0)
	{
		OutRect.left = 0;
	}

	if (OutRect.right < 0)
	{
		OutRect.right = 0;
	}

	if (OutRect.top < 0)
	{
		OutRect.top = 0;
	}

	if (OutRect.bottom < 0)
	{
		OutRect.bottom = 0;
	}

	if (OutRect.left > (LONG)lpDDSurfaceDesc2->dwWidth)
	{
		OutRect.left = lpDDSurfaceDesc2->dwWidth;
	}

	if (OutRect.right > (LONG)lpDDSurfaceDesc2->dwWidth)
	{
		OutRect.right = lpDDSurfaceDesc2->dwWidth;
	}

	if (OutRect.top > (LONG)lpDDSurfaceDesc2->dwHeight)
	{
		OutRect.top = lpDDSurfaceDesc2->dwHeight;
	}

	if (OutRect.bottom > (LONG)lpDDSurfaceDesc2->dwHeight)
	{
		OutRect.bottom = lpDDSurfaceDesc2->dwHeight;
	}

	return OutRect.left < OutRect.right && OutRect.top < OutRect.bottom;
}

void m_IDirectDrawSurfaceX::LockEmuLock(LPRECT lpDestRect, LPDDSURFACEDESC2 lpDDSurfaceDesc)
{
	// Only works if entire surface is locked
	if (!lpDDSurfaceDesc || !lpDDSurfaceDesc->lPitch || (lpDestRect && lpDestRect->top != 0 && lpDestRect->left != 0))
	{
		return;
	}

	DWORD BBP = surface.BitCount;
	LONG NewPitch = (BBP / 8) * lpDDSurfaceDesc->dwWidth;

	bool LockOffPlain = Config.DdrawEmulateLock;
	bool LockByteAlign = (Config.DdrawFixByteAlignment && lpDDSurfaceDesc->lPitch != NewPitch);

	// Emulate lock for offscreen surfaces
	if ((BBP == 8 || BBP == 16 || BBP == 24 || BBP == 32) && (LockOffPlain || LockByteAlign))
	{
		// Set correct pitch
		NewPitch = LockByteAlign ? NewPitch : lpDDSurfaceDesc->lPitch;

		// Store old variables
		EmuLock.Locked = true;
		EmuLock.Addr = lpDDSurfaceDesc->lpSurface;
		EmuLock.Pitch = lpDDSurfaceDesc->lPitch;
		EmuLock.NewPitch = NewPitch;
		EmuLock.BBP = BBP;
		EmuLock.Width = lpDDSurfaceDesc->dwWidth;
		EmuLock.Height = lpDDSurfaceDesc->dwHeight;

		// Update surface memory and pitch
		size_t Size = NewPitch * (lpDDSurfaceDesc->dwHeight + ExtraHeightPadding);
		if (EmuLock.Mem.size() < Size)
		{
			EmuLock.Mem.resize(Size);
		}
		lpDDSurfaceDesc->lpSurface = EmuLock.Mem.data();
		lpDDSurfaceDesc->lPitch = NewPitch;

		// Copy surface data to memory
		BYTE* InAddr = (BYTE*)EmuLock.Addr;
		DWORD InPitch = EmuLock.Pitch;
		BYTE* OutAddr = EmuLock.Mem.data();
		DWORD OutPitch = EmuLock.NewPitch;
		size_t MemWidth = (EmuLock.BBP / 8) * EmuLock.Width;
		for (DWORD x = 0; x < EmuLock.Height; x++)
		{
			memcpy(OutAddr, InAddr, MemWidth);
			InAddr += InPitch;
			OutAddr += OutPitch;
		}

		// Mark as byte align locked
		WasBitAlignLocked = LockByteAlign;
	}
}

void m_IDirectDrawSurfaceX::UnlockEmuLock()
{
	if (EmuLock.Locked && EmuLock.Addr)
	{
		// Copy memory back to surface
		BYTE* InAddr = EmuLock.Mem.data();
		DWORD InPitch = EmuLock.NewPitch;
		BYTE* OutAddr = (BYTE*)EmuLock.Addr;
		DWORD OutPitch = EmuLock.Pitch;
		size_t MemWidth = (EmuLock.BBP / 8) * EmuLock.Width;
		for (DWORD x = 0; x < EmuLock.Height; x++)
		{
			memcpy(OutAddr, InAddr, MemWidth);
			InAddr += InPitch;
			OutAddr += OutPitch;
		}

		EmuLock.Locked = false;
		EmuLock.Addr = nullptr;
	}
}

void m_IDirectDrawSurfaceX::RestoreScanlines(LASTLOCK& LLock) const
{
	DWORD ByteCount = surface.BitCount / 8;
	DWORD RectWidth = LLock.Rect.right - LLock.Rect.left;
	DWORD RectHeight = LLock.Rect.bottom - LLock.Rect.top;

	if (!IsPrimaryOrBackBuffer() || !LLock.LockedRect.pBits ||
		!ByteCount || ByteCount > 4 || RectWidth != EmuScanLine.ScanlineWidth)
	{
		return;
	}

	DWORD size = RectWidth * ByteCount;
	BYTE* DestBuffer = (BYTE*)LLock.LockedRect.pBits;

	// Restore even scanlines
	if (EmuScanLine.bEvenScanlines)
	{
		constexpr DWORD Starting = 0;
		DestBuffer += LLock.LockedRect.Pitch * Starting;

		for (DWORD y = Starting; y < RectHeight; y = y + 2)
		{
			memcpy(DestBuffer, EmuScanLine.EvenScanLine.data(), size);
			DestBuffer += LLock.LockedRect.Pitch * 2;
		}
	}
	// Restore odd scanlines
	else if (EmuScanLine.bOddScanlines)
	{
		constexpr DWORD Starting = 1;
		DestBuffer += LLock.LockedRect.Pitch * Starting;

		for (DWORD y = Starting; y < RectHeight; y = y + 2)
		{
			memcpy(DestBuffer, EmuScanLine.OddScanLine.data(), size);
			DestBuffer += LLock.LockedRect.Pitch * 2;
		}
	}
}

void m_IDirectDrawSurfaceX::RemoveScanlines(LASTLOCK& LLock)
{
	DWORD ByteCount = surface.BitCount / 8;
	DWORD RectWidth = LLock.Rect.right - LLock.Rect.left;
	DWORD RectHeight = LLock.Rect.bottom - LLock.Rect.top;

	// Reset scanline flags
	bool LastSet = (EmuScanLine.bEvenScanlines || EmuScanLine.bOddScanlines);
	EmuScanLine.bOddScanlines = false;
	EmuScanLine.bEvenScanlines = false;

	if (!IsPrimaryOrBackBuffer() || !LLock.LockedRect.pBits ||
		!ByteCount || ByteCount > 4 || RectHeight < 100)
	{
		return;
	}

	DWORD size = EmuScanLine.ScanlineWidth * ByteCount;
	if (EmuScanLine.EvenScanLine.size() < size || EmuScanLine.OddScanLine.size() < size)
	{
		EmuScanLine.EvenScanLine.resize(size);
		EmuScanLine.OddScanLine.resize(size);
	}
	EmuScanLine.ScanlineWidth = RectWidth;

	BYTE* DestBuffer = (BYTE*)LLock.LockedRect.pBits;

	// Check if video has scanlines
	for (DWORD y = 0; y < RectHeight; y++)
	{
		// Check for even scanlines
		if (y % 2 == 0)
		{
			if (y == 0)
			{
				EmuScanLine.bEvenScanlines = true;
				memcpy(EmuScanLine.EvenScanLine.data(), DestBuffer, size);
			}
			else if (EmuScanLine.bEvenScanlines)
			{
				EmuScanLine.bEvenScanlines = (memcmp(EmuScanLine.EvenScanLine.data(), DestBuffer, size) == 0);
			}
		}
		// Check for odd scanlines
		else
		{
			if (y == 1)
			{
				EmuScanLine.bOddScanlines = true;
				memcpy(EmuScanLine.OddScanLine.data(), DestBuffer, size);
			}
			else if (EmuScanLine.bOddScanlines)
			{
				EmuScanLine.bOddScanlines = (memcmp(EmuScanLine.OddScanLine.data(), DestBuffer, size) == 0);
			}
		}
		// Exit if no scanlines found
		if (!EmuScanLine.bOddScanlines && !EmuScanLine.bEvenScanlines)
		{
			break;
		}
		DestBuffer += LLock.LockedRect.Pitch;
	}

	// If all scanlines are set then do nothing
	if (!LastSet && EmuScanLine.bEvenScanlines && EmuScanLine.bOddScanlines)
	{
		EmuScanLine.bEvenScanlines = false;
		EmuScanLine.bOddScanlines = false;
	}

	// Reset destination buffer
	DestBuffer = (BYTE*)LLock.LockedRect.pBits;

	// Double even scanlines
	if (EmuScanLine.bEvenScanlines)
	{
		constexpr DWORD Starting = 0;
		DestBuffer += LLock.LockedRect.Pitch * Starting;

		for (DWORD y = Starting; y < RectHeight - 1; y = y + 2)
		{
			memcpy(DestBuffer, DestBuffer + LLock.LockedRect.Pitch, size);
			DestBuffer += LLock.LockedRect.Pitch * 2;
		}
	}
	// Double odd scanlines
	else if (EmuScanLine.bOddScanlines)
	{
		constexpr DWORD Starting = 1;
		DestBuffer += LLock.LockedRect.Pitch * Starting;

		for (DWORD y = Starting; y < RectHeight; y = y + 2)
		{
			memcpy(DestBuffer, DestBuffer - LLock.LockedRect.Pitch, size);
			DestBuffer += LLock.LockedRect.Pitch * 2;
		}
	}
}

HRESULT m_IDirectDrawSurfaceX::LockEmulatedSurface(D3DLOCKED_RECT* pLockedRect, LPRECT lpDestRect) const
{
	if (!pLockedRect)
	{
		return DDERR_GENERIC;
	}
	if (!IsUsingEmulation())
	{
		pLockedRect->Pitch = 0;
		pLockedRect->pBits = nullptr;
		return DDERR_GENERIC;
	}

	pLockedRect->Pitch = surface.emu->Pitch;
	pLockedRect->pBits = (lpDestRect) ? (void*)((DWORD)surface.emu->pBits + ((lpDestRect->top * pLockedRect->Pitch) + (lpDestRect->left * (surface.BitCount / 8)))) : surface.emu->pBits;

	return DD_OK;
}

void m_IDirectDrawSurfaceX::PrepareRenderTarget()
{
	if (surface.UsingShadowSurface && surface.Shadow)
	{
		if (SUCCEEDED((*d3d9Device)->UpdateSurface(surface.Shadow, nullptr, Get3DSurface(), nullptr)))
		{
			surface.UsingShadowSurface = false;
			return;
		}
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to update render target!");
	}
}

void m_IDirectDrawSurfaceX::SetRenderTargetShadow()
{
	if (!surface.UsingShadowSurface)
	{
		// Create shadow surface
		if (!surface.Shadow && (surface.Surface || surface.Texture) && (surface.Usage & D3DUSAGE_RENDERTARGET))
		{
			D3DSURFACE_DESC Desc;
			if (FAILED(surface.Surface ? surface.Surface->GetDesc(&Desc) : surface.Texture->GetLevelDesc(0, &Desc)) ||
				FAILED((*d3d9Device)->CreateOffscreenPlainSurface(Desc.Width, Desc.Height, Desc.Format, D3DPOOL_SYSTEMMEM, &surface.Shadow, nullptr)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create shadow surface. Size: " << surface.Width << "x" << surface.Height << " Format: " << surface.Format << " dwCaps: " << surfaceDesc2.ddsCaps);
				return;
			}
		}
		if (surface.Shadow)
		{
			if (SUCCEEDED((*d3d9Device)->GetRenderTargetData(Get3DSurface(), surface.Shadow)))
			{
				surface.UsingShadowSurface = true;
				return;
			}
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get render target data!");
		}
	}
}

void m_IDirectDrawSurfaceX::CaptureForPhaseA7FirstBind()
{
	if (!Config.DdrawContentCapture || !IsSurfaceTexture() || !surface.Texture) return;
	// Pass `this` (the wrapper pointer) so TEX_HASH log matches DRAW_XYZRHW tex= field
	CaptureSurfaceForPhaseA7FirstBind(surface.Texture, this);
}

m_IDirectDrawSurfaceX* m_IDirectDrawSurfaceX::GetCanonicalForPathB()
{
	if (!Config.DdrawCollapseAnimationPools) return this;
	auto it = memberToCanonical.find(this);
	if (it == memberToCanonical.end()) return this;
	poolRedirectsTotal++;
	return reinterpret_cast<m_IDirectDrawSurfaceX*>(it->second);
}

bool m_IDirectDrawSurfaceX::TryGetSubTextureForUV(float u_min, float v_min, float u_max, float v_max,
	IDirect3DTexture9*& subTexOut,
	float& regionU0Out, float& regionV0Out,
	float& regionDuOut, float& regionDvOut,
	int& regionIdxOut)
{
	subTexOut = nullptr; regionIdxOut = -1;
	if (!Config.DdrawAtlasDecompose) return false;
	if (!IsSurfaceTexture() || !surface.Texture) return false;

	// Resolve the region set for this atlas. Two paths:
	// (1) Fast path: exact content-hash lookup. Works for atlases with stable
	//     surface bytes across sessions. Regions are bare UVRects (no source PNG).
	// (2) Fallback: visual fingerprint match. Compute 32x32 grayscale once per
	//     wrapper, L2-match against known family fingerprints, cache. Works for
	//     pool atlases whose hashes drift but visual content is stable. Regions
	//     are AtlasRegion (UV + optional source_png override for clean-source
	//     synthesis -- see SOURCE_PNG path in the synthesis branch below).
	const UVRect* hashRegions = nullptr;
	const AtlasRegion* familyRegions = nullptr;
	size_t numRegions = 0;
	bool fromFingerprint = false;

	auto hashIt = capState.firstHashByTex.find(this);
	if (hashIt != capState.firstHashByTex.end())
	{
		auto regIt = kAtlasRegions.find(hashIt->second);
		if (regIt != kAtlasRegions.end())
		{
			hashRegions = regIt->second.data();
			numRegions = regIt->second.size();
		}
	}

	if (!hashRegions)
	{
		// Fingerprint fallback.
		auto fpIt = wrapperToFamilyIdx.find((void*)this);
		int famIdx;
		if (fpIt == wrapperToFamilyIdx.end())
		{
			uint8_t fp[1024];
			if (!ComputeFingerprintFromTexture(surface.Texture, d3d9Device, fp))
			{
				wrapperToFamilyIdx[(void*)this] = FP_NO_MATCH;
				return false;
			}
			float bestL2 = 0.0f;
			famIdx = FindFamilyForFingerprint(fp, &bestL2);
			wrapperToFamilyIdx[(void*)this] = famIdx;
			char buf[160];
			if (famIdx >= 0)
			{
				sprintf_s(buf, sizeof(buf), "[A.10 FP] match wrapper=%p family=%s L2=%.1f",
					(void*)this, kAtlasFamilies[famIdx].name, bestL2);
				fingerprintMatchTotal++;
			}
			else
			{
				sprintf_s(buf, sizeof(buf), "[A.10 FP] no-match wrapper=%p best_L2=%.1f (threshold=%.0f)",
					(void*)this, bestL2, kFingerprintL2Threshold);
			}
			Logging::Log() << buf;
		}
		else
		{
			famIdx = fpIt->second;
		}
		if (famIdx < 0) return false;
		familyRegions = kAtlasFamilies[famIdx].regions;
		numRegions = kAtlasFamilies[famIdx].region_count;
		fromFingerprint = true;
	}

	// First region whose rect contains the drawcall's UV bbox wins. Family
	// regions are sorted smallest-first so the most-specific region matches
	// (e.g. a 25%x25% subdivision wins over a 50%x50% containing one).
	// Drawcalls crossing region boundaries fall through -- caller binds atlas.
	for (size_t i = 0; i < numRegions; ++i)
	{
		const UVRect& rect = familyRegions ? familyRegions[i].uv : hashRegions[i];
		const char* source_png = familyRegions ? familyRegions[i].source_png : nullptr;
		if (!UVRectContains(rect, u_min, v_min, u_max, v_max)) continue;

		SubTexKey key{ (void*)this, (int)i };
		auto cacheIt = subTextureCache.find(key);
		IDirect3DTexture9* sub = (cacheIt != subTextureCache.end()) ? cacheIt->second : nullptr;

		// Path C: for fingerprint-matched atlases, redirect to canonical
		// sub-texture for this (family, region) pair so Remix sees ONE hash
		// per region across all N pool variants of this family.
		int currentFamIdx = -1;
		if (!sub && fromFingerprint)
		{
			currentFamIdx = wrapperToFamilyIdx[(void*)this];
			uint64_t crKey = MakeFamilyRegionKey(currentFamIdx, (int)i);
			auto crIt = canonicalByFamilyRegion.find(crKey);
			if (crIt != canonicalByFamilyRegion.end())
			{
				sub = crIt->second;
				subTextureCache[key] = sub;  // also wire per-wrapper cache for fast hit next time
				pathCRedirectsTotal++;
				char buf[200];
				sprintf_s(buf, sizeof(buf), "[A.10 PathC] redirect atlas=%p family=%s region=%d -> canonical=%p",
					(void*)this, kAtlasFamilies[currentFamIdx].name, (int)i, (void*)sub);
				Logging::Log() << buf;
			}
		}

		if (!sub)
		{
			// Lazy-create. Target dimensions match the atlas so Remix sees the
			// natural-resolution staging buffer; the cropped pixels just get
			// stretched. (Alternative: scale to region size for memory savings;
			// 128x128 is cheap enough that we prefer hash stability.)
			if (!d3d9Device || !*d3d9Device) return false;
			const UINT atlasW = surface.Width;
			const UINT atlasH = surface.Height;
			if (atlasW == 0 || atlasH == 0) return false;
			// Levels=0 -> allocate a FULL mip chain. Far/small torches sample reduced
			// LODs; with only mip 0 they render as a solid white block under Remix.
			HRESULT hr = (*d3d9Device)->CreateTexture(atlasW, atlasH, 0, 0,
				D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &sub, nullptr);
			if (FAILED(hr) || !sub) return false;

			IDirect3DSurface9* dstLevel = nullptr;
			if (FAILED(sub->GetSurfaceLevel(0, &dstLevel)) || !dstLevel)
			{
				sub->Release(); return false;
			}

			// SOURCE_PNG path: when the region declares a clean source PNG,
			// load it directly. The PNG bytes are byte-deterministic across
			// sessions/machines -> Remix sees a stable XXH3 hash every run.
			// This is the "1 logical texture per atlas" architecture: bypass
			// the runtime composite atlas entirely and bind a known clean image.
			HRESULT lh = D3DERR_INVALIDCALL;
			bool usedSourcePng = false;
			if (source_png)
			{
				char modulePath[MAX_PATH] = {};
				GetModuleFileNameA(nullptr, modulePath, MAX_PATH);
				std::string gameDir = modulePath;
				auto slash = gameDir.find_last_of("\\/");
				if (slash != std::string::npos) gameDir = gameDir.substr(0, slash);
				std::string fullPath = gameDir + "\\" + source_png;
				lh = D3DXLoadSurfaceFromFileA(dstLevel, nullptr, nullptr,
					fullPath.c_str(), nullptr, D3DX_FILTER_LINEAR, 0, nullptr);
				if (SUCCEEDED(lh))
				{
					usedSourcePng = true;
				}
				else
				{
					Logging::Log() << "[A.10] source PNG load failed (" << fullPath.c_str() << "), falling back to atlas crop. hr=" << (DWORD)lh;
				}
			}

			RECT srcRect = {};
			if (FAILED(lh))
			{
				// Atlas-crop fallback (existing behavior).
				IDirect3DSurface9* srcLevel = nullptr;
				if (FAILED(surface.Texture->GetSurfaceLevel(0, &srcLevel)) || !srcLevel)
				{
					dstLevel->Release(); sub->Release(); return false;
				}
				srcRect.left   = (LONG)(rect.u0 * atlasW);
				srcRect.top    = (LONG)(rect.v0 * atlasH);
				srcRect.right  = (LONG)(rect.u1 * atlasW);
				srcRect.bottom = (LONG)(rect.v1 * atlasH);
				lh = D3DXLoadSurfaceFromSurface(dstLevel, nullptr, nullptr,
					srcLevel, nullptr, &srcRect, D3DX_FILTER_LINEAR, 0);
				srcLevel->Release();
				if (FAILED(lh)) { dstLevel->Release(); sub->Release(); return false; }
			}

			// Generate the canonical's full mip chain from the freshly-loaded mip 0.
			// Static texture (synthesized once, never updated), so this is safe --
			// unlike the animated game surfaces the old CopyToDrawTexture filter touched.
			// Without lower mips, far/small torches (reduced LOD) render as a solid white
			// block under Remix. Mip-0 bytes are unchanged so the Remix XXH3 hash holds.
			if (sub->GetLevelCount() > 1)
			{
				D3DXFilterTexture(sub, nullptr, 0, D3DX_FILTER_LINEAR);
			}

			subTextureCache[key] = sub;

			// Path C: this is the FIRST synthesis for (family, region) -- claim
			// it as canonical so future pool variants of this family redirect to it.
			if (fromFingerprint && currentFamIdx >= 0)
			{
				uint64_t crKey = MakeFamilyRegionKey(currentFamIdx, (int)i);
				canonicalByFamilyRegion[crKey] = sub;
				pathCCanonicalsCreated++;
			}

			{
				char buf[280];
				if (usedSourcePng)
				{
					sprintf_s(buf, sizeof(buf), "[A.10] synthesized sub-texture atlas=%p key=family:%s region=%d FROM_PNG=%s canonical",
						(void*)this, kAtlasFamilies[currentFamIdx].name, (int)i, source_png);
				}
				else if (fromFingerprint)
				{
					sprintf_s(buf, sizeof(buf), "[A.10] synthesized sub-texture atlas=%p key=family:%s region=%d src=[%ld,%ld,%ld,%ld] canonical (atlas-crop)",
						(void*)this, kAtlasFamilies[currentFamIdx].name, (int)i,
						srcRect.left, srcRect.top, srcRect.right, srcRect.bottom);
				}
				else
				{
					sprintf_s(buf, sizeof(buf), "[A.10] synthesized sub-texture atlas=%p key=hash:%016llX region=%d src=[%ld,%ld,%ld,%ld]",
						(void*)this, (unsigned long long)hashIt->second, (int)i,
						srcRect.left, srcRect.top, srcRect.right, srcRect.bottom);
				}
				Logging::Log() << buf;
			}

			// Capture the synthesized content as a PNG + emit a TEX_HASH line keyed
			// by the sub-texture pointer. This is the only chance to record it --
			// the SetDirtyFlag / first-bind hooks both gate on m_IDirectDrawSurfaceX
			// wrappers, and synthesized sub-textures bypass that path entirely.
			// The "wrapper" key we pass is the sub-texture pointer itself; it's
			// stable for the cache lifetime and unique per (atlas, region).
			if (Config.DdrawContentCapture)
			{
				CaptureSurfaceContent(dstLevel, (void*)sub, /*isCrop=*/true);
			}
			dstLevel->Release();
		}
		subTexOut = sub;
		regionU0Out = rect.u0;
		regionV0Out = rect.v0;
		regionDuOut = rect.u1 - rect.u0;
		regionDvOut = rect.v1 - rect.v0;
		regionIdxOut = (int)i;
		atlasDecomposeBindsTotal++;
		return true;
	}
	return false;
}

// Dynamic-UI solo page check (map_texture etc.): lets the device draw sites apply
// the emissive blend override to FULL-page draws, which never enter the crop layer
// where per-placement classification happens.
bool m_IDirectDrawSurfaceX::IsNamekeyDynamicUiPage()
{
	if (!Config.DdrawNameKey || !surface.Texture) return false;
	auto it = namekeyRecipeByTex.find(surface.Texture);
	return it != namekeyRecipeByTex.end() && it->second.dynUiSolo;
}

// ===== [BLITQUAD] menu-text fix (2026-06-12) =====
// MENUDIAG proved the front end composites its UI -- ALL the menu text -- via 2D
// Blts straight onto the backbuffer (2013 backbuffer Blts per menu visit, color-
// keyed sprite layers at screen coords). The path tracer discards the rasterized
// frame, so that UI can never appear. We queue a texture copy of every such Blt;
// the device re-emits them at EndScene as XYZRHW quads through the normal draw
// path, where the orphan-overlay lift renders them near-depth + additive.
namespace
{
	struct MenuBlitOverlayEntry { IDirect3DTexture9* tex = nullptr; RECT dest = {}; };
	// v10: PERSISTENT overlay store keyed by dest screen region. One-frame display
	// caused mouse-move strobing (DK2 repaints regions on demand, not per frame);
	// entries now persist until the game repaints (upsert) or erases (ColorFill).
	std::unordered_map<uint64_t, MenuBlitOverlayEntry> g_menuBlitPersist;

	uint64_t MenuBlitDestKey(const RECT& r)
	{
		return ((uint64_t)(uint16_t)r.left << 48) | ((uint64_t)(uint16_t)r.top << 32) |
			((uint64_t)(uint16_t)r.right << 16) | (uint64_t)(uint16_t)r.bottom;
	}
	struct MenuBlitTexSlot { IDirect3DTexture9* tex = nullptr; LONG w = 0, h = 0; };
	std::unordered_map<uint64_t, MenuBlitTexSlot> g_menuBlitTexCache;
	constexpr size_t kMenuBlitQueueCap = 1024;
	bool g_menuBlitDrawActive = false;
	IDirect3DTexture9* g_menuBlitCurrentTex = nullptr;
	bool g_menuBlitBackbufferDirty = false;
	m_IDirectDrawSurfaceX* g_menuBlitLockedSurf = nullptr;

	// v11: OUR OWN 2D CANVAS. Full-frame snapshots were atomic -- the game's sub-rect
	// fill erases either killed the whole entry (flash) or were ignored (stale menus).
	// The canvas mirrors the game's screen 2D state region-by-region: lock-writes
	// paint pixels in, ColorFills erase regions, blit dests clear under their entry.
	// Displayed as ONE persistent quad, always complete.
	std::vector<DWORD> g_menuCanvas;       // ARGB, screen-size; alpha 0 = empty
	LONG g_menuCanvasW = 0, g_menuCanvasH = 0;
	RECT g_menuCanvasDirty = {};
	bool g_menuCanvasHasDirty = false;
	IDirect3DTexture9* g_menuCanvasTex = nullptr;

	DWORD NativePixelToARGB(D3DFORMAT fmt, const BYTE* p)
	{
		switch (fmt)
		{
		case D3DFMT_R5G6B5:
		{ const DWORD v = *(const WORD*)p; const DWORD r = (v >> 11) & 0x1F, g = (v >> 5) & 0x3F, b = v & 0x1F;
		  return 0xFF000000 | ((r * 255 / 31) << 16) | ((g * 255 / 63) << 8) | (b * 255 / 31); }
		case D3DFMT_X1R5G5B5: case D3DFMT_A1R5G5B5:
		{ const DWORD v = *(const WORD*)p; const DWORD r = (v >> 10) & 0x1F, g = (v >> 5) & 0x1F, b = v & 0x1F;
		  return 0xFF000000 | ((r * 255 / 31) << 16) | ((g * 255 / 31) << 8) | (b * 255 / 31); }
		case D3DFMT_A4R4G4B4: case D3DFMT_X4R4G4B4:
		{ const DWORD v = *(const WORD*)p; const DWORD r = (v >> 8) & 0xF, g = (v >> 4) & 0xF, b = v & 0xF;
		  return 0xFF000000 | ((r * 17) << 16) | ((g * 17) << 8) | (b * 17); }
		case D3DFMT_A8R8G8B8: case D3DFMT_X8R8G8B8:
			return 0xFF000000 | (*(const DWORD*)p & 0xFFFFFF);
		default:
			return 0;
		}
	}

	void MenuCanvasAddDirty(const RECT& r)
	{
		if (!g_menuCanvasHasDirty) { g_menuCanvasDirty = r; g_menuCanvasHasDirty = true; return; }
		g_menuCanvasDirty.left = min(g_menuCanvasDirty.left, r.left);
		g_menuCanvasDirty.top = min(g_menuCanvasDirty.top, r.top);
		g_menuCanvasDirty.right = max(g_menuCanvasDirty.right, r.right);
		g_menuCanvasDirty.bottom = max(g_menuCanvasDirty.bottom, r.bottom);
	}

	bool MenuCanvasEnsure(LONG w, LONG h, LPDIRECT3DDEVICE9 dev)
	{
		if (w <= 0 || h <= 0) return false;
		if (g_menuCanvasW != w || g_menuCanvasH != h)
		{
			g_menuCanvas.assign((size_t)w * h, 0u);
			g_menuCanvasW = w; g_menuCanvasH = h;
			if (g_menuCanvasTex) { g_menuCanvasTex->Release(); g_menuCanvasTex = nullptr; }
		}
		if (!g_menuCanvasTex && dev)
		{
			if (FAILED(dev->CreateTexture(w, h, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &g_menuCanvasTex, nullptr)))
			{
				g_menuCanvasTex = nullptr;
				return false;
			}
			RECT full = { 0, 0, w, h };
			MenuCanvasAddDirty(full);
		}
		return g_menuCanvasTex != nullptr;
	}

	// Clip r to the canvas; returns false if empty.
	bool MenuCanvasClip(RECT& r)
	{
		r.left = max(0L, r.left); r.top = max(0L, r.top);
		r.right = min(g_menuCanvasW, r.right); r.bottom = min(g_menuCanvasH, r.bottom);
		return (r.right > r.left && r.bottom > r.top);
	}

	// Region erase (game ColorFill or art blitted over): black/zero fill -> alpha 0.
	void MenuCanvasFill(const RECT& rin, DWORD argb)
	{
		if (g_menuCanvas.empty()) return;
		RECT r = rin;
		if (!MenuCanvasClip(r)) return;
		for (LONG y = r.top; y < r.bottom; ++y)
		{
			DWORD* row = g_menuCanvas.data() + (size_t)y * g_menuCanvasW;
			for (LONG x = r.left; x < r.right; ++x) row[x] = argb;
		}
		MenuCanvasAddDirty(r);
	}

	// Expand a native-format color key to the ARGB D3DX wants (key pixels -> alpha 0).
	D3DCOLOR NativeKeyToARGB(D3DFORMAT fmt, DWORD key)
	{
		switch (fmt)
		{
		case D3DFMT_R5G6B5:
		{ DWORD r = (key >> 11) & 0x1F, g = (key >> 5) & 0x3F, b = key & 0x1F;
		  return 0xFF000000 | ((r * 255 / 31) << 16) | ((g * 255 / 63) << 8) | (b * 255 / 31); }
		case D3DFMT_X1R5G5B5: case D3DFMT_A1R5G5B5:
		{ DWORD r = (key >> 10) & 0x1F, g = (key >> 5) & 0x1F, b = key & 0x1F;
		  return 0xFF000000 | ((r * 255 / 31) << 16) | ((g * 255 / 31) << 8) | (b * 255 / 31); }
		case D3DFMT_A4R4G4B4: case D3DFMT_X4R4G4B4:
		{ DWORD r = (key >> 8) & 0xF, g = (key >> 4) & 0xF, b = key & 0xF;
		  return 0xFF000000 | ((r * 17) << 16) | ((g * 17) << 8) | (b * 17); }
		case D3DFMT_A8R8G8B8: case D3DFMT_X8R8G8B8:
			return 0xFF000000 | (key & 0xFFFFFF);
		default:
			return 0;
		}
	}
}

// Lock-site hook (defined here so the anonymous-namespace globals are in scope at
// the call through this function regardless of where Lock sits in the file).
void MenuBlitOverlayMarkLockDirty(m_IDirectDrawSurfaceX* pSurf)
{
	g_menuBlitBackbufferDirty = true;
	g_menuBlitLockedSurf = pSurf;
}

// [BLITQUAD] v10.3: drop persistent entries that INTERSECT a newly written region --
// newer screen content supersedes. (Without this, cursor-area repaints at slightly
// shifted rects accumulated as a golden cursor TRAIL.)
void MenuBlitOverlayEraseIntersecting(const RECT& r)
{
	for (auto it = g_menuBlitPersist.begin(); it != g_menuBlitPersist.end(); )
	{
		const RECT& d = it->second.dest;
		const bool intersects = !(d.right <= r.left || d.left >= r.right ||
			d.bottom <= r.top || d.top >= r.bottom);
		if (intersects) it = g_menuBlitPersist.erase(it);
		else ++it;
	}
}

// [BLITQUAD] v11: a fill on EITHER screen surface is the game erasing that region
// of its 2D layer -- mirror it into the canvas exactly. (The atomic-snapshot dilemma
// of v10.x -- "erase kills everything" vs "erase ignores everything" -- is gone:
// the canvas is region-accurate by construction.)
void MenuBlitOverlayEraseRegion(LPRECT lpRect, LONG surfW, LONG surfH)
{
	if (g_menuCanvas.empty()) return;
	RECT r = {};
	if (lpRect) r = *lpRect;
	else { r.right = surfW; r.bottom = surfH; }
	MenuCanvasFill(r, 0u);
}

bool MenuBlitOverlayDrawActive() { return g_menuBlitDrawActive; }

// The device draw path rebinds textures from the game's CurrentTexture state on
// EVERY draw (SetDrawStates texture loop), which stomps any direct d3d9 SetTexture.
// The drain publishes its per-quad texture here; the draw sites bind it AFTER
// SetDrawStates -- the same after-the-rebind slot the atlas-decompose bind uses.
IDirect3DTexture9* MenuBlitOverlayCurrentTex() { return g_menuBlitDrawActive ? g_menuBlitCurrentTex : nullptr; }

void m_IDirectDrawSurfaceX::MenuBlitOverlayQueue(m_IDirectDrawSurfaceX* pSrc, LPRECT lpSrcRect, LPRECT lpDestRect, bool hasSrcKey, DWORD nativeSrcKey)
{
	if (!pSrc || !d3d9Device || !*d3d9Device) return;
	if (!MenuCanvasEnsure((LONG)surface.Width, (LONG)surface.Height, *d3d9Device)) return;

	RECT sr = {};
	if (lpSrcRect) sr = *lpSrcRect;
	else { sr.right = (LONG)pSrc->surface.Width; sr.bottom = (LONG)pSrc->surface.Height; }
	RECT dr = {};
	if (lpDestRect) dr = *lpDestRect;
	else { dr.right = (LONG)surface.Width; dr.bottom = (LONG)surface.Height; }
	const LONG dw = dr.right - dr.left, dh = dr.bottom - dr.top;
	if (dw <= 0 || dh <= 0 || sr.right <= sr.left || sr.bottom <= sr.top) return;

	IDirect3DSurface9* srcD9 = nullptr;
	bool releaseSrc = false;
	if (pSrc->surface.Texture)
	{
		if (FAILED(pSrc->surface.Texture->GetSurfaceLevel(0, &srcD9)) || !srcD9) return;
		releaseSrc = true;
	}
	else if (pSrc->surface.Surface)
	{
		srcD9 = pSrc->surface.Surface;
	}
	if (!srcD9)
	{
		LOG_LIMIT(100, "[BLITQUAD] skip: src " << pSrc << " has no d3d9 object (emulated-only)");
		return;
	}

	// v11: composite the blit into OUR canvas (CPU pixels via a reusable sysmem
	// staging surface; D3DX handles stretch + color-key->alpha). Keyed pixels leave
	// the canvas untouched; opaque pixels overwrite (black = erase under additive).
	static IDirect3DSurface9* stage = nullptr;
	static LONG stageW = 0, stageH = 0;
	if (stage && (stageW < dw || stageH < dh)) { stage->Release(); stage = nullptr; }
	if (!stage)
	{
		const LONG nw = max(dw, stageW), nh = max(dh, stageH);
		if (FAILED((*d3d9Device)->CreateOffscreenPlainSurface(nw, nh, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &stage, nullptr)) || !stage)
		{
			stage = nullptr;
			if (releaseSrc) srcD9->Release();
			return;
		}
		stageW = nw; stageH = nh;
	}
	RECT str = { 0, 0, dw, dh };
	const D3DCOLOR ck = hasSrcKey ? NativeKeyToARGB(pSrc->surface.Format, nativeSrcKey) : 0;
	HRESULT lh = D3DXLoadSurfaceFromSurface(stage, nullptr, &str, srcD9, nullptr, &sr, D3DX_FILTER_POINT, ck);
	if (releaseSrc) srcD9->Release();
	if (FAILED(lh))
	{
		LOG_LIMIT(100, "[BLITQUAD] stage load failed src=" << pSrc);
		return;
	}

	D3DLOCKED_RECT lrk;
	if (FAILED(stage->LockRect(&lrk, &str, D3DLOCK_READONLY))) return;
	RECT cd = dr;
	if (MenuCanvasClip(cd))
	{
		for (LONG y = cd.top; y < cd.bottom; ++y)
		{
			const DWORD* srow = (const DWORD*)((const BYTE*)lrk.pBits + (size_t)(y - dr.top) * lrk.Pitch);
			DWORD* drow = g_menuCanvas.data() + (size_t)y * g_menuCanvasW;
			for (LONG x = cd.left; x < cd.right; ++x)
			{
				const DWORD px = srow[x - dr.left];
				// Non-black pixels composite; BLACK NEVER ERASES via blits. The game's
				// cursor save/restore round-trips through the d3d9 backbuffer, which
				// under Remix does NOT retain the CPU composite -- the "saved" content
				// reads back black, and restoring it was wiping canvas text wherever
				// the cursor passed (permanently). Only explicit ColorFills erase.
				if ((px & 0xFF000000) && (px & 0xFFFFFF)) drow[x] = px;
			}
		}
		MenuCanvasAddDirty(cd);
	}
	stage->UnlockRect();
}

// [BLITQUAD] v3: capture written lock bits while the d3d9 mapping is valid (called
// from Unlock). One queued quad per (surface, rect) per frame -- repeated unlocks of
// the same region would stack additively and over-brighten.
void m_IDirectDrawSurfaceX::MenuBlitOverlayCaptureLockedBits(void* pBits, INT pitch, const RECT& lockedRect)
{
	if (!d3d9Device || !*d3d9Device || !pBits) return;

	// v11: write the locked region into OUR canvas -- region-accurate semantics
	// replace the atomic full-frame snapshots that caused both the flash (erase
	// kills everything) and the stale-menu overlap (erase ignores everything).
	if (!MenuCanvasEnsure((LONG)surface.Width, (LONG)surface.Height, *d3d9Device)) return;
	RECT r = lockedRect;
	if (!MenuCanvasClip(r)) return;
	const UINT bpp = (surface.Format == D3DFMT_X8R8G8B8 || surface.Format == D3DFMT_A8R8G8B8) ? 4 : 2;
	for (LONG y = r.top; y < r.bottom; ++y)
	{
		// pBits addresses the LOCKED rect's origin
		const BYTE* srcRow = (const BYTE*)pBits + (size_t)(y - lockedRect.top) * pitch;
		DWORD* dstRow = g_menuCanvas.data() + (size_t)y * g_menuCanvasW;
		for (LONG x = r.left; x < r.right; ++x)
		{
			const DWORD argb = NativePixelToARGB(surface.Format, srcRow + (size_t)(x - lockedRect.left) * bpp);
			// BLACK NEVER ERASES here either: the d3d9 lock mapping does not retain
			// the previous composite under Remix, so unwritten regions of a full-
			// surface lock read back black -- writing that would erase every other
			// region's text. Erasure is exclusively the game's explicit ColorFills.
			if (argb & 0xFFFFFF) dstRow[x] = argb;
		}
	}
	MenuCanvasAddDirty(r);
	LOG_LIMIT(2000, "[BLITQUAD] canvas write " << (r.right - r.left) << "x" << (r.bottom - r.top) << " at {" << r.left << "," << r.top << "}");
}

// [BLITQUAD] v2 (superseded by v3 unlock-time capture; the backbuffer turned out not
// to be emulated, so there is no sysmem mirror to read). Kept as a stub so the header
// declaration stays valid.
void m_IDirectDrawSurfaceX::MenuBlitOverlayCaptureLockedFrame()
{
}

// Drain at EndScene: draw each queued overlay as a screen-space quad through the
// device's normal draw path. g_menuBlitDrawActive makes the orphan-overlay lift
// accept these draws (near depth + additive) regardless of bound wrapper texture.
void MenuBlitOverlayDrain(m_IDirect3DDeviceX* pDevice, LPDIRECT3DDEVICE9 dev)
{
	// v11: ONE canvas quad. Upload the dirty region, then draw the always-complete
	// 2D layer through the device's own draw path (lift -> near depth + additive).
	g_menuBlitBackbufferDirty = false;

	if (!g_menuCanvasTex || !pDevice || !dev) return;

	if (g_menuCanvasHasDirty && !g_menuCanvas.empty())
	{
		RECT d = g_menuCanvasDirty;
		if (MenuCanvasClip(d))
		{
			D3DLOCKED_RECT lr;
			if (SUCCEEDED(g_menuCanvasTex->LockRect(0, &lr, &d, 0)))
			{
				for (LONG y = d.top; y < d.bottom; ++y)
				{
					memcpy((BYTE*)lr.pBits + (size_t)(y - d.top) * lr.Pitch,
						g_menuCanvas.data() + (size_t)y * g_menuCanvasW + d.left,
						(size_t)(d.right - d.left) * 4);
				}
				g_menuCanvasTex->UnlockRect(0);
			}
		}
		g_menuCanvasHasDirty = false;
	}

	g_menuBlitDrawActive = true;
	// Published for the device draw site to bind AFTER its texture-rebind loop
	// (a direct SetTexture here would be stomped by SetDrawStates).
	g_menuBlitCurrentTex = g_menuCanvasTex;
	struct V { float x, y, z, rhw, u, v; };
	const float r = (float)g_menuCanvasW, b = (float)g_menuCanvasH;
	V quad[4] =
	{
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f },
		{ r,    0.0f, 0.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, b,    0.0f, 1.0f, 0.0f, 1.0f },
		{ r,    b,    0.0f, 1.0f, 1.0f, 1.0f },
	};
	pDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, D3DFVF_XYZRHW | D3DFVF_TEX1, quad, 4, 0, 7);
	g_menuBlitDrawActive = false;
	g_menuBlitCurrentTex = nullptr;
	// No texture restore needed: the next real draw's SetDrawStates rebinds the
	// game's CurrentTexture state (the same loop that made the restore necessary).

	LOG_LIMIT(2000, "[BLITQUAD] canvas quad drawn " << g_menuCanvasW << "x" << g_menuCanvasH);
}

// Orphan-overlay lift classification (2026-06-12, menu-text fix): a page with NO
// recipe never received content from the tracked EngineTextures upload blit -- its
// pixels were runtime-rendered (MyTextRenderer glyph composites = menu text,
// loading-screen elements). See the device draw sites for how this is used.
bool m_IDirectDrawSurfaceX::IsNamekeyOrphanPage()
{
	if (!Config.DdrawNameKey || !surface.Texture) return false;
	return namekeyRecipeByTex.find(surface.Texture) == namekeyRecipeByTex.end();
}

// [LIFT] diagnostics: brief recipe description for the device-site telemetry.
void m_IDirectDrawSurfaceX::GetNamekeyRecipeBrief(char* buf, size_t bufSize)
{
	if (!buf || bufSize == 0) return;
	buf[0] = '\0';
	if (!Config.DdrawNameKey || !surface.Texture) { strcpy_s(buf, bufSize, "notex"); return; }
	auto it = namekeyRecipeByTex.find(surface.Texture);
	if (it == namekeyRecipeByTex.end()) { strcpy_s(buf, bufSize, "none"); return; }
	char names[192]; names[0] = '\0';
	uint32_t seen[3] = { 0, 0, 0 };
	int distinct = 0;
	for (const NamekeyPlacement& q : it->second.comps)
	{
		bool dup = false;
		for (int i = 0; i < distinct && i < 3; ++i) { if (seen[i] == q.idx) { dup = true; break; } }
		if (dup) continue;
		if (distinct < 3)
		{
			seen[distinct] = q.idx;
			const char* nm = NamekeyNameForIdx(q.idx);
			if (distinct) strcat_s(names, sizeof(names), "+");
			strcat_s(names, sizeof(names), nm ? nm : "?");
		}
		++distinct;
	}
	sprintf_s(buf, bufSize, "%s(distinct=%d,placements=%u)", names, distinct, (unsigned)it->second.comps.size());
}

// Gate-free water classification (2026-06-11): large water bodies are 34-340 vert
// meshes (run-11 DRAW_XYZRHW measurement) whose UV bbox sits exactly inside ONE
// Water placement -- they fail the universal-decompose caller gates (vcount<=32,
// UV area<0.5), so the crop layer's placement classification never saw them and
// the flatten missed every large body while puddles stood still. This is the
// classification ALONE: recipe lookup + snapped-texel containment, read-only --
// no crop, no UV rewrite (the 115-vert Remix AV history applies to REWRITING big
// meshes, not to reading them). Full-page draws on co-atlased pages span multiple
// placements -> not contained -> correctly excluded (flattening those would
// flatten torch billboards sharing the page).
bool m_IDirectDrawSurfaceX::IsNamekeyWaterDraw(float u_min, float v_min, float u_max, float v_max)
{
	if (!Config.DdrawNameKey || !surface.Texture) return false;
	auto it = namekeyRecipeByTex.find(surface.Texture);
	if (it == namekeyRecipeByTex.end() || !it->second.hasWater) return false;
	const UINT atlasW = surface.Width;
	const UINT atlasH = surface.Height;
	if (atlasW == 0 || atlasH == 0) return false;
	// Same texel snapping as the crop layer; deliberately NOT clamped to the page:
	// a tiling/wrapping bbox must fail containment rather than be clamped into it.
	const long px0 = (long)floorf(u_min * atlasW);
	const long py0 = (long)floorf(v_min * atlasH);
	const long px1 = (long)ceilf(u_max * atlasW);
	const long py1 = (long)ceilf(v_max * atlasH);
	for (const NamekeyPlacement& q : it->second.comps)
	{
		if (!q.isWater) continue;
		// Containment with 1-texel slack, mirroring the crop layer's placement match.
		if (px0 >= (long)q.a2 - 1 && py0 >= (long)q.a3 - 1 &&
			px1 <= (long)(q.a2 + q.sw) + 1 && py1 <= (long)(q.a3 + q.sh) + 1)
		{
			namekeyWaterLarge++;
			return true;
		}
	}
	return false;
}

// Universal UV-region decomposition (2026-05-21). See the cache declarations + the
// dk2_universal_decompose_plan memory for the design. Crops the drawcall's exact
// sampled texel rect into a content-hash-keyed sub-texture and returns the [0,1]
// transform. No fingerprints, no hardcoded hashes, no source PNGs.
bool m_IDirectDrawSurfaceX::TryGetUniversalSubTextureForUV(float u_min, float v_min, float u_max, float v_max,
	IDirect3DTexture9*& subTexOut,
	float& regionU0Out, float& regionV0Out,
	float& regionDuOut, float& regionDvOut,
	int& regionIdxOut)
{
	subTexOut = nullptr; regionIdxOut = -1;
	if (!Config.DdrawUniversalDecompose) return false;
	if (!IsSurfaceTexture() || !surface.Texture) return false;
	if (!d3d9Device || !*d3d9Device) return false;
	// Streaming-buffer guard (Phase 4, 2026-05-24): if this surface has been classified
	// as a streaming buffer (high crop rate, mostly-unique rects = engine constantly
	// writing new content into it), bail immediately. The caller falls back to using the
	// whole surface texture. This prevents the camera-pan-in-dense-scene perf cliff where
	// one streaming surface generates ~200 unique rects/sec and floods Remix with churn.
	if (surface.UnivSkipDecompose) { stage3PerSurface[(void*)this].blacklistSkips++; return false; }

	const UINT atlasW = surface.Width;
	const UINT atlasH = surface.Height;
	if (atlasW == 0 || atlasH == 0) return false;

	// Snap the UV bbox to integer texel boundaries: floor the min, ceil the max so
	// the cropped rect fully covers the sampled region. The snapped rect (in UV
	// space) is the transform we hand back for the downstream [0,1] UV rewrite.
	long px0 = (long)floorf(u_min * atlasW);
	long py0 = (long)floorf(v_min * atlasH);
	long px1 = (long)ceilf(u_max * atlasW);
	long py1 = (long)ceilf(v_max * atlasH);
	if (px0 < 0) px0 = 0;
	if (py0 < 0) py0 = 0;
	if (px1 > (long)atlasW) px1 = (long)atlasW;
	if (py1 > (long)atlasH) py1 = (long)atlasH;
	const long rw = px1 - px0;
	const long rh = py1 - py0;
	if (rw <= 0 || rh <= 0) return false;
	// If snapping pushed the rect to the whole texture, this isn't a proper sub-rect.
	if ((UINT)rw >= atlasW && (UINT)rh >= atlasH) return false;

	const float su0 = (float)px0 / (float)atlasW;
	const float sv0 = (float)py0 / (float)atlasH;
	const float su1 = (float)px1 / (float)atlasW;
	const float sv1 = (float)py1 / (float)atlasH;

	// (1) Per-(wrapper, quantized rect) fast path -- O(1), no crop/hash.
	// Gen-mismatch gate (Phase 5, 2026-05-24): RepurposeGen on the surface is bumped
	// by SetDirtyFlag ONLY when a Blt follows a long idle gap (sprite-repurpose
	// signature). Animations Blt continuously (small gaps) -> gen never bumps -> cache
	// stays valid -> torch-flicker fix preserved. Repurposes (idle then Blt) -> gen
	// bumps -> entries cached under old gen invalidate on this bind -> fresh crop from
	// current content. Replaces the cruder Phase 2/3 time-since-cache check, which fired
	// spuriously on long-running animations and missed fast-cadence repurposes.
	UnivDrawKey dkey{ (void*)this, QuantizeUV(su0), QuantizeUV(sv0), QuantizeUV(su1), QuantizeUV(sv1) };
	auto dIt = univDrawCache.find(dkey);
	if (dIt != univDrawCache.end())
	{
		bool isStale = (dIt->second.cachedAtRepurposeGen != surface.RepurposeGen);
		// Phase 7 (2026-05-24): per-Blt rect overlap gate. Even if RepurposeGen matches
		// (surface hasn't been idle-then-Blt'd), check whether any Blt since this entry
		// was cached touched the pixel rect this entry covers. If so, the cached pixels
		// no longer match what the source surface holds at that rect -> invalidate. This
		// catches the "continuously-Blt'd shared atlas" case where the per-surface gate
		// can't fire (no idle gap) but per-rect content has changed.
		if (!isStale)
		{
			const UnivSubTex& e = dIt->second;
			for (int i = 0; i < kBltHistorySize; ++i)
			{
				const auto& h = BltHistory[i];
				if (h.timeMs > e.cachedAtTimeMs)
				{
					// Standard 2D AABB intersection test
					if (h.x0 < e.px1 && h.x1 > e.px0 && h.y0 < e.py1 && h.y1 > e.py0)
					{
						isStale = true;
						break;
					}
				}
			}
		}
		if (isStale)
		{
			univLru.erase(dIt->second.lruIt);
			if (dIt->second.tex) dIt->second.tex->Release();
			univDrawCache.erase(dIt);
			univInvalidatesTotal++;
			stage3PerSurface[(void*)this].invals++;
			// fall through to miss path
		}
		else
		{
			// Touch: move to MRU end so frequently-drawn crops (torches) never get
			// evicted while transient map crops age toward the front. Skip splice if
			// already at MRU end -- saves redundant list-pointer writes per frame.
			if (dIt->second.lruIt != std::prev(univLru.end()))
				univLru.splice(univLru.end(), univLru, dIt->second.lruIt);
			subTexOut = dIt->second.tex;
			regionU0Out = dIt->second.u0;
			regionV0Out = dIt->second.v0;
			regionDuOut = dIt->second.u1 - dIt->second.u0;
			regionDvOut = dIt->second.v1 - dIt->second.v0;
			regionIdxOut = 0;
			if (subTexOut) { univDecomposeBindsTotal++; stage3PerSurface[(void*)this].hits++; }
			return subTexOut != nullptr;
		}
	}

	// [NAMEKEY] V2 (2026-06-11): placement-keyed crop resolution. If this page carries a
	// composition recipe (attached by the staging Blt from the 0x58E53B detour's
	// placements) and the draw's snapped texel rect sits inside exactly ONE placement,
	// the draw is sampling THAT sprite: resolve the sub-texture by (nameIdx, mip)
	// instead of by content. The bound crop is built FROM the canonical sidecar payload
	// (mip-0 verbatim -> its Remix hash == XXH3(payload), offline-known for all 5767
	// entries), and the live page rect is nibble-verified against that payload on EVERY
	// cache insert -- a failed verify falls through to the content-correct legacy crop,
	// so neither recipe mis-attribution nor stale recipes can bind wrong art.
	// Serving an already-built crop is a map lookup + small CPU verify (no synthesis),
	// so it sits BEFORE the scarcity gate: during effect bursts, sprites with built
	// crops keep their per-frame replacements. First-time synthesis stays gated below.
	auto NkServeCrop = [&](IDirect3DTexture9* ctex, long cx, long cy, long cw, long ch) -> bool
	{
		ctex->AddRef();	// the univDrawCache entry's reference; invalidate/evict Release() it
		const float pu0 = (float)cx / (float)atlasW;
		const float pv0 = (float)cy / (float)atlasH;
		const float pu1 = (float)(cx + cw) / (float)atlasW;
		const float pv1 = (float)(cy + ch) / (float)atlasH;
		univLru.push_back(dkey);
		auto nkLruIt = std::prev(univLru.end());
		univDrawCache[dkey] = UnivSubTex{ ctex, pu0, pv0, pu1, pv1, nkLruIt, surface.RepurposeGen,
			(LONG)cx, (LONG)cy, (LONG)(cx + cw), (LONG)(cy + ch), GetTickCount() };
		while (univDrawCache.size() > kUnivCacheCap && !univLru.empty())
		{
			const UnivDrawKey oldKey = univLru.front();
			univLru.pop_front();
			auto oIt = univDrawCache.find(oldKey);
			if (oIt != univDrawCache.end())
			{
				if (oIt->second.tex) oIt->second.tex->Release();
				univDrawCache.erase(oIt);
				univEvictTotal++;
			}
		}
		namekeyCropServe++;
		subTexOut = ctex;
		regionU0Out = pu0; regionV0Out = pv0;
		regionDuOut = pu1 - pu0; regionDvOut = pv1 - pv0;
		regionIdxOut = 0;
		univDecomposeBindsTotal++;
		return true;
	};
	const NamekeyPlacement* nkPlace = nullptr;
	if (Config.DdrawNameKey && Config.DdrawNameKeyCrop)
	{
		EnsureCanonSidecarsLoaded();	// no-op once loaded; binds normally run first, but don't depend on it
		auto nrIt = namekeyRecipeByTex.find(surface.Texture);
		if (nrIt != namekeyRecipeByTex.end() && !nrIt->second.comps.empty())
		{
			int contain = 0;
			for (const NamekeyPlacement& q : nrIt->second.comps)
			{
				// Containment with 1-texel slack (floor/ceil snapping vs the game's
				// half-texel-inset sampling); placements never overlap in a bin-pack,
				// so >1 containment means boundary ambiguity or a mixed accumulator.
				if (px0 >= (long)q.a2 - 1 && py0 >= (long)q.a3 - 1 &&
					px1 <= (long)(q.a2 + q.sw) + 1 && py1 <= (long)(q.a3 + q.sh) + 1)
				{
					contain++;
					nkPlace = &q;
				}
			}
			if (contain == 0) { namekeyCropSubmiss++; nkPlace = nullptr; }
			else if (contain > 1) { namekeyCropAmbiguous++; nkPlace = nullptr; }
			else
			{
				namekeyCropMatch++;
				const uint64_t ck = ((uint64_t)nkPlace->idx << 3) | (uint64_t)(nkPlace->mip & 7);
				auto sIt = namekeyCropByKey.find(ck);
				if (sIt != namekeyCropByKey.end())
				{
					NamekeyCropState& st = sIt->second;
					if (st.dynUi)
					{
						g_namekeyDrawDynUiAny = true;
						if (!st.isPointer) g_namekeyDrawDynUi = true;	// pointer renders via rtx.uiTextures, not additive
					}
					if (st.isWater) g_namekeyDrawWater = true;	// device flattens this draw's world heights
					if (st.st == 1 && st.tex && Config.DdrawNameKeyCrop >= 2)
					{
						float bf = 1.0f;
						if (NamekeyCropVerifyRect(surface.Texture, nkPlace->a2, nkPlace->a3, st.payload, st.w, st.h, &bf) == 0)
						{
							return NkServeCrop(st.tex, nkPlace->a2, nkPlace->a3, st.w, st.h);
						}
						// Stale/foreign content at the rect this draw: per-draw failure
						// only (state untouched); the legacy crop below is content-correct.
						namekeyCropServeVerifyFail++;
						nkPlace = nullptr;
					}
					else
					{
						nkPlace = nullptr;	// FAILED (-1) / shadow-logged (2): no re-attempt
					}
				}
				// else: first sight of this (name, mip) -- synthesis attempt below,
				// behind the scarcity gate (sidecar read + texture create is real work).
			}
		}
	}

	// Global rate scarcity gate (Phase 8): we're about to synthesize (cache miss or
	// just-invalidated entry). If the engine is gushing one-shot tiny crops faster
	// than we can absorb them (steam-on-water etc.), refuse new work for the rest of
	// this burst. Cache HITS above this point are unaffected -- already-cached torches
	// keep their replacements. New unique (atlas, rect) keys fall back to whole-surface
	// (caller-side null handling), which Remix replaces from its whole-surface mod
	// entries instead of POT crops. Window=1s, enter=200/s, exit=100/s (hysteresis).
	{
		const DWORD nowMsGate = GetTickCount();
		if (g_globalRateWindowStartMs == 0) g_globalRateWindowStartMs = nowMsGate;
		if (nowMsGate - g_globalRateWindowStartMs >= 1000)
		{
			const DWORD lastWindowRate = g_globalRateCropsInWindow;
			if (!g_globalScarcityMode && lastWindowRate >= kGlobalRateScarcityEnterPerSec)
			{
				g_globalScarcityMode = true;
				univScarcityEntersTotal++;
				char sbuf[220];
				sprintf_s(sbuf, sizeof(sbuf), "[UNIV] SCARCITY ENTER rate=%lu/s (enter>=%lu/s) -- new crops suspended until rate<%lu/s",
					(unsigned long)lastWindowRate, (unsigned long)kGlobalRateScarcityEnterPerSec, (unsigned long)kGlobalRateScarcityExitPerSec);
				Logging::Log() << sbuf;
			}
			else if (g_globalScarcityMode && lastWindowRate < kGlobalRateScarcityExitPerSec)
			{
				g_globalScarcityMode = false;
				char sbuf[220];
				sprintf_s(sbuf, sizeof(sbuf), "[UNIV] SCARCITY EXIT rate=%lu/s (exit<%lu/s) -- new crops resumed (skipped total=%lu, enters total=%lu)",
					(unsigned long)lastWindowRate, (unsigned long)kGlobalRateScarcityExitPerSec, (unsigned long)univScarcitySkippedTotal, (unsigned long)univScarcityEntersTotal);
				Logging::Log() << sbuf;
			}
			g_globalRateWindowStartMs = nowMsGate;
			g_globalRateCropsInWindow = 0;
		}
		if (g_globalScarcityMode)
		{
			univScarcitySkippedTotal++;
			stage3PerSurface[(void*)this].scarcitySkips++;
			return false;
		}
	}

	// [NAMEKEY] V2 first sight of (name, mip): resolve the sidecar payload, verify the
	// live rect against it, and (mode 2) build the canonical crop FROM the payload.
	// Shadow mode (1) logs the verdict and binds nothing. Any failure marks the key
	// (permanently for sidecar/dims/create failures) and falls through to the legacy
	// content crop below -- v2 is purely additive coverage.
	if (nkPlace)
	{
		const uint64_t ck = ((uint64_t)nkPlace->idx << 3) | (uint64_t)(nkPlace->mip & 7);
		NamekeyCropState st;
		st.st = -1;
		const char* nm = NamekeyNameForIdx(nkPlace->idx);
		char ckey[128] = {};
		if (nm)
		{
			sprintf_s(ckey, sizeof(ckey), "%sMM%u", nm, (unsigned)(nkPlace->mip & 7));
			for (char* c = ckey; *c; ++c) *c = (char)tolower((unsigned char)*c);
		}
		// Runtime-rendered UI names (pointer strips, tooltips, map) have no sidecar by
		// nature -- classify them for the emissive draw-state override before the
		// lookup fails, and remember it in the state so later draws classify O(1).
		if (nm && NamekeyNameIsDynamicUi(ckey))
		{
			st.dynUi = true;
			st.isPointer = NamekeyNameIsPointer(ckey);
			g_namekeyDrawDynUiAny = true;
			if (!st.isPointer) g_namekeyDrawDynUi = true;
		}
		if (nm && NamekeyNameIsWater(ckey))
		{
			st.isWater = true;
			g_namekeyDrawWater = true;
		}
		auto ixIt = nm ? namekeyCropIdx.find((uint64_t)XXH3_64bits(ckey, strlen(ckey))) : namekeyCropIdx.end();
		if (!nm || ixIt == namekeyCropIdx.end())
		{
			// Procedural/unnamed content (map_texture etc.) has no extracted source --
			// correctly excluded; the legacy content crop handles it.
			namekeyCropNoSidecar++;
		}
		else if ((long)ixIt->second.w != (long)nkPlace->sw || (long)ixIt->second.h != (long)nkPlace->sh)
		{
			// Extracted dims must equal the descriptor's source dims -- doubling as a
			// cheap attribution guard before any pixel work.
			namekeyCropDimsMismatch++;
		}
		else if (!NamekeyCropReadPayload(ixIt->second, st.payload))
		{
			namekeyCropReadFail++;
		}
		else
		{
			st.w = ixIt->second.w;
			st.h = ixIt->second.h;
			float bf = 1.0f;
			const int vr = NamekeyCropVerifyRect(surface.Texture, nkPlace->a2, nkPlace->a3, st.payload, st.w, st.h, &bf);
			if (Config.DdrawNameKeyCrop == 1)
			{
				// Shadow: verdict logged once per (name, mip); nothing created or bound.
				st.st = 2;
				if (vr == 0) namekeyCropShadowPass++; else namekeyCropShadowFail++;
				if (namekeyCropShadowLogged < 200)
				{
					namekeyCropShadowLogged++;
					char buf[300];
					sprintf_s(buf, sizeof(buf), "[NAMEKEY] CROPSHADOW key=\"%s\" at=%ld,%ld %ux%u page=%ux%u draw=[%ld,%ld,%ld,%ld] vr=%d bigfrac=%.4f duv=%.4f,%.4f,%.4f,%.4f",
						ckey, (long)nkPlace->a2, (long)nkPlace->a3, (unsigned)st.w, (unsigned)st.h,
						atlasW, atlasH, px0, py0, px1, py1, vr, bf,
						nkPlace->u0, nkPlace->v0, nkPlace->u1, nkPlace->v1);
					Logging::Log() << buf;
				}
				st.payload.clear();
				st.payload.shrink_to_fit();	// shadow keeps no payload resident
			}
			else if (vr != 0)
			{
				if (vr == 2) namekeyCropBadFormat++; else namekeyCropVerifyFail++;
				if (namekeyCropLogged < 200)
				{
					namekeyCropLogged++;
					char buf[260];
					sprintf_s(buf, sizeof(buf), "[NAMEKEY] CROPVERIFYFAIL key=\"%s\" at=%ld,%ld %ux%u vr=%d bigfrac=%.4f",
						ckey, (long)nkPlace->a2, (long)nkPlace->a3, (unsigned)st.w, (unsigned)st.h, vr, bf);
					Logging::Log() << buf;
				}
			}
			else
			{
				// SERVE-KEY COLLAPSE (2026-06-11). Two collapses, same mechanism, both
				// "verify own payload (attribution truth), serve canonical payload":
				// - MM0-collapse: any mip placement serves the sprite's MM0 content;
				//   Remix sees ONE hash per sprite across all game-side mip pages and
				//   the hi-res replacement's own mip chain handles distance.
				// - WATER FRAME-collapse (DdrawWaterCollapse): WaterN/WaterSOLIDN all
				//   serve frame 0 -- the surface stops cycling, Remix's denoiser gets
				//   ONE stable material (reflections converge instead of resetting 32x
				//   per cycle), and a single translucent mod entry covers every frame.
				//   The water MESH still moves if the game animates it; only the
				//   texture stills. waterfoot etc. don't match (digit required).
				// UV transforms are placement-rect-based = resolution-independent.
				const std::vector<uint8_t>* servePayload = &st.payload;
				uint32_t serveW = st.w, serveH = st.h;
				std::vector<uint8_t> collapsedPayload;
				char serveKey[128];
				strcpy_s(serveKey, sizeof(serveKey), ckey);
				bool waterFam = false;
				if (Config.DdrawWaterCollapse)
				{
					if (!strncmp(ckey, "watersolid", 10) && isdigit((unsigned char)ckey[10]))
					{
						strcpy_s(serveKey, sizeof(serveKey), "watersolid0mm0");
						waterFam = true;
					}
					else if (!strncmp(ckey, "water", 5) && isdigit((unsigned char)ckey[5]))
					{
						strcpy_s(serveKey, sizeof(serveKey), "water0mm0");
						waterFam = true;
					}
				}
				if (!waterFam && (nkPlace->mip & 7) != 0)
				{
					sprintf_s(serveKey, sizeof(serveKey), "%sMM0", nm);
					for (char* c = serveKey; *c; ++c) *c = (char)tolower((unsigned char)*c);
				}
				if (strcmp(serveKey, ckey) != 0)
				{
					auto ix0 = namekeyCropIdx.find((uint64_t)XXH3_64bits(serveKey, strlen(serveKey)));
					if (ix0 != namekeyCropIdx.end() && NamekeyCropReadPayload(ix0->second, collapsedPayload))
					{
						servePayload = &collapsedPayload;
						serveW = ix0->second.w;
						serveH = ix0->second.h;
						if (waterFam) namekeyWaterCollapse++; else namekeyCropMm0Collapse++;
					}
				}
				const uint64_t ph = (uint64_t)XXH3_64bits(servePayload->data(), servePayload->size());
				IDirect3DTexture9* ctex = NamekeyCropGetOrCreateTex(ph, *servePayload, serveW, serveH, d3d9Device);
				if (ctex)
				{
					st.st = 1;
					st.tex = ctex;
					namekeyCropSynth++;
					g_globalRateCropsInWindow++;	// creation is real work; count it in the burst window
					if (namekeyCropLogged < 200)
					{
						namekeyCropLogged++;
						char buf[300];
						sprintf_s(buf, sizeof(buf), "[NAMEKEY] CROP key=\"%s\" %ux%u at=%ld,%ld -> serve=%ux%u hash=%016llX (synth=%lu texes=%lu)",
							ckey, (unsigned)st.w, (unsigned)st.h, (long)nkPlace->a2, (long)nkPlace->a3,
							serveW, serveH, (unsigned long long)ph, (unsigned long)namekeyCropSynth, (unsigned long)namekeyCropTexCreated);
						Logging::Log() << buf;
					}
				}
			}
		}
		const bool nkServeNow = (st.st == 1 && st.tex != nullptr);
		IDirect3DTexture9* nkTex = st.tex;
		const long nkx = nkPlace->a2, nky = nkPlace->a3, nkw = (long)st.w, nkh = (long)st.h;
		namekeyCropByKey[ck] = std::move(st);
		if (nkServeNow)
		{
			return NkServeCrop(nkTex, nkx, nky, nkw, nkh);
		}
		// fall through: legacy content crop (correct regardless of why we're here)
	}

	// First sight of this (surface, rect). Crop the exact texel rect DIRECTLY into a
	// rect-sized MANAGED texture -- the SAME single GPU->MANAGED copy A.10 used and which
	// never froze. NO SYSTEMMEM readback + LockRect here: that mid-scene VRAM->sysmem
	// stall (added only to content-hash for cross-surface dedup) is what deadlocked
	// against Remix. Dedup dropped -- identical content on two different surfaces now
	// makes two sub-textures (minor memory, not correctness; each is still a clean
	// single-region image). See dk2_universal_decompose_plan memory.
	IDirect3DSurface9* srcLevel = nullptr;
	if (FAILED(surface.Texture->GetSurfaceLevel(0, &srcLevel)) || !srcLevel) return false;

	// POWER-OF-TWO destination (2026-05-21): Remix's server AV'd (0xC0000005, identical
	// crash address both times) on NPOT rect-sized sub-textures -- a near-square 65x65
	// crop squeaked by, but thin/extreme-aspect NPOT crops (41x52, 64x9) crash its
	// geometry/mip path. A.10's canonical (always created at the atlas's POT size with
	// the crop stretched to fill) NEVER crashed Remix in months. So size the sub-texture
	// to per-dimension next-POT and STRETCH the cropped region into it. The draw's UVs
	// are rewritten to [0,1] over the region (below), so the stretch is undone at sample
	// time -- visually identical, but Remix sees a clean POT texture. The UV mapping is
	// normalized so the POT size does not affect it. See dk2_universal_decompose_plan.
	auto nextPOT = [](long v) -> UINT { UINT p = 8; while ((long)p < v) p <<= 1; return p; };
	const UINT dstW = nextPOT(rw);
	const UINT dstH = nextPOT(rh);

	// MANAGED A8R8G8B8, full mip chain (levels=0). Without lower mips, far/small torches
	// render as a solid white block under Remix.
	IDirect3DTexture9* sub = nullptr;
	HRESULT hr = (*d3d9Device)->CreateTexture(dstW, dstH, 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &sub, nullptr);
	if (FAILED(hr) || !sub) { srcLevel->Release(); return false; }

	IDirect3DSurface9* dstLevel = nullptr;
	if (FAILED(sub->GetSurfaceLevel(0, &dstLevel)) || !dstLevel)
	{
		sub->Release(); srcLevel->Release(); return false;
	}

	// Stretch the cropped texel rect to FILL the POT destination (no dstRect = whole
	// surface). LINEAR resamples the rect up/down to the POT size.
	RECT srcRect{ px0, py0, px1, py1 };
	hr = D3DXLoadSurfaceFromSurface(dstLevel, nullptr, nullptr, srcLevel, nullptr, &srcRect, D3DX_FILTER_LINEAR, 0);
	if (FAILED(hr))
	{
		dstLevel->Release(); sub->Release(); srcLevel->Release(); return false;
	}

	if (sub->GetLevelCount() > 1)
		D3DXFilterTexture(sub, nullptr, 0, D3DX_FILTER_LINEAR);
	univCropsTotal++;
	g_globalRateCropsInWindow++;
	stage3PerSurface[(void*)this].crops++;

	// Per-surface streaming-buffer detection (Phase 4, periodic-recheck in Phase 6).
	// Count crops per surface; at every kBlacklistWindowCrops checkpoint, compute the
	// rate over the most recent window (NOT cumulative). If above threshold, blacklist.
	// Re-checking periodically catches late-emerging streamers -- surfaces whose
	// crop rate ramps up only during specific gameplay (late-game combat, mining, etc.)
	// and which wouldn't trip a one-shot cumulative-rate check at crop #256.
	// Logged on every blacklist event with the window stats.
	static const DWORD kBlacklistWindowCrops = 256;
	static const DWORD kBlacklistRateThresholdPerSec = 30;
	surface.UnivCropCount++;
	const DWORD nowMs = GetTickCount();
	if (surface.UnivCheckpointStartMs == 0) surface.UnivCheckpointStartMs = nowMs;
	if ((surface.UnivCropCount % kBlacklistWindowCrops) == 0)
	{
		const DWORD elapsedMs = nowMs - surface.UnivCheckpointStartMs;
		if (elapsedMs > 0)
		{
			const DWORD ratePerSec = (kBlacklistWindowCrops * 1000) / elapsedMs;
			if (ratePerSec > kBlacklistRateThresholdPerSec)
			{
				surface.UnivSkipDecompose = true;
				char bbuf[220];
				sprintf_s(bbuf, sizeof(bbuf), "[UNIV] BLACKLIST surface=%p totalCrops=%lu (window %lu in %lums -> %lu/s, threshold=%lu) -- streaming buffer, decompose disabled",
					(void*)this, (unsigned long)surface.UnivCropCount,
					(unsigned long)kBlacklistWindowCrops, (unsigned long)elapsedMs,
					(unsigned long)ratePerSec, (unsigned long)kBlacklistRateThresholdPerSec);
				Logging::Log() << bbuf;
			}
		}
		// Start a new measurement window for this surface.
		surface.UnivCheckpointStartMs = nowMs;
	}

	// Capture locks the MANAGED dst (sysmem-backed -> no GPU readback stall), unlike
	// the removed SYSTEMMEM scratch lock. Same call A.10 used safely.
	if (Config.DdrawContentCapture)
		CaptureSurfaceContent(dstLevel, (void*)sub, /*isCrop=*/true);

	{
		char buf[300];
		sprintf_s(buf, sizeof(buf), "[UNIV] synthesized atlas=%p rect=[%ld,%ld,%ld,%ld] %ldx%ld->%ux%u(POT) of %ux%u (binds=%lu crops=%lu cache=%zu/%zu evicts=%lu invals=%lu)",
			(void*)this, px0, py0, px1, py1, rw, rh, dstW, dstH, atlasW, atlasH,
			(unsigned long)univDecomposeBindsTotal, (unsigned long)univCropsTotal,
			univDrawCache.size(), kUnivCacheCap, (unsigned long)univEvictTotal,
			(unsigned long)univInvalidatesTotal);
		Logging::Log() << buf;
	}

	dstLevel->Release();
	srcLevel->Release();

	// [NAMEKEY] DYNUI CROP (2026-06-11): the EXACT hash Remix sees for a dynamic-UI
	// draw that fell to the legacy crop -- the pointer strips live here (no sidecar).
	// This is the authoring list for rtx.uiTextures: tag these and the rasterized-UI
	// path renders the hand in its original colors on top of the path-traced frame
	// (the Desktop copy's persistent-hand mechanism, re-keyed to our pipeline).
	if (g_namekeyDrawDynUiAny && namekeyDynUiCropLogLines < 60)
	{
		const uint64_t dh = Stage3HashD3d9Texture(sub);
		if (dh && namekeyDynUiCropLogged.insert(dh).second)
		{
			namekeyDynUiCropLogLines++;
			char dbuf[200];
			sprintf_s(dbuf, sizeof(dbuf), "[NAMEKEY] DYNUI CROP hash=%016llX rect=[%ld,%ld,%ld,%ld] page=%ux%u",
				(unsigned long long)dh, px0, py0, px1, py1, atlasW, atlasH);
			Logging::Log() << dbuf;
		}
	}

	// Cache this (surface, rect) -> sub-texture as most-recently-used. Record:
	// - cachedAtRepurposeGen for the per-surface idle-then-Blt gate (Phase 5)
	// - cachedAtTimeMs + px0/py0/px1/py1 pixel rect for the per-Blt overlap gate (Phase 7)
	univLru.push_back(dkey);
	auto lruIt = std::prev(univLru.end());
	univDrawCache[dkey] = UnivSubTex{ sub, su0, sv0, su1, sv1, lruIt, surface.RepurposeGen,
		px0, py0, px1, py1, GetTickCount() };

	// Evict least-recently-used entries past the cap, freeing their MANAGED textures
	// (the fix for the 32-bit OOM from unbounded map-churn crops). Torches stay (MRU).
	while (univDrawCache.size() > kUnivCacheCap && !univLru.empty())
	{
		const UnivDrawKey oldKey = univLru.front();
		univLru.pop_front();
		auto oIt = univDrawCache.find(oldKey);
		if (oIt != univDrawCache.end())
		{
			if (oIt->second.tex) oIt->second.tex->Release();
			univDrawCache.erase(oIt);
			univEvictTotal++;
		}
	}

	subTexOut = sub;
	regionU0Out = su0;
	regionV0Out = sv0;
	regionDuOut = su1 - su0;
	regionDvOut = sv1 - sv0;
	regionIdxOut = 0;
	univDecomposeBindsTotal++;
	return true;
}

// Phase 7 (2026-05-24): push a Blt dest rect into the surface's BltHistory ring buffer.
// dirtyRect=nullptr means "whole surface" -> records (0,0,width,height). Called from the
// success paths of Blt()/BltFast() (and could be called from any other code path that knows
// it just modified content at a known region of the surface). The universal-decompose hit
// gate uses this to invalidate per-rect when a cached entry's UV rect overlaps a recent Blt.
void m_IDirectDrawSurfaceX::RecordBltRect(const RECT* dirtyRect)
{
	BltHistEntry& e = BltHistory[BltHistoryHead];
	if (dirtyRect)
	{
		e.x0 = dirtyRect->left;
		e.y0 = dirtyRect->top;
		e.x1 = dirtyRect->right;
		e.y1 = dirtyRect->bottom;
	}
	else
	{
		e.x0 = 0;
		e.y0 = 0;
		e.x1 = (LONG)surface.Width;
		e.y1 = (LONG)surface.Height;
	}
	e.timeMs = GetTickCount();
	BltHistoryHead = (BltHistoryHead + 1) % kBltHistorySize;
}

void m_IDirectDrawSurfaceX::SetDirtyFlag(DWORD MipMapLevel, const RECT* dirtyRect)
{
	if (MipMapLevel == 0)
	{
		if (IsPrimarySurface() && ddrawParent && !ddrawParent->IsInScene())
		{
			dirtyFlag = true;
		}
		surface.IsDirtyFlag = true;
		surface.HasData = true;
		surface.IsDrawTextureDirty = true;
		// Per-Blt rect history (Phase 7, 2026-05-24): push the modified rect (or
		// "whole surface" when dirtyRect == nullptr) into BltHistory ring buffer.
		// The universal-decompose hit gate uses this to invalidate cache entries whose
		// UV rect overlaps a Blt that occurred after the entry was cached. Catches the
		// "continuously-Blt'd shared atlas" case where the per-surface RepurposeGen gate
		// alone can't fire (no idle gap).
		RecordBltRect(dirtyRect);
		// Gap-based repurpose detection (Phase 5, 2026-05-24): bump RepurposeGen only
		// when this Blt follows a long idle period. Animations Blt continuously (small
		// gaps) -> no bump -> cached entries stay valid. Sprite repurposes follow seconds
		// of idle -> gap > threshold -> bump -> cached entries for this surface invalidate
		// on next bind, forcing a fresh crop. Skip the bump on the very first Blt (prev=0)
		// because there are no cached entries yet to invalidate.
		static const DWORD kRepurposeIdleGapMs = 1000;
		const DWORD nowMs = GetTickCount();
		const DWORD prevBltMs = surface.LastBltTimeMs;
		surface.LastBltTimeMs = nowMs;
		if (prevBltMs != 0 && (nowMs - prevBltMs) > kRepurposeIdleGapMs)
		{
			surface.RepurposeGen++;
		}
		IsMipMapReadyToUse = (IsMipMapAutogen() || MipMaps.empty());

		// Update Uniqueness Value
		ChangeUniquenessValue(0);

		// Phase A.7: continuous content capture. Fires here (instead of only inside
		// CopyToDrawTexture) because non-color-key textures never go through that
		// path -- the d3d9 SetTexture for them binds surface.Texture directly via
		// Get3DTexture(). SetDirtyFlag is called after EVERY successful mip-0 write
		// regardless of code path, so this catches all content the game produces.
		if (Config.DdrawContentCapture && IsSurfaceTexture() && surface.Texture)
		{
			IDirect3DSurface9* level0 = nullptr;
			if (SUCCEEDED(surface.Texture->GetSurfaceLevel(0, &level0)) && level0)
			{
				CaptureSurfaceContent(level0);
				level0->Release();
			}
		}

		// Phase A.10: invalidate cached fingerprint-family match for this wrapper.
		// Game just wrote new content here -- our prior match decision (made on
		// the OLD content) is stale. Next TryGetSubTextureForUV call will
		// recompute the fingerprint against the current bytes. Fixes the
		// cache-staleness bug where surfaces first bound with non-torch content
		// stay cached as NO_MATCH even after getting Blt'd to torch content
		// later (observed empirically as missing-magenta torches at L2/L3 zoom
		// or after camera repositioning, 2026-05-20).
		if (Config.DdrawAtlasDecompose)
		{
			wrapperToFamilyIdx.erase((void*)this);
		}

		// [RECIPE] (2026-06-10): a mip-0 write NOT coming from our Blt path (Lock/
		// direct CPU write) on a recipe-tracked surface. Count it so emitted recipes
		// carry a direct=N caveat -- such content is partly invisible to Blt recording.
		// recipeLastBltTex filters Blt's own SetDirtyFlag call, which arrives after
		// ScopedFlagSet(IsInBlt) has already closed.
		if (Config.DdrawRecipeLog && IsSurfaceTexture() && surface.Texture)
		{
			if (surface.Texture == recipeLastBltTex)
			{
				recipeLastBltTex = nullptr;
			}
			else if (!IsSurfaceBlitting())
			{
				auto rit = recipeByTex.find(surface.Texture);
				if (rit != recipeByTex.end()) rit->second.directWrites++;

				// [LOCKW] flush: the game's Lock write(s) just landed -- log the
				// resulting staging content hash + every pending (eip, rect).
				// UniquenessValue is already bumped here, so the hash below is the
				// exact value the next Blt-out's srcHash cache will compute.
				auto lw = lockWPending.find((void*)this);
				if (lw != lockWPending.end() && !lw->second.empty())
				{
					const uint64_t hNow = Stage3HashD3d9Texture(surface.Texture);
					char buf[128];
					sprintf_s(buf, sizeof(buf), "[LOCKW] surf=%p hash=%016llX n=%zu writes=",
						(void*)this, (unsigned long long)hNow, lw->second.size());
					std::string line = buf;
					for (const LockWPend& p : lw->second)
					{
						if (p.hasRect)
						{
							sprintf_s(buf, sizeof(buf), "%08lX:%ld,%ld,%ld,%ld|", (unsigned long)p.eip,
								p.rect.left, p.rect.top, p.rect.right, p.rect.bottom);
						}
						else
						{
							sprintf_s(buf, sizeof(buf), "%08lX:FULL|", (unsigned long)p.eip);
						}
						line += buf;
					}
					Logging::Log() << line.c_str();
					lw->second.clear();
				}
			}
		}

		// Canonical Identity Layer (2026-06-09): content changed -> the cached
		// content hash and canonical resolution for this texture are stale. Erase
		// both; the next bind rehashes + re-resolves against the NEW bytes. After
		// kCanonChurnCap invalidations (animated/composited surfaces rewriting
		// every frame), permanently exempt the texture: cache a null resolution so
		// its binds stop paying lock+rehash entirely. The Stage 3 hash cache keeps
		// its last value for exempted textures, preserving pre-existing stats
		// behavior for hot writers.
		if (Config.DdrawCanonicalRebind && IsSurfaceTexture() && surface.Texture)
		{
			DWORD& churn = canonChurnByTex[surface.Texture];
			if (churn <= kCanonChurnCap)
			{
				churn++;
				if (churn > kCanonChurnCap)
				{
					canonResolveByTex[surface.Texture] = nullptr;
					canonChurnExemptTotal++;
				}
				else
				{
					stage3HashByD3d9Tex.erase(surface.Texture);
					canonResolveByTex.erase(surface.Texture);
				}
			}
		}
	}
	// Mark mipmap data flag
	if (MipMaps.size())
	{
		if (MipMapLevel && MipMapLevel <= MipMaps.size())
		{
			if (MipMaps[MipMapLevel - 1].UniquenessValue == UniquenessValue)
			{
				MipMaps[MipMapLevel - 1].UniquenessValue++;
			}
			else if (MipMaps[MipMapLevel - 1].UniquenessValue < UniquenessValue)
			{
				MipMaps[MipMapLevel - 1].UniquenessValue = UniquenessValue;
			}
		}
		CheckMipMapLevelGen();
	}
}

void m_IDirectDrawSurfaceX::ClearDirtyFlags()
{
	// Reset dirty flag
	dirtyFlag = false;
	surface.IsDirtyFlag = false;

	// Reset scene ready
	SceneReady = false;
}

bool m_IDirectDrawSurfaceX::CanSurfaceBeLost() const
{
	if ((surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_VIDEOMEMORY) && !IsSurfaceManaged() && IsD9UsingVideoMemory())
	{
		if (!ComplexChild)	// Complex children don't get surface lost notice
		{
			return true;
		}
	}
	return false;
}

void m_IDirectDrawSurfaceX::MarkSurfaceLost()
{
	if (CanSurfaceBeLost())
	{
		IsSurfaceLost = true;
	}
}

bool m_IDirectDrawSurfaceX::CheckRectforSkipScene(RECT& DestRect)
{
	bool isSingleLine = (DestRect.bottom - DestRect.top == 1);	// Only handles horizontal lines at this point

	return Config.DdrawRemoveInterlacing ? isSingleLine : false;
}

void m_IDirectDrawSurfaceX::BeginWritePresent(bool IsSkipScene)
{
	// Check if data needs to be presented before write
	if (dirtyFlag)
	{
		if (FAILED(PresentSurface(nullptr, IsSkipScene)))
		{
			PresentOnUnlock = true;
		}
	}
}

void m_IDirectDrawSurfaceX::EndWritePresent(LPRECT lpDestRect, bool IsSkipScene)
{
	// Handle overlays
	PresentOverlay(lpDestRect);

	if (ShouldWriteToGDI())
	{
		if (IsPrimaryOrBackBuffer())
		{
			CopyEmulatedSurfaceToGDI(lpDestRect);
		}
	}
	// Present surface after each draw unless removing interlacing
	else if (PresentOnUnlock || !Config.DdrawRemoveInterlacing)
	{
		PresentSurface(lpDestRect, IsSkipScene);

		// Reset endscene lock
		PresentOnUnlock = false;
	}
}

void m_IDirectDrawSurfaceX::EndWriteSyncSurfaces(LPRECT lpDestRect)
{
	// Copy emulated surface to real surface
	if (IsUsingEmulation())
	{
		CopyFromEmulatedSurface(lpDestRect);
	}

	// Pre-populate draw texture
	if (Using3D && IsSurfaceTexture() && IsColorKeyTexture() && !IsPrimaryOrBackBuffer() && !IsRenderTarget())
	{
		GetD3d9DrawTexture();
	}
}

bool m_IDirectDrawSurfaceX::IsSurfaceLocked(DWORD MipMapLevel)
{
	if (MipMapLevel == DXW_ALL_SURFACE_LEVELS)
	{
		for (auto& entry : LockedLevel)
		{
			if (entry.second.IsLocked)
			{
				return true;
			}
		}
		return false;
	}

	const auto it = LockedLevel.find(MipMapLevel);
	if (it != LockedLevel.end())
	{
		return it->second.IsLocked;
	}

	return false;
}

bool m_IDirectDrawSurfaceX::IsSurfaceInDC(DWORD MipMapLevel)
{
	if (MipMapLevel == DXW_ALL_SURFACE_LEVELS)
	{
		for (auto& entry : GetDCLevel)
		{
			if (entry.second)
			{
				return true;
			}
		}
		return false;
	}

	const auto it = GetDCLevel.find(MipMapLevel);
	if (it != GetDCLevel.end())
	{
		return it->second != nullptr;
	}

	return false;
}

bool m_IDirectDrawSurfaceX::IsLockedFromOtherThread(DWORD MipMapLevel)
{
	if (IsSurfaceBlitting() || IsSurfaceLocked(MipMapLevel))
	{
		const auto it = LockedLevel.find(MipMapLevel);
		if (it != LockedLevel.end())
		{
			DWORD LockedWithID = it->second.LockedWithID;
			if (LockedWithID)
			{
				return LockedWithID != GetCurrentThreadId();
			}
		}
	}
	return false;
}

void m_IDirectDrawSurfaceX::InitSurfaceDesc(DWORD DirectXVersion)
{
	// Update dds caps flags
	surfaceDesc2.dwFlags |= DDSD_CAPS;
	if (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_PRIMARYSURFACE)
	{
		surfaceDesc2.ddsCaps.dwCaps |= DDSCAPS_VISIBLE;
	}
	if (!(surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY) && !(surfaceDesc2.ddsCaps.dwCaps2 & DDSCAPS2_DONOTPERSIST))
	{
		surfaceDesc2.ddsCaps.dwCaps |= DDSCAPS_LOCALVIDMEM | DDSCAPS_VIDEOMEMORY;
		surfaceDesc2.ddsCaps.dwCaps &= ~DDSCAPS_NONLOCALVIDMEM;
	}

	// Create backbuffers
	if ((surfaceDesc2.dwFlags & DDSD_BACKBUFFERCOUNT) && surfaceDesc2.dwBackBufferCount)
	{
		DDSURFACEDESC2 Desc2 = surfaceDesc2;
		Desc2.ddsCaps.dwCaps4 &= ~(DDSCAPS4_CREATESURFACE);	// Clear surface creation flag
		Desc2.dwBackBufferCount--;
		Desc2.ddsCaps.dwCaps |= DDSCAPS_BACKBUFFER;
		Desc2.ddsCaps.dwCaps &= ~(DDSCAPS_VISIBLE | DDSCAPS_PRIMARYSURFACE | DDSCAPS_FRONTBUFFER);
		Desc2.ddsCaps.dwCaps4 |= DDSCAPS4_COMPLEXCHILD;

		if (surfaceDesc2.ddsCaps.dwCaps4 & DDSCAPS4_CREATESURFACE)
		{
			ComplexRoot = true;
			Desc2.dwReserved = (DWORD)this;
		}

		// Create complex surfaces
		BackBufferInterface = std::make_unique<m_IDirectDrawSurfaceX>(ddrawParent, DirectXVersion, &Desc2);

		m_IDirectDrawSurfaceX* attachedSurface = BackBufferInterface.get();

		AddAttachedSurfaceToMap(attachedSurface, false, DirectXVersion, 1);
	}

	// Set flags for complex child surface
	if (surfaceDesc2.ddsCaps.dwCaps4 & DDSCAPS4_COMPLEXCHILD)
	{
		// Add first surface as attached surface to the last surface in a surface chain
		if (surfaceDesc2.dwReserved && surfaceDesc2.dwBackBufferCount == 0)
		{
			m_IDirectDrawSurfaceX* attachedSurface = (m_IDirectDrawSurfaceX*)surfaceDesc2.dwReserved;

			// Check if source Surface exists add to surface map
			if (ddrawParent && ddrawParent->DoesSurfaceExist(attachedSurface))
			{
				AddAttachedSurfaceToMap(attachedSurface, false, DirectXVersion, 0);
			}
		}

		// Set complex child flags
		ComplexChild = true;
		surfaceDesc2.dwFlags &= ~DDSD_BACKBUFFERCOUNT;
		surfaceDesc2.dwBackBufferCount = 0;
	}

	// Handle mipmaps
	if ((!(surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) || ((surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) && surfaceDesc2.dwMipMapCount != 1)) &&
		(surfaceDesc2.ddsCaps.dwCaps & (DDSCAPS_MIPMAP | DDSCAPS_COMPLEX | DDSCAPS_TEXTURE)) == (DDSCAPS_MIPMAP | DDSCAPS_COMPLEX | DDSCAPS_TEXTURE))
	{
		// Compute width and height
		if ((!(surfaceDesc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT)) || (!surfaceDesc2.dwWidth && !surfaceDesc2.dwHeight)) &&
			(surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) && surfaceDesc2.dwMipMapCount > 0)
		{
			surfaceDesc2.dwFlags |= DDSD_WIDTH | DDSD_HEIGHT;
			surfaceDesc2.dwWidth = (DWORD)pow(2, surfaceDesc2.dwMipMapCount - 1);
			surfaceDesc2.dwHeight = surfaceDesc2.dwWidth;
		}
		// Compute mipcount
		DWORD MipMapLevelCount = ((surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT) && surfaceDesc2.dwMipMapCount) ? surfaceDesc2.dwMipMapCount :
			GetMaxMipMapLevel(surfaceDesc2.dwWidth, surfaceDesc2.dwHeight);
		MaxMipMapLevel = MipMapLevelCount - 1;
		surfaceDesc2.dwMipMapCount = MaxMipMapLevel + 1;
		surfaceDesc2.dwFlags |= DDSD_MIPMAPCOUNT;
	}
	// Mipmap textures
	else if (surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_MIPMAP)
	{
		if (surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT)
		{
			surfaceDesc2.dwMipMapCount = 1;
		}
	}
	// No mipmaps
	else
	{
		if (surfaceDesc2.dwFlags & DDSD_MIPMAPCOUNT)
		{
			surfaceDesc2.dwMipMapCount = 0;
		}
		surfaceDesc2.dwFlags &= ~DDSD_MIPMAPCOUNT;
		surfaceDesc2.ddsCaps.dwCaps &= ~DDSCAPS_MIPMAP;
	}

	// Clear pitch
	if (!(surfaceDesc2.dwFlags & DDSD_LPSURFACE))
	{
		surfaceDesc2.dwFlags &= ~DDSD_PITCH;
		surfaceDesc2.lPitch = 0;
	}
	surface.UsingSurfaceMemory = ((surfaceDesc2.dwFlags & DDSD_LPSURFACE) && surfaceDesc2.lpSurface);
	if (surface.UsingSurfaceMemory)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: lpSurface not fully Implemented.");
		if (surfaceDesc2.dwMipMapCount > 1)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: lpSurface not supported with MipMaps!");
		}
	}

	// Clear flags used in creating a surface structure
	surfaceDesc2.ddsCaps.dwCaps4 = 0;
	surfaceDesc2.dwReserved = 0;

	// Clear unused values
	ClearUnusedValues(surfaceDesc2);
}

void m_IDirectDrawSurfaceX::AddAttachedSurfaceToMap(m_IDirectDrawSurfaceX* lpSurfaceX, bool MarkAttached, DWORD DxVersion, DWORD RefCount)
{
	if (!lpSurfaceX)
	{
		return;
	}

	// Store surface
	AttachedSurfaceMap[++MapKey].pSurface = lpSurfaceX;
	AttachedSurfaceMap[MapKey].isAttachedSurfaceAdded = MarkAttached;

	AttachedSurfaceMap[MapKey].DxVersion = DxVersion;
	AttachedSurfaceMap[MapKey].RefCount = RefCount;

	if (RefCount == 1)
	{
		lpSurfaceX->AddRef(DxVersion);
	}
}

void m_IDirectDrawSurfaceX::RemoveAttachedSurfaceFromMap(m_IDirectDrawSurfaceX* lpSurfaceX)
{
	auto it = std::find_if(AttachedSurfaceMap.begin(), AttachedSurfaceMap.end(),
		[=](auto Map) -> bool { return Map.second.pSurface == lpSurfaceX; });

	if (it != std::end(AttachedSurfaceMap))
	{
		DWORD RefCount = it->second.RefCount;
		DWORD DxVersion = it->second.DxVersion;
		AttachedSurfaceMap.erase(it);	// Erase from list before releasing
		if (RefCount == 1)
		{
			lpSurfaceX->Release(DxVersion);
		}
	}
}

bool m_IDirectDrawSurfaceX::DoesAttachedSurfaceExist(m_IDirectDrawSurfaceX* lpSurfaceX)
{
	if (!lpSurfaceX)
	{
		return false;
	}

	return (std::find_if(AttachedSurfaceMap.begin(), AttachedSurfaceMap.end(),
		[=](auto Map) -> bool { return Map.second.pSurface == lpSurfaceX; }) != std::end(AttachedSurfaceMap));
}

bool m_IDirectDrawSurfaceX::WasAttachedSurfaceAdded(m_IDirectDrawSurfaceX* lpSurfaceX)
{
	if (!lpSurfaceX)
	{
		return false;
	}

	return (std::find_if(AttachedSurfaceMap.begin(), AttachedSurfaceMap.end(),
		[=](auto Map) -> bool { return (Map.second.pSurface == lpSurfaceX) && Map.second.isAttachedSurfaceAdded; }) != std::end(AttachedSurfaceMap));
}

bool m_IDirectDrawSurfaceX::DoesFlipBackBufferExist(m_IDirectDrawSurfaceX* lpSurfaceX)
{
	if (!lpSurfaceX)
	{
		return false;
	}

	DWORD dwCaps = 0;
	m_IDirectDrawSurfaceX *lpTargetSurface = nullptr;

	// Loop through each surface
	for (auto& it : AttachedSurfaceMap)
	{
		if (it.second.pSurface && (it.second.pSurface->GetSurfaceCaps().dwCaps & DDSCAPS_FLIP))
		{
			lpTargetSurface = it.second.pSurface;

			break;
		}
	}

	// Check if attached surface was not found
	if (!lpTargetSurface || (dwCaps & DDSCAPS_FRONTBUFFER))
	{
		return false;
	}

	// Check if attached surface was found
	if (lpTargetSurface == lpSurfaceX)
	{
		return true;
	}

	// Check next surface
	return lpTargetSurface->DoesFlipBackBufferExist(lpSurfaceX);
}

HRESULT m_IDirectDrawSurfaceX::ColorFill(RECT* pRect, D3DCOLOR dwFillColor, DWORD MipMapLevel)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	// Check for device interface
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr))
	{
		return c_hr;
	}

	// Get surface desc for mipmap
	DDSURFACEDESC2 Desc2 = {};
	Desc2.dwSize = sizeof(DDSURFACEDESC2);
	GetSurfaceDesc2(&Desc2, MipMapLevel, 7);

	// Check and copy rect
	RECT DestRect = {};
	if (!CheckCoordinates(DestRect, pRect, &Desc2))
	{
		return DDERR_INVALIDRECT;
	}

	// Handle clipper
	if (attachedClipper)
	{
		RECT clipBounds = {};
		if (attachedClipper->GetClipBoundsFromData(clipBounds))
		{
			// Intersect destination rect with clip region
			RECT clippedDest = {};
			if (!IntersectRect(&clippedDest, &DestRect, &clipBounds))
			{
				// Fully clipped - no color fill needed
				LOG_LIMIT(100, __FUNCTION__ << " Warning: dest rect is fully clipped!");
				return DD_OK;
			}

			// Replace original rects with clipped/adjusted ones
			DestRect = clippedDest;
		}
	}

	HRESULT hr = DDERR_GENERIC;

	// Use GPU ColorFill
	if (!IsUsingShadowSurface() && ((surface.Usage & D3DUSAGE_RENDERTARGET) || surface.Type == D3DTYPE_OFFPLAINSURFACE) && surface.Pool == D3DPOOL_DEFAULT)
	{
		ScopedGetMipMapContext Dest(this, MipMapLevel);
		if (Dest.GetSurface())
		{
			hr = (*d3d9Device)->ColorFill(Dest.GetSurface(), &DestRect, dwFillColor);

			if (FAILED(hr))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not color fill: " << (D3DERR)hr);
			}
		}
	}
	
	// Lock surface and manually fill with color
	if (FAILED(hr))
	{
		// Get width and height of rect
		LONG FillWidth = DestRect.right - DestRect.left;
		LONG FillHeight = DestRect.bottom - DestRect.top;

		// Check bit count
		if (surface.BitCount != 8 && surface.BitCount != 12 && surface.BitCount != 16 && surface.BitCount != 24 && surface.BitCount != 32)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid bit count: " << surface.BitCount << " Width: " << FillWidth);
			return DDERR_GENERIC;
		}

		// Check if surface is not locked then lock it
		D3DLOCKED_RECT DestLockRect = {};
		if (FAILED(IsUsingEmulation() ? LockEmulatedSurface(&DestLockRect, &DestRect) :
			LockD3d9Surface(&DestLockRect, &DestRect, 0, MipMapLevel)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not lock destination surface " << DestRect);
			return (IsSurfaceLocked(MipMapLevel)) ? DDERR_SURFACEBUSY : DDERR_GENERIC;
		}

		bool CanUseMemSet = surface.BitCount == 8 ? true :
			surface.BitCount == 12 ||
			surface.BitCount == 16 ? (dwFillColor & 0xFF) == ((dwFillColor >> 8) & 0xFF) :
			surface.BitCount == 24 ? (dwFillColor & 0xFF) == ((dwFillColor >> 8) & 0xFF) &&
									 (dwFillColor & 0xFF) == ((dwFillColor >> 16) & 0xFF) :
			surface.BitCount == 32 ? (dwFillColor & 0xFF) == ((dwFillColor >> 8) & 0xFF) &&
									 (dwFillColor & 0xFF) == ((dwFillColor >> 16) & 0xFF) &&
									 (dwFillColor & 0xFF) == ((dwFillColor >> 24) & 0xFF) : false;

		if (FillWidth == (LONG)surfaceDesc2.dwWidth && CanUseMemSet)
		{
			memset(DestLockRect.pBits, dwFillColor, DestLockRect.Pitch * FillHeight);
		}
		else if (surface.BitCount == 8 || (surface.BitCount == 12 && FillWidth % 2 == 0) || surface.BitCount == 16 || surface.BitCount == 24 || surface.BitCount == 32)
		{
			// Get byte count
			DWORD ByteCount = surface.BitCount / 8;

			// Handle 12-bit surface
			if (surface.BitCount == 12)
			{
				ByteCount = 3;
				dwFillColor = (dwFillColor & 0xFFF) + ((dwFillColor & 0xFFF) << 12);
				FillWidth /= 2;
			}

			// Fill first line memory
			if ((surface.BitCount == 8 || surface.BitCount == 16 || surface.BitCount == 32) &&								// Check bit count
				(FillWidth % (sizeof(DWORD) / ByteCount) == 0) && reinterpret_cast<uintptr_t>(DestLockRect.pBits) % sizeof(DWORD) == 0)	// Check for aligned width and memory
			{
				DWORD Color = (surface.BitCount == 8) ? (dwFillColor & 0xFF) * 0x01010101 :
					(surface.BitCount == 16) ? (dwFillColor & 0xFFFF) * 0x00010001 : dwFillColor;

				DWORD* DestBuffer = reinterpret_cast<DWORD*>(DestLockRect.pBits);
				LONG Iterations = FillWidth / (sizeof(DWORD) / ByteCount);

				for (LONG x = 0; x < Iterations; ++x)
				{
					*DestBuffer++ = Color;
				}
			}
			else
			{
				BYTE* SrcColor = reinterpret_cast<BYTE*>(&dwFillColor);
				BYTE* DestBuffer = reinterpret_cast<BYTE*>(DestLockRect.pBits);

				for (LONG x = 0; x < FillWidth; ++x)
				{
					BYTE* Color = SrcColor;
					for (DWORD y = 0; y < ByteCount; ++y)
					{
						*DestBuffer++ = *Color;
						Color++;
					}
				}
			}

			// Fill rest of surface rect using the first line as a template
			BYTE* SrcBuffer = (BYTE*)DestLockRect.pBits;
			BYTE* DestBuffer = (BYTE*)DestLockRect.pBits + DestLockRect.Pitch;
			size_t Size = FillWidth * ByteCount;
			for (LONG y = 1; y < FillHeight; y++)
			{
				memcpy(DestBuffer, SrcBuffer, Size);
				DestBuffer += DestLockRect.Pitch;
			}
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid bit count: " << surface.BitCount << " Width: " << FillWidth);
			return DDERR_GENERIC;
		}

		// Unlock surface
		if (!IsUsingEmulation())
		{
			UnLockD3d9Surface(MipMapLevel);
		}
	}

	if (MipMapLevel == 0)
	{
		// Keep surface insync
		EndWriteSyncSurfaces(&DestRect);
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::SaveDXTDataToDDS(const void *data, size_t dataSize, const char *filename, int dxtVersion) const
{
	int blockSize = 0;
	DWORD fourCC = 0;

	switch(dxtVersion)
	{
	case 1:
		blockSize = 8;
		fourCC = '1TXD';
		break;

	case 3:
		blockSize = 16;
		fourCC = '3TXD';
		break;

	case 5:
		blockSize = 16;
		fourCC = '5TXD';
		break;

	default:
		Logging::Log() << __FUNCTION__ << " Error: unsupported DXT version!";
		return D3DERR_INVALIDCALL;
	}

	std::ofstream outFile(filename, std::ios::binary | std::ios::out);
	if (outFile.is_open())
	{
		DDS_HEADER header = {};
		header.dwSize = sizeof(DDS_HEADER);
		header.dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE;
		header.dwHeight = surfaceDesc2.dwHeight;
		header.dwWidth = surfaceDesc2.dwHeight;
		header.dwPitchOrLinearSize = max(1, (surfaceDesc2.dwWidth + 3) / 4) * blockSize;  // 8 for DXT1, 16 for others
		header.dwDepth = 0;
		header.dwMipMapCount = 0;
		header.ddspf.dwSize = sizeof(DDS_PIXELFORMAT);
		header.ddspf.dwFlags = DDPF_FOURCC;
		header.ddspf.dwFourCC = fourCC;
		header.dwCaps = DDSCAPS_TEXTURE;// | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
		header.dwCaps2 = 0x00000000;
		header.dwCaps3 = 0x00000000;
		header.dwCaps4 = 0x00000000;
		header.dwReserved2 = 0;

		outFile.write("DDS ", 4);
		outFile.write((char*)&header, sizeof(DDS_HEADER));
		outFile.write((char*)data, dataSize);
		outFile.close();

		return D3D_OK;
	}

	return DDERR_GENERIC;
}

HRESULT m_IDirectDrawSurfaceX::Load(LPDIRECTDRAWSURFACE7 lpDestTex, LPPOINT lpDestPoint, LPDIRECTDRAWSURFACE7 lpSrcTex, LPRECT lprcSrcRect, DWORD dwFlags)
{
	if (!lpDestTex || !lpSrcTex)
	{
		return  DDERR_INVALIDPARAMS;
	}

	// ToDo: support the following dwFlags: 
	// DDSCAPS2_CUBEMAP_ALLFACES - All faces should be loaded with the image data within the source texture.
	// DDSCAPS2_CUBEMAP_NEGATIVEX, DDSCAPS2_CUBEMAP_NEGATIVEY, or DDSCAPS2_CUBEMAP_NEGATIVEZ
	//     The negative x, y, or z faces should receive the image data.
	// DDSCAPS2_CUBEMAP_POSITIVEX, DDSCAPS2_CUBEMAP_POSITIVEY, or DDSCAPS2_CUBEMAP_POSITIVEZ
	//     The positive x, y, or z faces should receive the image data.

	if (dwFlags)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: flags not supported. dwFlags: " << Logging::hex(dwFlags));
	}

	HRESULT hr;
	if (!lprcSrcRect && (!lpDestPoint || (lpDestPoint && lpDestPoint->x == 0 && lpDestPoint->y == 0)))
	{
		hr = lpDestTex->Blt(nullptr, lpSrcTex, nullptr, 0, nullptr);
	}
	else
	{
		// Get source rect
		RECT SrcRect = {};
		if (lprcSrcRect)
		{
			SrcRect = *lprcSrcRect;
		}
		else
		{
			DDSURFACEDESC2 Desc2 = {};
			Desc2.dwSize = sizeof(DDSURFACEDESC2);
			lpSrcTex->GetSurfaceDesc(&Desc2);

			if ((Desc2.dwFlags & (DDSD_WIDTH | DDSD_HEIGHT)) != (DDSD_WIDTH | DDSD_HEIGHT))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: rect size doesn't match!");
				return DDERR_GENERIC;
			}

			SrcRect = { 0, 0, (LONG)Desc2.dwWidth, (LONG)Desc2.dwHeight };
		}

		// Get destination point
		POINT DestPoint = {};
		if (lpDestPoint)
		{
			DestPoint = *lpDestPoint;
		}

		// Get destination rect
		RECT DestRect = {
			DestPoint.x,									// left
			DestPoint.y,									// top
			DestPoint.x + (SrcRect.right - SrcRect.left),	// right
			DestPoint.y + (SrcRect.bottom - SrcRect.top),	// bottom
		};

		hr = lpDestTex->Blt(&DestRect, lpSrcTex, &SrcRect, 0, nullptr);
	}

	if (SUCCEEDED(hr))
	{
		// Load color key
		m_IDirectDrawSurfaceX* pSrcSurfaceX = nullptr;
		if (SUCCEEDED(lpSrcTex->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pSrcSurfaceX)))
		{
			surfaceDesc2.dwFlags |= pSrcSurfaceX->surfaceDesc2.dwFlags & (DDSD_CKDESTBLT | DDSD_CKDESTOVERLAY | DDSD_CKSRCBLT | DDSD_CKSRCOVERLAY);
			if (surfaceDesc2.dwFlags & DDSD_CKDESTOVERLAY) surfaceDesc2.ddckCKDestOverlay = pSrcSurfaceX->surfaceDesc2.ddckCKDestOverlay;
			if (surfaceDesc2.dwFlags & DDSD_CKDESTBLT) surfaceDesc2.ddckCKDestBlt = pSrcSurfaceX->surfaceDesc2.ddckCKDestBlt;
			if (surfaceDesc2.dwFlags & DDSD_CKSRCOVERLAY) surfaceDesc2.ddckCKSrcOverlay = pSrcSurfaceX->surfaceDesc2.ddckCKSrcOverlay;
			if (surfaceDesc2.dwFlags & DDSD_CKSRCBLT) surfaceDesc2.ddckCKSrcBlt = pSrcSurfaceX->surfaceDesc2.ddckCKSrcBlt;
		}
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::SaveSurfaceToFile(const char *filename, D3DXIMAGE_FILEFORMAT format)
{
	ComPtr<ID3DXBuffer> pDestBuf;
	HRESULT hr = D3DXSaveSurfaceToFileInMemory(pDestBuf.GetAddressOf(), format, Get3DSurface(), nullptr, nullptr);

	if (SUCCEEDED(hr))
	{
		// Save the buffer to a file
		std::ofstream outFile(filename, std::ios::binary | std::ios::out);
		if (outFile.is_open())
		{
			outFile.write((const char*)pDestBuf->GetBufferPointer(), pDestBuf->GetBufferSize());
			outFile.close();
		}
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::CopySurface(m_IDirectDrawSurfaceX* pSourceSurface, RECT* pSourceRect, RECT* pDestRect, D3DTEXTUREFILTERTYPE Filter, D3DCOLOR ColorKey, DWORD dwFlags, DWORD SrcMipMapLevel, DWORD MipMapLevel)
{
	// Check parameters
	if (!pSourceSurface)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: invalid parameters!");
		return DDERR_INVALIDPARAMS;
	}

	// Check for device interface
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
	HRESULT s_hr = (pSourceSurface == this) ? c_hr : pSourceSurface->CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr) || FAILED(s_hr))
	{
		if (c_hr == DDERR_SURFACELOST || s_hr == DDERR_SURFACELOST)
		{
			return DDERR_SURFACELOST;
		}
		return FAILED(c_hr) ? c_hr : s_hr;
	}

	// Get surface desc for mipmap
	DDSURFACEDESC2 SrcDesc2 = {}, DestDesc2 = {};
	SrcDesc2.dwSize = sizeof(DDSURFACEDESC2);
	DestDesc2.dwSize = sizeof(DDSURFACEDESC2);
	pSourceSurface->GetSurfaceDesc2(&SrcDesc2, SrcMipMapLevel, 7);
	GetSurfaceDesc2(&DestDesc2, MipMapLevel, 7);

	// Copy rect and do clipping
	RECT SrcRect = (pSourceRect ? *pSourceRect : RECT{ 0, 0, (LONG)SrcDesc2.dwWidth, (LONG)SrcDesc2.dwHeight });
	RECT DestRect = (pDestRect ? *pDestRect : RECT{ 0, 0, (LONG)DestDesc2.dwWidth, (LONG)DestDesc2.dwHeight });
	LONG Left = min(SrcRect.left, DestRect.left);
	if (Left < 0)
	{
		SrcRect.left -= Left;
		DestRect.left -= Left;
	}
	LONG Top = min(SrcRect.top, DestRect.top);
	if (Top < 0)
	{
		SrcRect.top -= Top;
		DestRect.top -= Top;
	}

	// Get source and dest format
	const D3DFORMAT SrcFormat = pSourceSurface->GetSurfaceFormat();
	const D3DFORMAT DestFormat = GetSurfaceFormat();

	// Check source and destination format
	const bool FormatMismatch = !(SrcFormat == DestFormat || (ISDXTEX(SrcFormat) && ISDXTEX(DestFormat)) ||
		((SrcFormat == D3DFMT_A1R5G5B5 || SrcFormat == D3DFMT_X1R5G5B5) && (DestFormat == D3DFMT_A1R5G5B5 || DestFormat == D3DFMT_X1R5G5B5)) ||
		((SrcFormat == D3DFMT_A4R4G4B4 || SrcFormat == D3DFMT_X4R4G4B4) && (DestFormat == D3DFMT_A4R4G4B4 || DestFormat == D3DFMT_X4R4G4B4)) ||
		((SrcFormat == D3DFMT_A8R8G8B8 || SrcFormat == D3DFMT_X8R8G8B8) && (DestFormat == D3DFMT_A8R8G8B8 || DestFormat == D3DFMT_X8R8G8B8)) ||
		((SrcFormat == D3DFMT_A8B8G8R8 || SrcFormat == D3DFMT_X8B8G8R8) && (DestFormat == D3DFMT_A8B8G8R8 || DestFormat == D3DFMT_X8B8G8R8)));

	// Get copy flags
	const bool IsStretchRect =
		abs((SrcRect.right - SrcRect.left) - (DestRect.right - DestRect.left)) > 1 ||		// Width size
		abs((SrcRect.bottom - SrcRect.top) - (DestRect.bottom - DestRect.top)) > 1;			// Height size
	const bool IsColorKey = ((dwFlags & BLT_COLORKEY) != 0);
	const bool IsMirrorLeftRight = ((dwFlags & BLT_MIRRORLEFTRIGHT) != 0);
	const bool IsMirrorUpDown = ((dwFlags & BLT_MIRRORUPDOWN) != 0);
	const DWORD D3DXFilter =
		(IsStretchRect && IsPalette()) || (Filter & D3DTEXF_POINT) ? D3DX_FILTER_POINT :	// Force palette surfaces to use point filtering to prevent color banding
		(Filter & D3DTEXF_LINEAR) ? D3DX_FILTER_LINEAR :									// Use linear filtering when requested by the application
		(IsStretchRect) ? D3DX_FILTER_POINT :												// Default to point filtering when stretching the rect, same as DirectDraw
		D3DX_FILTER_NONE;

#ifdef ENABLE_PROFILING
	Logging::Log() << __FUNCTION__ << " (" << pSourceSurface << ") -> (" << this << ")" <<
		" StretchRect = " << IsStretchRect <<
		" ColorKey = " << IsColorKey <<
		" MirrorLeftRight = " << IsMirrorLeftRight <<
		" MirrorUpDown = " << IsMirrorUpDown;
#endif

	// Check rect and do clipping
	if (!pSourceSurface->CheckCoordinates(SrcRect, &SrcRect, &SrcDesc2) || !CheckCoordinates(DestRect, &DestRect, &DestDesc2))
	{
		return DDERR_INVALIDRECT;
	}

	// Handle clipper
	if (attachedClipper)
	{
		RECT clipBounds = {};
		if (attachedClipper->GetClipBoundsFromData(clipBounds))
		{
			// Intersect destination rect with clip region
			RECT clippedDest = {};
			if (!IntersectRect(&clippedDest, &DestRect, &clipBounds))
			{
				// Fully clipped - no blit needed
				LOG_LIMIT(100, __FUNCTION__ << " Warning: dest rect is fully clipped!");
				return DD_OK;
			}

			if (IsStretchRect)
			{
				// Calculate scaling factors
				float scaleX = float(SrcRect.right - SrcRect.left) / float(DestRect.right - DestRect.left);
				float scaleY = float(SrcRect.bottom - SrcRect.top) / float(DestRect.bottom - DestRect.top);

				// Lambda function to round up
				auto RoundF = [](float value) {
					return int(value + 0.5f);
					};

				// Adjust source rect proportionally
				SrcRect = {
					SrcRect.left + RoundF((clippedDest.left - DestRect.left) * scaleX),
					SrcRect.top + RoundF((clippedDest.top - DestRect.top) * scaleY),
					SrcRect.left + RoundF((clippedDest.right - DestRect.left) * scaleX),
					SrcRect.top + RoundF((clippedDest.bottom - DestRect.top) * scaleY)
				};
			}
			else
			{
				// Calculate how many pixels were clipped off each side
				SrcRect.left += clippedDest.left - DestRect.left;
				SrcRect.top += clippedDest.top - DestRect.top;
				SrcRect.right -= DestRect.right - clippedDest.right;
				SrcRect.bottom -= DestRect.bottom - clippedDest.bottom;
			}

			// Check if rect is fully clipped
			if (SrcRect.left >= SrcRect.right || SrcRect.top >= SrcRect.bottom)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: source rect is fully clipped!");
				return DD_OK;
			}

			// Adjusted dest rects
			DestRect = clippedDest;
		}
	}

	// Get width and height of rect
	LONG SrcRectWidth = SrcRect.right - SrcRect.left;
	LONG SrcRectHeight = SrcRect.bottom - SrcRect.top;
	LONG DestRectWidth = DestRect.right - DestRect.left;
	LONG DestRectHeight = DestRect.bottom - DestRect.top;

	if (!IsStretchRect)
	{
		Filter = D3DTEXF_NONE;
		SrcRectWidth = min(SrcRectWidth, DestRectWidth);
		SrcRectHeight = min(SrcRectHeight, DestRectHeight);
		DestRectWidth = SrcRectWidth;
		DestRectHeight = SrcRectHeight;
		SrcRect.right = SrcRect.left + SrcRectWidth;
		SrcRect.bottom = SrcRect.top + SrcRectHeight;
		DestRect.right = DestRect.left + DestRectWidth;
		DestRect.bottom = DestRect.top + DestRectHeight;
	}

	// Read surface from GDI
	if (ShouldReadFromGDI())
	{
		CopyEmulatedSurfaceFromGDI(&DestRect);
	}

	// Variables
	HRESULT hr = DDERR_GENERIC;
	bool UnlockSrc = false, UnlockDest = false;
	D3DLOCKED_RECT DestLockRect = {};

	do {
		// Use StretchRect for video memory to prevent copying out of video memory
		if (!IsUsingEmulation() && !IsUsingShadowSurface() && !pSourceSurface->IsUsingShadowSurface() &&
			(pSourceSurface->surface.Pool == D3DPOOL_DEFAULT && surface.Pool == D3DPOOL_DEFAULT) &&
			(pSourceSurface->surface.Type == surface.Type || (pSourceSurface->surface.Type == D3DTYPE_OFFPLAINSURFACE && (surface.Usage & D3DUSAGE_RENDERTARGET))) &&
			(!IsStretchRect || (this != pSourceSurface && !ISDXTEX(SrcFormat) && !ISDXTEX(DestFormat) && (surface.Usage & D3DUSAGE_RENDERTARGET))) &&
			(surface.Type != D3DTYPE_TEXTURE) &&
			(!pSourceSurface->IsPalette() && !IsPalette()) &&
			!IsMirrorLeftRight && !IsMirrorUpDown && !IsColorKey)
		{
			ScopedGetMipMapContext Src(pSourceSurface, SrcMipMapLevel);
			ScopedGetMipMapContext Dest(this, MipMapLevel);

			if (Src.GetSurface() && Dest.GetSurface())
			{
				hr = (*d3d9Device)->StretchRect(Src.GetSurface(), &SrcRect, Dest.GetSurface(), &DestRect, Filter);

				if (hr == D3DERR_INVALIDCALL && Src.GetSurface() == Dest.GetSurface())
				{
					if (!tmpVideo.Surface)
					{
						LOG_LIMIT(100, __FUNCTION__ << " Creating tmpVideo surface.");

						D3DSURFACE_DESC Desc = {};
						Dest.GetSurface()->GetDesc(&Desc);

						if (surface.Type == D3DTYPE_OFFPLAINSURFACE)
						{
							if (FAILED((*d3d9Device)->CreateOffscreenPlainSurface(Desc.Width, Desc.Height, Desc.Format, surface.Pool, &tmpVideo.Surface, nullptr)))
							{
								LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create offplain tmpVideo.Surface. Size: " << Desc.Width << "x" << Desc.Height << " Format: " << Desc.Format);
							}
						}
						else if (surface.Type == D3DTYPE_RENDERTARGET)
						{
							if (FAILED((*d3d9Device)->CreateRenderTarget(Desc.Width, Desc.Height, Desc.Format, D3DMULTISAMPLE_NONE, 0, FALSE, &tmpVideo.Surface, nullptr)))
							{
								LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create render target tmpVideo.Surface. Size: " << Desc.Width << "x" << Desc.Height << " Format: " << Desc.Format);
							}
						}
						else if (surface.Type == D3DTYPE_TEXTURE)
						{
							if (!tmpVideo.Texture && FAILED((*d3d9Device)->CreateTexture(Desc.Width, Desc.Height, 1, surface.Usage, Desc.Format, surface.Pool, &tmpVideo.Texture, nullptr)))
							{
								LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create texture tmpVideo.Texture. Size: " << Desc.Width << "x" << Desc.Height << " Format: " << Desc.Format);
							}
							if (tmpVideo.Texture && FAILED(tmpVideo.Texture->GetSurfaceLevel(0, &tmpVideo.Surface)))
							{
								LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface level for tmpVideo.Texture. Size: " << Desc.Width << "x" << Desc.Height << " Format: " << Desc.Format);
							}
						}
					}

					if (tmpVideo.Surface)
					{
						hr = (*d3d9Device)->StretchRect(Src.GetSurface(), &SrcRect, tmpVideo.Surface, &DestRect, Filter);

						if (FAILED(hr))
						{
							LOG_LIMIT(100, __FUNCTION__ << " Error: failed to StretchRect to tmpVideo.Surface!");
						}

						if (SUCCEEDED(hr))
						{
							hr = (*d3d9Device)->StretchRect(tmpVideo.Surface, &DestRect, Dest.GetSurface(), &DestRect, D3DTEXF_NONE);

							if (FAILED(hr))
							{
								LOG_LIMIT(100, __FUNCTION__ << " Error: failed to StretchRect from tmpVideo.Surface!");
							}
						}
					}
				}

				if (FAILED(hr))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not copy rect: " << SrcDesc2.ddsCaps << " -> " << DestDesc2.ddsCaps << " " <<
						SrcFormat << " -> " << DestFormat << " " << SrcRect << " -> " << DestRect << " " << IsStretchRect << " " <<
						Src.GetSurface() << " -> " << Dest.GetSurface() << " " << (D3DERR)hr);
				}
			}

			if (SUCCEEDED(hr))
			{
				break;
			}
		}

		// Use UpdateSurface for copying system memory to video memory
		if (!IsUsingEmulation() && !IsUsingShadowSurface() && surface.Pool == D3DPOOL_DEFAULT &&
			(pSourceSurface->surface.Pool == D3DPOOL_SYSTEMMEM || pSourceSurface->IsUsingShadowSurface() ||
				(pSourceSurface->surface.Pool == D3DPOOL_MANAGED && surface.Shadow && (surface.BitCount == 8 || surface.BitCount == 16 || surface.BitCount == 24 || surface.BitCount == 32))) &&
			(pSourceSurface->surface.Type != D3DTYPE_DEPTHSTENCIL && surface.Type != D3DTYPE_DEPTHSTENCIL) &&
			(pSourceSurface->surface.Format == surface.Format) &&
			(!pSourceSurface->IsPalette() && !IsPalette()) &&
			!IsStretchRect && !IsMirrorLeftRight && !IsMirrorUpDown && !IsColorKey)
		{
			ScopedGetMipMapContext Src(pSourceSurface, SrcMipMapLevel);
			ScopedGetMipMapContext Dest(this, MipMapLevel);

			if (Src.GetSurface() && Dest.GetSurface())
			{
				if (pSourceSurface->surface.Pool == D3DPOOL_SYSTEMMEM || pSourceSurface->IsUsingShadowSurface())
				{
					hr = (*d3d9Device)->UpdateSurface(Src.GetSurface(), &SrcRect, Dest.GetSurface(), (LPPOINT)&DestRect);
				}
				else
				{
					do {
						D3DLOCKED_RECT SrcLockedRect = {};
						if (FAILED(Src.GetSurface()->LockRect(&SrcLockedRect, &SrcRect, D3DLOCK_READONLY)))
						{
							LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock source surface for update!");
							break;
						}
						D3DLOCKED_RECT DestLockedRect = {};
						if (FAILED(surface.Shadow->LockRect(&DestLockedRect, &DestRect, 0)))
						{
							LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock shadow surface for update!");
							Src.GetSurface()->UnlockRect();
							break;
						}

						BYTE* SrcBytes = (BYTE*)SrcLockedRect.pBits;
						BYTE* DestBytes = (BYTE*)DestLockedRect.pBits;
						size_t Size = DestRectWidth * surface.BitCount / 8;
						for (int x = 0; x < DestRectHeight ; x++)
						{
							memcpy(DestBytes, SrcBytes, Size);
							SrcBytes += SrcLockedRect.Pitch;
							DestBytes += DestLockedRect.Pitch;
						}

						surface.Shadow->UnlockRect();
						Src.GetSurface()->UnlockRect();

						hr = (*d3d9Device)->UpdateSurface(surface.Shadow, &DestRect, Dest.GetSurface(), (LPPOINT)&DestRect);

					} while (false);
				}

				if (FAILED(hr))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not update surface: " << SrcDesc2.ddsCaps << " -> " << DestDesc2.ddsCaps << " " <<
						SrcFormat << " -> " << DestFormat << " " << SrcRect << " -> " << DestRect << " " << IsStretchRect << " " << (D3DERR)hr);
				}
			}

			if (SUCCEEDED(hr))
			{
				break;
			}
		}

		// Check if source render target should use shadow
		if (SrcMipMapLevel == 0 && (pSourceSurface->surface.Usage & D3DUSAGE_RENDERTARGET) && !pSourceSurface->IsUsingShadowSurface())
		{
			pSourceSurface->SetRenderTargetShadow();
		}

		// Check if render target should use shadow
		if (MipMapLevel == 0 && (surface.Usage & D3DUSAGE_RENDERTARGET) && !IsUsingShadowSurface())
		{
			SetRenderTargetShadow();
		}

		// Decode DirectX textures and FourCCs
		if ((FormatMismatch && !IsUsingEmulation()) ||
			(!IsPixelFormatRGB(pSourceSurface->surfaceDesc2.ddpfPixelFormat) && !IsPixelFormatPalette(pSourceSurface->surfaceDesc2.ddpfPixelFormat)))
		{
			if (IsColorKey)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: color key not supported with DirectX textures!");
			}

			if (IsMirrorLeftRight || IsMirrorUpDown)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: mirroring not supported with DirectX textures!");
			}

			if (IsUsingEmulation())
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: copying DirectX textures to emulated surfaces is not supported!");
				hr = DDERR_GENERIC;
				break;
			}

			ScopedGetMipMapContext Src(pSourceSurface, SrcMipMapLevel);
			ScopedGetMipMapContext Dest(this, MipMapLevel);

			if (Src.GetSurface() && Dest.GetSurface())
			{
				hr = D3DXLoadSurfaceFromSurface(Dest.GetSurface(), nullptr, &DestRect, Src.GetSurface(), nullptr, &SrcRect, D3DXFilter, 0);

				if (FAILED(hr))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: could not decode source texture. " << (D3DERR)hr << " " << SrcFormat << "->" << DestFormat);
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get source or destination surface level: " << Src.GetSurface() << "->" << Dest.GetSurface());
			}

			if (SUCCEEDED(hr))
			{
				break;
			}

			break;
		}

		// Use BitBlt/StretchBlt to copy the surface
		if (IsEmulationDCReady() && pSourceSurface->IsEmulationDCReady() && !IsColorKey)
		{
			LONG DestLeft = DestRect.left;
			LONG DestTop = DestRect.top;
			LONG DestWidth = DestRectWidth;
			LONG DestHeight = DestRectHeight;

			if (IsMirrorLeftRight)
			{
				DestLeft = DestRect.right;
				DestWidth = -DestWidth;
			}
			if (IsMirrorUpDown)
			{
				DestTop = DestRect.bottom;
				DestHeight = -DestHeight;
			}

			// Set new palette data
			UpdatePaletteData();
			pSourceSurface->UpdatePaletteData();

			// Set stretch mode
			if (IsStretchRect)
			{
				// After setting the HALFTONE stretching mode, an application must call the SetBrushOrgEx
				// function to set the brush origin. If it fails to do so, brush misalignment occurs.
				POINT org;
				GetBrushOrgEx(surface.emu->DC, &org);
				SetStretchBltMode(surface.emu->DC, (Filter & D3DTEXF_LINEAR) ? HALFTONE : COLORONCOLOR);
				SetBrushOrgEx(surface.emu->DC, org.x, org.y, nullptr);
			}

			if ((IsStretchRect || IsMirrorLeftRight || IsMirrorUpDown) ?
				StretchBlt(surface.emu->DC, DestLeft, DestTop, DestWidth, DestHeight,
					pSourceSurface->surface.emu->DC, SrcRect.left, SrcRect.top, SrcRect.right - SrcRect.left, SrcRect.bottom - SrcRect.top, SRCCOPY) :
				BitBlt(surface.emu->DC, DestRect.left, DestRect.top, DestRectWidth, DestRectHeight,
					pSourceSurface->surface.emu->DC, SrcRect.left, SrcRect.top, SRCCOPY))
			{
				hr = DD_OK;
				break;
			}
		}

		// Use D3DXLoadSurfaceFromSurface to copy the surface
		if (!IsUsingEmulation() && !IsColorKey && !IsMirrorLeftRight && !IsMirrorUpDown &&
			pSourceSurface->surface.Type == surface.Type &&	// D3DXLoadSurfaceFromSurface is very slow when copying from offplain to texture
			!surface.UsingSurfaceMemory && !pSourceSurface->surface.UsingSurfaceMemory &&
			(pSourceSurface->IsPalette() == IsPalette()))
		{
			ScopedGetMipMapContext Src(pSourceSurface, SrcMipMapLevel);
			ScopedGetMipMapContext Dest(this, MipMapLevel);

			if (Src.GetSurface() && Dest.GetSurface())
			{
				hr = D3DXLoadSurfaceFromSurface(Dest.GetSurface(), nullptr, &DestRect, Src.GetSurface(), nullptr, &SrcRect, D3DXFilter, 0);

				if (FAILED(hr))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to load surface from surface. " << (D3DERR)hr);
				}
			}

			if (SUCCEEDED(hr))
			{
				break;
			}
		}

		// Check for format mismatch
		const bool FormatR5G6B5toX8R8G8B8 = (SrcFormat == D3DFMT_R5G6B5 && (DestFormat == D3DFMT_A8R8G8B8 || DestFormat == D3DFMT_X8R8G8B8));
		if (FormatMismatch)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: source and destination formats don't match! " << SrcFormat << "-->" << DestFormat);

			if (!FormatR5G6B5toX8R8G8B8)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: not supported for specified source and destination formats! " << SrcFormat << "-->" << DestFormat);
				hr = DDERR_GENERIC;
				break;
			}
		}

		// Get byte count
		DWORD DestBitCount = surface.BitCount;
		DWORD ByteCount = DestBitCount / 8;
		if (!ByteCount || ByteCount > 4 || DestBitCount % 8 != 0)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: wrong bit count " << DestBitCount);
			hr = DDERR_GENERIC;
			break;
		}

		// Check if source surface is not locked then lock it
		D3DLOCKED_RECT SrcLockRect = {};
		if (FAILED(pSourceSurface->IsUsingEmulation() ? pSourceSurface->LockEmulatedSurface(&SrcLockRect, &SrcRect) :
			pSourceSurface->LockD3d9Surface(&SrcLockRect, &SrcRect, D3DLOCK_READONLY, SrcMipMapLevel)) || !SrcLockRect.pBits)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not lock source surface " << SrcRect);
			hr = (pSourceSurface->IsSurfaceBusy(MipMapLevel)) ? DDERR_SURFACEBUSY : DDERR_GENERIC;
			break;
		}
		UnlockSrc = true;

		// Use seperate memory cache if source and destination formats mismatch or are on the same surface
		if ((pSourceSurface == this && MipMapLevel == SrcMipMapLevel) || FormatMismatch)
		{
			size_t size = SrcRectWidth * ByteCount * SrcRectHeight;
			if (size > ByteArray.size())
			{
				ByteArray.resize(size);
			}
			BYTE* SrcBuffer = (BYTE*)SrcLockRect.pBits;
			BYTE* DestBuffer = (BYTE*)ByteArray.data();
			INT DestPitch = SrcRectWidth * ByteCount;
			if (FormatR5G6B5toX8R8G8B8)
			{
				for (LONG y = 0; y < SrcRectHeight; y++)
				{
					for (LONG x = 0; x < SrcRectWidth; x++)
					{
						((DWORD*)DestBuffer)[x] = D3DFMT_R5G6B5_TO_X8R8G8B8(((WORD*)SrcBuffer)[x]);
					}
					SrcBuffer += SrcLockRect.Pitch;
					DestBuffer += DestPitch;
				}
				ColorKey = D3DFMT_R5G6B5_TO_X8R8G8B8(ColorKey);
			}
			else
			{
				for (LONG y = 0; y < SrcRectHeight; y++)
				{
					memcpy(DestBuffer, SrcBuffer, SrcRectWidth * ByteCount);
					SrcBuffer += SrcLockRect.Pitch;
					DestBuffer += DestPitch;
				}
			}
			SrcLockRect.pBits = ByteArray.data();
			SrcLockRect.Pitch = DestPitch;
			if (UnlockSrc)
			{
				pSourceSurface->IsUsingEmulation() ? DD_OK : pSourceSurface->UnLockD3d9Surface(SrcMipMapLevel);
				UnlockSrc = false;
			}
		}

		// Check if destination surface is not locked then lock it
		if (FAILED(IsUsingEmulation() ? LockEmulatedSurface(&DestLockRect, &DestRect) :
			LockD3d9Surface(&DestLockRect, &DestRect, 0, MipMapLevel)) || !DestLockRect.pBits)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not lock destination surface " << DestRect);
			hr = (IsSurfaceLocked(MipMapLevel)) ? DDERR_SURFACEBUSY : DDERR_GENERIC;
			break;
		}
		UnlockDest = true;

		// Create buffer variables
		BYTE* SrcBuffer = (BYTE*)SrcLockRect.pBits;
		BYTE* DestBuffer = (BYTE*)DestLockRect.pBits;

		// For mirror copy up/down
		INT DestPitch = DestLockRect.Pitch;
		if (IsMirrorUpDown)
		{
			DestPitch = -DestLockRect.Pitch;
			DestBuffer += DestLockRect.Pitch * (DestRectHeight - 1);
		}

		// Simple memory copy (QuickCopy)
		if (!IsStretchRect && !IsColorKey && !IsMirrorLeftRight)
		{
			if (!IsMirrorUpDown && SrcLockRect.Pitch == DestLockRect.Pitch && (DWORD)DestRectWidth == DestDesc2.dwWidth)
			{
				memcpy(DestBuffer, SrcBuffer, DestRectHeight * DestPitch);
			}
			else
			{
				for (LONG y = 0; y < DestRectHeight; y++)
				{
					memcpy(DestBuffer, SrcBuffer, DestRectWidth * ByteCount);
					SrcBuffer += SrcLockRect.Pitch;
					DestBuffer += DestPitch;
				}
			}
			hr = DD_OK;
			break;
		}

		// Simple copy with ColorKey and Mirroring
		if (!IsStretchRect)
		{
			switch (ByteCount)
			{
			case 1:
				SimpleColorKeyCopy<BYTE>((BYTE)ColorKey, SrcBuffer, DestBuffer, SrcLockRect.Pitch, DestPitch, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorLeftRight);
				break;
			case 2:
				SimpleColorKeyCopy<WORD>((WORD)ColorKey, SrcBuffer, DestBuffer, SrcLockRect.Pitch, DestPitch, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorLeftRight);
				break;
			case 3:
				SimpleColorKeyCopy<TRIBYTE>((TRIBYTE)ColorKey, SrcBuffer, DestBuffer, SrcLockRect.Pitch, DestPitch, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorLeftRight);
				break;
			case 4:
				SimpleColorKeyCopy<DWORD>((DWORD)ColorKey, SrcBuffer, DestBuffer, SrcLockRect.Pitch, DestPitch, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorLeftRight);
				break;
			}
			hr = DD_OK;
			break;
		}

		// Copy memory (complex)
		switch (ByteCount)
		{
		case 1:
			ComplexCopy<BYTE>((BYTE)ColorKey, SrcLockRect, DestLockRect, SrcRectWidth, SrcRectHeight, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorUpDown, IsMirrorLeftRight);
			break;
		case 2:
			ComplexCopy<WORD>((WORD)ColorKey, SrcLockRect, DestLockRect, SrcRectWidth, SrcRectHeight, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorUpDown, IsMirrorLeftRight);
			break;
		case 3:
			ComplexCopy<TRIBYTE>((TRIBYTE)ColorKey, SrcLockRect, DestLockRect, SrcRectWidth, SrcRectHeight, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorUpDown, IsMirrorLeftRight);
			break;
		case 4:
			ComplexCopy<DWORD>((DWORD)ColorKey, SrcLockRect, DestLockRect, SrcRectWidth, SrcRectHeight, DestRectWidth, DestRectHeight, IsColorKey, IsMirrorUpDown, IsMirrorLeftRight);
			break;
		}
		hr = DD_OK;
		break;

	} while (false);

	// Remove scanlines before unlocking surface
	if (SUCCEEDED(hr) && Config.DdrawRemoveScanlines && IsPrimaryOrBackBuffer())
	{
		// Set last rect before removing scanlines
		LASTLOCK LLock;
		EmuScanLine.ScanlineWidth = DestRectWidth;
		LLock.Rect = DestRect;
		if (IsUsingEmulation())
		{
			LockEmulatedSurface(&LLock.LockedRect, &DestRect);
			RemoveScanlines(LLock);
		}
		else if (UnlockDest)
		{
			LLock.LockedRect = DestLockRect;
			RemoveScanlines(LLock);
		}
	}

	// Unlock surfaces if needed
	if (UnlockSrc)
	{
		pSourceSurface->IsUsingEmulation() ? DD_OK : pSourceSurface->UnLockD3d9Surface(SrcMipMapLevel);
	}
	if (UnlockDest)
	{
		IsUsingEmulation() ? DD_OK : UnLockD3d9Surface(MipMapLevel);
	}

	if (SUCCEEDED(hr))
	{
		if (MipMapLevel == 0)
		{
			// Keep surface insync
			EndWriteSyncSurfaces(&DestRect);
		}
	}

	// Return
	return hr;
}

HRESULT m_IDirectDrawSurfaceX::CopyZBuffer(m_IDirectDrawSurfaceX* pSourceSurface, RECT* pSourceRect, RECT* pDestRect, bool DepthFill, DWORD DepthColor)
{
	// Check parameters
	if (!pSourceSurface)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: invalid parameters!");
		return DDERR_INVALIDPARAMS;
	}

	// Check for device interface
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, true);
	HRESULT s_hr = (pSourceSurface == this) ? c_hr : pSourceSurface->CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr) || FAILED(s_hr))
	{
		if (c_hr == DDERR_SURFACELOST || s_hr == DDERR_SURFACELOST)
		{
			return DDERR_SURFACELOST;
		}
		return FAILED(c_hr) ? c_hr : s_hr;
	}

	// Check rect
	RECT SrcRect = {}, DestRect = {};
	if (!pSourceSurface->CheckCoordinates(SrcRect, pSourceRect, &pSourceSurface->surfaceDesc2) || !CheckCoordinates(DestRect, pDestRect, &surfaceDesc2))
	{
		return DDERR_INVALIDRECT;
	}

	// Check dest is memory pool
	if (surface.Pool == D3DPOOL_SYSTEMMEM && (pSourceSurface->surface.Pool != D3DPOOL_SYSTEMMEM || pSourceSurface->surface.Format != surface.Format))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: destination surface cannot be system memory for mismatching surfaces!");
		return DDERR_UNSUPPORTED;
	}

	bool VideoMemoryBlt = (pSourceSurface->surface.Pool == D3DPOOL_DEFAULT && surface.Pool == D3DPOOL_DEFAULT);

	// zBuffer fill value
	float depthValue = (DepthColor & 0xFFFF0000) ?
		static_cast<float>(DepthColor) / static_cast<float>(0xFFFFFFFF) :
		static_cast<float>(DepthColor & 0xFFFF) / static_cast<float>(0xFFFF);
	depthValue = CLAMP(depthValue, 0.0f, 1.0f);

	// Check conditions for copying Z-buffer
	if (!DepthFill)
	{
		// Check if source and dest surfaces are the same
		if (pSourceSurface == this)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: source and destination surfaces cannot be the same!");
			return DDERR_UNSUPPORTED;
		}

		// Check if source and dest surfaces are the same
		if (pSourceSurface->surface.Format != surface.Format)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: source and destination surfaces must be the same format: " << pSourceSurface->surface.Format << " -> " << surface.Format);
			return DDERR_UNSUPPORTED;
		}

		// Check for sub-rectangle copies
		if (VideoMemoryBlt &&
			(DestRect.left || DestRect.top || DestRect.right < (LONG)surface.Width || DestRect.bottom < (LONG)surface.Height ||
				SrcRect.left || SrcRect.top || SrcRect.right != DestRect.right || SrcRect.bottom != DestRect.bottom ||
				SrcRect.right < (LONG)pSourceSurface->surface.Width || SrcRect.bottom < (LONG)pSourceSurface->surface.Height))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: depth buffer copy cannot copy sub-rectangles!");
			return DDERR_UNSUPPORTED;
		}

		// Check if rect is being stretched
		if (abs((SrcRect.right - SrcRect.left) - (DestRect.right - DestRect.left)) > 1 ||		// Width size
			abs((SrcRect.bottom - SrcRect.top) - (DestRect.bottom - DestRect.top)) > 1)			// Height size
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: stretched rect not supported!");
			return DDERR_UNSUPPORTED;
		}

		// Check BitCount
		if (pSourceSurface->surface.BitCount != 16 && pSourceSurface->surface.BitCount != 24 && pSourceSurface->surface.BitCount != 32)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid BitCount for source depth buffer: " << pSourceSurface->surface.BitCount);
			return DDERR_UNSUPPORTED;
		}
	}

	// Do rect clipping
	LONG Width = min(SrcRect.right - SrcRect.left, DestRect.right - DestRect.left);
	LONG Height = min(SrcRect.bottom - SrcRect.top, DestRect.bottom - DestRect.top);
	SrcRect.right = SrcRect.left + Width;
	SrcRect.bottom = SrcRect.top + Height;
	DestRect.right = DestRect.left + Width;
	DestRect.bottom = DestRect.top + Height;

	// Handle system memory copy
	if (surface.Pool == D3DPOOL_SYSTEMMEM)
	{
		if (DepthFill)
		{
			return ColorFill(&DestRect, GetDepthFillValue(depthValue, surface.Format), 0);
		}
		else
		{
			// Lock source
			D3DLOCKED_RECT SrcLockedRect = {}, DestLockedRect = {};
			if (FAILED(pSourceSurface->surface.Surface->LockRect(&SrcLockedRect, &SrcRect, D3DLOCK_READONLY)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock source surface!");
				return DDERR_GENERIC;
			}

			// Lock dest
			if (FAILED(surface.Surface->LockRect(&DestLockedRect, &DestRect, 0)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock destination surface!");
				pSourceSurface->surface.Surface->UnlockRect();
				return DDERR_GENERIC;
			}

			DWORD CopyPitch = min(SrcLockedRect.Pitch, DestLockedRect.Pitch);

			BYTE* SrcBuffer = reinterpret_cast<BYTE*>(SrcLockedRect.pBits);
			BYTE* DestBuffer = reinterpret_cast<BYTE*>(DestLockedRect.pBits);

			// Copy data
			for (int x = 0; x < Height; ++x)
			{
				memcpy(DestBuffer, SrcBuffer, CopyPitch);
				SrcBuffer += SrcLockedRect.Pitch;
				DestBuffer += DestLockedRect.Pitch;
			}

			// Unlock
			surface.Surface->UnlockRect();
			pSourceSurface->surface.Surface->UnlockRect();

			return DD_OK;
		}
	}

	// Handle video memory copy
	if (!DepthFill && VideoMemoryBlt)
	{
		/*bool InScene = ddrawParent->IsInScene();
		if (InScene)
		{
			(*d3d9Device)->EndScene();
		}

		HRESULT hr = (*d3d9Device)->StretchRect(pSourceSurface->surface.Surface, nullptr, surface.Surface, nullptr, D3DTEXF_NONE);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not copy depth buffer: " << pSourceSurface->surfaceDesc2.ddsCaps << " -> " << surfaceDesc2.ddsCaps << " " <<
				pSourceSurface->surface.Format << " -> " << surface.Format << " " << (D3DERR)hr);
			hr = DDERR_GENERIC;
		}

		if (InScene)
		{
			(*d3d9Device)->BeginScene();
		}

		return hr;*/

		// Just return not supported for now
		LOG_LIMIT(100, __FUNCTION__ << " Error: video memory zbuffer Blt not implemented!");
		return DDERR_UNSUPPORTED;
	}

	ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

	bool IsUsingCurrentZBuffer =
		(ddrawParent->GetDepthStencilSurface() != this && ddrawParent->GetDepthStencilSurface() != GetAttachedDepthStencil());

	// Set new depth stencil
	ComPtr<IDirect3DSurface9> pDepthStencil = nullptr;
	if (!IsUsingCurrentZBuffer)
	{
		HRESULT hr = (*d3d9Device)->GetDepthStencilSurface(pDepthStencil.GetAddressOf());
		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get depth buffer: " << (DDERR)hr);
			return DDERR_GENERIC;
		}

		hr = (*d3d9Device)->SetDepthStencilSurface(surface.Surface);
		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to set depth buffer: " << (DDERR)hr);
			return DDERR_GENERIC;
		}
	}

	// Query surface sizes
	ComPtr<IDirect3DSurface9> pRenderTarget;
	D3DSURFACE_DESC rtDesc = {}, dsDesc = {};
	surface.Surface->GetDesc(&dsDesc);

	if (SUCCEEDED((*d3d9Device)->GetRenderTarget(0, pRenderTarget.GetAddressOf())))
	{
		pRenderTarget->GetDesc(&rtDesc);
	}

	// Check for mismatch
	if (rtDesc.Width != dsDesc.Width || rtDesc.Height != dsDesc.Height)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: render target (" << rtDesc.Width << "x" << rtDesc.Height
			<< ") and depth buffer (" << dsDesc.Width << "x" << dsDesc.Height << ") dimensions do not match!");
	}

	// Get current viewport
	D3DVIEWPORT9 Viewport = {};
	(*d3d9Device)->GetViewport(&Viewport);

	// Set new viewport
	{
		D3DVIEWPORT9 NewViewport = { 0, 0, dsDesc.Width, dsDesc.Height, 0.0f, 1.0f };
		(*d3d9Device)->SetViewport(&NewViewport);
	}

	HRESULT hr = DD_OK;

	// Depth stencil fill
	if (DepthFill)
	{
		if (pDestRect)
		{
			pDestRect = &DestRect;
		}

		hr = (*d3d9Device)->Clear(pDestRect ? 1 : 0, (D3DRECT*)pDestRect, D3DCLEAR_ZBUFFER, 0, depthValue, 0);
		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to fill depth buffer: " << (DDERR)hr);
		}
	}
	// Copy depth stencil
	else
	{
		switch (pSourceSurface->surface.BitCount)
		{
		case 16:
			hr = ComplexZBufferCopy<WORD>(*d3d9Device, pSourceSurface->surface.Surface, SrcRect, DestRect, surface.Format);
			break;
		case 24:	// Depth surfaces are always padded to multiples of 4 bytes per pixel. 
		case 32:
			hr = ComplexZBufferCopy<DWORD>(*d3d9Device, pSourceSurface->surface.Surface, SrcRect, DestRect, surface.Format);
			break;
		}
	}

	// Reset viewport
	(*d3d9Device)->SetViewport(&Viewport);

	// Reset depth stencil
	if (!IsUsingCurrentZBuffer)
	{
		(*d3d9Device)->SetDepthStencilSurface(pDepthStencil.Get());
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::CopyToDrawTexture(LPRECT lpDestRect)
{
	if (!surface.DrawTexture || !surface.Texture)
	{
		return DDERR_GENERIC;
	}

	IDirect3DSurface9* SrcSurface = Get3DMipMapSurface(0);
	ComPtr<IDirect3DSurface9> DestSurface;
	if (!SrcSurface || FAILED(surface.DrawTexture->GetSurfaceLevel(0, DestSurface.GetAddressOf())))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface texture!");
		return DDERR_GENERIC;
	}

	// Get color key
	DWORD ColorKey = 0;
	if (surfaceDesc2.dwFlags & DDSD_CKSRCBLT)
	{
		if (IsPalette())
		{
			UpdatePaletteData();
			if (surface.PaletteEntryArray)
			{
				PALETTEENTRY PaletteEntry = surface.PaletteEntryArray[surfaceDesc2.ddckCKSrcBlt.dwColorSpaceLowValue & 0xFF];
				ColorKey = D3DCOLOR_ARGB(PaletteEntry.peFlags, PaletteEntry.peRed, PaletteEntry.peGreen, PaletteEntry.peBlue);
			}
		}
		else if (surfaceDesc2.ddpfPixelFormat.dwRGBBitCount)
		{
			ColorKey = GetARGBColorKey(surfaceDesc2.ddckCKSrcBlt.dwColorSpaceLowValue, surfaceDesc2.ddpfPixelFormat);
		}
	}

	if (FAILED(D3DXLoadSurfaceFromSurface(DestSurface.Get(), nullptr, lpDestRect, SrcSurface, surface.PaletteEntryArray, lpDestRect, D3DX_FILTER_NONE, ColorKey)))
	{
		Logging::Log() << __FUNCTION__ " Error: failed to copy data from surface: " << surface.Format << " " << (void*)ColorKey << " " << lpDestRect;

		return DDERR_GENERIC;
	}

	// Phase A.7: continuous content capture. Hashes the freshly-copied mip 0
	// bytes (the layer the Remix bridge sees on upload). Dumps PNG + manifest
	// row for each new content-hash this session. See ContentCaptureState block
	// near the top of this file for full notes.
	if (Config.DdrawContentCapture)
	{
		// Pass `this` (the wrapper) so TEX_HASH log matches DRAW_XYZRHW tex= field
		CaptureSurfaceContent(DestSurface.Get(), this);
	}

	// [may20-minus-mip] REMOVED the unconditional D3DXFilterTexture(D3DX_FILTER_BOX)
	// mip regeneration added 2026-05-18. It box-filters the lower mips of every
	// multi-mip draw texture on every copy; on animated torch surfaces this
	// corrupts the lower LODs -> torches disappear by distance/angle. This is the
	// suspected disappearing-torch regression; everything else in the working
	// May-20 build (RTX camera path, A.10) is preserved.

	surface.IsDrawTextureDirty = false;

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::LoadSurfaceFromMemory(LPDIRECT3DSURFACE9 pDestSurface, const RECT& Rect, LPCVOID pSrcMemory, D3DFORMAT SrcFormat, UINT SrcPitch)
{
	if (!pDestSurface)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: destination surface is NULL!");
		return DDERR_GENERIC;
	}

	if (!pSrcMemory)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: source memory is NULL!");
		return DDERR_GENERIC;
	}

	// Get actual surface format
	D3DSURFACE_DESC Desc = {};
	if (FAILED(pDestSurface->GetDesc(&Desc)))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get surface description!");
		return DDERR_GENERIC;
	}

	// Ensure bit counts match for manual copy
	const UINT SrcBitCount = GetBitCount(SrcFormat);
	const UINT DestBitCount = GetBitCount(Desc.Format);

	if (SrcBitCount == DestBitCount && !(Desc.Usage & D3DUSAGE_RENDERTARGET) && (SrcFormat == Desc.Format || GetFailoverFormat(SrcFormat) == Desc.Format))
	{
		// Lock destination surface
		D3DLOCKED_RECT LockedRect = {};
		if (FAILED(pDestSurface->LockRect(&LockedRect, nullptr, 0)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to lock destination surface!");
			return DDERR_GENERIC;
		}

		// Calculate bytes per pixel
		const LONG BytesPerPixel = SrcBitCount / 8;

		// Validate rectangle dimensions
		if (Rect.left < 0 || Rect.top < 0 ||
			Rect.right >(LONG)Desc.Width || Rect.bottom >(LONG)Desc.Height)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid rectangle dimensions!");
			pDestSurface->UnlockRect();
			return DDERR_INVALIDRECT;
		}

		// Calculate source and destination buffers
		const BYTE* SrcBuffer = (const BYTE*)pSrcMemory + (SrcPitch * Rect.top) + (BytesPerPixel * Rect.left);
		BYTE* DestBuffer = (BYTE*)LockedRect.pBits + (LockedRect.Pitch * Rect.top) + (BytesPerPixel * Rect.left);

		// Check dest buffer
		if (!DestBuffer || !SrcBuffer)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: source or destination buffer is null!");
			pDestSurface->UnlockRect();
			return DDERR_GENERIC;
		}

		// Calculate copy pitch and height
		const LONG CopyPitch = (Rect.right - Rect.left) == (LONG)Desc.Width
			? min(LockedRect.Pitch, (INT)SrcPitch)
			: (Rect.right - Rect.left) * BytesPerPixel;
		const LONG CopyHeight = Rect.bottom - Rect.top;

		// Copy surface data row by row
		for (LONG row = 0; row < CopyHeight; ++row)
		{
			memcpy(DestBuffer, SrcBuffer, CopyPitch);
			SrcBuffer += SrcPitch;
			DestBuffer += LockedRect.Pitch;
		}

		// Unlock destination surface
		pDestSurface->UnlockRect();
	}
	else
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: using slower D3DXLoadSurfaceFromMemory for copy: " << SrcFormat << "->" << Desc.Format);

		// Use D3DXLoadSurfaceFromMemory for format conversion
		if (FAILED(D3DXLoadSurfaceFromMemory(pDestSurface, nullptr, &Rect, pSrcMemory, SrcFormat, SrcPitch, nullptr, &Rect, D3DX_FILTER_NONE, 0)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not copy surface using D3DXLoadSurfaceFromMemory!");
			return DDERR_GENERIC;
		}
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::CopyFromEmulatedSurface(LPRECT lpDestRect)
{
	if (!IsUsingEmulation())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not using emulation!");
		return DDERR_GENERIC;
	}

	// Update rect
	RECT DestRect = {};
	if (!CheckCoordinates(DestRect, lpDestRect, nullptr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
		return DDERR_INVALIDRECT;
	}

	// Get real d3d9 surface
	IDirect3DSurface9* pDestSurfaceD9 = Get3DMipMapSurface(0);
	if (!pDestSurfaceD9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not get real surface!");
		return DDERR_GENERIC;
	}

	// LoadSurfaceFromMemory to copy to the surface
	if (FAILED(LoadSurfaceFromMemory(pDestSurfaceD9, DestRect, surface.emu->pBits, (surface.Format == D3DFMT_P8) ? D3DFMT_L8 : surface.Format, surface.emu->Pitch)))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not copy emulated surface: " << surface.Format);
		return DDERR_GENERIC;
	}

	// Update palette surface data
	if (!surface.DisplayTexture || FAILED(CopyEmulatedPaletteSurface(&DestRect)))
	{
		surface.IsPaletteDirty = IsPalette();
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::CopyToEmulatedSurface(LPRECT lpDestRect)
{
	if (!IsUsingEmulation())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not using emulation!");
		return DDERR_GENERIC;
	}

	// Update rect
	RECT DestRect = {};
	if (!CheckCoordinates(DestRect, lpDestRect, nullptr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
		return DDERR_INVALIDRECT;
	}

	// Get lock for emulated surface
	D3DLOCKED_RECT EmulatedLockRect = {};
	if (FAILED(LockEmulatedSurface(&EmulatedLockRect, &DestRect)))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not get emulated surface lock!");
		return DDERR_GENERIC;
	}

	// Get lock for real surface
	D3DLOCKED_RECT SrcLockRect = {};
	if (FAILED(LockD3d9Surface(&SrcLockRect, &DestRect, D3DLOCK_READONLY, 0)))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: could not lock destination surface " << DestRect);
		return (IsSurfaceLocked()) ? DDERR_SURFACEBUSY : DDERR_GENERIC;
	}

	// Create buffer variables
	BYTE* EmulatedBuffer = (BYTE*)EmulatedLockRect.pBits;
	BYTE* SurfaceBuffer = (BYTE*)SrcLockRect.pBits;

	DWORD Height = (DestRect.bottom - DestRect.top);
	INT WidthPitch = min(SrcLockRect.Pitch, EmulatedLockRect.Pitch);

	HRESULT hr = DD_OK;

	// Copy real surface data to emulated surface
	switch ((DWORD)surface.Format)
	{
	case D3DFMT_X4R4G4B4:
	case D3DFMT_A4R4G4B4:
		for (LONG x = DestRect.top; x < DestRect.bottom; x++)
		{
			WORD* EmulatedBufferLoop = (WORD*)EmulatedBuffer;
			DWORD* SurfaceBufferLoop = (DWORD*)SurfaceBuffer;
			for (LONG y = DestRect.left; y < DestRect.right; y++)
			{
				*EmulatedBufferLoop = D3DFMT_A8R8G8B8_TO_A4R4G4B4(*SurfaceBufferLoop);
				EmulatedBufferLoop++;
				SurfaceBufferLoop++;
			}
			EmulatedBuffer += EmulatedLockRect.Pitch;
			SurfaceBuffer += SrcLockRect.Pitch;
		}
		break;
	case D3DFMT_R8G8B8:
		for (LONG x = DestRect.top; x < DestRect.bottom; x++)
		{
			TRIBYTE* EmulatedBufferLoop = (TRIBYTE*)EmulatedBuffer;
			DWORD* SurfaceBufferLoop = (DWORD*)SurfaceBuffer;
			for (LONG y = DestRect.left; y < DestRect.right; y++)
			{
				*EmulatedBufferLoop = *(TRIBYTE*)SurfaceBufferLoop;
				EmulatedBufferLoop++;
				SurfaceBufferLoop++;
			}
			EmulatedBuffer += EmulatedLockRect.Pitch;
			SurfaceBuffer += SrcLockRect.Pitch;
		}
		break;
	case D3DFMT_B8G8R8:
		for (LONG x = DestRect.top; x < DestRect.bottom; x++)
		{
			TRIBYTE* EmulatedBufferLoop = (TRIBYTE*)EmulatedBuffer;
			DWORD* SurfaceBufferLoop = (DWORD*)SurfaceBuffer;
			for (LONG y = DestRect.left; y < DestRect.right; y++)
			{
				DWORD Pixel = D3DFMT_X8R8G8B8_TO_B8G8R8(*SurfaceBufferLoop);
				*EmulatedBufferLoop = *(TRIBYTE*)&Pixel;
				EmulatedBufferLoop++;
				SurfaceBufferLoop++;
			}
			EmulatedBuffer += EmulatedLockRect.Pitch;
			SurfaceBuffer += SrcLockRect.Pitch;
		}
		break;
	case D3DFMT_X8B8G8R8:
	case D3DFMT_A8B8G8R8:
		for (LONG x = DestRect.top; x < DestRect.bottom; x++)
		{
			DWORD* EmulatedBufferLoop = (DWORD*)EmulatedBuffer;
			DWORD* SurfaceBufferLoop = (DWORD*)SurfaceBuffer;
			for (LONG y = DestRect.left; y < DestRect.right; y++)
			{
				*EmulatedBufferLoop = D3DFMT_A8R8G8B8_TO_A8B8G8R8(*SurfaceBufferLoop);
				EmulatedBufferLoop++;
				SurfaceBufferLoop++;
			}
			EmulatedBuffer += EmulatedLockRect.Pitch;
			SurfaceBuffer += SrcLockRect.Pitch;
		}
		break;
	default:
		if (SrcLockRect.Pitch == EmulatedLockRect.Pitch && (DWORD)(DestRect.right - DestRect.left) == surfaceDesc2.dwWidth)
		{
			memcpy(EmulatedBuffer, SurfaceBuffer, SrcLockRect.Pitch * Height);
		}
		else if (surface.emu->bmi->bmiHeader.biBitCount == surface.BitCount)
		{
			for (UINT x = 0; x < Height; x++)
			{
				memcpy(EmulatedBuffer, SurfaceBuffer, WidthPitch);
				EmulatedBuffer += EmulatedLockRect.Pitch;
				SurfaceBuffer += SrcLockRect.Pitch;
			}
		}
		else
		{
			hr = DDERR_GENERIC;
			LOG_LIMIT(100, __FUNCTION__ << " Error: emulated surface format not supported: " << surface.Format);
		}
	}

	// Unlock surface
	UnLockD3d9Surface(0);

	// Update palette surface data
	if (!surface.DisplayTexture || FAILED(CopyEmulatedPaletteSurface(&DestRect)))
	{
		surface.IsPaletteDirty = IsPalette();
	}

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::CopyEmulatedPaletteSurface(LPRECT lpDestRect)
{
	if (!IsPalette() || !d3d9Device || !*d3d9Device)
	{
		return DDERR_GENERIC;
	}

	if (!IsUsingEmulation())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not using emulation!");
		return DDERR_GENERIC;
	}

	HRESULT hr = DD_OK;

	do {
		ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

		// Set new palette data
		UpdatePaletteData();

		// Check for palette entry data
		if (!surface.PaletteEntryArray)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get palette data!");
			hr = DDERR_GENERIC;
			break;
		}

		// Create emulated texture for palettes
		if (!surface.DisplayTexture)
		{
			const D3DPOOL TexturePool = IsPrimaryOrBackBuffer() ? D3DPOOL_MANAGED :
				(surfaceDesc2.ddsCaps.dwCaps & DDSCAPS_SYSTEMMEMORY) ? D3DPOOL_SYSTEMMEM : D3DPOOL_MANAGED;
			const DWORD Width = surface.Width;
			const DWORD Height = surfaceDesc2.dwHeight;
			LOG_LIMIT(3, __FUNCTION__ << " Creating palette display surface texture. Size: " << Width << "x" << Height << " dwCaps: " << surfaceDesc2.ddsCaps);
			if (FAILED(((*d3d9Device)->CreateTexture(Width, Height, 1, 0, D3DFMT_X8R8G8B8, TexturePool, &surface.DisplayTexture, nullptr))))
			{
				// Try failover format
				if (FAILED(((*d3d9Device)->CreateTexture(Width, Height, 1, 0, GetFailoverFormat(D3DFMT_X8R8G8B8), TexturePool, &surface.DisplayTexture, nullptr))))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: failed to create palette display surface texture. Size: " << Width << "x" << Height << " Format: " << D3DFMT_X8R8G8B8 << " dwCaps: " << surfaceDesc2.ddsCaps);
					hr = DDERR_GENERIC;
					break;
				}
			}
		}

		// Update rect, if palette surface is dirty then update the whole surface
		RECT DestRect = {};
		if (!CheckCoordinates(DestRect, (surface.IsPaletteDirty ? nullptr : lpDestRect), nullptr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
			hr = DDERR_INVALIDRECT;
			break;
		}

		// Get palette display context surface
		if (!surface.DisplayContext)
		{
			if (FAILED(surface.DisplayTexture->GetSurfaceLevel(0, &surface.DisplayContext)))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get palette display context surface!");
				hr = DDERR_GENERIC;
				break;
			}
		}

		// Use LoadSurfaceFromMemory to copy to the surface
		if (FAILED(LoadSurfaceFromMemory(surface.DisplayContext, DestRect, surface.emu->pBits, D3DFMT_P8, surface.emu->Pitch)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: could not copy palette display texture: " << surface.Format);
			hr = DDERR_GENERIC;
			break;
		}

		// Reset palette texture dirty flag
		surface.IsPaletteDirty = false;

	} while (false);

	return hr;
}

HRESULT m_IDirectDrawSurfaceX::GetFlipList(std::vector<m_IDirectDrawSurfaceX*>& FlipList, LPDIRECTDRAWSURFACE7 lpDDSurfaceTargetOverride)
{
	FlipList.push_back(this);

	// If SurfaceTargetOverride then use that surface
	if (lpDDSurfaceTargetOverride)
	{
		m_IDirectDrawSurfaceX* lpTargetSurface = nullptr;

		lpDDSurfaceTargetOverride->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpTargetSurface);

		// Check backbuffer
		HRESULT hr = CheckBackBufferForFlip(lpTargetSurface);
		if (FAILED(hr))
		{
			return hr;
		}

		FlipList.push_back(lpTargetSurface);
	}
	// Get list for all attached surfaces
	else
	{
		m_IDirectDrawSurfaceX* lpTargetSurface = this;
		do {
			DWORD dwCaps = 0;
			m_IDirectDrawSurfaceX* lpNewTargetSurface = nullptr;

			// Loop through each surface
			for (auto& it : lpTargetSurface->AttachedSurfaceMap)
			{
				dwCaps = it.second.pSurface->GetSurfaceCaps().dwCaps;
				if (dwCaps & DDSCAPS_FLIP)
				{
					lpNewTargetSurface = it.second.pSurface;
					break;
				}
			}
			lpTargetSurface = lpNewTargetSurface;

			// Stop looping when frontbuffer is found
			if (!lpTargetSurface || lpTargetSurface == this || dwCaps & DDSCAPS_FRONTBUFFER)
			{
				break;
			}

			// Check backbuffer
			HRESULT hr = CheckBackBufferForFlip(lpTargetSurface);
			if (FAILED(hr))
			{
				return hr;
			}

			// Add target surface to list
			FlipList.push_back(lpTargetSurface);

		} while (true);
	}

	return DD_OK;
}

void m_IDirectDrawSurfaceX::CopyGDIToPrimaryAndBackbuffer()
{
	if (Config.D3d9to9Ex || Config.EnableWindowMode)
	{
		return;
	}

	// Check for device interface
	HRESULT c_hr = CheckInterface(__FUNCTION__, true, true, false);
	if (FAILED(c_hr))
	{
		return;
	}

	// Create flip list
	std::vector<m_IDirectDrawSurfaceX*> FlipList;
	if (FAILED(GetFlipList(FlipList, nullptr)))
	{
		return;
	}

	HWND hWnd = ddrawParent->GetPresentationHwnd();
	if (!IsWindow(hWnd) || IsIconic(hWnd))
	{
		Logging::Log() << __FUNCTION__ << " Error: Failed to copy from GDI after device lost!";
		return;
	}

	HDC hdcWindow = ::GetDC(nullptr);
	if (!hdcWindow)
	{
		Logging::Log() << __FUNCTION__ << " Error: Failed to get DC after device lost!";
		return;
	}

	for (auto& entry : FlipList)
	{
		IDirect3DSurface9* pSurface = entry->Get3DSurface();
		if (!pSurface)
		{
			Logging::Log() << __FUNCTION__ << " Error: Failed to get primary surface after device lost!";
			break;
		}

		HDC hdcSurface = nullptr;
		if (FAILED(pSurface->GetDC(&hdcSurface)))
		{
			Logging::Log() << __FUNCTION__ << " Error: Failed to get DC from surface after device lost!";
			break;
		}

		if (!BitBlt(hdcSurface, 0, 0, surface.Width, surface.Height, hdcWindow, 0, 0, SRCCOPY))
		{
			Logging::Log() << __FUNCTION__ << " Error: Failed to BitBlt GDI to surface after device lost!";
		}
		entry->LostDeviceBackup.clear();

		pSurface->ReleaseDC(hdcSurface);
	}

	::ReleaseDC(hWnd, hdcWindow);
}

HRESULT m_IDirectDrawSurfaceX::CopyEmulatedSurfaceFromGDI(LPRECT lpDestRect)
{
	if (!IsUsingEmulation())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not using emulation!");
		return DDERR_GENERIC;
	}

	// Check for forground window
	HWND DDraw_hWnd = ddrawParent->GetHwnd();
	HWND Forground_hWnd = Utils::GetTopLevelWindowOfCurrentProcess();
	bool UsingForgroundWindow = (DDraw_hWnd != Forground_hWnd && Utils::IsWindowRectEqualOrLarger(Forground_hWnd, DDraw_hWnd));

	// Get hWnd
	HWND hWnd = (UsingForgroundWindow) ? Forground_hWnd : DDraw_hWnd;
	if (!hWnd || !DDraw_hWnd)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Cannot get window handle!");
		return DDERR_GENERIC;
	}

	// Check for iconic window
	if (IsIconic(hWnd) || IsIconic(DDraw_hWnd))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Window is iconic!");
		return DDERR_GENERIC;
	}

	// Update rect
	RECT DestRect = {};
	if (!CheckCoordinates(DestRect, lpDestRect, nullptr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
		return DDERR_INVALIDRECT;
	}

	// Clip rect
	RECT ClientRect = {};
	if (GetClientRect(DDraw_hWnd, &ClientRect) && MapWindowPoints(DDraw_hWnd, HWND_DESKTOP, (LPPOINT)&ClientRect, 2))
	{
		DestRect.left = max(DestRect.left, ClientRect.left);
		DestRect.top = max(DestRect.top, ClientRect.top);
		DestRect.right = min(DestRect.right, ClientRect.right);
		DestRect.bottom = min(DestRect.bottom, ClientRect.bottom);
	}

	// Validate rect
	if (DestRect.left >= DestRect.right || DestRect.top >= DestRect.bottom)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid DestRect: " << DestRect);
		return DDERR_GENERIC;
	}

	// Get rect size
	RECT MapRect = DestRect;
	MapWindowPoints(hWnd, HWND_DESKTOP, (LPPOINT)&MapRect, 2);

	// Get hdc
	HDC hdc = (UsingForgroundWindow) ? ::GetDC(hWnd) : ddrawParent->GetDC();
	if (!hdc)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Cannot get window DC!");
		return DDERR_GENERIC;
	}

	// Set new palette data
	UpdatePaletteData();

	// Blt to GDI
	BitBlt(surface.emu->DC, MapRect.left, MapRect.top, MapRect.right - MapRect.left, MapRect.bottom - MapRect.top, hdc, DestRect.left, DestRect.top, SRCCOPY);

	// Release DC
	if (UsingForgroundWindow)
	{
		::ReleaseDC(hWnd, hdc);
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::CopyEmulatedSurfaceToGDI(LPRECT lpDestRect)
{
	if (!IsUsingEmulation())
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: surface is not using emulation!");
		return DDERR_GENERIC;
	}

	// Check for forground window
	HWND DDraw_hWnd = ddrawParent->GetHwnd();
	HWND Forground_hWnd = Utils::GetTopLevelWindowOfCurrentProcess();
	bool UsingForgroundWindow = (DDraw_hWnd != Forground_hWnd && Utils::IsWindowRectEqualOrLarger(Forground_hWnd, DDraw_hWnd));

	// Get hWnd
	HWND hWnd = (UsingForgroundWindow) ? Forground_hWnd : DDraw_hWnd;
	if (!hWnd || !DDraw_hWnd)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Cannot get window handle!");
		return DDERR_GENERIC;
	}

	// Check for iconic window
	if (IsIconic(hWnd) || IsIconic(DDraw_hWnd))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Window is iconic!");
		return DDERR_GENERIC;
	}

	// Update rect
	RECT DestRect = {};
	if (!CheckCoordinates(DestRect, lpDestRect, nullptr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << lpDestRect);
		return DDERR_INVALIDRECT;
	}

	// Clip rect
	RECT ClientRect = {};
	if (GetClientRect(DDraw_hWnd, &ClientRect) && MapWindowPoints(DDraw_hWnd, HWND_DESKTOP, (LPPOINT)&ClientRect, 2))
	{
		DestRect.left = max(DestRect.left, ClientRect.left);
		DestRect.top = max(DestRect.top, ClientRect.top);
		DestRect.right = min(DestRect.right, ClientRect.right);
		DestRect.bottom = min(DestRect.bottom, ClientRect.bottom);
	}

	// Validate rect
	if (DestRect.left >= DestRect.right || DestRect.top >= DestRect.bottom)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << DestRect);
		return DDERR_GENERIC;
	}

	// Get rect size
	RECT MapRect = DestRect;
	MapWindowPoints(HWND_DESKTOP, hWnd, (LPPOINT)&MapRect, 2);

	// Get hdc
	HDC hdc = (UsingForgroundWindow) ? ::GetDC(hWnd) : ddrawParent->GetDC();
	if (!hdc)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Cannot get window DC!");
		return DDERR_GENERIC;
	}

	// Set new palette data
	UpdatePaletteData();

	// Blt to GDI
	BitBlt(hdc, MapRect.left, MapRect.top, MapRect.right - MapRect.left, MapRect.bottom - MapRect.top, surface.emu->DC, DestRect.left, DestRect.top, SRCCOPY);

	// Release DC
	if (UsingForgroundWindow)
	{
		::ReleaseDC(hWnd, hdc);
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::GetPresentWindowRect(LPRECT pRect, RECT& DestRect)
{
	if (!PrimaryDisplayTexture)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Primary display texture missing!");
		return DDERR_GENERIC;
	}

	// Get hWnd
	HWND hWnd = ddrawParent->GetHwnd();
	if (!hWnd)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Cannot get window handle!");
		return DDERR_GENERIC;
	}

	// Check for iconic window
	if (IsIconic(hWnd))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Window is iconic!");
		return DDERR_GENERIC;
	}

	// Clip rect
	RECT Rect = pRect ? *pRect : RECT{ 0, 0, (LONG)surfaceDesc2.dwWidth, (LONG)surfaceDesc2.dwHeight };
	RECT ClientRect = {};
	if (GetClientRect(hWnd, &ClientRect) && MapWindowPoints(hWnd, HWND_DESKTOP, (LPPOINT)&ClientRect, 2))
	{
		Rect.left = max(Rect.left, ClientRect.left);
		Rect.top = max(Rect.top, ClientRect.top);
		Rect.right = min(Rect.right, ClientRect.right);
		Rect.bottom = min(Rect.bottom, ClientRect.bottom);
	}

	// Get map points
	RECT MapClient = Rect;
	MapWindowPoints(HWND_DESKTOP, hWnd, (LPPOINT)&MapClient, 2);

	// Validate MapClient rect
	if (!ClipRectToBounds(&MapClient, surfaceDesc2.dwWidth, surfaceDesc2.dwHeight))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid MapClient rect: " << Rect);
		return DDERR_GENERIC;
	}

	// Validate rect
	if (!ClipRectToBounds(&Rect, surfaceDesc2.dwWidth, surfaceDesc2.dwHeight))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid rect: " << Rect);
		return DDERR_GENERIC;
	}

	// Get source surface
	IDirect3DSurface9* pSourceSurfaceD9 = Get3DMipMapSurface(0);
	if (!pSourceSurfaceD9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to get source surface!");
		return DDERR_GENERIC;
	}

	// Get destination surface
	ComPtr<IDirect3DSurface9> pDestSurfaceD9;
	if (FAILED(PrimaryDisplayTexture->GetSurfaceLevel(0, pDestSurfaceD9.GetAddressOf())))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to get destination surface!");
		return DDERR_GENERIC;
	}

	// Copy surface
	HRESULT hr = DDERR_GENERIC;
	if (IsD9UsingVideoMemory())
	{
		hr = (*d3d9Device)->StretchRect(pSourceSurfaceD9, &Rect, pDestSurfaceD9.Get(), &MapClient, D3DTEXF_NONE);
	}
	else
	{
		hr = (*d3d9Device)->UpdateSurface(pSourceSurfaceD9, &Rect, pDestSurfaceD9.Get(), (LPPOINT)&MapClient);
	}

	if (FAILED(hr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to copy surface: " << Rect << " -> " << MapClient);
		return DDERR_GENERIC;
	}

	DestRect = MapClient;

	return DD_OK;
}

void m_IDirectDrawSurfaceX::RemoveClipper(m_IDirectDrawClipper* ClipperToRemove)
{
	if (ClipperToRemove == attachedClipper)
	{
		SetClipper(nullptr);
	}
}

void m_IDirectDrawSurfaceX::RemovePalette(m_IDirectDrawPalette* PaletteToRemove)
{
	if (PaletteToRemove == attachedPalette)
	{
		SetPalette(nullptr);
	}
}

void m_IDirectDrawSurfaceX::UpdatePaletteData()
{
	// Check surface format
	if (!IsPalette())
	{
		return;
	}

	DWORD NewPaletteUSN = 0;
	const PALETTEENTRY* NewPaletteEntry = nullptr;
	const RGBQUAD* NewRGBPalette = nullptr;

	// Get palette data
	if (attachedPalette)
	{
		NewPaletteUSN = attachedPalette->GetPaletteUSN();
		NewPaletteEntry = attachedPalette->GetPaletteEntries();
		NewRGBPalette = attachedPalette->GetRGBPalette();
	}
	// Get palette from primary surface if this is not primary
	else if (!IsPrimarySurface())
	{
		m_IDirectDrawSurfaceX* lpPrimarySurface = ddrawParent->GetPrimarySurface();
		if (lpPrimarySurface)
		{
			m_IDirectDrawPalette* lpPalette = lpPrimarySurface->GetAttachedPalette();
			if (lpPalette)
			{
				NewPaletteUSN = lpPalette->GetPaletteUSN();
				NewPaletteEntry = lpPalette->GetPaletteEntries();
				NewRGBPalette = lpPalette->GetRGBPalette();
			}
		}
	}

	bool IsPrimaryPaletteUpdated = (primary.PaletteTexture && NewPaletteEntry && primary.LastPaletteUSN != NewPaletteUSN);
	bool IsEmulatedPaletteUpdated = (IsUsingEmulation() && NewRGBPalette && surface.emu->LastPaletteUSN != NewPaletteUSN);
	bool IsPaletteDataUpdated = (NewPaletteEntry && surface.LastPaletteUSN != NewPaletteUSN);

	ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection(), IsPrimaryPaletteUpdated || IsEmulatedPaletteUpdated || IsPaletteDataUpdated);

	// Add palette data to texture
	if (IsPrimaryPaletteUpdated)
	{
		// Get palette display context surface
		ComPtr<IDirect3DSurface9> paletteSurface;
		if (SUCCEEDED(primary.PaletteTexture->GetSurfaceLevel(0, paletteSurface.GetAddressOf())))
		{
			// Use LoadSurfaceFromMemory to copy to the surface
			RECT Rect = { 0, 0, MaxPaletteSize, 1 };
			if (FAILED(LoadSurfaceFromMemory(paletteSurface.Get(), Rect, NewRGBPalette, D3DFMT_X8R8G8B8, MaxPaletteSize * sizeof(D3DCOLOR))))
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: could not full palette textur!");
			}
			primary.LastPaletteUSN = NewPaletteUSN;
		}
	}

	// Set color palette for emulation device context
	if (IsEmulatedPaletteUpdated)
	{
		SetDIBColorTable(surface.emu->DC, 0, MaxPaletteSize, NewRGBPalette);
		SetDIBColorTable(surface.emu->GameDC, 0, MaxPaletteSize, NewRGBPalette);
		surface.emu->LastPaletteUSN = NewPaletteUSN;
	}

	// Set new palette data
	if (IsPaletteDataUpdated)
	{
		surface.IsPaletteDirty = true;
		surface.LastPaletteUSN = NewPaletteUSN;
		surface.PaletteEntryArray = NewPaletteEntry;
	}
}

m_IDirectDrawSurfaceX* m_IDirectDrawSurfaceX::GetAttachedDepthStencil()
{
	for (auto& it : AttachedSurfaceMap)
	{
		if (it.second.pSurface->IsDepthStencil())
		{
			return it.second.pSurface;
		}
	}
	return nullptr;
}

HRESULT m_IDirectDrawSurfaceX::GetMipMapLevelAddr(LPDIRECTDRAWSURFACE7 FAR* lplpDDAttachedSurface, MIPMAP& MipMapSurface, DWORD MipMapLevel, DWORD DirectXVersion)
{
	switch (DirectXVersion)
	{
	case 1:
		if (!MipMapSurface.Addr)
		{
			MipMapSurface.Addr = new m_IDirectDrawSurface(this, MipMapLevel);
		}
		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)MipMapSurface.Addr;
		break;
	case 2:
		if (!MipMapSurface.Addr2)
		{
			MipMapSurface.Addr2 = new m_IDirectDrawSurface2(this, MipMapLevel);
		}
		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)MipMapSurface.Addr2;
		break;
	case 3:
		if (!MipMapSurface.Addr3)
		{
			MipMapSurface.Addr3 = new m_IDirectDrawSurface3(this, MipMapLevel);
		}
		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)MipMapSurface.Addr3;
		break;
	case 4:
		if (!MipMapSurface.Addr4)
		{
			MipMapSurface.Addr4 = new m_IDirectDrawSurface4(this, MipMapLevel);
		}
		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)MipMapSurface.Addr4;
		break;
	case 7:
		if (!MipMapSurface.Addr7)
		{
			MipMapSurface.Addr7 = new m_IDirectDrawSurface7(this, MipMapLevel);
		}
		*lplpDDAttachedSurface = (LPDIRECTDRAWSURFACE7)MipMapSurface.Addr7;
		break;
	default:
		LOG_LIMIT(100, __FUNCTION__ << " Error: incorrect DirectX version: " << DirectXVersion);
		return DDERR_NOTFOUND;
	}

	return DD_OK;
}

HRESULT m_IDirectDrawSurfaceX::GetMipMapSubLevel(LPDIRECTDRAWSURFACE7 FAR* lplpDDAttachedSurface, DWORD MipMapLevel, DWORD DirectXVersion)
{
	// Check for device interface to ensure correct max MipMap level
	CheckInterface(__FUNCTION__, true, true, false);

	if (MaxMipMapLevel > MipMapLevel)
	{
		while (MipMaps.size() < MipMapLevel + 1)
		{
			MIPMAP MipMap;
			MipMaps.push_back(MipMap);
		}

		return GetMipMapLevelAddr(lplpDDAttachedSurface, MipMaps[MipMapLevel], MipMapLevel + 1, DirectXVersion);
	}
	return DDERR_NOTFOUND;
}

HRESULT m_IDirectDrawSurfaceX::CheckBackBufferForFlip(m_IDirectDrawSurfaceX* lpTargetSurface)
{
	// Check if target surface exists
	if (!lpTargetSurface || lpTargetSurface == this || !DoesFlipBackBufferExist(lpTargetSurface))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: invalid surface!");
		return DDERR_INVALIDPARAMS;
	}

	// Make sure that surface description on target is updated
	lpTargetSurface->UpdateSurfaceDesc();

	// Check for device interface
	HRESULT c_hr = lpTargetSurface->CheckInterface(__FUNCTION__, true, true, true);
	if (FAILED(c_hr))
	{
		return c_hr;
	}

	// Check if surface format and size matches
	if (surface.Format != lpTargetSurface->surface.Format ||
		surfaceDesc2.dwWidth != lpTargetSurface->surfaceDesc2.dwWidth ||
		surfaceDesc2.dwHeight != lpTargetSurface->surfaceDesc2.dwHeight)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: backbuffer surface does not match: " <<
			surface.Format << " -> " << lpTargetSurface->surface.Format << " " <<
			surfaceDesc2.dwWidth << "x" << surfaceDesc2.dwHeight << " -> " <<
			lpTargetSurface->surfaceDesc2.dwWidth << "x" << lpTargetSurface->surfaceDesc2.dwHeight);
		return DDERR_INVALIDPARAMS;
	}

	return DD_OK;
}

bool m_IDirectDrawSurfaceX::GetColorKeyForPrimaryShader(float(&lowColorKey)[4], float(&highColorKey)[4])
{
	// Primary 2D surface background color
	if (IsPrimarySurface())
	{
		if (!primary.ShaderColorKey.IsSet)
		{
			GetColorKeyArray(primary.ShaderColorKey.lowColorKey, primary.ShaderColorKey.highColorKey, Config.DdrawFlipFillColor, Config.DdrawFlipFillColor, surfaceDesc2.ddpfPixelFormat);
			primary.ShaderColorKey.IsSet = true;
		}
		lowColorKey[0] = primary.ShaderColorKey.lowColorKey[0];
		lowColorKey[1] = primary.ShaderColorKey.lowColorKey[1];
		lowColorKey[2] = primary.ShaderColorKey.lowColorKey[2];
		lowColorKey[3] = primary.ShaderColorKey.lowColorKey[3];
		highColorKey[0] = primary.ShaderColorKey.highColorKey[0];
		highColorKey[1] = primary.ShaderColorKey.highColorKey[1];
		highColorKey[2] = primary.ShaderColorKey.highColorKey[2];
		highColorKey[3] = primary.ShaderColorKey.highColorKey[3];
		return true;
	}
	return false;
}

bool m_IDirectDrawSurfaceX::GetColorKeyForShader(float(&lowColorKey)[4], float(&highColorKey)[4])
{
	// Surface low and high color space
	if (!ShaderColorKey.IsSet)
	{
		if (surfaceDesc2.dwFlags & DDSD_CKSRCBLT)
		{
			GetColorKeyArray(ShaderColorKey.lowColorKey, ShaderColorKey.highColorKey,
				surfaceDesc2.ddckCKSrcBlt.dwColorSpaceLowValue, surfaceDesc2.ddckCKSrcBlt.dwColorSpaceHighValue, surfaceDesc2.ddpfPixelFormat);
			ShaderColorKey.IsSet = true;
		}
		else
		{
			return false;
		}
	}
	lowColorKey[0] = ShaderColorKey.lowColorKey[0];
	lowColorKey[1] = ShaderColorKey.lowColorKey[1];
	lowColorKey[2] = ShaderColorKey.lowColorKey[2];
	lowColorKey[3] = ShaderColorKey.lowColorKey[3];
	highColorKey[0] = ShaderColorKey.highColorKey[0];
	highColorKey[1] = ShaderColorKey.highColorKey[1];
	highColorKey[2] = ShaderColorKey.highColorKey[2];
	highColorKey[3] = ShaderColorKey.highColorKey[3];
	return true;
}

void m_IDirectDrawSurfaceX::FixTextureFlags(LPDDSURFACEDESC2 lpDDSurfaceDesc2)
{
	if (lpDDSurfaceDesc2)
	{
		if (lpDDSurfaceDesc2->dwFlags & DDSD_PITCH)
		{
			lpDDSurfaceDesc2->dwFlags |= DDSD_LINEARSIZE;
		}
		lpDDSurfaceDesc2->dwFlags &= ~(DDSD_PITCH | DDSD_LPSURFACE);
	}
}

HRESULT m_IDirectDrawSurfaceX::LockD3d9Surface(D3DLOCKED_RECT* pLockedRect, RECT* pRect, DWORD Flags, DWORD MipMapLevel)
{
	if (surface.UsingSurfaceMemory)
	{
		pLockedRect->Pitch = surfaceDesc2.dwWidth * surface.BitCount / 8;
		pLockedRect->pBits = (pRect) ? (void*)((DWORD)surfaceDesc2.lpSurface + ((pRect->top * pLockedRect->Pitch) + (pRect->left * (surface.BitCount / 8)))) : surfaceDesc2.lpSurface;
		return DD_OK;
	}
	// Lock shadow surface
	else if (IsUsingShadowSurface())
	{
		HRESULT hr = surface.Shadow->LockRect(pLockedRect, pRect, Flags);
		if (FAILED(hr) && (Flags & D3DLOCK_NOSYSLOCK))
		{
			hr = surface.Shadow->LockRect(pLockedRect, pRect, Flags & ~D3DLOCK_NOSYSLOCK);
		}
		return hr;
	}
	// Lock 3D surface
	else if (surface.Surface)
	{
		HRESULT hr = surface.Surface->LockRect(pLockedRect, pRect, Flags);
		if (FAILED(hr) && (Flags & D3DLOCK_NOSYSLOCK))
		{
			hr = surface.Surface->LockRect(pLockedRect, pRect, Flags & ~D3DLOCK_NOSYSLOCK);
		}
		return hr;
	}
	// Lock surface texture
	else if (surface.Texture)
	{
		HRESULT hr = surface.Texture->LockRect(GetD3d9MipMapLevel(MipMapLevel), pLockedRect, pRect, Flags);
		if (FAILED(hr) && (Flags & D3DLOCK_NOSYSLOCK))
		{
			hr = surface.Texture->LockRect(GetD3d9MipMapLevel(MipMapLevel), pLockedRect, pRect, Flags & ~D3DLOCK_NOSYSLOCK);
		}
		return hr;
	}

	return DDERR_GENERIC;
}

HRESULT m_IDirectDrawSurfaceX::UnLockD3d9Surface(DWORD MipMapLevel)
{
	if (surface.UsingSurfaceMemory)
	{
		return DD_OK;
	}
	// Unlock shadow surface
	else if (IsUsingShadowSurface())
	{
		return surface.Shadow->UnlockRect();
	}
	// Unlock 3D surface
	else if (surface.Surface)
	{
		return surface.Surface->UnlockRect();
	}
	// Unlock surface texture
	else if (surface.Texture)
	{
		return surface.Texture->UnlockRect(GetD3d9MipMapLevel(MipMapLevel));
	}

	return DDERR_GENERIC;
}

HRESULT m_IDirectDrawSurfaceX::PresentOverlay(LPRECT lpSrcRect)
{
	if (SurfaceOverlay.OverlayEnabled)
	{
		RECT SrcRect = {};
		if (!lpSrcRect || SurfaceOverlay.isSrcRectNull || GetOverlappingRect(*lpSrcRect, SurfaceOverlay.SrcRect, SrcRect))
		{
			LPRECT lpNewSrcRect = SurfaceOverlay.isSrcRectNull ? nullptr : &SurfaceOverlay.SrcRect;
			LPRECT lpNewDestRect = SurfaceOverlay.isDestRectNull ? nullptr : &SurfaceOverlay.DestRect;

			DWORD DDBltFxFlags = SurfaceOverlay.DDBltFxFlags;
			DDBLTFX DDBltFx = SurfaceOverlay.DDBltFx;

			// Handle color keying
			if (!(DDBltFxFlags & (DDBLT_KEYDESTOVERRIDE | DDBLT_KEYSRCOVERRIDE)))
			{
				if ((SurfaceOverlay.DDOverlayFxFlags & DDOVER_KEYDEST) && (SurfaceOverlay.lpDDDestSurfaceX->surfaceDesc2.dwFlags & DDSD_CKDESTOVERLAY))
				{
					DDBltFxFlags |= (DDBLT_DDFX | DDBLT_KEYDESTOVERRIDE);
					DDBltFx.ddckDestColorkey = SurfaceOverlay.lpDDDestSurfaceX->surfaceDesc2.ddckCKDestOverlay;
				}
				else if ((SurfaceOverlay.DDOverlayFxFlags & DDOVER_KEYSRC) && (surfaceDesc2.dwFlags & DDSD_CKSRCOVERLAY))
				{
					DDBltFxFlags |= (DDBLT_DDFX | DDBLT_KEYSRCOVERRIDE);
					DDBltFx.ddckSrcColorkey = surfaceDesc2.ddckCKSrcOverlay;
				}
			}

			SurfaceOverlay.lpDDDestSurfaceX->Blt(lpNewDestRect, (LPDIRECTDRAWSURFACE7)GetWrapperInterfaceX(0), lpNewSrcRect, DDBltFxFlags, &DDBltFx, 0);
		}
	}
	return DD_OK;
}

// ******************************
// External static functions
// ******************************

void m_IDirectDrawSurfaceX::StartSharedEmulatedMemory()
{
	ShareEmulatedMemory = true;
}

void m_IDirectDrawSurfaceX::DeleteEmulatedMemory(EMUSURFACE **ppEmuSurface)
{
	if (!ppEmuSurface || !*ppEmuSurface)
	{
		return;
	}

	LOG_LIMIT(100, __FUNCTION__ << " Deleting emulated surface (" << *ppEmuSurface << ")");

	ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

	// Release device context memory
	if ((*ppEmuSurface)->DC)
	{
		SelectObject((*ppEmuSurface)->DC, (*ppEmuSurface)->OldDCObject);
		DeleteDC((*ppEmuSurface)->DC);
	}
	if ((*ppEmuSurface)->GameDC)
	{
		if ((*ppEmuSurface)->OldGameDCObject)
		{
			SelectObject((*ppEmuSurface)->GameDC, (*ppEmuSurface)->OldGameDCObject);
		}
		DeleteDC((*ppEmuSurface)->GameDC);
	}
	if ((*ppEmuSurface)->bitmap)
	{
		DeleteObject((*ppEmuSurface)->bitmap);
		(*ppEmuSurface)->pBits = nullptr;
	}
	if ((*ppEmuSurface)->pBits)
	{
		HeapFree(GetProcessHeap(), NULL, (*ppEmuSurface)->pBits);
		(*ppEmuSurface)->pBits = nullptr;
	}
	delete (*ppEmuSurface);
	*ppEmuSurface = nullptr;
}

void m_IDirectDrawSurfaceX::CleanupSharedEmulatedMemory()
{
	// Disable shared memory
	ShareEmulatedMemory = false;
	
	LOG_LIMIT(100, __FUNCTION__ << " Deleting " << memorySurfaces.size() << " emulated surface" << ((memorySurfaces.size() != 1) ? "s" : "") << "!");

	ScopedCriticalSection ThreadLockPE(DdrawWrapper::GetPECriticalSection());

	// Clean up unused emulated surfaces
	for (EMUSURFACE* pEmuSurface: memorySurfaces)
	{
		DeleteEmulatedMemory(&pEmuSurface);
	}
	memorySurfaces.clear();
}

void m_IDirectDrawSurfaceX::SizeDummySurface(size_t size)
{
	dummySurface.resize(size);
}

void m_IDirectDrawSurfaceX::CleanupDummySurface()
{
	dummySurface.clear();
}

void m_IDirectDrawSurfaceX::LogAtlasTrackingAndReset()
{
	if (!Config.DdrawLogTextureAtlas)
	{
		return;
	}

	// Increment frame counter
	lockTrackingFrame++;

	// Only log every 60 frames to reduce spam
	if (lockTrackingFrame % 60 != 0)
	{
		lockTrackingMap.clear();
		return;
	}

	// Surface-pool dump (Phase A): live/peak texture-surface count, create/release
	// churn over the last 60 frames, and the DKII.exe call sites doing the creating.
	Logging::Log() << "=== SURFACE POOL FRAME " << lockTrackingFrame
		<< " === live=" << poolStats.liveTextureSurfaces
		<< " peak=" << poolStats.peakTextureSurfaces
		<< " created/60f=" << poolStats.createdThisWindow
		<< " released/60f=" << poolStats.releasedThisWindow
		<< " bltToTex/60f=" << poolStats.bltToTextureThisWindow;
	for (const auto& c : createCallerHist)
	{
		Logging::Log() << "  CREATE CALLER " << Logging::hex(c.first) << " count=" << c.second;
	}
	// Phase A.6: dump top blt destinations so we know which surfaces are
	// receiving the baseline blts (candidate torch / animated surfaces).
	if (!bltTargetsThisWindow.empty())
	{
		std::vector<std::pair<void*, BltDestInfo>> sorted(bltTargetsThisWindow.begin(), bltTargetsThisWindow.end());
		std::sort(sorted.begin(), sorted.end(),
			[](const std::pair<void*, BltDestInfo>& a, const std::pair<void*, BltDestInfo>& b) {
				return a.second.count > b.second.count;
			});
		int emitted = 0;
		for (const auto& p : sorted)
		{
			if (emitted++ >= 10) break;
			Logging::Log() << "  BLT_TO " << p.first
				<< " size=" << p.second.width << "x" << p.second.height
				<< " count=" << p.second.count
				<< " lastSrc=" << p.second.lastSrc;
		}
	}
	bltTargetsThisWindow.clear();
	poolStats.createdThisWindow = 0;
	poolStats.releasedThisWindow = 0;
	poolStats.bltToTextureThisWindow = 0;
	createCallerHist.clear();

	// Phase A.7: capture progress visible in the same per-window stream
	if (Config.DdrawContentCapture && capState.initialized)
	{
		Logging::Log() << "  CAPTURE STATS bltSeenTotal=" << capState.bltSeenTotal
			<< " bindSeenTotal=" << capState.bindSeenTotal
			<< " uniqueSavedTotal=" << capState.savedTotal
			<< " inMemoryHashes=" << capState.seenHashes.size()
			<< " surfacesBound=" << capState.firstBindSeen.size();
	}
	if (Config.DdrawCollapseAnimationPools)
	{
		Logging::Log() << "  PATHB STATS activePools=" << poolsActiveTotal
			<< " redirectMembers=" << memberToCanonical.size()
			<< " redirectsThisSession=" << poolRedirectsTotal;
	}

	if (lockTrackingMap.empty())
	{
		Logging::Log() << "=== LOCK TRACKING FRAME " << lockTrackingFrame << " === (no locks)";
		return;
	}

	Logging::Log() << "=== LOCK TRACKING FRAME " << lockTrackingFrame << " ===";

	// Find surfaces with high lock counts (likely atlases being written to)
	for (const auto& pair : lockTrackingMap)
	{
		const SurfaceLockStats& stats = pair.second;

		// Log surfaces with write locks (potential atlas targets)
		Logging::Log() << "SURFACE: " << pair.first
			<< " size=" << stats.width << "x" << stats.height
			<< " lockCount=" << stats.lockCount
			<< " writeLocks=" << stats.writeLockCount;
	}

	Logging::Log() << "=== END LOCK TRACKING ===";

	// Reset tracking for next frame
	lockTrackingMap.clear();
}

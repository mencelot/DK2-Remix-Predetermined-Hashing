#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>
#include <string>
#include <ostream>
#include "ReadParse.h"

#define NOT_EXIST 0xFFFF
#define MAX_ENV_VAR 0x7FFF

struct DHEX {
	DWORD value = 0;

	explicit operator bool() const { return value != 0; }
	operator DWORD() const { return value; }
	DHEX& operator=(DWORD v) { value = v; return *this; }
};

inline std::ostream& operator<<(std::ostream& os, const DHEX& dhex) {
	return os << dhex.value;
}

#define VISIT_CONFIG_SETTINGS(visit) \
	visit(AnisotropicFiltering) \
	visit(AntiAliasing) \
	visit(AudioClipDetection) \
	visit(AudioFadeOutDelayMS) \
	visit(CustomDisplayWidth) \
	visit(CustomDisplayHeight) \
	visit(Dd7to9) \
	visit(D3d8to9) \
	visit(D3d9to9Ex) \
	visit(D3d9on12) \
	visit(Dinputto8) \
	visit(DDrawCompat) \
	visit(DDrawCompat20) \
	visit(DDrawCompat21) \
	visit(DDrawCompat32) \
	visit(DDrawCompatDisableGDIHook) \
	visit(DDrawCompatNoProcAffinity) \
	visit(DdrawAlternatePixelCenter) \
	visit(DdrawAutoFrameSkip) \
	visit(DdrawClampVertexZDepth) \
	visit(DdrawClippedWidth) \
	visit(DdrawClippedHeight) \
	visit(DdrawCustomWidth) \
	visit(DdrawCustomHeight) \
	visit(DdrawEnableByteAlignment) \
	visit(DdrawDisableDirect3DCaps) \
	visit(DdrawEmulateLock) \
	visit(DdrawFillSurfaceColor) \
	visit(DdrawForceMipMapAutoGen) \
	visit(DdrawFlipFillColor) \
	visit(DdrawKeepAllInterfaceCache) \
	visit(DdrawRemoveScanlines) \
	visit(DdrawRemoveInterlacing) \
	visit(DdrawFixByteAlignment) \
	visit(DdrawIntroVideoFix) \
	visit(DdrawEmulateSurface) \
	visit(DdrawReadFromGDI) \
	visit(DdrawWriteToGDI) \
	visit(DdrawIntegerScalingClamp) \
	visit(DdrawLimitDisplayModeCount) \
	visit(DdrawLimitTextureFormats) \
	visit(DdrawMaintainAspectRatio) \
	visit(DdrawNoDrawBufferSysLock) \
	visit(DdrawNoMultiThreaded) \
	visit(DdrawOverrideBitMode) \
	visit(DdrawOverrideWidth) \
	visit(DdrawOverrideHeight) \
	visit(DdrawOverrideStencilFormat) \
	visit(DdrawResolutionHack) \
	visit(DdrawConvertHomogeneousW) \
	visit(DdrawConvertHomogeneousToWorld) \
	visit(DdrawConvertHomogeneousToWorldUseGameCamera) \
	visit(DdrawConvertHomogeneousToWorldFOV) \
	visit(DdrawConvertHomogeneousToWorldNearPlane) \
	visit(DdrawConvertHomogeneousToWorldFarPlane) \
	visit(DdrawConvertHomogeneousToWorldDepthOffset) \
	visit(DdrawLogTextureAtlas) \
	visit(DdrawContentCapture) \
	visit(DdrawStage3BoundDump) \
	visit(DdrawRecipeLog) \
	visit(DdrawOwnerLog) \
	visit(DdrawNameKey) \
	visit(DdrawNameKeyCrop) \
	visit(DdrawDynamicUiEmissive) \
	visit(DdrawWaterCollapse) \
	visit(DdrawWaterFlatten) \
	visit(DdrawWaterCropLarge) \
	visit(DdrawCanonicalRebind) \
	visit(DdrawReplaceHeap) \
	visit(DdrawMenuResWidth) \
	visit(DdrawMenuResHeight) \
	visit(DdrawOrphanOverlayLift) \
	visit(DdrawMenuBlitOverlay) \
	visit(DdrawCollapseAnimationPools) \
	visit(DdrawAtlasDecompose) \
	visit(DdrawUniversalDecompose) \
	visit(DdrawGeomProbe) \
	visit(DdrawDrawCallerLog) \
	visit(DdrawUseDirect3D9Caps) \
	visit(DdrawUseShadowSurface) \
	visit(DdrawUseNativeResolution) \
	visit(DdrawVertexLockDiscard) \
	visit(DdrawEnableMouseHook) \
	visit(DdrawDisableLighting) \
	visit(DdrawHookSystem32) \
	visit(D3d8HookSystem32) \
	visit(D3d9HookSystem32) \
	visit(DinputHookSystem32) \
	visit(Dinput8HookSystem32) \
	visit(DsoundHookSystem32) \
	visit(SetSwapEffectShim) \
	visit(DeviceLookupCacheTime) \
	visit(DisableGameUX) \
	visit(DisableGDIGammaRamp) \
	visit(DisableHighDPIScaling) \
	visit(DisableLogging) \
	visit(DirectShowEmulation) \
	visit(CacheClipPlane) \
	visit(EnvironmentCubeMapFix) \
	visit(EnableDdrawWrapper) \
	visit(EnableD3d9Wrapper) \
	visit(EnableDinput8Wrapper) \
	visit(EnableDsoundWrapper) \
	visit(EnableImgui) \
	visit(EnableMultisamplingATOC) \
	visit(EnableOpenDialogHook) \
	visit(EnableVSync) \
	visit(EnableWindowMode) \
	visit(ExcludeProcess) \
	visit(ForceExclusiveFullscreen) \
	visit(ForceKeyboardLayout) \
	visit(ForceMixedVertexProcessing) \
	visit(ForceSystemMemVertexCache) \
	visit(ForceSingleBeginEndScene) \
	visit(FilterNonActiveInput) \
	visit(FixHighFrequencyMouse) \
	visit(FixPerfCounterUptime) \
	visit(FixSpeakerConfigType) \
	visit(FlipEx) \
	visit(ForceExclusiveMode) \
	visit(ForceHardwareMixing) \
	visit(ForceHQ3DSoftMixing) \
	visit(ForceNonStaticBuffers) \
	visit(ForcePrimaryBufferFormat) \
	visit(ForceSoftwareMixing) \
	visit(ForceTermination) \
	visit(ForceVoiceManagement) \
	visit(ForceVsyncMode) \
	visit(ForceWindowResize) \
	visit(FullScreen) \
	visit(FullscreenWindowMode) \
	visit(GraphicsHybridAdapter) \
	visit(HandleExceptions) \
	visit(IgnoreWindowName) \
	visit(IncludeProcess) \
	visit(InitialWindowPositionLeft) \
	visit(InitialWindowPositionTop) \
	visit(isAppCompatDataSet) \
	visit(LimitDisplayModeCount) \
	visit(LimitPerFrameFPS) \
	visit(LoadCustomDllPath) \
	visit(LoadFromScriptsOnly) \
	visit(LoadPlugins) \
	visit(LockColorkey) \
	visit(LoopSleepTime) \
	visit(MouseMovementFactor) \
	visit(MouseMovementPadding) \
	visit(Num2DBuffers) \
	visit(Num3DBuffers) \
	visit(OverrideRefreshRate) \
	visit(OverrideStencilFormat) \
	visit(PrimaryBufferBits) \
	visit(PrimaryBufferChannels) \
	visit(PrimaryBufferSamples) \
	visit(RealDllPath) \
	visit(ResetMemoryAfter) \
	visit(ResetScreenRes) \
	visit(LimitStateBlocks) \
	visit(RunProcess) \
	visit(SendAltEnter) \
	visit(SetFullScreenLayer) \
	visit(SetInitialWindowPosition) \
	visit(SetNamedLayer) \
	visit(SetPOW2Caps) \
	visit(ShowFPSCounter) \
	visit(SingleProcAffinity) \
	visit(StoppedDriverWorkaround) \
	visit(UseShadowBackbuffer) \
	visit(WaitForProcess) \
	visit(WaitForWindowChanges) \
	visit(WindowSleepTime) \
	visit(WindowModeBorder) \
	visit(WinVersionLie) \
	visit(WinVersionLieSP) \
	visit(WindowModeGammaShader) \
	visit(WrapperMode)

#define VISIT_APPCOMPATDATA_SETTINGS(visit) \
	visit(LockEmulation) \
	visit(BltEmulation) \
	visit(ForceLockNoWindow) \
	visit(ForceBltNoWindow) \
	visit(LockColorkey) \
	visit(FullscreenWithDWM) \
	visit(DisableLockEmulation) \
	visit(EnableOverlays) \
	visit(DisableSurfaceLocks) \
	visit(RedirectPrimarySurfBlts) \
	visit(StripBorderStyle) \
	visit(DisableMaxWindowedMode)

typedef unsigned char byte;

struct MEMORYINFO						// Used for hot patching memory
{
	void* AddressPointer = nullptr;		// Hot patch address
	std::string PatternString;			// Hot patch pattern
	std::vector<byte> Bytes;			// Hot patch bytes
};

struct DLLTYPE
{
	const DWORD dxwrapper = 0;
	const DWORD ddraw = 1;
	const DWORD d3d8 = 2;
	const DWORD d3d9 = 3;
	const DWORD dsound = 4;
	const DWORD dinput = 5;
	const DWORD dinput8 = 6;
	const DWORD winmm = 7;
};
static const DLLTYPE dtype;

// Designated Initializer does not work in VS 2015 so must pay attention to the order
static constexpr const char* dtypename[] = {
	"dxwrapper.dll",// 0
	"ddraw.dll",	// 1
	"d3d8.dll",		// 2
	"d3d9.dll",		// 3
	"dsound.dll",	// 4
	"dinput.dll",	// 5
	"dinput8.dll",	// 6
	"winmm.dll",	// 7
};
static constexpr int dtypeArraySize = (sizeof(dtypename) / sizeof(*dtypename));

struct APPCOMPATDATATYPE
{
	const DWORD Empty = 0;
	const DWORD LockEmulation = 1;
	const DWORD BltEmulation = 2;
	const DWORD ForceLockNoWindow = 3;
	const DWORD ForceBltNoWindow = 4;
	const DWORD LockColorkey = 5;
	const DWORD FullscreenWithDWM = 6;
	const DWORD DisableLockEmulation = 7;
	const DWORD EnableOverlays = 8;
	const DWORD DisableSurfaceLocks = 9;
	const DWORD RedirectPrimarySurfBlts = 10;
	const DWORD StripBorderStyle = 11;
	const DWORD DisableMaxWindowedMode = 12;
};
static const APPCOMPATDATATYPE AppCompatDataType;

struct CONFIG
{
	void Init();								// Initialize the config setting
	void SetConfig();							// Set additional settings
	bool IsSet(DWORD Value);					// Check if a value is set
	bool Exiting = false;						// Dxwrapper is being unloaded
	bool Dd7to9 = false;						// Converts DirectDraw/Direct3D (ddraw.dll) to Direct3D9 (d3d9.dll)
	bool D3d8to9 = false;						// Converts Direct3D8 (d3d8.dll) to Direct3D9 (d3d9.dll) https://github.com/crosire/d3d8to9
	bool D3d9to9Ex = false;						// Converts Direct3D9 to Direct3D9Ex
	bool D3d9on12 = false;						// Converts Direct3D9 to use CreateDirect3D9On12
	bool Dinputto8 = false;						// Converts DirectInput (dinput.dll) to DirectInput8 (dinput8.dll)
	bool DDrawCompat = false;					// Enables the default DDrawCompat functions https://github.com/narzoul/DDrawCompat/
	bool DDrawCompat20 = false;					// Enables DDrawCompat v0.2.0b
	bool DDrawCompat21 = false;					// Enables DDrawCompat v0.2.1
	bool DDrawCompat32 = false;					// Enables DDrawCompat v0.3.2
	bool DDrawCompatDisableGDIHook = false;		// Disables DDrawCompat GDI hooks
	bool DDrawCompatNoProcAffinity = false;		// Disables DDrawCompat single processor affinity
	bool DdrawAlternatePixelCenter = false;		// Enables alternate pixel center -0.5f vs 0.0
	DWORD DdrawClampVertexZDepth = 0;			// 1) Clamps z depth to a max of 1.0f, 2) Clamps the z depth between 0.0f and 1.0f and recomputes w/rhw
	bool DdrawAutoFrameSkip = false;			// Automatically skips frames to reduce input lag
	DWORD DdrawFixByteAlignment = false;		// Fixes lock with surfaces that have unaligned byte sizes, 1) just byte align, 2) byte align + D3DTEXF_NONE, 3) byte align + D3DTEXF_LINEAR
	bool DdrawEnableByteAlignment = false;		// Disables 32bit / 64bit byte alignment
	bool DdrawIntroVideoFix = false;			// Enables some fixes that may help with showing intro videos
	DWORD DdrawResolutionHack = 0;				// Removes the artificial resolution limit from Direct3D7 and below https://github.com/UCyborg/LegacyD3DResolutionHack
	bool DdrawRemoveScanlines = false;			// Experimental feature to removing interlaced black lines in a single frame
	bool DdrawRemoveInterlacing = false;		// Experimental feature to removing interlacing between frames
	bool DdrawFillSurfaceColor = false;			// After creating surface fill with random color for testing black screen or objects
	bool DdrawKeepAllInterfaceCache = false;	// Preserve the interface cache all ddraw interfaces, which may casue higher memory usage
	bool DdrawEmulateSurface = false;			// Emulates the ddraw surface using device context for Dd7to9
	bool DdrawEmulateLock = false;				// Emulates the lock to prevent crashes when an application tries to read data outside Lock/Unlock pair
	bool DdrawReadFromGDI = false;				// Read from GDI bfore passing surface to program
	bool DdrawWriteToGDI = false;				// Blt surface directly to GDI rather than Direct3D9
	bool DdrawIntegerScalingClamp = false;		// Scales the screen by an integer value to help preserve video quality
	bool DdrawMaintainAspectRatio = false;		// Keeps the current DirectDraw aspect ratio when overriding the game's resolution
	bool DdrawConvertHomogeneousW = false;		// Convert primites using D3DFVF_XYZRHW to D3DFVF_XYZW.
	bool DdrawConvertHomogeneousToWorld = false;				// Convert primitives back into a world space. Needed for RTX.
	bool DdrawConvertHomogeneousToWorldUseGameCamera = false;	// Use the game's view matrix instead of replacing it with our own.
	float DdrawConvertHomogeneousToWorldFOV = 0.0f;				// The field of view of the camera used to reconstruct the original 3D world.
	float DdrawConvertHomogeneousToWorldNearPlane = 0.0f;		// The near plane of the camera used to reconstruct the original 3D world.
	float DdrawConvertHomogeneousToWorldFarPlane = 0.0f;		// The far plane of the camera used to reconstruct the original 3D world.
	float DdrawConvertHomogeneousToWorldDepthOffset = 0.0f;		// The offset to add to the geometry so it does not clip into the near plane.
	bool DdrawLogTextureAtlas = false;			// Log texture atlas blit operations for RTX Remix analysis
	bool DdrawContentCapture = false;			// Phase A.7: dump unique texture content as PNGs under <gamedir>\_capture_phase_a7\ with a manifest CSV
	bool DdrawStage3BoundDump = false;			// Stage 3 bound-corpus dump ONLY (one PNG per distinct whole-surface bound hash into _capture_phase_a7*\stage3_bound\). Split from DdrawContentCapture (2026-06-09) so the cheap ~hundreds-of-files dump can stay on without paying the full A.7 capture's ~60k files/session.
	bool DdrawRecipeLog = false;				// [RECIPE] (2026-06-10): log each distinct bound whole-surface hash's composition recipe (ordered srcContentHash:srcRect->dstRect Blt list). Decisive cross-session test for write-side canonicalization: drifting hashes sharing identical recipes => deterministic shadow recomposition is viable.
	bool DdrawOwnerLog = false;					// [OWNUP] (2026-06-10): read-only call-site detour at DKII.exe 0x58E53B (the 78% texture-upload blit) that logs the source descriptor's identity field (+0x24 name index -> EngineTextures fullname via name table [0x78E564]). Confirms whether DK2's pooled atlas slots carry STABLE source identity (so replacements can be authored against identity, not drifting hashes). Patches one rel32 in DKII.exe; the blit itself is unchanged. One run -> revert to 0. See dk2_recipe_canonicalization.
	DWORD DdrawNameKey = 0;						// [NAMEKEY] B-v1 (2026-06-11): name-keyed canon resolution. Installs the (read-only) 0x58E53B upload detour to record per-page placements {nameIdx, mip, x, y, w, h}; a full-surface staging->texture Blt attaches them to the dest texture. At bind, a single-name full-page recipe resolves via _canonical\name_map.csv (key = fullname+MM{mip}) to a canonical -- EXACT identity, no fingerprint. 0=off, 1=log-only shadow mode (fingerprint path stays authoritative; logs what the name key WOULD do + agreement stats), 2=name key authoritative (pixel-verified; fingerprint as fallback). Requires DdrawCanonicalRebind=1 to take effect.
	DWORD DdrawNameKeyCrop = 0;					// [NAMEKEY] V2 (2026-06-11): placement-keyed CROP resolution. When a draw's UV bbox falls inside exactly one recorded placement of the bound page's recipe, the UniversalDecompose sub-texture is resolved by (name, mip) through _canonical\crop_payloads.bin: the live page rect is nibble-verified against the expected payload (CanonVerify gate) and the bound crop is BUILT from that payload (mip-0 verbatim -> Remix hash == XXH3(payload), offline-known for all 5767 entries; see _scratch_discovery\crop_hashes.csv). Makes ktorch0-19/gtorch0-19/GSparkle* etc. individually addressable per FRAME. 0=off, 1=shadow (match+verify+log only, binds nothing), 2=serve crops. Requires DdrawNameKey>0 and DdrawUniversalDecompose=1.
	bool DdrawDynamicUiEmissive = false;		// [NAMEKEY] dynamic-UI emissive (2026-06-11): draws whose placement/recipe resolves to a RUNTIME-RENDERED UI name (PointerInfo* mouse pointer strips, ToolTip*, map_texture minimap, FollowPathSubt*) get their blend state forced to additive for that one draw -- Remix treats additive as emissive (torch-flame mechanism), making hash-unstable dynamic UI self-lit in the dark. Requires DdrawNameKey>0 (the name identification) + DdrawNameKeyCrop>0 (placement classification).
	bool DdrawWaterCollapse = false;			// [NAMEKEY] water frame-collapse (2026-06-11): WaterN/WaterSOLIDN crop serves are all collapsed onto frame 0's payload (verify still runs against the page's REAL frame). The water surface stops cycling -> Remix's denoiser converges (reflections stop resetting 32x/cycle) and ONE translucent mod entry covers every frame. Requires DdrawNameKeyCrop=2.
	bool DdrawWaterFlatten = false;				// [NAMEKEY] water mesh flatten (2026-06-11): the game waves water surface vertices per frame (registry Sine Wave Water=0 notwithstanding); under translucent path-traced water the bobbing churns refraction/reflection. Draws classified as water (name WaterN/WaterSOLIDN via the placement system) get their inverse-projected WORLD heights snapped to the draw's average -> glass-still surface. Texture stilling is DdrawWaterCollapse; this stills the GEOMETRY.
	bool DdrawWaterCropLarge = false;			// [NAMEKEY] water crop serving for LARGE draws (2026-06-11, run-15): large water meshes (34-340 verts) fail the UniversalDecompose vcount gate, so they bound the WHOLE page -- opaque, frame-animated art while small water draws got the collapsed translucent crop (two different-looking water pipelines, user-visible at different zooms). This lets recipe-classified water draws through the gate regardless of vcount: their UV bboxes are measured to sit inside ONE placement and the served crops are POT 32x32 (the historic 115-vert Remix AV involved an NPOT crop). Own flag = one-switch rollback if Remix's BVH objects to big-mesh UV rewrites after all. Requires DdrawWaterFlatten + DdrawNameKeyCrop=2.
	bool DdrawCanonicalRebind = false;			// Canonical Identity Layer (2026-06-09): at whole-surface bind, resolve the bound texture's content (exact hash, then 32x32-L2 fingerprint vs _canonical\canon_fps.bin) to a frozen canonical texture loaded from _canonical\tex\<HASH>.a4r4 and bind THAT instead. Remix then sees one stable, offline-precomputed hash per content regardless of DK2's per-session atlas-composition drift.
	bool DdrawReplaceHeap = false;				// CRT heap replacement (2026-06-12, perf): detour DKII.exe's five VC6 CRT allocator entry points (__nh_malloc 0x632680, _realloc 0x6324B0, _free 0x632CA0, _calloc 0x632FA0, __msize 0x633EA0) to a private low-fragmentation Win32 heap, bypassing the 1999 small-block-heap + global lock(9) that serializes every alloc/free across the sound+main threads. Install is all-or-nothing and verify-gated on exact prologue bytes (mismatch => game stays on its original heap). Ported in spirit from DiaLight/Flame replace_heap. See dk2_heap_replacement_2026_06_12 + ddraw\HeapReplace.cpp. Default OFF; A/B against it for the scroll/heavy-scene stutter.
	DWORD DdrawMenuResWidth = 0;				// Front-end menu resolution override width (2026-06-12, port of DiaLight/Flame flame:menu-res): rewrites launchGame's hardcoded 640x480 push-immediates at DKII.exe 0x52BB66 so the main menu opens at this size; the exe's own GUI scaler handles layout + hit-testing at any size (see ddraw\MenuRes.cpp). 0 = off; both width and height must be set. Verify-gated; inert on -level launches.
	DWORD DdrawMenuResHeight = 0;				// Front-end menu resolution override height. See DdrawMenuResWidth.
	bool DdrawOrphanOverlayLift = false;		// Orphan-overlay lift (2026-06-12, menu-text fix): constant-RHW draws binding a RECIPELESS page (runtime-composited content = menu text, loading bars; never sourced from the tracked EngineTextures upload) get their screen z lifted to a fixed near depth in the inverse projection (painter's-order epsilon per draw) and are drawn additive (self-lit, dynUi mechanism). Fixes front-end menu text being depth-buried inside the path-traced 3D backdrop (floodlight-tested invisible). Torch billboards et al. have recipes and are excluded by construction. Requires DdrawNameKey>0.
	bool DdrawMenuBlitOverlay = false;			// [BLITQUAD] (2026-06-12, menu-text fix root cause): DK2's front end composites its UI -- ALL the menu text -- via 2D Blts straight onto the backbuffer (MENUDIAG run: 2013 backbuffer Blts/menu visit; color-keyed 79x38 sprite layers at screen coords). The path tracer discards the rasterized frame, so that UI can never appear. This queues a texture copy of every such Blt and re-emits them at EndScene as XYZRHW quads through the normal draw path, where the orphan-overlay lift makes them near-depth + additive (self-lit). Inert in-game (the game proper never Blts to the backbuffer mid-scene). Requires DdrawOrphanOverlayLift.
	bool DdrawCollapseAnimationPools = false;	// Path B: detect per-instance animation pools (one source -> N dests) and redirect SetTexture to a canonical member to stabilize hashes for Remix replacement
	bool DdrawAtlasDecompose = false;			// Phase A.10: split known k-in-1 atlases into per-region sub-textures at SetTexture time and rewrite drawcall UVs to [0,1] so Remix sees per-region content hashes
	bool DdrawDrawCallerLog = false;			// [DRAWCALLER] (2026-06-18, RE): at DrawIndexedPrimitive, stack-scan for the GAME's return address (the caller in DKII.exe .text 0x401000-0x64D431) to find DK2's render/draw call sites. The LOD-selection routine sits just before these calls. Logs [DRAWCALLER] NEW/SUMMARY (distinct sites + icount/vcount ranges). Read-only.
	bool DdrawGeomProbe = false;				// [GEOMPROBE] geometry identity stability probe (2026-06-16, read-only): at DrawIndexedPrimitive, hash the GAME-submitted index list + UVs + counts (topology & UVs are invariant across animation/position; only XYZ bakes per frame) into a position-independent geomId, and track per geomId how many DISTINCT position-sets it's drawn with. Decides whether DK2 geometry has a stable identity Remix replacements could bind to (the mesh analog of the texture name-key). Logs [GEOMPROBE] NEW / SUMMARY. Zero behavioral change; never rewrites geometry.
	bool DdrawUniversalDecompose = false;		// Universal UV-region decompose (2026-05-21): for ANY drawcall whose stage-0 UV bbox is a proper sub-rectangle of its bound texture, crop that exact region into a content-hash-keyed sub-texture and rewrite UVs to [0,1]. No fingerprints/hardcoded hashes. General fix for world-texture-co-atlased-with-UI (DK2 torch white-flash). Separate from / mutually-exclusive with DdrawAtlasDecompose.
	bool DdrawNoDrawBufferSysLock = false;		// Disables Draw CriticalSection and sets NOSYSLOCK on Index and Vertex Buffer locks
	bool DdrawNoMultiThreaded = false;			// Don't add D3DCREATE_MULTITHREADED flag when creating Direct3D9 device unless the game requests it
	bool DdrawUseDirect3D9Caps = false;			// Use Direct3D9 (Dd7to9) for GetCaps
	bool DdrawUseShadowSurface = false;			// Use shadow surface with Dd7to9 for render target Locks/GetDC
	bool DdrawUseNativeResolution = false;		// Uses the current screen resolution for Dd7to9
	bool DdrawVertexLockDiscard = false;		// Sets the discard flag for vertex Lock
	DWORD DdrawClippedWidth = 0;				// Used to scaled Direct3d9 to use this width when using Dd7to9
	DWORD DdrawClippedHeight = 0;				// Used to scaled Direct3d9 to use this height when using Dd7to9
	DWORD DdrawCustomWidth = 0;					// Custom resolution width for Dd7to9 when using DdrawLimitDisplayModeCount, resolution must be supported by video card and monitor
	DWORD DdrawCustomHeight = 0;				// Custom resolution height for Dd7to9 when using DdrawLimitDisplayModeCount, resolution must be supported by video card and monitor
	bool DdrawDisableDirect3DCaps = false;		// Disable caps for Direct3D to try and force the game to use DirectDraw instaed of Direct3D
	bool DdrawLimitTextureFormats = false;		// Limits the number of texture formats sent to the program, some games crash when you feed them with too many textures
	bool DdrawLimitDisplayModeCount = false;	// Limits the number of display modes sent to program, some games crash when you feed them with too many resolutions
	DWORD DdrawOverrideBitMode = 0;				// Forces DirectX to use specified bit mode: 8, 16, 24, 32
	DWORD DdrawOverrideWidth = 0;				// Force Direct3d9 to use this width when using Dd7to9
	DWORD DdrawOverrideHeight = 0;				// Force Direct3d9 to use this height when using Dd7to9
	DWORD OverrideRefreshRate = 0;				// Force Direct3d9 to use this refresh rate, only works in exclusive fullscreen mode
	DWORD OverrideStencilFormat = 0;			// Force Direct3d9 to use this AutoStencilFormat
	DWORD DdrawOverrideStencilFormat = 0;		// Force Direct3d9 to use this AutoStencilFormat when using Dd7to9
	DWORD DdrawFlipFillColor = 0;				// Color used to fill the primary surface before flipping
	bool DdrawForceMipMapAutoGen = false;		// Force Direct3d9 to use this AutoStencilFormat when using Dd7to9
	bool DdrawEnableMouseHook = false;			// Allow to hook into mouse to limit it to the chosen resolution
	bool DdrawDisableLighting = false;			// Allow to disable lighting
	DWORD DdrawHookSystem32 = 0;				// Hooks the ddraw.dll file in the Windows System32 folder
	DWORD D3d8HookSystem32 = 0;					// Hooks the d3d8.dll file in the Windows System32 folder
	DWORD D3d9HookSystem32 = 0;					// Hooks the d3d9.dll file in the Windows System32 folder
	DWORD DinputHookSystem32 = 0;				// Hooks the dinput.dll file in the Windows System32 folder
	DWORD Dinput8HookSystem32 = 0;				// Hooks the dinput8.dll file in the Windows System32 folder
	DWORD DsoundHookSystem32 = 0;				// Hooks the dsound.dll file in the Windows System32 folder
	DWORD DeviceLookupCacheTime = 0;			// Number of seconds to cache the DeviceEnum callback data
	bool DirectShowEmulation = false;			// Emulates DirectShow APIs
	bool DisableGameUX = false;					// Disables the Microsoft Game Explorer which can sometimes cause high CPU in rundll32.exe and hang the game process
	bool DisableGDIGammaRamp = false;			// Disables gamma ramp for GDI, some games look washed out with gamme ramp enabled
	bool DisableHighDPIScaling = false;			// Disables display scaling on high DPI settings
	bool DisableLogging = false;				// Disables the logging file
	DWORD SetSwapEffectShim = 0;				// Disables the call to d3d9.dll 'Direct3D9SetSwapEffectUpgradeShim' to switch present mode
	DWORD CacheClipPlane = 0;					// Caches the ClipPlane for Direct3D9 to fix an issue in d3d9 on Windows 8 and newer
	bool EnvironmentCubeMapFix = false;			// Fixes environment cube maps when no texture is applied, issue exists in d3d8
	DWORD CustomDisplayWidth = 0;				// Custom resolution width when using LimitDisplayModeCount, resolution must be supported by video card and monitor
	DWORD CustomDisplayHeight = 0;				// Custom resolution height when using LimitDisplayModeCount, resolution must be supported by video card and monitor
	bool EnableDdrawWrapper = false;			// Enables the ddraw wrapper
	DWORD EnableD3d9Wrapper = 0;				// Enables the d3d9 wrapper
	bool EnableDinput8Wrapper = false;			// Enables the dinput8 wrapper
	bool EnableDsoundWrapper = false;			// Enables the dsound wrapper
	bool EnableImgui = false;					// Enables imgui for debugging
	DWORD EnableMultisamplingATOC = 0;			// Enables transparency multisampling (ATOC). 1) Just enable ATOC. 2) Enable ATOC and AlphaTest Render State
	bool EnableOpenDialogHook = false;			// Enables the hooks for the open dialog box
	bool EnableWindowMode = false;				// Enables WndMode for d3d9 wrapper
	bool EnableVSync = false;					// Enables VSync for d3d9 wrapper
	bool FixHighFrequencyMouse = false;			// Gets the latest mouse status by merging the DirectInput buffer data
	float MouseMovementFactor = 1.0f;			// Sets the mouse movement speed factor, requires enabling FixHighFrequencyMouse
	DWORD MouseMovementPadding = 0;				// Adds extra mouse movement to overcome issues with input deadzone in some games, requires enabling FixHighFrequencyMouse
	DWORD FixPerfCounterUptime = 0;				// Reduces uptime counters to prevent slowdowns in games
	bool ForceExclusiveFullscreen = false;		// Forces exclusive fullscreen mode in d3d9
	DHEX ForceKeyboardLayout = {};				// Force specific keyboard layout
	bool ForceMixedVertexProcessing = false;	// Forces Mixed mode for vertex processing in d3d9
	bool ForceSystemMemVertexCache = false;		// Forces System Memory caching for vertexes in d3d9
	bool ForceSingleBeginEndScene = false;		// Ensures that only a single EndScene/BeginScene pair are called per frame
	bool FullScreen = false;					// Sets the main window to fullscreen
	bool FullscreenWindowMode = false;			// Enables fullscreen windowed mode, requires EnableWindowMode
	bool ForceTermination = false;				// Terminates application when main window closes
	bool ForceWindowResize = false;				// Forces main window to fullscreen, requires FullScreen
	bool ForceVsyncMode = false;				// Forces d3d9 game to use EnableVsync option
	DWORD GraphicsHybridAdapter = 0;			// Sets the Direct3D9 Hybrid Enumeration Mode to allow using a secondary display adapter
	bool HandleExceptions = false;				// Handles unhandled exceptions in the application
	bool isAppCompatDataSet = false;			// Flag that holds tells whether any of the AppCompatData flags are set
	bool LimitDisplayModeCount = false;			// Limits the number of display modes sent to program, some games crash when you feed them with too many resolutions
	float LimitPerFrameFPS = 0;					// Limits each frame by adding a delay if the frame is to fast
	bool LoadPlugins = false;					// Loads ASI plugins
	bool LoadFromScriptsOnly = false;			// Loads ASI plugins from 'scripts' and 'plugins' folder only
	bool ProcessExcluded = false;				// Set if this process is excluded from dxwrapper functions
	bool ResetScreenRes = false;				// Reset the screen resolution on close
	DWORD LimitStateBlocks = 0;					// Reuses state block interfaces to prevent memory leaks
	bool SendAltEnter = false;					// Sends an Alt+Enter message to the wind to tell it to go into fullscreen, requires FullScreen
	bool UseShadowBackbuffer = false;			// Enables shadow backbuffer for d3d8to and Direct3D9 games
	bool WaitForProcess = false;				// Waits for process to end before continuing, requires FullScreen
	bool WaitForWindowChanges = false;			// Waits for window handle to stabilize before setting fullsreen, requires FullScreen
	bool WindowModeBorder = false;				// Enables the window border when EnableWindowMode is set, requires EnableWindowMode
	DWORD WindowModeGammaShader = 0;			// Use shader for gamma: 1 = when in window mode; 2 = for both window and exclusive fullscreen mode
	bool SetInitialWindowPosition = false;		// Enable Initial window position
	DWORD InitialWindowPositionLeft;			// Initial left window position for application
	DWORD InitialWindowPositionTop;				// Initial top window position for application
	DWORD LoopSleepTime = 0;					// Time to sleep between each window handle check loop, requires FullScreen
	DWORD ResetMemoryAfter = 0;					// Undo hot patch after this amount of time
	DWORD WindowSleepTime = 0;					// Time to wait (sleep) for window handle and screen updates to finish, requires FullScreen
	DWORD ShowFPSCounter = 0;					// Shows the FPS counter. 1 = top left; 2 = top right; 3 = bottom right; 4 = bottom left
	DWORD SingleProcAffinity = 0;				// Sets the CPU affinity for this process
	DWORD SetFullScreenLayer = 0;				// The layer to be selected for fullscreen, requires FullScreen
	DWORD SetPOW2Caps = 0;						// Force caps change: 1 = force both, 2 = force D3DPTEXTURECAPS_NONPOW2CONDITIONAL, 3 = force D3DPTEXTURECAPS_POW2, 4 = remove both
	DWORD AnisotropicFiltering = 0;				// Enable Anisotropic Filtering for d3d9
	DWORD AntiAliasing = 0;						// Enable AntiAliasing for d3d9 CreateDevice
	bool FlipEx = false;						// Enable FlipEx presentation mode for D3D9Ex. Disables AntiAliasing
	DWORD RealWrapperMode = 0;					// Internal wrapper mode
	MEMORYINFO VerifyMemoryInfo;				// Memory used for verification before hot patching
	std::string WinVersionLie = "";				// Using DDrawCompat WinVersionLie to tell the OS a different OS
	DWORD WinVersionLieSP = 0;					// Using DDrawCompat WinVersionLie to tell the OS a different OS
	std::vector<MEMORYINFO> MemoryInfo;			// Addresses and memory used in hot patching
	std::string RealDllPath;					// Manually set Dll to wrap
	std::string RunProcess;						// Process to run on load
	std::string WrapperMode;					// Mode of dxwrapper from config file
	std::string WrapperName;					// dxwrapper dll filename
	std::vector<std::string> SetNamedLayer;		// List of named layers to select for fullscreen
	std::vector<std::string> IgnoreWindowName;	// List of window classes to ignore
	std::vector<std::string> LoadCustomDllPath;	// List of custom dlls to load
	std::vector<std::string> ExcludeProcess;	// List of excluded applications
	std::vector<std::string> IncludeProcess;	// List of included applications

	// Dinput8
	bool FilterNonActiveInput = 0;

	// SetAppCompatData
	bool DXPrimaryEmulation[13] = { false };	// SetAppCompatData exported functions from ddraw.dll
	DWORD LockColorkey = 0;						// DXPrimaryEmulation option that needs a second parameter
	bool DisableMaxWindowedModeNotSet = false;	// If the DisableMaxWindowedMode option exists in the config file

	// DirectSoundControl https://github.com/nRaecheR/DirectSoundControl
	DWORD Num2DBuffers = 0;
	DWORD Num3DBuffers = 0;
	bool ForceCertification = false;
	bool ForceExclusiveMode = false;
	bool ForceSoftwareMixing = false;
	bool ForceHardwareMixing = false;
	bool ForceHQ3DSoftMixing = false;
	bool ForceNonStaticBuffers = false;
	bool ForceVoiceManagement = false;
	bool ForcePrimaryBufferFormat = false;
	DWORD PrimaryBufferBits = 0;
	DWORD PrimaryBufferSamples = 0;
	DWORD PrimaryBufferChannels = 0;
	bool AudioClipDetection = false;
	DWORD AudioFadeOutDelayMS = 0;
	bool FixSpeakerConfigType = false;
	bool StoppedDriverWorkaround = false;
};
extern CONFIG Config;

namespace Settings
{
	bool IfStringExistsInList(const char*, std::vector<std::string>, bool = true);
	void SetValue(char*, char*, std::vector<std::string>*);
	void ClearConfigSettings();
}

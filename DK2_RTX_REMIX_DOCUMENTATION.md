# Dungeon Keeper 2 - RTX Remix Path Tracing Project Documentation

## Project Overview

This project achieves **path tracing on a DirectX 6 game** (Dungeon Keeper 2, released 1999) using a modified dxwrapper and NVIDIA RTX Remix. This is believed to be the first successful implementation of RTX path tracing on a true DX6 title.

**Key Achievement:** Reverse-projecting pre-transformed screen-space vertices (XYZRHW) back into world-space 3D geometry that RTX Remix can ray trace.

---

## Table of Contents

1. [DirectX Version Analysis](#directx-version-analysis)
2. [Architecture Overview](#architecture-overview)
3. [XYZRHW to World Space Conversion](#xyzrhw-to-world-space-conversion)
4. [Files Modified (January 19, 2026)](#files-modified-january-19-2026)
5. [Configuration Settings](#configuration-settings)
6. [Texture Atlas Problem](#texture-atlas-problem)
7. [Proposed Texture Atlas Solution](#proposed-texture-atlas-solution)
8. [Key Code Locations Reference](#key-code-locations-reference)
9. [Next Steps](#next-steps)

---

## DirectX Version Analysis

### Confirmed: DirectX 6 (IDirectDraw4 / IDirect3D3)

Binary analysis of `DKII.exe` using radare2 disassembly confirmed the exact DirectX interfaces used.

#### Import Table Analysis
```
DLL Name: DDRAW.dll
    DirectDrawEnumerateA
    DirectDrawCreate        ← NOT DirectDrawCreateEx (DX7 required)
```

The game uses `DirectDrawCreate` (not `DirectDrawCreateEx`), which means it **cannot** use DX7 interfaces.

#### COM Interface GUIDs Found in Binary

| Interface | DirectX Version | Found in Binary | Address |
|-----------|-----------------|-----------------|---------|
| IDirectDraw | DX1 | YES | - |
| IDirectDraw2 | DX5 | YES | 0x00269a90 |
| **IDirectDraw4** | **DX6** | **YES** | **0x0066bca8** |
| IDirectDraw7 | DX7 | NO | - |
| IDirect3D | DX3 | YES | - |
| IDirect3D2 | DX5 | YES | 0x00269db0 |
| **IDirect3D3** | **DX6** | **YES** | **0x0066bfc8** |
| IDirect3D7 | DX7 | NO | - |

#### Disassembly Proof of QueryInterface Calls

```asm
0x56f3a3: push 0x66bca8    ; IID_IDirectDraw4 GUID
0x56f3a8: mov ecx, [eax]
0x56f3aa: push eax
0x56f3ab: call [ecx]       ; QueryInterface()

0x56f4ab: push 0x66bfc8    ; IID_IDirect3D3 GUID
0x56f4b0: push eax
0x56f4b1: mov ecx, [eax]
0x56f4b3: call [ecx]       ; QueryInterface()
```

#### Additional Evidence
- MMX Software Renderer dated "1998/9" embedded in binary
- "RGB Emulation" and "Ramp" rendering modes (DX5/6 era)
- No shader references (shaders introduced in DX8)
- No Hardware T&L references (introduced in DX7)
- Game shipped June 1999, DX6.1 was current

---

## Architecture Overview

### DLL Chain
```
DKII.exe (32-bit, DirectX 6)
    │
    ▼
dxwrapper.dll (32-bit) - Custom build with XYZRHW conversion
    │
    ├── ddraw.dll wrapper - Converts DDraw7 → D3D9
    │
    ▼
d3d9.dll (32-bit stub) - NV Bridge client
    │
    ▼ [IPC across 32→64 bit boundary]
    │
NV Bridge (64-bit)
    │
    ▼
RTX Remix Runtime (64-bit) - Path tracing
    │
    ▼
GPU (RTX hardware)
```

### Data Flow
```
DK2 (IDirectDraw4/IDirect3D3)
    ↓ [compatibility shim]
DDraw7 interface
    ↓ [dxwrapper Dd7to9 conversion]
D3D9 + XYZRHW→World reconstruction
    ↓ [RTX Remix interception]
Path Traced Output
```

---

## XYZRHW to World Space Conversion

### The Problem

Old DirectX 6/7 games use **pre-transformed vertices** (D3DFVF_XYZRHW):
- Vertices are already projected to 2D screen space
- Format: (X, Y, Z, RHW) where X,Y are screen coordinates
- RTX Remix needs **world-space 3D geometry** to cast rays
- No existing wrapper solved this problem

### The Solution

**Inverse projection pipeline** implemented in dxwrapper:

```
Screen Space (X, Y, Z, RHW)
    ↓ [Inverse View-Projection Matrix]
World Space (X, Y, Z)
    ↓ [New View + Projection matrices for RTX]
RTX Remix Path Tracing
```

### Core Algorithm (IDirect3DDeviceX.cpp:2977-2997)

```cpp
for (UINT x = 0; x < dwVertexCount; x++)
{
    // Read screen-space vertex
    float* srcpos = (float*)sourceVertex;
    float* trgtpos = (float*)targetVertex;

    // Transform from screen space back to world space
    DirectX::XMVECTOR xpos = DirectX::XMVectorSet(srcpos[0], srcpos[1], srcpos[2], srcpos[3]);
    DirectX::XMVECTOR xpos_global = DirectX::XMVector3TransformCoord(xpos, ConvertHomogeneous.ToWorld_ViewMatrixInverse);
    xpos_global = DirectX::XMVectorDivide(xpos_global, DirectX::XMVectorSplatW(xpos_global));

    // Write world-space vertex
    trgtpos[0] = DirectX::XMVectorGetX(xpos_global);
    trgtpos[1] = DirectX::XMVectorGetY(xpos_global);
    trgtpos[2] = DirectX::XMVectorGetZ(xpos_global);

    // Copy remaining vertex data (colors, UVs, etc.)
    std::memcpy(targetVertex + sizeof(float) * 3, sourceVertex + sizeof(float) * 4, restSize);

    sourceVertex += stride;
    targetVertex += targetStride;
}
```

### Inverse Matrix Calculation (IDirect3DDeviceX.cpp:2752-2758)

```cpp
// Build inverse transformation matrix
DirectX::XMMATRIX toViewSpace = DirectX::XMLoadFloat4x4((DirectX::XMFLOAT4X4*)&view);
DirectX::XMMATRIX vp = DirectX::XMMatrixMultiply(viewMatrix, proj);
DirectX::XMMATRIX vpinv = DirectX::XMMatrixInverse(nullptr, vp);
DirectX::XMMATRIX depthoffset = DirectX::XMMatrixTranslation(0.0f, 0.0f, depthOffset);

// Full transform: screen → NDC → clip → view → world
ConvertHomogeneous.ToWorld_ViewMatrixInverse = DirectX::XMMatrixMultiply(depthoffset, DirectX::XMMatrixMultiply(toViewSpace, vpinv));
```

### DK2-Specific Camera Setup (IDirect3DDeviceX.cpp:2728-2730)

```cpp
// DK2 isometric camera - positioned above, looking down at ~45 degree angle
position = DirectX::XMVectorSet(0.0f, 800.0f, -800.0f, 0.0f);
direction = DirectX::XMVector3Normalize(DirectX::XMVectorSet(0.0f, -0.707f, 0.707f, 0.0f));
```

### Data Structures

#### CONVERTHOMOGENEOUS Struct (IDirect3DTypes.h:389-399)

```cpp
struct CONVERTHOMOGENEOUS
{
    bool IsTransformViewSet = false;                    // Remembers if game sets the view matrix
    D3DMATRIX ToWorld_ProjectionMatrix;                 // Projection matrix for GPU transform
    D3DMATRIX ToWorld_ViewMatrix;                       // View matrix for GPU transform
    D3DMATRIX ToWorld_ViewMatrixOriginal;               // Original view matrix (for restoration)
    DirectX::XMMATRIX ToWorld_ViewMatrixInverse;        // Inverse matrix for CPU vertex transform
    std::vector<uint8_t> ToWorld_IntermediateGeometry;  // Buffer for converted vertices
    float ToWorld_GameCameraYaw = 0.0f;
    float ToWorld_GameCameraPitch = 0.0f;
};
```

---

## Files Modified (January 19, 2026)

These are the files with actual code changes made during the recent development session (not inherited from the fork).

### 1. NEW FILE: `ddraw/RemixAPI.h`
**Modified:** 2026-01-19 06:41

Complete RTX Remix C API header for direct camera control, bypassing D3D9 detection.

**Contents:**
- Error codes (`remixapi_ErrorCode`)
- Struct types (`remixapi_StructType`)
- Camera types (`remixapi_CameraType`)
- `remixapi_CameraInfo` struct
- `remixapi_CameraInfoParameterizedEXT` struct
- `remixapi_Interface` struct (vtable)
- `RemixAPIManager` C++ singleton class

**Key Code:**
```cpp
class RemixAPIManager {
public:
    static RemixAPIManager& Instance();
    bool Initialize(const char** outError = nullptr);
    bool IsInitialized() const;
    remixapi_ErrorCode SetupCamera(const remixapi_CameraInfo* info);
    remixapi_ErrorCode SetConfigVariable(const char* var, const char* value);
private:
    bool m_initialized;
    remixapi_Interface m_interface;
    PFN_remixapi_InitializeLibrary m_pfnInitialize;
};
```

**Note:** Direct Remix API may not work through the 32→64 bit NV Bridge. The D3D9 transform method is the working solution.

---

### 2. `d3d9/d3d9.cpp`
**Modified:** 2026-01-19 05:45

**Change:** Prefer already-loaded d3d9.dll (RTX Remix) over system DLL.

```cpp
// BEFORE:
// Get System d3d9.dll

// AFTER:
// Get d3d9.dll - prefer already loaded (RTX Remix) over system
if (!h_d3d9)
{
    // First try to get already loaded d3d9.dll (RTX Remix if present)
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, "d3d9.dll", &h_d3d9);
    // Fall back to system d3d9.dll if not found
}
```

---

### 3. `ddraw/IDirect3DDeviceX.cpp`
**Modified:** 2026-01-19 14:01

**Major Changes:**

#### A. Include RemixAPI
```cpp
#include "RemixAPI.h"
```

#### B. BeginScene - Set camera matrices BEFORE draw calls
```cpp
// Set 3D camera matrices at START of scene for RTX Remix camera detection
// Must be done BEFORE draw calls, as camera is detected during draw call processing
if (Config.DdrawConvertHomogeneousToWorld)
{
    if (!ConvertHomogeneous.IsTransformViewSet)
    {
        // Initialize camera matrices...
        ConvertHomogeneous.IsTransformViewSet = true;
    }
    (*d3d9Device)->SetTransform(D3DTS_VIEW, &ConvertHomogeneous.ToWorld_ViewMatrix);
    (*d3d9Device)->SetTransform(D3DTS_PROJECTION, &ConvertHomogeneous.ToWorld_ProjectionMatrix);
}
```

#### C. EndScene - RemixAPI integration attempt
```cpp
if (Config.DdrawConvertHomogeneousToWorld)
{
    RemixAPIManager& remixApi = RemixAPIManager::Instance();
    if (!remixApi.IsInitialized())
    {
        if (remixApi.Initialize(&initError))
        {
            Logging::Log() << __FUNCTION__ << " RTX Remix API initialized successfully";
        }
    }
}
```

#### D. CRITICAL FIX - Don't restore transforms after drawing
```cpp
// BEFORE (broken):
(*d3d9Device)->SetTransform(D3DTS_VIEW, &ConvertHomogeneous.ToWorld_ViewMatrixOriginal);

// AFTER (working):
// NOTE: Do NOT restore transform - keep 3D camera matrices set for RTX Remix detection
// The original code restored the matrices here which prevented RTX Remix from detecting the camera
```

**This was the key bug fix!** RTX Remix reads matrices at end of frame, but they were being restored to incorrect values.

#### E. Added diagnostic logging
```cpp
Logging::Log() << __FUNCTION__ << " Converting XYZRHW to world space (DrawPrimitive)";
Logging::Log() << "  Projection matrix [0][0]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._11
    << " [1][1]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._22;
```

---

### 4. `bin/Release/dx*/dxwrapper.ini`
**Modified:** 2026-01-19 14:01

Updated configuration files with working settings.

---

## Configuration Settings

### dxwrapper.ini Key Settings

```ini
; === Conversion Pipeline ===
Dd7to9 = 1                                    ; Convert DirectDraw 7 → Direct3D 9
EnableD3d9Wrapper = 1                         ; Enable D3D9 wrapper for RTX Remix

; === XYZRHW Vertex Conversion (Critical for RTX) ===
DdrawDisableLighting = 1                      ; Disable fixed-function lighting
DdrawConvertHomogeneousW = 1                  ; Enable XYZRHW → XYZW conversion
DdrawConvertHomogeneousToWorld = 1            ; Reconstruct world-space geometry
DdrawConvertHomogeneousToWorldFOV = 60.0      ; Camera FOV for reconstruction
DdrawConvertHomogeneousToWorldNearPlane = 0.1 ; Near clip plane
DdrawConvertHomogeneousToWorldFarPlane = 5000.0 ; Far clip plane
DdrawConvertHomogeneousToWorldDepthOffset = 0.0 ; Depth offset to prevent near-plane clipping

; === Window Mode ===
EnableWindowMode = 1
FullscreenWindowMode = 1
```

### rtx.conf Key Settings

```ini
rtx.orthographicIsUI = False          ; Prevent geometry from being classified as UI
rtx.camera.freeCameraPosition = 0, 800, -800
rtx.camera.freeCameraPitch = -45
rtx.emissiveIntensity = 5
; Texture hashes for UI, terrain, lightmaps configured
```

### Settings.h Definitions (Lines 75-81, 282-288)

```cpp
// Macro-based setting definitions
visit(DdrawConvertHomogeneousW)
visit(DdrawConvertHomogeneousToWorld)
visit(DdrawConvertHomogeneousToWorldUseGameCamera)
visit(DdrawConvertHomogeneousToWorldFOV)
visit(DdrawConvertHomogeneousToWorldNearPlane)
visit(DdrawConvertHomogeneousToWorldFarPlane)
visit(DdrawConvertHomogeneousToWorldDepthOffset)

// Default values
bool DdrawConvertHomogeneousW = false;
bool DdrawConvertHomogeneousToWorld = false;
bool DdrawConvertHomogeneousToWorldUseGameCamera = false;
float DdrawConvertHomogeneousToWorldFOV = 0.0f;         // Default: 90.0f
float DdrawConvertHomogeneousToWorldNearPlane = 0.0f;   // Default: 1.0f
float DdrawConvertHomogeneousToWorldFarPlane = 0.0f;    // Default: 1000.0f
float DdrawConvertHomogeneousToWorldDepthOffset = 0.0f;
```

---

## Texture Atlas Problem

### Description

DK2 uses a **dynamic texture atlas** system that breaks RTX Remix's hash-based texture replacement:

1. Game loads individual source textures
2. Blits (copies) multiple textures into a shared "atlas" surface
3. If mipmap level > 0, combines multiple textures
4. Uses the atlas surface for rendering
5. **RTX Remix hashes the atlas content**
6. Atlas content changes every frame → hash changes → **texture replacement fails**

### Visual Representation

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GAME SIDE                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Source Texture A ──┐                                                   │
│  (stable content)   │                                                   │
│  Hash: 0xAABBCCDD   │    Blt()                                          │
│                     │                                                   │
│  Source Texture B ──┼──────────────►  Atlas Surface                     │
│  (stable content)   │                 (dynamic content)                 │
│  Hash: 0x11223344   │                 Hash: ??? (changes!)              │
│                     │                       │                           │
│  Source Texture C ──┘                       │                           │
│  (stable content)                           │                           │
│  Hash: 0x55667788                           ▼                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  DXWRAPPER (ddraw/IDirectDrawSurfaceX.cpp)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Blt() @ line 421                                                       │
│    │                                                                    │
│    ├─► lpDDSrcSurface  (source - HASHABLE, stable)                      │
│    │     └─► m_IDirectDrawSurfaceX* lpDDSrcSurfaceX                     │
│    │           └─► GetD3d9Texture() → IDirect3DTexture9*                │
│    │                                                                    │
│    └─► this (destination atlas - UNHASHABLE, changes)                   │
│          └─► CopySurface() @ line 6587                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  D3D9 / RTX REMIX                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SetTexture(stage, atlas_surface)                                       │
│       │                                                                 │
│       ▼                                                                 │
│  RTX Remix hashes atlas content ──► Hash changes! ──► No replacement    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Happens

The `Blt()` function copies pixel data from source textures into a shared destination surface:

```cpp
// IDirectDrawSurfaceX.cpp:656
hr = CopySurface(lpDDSrcSurfaceX, lpSrcRect, lpDestRect, Filter, ColorKey, Flags, SrcMipMapLevel, MipMapLevel);
```

When the game later calls `SetTexture()` with the atlas:

```cpp
// IDirect3DDeviceX.cpp:4598
IDirect3DTexture9* pTexture9 = lpDDSrcSurfaceX->GetD3d9Texture();
```

RTX Remix sees only the final composited atlas, not the individual source textures.

---

## Proposed Texture Atlas Solution

### Approach 1: Track Source→Region Mapping

Intercept blits and remember which source texture was placed where in the atlas.

#### Data Structure
```cpp
struct AtlasRegionInfo {
    m_IDirectDrawSurfaceX* sourceSurface;  // Original texture pointer
    RECT sourceRect;                        // Which part of source
    RECT destRect;                          // Where in atlas
    DWORD sourceTextureHash;                // Stable hash of source content
    IDirect3DTexture9* sourceD3D9Texture;   // D3D9 texture for hashing
};

// Per-atlas tracking
std::map<m_IDirectDrawSurfaceX*, std::vector<AtlasRegionInfo>> atlasRegions;
```

#### Interception Point (IDirectDrawSurfaceX.cpp, after line 520)

```cpp
// In Blt(), after getting lpDDSrcSurfaceX:

// Check if destination is an atlas surface
if (IsAtlasSurface(this))
{
    AtlasRegionInfo info;
    info.sourceSurface = lpDDSrcSurfaceX;
    info.sourceRect = (lpSrcRect ? *lpSrcRect : RECT{0, 0, srcWidth, srcHeight});
    info.destRect = (lpDestRect ? *lpDestRect : RECT{0, 0, destWidth, destHeight});
    info.sourceD3D9Texture = lpDDSrcSurfaceX->GetD3d9Texture();

    // Compute stable hash of source texture content
    info.sourceTextureHash = ComputeTextureContentHash(info.sourceD3D9Texture);

    atlasRegions[this].push_back(info);

    Logging::Log() << "ATLAS BLIT: src=" << lpDDSrcSurfaceX
                   << " hash=" << std::hex << info.sourceTextureHash
                   << " destRect=(" << info.destRect.left << "," << info.destRect.top
                   << "," << info.destRect.right << "," << info.destRect.bottom << ")";
}
```

### Approach 2: Bypass Atlas, Draw Source Textures Directly

Instead of using the atlas, emit draw calls for each source texture with transformed UVs.

#### In SetTexture or Draw calls:
```cpp
if (IsAtlasSurface(textureSurface) && atlasRegions.count(textureSurface))
{
    // Don't use atlas - draw each source texture separately
    for (const auto& region : atlasRegions[textureSurface])
    {
        // Calculate UV transform to map atlas UVs to source texture UVs
        // Emit separate draw call with source texture
        (*d3d9Device)->SetTexture(stage, region.sourceD3D9Texture);
        // Transform UVs and draw...
    }
}
```

### Approach 3: Hash Source Textures for Remix

Before the atlas is used, emit source texture hashes to Remix:

```cpp
// When atlas is bound for drawing, inform Remix about source textures
if (IsAtlasSurface(textureSurface))
{
    for (const auto& region : atlasRegions[textureSurface])
    {
        // Tell Remix: "this region of the atlas came from texture with hash X"
        // Remix can then apply the correct replacement material
    }
}
```

### Atlas Detection Heuristics

How to identify which surfaces are atlases:

1. **Size** - Atlases are typically larger (256x256, 512x512, 1024x1024)
2. **Multiple blits per frame** - Same destination receives many Blt() calls
3. **MipMap level** - User noted "if mipmap level > 0" triggers atlasing
4. **Surface caps** - Check `DDSCAPS_TEXTURE` with specific patterns
5. **Blit count tracking** - Surfaces receiving >N blits are likely atlases

#### Detection Code
```cpp
// Add to IDirectDrawSurfaceX class or global tracking
static std::map<m_IDirectDrawSurfaceX*, int> blitCountsPerFrame;
static int currentFrame = 0;

// In Blt():
blitCountsPerFrame[this]++;

// At end of frame:
for (auto& [surface, count] : blitCountsPerFrame)
{
    if (count > 5)  // Threshold
    {
        Logging::Log() << "ATLAS DETECTED: " << surface
                       << " Size: " << surface->GetWidth() << "x" << surface->GetHeight()
                       << " BlitCount: " << count;
    }
}
blitCountsPerFrame.clear();
```

---

## Key Code Locations Reference

### Texture/Surface Operations

| File | Line | Function | Purpose |
|------|------|----------|---------|
| `IDirectDrawSurfaceX.cpp` | 421 | `Blt()` | Main blit entry point |
| `IDirectDrawSurfaceX.cpp` | 510-520 | `Blt()` | Gets source surface pointer |
| `IDirectDrawSurfaceX.cpp` | 656 | `Blt()` | Calls `CopySurface()` |
| `IDirectDrawSurfaceX.cpp` | 815 | `BltFast()` | Fast blit (calls Blt internally) |
| `IDirectDrawSurfaceX.cpp` | 6587 | `CopySurface()` | Actual pixel copy operation |
| `IDirect3DDeviceX.cpp` | 4570 | `SetTexture()` | Binds texture for drawing |
| `IDirect3DDeviceX.cpp` | 4598 | `SetTexture()` | Gets D3D9 texture from surface |
| `IDirect3DTextureX.cpp` | 156 | `GetHandle()` | Texture handle assignment |

### XYZRHW Conversion

| File | Line | Function | Purpose |
|------|------|----------|---------|
| `IDirect3DDeviceX.cpp` | 1301-1355 | `BeginScene()` | Camera matrix setup |
| `IDirect3DDeviceX.cpp` | 2663-2774 | `SetTransform()` | Transform matrix handling |
| `IDirect3DDeviceX.cpp` | 2728-2730 | `SetTransform()` | DK2-specific camera position |
| `IDirect3DDeviceX.cpp` | 2943-3024 | `DrawPrimitive()` | XYZRHW vertex conversion |
| `IDirect3DDeviceX.cpp` | 2977-2997 | `DrawPrimitive()` | Core conversion loop |
| `IDirect3DTypes.h` | 389-399 | struct | `CONVERTHOMOGENEOUS` definition |

### Settings

| File | Line | Purpose |
|------|------|---------|
| `Settings/Settings.h` | 75-81 | Setting macro definitions |
| `Settings/Settings.h` | 282-288 | Setting variable declarations |
| `Settings/Settings.cpp` | 523-525 | Default values |

---

## Investigation Log

### Attempt 1: Blt() Tracking (2026-01-22)

**Hypothesis:** Atlas is built via Blt() calls each frame

**Implementation:**
- Added `DdrawLogTextureAtlas` config option
- Tracked all Blt() calls with source/dest surface info
- Logged surfaces receiving 2+ blits per frame as "POTENTIAL ATLAS"

**Results:**
```
POTENTIAL ATLAS: surface=1062E6D0 size=205x132 blitCount=3
  <- SRC: 1054C688 size=205x5412 srcRect=(0,2244,205,2376) destRect=(0,0,205,132)

POTENTIAL ATLAS: surface=1062BC70 size=210x215 blitCount=3
  <- SRC: 1054B868 size=207x97 srcRect=(0,0,207,97) destRect=(0,0,207,97)
```

**Analysis:**
| What We Found | What We Expected |
|---------------|------------------|
| 3 blits/frame | >5 blits/frame |
| Small surfaces (205x132, 210x215) | Large atlases (256x256+) |
| Same source repeated | Multiple different sources |
| Tall sprite sheets (205x5412) | Multiple textures composited |

**Conclusion:** This is **sprite animation extraction**, NOT texture atlasing:
- Source `1054C688` (205x5412) = vertical sprite sheet with all animation frames
- Game extracts 132px slices from different Y positions each frame
- This is standard sprite animation, not the atlas problem

**Files Modified:**
- `Settings/Settings.h` - Added `DdrawLogTextureAtlas` setting
- `IDirectDrawSurfaceX.cpp` - Added blit tracking structures and logging
- `IDirectDrawSurfaceX.h` - Added `LogAtlasTrackingAndReset()` declaration
- `IDirect3DDeviceX.cpp` - Called atlas logging from EndScene

**Next Investigation:** Track Lock/Unlock operations - atlas may be built via direct pixel writes

---

### Attempt 2: Lock/Unlock + SetTexture Tracking (2026-01-22)

**Hypothesis:** Atlas is built via Lock/Unlock (direct pixel writes), or large textures are used

**Implementation:**
- Tracked all Lock() calls with surface size and write flags
- Tracked SetTexture() calls with texture sizes (every 1000th call logged)

**Results:**
```
Lock Tracking:
SURFACE: 0D84AAC0 size=128x128 lockCount=9 writeLocks=9

SetTexture (sampled ~170,000+ calls):
SETTEXTURE #1 stage=0 surface=1053F2D0 size=128x128
SETTEXTURE #1001 stage=0 surface=1053D690 size=128x128
... (all 128x128)
```

**MAJOR FINDING: ALL TEXTURES ARE 128x128!**

| Metric | Value |
|--------|-------|
| Total SetTexture calls | 170,000+ |
| Texture sizes | **100% are 128x128** |
| Large atlases (256x256+) | **NONE** |
| Locked surfaces | 1 surface (128x128), ~9 write locks/60 frames |

**Conclusion:**
- DK2 does NOT use large texture atlases
- All game textures are uniform 128x128 size
- The "texture atlas problem" described earlier may not exist
- RTX Remix should be able to hash these small, uniform textures directly

**Implications:**
1. The texture replacement issue may have a different cause
2. Need to investigate WHY texture hashes might be unstable (if they are)
3. The single locked surface (0D84AAC0) could be cursor/UI - investigate further

**Files Modified:** Same as Attempt 1 (repurposed tracking code)

---

### Clarification: Pre-Built vs Runtime Atlases (2026-01-22)

**User provided screenshot showing texture atlas in RTX Remix Developer Menu**

The screenshot shows multiple sprites (creature silhouettes, fire, UI elements) packed into a single 128x128 texture. This confirms:

1. **DK2 DOES use atlases** - but they are **PRE-BUILT** in the game's asset files
2. **Not runtime assembled** - the atlases come pre-packed, not dynamically built
3. **Our tracking didn't catch it** because there's no runtime Blt/Lock operations building atlases
4. **The 128x128 size is correct** - atlases are just 128x128 with multiple sprites packed inside

**The Real Problem (if any):**
- RTX Remix can hash these pre-built atlases (they're stable)
- BUT: replacing means replacing the ENTIRE atlas, not individual sprites
- If you want to replace just the "fire" sprite, you must replace the whole atlas texture

---

### Attempt 3: Texture Replacement Testing - SUCCESS! (2026-01-22)

**Goal:** Verify if RTX Remix texture replacement works at all

**Initial Failed Attempts:**

*Test 1: Simple file placement (FAILED)*
- Placed replacement texture at `rtx-remix/textures/3202B376AFBFA049.dds`
- Result: No change - RTX Remix doesn't scan folders for replacements

*Test 2: Tagged as Terrain (FAILED)*
- Tagged texture in RTX Remix UI as terrain
- Result: **Texture rendered as SOLID BLACK** - terrain baker couldn't process it

**Analysis of Material USD file (`captures/materials/mat_3202B376AFBFA049.usda`):**
```usd
def Material "mat_3202B376AFBFA049"
{
    def Shader "Shader"
    {
        asset inputs:diffuse_texture = @..\textures\3202B376AFBFA049.dds@
    }
}
```

**Key Discovery:** RTX Remix requires USD mod files to override material properties - simply placing DDS files doesn't work.

---

### SUCCESSFUL Texture Replacement Method (2026-01-22)

**Solution:** Create a proper USD mod structure

**Mod Structure Created:**
```
rtx-remix/mods/TextureTest/
├── mod.usda                        (material override)
└── textures/
    └── 3202B376AFBFA049.dds        (replacement texture)
```

**mod.usda Contents:**
```usda
#usda 1.0
(
    customLayerData = {
        string lightspeed_game_name = "Dungeon Keeper II"
        string lightspeed_layer_type = "replacement"
    }
    timeCodesPerSecond = 24
    upAxis = "Y"
)

over "RootNode"
{
    over "Looks"
    {
        over "mat_3202B376AFBFA049"
        {
            over "Shader"
            {
                asset inputs:diffuse_texture = @./textures/3202B376AFBFA049.dds@ (
                    colorSpace = "auto"
                )
            }
        }
    }
}
```

**Key Elements:**
1. `lightspeed_layer_type = "replacement"` - Tells Remix this is a replacement mod
2. `over "mat_3202B376AFBFA049"` - Overrides the specific material by hash
3. `over "Shader"` - Overrides the shader properties
4. `inputs:diffuse_texture = @./textures/...@` - Points to replacement texture (relative path)

**Result: SUCCESS!** Texture replacement confirmed working.

**What This Proves:**
1. RTX Remix texture replacement DOES work on DK2
2. The "atlas problem" is NOT a technical blocker - it's a content creation workflow issue
3. Pre-built atlases CAN be replaced - you just replace the entire atlas texture
4. Individual sprites within an atlas require editing the atlas image itself

**Files Created/Modified:**
- `rtx-remix/mods/TextureTest/mod.usda` - NEW: Material override USD file
- `rtx-remix/mods/TextureTest/textures/3202B376AFBFA049.dds` - Replacement texture
- `rtx.conf` - Removed texture from terrainTextures list (was causing black rendering)

**Additional Finding:** Tested with original unmodified dxwrapper - texture replacement still works. This confirms:
- The dxwrapper atlas tracking code (Attempts 1 & 2) was **NOT necessary** for texture replacement
- Texture replacement is purely an RTX Remix feature via USD mods
- The diagnostic logging code can be removed or disabled (`DdrawLogTextureAtlas = 0`)

---

### Creature Texture Replacement - SUCCESS! (2026-01-22)

**Test:** Replace Imp texture with wrong texture to verify creature textures can be replaced

**Hash:** `284668746F790EB3` (Imp texture)

**Result: SUCCESS!** Imps rendered with replacement texture, confirming:
- Creature/sprite textures work the same as environment textures
- No special handling needed for animated sprites
- Same USD mod pattern applies to all texture types

**Mod structure now contains:**
```
rtx-remix/mods/TextureTest/
├── mod.usda
└── textures/
    ├── 3202B376AFBFA049.dds    (environment test)
    └── 284668746F790EB3.dds    (imp/creature test)
```

---

## Next Steps

### Completed

1. ~~**Enable atlas detection logging**~~ ✓ Done - Found sprite animation, not runtime atlas building
2. ~~**Track Lock/Unlock + SetTexture**~~ ✓ Done - All textures are 128x128, pre-built atlases
3. ~~**Texture replacement test**~~ ✓ **SUCCESS!** - USD mod structure works

### Current Status

**Texture replacement is WORKING.** The workflow is:
1. Identify texture hash in RTX Remix UI
2. Create mod folder: `rtx-remix/mods/<ModName>/`
3. Create `mod.usda` with material override (see template above)
4. Place replacement DDS in `textures/` subfolder
5. Run game - replacement loads automatically

### Future Work

1. **Create replacement textures** - Edit actual game textures/atlases with improved art
2. **Material properties** - Adjust PBR properties (roughness, metallic, emission) via USD
3. **Lighting setup** - Add/modify lights for better path-traced visuals
4. **Mesh replacement** - Replace low-poly game meshes with higher quality versions
5. **Document all texture hashes** - Catalog which textures are used where in game

---

## Git Information

### Branch
```
reverse_xyzrhw_new
```

### Key Historical Commits (from original fork)

| Date | Commit | Description |
|------|--------|-------------|
| 2023-05-04 | `cc098a9a` | Added `DdrawConvertHomogeneousW` setting |
| 2023-05-04 | `33e1cec9` | Added geometry→world space transformation |
| 2023-05-05 | `0f1dfdb5` | Camera now accepted by RTX Remix |
| 2023-05-06 | `dad3f0d3` | "Hack for RTX Remix!!" |
| 2023-05-14 | `917d6d67` | Full homogeneous→world geometry support |

### Uncommitted Changes (as of 2026-01-19)
- `ddraw/RemixAPI.h` (new file)
- `d3d9/d3d9.cpp`
- `ddraw/IDirect3DDeviceX.cpp`
- `bin/Release/dx*/dxwrapper.ini`

---

## Summary

This project successfully achieved path tracing on a DirectX 6 game by:

1. **Identifying exact DX version** - Binary analysis confirmed DX6 (IDirectDraw4/IDirect3D3)
2. **Implementing inverse projection** - Converts screen-space XYZRHW vertices back to world space
3. **Integrating with RTX Remix** - Via D3D9 transform matrices (not direct API due to 32-bit bridge limitation)
4. **Fixing camera detection** - Key bug fix: don't restore transforms after drawing

**Texture Replacement:** WORKING as of 2026-01-22. Requires USD mod structure (not just DDS file placement). See "Successful Texture Replacement Method" section.

---

*Document updated: 2026-01-22*
*Texture replacement: CONFIRMED WORKING*
*For: dxwrapper RTX Remix project*

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

#include "ddraw.h"
#include "d3d9\d3d9External.h"
#include "RemixAPI.h"
#include <unordered_map>		// [GEOMPROBE]
#include <unordered_set>		// [GEOMPROBE]
#include <intrin.h>			// [DRAWCALLER] _AddressOfReturnAddress

// Stage 3 measurement (2026-05-27): defined in IDirectDrawSurfaceX.cpp. Called on
// every d3d9 SetTexture binding so we can aggregate per-hash bind counts and answer
// "which hashes does Remix actually request" with real data. See
// dk2_mission_measure_before_claiming memory.
extern void Stage3OnRemixBind(IDirect3DTexture9* d3d9Tex, bool isCrop);
// Dynamic-UI emissive (2026-06-11): per-draw classification accessors (set inside
// TryGetUniversalSubTextureForUV when the draw's placement resolves to a
// runtime-rendered UI name -- pointer strips / tooltips / minimap).
extern void NamekeyResetDrawDynUi();
// [BLITQUAD] menu-text fix (defined in IDirectDrawSurfaceX.cpp)
extern bool MenuBlitOverlayDrawActive();
extern void MenuBlitOverlayDrain(m_IDirect3DDeviceX* pDevice, LPDIRECT3DDEVICE9 dev);
extern IDirect3DTexture9* MenuBlitOverlayCurrentTex();
extern bool NamekeyGetDrawDynUi();
extern void NamekeyNoteDynUiDraw();
extern bool NamekeyGetDrawWater();
extern void NamekeyNoteWaterFlatten();

// Water surface fit (2026-06-11, build 8163). History: per-draw Y-average flatten
// (8161) -> seam lattice (every draw its own plane); single global level (8162) ->
// water vanished except where the level crossed the terrain, because run-13's
// WATERLVL data showed the inverse projection reconstructs the FLAT water plane as
// a gently bowed surface (true wave amplitude ~0.2-0.6 units, but water verts span
// 2-8+ units across one frame -- a 0.3-0.5% spatial distortion of world space).
// The flatten must FOLLOW that bow: each frame we least-squares-fit a quadratic
// surface y(x,z) = c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2 (u,v = centered,
// scaled x,z) to the frame's raw water verts, reject outliers (shore skirts, the
// occasional wild vert), refit, and snap NEXT frame's water verts to the surface
// evaluated at their own (x,z) -- one analytic surface, so no seams; it tracks the
// bow, so no vanishing; thousands of verts average the sine out, so it is
// temporally still. Verts farther than the snap band from the surface keep their
// height (shore skirts stay anchored). 5s without water draws resets everything
// (loading/map change). Draw path and EndScene share the DD critical section.
// STATELESS ERA (2026-06-12): the temporal estimators (EMA'd normal equations,
// zoom-step detector, sparse-frame adaptation -- see git history, builds 8164-8167)
// chased the camera-dependent reconstruction with visible rise/fall lag during
// cinematic and possession camera motion; the user directed abandoning estimation.
// Now the serving surface IS the previous frame's fit, verbatim: one frame of lag,
// camera-independent by construction. Water keeps the game's gentle swell (run-14
// finding: within one frame, verts lie on a smooth surface; the swell is that
// surface breathing over time) -- spatially smoothed, no chop, no seams, no chase.
// The true kill (game-side wave patch; "Sine Wave Water" string at exe 0x2B5DDC)
// is the prepared endgame if the residual swell bothers under translucency.
static bool   g_waterSurfValid = false;
static double g_waterSurfCoef[6] = {};					// SERVING surface (solved from the EMA'd normals)
static double g_waterSurfCx = 0.0, g_waterSurfCz = 0.0;	// centering point (fit epoch)
static double g_waterSurfScale = 1.0;					// coordinate scale -> u,v ~ O(1)
static bool   g_waterSurfEpochSet = false;
static float  g_waterSurfBand = 1.0f;					// snap only |y - S(x,z)| <= band
static DWORD  g_waterSurfAgeFrames = 0;					// frames since the serving fit was refreshed (stateless era: ages out fast)
static std::vector<float> g_waterFrameXYZ;				// raw x,y,z triplets this frame
static DWORD  g_waterFrameDraws = 0;
static DWORD  g_waterVertsTotal = 0;					// water verts seen this frame
static DWORD  g_waterVertsSnapped = 0;					// ...of which snapped (coverage telemetry)
static float  g_waterFrameYMin = 0.0f;					// raw height extremes this frame
static float  g_waterFrameYMax = 0.0f;
static DWORD  g_waterLastDrawMs = 0;
static DWORD  g_waterLvlPublishes = 0;
static const size_t kWaterFrameVertCap = 60000;			// 720 KB worst case, then sampling stops
static const float  kWaterBandFloor = 0.45f;			// band covers the residual swell around the live fit...
static const float  kWaterBandCap = 1.2f;				// ...but stays under shore-lip height (run-15 water-over-land)

// Evaluate a quadratic surface (given coefficients) at world (x, z).
static inline float WaterEvalCoef(const double coef[6], float x, float z)
{
	const double u = ((double)x - g_waterSurfCx) * g_waterSurfScale;
	const double v = ((double)z - g_waterSurfCz) * g_waterSurfScale;
	return (float)(coef[0] + coef[1] * u + coef[2] * v +
		coef[3] * u * u + coef[4] * u * v + coef[5] * v * v);
}

// Evaluate the CURRENT fitted surface at world (x, z).
static inline float WaterSurfEval(float x, float z)
{
	return WaterEvalCoef(g_waterSurfCoef, x, z);
}

// Solve the 6x6 normal equations (Gaussian elimination, partial pivoting).
static bool WaterSolve6(double A[6][6], double b[6], double out[6])
{
	int piv[6] = { 0, 1, 2, 3, 4, 5 };
	for (int col = 0; col < 6; col++)
	{
		int best = col;
		for (int r = col + 1; r < 6; r++)
			if (fabs(A[piv[r]][col]) > fabs(A[piv[best]][col])) best = r;
		int t = piv[col]; piv[col] = piv[best]; piv[best] = t;
		const double d = A[piv[col]][col];
		if (fabs(d) < 1e-12) return false;	// degenerate (e.g. all water collinear on screen)
		for (int r = col + 1; r < 6; r++)
		{
			const double f = A[piv[r]][col] / d;
			if (f == 0.0) continue;
			for (int c = col; c < 6; c++) A[piv[r]][c] -= f * A[piv[col]][c];
			b[piv[r]] -= f * b[piv[col]];
		}
	}
	for (int col = 5; col >= 0; col--)
	{
		double s = b[piv[col]];
		for (int c = col + 1; c < 6; c++) s -= A[piv[col]][c] * out[c];
		out[col] = s / A[piv[col]][col];
	}
	return true;
}

// Accumulate the per-vert-normalized 6x6 normal equations over the vert list.
// When baseline != nullptr, only verts within keepBand of the baseline surface
// contribute (outlier reject: shore skirts and stray geometry drop out).
// Returns the kept count (0 = nothing usable). A/b come back DIVIDED by kept so
// frames of different vert counts carry equal weight in the temporal EMA.
static size_t WaterAccumNormals(const std::vector<float>& xyz, const double* baseline, float keepBand, double A[6][6], double b[6])
{
	for (int r = 0; r < 6; r++) { b[r] = 0.0; for (int c = 0; c < 6; c++) A[r][c] = 0.0; }
	size_t kept = 0;
	const size_t n = xyz.size() / 3;
	for (size_t i = 0; i < n; i++)
	{
		const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
		if (baseline && fabsf(y - WaterEvalCoef(baseline, x, z)) > keepBand) continue;
		const double u = ((double)x - g_waterSurfCx) * g_waterSurfScale;
		const double v = ((double)z - g_waterSurfCz) * g_waterSurfScale;
		const double row[6] = { 1.0, u, v, u * u, u * v, v * v };
		for (int r = 0; r < 6; r++)
		{
			for (int c = r; c < 6; c++) A[r][c] += row[r] * row[c];
			b[r] += row[r] * (double)y;
		}
		kept++;
	}
	if (kept < 24) return 0;	// need comfortable headroom over 6 unknowns
	for (int r = 1; r < 6; r++)
		for (int c = 0; c < r; c++) A[r][c] = A[c][r];
	const double inv = 1.0 / (double)kept;
	for (int r = 0; r < 6; r++) { b[r] *= inv; for (int c = 0; c < 6; c++) A[r][c] *= inv; }
	return kept;
}

// RMS residual of the (baseline-filtered) verts against the given surface.
// meanOut (optional) receives the SIGNED mean residual -- the zoom-step signal:
// a zoom change shifts the whole reconstruction, showing up as a sustained
// nonzero mean while the wave averages near zero.
static double WaterRmsVsCoef(const std::vector<float>& xyz, const double* baseline, float keepBand, const double coef[6], double* meanOut = nullptr)
{
	double se = 0.0, sr = 0.0; size_t cnt = 0;
	const size_t n = xyz.size() / 3;
	for (size_t i = 0; i < n; i++)
	{
		const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
		if (baseline && fabsf(y - WaterEvalCoef(baseline, x, z)) > keepBand) continue;
		const double r = (double)y - (double)WaterEvalCoef(coef, x, z);
		se += r * r; sr += r; cnt++;
	}
	if (meanOut) *meanOut = cnt ? (sr / (double)cnt) : 0.0;
	return cnt ? sqrt(se / (double)cnt) : 0.0;
}

// Force ONE draw to additive blending so Remix treats the surface as emissive
// (the torch-flame mechanism) -- the only way to make hash-unstable dynamic UI
// self-lit. Applied AFTER SetDrawStates (so it wins), restored right after the
// draw. Single-threaded draw path; static save-slots are safe (same pattern as
// ToWorld_IntermediateGeometry).
static bool sDynUiOverrideActive = false;
static DWORD sDynUiOldAB = 0, sDynUiOldSrc = 0, sDynUiOldDst = 0;
// Orphan-overlay lift (2026-06-12, menu-text fix): per-frame near-plane depth for
// lifted overlay draws (constant-RHW + recipeless page = runtime-composited text).
// Each lifted draw steps slightly nearer so painter's order (text over panels)
// survives the inverse projection. Reset every BeginScene.
static float g_orphanLiftZ = 0.05f;

static void DynUiBeginDraw(LPDIRECT3DDEVICE9 dev)
{
	dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &sDynUiOldAB);
	dev->GetRenderState(D3DRS_SRCBLEND, &sDynUiOldSrc);
	dev->GetRenderState(D3DRS_DESTBLEND, &sDynUiOldDst);
	dev->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
	dev->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
	dev->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ONE);
	sDynUiOverrideActive = true;
	NamekeyNoteDynUiDraw();
}
static void DynUiEndDraw(LPDIRECT3DDEVICE9 dev)
{
	if (!sDynUiOverrideActive) return;
	dev->SetRenderState(D3DRS_ALPHABLENDENABLE, sDynUiOldAB);
	dev->SetRenderState(D3DRS_SRCBLEND, sDynUiOldSrc);
	dev->SetRenderState(D3DRS_DESTBLEND, sDynUiOldDst);
	sDynUiOverrideActive = false;
}

// Canonical Identity Layer (2026-06-09): defined in IDirectDrawSurfaceX.cpp.
// Resolves a whole-surface texture's CONTENT to a frozen canonical texture
// (loaded from _canonical\tex\<HASH>.a4r4) so Remix sees one stable,
// offline-precomputed hash per content regardless of DK2's per-session
// atlas-composition drift. Returns nullptr when no rebind applies.
extern IDirect3DTexture9* Stage3CanonicalResolve(IDirect3DTexture9* d3d9Tex, LPDIRECT3DDEVICE9* d3d9Device);

// ******************************
// IUnknown functions
// ******************************

HRESULT m_IDirect3DDeviceX::QueryInterface(REFIID riid, LPVOID FAR * ppvObj, DWORD DirectXVersion)
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
		return D3D_OK;
	}
	if (riid == IID_GetInterfaceX)
	{
		*ppvObj = this;
		return D3D_OK;
	}

	DWORD DxVersion = (Config.Dd7to9 && CheckWrapperType(riid)) ? GetGUIDVersion(riid) : DirectXVersion;

	if (riid == GetWrapperType(DxVersion) || riid == IID_IUnknown)
	{
		*ppvObj = GetWrapperInterfaceX(DxVersion);

		AddRef(DxVersion);

		return D3D_OK;
	}

	return ProxyQueryInterface(ProxyInterface, riid, ppvObj, GetWrapperType(DirectXVersion));
}

ULONG m_IDirect3DDeviceX::AddRef(DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") v" << DirectXVersion;

	if (Config.Dd7to9)
	{
		// Some Direct3DDevices share reference count with parent surfaces
		if (parent3DSurface.Interface)
		{
			return parent3DSurface.Interface->AddRef(parent3DSurface.DxVersion);
		}

		switch (DirectXVersion)
		{
		case 1:
			return InterlockedIncrement(&RefCount1);
		case 2:
			return InterlockedIncrement(&RefCount2);
		case 3:
			return InterlockedIncrement(&RefCount3);
		case 7:
			return InterlockedIncrement(&RefCount7);
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
			return 0;
		}
	}

	return ProxyInterface->AddRef();
}

ULONG m_IDirect3DDeviceX::Release(DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") v" << DirectXVersion;

	if (Config.Dd7to9)
	{
		ULONG ref;

		// Some Direct3DDevices share reference count with parent surfaces
		if (parent3DSurface.Interface)
		{
			return parent3DSurface.Interface->Release(parent3DSurface.DxVersion);
		}

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
		case 7:
			ref = (InterlockedCompareExchange(&RefCount7, 0, 0)) ? InterlockedDecrement(&RefCount7) : 0;
			break;
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
			ref = 0;
		}

		if (InterlockedCompareExchange(&RefCount1, 0, 0) + InterlockedCompareExchange(&RefCount2, 0, 0) +
			InterlockedCompareExchange(&RefCount3, 0, 0) + InterlockedCompareExchange(&RefCount7, 0, 0) == 0)
		{
			delete this;
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
// IDirect3DDevice v1 functions
// ******************************

HRESULT m_IDirect3DDeviceX::Initialize(LPDIRECT3D lpd3d, LPGUID lpGUID, LPD3DDEVICEDESC lpd3ddvdesc)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Returns D3D_OK if successful, otherwise it returns an error.
		return D3D_OK;
	}

	if (lpd3d)
	{
		lpd3d->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpd3d);
	}

	return GetProxyInterfaceV1()->Initialize(lpd3d, lpGUID, lpd3ddvdesc);
}

HRESULT m_IDirect3DDeviceX::GetCaps(LPD3DDEVICEDESC lpD3DHWDevDesc, LPD3DDEVICEDESC lpD3DHELDevDesc)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if ((!lpD3DHWDevDesc && !lpD3DHELDevDesc) ||
			(lpD3DHWDevDesc && lpD3DHWDevDesc->dwSize != D3DDEVICEDESC1_SIZE && lpD3DHWDevDesc->dwSize != D3DDEVICEDESC5_SIZE && lpD3DHWDevDesc->dwSize != D3DDEVICEDESC6_SIZE) ||
			(lpD3DHELDevDesc && lpD3DHELDevDesc->dwSize != D3DDEVICEDESC1_SIZE && lpD3DHELDevDesc->dwSize != D3DDEVICEDESC5_SIZE && lpD3DHELDevDesc->dwSize != D3DDEVICEDESC6_SIZE))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Incorrect dwSize: " << ((lpD3DHWDevDesc) ? lpD3DHWDevDesc->dwSize : -1) << " " << ((lpD3DHELDevDesc) ? lpD3DHELDevDesc->dwSize : -1));
			return DDERR_INVALIDPARAMS;
		}

		D3DDEVICEDESC7 D3DDevDesc;
		HRESULT hr = GetCaps(&D3DDevDesc);

		if (SUCCEEDED(hr))
		{
			if (lpD3DHWDevDesc)
			{
				ConvertDeviceDesc(*lpD3DHWDevDesc, D3DDevDesc);
			}

			if (lpD3DHELDevDesc)
			{
				ConvertDeviceDesc(*lpD3DHELDevDesc, D3DDevDesc);
			}
		}

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->GetCaps(lpD3DHWDevDesc, lpD3DHELDevDesc);
	case 2:
		return GetProxyInterfaceV2()->GetCaps(lpD3DHWDevDesc, lpD3DHELDevDesc);
	case 3:
		return GetProxyInterfaceV3()->GetCaps(lpD3DHWDevDesc, lpD3DHELDevDesc);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::SwapTextureHandles(LPDIRECT3DTEXTURE2 lpD3DTex1, LPDIRECT3DTEXTURE2 lpD3DTex2)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DTex1 || !lpD3DTex2)
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirect3DTextureX* pTextureX1 = nullptr;
		lpD3DTex1->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pTextureX1);

		m_IDirect3DTextureX* pTextureX2 = nullptr;
		lpD3DTex2->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pTextureX2);

		if (!pTextureX1 || !pTextureX2)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture wrapper!");
			return DDERR_INVALIDPARAMS;
		}

		// Find handle associated with texture1
		D3DTEXTUREHANDLE TexHandle1 = 0;
		if (FAILED(pTextureX1->GetHandle((LPDIRECT3DDEVICE2)GetWrapperInterfaceX(0), &TexHandle1)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find texture1 handle!");
			return DDERR_INVALIDPARAMS;
		}

		// Find handle associated with texture2
		D3DTEXTUREHANDLE TexHandle2 = 0;
		if (FAILED(pTextureX2->GetHandle((LPDIRECT3DDEVICE2)GetWrapperInterfaceX(0), &TexHandle2)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not find texture2 handle!");
			return DDERR_INVALIDPARAMS;
		}

		// Swap texture handle1
		pTextureX1->SetHandle(TexHandle2);
		TextureHandleMap[TexHandle2] = pTextureX1;

		// Swap texture handle2
		pTextureX2->SetHandle(TexHandle1);
		TextureHandleMap[TexHandle1] = pTextureX2;

		// If texture handle is set then use new texture
		if (TexHandle1 == DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREHANDLE].State ||
			TexHandle2 == DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREHANDLE].State)
		{
			SetTextureHandle(DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREHANDLE].State);
		}

		return D3D_OK;
	}

	if (lpD3DTex1)
	{
		lpD3DTex1->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpD3DTex1);
	}
	if (lpD3DTex2)
	{
		lpD3DTex2->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpD3DTex2);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->SwapTextureHandles((LPDIRECT3DTEXTURE)lpD3DTex1, (LPDIRECT3DTEXTURE)lpD3DTex2);
	case 2:
		return GetProxyInterfaceV2()->SwapTextureHandles(lpD3DTex1, lpD3DTex2);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::CreateExecuteBuffer(LPD3DEXECUTEBUFFERDESC lpDesc, LPDIRECT3DEXECUTEBUFFER * lplpDirect3DExecuteBuffer, IUnknown * pUnkOuter)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDirect3DExecuteBuffer || !lpDesc)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpDirect3DExecuteBuffer = nullptr;

		if (pUnkOuter)
		{
			LOG_LIMIT(3, __FUNCTION__ << " Warning: 'pUnkOuter' is not null: " << pUnkOuter);
		}

		if (lpDesc->dwSize != sizeof(D3DEXECUTEBUFFERDESC))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Incorrect dwSize: " << lpDesc->dwSize);
			return DDERR_INVALIDPARAMS;
		}

		// Validate dwFlags
		if (!(lpDesc->dwFlags & D3DDEB_BUFSIZE))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: D3DDEB_BUFSIZE flag not set.");
			return DDERR_INVALIDPARAMS;
		}

		// Validate dwBufferSize
		if (lpDesc->dwBufferSize == 0 || lpDesc->dwBufferSize > MAX_EXECUTE_BUFFER_SIZE)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Invalid dwBufferSize: " << lpDesc->dwBufferSize);
			return DDERR_INVALIDPARAMS;
		}

		// Validate dwCaps
		if ((lpDesc->dwFlags & D3DDEB_CAPS) && (lpDesc->dwCaps & D3DDEBCAPS_SYSTEMMEMORY) && (lpDesc->dwCaps & D3DDEBCAPS_VIDEOMEMORY))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Unsupported dwCaps: " << Logging::hex(lpDesc->dwCaps));
			return DDERR_INVALIDPARAMS;
		}

		// Validate lpData
		if ((lpDesc->dwFlags & D3DDEB_LPDATA) && lpDesc->lpData)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: lpData is non-null, using application data.");
		}

		m_IDirect3DExecuteBuffer* pExecuteBuffer = m_IDirect3DExecuteBuffer::CreateDirect3DExecuteBuffer(nullptr, this, lpDesc);

		*lplpDirect3DExecuteBuffer = pExecuteBuffer;

		return D3D_OK;
	}

	HRESULT hr = GetProxyInterfaceV1()->CreateExecuteBuffer(lpDesc, lplpDirect3DExecuteBuffer, pUnkOuter);

	if (SUCCEEDED(hr) && lplpDirect3DExecuteBuffer)
	{
		*lplpDirect3DExecuteBuffer = m_IDirect3DExecuteBuffer::CreateDirect3DExecuteBuffer(*lplpDirect3DExecuteBuffer, nullptr, nullptr);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::GetStats(LPD3DSTATS lpD3DStats)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// The method returns E_NOTIMPL / DDERR_UNSUPPORTED.
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->GetStats(lpD3DStats);
	case 2:
		return GetProxyInterfaceV2()->GetStats(lpD3DStats);
	case 3:
		return GetProxyInterfaceV3()->GetStats(lpD3DStats);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::Execute(LPDIRECT3DEXECUTEBUFFER lpDirect3DExecuteBuffer, LPDIRECT3DVIEWPORT lpDirect3DViewport, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDirect3DExecuteBuffer || !lpDirect3DViewport)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid params: " << lpDirect3DExecuteBuffer << " " << lpDirect3DViewport);
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		// Flags
		// D3DEXECUTE_CLIPPED - Clip any primitives in the buffer that are outside or partially outside the viewport. 
		// D3DEXECUTE_UNCLIPPED - All primitives in the buffer are contained within the viewport.

		m_IDirect3DExecuteBuffer* pExecuteBuffer = (m_IDirect3DExecuteBuffer*)lpDirect3DExecuteBuffer;

		LPVOID lpData;
		D3DEXECUTEDATA ExecuteData;
		LPD3DSTATUS lpStatus;

		// Check lock
		bool IsLocked = false;
		ScopedFlagSet(pExecuteBuffer->CheckLockStatus(IsLocked));
		if (IsLocked)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: execute buffer still locked!");
		}

		// Get execute data and desc
		if (FAILED(pExecuteBuffer->GetBuffer(&lpData, ExecuteData, &lpStatus)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: get execute data failed!");
			return DDERR_INVALIDPARAMS;
		}

		if (FAILED(SetCurrentViewport((LPDIRECT3DVIEWPORT3)lpDirect3DViewport)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to set the specified viewport!");
			return DDERR_INVALIDPARAMS;
		}

		// Pointer to the start of the instruction data
		BYTE* instructionData = reinterpret_cast<BYTE*>(lpData) + ExecuteData.dwInstructionOffset;
		BYTE* instructionEnd = instructionData + ExecuteData.dwInstructionLength;

		DWORD opcode = NULL;

		// ToDo: figure out which vertex type is being used D3DFVF_VERTEX, D3DFVF_LVERTEX or D3DFVF_TLVERTEX
		DWORD VertexTypeDesc = D3DFVF_TLVERTEX;

		// Primitive structures and related defines. Vertex offsets are to types D3DVERTEX, D3DLVERTEX, or D3DTLVERTEX.
		BYTE* vertexBuffer = reinterpret_cast<BYTE*>(lpData) + ExecuteData.dwVertexOffset;
		const DWORD vertexCount = ExecuteData.dwVertexCount;

		// Iterate through the instructions
		while (instructionData < instructionEnd)
		{
			const D3DINSTRUCTION* instruction = (const D3DINSTRUCTION*)(instructionData);
			const DWORD instructionSize = sizeof(D3DINSTRUCTION) + (instruction->wCount * instruction->bSize);

			opcode = instruction->bOpcode;
			BYTE* opstruct = instructionData + sizeof(D3DINSTRUCTION);

			if (opcode == D3DOP_EXIT || instructionData + instructionSize > instructionEnd)
			{
				break;
			}

			bool SkipNextMove = false;

			switch (opcode)
			{
			case D3DOP_POINT:
				// Sends a point to the renderer. Operand data is described by the D3DPOINT structure.
			{
				D3DPOINT* point = reinterpret_cast<D3DPOINT*>(opstruct);

				if (instruction->bSize != sizeof(D3DPOINT))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: point instruction size does not match!");
				}

				DrawExecutePoint(point, instruction->wCount, vertexCount, vertexBuffer, VertexTypeDesc);

				break;
			}
			case D3DOP_SPAN:
				// Spans a list of points with the same y value. For more information, see the D3DSPAN structure.
			{
				D3DSPAN* span = reinterpret_cast<D3DSPAN*>(opstruct);

				if (instruction->bSize != sizeof(D3DSPAN))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: span instruction size does not match!");
				}

				DrawExecuteSpan(span, instruction->wCount, vertexCount, vertexBuffer, VertexTypeDesc);

				break;
			}
			case D3DOP_LINE:
				// Sends a line to the renderer. Operand data is described by the D3DLINE structure.
			{
				D3DLINE* line = reinterpret_cast<D3DLINE*>(opstruct);

				if (instruction->bSize != sizeof(D3DLINE))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: line instruction size does not match!");
				}

				DrawExecuteLine(line, instruction->wCount, vertexCount, vertexBuffer, VertexTypeDesc);

				break;
			}
			case D3DOP_TRIANGLE:
				// Sends a triangle to the renderer. Operand data is described by the D3DTRIANGLE structure.
			{
				D3DTRIANGLE* triangle = reinterpret_cast<D3DTRIANGLE*>(opstruct);

				if (instruction->bSize != sizeof(D3DTRIANGLE))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: triangle instruction size does not match!");
				}

				DrawExecuteTriangle(triangle, instruction->wCount, vertexCount, vertexBuffer, VertexTypeDesc);

				break;
			}
			case D3DOP_MATRIXLOAD:
				// Triggers a data transfer in the rendering engine. Operand data is described by the D3DMATRIXLOAD structure.
			{
				D3DMATRIXLOAD* matrixLoad = reinterpret_cast<D3DMATRIXLOAD*>(opstruct);

				if (instruction->bSize != sizeof(D3DMATRIXLOAD))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: matrix load instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					// Copy matrix to dest
					D3DMATRIX* pSrcMatrix = GetMatrix(matrixLoad[i].hSrcMatrix);
					D3DMATRIX* pDestMatrix = GetMatrix(matrixLoad[i].hDestMatrix);
					if (pSrcMatrix && pDestMatrix)
					{
						*pDestMatrix = *pSrcMatrix;
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to find matrix handle for load!");
					}
				}

				break;
			}
			case D3DOP_MATRIXMULTIPLY:
				// Triggers a data transfer in the rendering engine. Operand data is described by the D3DMATRIXMULTIPLY structure.
			{
				D3DMATRIXMULTIPLY* matrixMultiply = reinterpret_cast<D3DMATRIXMULTIPLY*>(opstruct);

				if (instruction->bSize != sizeof(D3DMATRIXMULTIPLY))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: matrix multiply instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					// Multiply matrix to dest
					D3DMATRIX* pSrcMatrix1 = GetMatrix(matrixMultiply[i].hSrcMatrix1);
					D3DMATRIX* pSrcMatrix2 = GetMatrix(matrixMultiply[i].hSrcMatrix2);
					D3DMATRIX* pDestMatrix = GetMatrix(matrixMultiply[i].hDestMatrix);
					if (pSrcMatrix1 && pSrcMatrix2 && pDestMatrix)
					{
						using namespace DirectX;

						// Load D3DMATRIX into XMMATRIX
						XMMATRIX xmMatrix1 = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(pSrcMatrix1));
						XMMATRIX xmMatrix2 = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(pSrcMatrix2));

						// Perform the multiplication
						XMMATRIX xmResult = XMMatrixMultiply(xmMatrix1, xmMatrix2);

						// Store the result back into a D3DMATRIX
						XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(pDestMatrix), xmResult);
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to find matrix handle for multiply!");
					}
				}

				break;
			}
			case D3DOP_STATETRANSFORM:
				// Sets the value of internal state variables in the rendering engine for the transformation module. Operand data is a variable token
				// and the new value. The token identifies the internal state variable, and the new value is the value to which that variable should
				// be set. For more information about these variables, see the D3DSTATE structure and the D3DLIGHTSTATETYPE enumerated type.
			{
				D3DSTATE* state = reinterpret_cast<D3DSTATE*>(opstruct);

				if (instruction->bSize != sizeof(D3DSTATE))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: state transform instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					D3DMATRIX* pMatrix = GetMatrix(state[i].dwArg[0]);
					if (pMatrix)
					{
						SetTransform(state[i].dtstTransformStateType, pMatrix);
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to find matrix handle for transform!");
					}
				}

				break;
			}
			case D3DOP_STATELIGHT:
				// Sets the value of internal state variables in the rendering engine for the lighting module. Operand data is a variable token
				// and the new value. The token identifies the internal state variable, and the new value is the value to which that variable
				// should be set. For more information about these variables, see the D3DSTATE structure and the D3DLIGHTSTATETYPE enumerated type.
			{
				D3DSTATE* state = reinterpret_cast<D3DSTATE*>(opstruct);

				if (instruction->bSize != sizeof(D3DSTATE))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: state light instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					SetLightState(state[i].dlstLightStateType, state[i].dwArg[0]);
				}

				break;
			}
			case D3DOP_STATERENDER:
				// Sets the value of internal state variables in the rendering engine for the rendering module. Operand data is a variable token
				// and the new value. The token identifies the internal state variable, and the new value is the value to which that variable
				// should be set. For more information about these variables, see the D3DSTATE structure and the D3DLIGHTSTATETYPE enumerated type.
			{
				D3DSTATE* state = reinterpret_cast<D3DSTATE*>(opstruct);

				if (instruction->bSize != sizeof(D3DSTATE))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: state render instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					SetRenderState(state[i].drstRenderStateType, state[i].dwArg[0]);
				}

				break;
			}
			case D3DOP_TEXTURELOAD:
				// Triggers a data transfer in the rendering engine. Operand data is described by the D3DTEXTURELOAD structure.
			{
				D3DTEXTURELOAD* textureLoad = reinterpret_cast<D3DTEXTURELOAD*>(opstruct);

				if (instruction->bSize != sizeof(D3DTEXTURELOAD))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: texture load instruction size does not match!");
				}

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					// Copy texture to dest
					m_IDirect3DTextureX* lpTextureSrcX = GetTexture(textureLoad[i].hSrcTexture);
					m_IDirect3DTextureX* lpTextureDestX = GetTexture(textureLoad[i].hDestTexture);
					if (lpTextureSrcX && lpTextureDestX)
					{
						LPDIRECT3DTEXTURE2 lpTextureSrc = (LPDIRECT3DTEXTURE2)lpTextureSrcX->GetWrapperInterfaceX(0);
						lpTextureDestX->Load(lpTextureSrc);
					}
					else
					{
						LOG_LIMIT(100, __FUNCTION__ << " Error: failed to find texture handle!");
					}
				}

				break;
			}
			case D3DOP_PROCESSVERTICES:
				// Sets both lighting and transformations for vertices. Operand data is described by the D3DPROCESSVERTICES structure.
			{
				D3DPROCESSVERTICES* processVertices = reinterpret_cast<D3DPROCESSVERTICES*>(opstruct);

				if (instruction->bSize != sizeof(D3DPROCESSVERTICES))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: process vertices instruction size does not match!");
				}

#ifdef ENABLE_PROFILING
				auto startTime = std::chrono::high_resolution_clock::now();
#endif

				bool IsHVertexUsed = false;

				HRESULT hr = D3D_OK;

				BYTE* inputVerts = reinterpret_cast<BYTE*>(lpData) + ExecuteData.dwVertexOffset;
				BYTE* outputVerts = reinterpret_cast<BYTE*>(lpData) + ExecuteData.dwHVertexOffset;

				for (DWORD i = 0; i < instruction->wCount; i++)
				{
					DWORD Flags = processVertices[i].dwFlags;

					if ((Flags & (D3DPROCESSVERTICES_COPY | D3DPROCESSVERTICES_TRANSFORM | D3DPROCESSVERTICES_TRANSFORMLIGHT)) == 0)
					{
						Flags |= D3DPROCESSVERTICES_TRANSFORM;
						LOG_LIMIT(100, __FUNCTION__ << " Warning: ProcessVertices dwFlags=0, defaulting to TRANSFORM");
					}

					if (processVertices[i].wStart >= vertexCount || processVertices[i].wDest >= vertexCount)
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: index exceeds vertices count.  Skip processing!");
						continue;
					}
					// Compute maximum safe count based on buffer size
					DWORD maxSrc = vertexCount - processVertices[i].wStart;
					DWORD maxDest = vertexCount - processVertices[i].wDest;
					DWORD Count = min(processVertices[i].dwCount, min(maxSrc, maxDest));

					// Check Count
					if (Count == 0)
					{
						LOG_LIMIT(100, __FUNCTION__ << " Warning: zero vertices to process. Skip processing!");
						continue;
					}

					D3DLVERTEX* srcVertices = reinterpret_cast<D3DLVERTEX*>(inputVerts) + processVertices[i].wStart;
					D3DTLVERTEX* destVertices = reinterpret_cast<D3DTLVERTEX*>(outputVerts) + processVertices[i].wDest;

					// Copy vertices only
					if (Flags & D3DPROCESSVERTICES_COPY)
					{
						IsHVertexUsed = true;
						for (UINT x = 0; x < Count; x++)
						{
							destVertices[x] = *(D3DTLVERTEX*)&srcVertices[x];
						}
					}
					// Apply transform & lighting
					else if (Flags & (D3DPROCESSVERTICES_TRANSFORM | D3DPROCESSVERTICES_TRANSFORMLIGHT))
					{
						IsHVertexUsed = true;
						bool IsLight = (Flags & D3DPROCESSVERTICES_TRANSFORMLIGHT) && !(Flags & D3DPROCESSVERTICES_NOCOLOR);
						bool UpdateExtents = (Flags & D3DPROCESSVERTICES_UPDATEEXTENTS);
						hr = m_IDirect3DVertexBufferX::TransformVertexUP(this, srcVertices, destVertices, nullptr, Count, lpStatus->drExtent, IsLight, UpdateExtents);

						if (SUCCEEDED(hr))
						{
							lpStatus->dwFlags |= D3DSETSTATUS_STATUS;
							lpStatus->dwStatus = 0; // Just set no clip flags and no ZNOTVISIBLE

							if (UpdateExtents)
							{
								lpStatus->dwFlags |= D3DSETSTATUS_EXTENTS;
							}
						}
					}
				}

#ifdef ENABLE_PROFILING
				Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

				// Update vertex buffer to use output
				if (IsHVertexUsed)
				{
					vertexBuffer = reinterpret_cast<BYTE*>(lpData) + ExecuteData.dwHVertexOffset;
				}

				break;
			}
			case D3DOP_SETSTATUS:
				// Resets the status of the execute buffer. For more information, see the D3DSTATUS structure.
			{
				D3DSTATUS* status = reinterpret_cast<D3DSTATUS*>(opstruct);

				if (instruction->bSize != sizeof(D3DSTATUS))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: status instruction size does not match!");
				}

				if (instruction->wCount > 1)
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: more than 1 count in set status instruction!");
				}

				// Update only the requested fields
				if (status->dwFlags & D3DSETSTATUS_STATUS)
				{
					lpStatus->dwStatus = status->dwStatus;
				}

				if (status->dwFlags & D3DSETSTATUS_EXTENTS)
				{
					lpStatus->drExtent = status->drExtent;
				}

				// Always update the flags field to reflect what was set
				lpStatus->dwFlags = status->dwFlags;

				break;
			}
			case D3DOP_BRANCHFORWARD:
				// Enables a branching mechanism within the execute buffer. For more information, see the D3DBRANCH structure.
			{
				// Parse the branch structure
				D3DBRANCH* branch = reinterpret_cast<D3DBRANCH*>(opstruct);

				if (instruction->bSize != sizeof(D3DBRANCH))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: branch instruction size does not match!");
				}

				if (instruction->wCount > 1)
				{
					LOG_LIMIT(100, __FUNCTION__ << " Warning: more than 1 count in branch forward instruction!");
				}

				// Apply the mask to the current status
				DWORD maskedStatus = lpStatus->dwStatus & branch->dwMask;

				// Compare the masked status with the value
				bool condition = (maskedStatus == branch->dwValue);

				// Negate the condition if bNegate is TRUE
				if (branch->bNegate)
				{
					condition = !condition;
				}

				// If the condition is true, branch forward
				if (condition)
				{
					SkipNextMove = true;
					if (branch->dwOffset == 0)
					{
						// Exit the execute buffer if offset is 0
						opcode = D3DOP_EXIT;
						break;
					}
					else
					{
						// Move the instruction pointer forward by the offset
						instructionData += branch->dwOffset;
					}
				}

				// Otherwise, continue to the next instruction
				break;
			}
			case D3DOP_EXIT:
				// Signals that the end of the list has been reached.
				break;
			default:
				// Handle unknown or unsupported opcodes
				LOG_LIMIT(100, __FUNCTION__ << " Warning: Unknown opcode: " << opcode);
				break;
			}

			// Exit loop
			if (opcode == D3DOP_EXIT)
			{
				break;
			}

			// Move to the next instruction
			if (!SkipNextMove)
			{
				instructionData += instructionSize;
			}
		}

		return D3D_OK;
	}

	if (lpDirect3DExecuteBuffer)
	{
		lpDirect3DExecuteBuffer->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DExecuteBuffer);
	}
	if (lpDirect3DViewport)
	{
		lpDirect3DViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DViewport);
	}

	return GetProxyInterfaceV1()->Execute(lpDirect3DExecuteBuffer, lpDirect3DViewport, dwFlags);
}

HRESULT m_IDirect3DDeviceX::AddViewport(LPDIRECT3DVIEWPORT3 lpDirect3DViewport)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// This method will fail, returning DDERR_INVALIDPARAMS, if you attempt to add a viewport that has already been assigned to the device.
		if (!lpDirect3DViewport || IsViewportAttached(lpDirect3DViewport))
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirect3DViewportX* lpViewportX = nullptr;
		if (FAILED(lpDirect3DViewport->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpViewportX)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get ViewportX interface!");
			return DDERR_GENERIC;
		}

		// Associate device with the viewport
		lpViewportX->AddD3DDevice(this);

		AttachedViewports.push_back(lpDirect3DViewport);

		lpDirect3DViewport->AddRef();

		// The first version of the device doesn't have SetCurrentViewport()
		if (ClientDirectXVersion == 1)
		{
			if (AttachedViewports.size() == 1)
			{
				SetCurrentViewport(lpDirect3DViewport);
			}
		}

		return D3D_OK;
	}

	if (lpDirect3DViewport)
	{
		lpDirect3DViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DViewport);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->AddViewport(lpDirect3DViewport);
	case 2:
		return GetProxyInterfaceV2()->AddViewport(lpDirect3DViewport);
	case 3:
		return GetProxyInterfaceV3()->AddViewport(lpDirect3DViewport);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::DeleteViewport(LPDIRECT3DVIEWPORT3 lpDirect3DViewport)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDirect3DViewport)
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirect3DViewportX* lpViewportX = nullptr;
		if (FAILED(lpDirect3DViewport->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpViewportX)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get ViewportX interface!");
			return DDERR_GENERIC;
		}

		lpViewportX->ClearD3DDevice(this);

		bool ret = DeleteAttachedViewport(lpDirect3DViewport);

		if (!ret)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (lpDirect3DViewport == lpCurrentViewport)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: deleting active viewport!");

			// Backup old viewport
			m_IDirect3DViewportX* lpOldViewportX = lpCurrentViewportX;

			// Set to null first
			lpCurrentViewport = nullptr;
			lpCurrentViewportX = nullptr;

			// Clear old viewport data
			if (lpOldViewportX)
			{
				lpOldViewportX->ClearCurrentViewport(this, true);
			}
		}

		lpDirect3DViewport->Release();

		return D3D_OK;
	}

	if (lpDirect3DViewport)
	{
		lpDirect3DViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DViewport);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->DeleteViewport(lpDirect3DViewport);
	case 2:
		return GetProxyInterfaceV2()->DeleteViewport(lpDirect3DViewport);
	case 3:
		return GetProxyInterfaceV3()->DeleteViewport(lpDirect3DViewport);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::NextViewport(LPDIRECT3DVIEWPORT3 lpDirect3DViewport, LPDIRECT3DVIEWPORT3* lplpDirect3DViewport, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpDirect3DViewport || (dwFlags == D3DNEXT_NEXT && !lpDirect3DViewport))
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpDirect3DViewport = nullptr;

		if (AttachedViewports.size() == 0)
		{
			return D3DERR_NOVIEWPORTS;
		}

		switch (dwFlags)
		{
		case D3DNEXT_HEAD:
			// Retrieve the item at the beginning of the list.
			*lplpDirect3DViewport = AttachedViewports.front();
			break;
		case D3DNEXT_TAIL:
			// Retrieve the item at the end of the list.
			*lplpDirect3DViewport = AttachedViewports.back();
			break;
		case D3DNEXT_NEXT:
			// Retrieve the next item in the list.
			// If you attempt to retrieve the next viewport in the list when you are at the end of the list, this method returns D3D_OK but lplpAnotherViewport is NULL.
			for (UINT x = 1; x < AttachedViewports.size(); x++)
			{
				if (AttachedViewports[x - 1] == lpDirect3DViewport)
				{
					*lplpDirect3DViewport = AttachedViewports[x];
					break;
				}
			}
			break;
		default:
			return DDERR_INVALIDPARAMS;
			break;
		}

		// The first version of the device doesn't have SetCurrentViewport()
		if (DirectXVersion == 1)
		{
			if (*lplpDirect3DViewport)
			{
				SetCurrentViewport(*lplpDirect3DViewport);
			}
		}

		return D3D_OK;
	}

	if (lpDirect3DViewport)
	{
		lpDirect3DViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DViewport);
	}

	HRESULT hr = DDERR_GENERIC;

	switch (ProxyDirectXVersion)
	{
	case 1:
		hr = GetProxyInterfaceV1()->NextViewport(lpDirect3DViewport, (LPDIRECT3DVIEWPORT*)lplpDirect3DViewport, dwFlags);
		break;
	case 2:
		hr = GetProxyInterfaceV2()->NextViewport(lpDirect3DViewport, (LPDIRECT3DVIEWPORT2*)lplpDirect3DViewport, dwFlags);
		break;
	case 3:
		hr = GetProxyInterfaceV3()->NextViewport(lpDirect3DViewport, lplpDirect3DViewport, dwFlags);
		break;
	}

	if (SUCCEEDED(hr) && lplpDirect3DViewport)
	{
		*lplpDirect3DViewport = ProxyAddressLookupTable.FindAddress<m_IDirect3DViewport3>(*lplpDirect3DViewport, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::Pick(LPDIRECT3DEXECUTEBUFFER lpDirect3DExecuteBuffer, LPDIRECT3DVIEWPORT lpDirect3DViewport, DWORD dwFlags, LPD3DRECT lpRect)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	if (lpDirect3DExecuteBuffer)
	{
		lpDirect3DExecuteBuffer->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DExecuteBuffer);
	}
	if (lpDirect3DViewport)
	{
		lpDirect3DViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDirect3DViewport);
	}

	return GetProxyInterfaceV1()->Pick(lpDirect3DExecuteBuffer, lpDirect3DViewport, dwFlags, lpRect);
}

HRESULT m_IDirect3DDeviceX::GetPickRecords(LPDWORD lpCount, LPD3DPICKRECORD lpD3DPickRec)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	return GetProxyInterfaceV1()->GetPickRecords(lpCount, lpD3DPickRec);
}

HRESULT m_IDirect3DDeviceX::EnumTextureFormats(LPD3DENUMTEXTUREFORMATSCALLBACK lpd3dEnumTextureProc, LPVOID lpArg)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpd3dEnumTextureProc)
		{
			return DDERR_INVALIDPARAMS;
		}

		struct EnumPixelFormat
		{
			LPVOID lpContext;
			LPD3DENUMTEXTUREFORMATSCALLBACK lpCallback;

			static HRESULT CALLBACK ConvertCallback(LPDDPIXELFORMAT lpDDPixFmt, LPVOID lpContext)
			{
				EnumPixelFormat* self = (EnumPixelFormat*)lpContext;

				// Only RGB formats are supported
				if ((lpDDPixFmt->dwFlags & DDPF_RGB) == NULL)
				{
					return DDENUMRET_OK;
				}

				DDSURFACEDESC Desc = {};
				Desc.dwSize = sizeof(DDSURFACEDESC);
				Desc.dwFlags = DDSD_CAPS | DDSD_PIXELFORMAT;
				Desc.ddpfPixelFormat = *lpDDPixFmt;
				Desc.ddsCaps.dwCaps = DDSCAPS_TEXTURE;

				return self->lpCallback(&Desc, self->lpContext);
			}
		} CallbackContext = {};
		CallbackContext.lpContext = lpArg;
		CallbackContext.lpCallback = lpd3dEnumTextureProc;

		return EnumTextureFormats(EnumPixelFormat::ConvertCallback, &CallbackContext);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->EnumTextureFormats(lpd3dEnumTextureProc, lpArg);
	case 2:
		return GetProxyInterfaceV2()->EnumTextureFormats(lpd3dEnumTextureProc, lpArg);
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::CreateMatrix(LPD3DMATRIXHANDLE lpD3DMatHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DMatHandle)
		{
			return DDERR_INVALIDPARAMS;
		}

		D3DMATRIX Matrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};

		D3DMATRIXHANDLE D3DMatHandle = ComputeRND((DWORD)&Matrix, (DWORD)lpD3DMatHandle);

		// Make sure the material handle is unique
		while (D3DMatHandle == NULL || GetMatrix(D3DMatHandle))
		{
			D3DMatHandle += 4;
		}

		MatrixMap[D3DMatHandle] = { true, Matrix };

		*lpD3DMatHandle = D3DMatHandle;

		return D3D_OK;
	}

	return GetProxyInterfaceV1()->CreateMatrix(lpD3DMatHandle);
}

HRESULT m_IDirect3DDeviceX::SetMatrix(D3DMATRIXHANDLE D3DMatHandle, const LPD3DMATRIX lpD3DMatrix)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DMatrix || !GetMatrix(D3DMatHandle))
		{
			return DDERR_INVALIDPARAMS;
		}

		MatrixMap[D3DMatHandle] = { true, *lpD3DMatrix };

		return D3D_OK;
	}

	return GetProxyInterfaceV1()->SetMatrix(D3DMatHandle, lpD3DMatrix);
}

HRESULT m_IDirect3DDeviceX::GetMatrix(D3DMATRIXHANDLE D3DMatHandle, LPD3DMATRIX lpD3DMatrix)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DMatrix || !GetMatrix(D3DMatHandle))
		{
			return DDERR_INVALIDPARAMS;
		}

		*lpD3DMatrix = MatrixMap[D3DMatHandle].m;

		return D3D_OK;
	}

	return GetProxyInterfaceV1()->GetMatrix(D3DMatHandle, lpD3DMatrix);
}

HRESULT m_IDirect3DDeviceX::DeleteMatrix(D3DMATRIXHANDLE D3DMatHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!GetMatrix(D3DMatHandle))
		{
			return DDERR_INVALIDPARAMS;
		}

		MatrixMap.erase(D3DMatHandle);

		return D3D_OK;
	}

	return GetProxyInterfaceV1()->DeleteMatrix(D3DMatHandle);
}

HRESULT m_IDirect3DDeviceX::BeginScene()
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	// New frame: reset the orphan-overlay painter's-order depth.
	g_orphanLiftZ = 0.05f;

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		// Set 3D Enabled
		ddrawParent->Enable3D();

		HRESULT hr = (*d3d9Device)->BeginScene();

		if (SUCCEEDED(hr))
		{
			IsInScene = true;

#ifdef ENABLE_PROFILING
			Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr;
			sceneTime = std::chrono::high_resolution_clock::now();
#endif

			PrepDevice();

			// Set 3D camera matrices at START of scene for RTX Remix camera detection
			// Must be done BEFORE draw calls, as camera is detected during draw call processing
			if (Config.DdrawConvertHomogeneousToWorld)
			{
				if (!ConvertHomogeneous.IsTransformViewSet)
				{
					// Game never called SetTransform(D3DTS_VIEW) - compute default camera matrices
					D3DVIEWPORT9 vp;
					(*d3d9Device)->GetViewport(&vp);
					const float width = (float)vp.Width;
					const float height = (float)vp.Height;
					const float fov = Config.DdrawConvertHomogeneousToWorldFOV;
					const float nearplane = Config.DdrawConvertHomogeneousToWorldNearPlane;
					const float farplane = Config.DdrawConvertHomogeneousToWorldFarPlane;
					const float ratio = width / height;

					// Create projection matrix (perspective, NOT orthographic)
					DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(
						fov * (3.14159265359f / 180.0f), ratio, nearplane, farplane);
					DirectX::XMStoreFloat4x4((DirectX::XMFLOAT4X4*)&ConvertHomogeneous.ToWorld_ProjectionMatrix, proj);

					// Create view matrix - isometric camera looking down at 45 degrees
					DirectX::XMVECTOR position = DirectX::XMVectorSet(0.0f, 800.0f, -800.0f, 0.0f);
					DirectX::XMVECTOR direction = DirectX::XMVector3Normalize(
						DirectX::XMVectorSet(0.0f, -0.707f, 0.707f, 0.0f));
					DirectX::XMVECTOR upVector = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
					DirectX::XMMATRIX viewMatrix = DirectX::XMMatrixLookToLH(position, direction, upVector);
					DirectX::XMStoreFloat4x4((DirectX::XMFLOAT4X4*)&ConvertHomogeneous.ToWorld_ViewMatrix, viewMatrix);

					// Create screen-to-NDC matrix (transforms screen coords 0-width,0-height to NDC -1 to 1)
					// This is the same matrix used in SetTransform for XYZRHW handling
					DirectX::XMMATRIX screenToNDC = DirectX::XMMatrixSet(
						2.0f / width, 0.0f, 0.0f, 0.0f,
						0.0f, -2.0f / height, 0.0f, 0.0f,
						0.0f, 0.0f, 1.0f, 0.0f,
						-1.0f, 1.0f, 0.0f, 1.0f
					);

					// Compute the inverse matrix for vertex transformation
					// Full transform: screen -> NDC -> clip -> view -> world
					DirectX::XMMATRIX vp_matrix = DirectX::XMMatrixMultiply(viewMatrix, proj);
					DirectX::XMMATRIX vpinv = DirectX::XMMatrixInverse(nullptr, vp_matrix);
					ConvertHomogeneous.ToWorld_ViewMatrixInverse = DirectX::XMMatrixMultiply(screenToNDC, vpinv);

					ConvertHomogeneous.IsTransformViewSet = true;

					static bool loggedDefault = false;
					if (!loggedDefault)
					{
						Logging::Log() << __FUNCTION__ << " Initialized camera matrices in BeginScene (width=" << width << " height=" << height << ")";
						loggedDefault = true;
					}
				}

				// Set matrices via D3D9 BEFORE any draw calls
				(*d3d9Device)->SetTransform(D3DTS_VIEW, &ConvertHomogeneous.ToWorld_ViewMatrix);
				(*d3d9Device)->SetTransform(D3DTS_PROJECTION, &ConvertHomogeneous.ToWorld_ProjectionMatrix);
			}
		}

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->BeginScene();
	case 2:
		return GetProxyInterfaceV2()->BeginScene();
	case 3:
		return GetProxyInterfaceV3()->BeginScene();
	case 7:
		return GetProxyInterfaceV7()->BeginScene();
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::EndScene()
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		// The IDirect3DDevice7::EndScene method ends a scene that was begun by calling the IDirect3DDevice7::BeginScene method.
		// When this method succeeds, the scene has been rendered, and the device surface holds the rendered scene.

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		// [BLITQUAD] menu-text fix: re-emit this frame's queued backbuffer blits
		// (front-end UI/text composites) as lifted self-lit quads inside the scene.
		if (Config.DdrawMenuBlitOverlay && d3d9Device && *d3d9Device)
		{
			MenuBlitOverlayDrain(this, *d3d9Device);
		}

		// Note: Camera matrices are now set in BeginScene (before draw calls)
		// The EndScene code below is kept for Remix API (may not work on 32-bit through bridge)
		if (Config.DdrawConvertHomogeneousToWorld)
		{
			// Try Remix API if available
			RemixAPIManager& remixApi = RemixAPIManager::Instance();
			if (!remixApi.IsInitialized())
			{
				const char* initError = nullptr;
				if (remixApi.Initialize(&initError))
				{
					Logging::Log() << __FUNCTION__ << " RTX Remix API initialized successfully";
				}
			}
		}

		HRESULT hr = (*d3d9Device)->EndScene();

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		if (SUCCEEDED(hr))
		{
			IsInScene = false;

#ifdef ENABLE_PROFILING
			Logging::Log() << __FUNCTION__ << " (" << this << ") Full Scene Time = " << Logging::GetTimeLapseInMS(sceneTime);
#endif

			if (lpCurrentRenderTargetX)
			{
				lpCurrentRenderTargetX->EndWritePresent(nullptr, false);
			}

			// Log atlas tracking at end of frame
			m_IDirectDrawSurfaceX::LogAtlasTrackingAndReset();

			// Water surface update (run-14 temporal fix -- see the block comment at
			// the top of this file). Per frame: live fit (pass 1) -> reject skirts
			// against it -> fold the kept verts' normalized normal equations into
			// the temporal EMA -> solve the EMA for the SERVING surface. The wave
			// (which the live fit tracks) averages out across phases in the EMA;
			// a sustained live-vs-serve gap (real scene change) reseeds instead.
			{
				const size_t frameVerts = g_waterFrameXYZ.size() / 3;
				if (frameVerts >= 48)
				{
					if (!g_waterSurfEpochSet)
					{
						// Fix the fit's coordinate frame on this epoch's water centroid and
						// spread, so u,v stay O(1) (4th-order moments need the conditioning).
						double sx = 0.0, sz = 0.0;
						for (size_t i = 0; i < frameVerts; i++) { sx += g_waterFrameXYZ[i * 3]; sz += g_waterFrameXYZ[i * 3 + 2]; }
						g_waterSurfCx = sx / (double)frameVerts;
						g_waterSurfCz = sz / (double)frameVerts;
						double sd = 0.0;
						for (size_t i = 0; i < frameVerts; i++)
						{
							const double dx = g_waterFrameXYZ[i * 3] - g_waterSurfCx;
							const double dz = g_waterFrameXYZ[i * 3 + 2] - g_waterSurfCz;
							sd += dx * dx + dz * dz;
						}
						const double spread = sqrt(sd / (double)frameVerts);
						g_waterSurfScale = (spread > 1e-3) ? (1.0 / spread) : 1.0;
						g_waterSurfEpochSet = true;
					}
					double Af[6][6], bf[6], As[6][6], bs[6], pass1[6], live[6], serve[6];
					bool haveLive = false;
					size_t kept = 0;
					if (WaterAccumNormals(g_waterFrameXYZ, nullptr, 0.0f, Af, bf))
					{
						memcpy(As, Af, sizeof(As)); memcpy(bs, bf, sizeof(bs));
						if (WaterSolve6(As, bs, pass1))
						{
							const double rms1 = WaterRmsVsCoef(g_waterFrameXYZ, nullptr, 0.0f, pass1);
							float rejBand = (float)(rms1 * 2.5);
							if (rejBand < 0.25f) rejBand = 0.25f;
							if (rejBand > 3.0f) rejBand = 3.0f;
							kept = WaterAccumNormals(g_waterFrameXYZ, pass1, rejBand, Af, bf);
							if (kept >= frameVerts / 4)
							{
								memcpy(As, Af, sizeof(As)); memcpy(bs, bf, sizeof(bs));
								haveLive = WaterSolve6(As, bs, live);
								if (haveLive)
								{
									// STATELESS ERA (2026-06-12, user directive "abandon that method"):
									// the temporal EMA/step/sparse estimators chased the camera-
									// dependent reconstruction with visible lag during cinematic and
									// possession camera motion (water rose/fell for seconds). Serve
									// the LIVE fit directly -- one frame of lag, ~16ms, camera-
									// independent by construction. Trade-off accepted: water keeps
									// the game's gentle swell, spatially smoothed (no chop, no seams).
									memcpy(serve, live, sizeof(serve));
									{
										const double rmsServe = WaterRmsVsCoef(g_waterFrameXYZ, pass1, rejBand, serve);
										for (int i = 0; i < 6; i++) g_waterSurfCoef[i] = serve[i];
										g_waterSurfAgeFrames = 0;
										float band = (float)(rmsServe * 4.0);
										if (band < kWaterBandFloor) band = kWaterBandFloor;
										if (band > kWaterBandCap) band = kWaterBandCap;
										g_waterSurfBand = band;
										g_waterSurfValid = true;
										g_waterLvlPublishes++;
										if (g_waterLvlPublishes <= 20 || (g_waterLvlPublishes % 600) == 0)
										{
											const DWORD snapPct = g_waterVertsTotal ? (g_waterVertsSnapped * 100 / g_waterVertsTotal) : 0;
											char wbuf[300];
											sprintf_s(wbuf, sizeof(wbuf), "[NAMEKEY] WATERSURF n=%lu c0=%.4f liveC0=%.4f cx=%.4f cz=%.4f cxx=%.5f cxz=%.5f czz=%.5f rmsServe=%.4f band=%.3f kept=%zu/%zu snap=%lu%% ymin=%.3f ymax=%.3f draws=%lu",
												(unsigned long)g_waterLvlPublishes, g_waterSurfCoef[0], live[0], g_waterSurfCoef[1], g_waterSurfCoef[2],
												g_waterSurfCoef[3], g_waterSurfCoef[4], g_waterSurfCoef[5],
												rmsServe, band, kept, frameVerts, (unsigned long)snapPct,
												g_waterFrameYMin, g_waterFrameYMax, (unsigned long)g_waterFrameDraws);
											Logging::Log() << wbuf;
										}
									}
								}
							}
						}
					}
				}
				// frameVerts in (0, 48): sparse water (extreme close zoom / possession
				// near a puddle). STATELESS ERA: no adaptation -- a fresh surface may
				// serve briefly, but sustained sparseness AGES IT OUT so draws fall
				// back to their own per-draw averages (always locally correct, no
				// camera-chase possible).
				else if (frameVerts > 0)
				{
					if (g_waterSurfValid && ++g_waterSurfAgeFrames > 10)
					{
						g_waterSurfValid = false;	// stale fit + moving camera = the chase artifact; let per-draw averages take over
					}
					double sy = 0.0;
					for (size_t i = 0; i < frameVerts; i++) sy += g_waterFrameXYZ[i * 3 + 1];
					static DWORD waterSparseSeen = 0;
					waterSparseSeen++;
					if (waterSparseSeen <= 40 || (waterSparseSeen % 300) == 0)
					{
						char sbuf[240];
						sprintf_s(sbuf, sizeof(sbuf), "[NAMEKEY] WATERSPARSE n=%lu verts=%zu draws=%lu snapped=%lu yAvg=%.3f ymin=%.3f ymax=%.3f age=%lu surfValid=%d",
							(unsigned long)waterSparseSeen, frameVerts, (unsigned long)g_waterFrameDraws,
							(unsigned long)g_waterVertsSnapped, (float)(sy / (double)frameVerts),
							g_waterFrameYMin, g_waterFrameYMax,
							(unsigned long)g_waterSurfAgeFrames, g_waterSurfValid ? 1 : 0);
						Logging::Log() << sbuf;
					}
				}
				g_waterFrameXYZ.clear();
				g_waterFrameDraws = 0;
				g_waterVertsTotal = 0;
				g_waterVertsSnapped = 0;
			}
		}

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
		return GetProxyInterfaceV1()->EndScene();
	case 2:
		return GetProxyInterfaceV2()->EndScene();
	case 3:
		return GetProxyInterfaceV3()->EndScene();
	case 7:
		return GetProxyInterfaceV7()->EndScene();
	default:
		return DDERR_GENERIC;
	}
}

HRESULT m_IDirect3DDeviceX::GetDirect3D(LPDIRECT3D7* lplpD3D, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpD3D)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpD3D = nullptr;

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, false)))
		{
			return DDERR_INVALIDOBJECT;
		}

		m_IDirect3DX** lplpD3DX = ddrawParent->GetCurrentD3D();

		if (!(*lplpD3DX))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: missing Direct3D wrapper!");
			return DDERR_GENERIC;
		}

		*lplpD3D = (LPDIRECT3D7)(*lplpD3DX)->GetWrapperInterfaceX(DirectXVersion);

		if (!(*lplpD3D))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get Direct3D interface!");
			return DDERR_GENERIC;
		}

		(*lplpD3D)->AddRef();

		return D3D_OK;
	}

	HRESULT hr = DDERR_GENERIC;

	switch (ProxyDirectXVersion)
	{
	case 1:
		hr = GetProxyInterfaceV1()->GetDirect3D((LPDIRECT3D*)lplpD3D);
		break;
	case 2:
		hr = GetProxyInterfaceV2()->GetDirect3D((LPDIRECT3D2*)lplpD3D);
		break;
	case 3:
		hr = GetProxyInterfaceV3()->GetDirect3D((LPDIRECT3D3*)lplpD3D);
		break;
	case 7:
		hr = GetProxyInterfaceV7()->GetDirect3D(lplpD3D);
		break;
	}

	if (SUCCEEDED(hr) && lplpD3D)
	{
		*lplpD3D = ProxyAddressLookupTable.FindAddress<m_IDirect3D7>(*lplpD3D, DirectXVersion);
	}

	return hr;
}

// ******************************
// IDirect3DDevice v2 functions
// ******************************

HRESULT m_IDirect3DDeviceX::SetCurrentViewport(LPDIRECT3DVIEWPORT3 lpd3dViewport)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Before calling this method, applications must have already called the AddViewport method to add the viewport to the device.
		if (!lpd3dViewport || !IsViewportAttached(lpd3dViewport))
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirect3DViewportX* lpViewportX = nullptr;
		if (FAILED(lpd3dViewport->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpViewportX)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get ViewportX interface!");
			return DDERR_GENERIC;
		}

		D3DVIEWPORT Viewport = {};
		Viewport.dwSize = sizeof(D3DVIEWPORT);

		HRESULT hr = lpd3dViewport->GetViewport(&Viewport);

		if (SUCCEEDED(hr))
		{
			D3DVIEWPORT7 Viewport7;

			ConvertViewport(Viewport7, Viewport);

			hr = SetViewport(&Viewport7);

			if (SUCCEEDED(hr))
			{
				// Backup old viewport
				m_IDirect3DViewportX* lpOldViewportX = lpCurrentViewportX;

				// Set new viewport first
				lpCurrentViewport = lpd3dViewport;
				lpCurrentViewportX = lpViewportX;

				// Clear old viewport data and apply new viewport data
				if (lpOldViewportX != lpCurrentViewportX)
				{
					if (lpOldViewportX)
					{
						lpOldViewportX->ClearCurrentViewport(this, false);
					}

					lpCurrentViewportX->SetCurrentViewportActive(false, true, true);
				}
			}
		}

		return hr;
	}

	if (lpd3dViewport)
	{
		lpd3dViewport->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpd3dViewport);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetCurrentViewport(lpd3dViewport);
	case 3:
		return GetProxyInterfaceV3()->SetCurrentViewport(lpd3dViewport);
	}
}

HRESULT m_IDirect3DDeviceX::GetCurrentViewport(LPDIRECT3DVIEWPORT3* lplpd3dViewport, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpd3dViewport)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (!lpCurrentViewport)
		{
			return D3DERR_NOCURRENTVIEWPORT;
		}

		*lplpd3dViewport = lpCurrentViewport;

		lpCurrentViewport->AddRef();

		return D3D_OK;
	}

	HRESULT hr = DDERR_GENERIC;

	switch (ProxyDirectXVersion)
	{
	case 2:
		hr = GetProxyInterfaceV2()->GetCurrentViewport((LPDIRECT3DVIEWPORT2*)lplpd3dViewport);
		break;
	case 3:
		hr = GetProxyInterfaceV3()->GetCurrentViewport(lplpd3dViewport);
		break;
	}

	if (SUCCEEDED(hr) && lplpd3dViewport)
	{
		*lplpd3dViewport = ProxyAddressLookupTable.FindAddress<m_IDirect3DViewport3>(*lplpd3dViewport, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::SetRenderTarget(LPDIRECTDRAWSURFACE7 lpNewRenderTarget, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpNewRenderTarget)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Don't reset existing render target
		if (CurrentRenderTarget == lpNewRenderTarget)
		{
			return D3D_OK;
		}

		// dwFlags: Not currently used; set to 0.

		// ToDo: if DirectXVersion < 7 then invalidate the current material and viewport:
		// Unlike this method's implementation in previous interfaces, IDirect3DDevice7::SetRenderTarget does not invalidate the current material or viewport for the device.

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = nullptr;
		lpNewRenderTarget->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDSrcSurfaceX);

		if (!lpDDSrcSurfaceX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface wrapper!");
			return DDERR_INVALIDPARAMS;
		}

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		HRESULT hr = ddrawParent->SetRenderTargetSurface(lpDDSrcSurfaceX);

		if (SUCCEEDED(hr))
		{
			if (CurrentRenderTarget)
			{
				CurrentRenderTarget->Release();
			}

			CurrentRenderTarget = lpNewRenderTarget;

			CurrentRenderTarget->AddRef();

			lpCurrentRenderTargetX = lpDDSrcSurfaceX;

			RenderTargetMultiSampleType = lpDDSrcSurfaceX->GetMultiSampleType();

			DWORD OldDepthBits = DepthBitCount;

			DepthBitCount = lpDDSrcSurfaceX->GetAttachedStencilSurfaceZBits();

			if (OldDepthBits != DepthBitCount)
			{
				SetD9RenderState(D3DRS_DEPTHBIAS, GetDepthBias(DeviceStates.RenderState[D3DRENDERSTATE_ZBIAS].State, DepthBitCount));
			}
		}

		return D3D_OK;
	}

	if (lpNewRenderTarget)
	{
		lpNewRenderTarget->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpNewRenderTarget);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetRenderTarget((LPDIRECTDRAWSURFACE)lpNewRenderTarget, dwFlags);
	case 3:
		return GetProxyInterfaceV3()->SetRenderTarget((LPDIRECTDRAWSURFACE4)lpNewRenderTarget, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->SetRenderTarget(lpNewRenderTarget, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::GetRenderTarget(LPDIRECTDRAWSURFACE7* lplpRenderTarget, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpRenderTarget)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (!CurrentRenderTarget)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: render target not set.");
			return DDERR_GENERIC;
		}

		*lplpRenderTarget = CurrentRenderTarget;

		CurrentRenderTarget->AddRef();

		return D3D_OK;
	}

	HRESULT hr = DDERR_GENERIC;

	switch (ProxyDirectXVersion)
	{
	case 2:
		hr = GetProxyInterfaceV2()->GetRenderTarget((LPDIRECTDRAWSURFACE*)lplpRenderTarget);
		break;
	case 3:
		hr = GetProxyInterfaceV3()->GetRenderTarget((LPDIRECTDRAWSURFACE4*)lplpRenderTarget);
		break;
	case 7:
		hr = GetProxyInterfaceV7()->GetRenderTarget(lplpRenderTarget);
		break;
	}

	if (SUCCEEDED(hr) && lplpRenderTarget)
	{
		*lplpRenderTarget = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(*lplpRenderTarget, DirectXVersion);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::Begin(D3DPRIMITIVETYPE d3dpt, DWORD d3dvt, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->Begin(d3dpt, (D3DVERTEXTYPE)d3dvt, dwFlags);
	case 3:
		return GetProxyInterfaceV3()->Begin(d3dpt, d3dvt, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::BeginIndexed(D3DPRIMITIVETYPE dptPrimitiveType, DWORD dvtVertexType, LPVOID lpvVertices, DWORD dwNumVertices, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->BeginIndexed(dptPrimitiveType, (D3DVERTEXTYPE)dvtVertexType, lpvVertices, dwNumVertices, dwFlags);
	case 3:
		return GetProxyInterfaceV3()->BeginIndexed(dptPrimitiveType, dvtVertexType, lpvVertices, dwNumVertices, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::Vertex(LPVOID lpVertexType)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->Vertex(lpVertexType);
	case 3:
		return GetProxyInterfaceV3()->Vertex(lpVertexType);
	}
}

HRESULT m_IDirect3DDeviceX::Index(WORD wVertexIndex)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->Index(wVertexIndex);
	case 3:
		return GetProxyInterfaceV3()->Index(wVertexIndex);
	}
}

HRESULT m_IDirect3DDeviceX::End(DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Not Implemented");
		return DDERR_UNSUPPORTED;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->End(dwFlags);
	case 3:
		return GetProxyInterfaceV3()->End(dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::GetRenderState(D3DRENDERSTATETYPE dwRenderStateType, LPDWORD lpdwRenderState)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") " << dwRenderStateType;

	if (Config.Dd7to9)
	{
		if (!lpdwRenderState)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			*lpdwRenderState = 0;
			return DDERR_INVALIDOBJECT;
		}

		if (!IsValidRenderState(dwRenderStateType, ClientDirectXVersion))
		{
			if (dwRenderStateType > 0 && dwRenderStateType < D3D_MAXRENDERSTATES && ClientDirectXVersion < 7)
			{
				*lpdwRenderState = DeviceStates.RenderState[dwRenderStateType].State;
				return D3D_OK;
			}
			else
			{
				*lpdwRenderState = 0;
				return DDERR_INVALIDPARAMS;
			}
		}

		if (dwRenderStateType > 95 && (dwRenderStateType < 128 || IsOutOfRangeRenderState(dwRenderStateType, ClientDirectXVersion)))
		{
			if (dwRenderStateType < D3D_MAXRENDERSTATES)
			{
				switch (dwRenderStateType)
				{
				case D3DRS_WRAP0:					// 128
					*lpdwRenderState = DeviceStates.rsMap128;
					break;
				case D3DRS_AMBIENT:					// 139
					*lpdwRenderState = DeviceStates.rsMap139;
					break;
				case D3DRS_FOGVERTEXMODE:			// 140
					*lpdwRenderState = DeviceStates.rsMap140;
					break;
				case D3DRS_COLORVERTEX:				// 141
					*lpdwRenderState = DeviceStates.rsMap141;
					break;
				case D3DRS_MULTISAMPLEANTIALIAS:	// 161
					*lpdwRenderState = DeviceStates.rsMap161;
					break;
				case D3DRS_MULTISAMPLEMASK:			// 162
					*lpdwRenderState = DeviceStates.rsMap162;
					break;
				case D3DRS_DEPTHBIAS:				// 195
					*lpdwRenderState = DeviceStates.rsMap195;
					break;
				default:
					*lpdwRenderState = DeviceStates.RenderState[dwRenderStateType].State;
					break;
				}
				return D3D_OK;
			}
			*lpdwRenderState = (DWORD)-1;
			return DDERR_INVALIDPARAMS;
		}

		switch ((DWORD)dwRenderStateType)
		{
		case D3DRENDERSTATE_TEXTUREADDRESS:		// 3
		{
			DWORD ValueU = 0, ValueV = 0;
			GetD9SamplerState(0, D3DSAMP_ADDRESSU, &ValueU);
			GetD9SamplerState(0, D3DSAMP_ADDRESSV, &ValueV);
			if (ValueU != ValueV)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: Texture Address U and V states don't match: " << ValueU << " and " << ValueV);
			}
			*lpdwRenderState = ValueU;
			return D3D_OK;
		}
		case D3DRENDERSTATE_WRAPU:				// 5
			GetD9RenderState(D3DRS_WRAP0, lpdwRenderState);
			*lpdwRenderState &= D3DWRAP_U;
			return D3D_OK;
		case D3DRENDERSTATE_WRAPV:				// 6
			GetD9RenderState(D3DRS_WRAP0, lpdwRenderState);
			*lpdwRenderState = (bool)(*lpdwRenderState & D3DWRAP_V);
			return D3D_OK;
		case D3DRENDERSTATE_OLDALPHABLENDENABLE:// 42
			if (DeviceStates.RenderState[D3DRENDERSTATE_OLDALPHABLENDENABLE].State == (DWORD)-1)
			{
				*lpdwRenderState = (DWORD)-1;
				return D3D_OK;
			}
			return GetD9RenderState(D3DRENDERSTATE_BLENDENABLE, lpdwRenderState);
		case D3DRENDERSTATE_BORDERCOLOR:		// 43
			return GetD9SamplerState(0, D3DSAMP_BORDERCOLOR, lpdwRenderState);
		case D3DRENDERSTATE_TEXTUREADDRESSU:	// 44
			return GetD9SamplerState(0, D3DSAMP_ADDRESSU, lpdwRenderState);
		case D3DRENDERSTATE_TEXTUREADDRESSV:	// 45
			return GetD9SamplerState(0, D3DSAMP_ADDRESSV, lpdwRenderState);
		case D3DRENDERSTATE_MIPMAPLODBIAS:		// 46
			return GetD9SamplerState(0, D3DSAMP_MIPMAPLODBIAS, lpdwRenderState);
		case D3DRENDERSTATE_ANISOTROPY:			// 49
			return GetD9SamplerState(0, D3DSAMP_MAXANISOTROPY, lpdwRenderState);
		case D3DRENDERSTATE_NONE:				// 0
		case D3DRENDERSTATE_TEXTUREHANDLE:		// 1
		case D3DRENDERSTATE_ANTIALIAS:			// 2
		case D3DRENDERSTATE_TEXTUREPERSPECTIVE:	// 4
		case D3DRENDERSTATE_LINEPATTERN:		// 10
		case D3DRENDERSTATE_MONOENABLE:			// 11
		case D3DRENDERSTATE_ROP2:				// 12
		case D3DRENDERSTATE_PLANEMASK:			// 13
		case D3DRENDERSTATE_TEXTUREMAG:			// 17
		case D3DRENDERSTATE_TEXTUREMIN:			// 18
		case D3DRENDERSTATE_TEXTUREMAPBLEND:	// 21
		case D3DRENDERSTATE_ZVISIBLE:			// 30
		case D3DRENDERSTATE_SUBPIXEL:			// 31
		case D3DRENDERSTATE_SUBPIXELX:			// 32
		case D3DRENDERSTATE_STIPPLEDALPHA:		// 33
		case D3DRENDERSTATE_STIPPLEENABLE:		// 39
		case D3DRENDERSTATE_EDGEANTIALIAS:		// 40
		case D3DRENDERSTATE_COLORKEYENABLE:		// 41
		case D3DRENDERSTATE_ZBIAS:				// 47
		case D3DRENDERSTATE_FLUSHBATCH:			// 50
		case D3DRENDERSTATE_TRANSLUCENTSORTINDEPENDENT:	// 51
		case 61:
		case 62:
		case 63:
		case D3DRENDERSTATE_STIPPLEPATTERN00:	// 64
		case D3DRENDERSTATE_STIPPLEPATTERN01:	// 65
		case D3DRENDERSTATE_STIPPLEPATTERN02:	// 66
		case D3DRENDERSTATE_STIPPLEPATTERN03:	// 67
		case D3DRENDERSTATE_STIPPLEPATTERN04:	// 68
		case D3DRENDERSTATE_STIPPLEPATTERN05:	// 69
		case D3DRENDERSTATE_STIPPLEPATTERN06:	// 70
		case D3DRENDERSTATE_STIPPLEPATTERN07:	// 71
		case D3DRENDERSTATE_STIPPLEPATTERN08:	// 72
		case D3DRENDERSTATE_STIPPLEPATTERN09:	// 73
		case D3DRENDERSTATE_STIPPLEPATTERN10:	// 74
		case D3DRENDERSTATE_STIPPLEPATTERN11:	// 75
		case D3DRENDERSTATE_STIPPLEPATTERN12:	// 76
		case D3DRENDERSTATE_STIPPLEPATTERN13:	// 77
		case D3DRENDERSTATE_STIPPLEPATTERN14:	// 78
		case D3DRENDERSTATE_STIPPLEPATTERN15:	// 79
		case D3DRENDERSTATE_STIPPLEPATTERN16:	// 80
		case D3DRENDERSTATE_STIPPLEPATTERN17:	// 81
		case D3DRENDERSTATE_STIPPLEPATTERN18:	// 82
		case D3DRENDERSTATE_STIPPLEPATTERN19:	// 83
		case D3DRENDERSTATE_STIPPLEPATTERN20:	// 84
		case D3DRENDERSTATE_STIPPLEPATTERN21:	// 85
		case D3DRENDERSTATE_STIPPLEPATTERN22:	// 86
		case D3DRENDERSTATE_STIPPLEPATTERN23:	// 87
		case D3DRENDERSTATE_STIPPLEPATTERN24:	// 88
		case D3DRENDERSTATE_STIPPLEPATTERN25:	// 89
		case D3DRENDERSTATE_STIPPLEPATTERN26:	// 90
		case D3DRENDERSTATE_STIPPLEPATTERN27:	// 91
		case D3DRENDERSTATE_STIPPLEPATTERN28:	// 92
		case D3DRENDERSTATE_STIPPLEPATTERN29:	// 93
		case D3DRENDERSTATE_STIPPLEPATTERN30:	// 94
		case D3DRENDERSTATE_STIPPLEPATTERN31:	// 95
		case D3DRENDERSTATE_EXTENTS:			// 138
		case D3DRENDERSTATE_COLORKEYBLENDENABLE:// 144
			*lpdwRenderState = DeviceStates.RenderState[dwRenderStateType].State;
			return D3D_OK;
		}

		return GetD9RenderState(dwRenderStateType, lpdwRenderState);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->GetRenderState(dwRenderStateType, lpdwRenderState);
	case 3:
		return GetProxyInterfaceV3()->GetRenderState(dwRenderStateType, lpdwRenderState);
	case 7:
		return GetProxyInterfaceV7()->GetRenderState(dwRenderStateType, lpdwRenderState);
	}
}

HRESULT m_IDirect3DDeviceX::SetRenderState(D3DRENDERSTATETYPE dwRenderStateType, DWORD dwRenderState)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") " << dwRenderStateType << " " << dwRenderState;

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		if (!IsValidRenderState(dwRenderStateType, ClientDirectXVersion))
		{
			return DDERR_INVALIDPARAMS;
		}

		if (dwRenderStateType > 95 && (dwRenderStateType < 128 || IsOutOfRangeRenderState(dwRenderStateType, ClientDirectXVersion)))
		{
			if (dwRenderStateType < D3D_MAXRENDERSTATES)
			{
				switch (dwRenderStateType)
				{
				case D3DRS_WRAP0:					// 128
					DeviceStates.rsMap128 = dwRenderState;
					break;
				case D3DRS_AMBIENT:					// 139
					DeviceStates.rsMap139 = dwRenderState;
					break;
				case D3DRS_FOGVERTEXMODE:			// 140
					DeviceStates.rsMap140 = dwRenderState;
					break;
				case D3DRS_COLORVERTEX:				// 141
					DeviceStates.rsMap141 = dwRenderState;
					break;
				case D3DRS_MULTISAMPLEANTIALIAS:	// 161
					DeviceStates.rsMap161 = dwRenderState;
					break;
				case D3DRS_MULTISAMPLEMASK:			// 162
					DeviceStates.rsMap162 = dwRenderState;
					break;
				case D3DRS_DEPTHBIAS:				// 195
					DeviceStates.rsMap195 = dwRenderState;
					break;
				default:
					DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
					break;
				}
				return D3D_OK;
			}
			return DDERR_INVALIDPARAMS;
		}

		switch ((DWORD)dwRenderStateType)
		{
		case D3DRENDERSTATE_NONE:				// 0
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			return D3D_OK;
		case D3DRENDERSTATE_TEXTUREHANDLE:		// 1
			return SetTextureHandle(dwRenderState);
		case D3DRENDERSTATE_ANTIALIAS:			// 2
		{
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			BOOL AntiAliasEnabled = (
				(D3DANTIALIASMODE)dwRenderState == D3DANTIALIAS_SORTDEPENDENT ||
				(D3DANTIALIASMODE)dwRenderState == D3DANTIALIAS_SORTINDEPENDENT) ? TRUE : FALSE;
			SetStateBlockRenderState(dwRenderStateType, dwRenderState);
			return SetD9RenderState(D3DRS_MULTISAMPLEANTIALIAS, AntiAliasEnabled);
		}
		case D3DRENDERSTATE_TEXTUREADDRESS:		// 3
			SetD9SamplerState(0, D3DSAMP_ADDRESSU, dwRenderState);
			return SetD9SamplerState(0, D3DSAMP_ADDRESSV, dwRenderState);
		case D3DRENDERSTATE_TEXTUREPERSPECTIVE:	// 4
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_TEXTUREPERSPECTIVE' not implemented: " << dwRenderState);
			}
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_WRAPU:				// 5
			return SetD9RenderState(D3DRS_WRAP0, (dwRenderState ? D3DWRAP_U : 0) | (DeviceStates.RenderState[D3DRENDERSTATE_WRAPV].State ? D3DWRAP_V : 0));
		case D3DRENDERSTATE_WRAPV:				// 6
			return SetD9RenderState(D3DRS_WRAP0, (DeviceStates.RenderState[D3DRENDERSTATE_WRAPU].State ? D3DWRAP_U : 0) | (dwRenderState ? D3DWRAP_V : 0));
		case D3DRENDERSTATE_LINEPATTERN:		// 10
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != 0)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_LINEPATTERN' not implemented: " << dwRenderState);
			}
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_MONOENABLE:			// 11
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_MONOENABLE' not implemented: " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_ROP2:				// 12
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != R2_COPYPEN)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_ROP2' not implemented: " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_PLANEMASK:			// 13
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != ~0)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_PLANEMASK' not implemented: " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_TEXTUREMAG:			// 17
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			switch (dwRenderState)
			{
			// Only the first two (D3DFILTER_NEAREST and D3DFILTER_LINEAR) are valid with D3DRENDERSTATE_TEXTUREMAG.
			case NULL:
			case D3DFILTER_NEAREST:
			case D3DFILTER_LINEAR:
				return SetD9SamplerState(0, D3DSAMP_MAGFILTER, dwRenderState);
			default:
				LOG_LIMIT(100, __FUNCTION__ << " Warning: unsupported 'D3DRENDERSTATE_TEXTUREMAG' state: " << dwRenderState);
				return D3D_OK;
			}
		case D3DRENDERSTATE_TEXTUREMIN:			// 18
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			switch (dwRenderState)
			{
			case NULL:
			case D3DFILTER_NEAREST:
			case D3DFILTER_LINEAR:
				SetD9SamplerState(0, D3DSAMP_MINFILTER, dwRenderState);
				return SetD9SamplerState(0, D3DSAMP_MIPFILTER, D3DTFP_NONE);
			case D3DFILTER_MIPNEAREST:
				SetD9SamplerState(0, D3DSAMP_MINFILTER, D3DTFN_POINT);
				return SetD9SamplerState(0, D3DSAMP_MIPFILTER, D3DTFP_POINT);
			case D3DFILTER_MIPLINEAR:
				SetD9SamplerState(0, D3DSAMP_MINFILTER, D3DTFN_LINEAR);
				return SetD9SamplerState(0, D3DSAMP_MIPFILTER, D3DTFP_POINT);
			case D3DFILTER_LINEARMIPNEAREST:
				SetD9SamplerState(0, D3DSAMP_MINFILTER, D3DTFN_POINT);
				return SetD9SamplerState(0, D3DSAMP_MIPFILTER, D3DTFP_LINEAR);
			case D3DFILTER_LINEARMIPLINEAR:
				SetD9SamplerState(0, D3DSAMP_MINFILTER, D3DTFN_LINEAR);
				return SetD9SamplerState(0, D3DSAMP_MIPFILTER, D3DTFP_LINEAR);
			default:
				LOG_LIMIT(100, __FUNCTION__ << " Warning: unsupported 'D3DRENDERSTATE_TEXTUREMIN' state: " << dwRenderState);
				return D3D_OK;
			}
		case D3DRENDERSTATE_TEXTUREMAPBLEND:	// 21
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			switch (dwRenderState)
			{
			case D3DTBLEND_COPY:
			case D3DTBLEND_DECAL:
				// Reset states
				SetD9TextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_COLORARG2, D3DTA_CURRENT);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_CURRENT);

				// Decal texture-blending mode is supported. In this mode, the RGB and alpha values of the texture replace the colors that would have been used with no texturing.
				SetD9TextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
				SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);

				return D3D_OK;
			case D3DTBLEND_DECALALPHA:
				// Reset states
				SetD9TextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);

				// Decal-alpha texture-blending mode is supported. In this mode, the RGB and alpha values of the texture are 
				// blended with the colors that would have been used with no texturing.
				SetD9TextureStageState(0, D3DTSS_COLOROP, D3DTOP_BLENDTEXTUREALPHA);
				SetD9TextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
				SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);

				return D3D_OK;
			case D3DTBLEND_DECALMASK:
				// This blending mode is not supported. When the least-significant bit of the texture's alpha component is zero, 
				// the effect is as if texturing were disabled.
				LOG_LIMIT(100, __FUNCTION__ << " Warning: unsupported 'D3DTBLEND_DECALMASK' state: " << dwRenderState);

				return D3D_OK;
			case D3DTBLEND_MODULATE:
				// Reset states
				SetD9TextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);

				// Modulate texture-blending mode is supported. In this mode, the RGB values of the texture are multiplied 
				// with the RGB values that would have been used with no texturing. Any alpha values in the texture replace 
				// the alpha values in the colors that would have been used with no texturing; if the texture does not contain 
				// an alpha component, alpha values at the vertices in the source are interpolated between vertices.
				SetD9TextureStageState(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
				SetD9TextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
				SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
				DeviceStates.TextureStageState[0][D3DTSS_ALPHAOP].Set = FALSE;	// Sets alphaop to auto based on texture alpha channel

				return D3D_OK;
			case D3DTBLEND_MODULATEALPHA:
				// Reset states
				SetD9TextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);

				// Modulate-alpha texture-blending mode is supported. In this mode, the RGB values of the texture are multiplied 
				// with the RGB values that would have been used with no texturing, and the alpha values of the texture are multiplied 
				// with the alpha values that would have been used with no texturing.
				SetD9TextureStageState(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
				SetD9TextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
				SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_MODULATE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);

				return D3D_OK;
			case D3DTBLEND_MODULATEMASK:
				// This blending mode is not supported. When the least-significant bit of the texture's alpha component is zero, 
				// the effect is as if texturing were disabled.
				LOG_LIMIT(100, __FUNCTION__ << " Warning: unsupported 'D3DTBLEND_MODULATEMASK' state: " << dwRenderState);

				return D3D_OK;
			case D3DTBLEND_ADD:
				// Reset states
				SetD9TextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);

				// Add the Gouraud interpolants to the texture lookup with saturation semantics
				// (that is, if the color value overflows it is set to the maximum possible value).
				SetD9TextureStageState(0, D3DTSS_COLOROP, D3DTOP_ADD);
				SetD9TextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
				SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2);
				SetD9TextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);

				return D3D_OK;
			default:
				LOG_LIMIT(100, __FUNCTION__ << " Warning: unsupported 'D3DRENDERSTATE_TEXTUREMAPBLEND' state: " << dwRenderState);
				return D3D_OK;
			}
		case D3DRENDERSTATE_ALPHAREF:			// 24
			dwRenderState &= 0xFF;
			break;
		case D3DRENDERSTATE_BLENDENABLE:		// 27
			if (ClientDirectXVersion == 1)
			{
				DeviceStates.RenderState[D3DRENDERSTATE_COLORKEYENABLE].State = dwRenderState;
			}
			break;
		case D3DRENDERSTATE_ZVISIBLE:			// 30
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_ZVISIBLE' not implemented: " << dwRenderState);
			}
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_SUBPIXEL:			// 31
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_SUBPIXEL' not implemented: " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_SUBPIXELX:			// 32
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_SUBPIXELX' not implemented: " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_STIPPLEDALPHA:		// 33
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_STIPPLEDALPHA' not implemented! " << dwRenderState);
			}
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_STIPPLEENABLE:		// 39
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_STIPPLEENABLE' not implemented! " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_EDGEANTIALIAS:		// 40
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_EDGEANTIALIAS' not implemented! " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_COLORKEYENABLE:		// 41
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_OLDALPHABLENDENABLE:// 42
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			return SetD9RenderState(D3DRENDERSTATE_BLENDENABLE, dwRenderState);
		case D3DRENDERSTATE_BORDERCOLOR:		// 43
			return SetD9SamplerState(0, D3DSAMP_BORDERCOLOR, dwRenderState);
		case D3DRENDERSTATE_TEXTUREADDRESSU:	// 44
			return SetD9SamplerState(0, D3DSAMP_ADDRESSU, dwRenderState);
		case D3DRENDERSTATE_TEXTUREADDRESSV:	// 45
			return SetD9SamplerState(0, D3DSAMP_ADDRESSV, dwRenderState);
		case D3DRENDERSTATE_MIPMAPLODBIAS:		// 46
			return SetD9SamplerState(0, D3DSAMP_MIPMAPLODBIAS, dwRenderState);
		case D3DRENDERSTATE_ZBIAS:				// 47
			if (dwRenderState != DeviceStates.RenderState[dwRenderStateType].State)
			{
				SetD9RenderState(D3DRS_DEPTHBIAS, GetDepthBias(dwRenderState, DepthBitCount));
			}
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_ANISOTROPY:			// 49
			return SetD9SamplerState(0, D3DSAMP_MAXANISOTROPY, dwRenderState);
		case D3DRENDERSTATE_FLUSHBATCH:			// 50
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_FLUSHBATCH' not implemented! " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_TRANSLUCENTSORTINDEPENDENT:	// 51
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_TRANSLUCENTSORTINDEPENDENT' not implemented! " << dwRenderState);
			}
			return D3D_OK;
		case 61:
		case 62:
		case 63:
		case D3DRENDERSTATE_STIPPLEPATTERN00:	// 64
		case D3DRENDERSTATE_STIPPLEPATTERN01:	// 65
		case D3DRENDERSTATE_STIPPLEPATTERN02:	// 66
		case D3DRENDERSTATE_STIPPLEPATTERN03:	// 67
		case D3DRENDERSTATE_STIPPLEPATTERN04:	// 68
		case D3DRENDERSTATE_STIPPLEPATTERN05:	// 69
		case D3DRENDERSTATE_STIPPLEPATTERN06:	// 70
		case D3DRENDERSTATE_STIPPLEPATTERN07:	// 71
		case D3DRENDERSTATE_STIPPLEPATTERN08:	// 72
		case D3DRENDERSTATE_STIPPLEPATTERN09:	// 73
		case D3DRENDERSTATE_STIPPLEPATTERN10:	// 74
		case D3DRENDERSTATE_STIPPLEPATTERN11:	// 75
		case D3DRENDERSTATE_STIPPLEPATTERN12:	// 76
		case D3DRENDERSTATE_STIPPLEPATTERN13:	// 77
		case D3DRENDERSTATE_STIPPLEPATTERN14:	// 78
		case D3DRENDERSTATE_STIPPLEPATTERN15:	// 79
		case D3DRENDERSTATE_STIPPLEPATTERN16:	// 80
		case D3DRENDERSTATE_STIPPLEPATTERN17:	// 81
		case D3DRENDERSTATE_STIPPLEPATTERN18:	// 82
		case D3DRENDERSTATE_STIPPLEPATTERN19:	// 83
		case D3DRENDERSTATE_STIPPLEPATTERN20:	// 84
		case D3DRENDERSTATE_STIPPLEPATTERN21:	// 85
		case D3DRENDERSTATE_STIPPLEPATTERN22:	// 86
		case D3DRENDERSTATE_STIPPLEPATTERN23:	// 87
		case D3DRENDERSTATE_STIPPLEPATTERN24:	// 88
		case D3DRENDERSTATE_STIPPLEPATTERN25:	// 89
		case D3DRENDERSTATE_STIPPLEPATTERN26:	// 90
		case D3DRENDERSTATE_STIPPLEPATTERN27:	// 91
		case D3DRENDERSTATE_STIPPLEPATTERN28:	// 92
		case D3DRENDERSTATE_STIPPLEPATTERN29:	// 93
		case D3DRENDERSTATE_STIPPLEPATTERN30:	// 94
		case D3DRENDERSTATE_STIPPLEPATTERN31:	// 95
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != 0)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_STIPPLEPATTERN00' not implemented! " << dwRenderState);
			}
			return D3D_OK;
		case D3DRENDERSTATE_LIGHTING:			// 137
			if (Config.DdrawDisableLighting)
			{
				dwRenderState = FALSE;
			}
		break;
		case D3DRENDERSTATE_EXTENTS:			// 138
			// ToDo: use this to enable/disable clip plane extents set by SetClipStatus()
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			if (dwRenderState != FALSE)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DRENDERSTATE_EXTENTS' not implemented! " << dwRenderState);
			}
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		case D3DRENDERSTATE_COLORKEYBLENDENABLE:// 144
			DeviceStates.RenderState[dwRenderStateType].State = dwRenderState;
			return SetStateBlockRenderState(dwRenderStateType, dwRenderState);
		}

		return SetD9RenderState(dwRenderStateType, dwRenderState);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetRenderState(dwRenderStateType, dwRenderState);
	case 3:
		return GetProxyInterfaceV3()->SetRenderState(dwRenderStateType, dwRenderState);
	case 7:
		return GetProxyInterfaceV7()->SetRenderState(dwRenderStateType, dwRenderState);
	}
}

HRESULT m_IDirect3DDeviceX::GetLightState(D3DLIGHTSTATETYPE dwLightStateType, LPDWORD lpdwLightState)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwLightState)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Light state called with nullptr: " << dwLightStateType);
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			*lpdwLightState = 0;
			return DDERR_INVALIDOBJECT;
		}

		if (dwLightStateType == 0
			|| (ClientDirectXVersion == 2 && dwLightStateType > 7)
			|| (ClientDirectXVersion == 3 && dwLightStateType > 8))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: unknown LightStateType: " << dwLightStateType);
			*lpdwLightState = 0;
			return DDERR_INVALIDPARAMS;
		}

		switch (dwLightStateType)
		{
		case D3DLIGHTSTATE_MATERIAL:
			*lpdwLightState = DeviceStates.LightState[dwLightStateType];
			return D3D_OK;
		case D3DLIGHTSTATE_AMBIENT:
			return GetD9RenderState(D3DRS_AMBIENT, lpdwLightState);
		case D3DLIGHTSTATE_COLORMODEL:
			*lpdwLightState = DeviceStates.LightState[dwLightStateType];
			return D3D_OK;
		case D3DLIGHTSTATE_FOGMODE:
			return GetD9RenderState(D3DRS_FOGVERTEXMODE, lpdwLightState);
		case D3DLIGHTSTATE_FOGSTART:
			// Fog start is 0 until it is assigned
			if (DeviceStates.LightState[dwLightStateType])
			{
				return GetD9RenderState(D3DRS_FOGSTART, lpdwLightState);
			}
			*lpdwLightState = DeviceStates.LightState[dwLightStateType];
			return D3D_OK;
		case D3DLIGHTSTATE_FOGEND:
			// Fog end is 0 until it is assigned
			if (DeviceStates.LightState[dwLightStateType])
			{
				return GetD9RenderState(D3DRS_FOGSTART, lpdwLightState);
			}
			*lpdwLightState = DeviceStates.LightState[dwLightStateType];
			return D3D_OK;
		case D3DLIGHTSTATE_FOGDENSITY:
			// Fog density is 0 until it is assigned
			if (DeviceStates.LightState[dwLightStateType])
			{
				return GetD9RenderState(D3DRS_FOGDENSITY, lpdwLightState);
			}
			*lpdwLightState = DeviceStates.LightState[dwLightStateType];
			return D3D_OK;
		case D3DLIGHTSTATE_COLORVERTEX:
			return GetD9RenderState(D3DRS_COLORVERTEX, lpdwLightState);
		default:
			break;
		}

		LOG_LIMIT(100, __FUNCTION__ << " Error: unknown LightStateType: " << dwLightStateType);
		*lpdwLightState = 0;
		return DDERR_INVALIDPARAMS;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->GetLightState(dwLightStateType, lpdwLightState);
	case 3:
		return GetProxyInterfaceV3()->GetLightState(dwLightStateType, lpdwLightState);
	}
}

HRESULT m_IDirect3DDeviceX::SetLightState(D3DLIGHTSTATETYPE dwLightStateType, DWORD dwLightState)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ") " << dwLightStateType << " " << dwLightState;

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		if (dwLightStateType == 0
			|| (ClientDirectXVersion == 2 && dwLightStateType > 7)
			|| (ClientDirectXVersion == 3 && dwLightStateType > 8))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: unknown LightStateType: " << dwLightStateType);
			return DDERR_INVALIDPARAMS;
		}

		switch (dwLightStateType)
		{
		case D3DLIGHTSTATE_MATERIAL:
			return SetMaterialHandle(dwLightState);
		case D3DLIGHTSTATE_AMBIENT:
			return SetD9RenderState(D3DRS_AMBIENT, dwLightState);
		case D3DLIGHTSTATE_COLORMODEL:
			if (dwLightState != D3DCOLOR_RGB)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: 'D3DLIGHTSTATE_COLORMODEL' not implemented! " << dwLightState);
			}
			break;
		case D3DLIGHTSTATE_FOGMODE:
			return SetD9RenderState(D3DRS_FOGVERTEXMODE, dwLightState);
		case D3DLIGHTSTATE_FOGSTART:
			// Fog start is 0 until it is assigned
			DeviceStates.LightState[dwLightStateType] = dwLightState;
			return SetD9RenderState(D3DRS_FOGSTART, dwLightState);
		case D3DLIGHTSTATE_FOGEND:
			// Fog end is 0 until it is assigned
			DeviceStates.LightState[dwLightStateType] = dwLightState;
			return SetD9RenderState(D3DRS_FOGEND, dwLightState);
		case D3DLIGHTSTATE_FOGDENSITY:
			// Fog density is 0 until it is assigned
			DeviceStates.LightState[dwLightStateType] = dwLightState;
			return SetD9RenderState(D3DRS_FOGDENSITY, dwLightState);
		case D3DLIGHTSTATE_COLORVERTEX:
			return SetD9RenderState(D3DRS_COLORVERTEX, dwLightState);
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: unknown LightStateType: " << dwLightStateType);
			return DDERR_INVALIDPARAMS;
		}

		DeviceStates.LightState[dwLightStateType] = dwLightState;
		return D3D_OK;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetLightState(dwLightStateType, dwLightState);
	case 3:
		return GetProxyInterfaceV3()->SetLightState(dwLightStateType, dwLightState);
	}
}

HRESULT m_IDirect3DDeviceX::SetTransform(D3DTRANSFORMSTATETYPE dtstTransformStateType, LPD3DMATRIX lpD3DMatrix)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DMatrix)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		switch ((DWORD)dtstTransformStateType)
		{
		case D3DTRANSFORMSTATE_WORLD:
			dtstTransformStateType = D3DTS_WORLD;
			break;
		case D3DTRANSFORMSTATE_WORLD1:
			dtstTransformStateType = D3DTS_WORLD1;
			break;
		case D3DTRANSFORMSTATE_WORLD2:
			dtstTransformStateType = D3DTS_WORLD2;
			break;
		case D3DTRANSFORMSTATE_WORLD3:
			dtstTransformStateType = D3DTS_WORLD3;
			break;
		}

		D3DMATRIX view;
		if (Config.DdrawConvertHomogeneousW)
		{
			if (dtstTransformStateType == D3DTS_VIEW)
			{
				D3DVIEWPORT9 Viewport9;
				if (SUCCEEDED((*d3d9Device)->GetViewport(&Viewport9)))
				{
					const float width = (float)Viewport9.Width;
					const float height = (float)Viewport9.Height;

					// Replace the matrix with one that handles D3DFVF_XYZRHW geometry
					ZeroMemory(&view, sizeof(D3DMATRIX));
					view._11 = 2.0f / width;
					view._22 = -2.0f / height;
					view._33 = 1.0f;
					view._41 = -1.0f;  // translate X
					view._42 = 1.0f;   // translate Y
					view._44 = 1.0f;

					// Set flag
					ConvertHomogeneous.IsTransformViewSet = true;

					if (Config.DdrawConvertHomogeneousToWorld)
					{
						DirectX::XMVECTOR position, direction;
						float depthOffset = 0.0f;
						if (Config.DdrawConvertHomogeneousToWorldUseGameCamera)
						{
							// To reconstruct the 3D world, we need to know where the camera is and where it is looking
							position = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);

							const float x = lpD3DMatrix->_11;
							const float y = lpD3DMatrix->_12;
							const float z = lpD3DMatrix->_13;

							float pitch = std::atan2(y, z);
							if (pitch < 0.0f && y * z > 0.0f)  // check if y and z have the same sign
							{
								// handle flipping of the pitch. This is not because the camera is looking up.
								pitch += DirectX::XM_PI;
							}

							float yaw = std::asin(x);
							if (yaw < 0.0f)
							{
								yaw += DirectX::XM_2PI;
							}

							// mirror the transform
							float pitchneg = -pitch;

							float pitch_cos = std::cos(pitchneg);
							float x2 = 0.0f;  //std::cos(yaw) * pitch_cos;
							float y2 = std::sin(pitchneg);
							float z2 = /*std::sin(yaw) **/ pitch_cos;

							direction = DirectX::XMVectorSet(x2, y2, z2, 0.0f);

							depthOffset = Config.DdrawConvertHomogeneousToWorldDepthOffset;

							ConvertHomogeneous.ToWorld_GameCameraYaw = yaw;
							ConvertHomogeneous.ToWorld_GameCameraPitch = pitch;
						}
						else
						{
							// DK2 isometric camera - positioned above, looking down at ~45 degree angle
							position = DirectX::XMVectorSet(0.0f, 800.0f, -800.0f, 0.0f);
							direction = DirectX::XMVector3Normalize(DirectX::XMVectorSet(0.0f, -0.707f, 0.707f, 0.0f));
						}

						// Store the original matrix so it can be restored
						ConvertHomogeneous.ToWorld_ViewMatrixOriginal = view;

						// The Black & White matrix is an ortho camera, so create a perspective one matching the game
						const float fov = Config.DdrawConvertHomogeneousToWorldFOV;
						const float nearplane = Config.DdrawConvertHomogeneousToWorldNearPlane;
						const float farplane = Config.DdrawConvertHomogeneousToWorldFarPlane;
						const float ratio = width / height;
						DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovLH(fov * (3.14159265359f / 180.0f), ratio, nearplane, farplane);

						DirectX::XMStoreFloat4x4((DirectX::XMFLOAT4X4*)&ConvertHomogeneous.ToWorld_ProjectionMatrix, proj);

						DirectX::XMVECTOR upVector = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
						DirectX::XMMATRIX viewMatrix = DirectX::XMMatrixLookToLH(position, direction, upVector);

						// Store the 3D view matrix so it can be set later
						DirectX::XMStoreFloat4x4((DirectX::XMFLOAT4X4*)&ConvertHomogeneous.ToWorld_ViewMatrix, viewMatrix);

						// Store the view inverse matrix of the game, so we can transform the geometry with it
						DirectX::XMMATRIX toViewSpace = DirectX::XMLoadFloat4x4((DirectX::XMFLOAT4X4*)&view);
						DirectX::XMMATRIX vp = DirectX::XMMatrixMultiply(viewMatrix, proj);
						DirectX::XMMATRIX vpinv = DirectX::XMMatrixInverse(nullptr, vp);

						DirectX::XMMATRIX depthoffset = DirectX::XMMatrixTranslation(0.0f, 0.0f, depthOffset);

						ConvertHomogeneous.ToWorld_ViewMatrixInverse = DirectX::XMMatrixMultiply(depthoffset, DirectX::XMMatrixMultiply(toViewSpace, vpinv));
					}

					// Override original matrix pointer
					lpD3DMatrix = &view;
				}
			}
			else
			{
				return D3D_OK;
			}
		}

		HRESULT hr = SetD9Transform(dtstTransformStateType, lpD3DMatrix);

#ifdef ENABLE_DEBUGOVERLAY
		if (SUCCEEDED(hr) && !Config.DdrawConvertHomogeneousW)
		{
			DOverlay.SetTransform(dtstTransformStateType, lpD3DMatrix);
		}
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetTransform(dtstTransformStateType, lpD3DMatrix);
	case 3:
		return GetProxyInterfaceV3()->SetTransform(dtstTransformStateType, lpD3DMatrix);
	case 7:
		return GetProxyInterfaceV7()->SetTransform(dtstTransformStateType, lpD3DMatrix);
	}
}

HRESULT m_IDirect3DDeviceX::GetTransform(D3DTRANSFORMSTATETYPE dtstTransformStateType, LPD3DMATRIX lpD3DMatrix)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DMatrix)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			*lpD3DMatrix = {};
			return DDERR_INVALIDOBJECT;
		}

		switch ((DWORD)dtstTransformStateType)
		{
		case D3DTRANSFORMSTATE_WORLD:
			dtstTransformStateType = D3DTS_WORLD;
			break;
		case D3DTRANSFORMSTATE_WORLD1:
			dtstTransformStateType = D3DTS_WORLD1;
			break;
		case D3DTRANSFORMSTATE_WORLD2:
			dtstTransformStateType = D3DTS_WORLD2;
			break;
		case D3DTRANSFORMSTATE_WORLD3:
			dtstTransformStateType = D3DTS_WORLD3;
			break;
		}

		return GetD9Transform(dtstTransformStateType, lpD3DMatrix);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->GetTransform(dtstTransformStateType, lpD3DMatrix);
	case 3:
		return GetProxyInterfaceV3()->GetTransform(dtstTransformStateType, lpD3DMatrix);
	case 7:
		return GetProxyInterfaceV7()->GetTransform(dtstTransformStateType, lpD3DMatrix);
	}
}

HRESULT m_IDirect3DDeviceX::MultiplyTransform(D3DTRANSFORMSTATETYPE dtstTransformStateType, LPD3DMATRIX lpD3DMatrix)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		switch ((DWORD)dtstTransformStateType)
		{
		case D3DTRANSFORMSTATE_WORLD:
			dtstTransformStateType = D3DTS_WORLD;
			break;
		case D3DTRANSFORMSTATE_WORLD1:
			dtstTransformStateType = D3DTS_WORLD1;
			break;
		case D3DTRANSFORMSTATE_WORLD2:
			dtstTransformStateType = D3DTS_WORLD2;
			break;
		case D3DTRANSFORMSTATE_WORLD3:
			dtstTransformStateType = D3DTS_WORLD3;
			break;
		}

		return D9MultiplyTransform(dtstTransformStateType, lpD3DMatrix);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->MultiplyTransform(dtstTransformStateType, lpD3DMatrix);
	case 3:
		return GetProxyInterfaceV3()->MultiplyTransform(dtstTransformStateType, lpD3DMatrix);
	case 7:
		return GetProxyInterfaceV7()->MultiplyTransform(dtstTransformStateType, lpD3DMatrix);
	}
}

HRESULT m_IDirect3DDeviceX::DrawPrimitive(D3DPRIMITIVETYPE dptPrimitiveType, DWORD dwVertexTypeDesc, LPVOID lpVertices, DWORD dwVertexCount, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" VertexType = " << Logging::hex(dptPrimitiveType) <<
		" VertexDesc = " << Logging::hex(dwVertexTypeDesc) <<
		" Vertices = " << lpVertices <<
		" VertexCount = " << dwVertexCount <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Version = " << DirectXVersion;

	if (Config.Dd7to9)
	{
		if (dwVertexCount == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpVertices)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (DirectXVersion == 2)
		{
			if (dwVertexTypeDesc != D3DVT_VERTEX && dwVertexTypeDesc != D3DVT_LVERTEX && dwVertexTypeDesc != D3DVT_TLVERTEX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: invalid Vertex type: " << dwVertexTypeDesc);
				return D3DERR_INVALIDVERTEXTYPE;
			}
			dwVertexTypeDesc = ConvertVertexTypeToFVF((D3DVERTEXTYPE)dwVertexTypeDesc);
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection(), !Config.DdrawNoDrawBufferSysLock);

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		// Update vertices for Direct3D9 (needs to be first)
		UpdateVertices(dwVertexTypeDesc, lpVertices, 0, dwVertexCount);

		// === UI detection via RHW-constancy heuristic ===
		// Pre-transformed UI overlays use a constant RHW per draw call (no perspective
		// division applied). Pre-transformed 3D world geometry has varying RHW (= 1/W).
		// If RHW is constant across the verts, treat this draw as UI: skip the inverse
		// projection so Remix sees pure XYZRHW screen-space verts and can render it as
		// a 2D overlay rather than path-traced world geometry.
		bool isUIPassthrough = false;
		bool dynUiEmissiveDraw = false;
		bool orphanOverlayDraw = false;
		float uiHeuristicMinRHW = 0.0f, uiHeuristicMaxRHW = 0.0f;
		if ((dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW && dwVertexCount > 0 && lpVertices)
		{
			const UINT scanStride = GetVertexStride(dwVertexTypeDesc);
			const UINT8* scanVertex = (const UINT8*)lpVertices;
			uiHeuristicMinRHW = uiHeuristicMaxRHW = ((const float*)scanVertex)[3];
			for (UINT i = 1; i < dwVertexCount; i++)
			{
				scanVertex += scanStride;
				const float rhw = ((const float*)scanVertex)[3];
				if (rhw < uiHeuristicMinRHW) uiHeuristicMinRHW = rhw;
				if (rhw > uiHeuristicMaxRHW) uiHeuristicMaxRHW = rhw;
			}
			const float denom = (uiHeuristicMaxRHW > 0.0f) ? uiHeuristicMaxRHW : 1.0f;
			const bool isConstantRHW = ((uiHeuristicMaxRHW - uiHeuristicMinRHW) / denom) < 1e-4f;
			isUIPassthrough = isConstantRHW;
			// [may20-minus-mip] DISABLED: the UI-passthrough heuristic classified
			// constant-RHW torch billboards as UI -> they skipped the world inverse
			// projection -> torches vanish by angle/distance under Remix. OG (pre-May-2)
			// projected ALL XYZRHW draws. Force false to restore that baseline.
			isUIPassthrough = false;
			orphanOverlayDraw = Config.DdrawOrphanOverlayLift && isConstantRHW;

			// === Phase A.8 (atlas decomposition prep): UV bounds for stage 0 ===
			// Each drawcall samples a sub-rectangle of the bound atlas. Aggregating
			// (tex_ptr, u_min, v_min, u_max, v_max) per draw lets us cluster atlas
			// usage by UV region offline -- 1-in-1 atlases show single region,
			// k-in-1 atlases show k discrete clusters of varying shape.
			float u_min = 0, u_max = 0, v_min = 0, v_max = 0;
			bool has_uv = false;
			const DWORD numTexCoords = (dwVertexTypeDesc & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
			if (numTexCoords > 0)
			{
				UINT uvOff = 16;  // XYZRHW = 4 floats
				if (dwVertexTypeDesc & D3DFVF_RESERVED1) uvOff += 4;
				if (dwVertexTypeDesc & D3DFVF_DIFFUSE) uvOff += 4;
				if (dwVertexTypeDesc & D3DFVF_SPECULAR) uvOff += 4;
				const UINT8* vbase = (const UINT8*)lpVertices;
				const float* uv0 = (const float*)(vbase + uvOff);
				u_min = u_max = uv0[0];
				v_min = v_max = uv0[1];
				for (UINT i = 1; i < dwVertexCount; i++)
				{
					const float* uv = (const float*)(vbase + i * scanStride + uvOff);
					if (uv[0] < u_min) u_min = uv[0];
					if (uv[0] > u_max) u_max = uv[0];
					if (uv[1] < v_min) v_min = uv[1];
					if (uv[1] > v_max) v_max = uv[1];
				}
				has_uv = true;
			}

			void* uiTexPtr = (void*)CurrentTextureSurfaceX[0];
			LOG_LIMIT(20000, "DRAW_XYZRHW DrawPrimitive ui=" << (isUIPassthrough ? 1 : 0)
				<< " vcount=" << dwVertexCount
				<< " rhw=[" << uiHeuristicMinRHW << "," << uiHeuristicMaxRHW << "]"
				<< " tex=" << uiTexPtr
				<< (has_uv ? (std::string(" uv=[") + std::to_string(u_min) + "," + std::to_string(v_min) + "," + std::to_string(u_max) + "," + std::to_string(v_max) + "]").c_str() : ""));

			// Dynamic-UI emissive: DrawPrimitive has no decompose probe, so only the
			// recipe-level (solo runtime-rendered page, e.g. the minimap) check applies.
			NamekeyResetDrawDynUi();
			dynUiEmissiveDraw = Config.DdrawDynamicUiEmissive &&
				CurrentTextureSurfaceX[0] && CurrentTextureSurfaceX[0]->IsNamekeyDynamicUiPage();

			// Orphan-overlay lift (2026-06-12, menu-text fix): a constant-RHW quad
			// binding a RECIPELESS page (content never sourced from the tracked
			// EngineTextures upload = runtime-composited text/overlay, e.g. front-end
			// menu labels) is invisible under the inverse projection: its screen z
			// buries it inside the path-traced backdrop and no scene light reaches it
			// (floodlight-tested 2026-06-12). Lift it to a fixed near depth and draw
			// it additive (self-lit). Recipe-bearing pages (torch billboards etc.) are
			// excluded by construction -- this cannot re-run the May-20 passthrough
			// regression. See dk2_menu_res_port memory.
			orphanOverlayDraw = orphanOverlayDraw &&
				((CurrentTextureSurfaceX[0] && CurrentTextureSurfaceX[0]->IsNamekeyOrphanPage()) || MenuBlitOverlayDrawActive());
			if (orphanOverlayDraw) dynUiEmissiveDraw = true;

			// [LIFT] telemetry v2 (own budget): constant-RHW draws ONLY -- the
			// orphan-page OR-branch flooded the budget in 2.6s (the front-end's 3D
			// backdrop binds recipe-less textures wholesale, so "orphan" is the NORM
			// there, not a discriminator). Recipe names identify what kind of page
			// each overlay draw binds (font/text pages vs sprite pages).
			if (Config.DdrawOrphanOverlayLift && isConstantRHW)
			{
				char recipeBrief[256];
				if (CurrentTextureSurfaceX[0]) CurrentTextureSurfaceX[0]->GetNamekeyRecipeBrief(recipeBrief, sizeof(recipeBrief));
				else strcpy_s(recipeBrief, sizeof(recipeBrief), "notex");
				LOG_LIMIT(100000, "[LIFT] DrawPrimitive " << (orphanOverlayDraw ? "LIFTED" : "skip")
					<< " tex=" << (void*)CurrentTextureSurfaceX[0]
					<< " recipe=" << recipeBrief
					<< " vcount=" << dwVertexCount
					<< " rhw=" << uiHeuristicMinRHW
					<< " srcz=" << ((const float*)lpVertices)[2]
					<< " liftz=" << g_orphanLiftZ);
			}
		}

		// Handle PositionT (pre-transformed vertices) - convert to world space for RTX Remix
		// UI passthrough draws skip this branch entirely so they emit as XYZRHW screen-space.
		if (Config.DdrawConvertHomogeneousW && (dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW && !isUIPassthrough)
		{
			static bool loggedDrawPrimConversion = false;
			if (!loggedDrawPrimConversion)
			{
				Logging::Log() << __FUNCTION__ << " Converting XYZRHW to world space (DrawPrimitive)";
				loggedDrawPrimConversion = true;
			}

			if (!ConvertHomogeneous.IsTransformViewSet)
			{
				D3DMATRIX Matrix = {};
				GetTransform(D3DTS_VIEW, &Matrix);
				SetTransform(D3DTS_VIEW, &Matrix);
			}

			if (!Config.DdrawConvertHomogeneousToWorld)
			{
				// Update the FVF
				dwVertexTypeDesc = (dwVertexTypeDesc & ~D3DFVF_XYZRHW) | D3DFVF_XYZW;
			}
			else
			{
				const UINT stride = GetVertexStride(dwVertexTypeDesc);
				const UINT targetStride = stride - sizeof(float);
				const UINT restSize = stride - sizeof(float) * 4;

				ConvertHomogeneous.ToWorld_IntermediateGeometry.resize(targetStride * dwVertexCount);

				UINT8* sourceVertex = (UINT8*)lpVertices;
				UINT8* targetVertex = (UINT8*)ConvertHomogeneous.ToWorld_IntermediateGeometry.data();

				lpVertices = targetVertex;

				for (UINT x = 0; x < dwVertexCount; x++)
				{
					// Transform the vertices into world space
					float* srcpos = (float*)sourceVertex;
					float* trgtpos = (float*)targetVertex;

					DirectX::XMVECTOR xpos = DirectX::XMVectorSet(srcpos[0], srcpos[1],
						orphanOverlayDraw ? g_orphanLiftZ : srcpos[2], srcpos[3]);
					DirectX::XMVECTOR xpos_global = DirectX::XMVector3TransformCoord(xpos, ConvertHomogeneous.ToWorld_ViewMatrixInverse);
					xpos_global = DirectX::XMVectorDivide(xpos_global, DirectX::XMVectorSplatW(xpos_global));

					trgtpos[0] = DirectX::XMVectorGetX(xpos_global);
					trgtpos[1] = DirectX::XMVectorGetY(xpos_global);
					trgtpos[2] = DirectX::XMVectorGetZ(xpos_global);

					// Copy the rest
					std::memcpy(targetVertex + sizeof(float) * 3, sourceVertex + sizeof(float) * 4, restSize);

					// Move to next vertex
					sourceVertex += stride;
					targetVertex += targetStride;
				}

				// Painter's order for lifted overlays: the next lifted draw sits
				// slightly nearer so later draws (text over panels) resolve on top.
				if (orphanOverlayDraw && g_orphanLiftZ > 0.01f) g_orphanLiftZ -= 1e-5f;

				// Set transform
				(*d3d9Device)->SetTransform(D3DTS_VIEW, &ConvertHomogeneous.ToWorld_ViewMatrix);
				(*d3d9Device)->SetTransform(D3DTS_PROJECTION, &ConvertHomogeneous.ToWorld_ProjectionMatrix);

				// Update the FVF
				const DWORD newVertexTypeDesc = (dwVertexTypeDesc & ~D3DFVF_XYZRHW) | D3DFVF_XYZ;

				// Set fixed function vertex type
				if (FAILED((*d3d9Device)->SetFVF(newVertexTypeDesc)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
					return D3DERR_INVALIDVERTEXTYPE;
				}

				// Handle dwFlags
				SetDrawStates(newVertexTypeDesc, dwFlags, DirectXVersion);

				// [BLITQUAD]: bind the overlay texture AFTER SetDrawStates (whose texture
				// loop rebinds the game's CurrentTexture state and would stomp it).
				if (IDirect3DTexture9* overlayTex = MenuBlitOverlayCurrentTex())
				{
					(*d3d9Device)->SetTexture(0, overlayTex);
				}

				// Draw primitive UP (dynamic-UI draws get the additive/emissive override)
				if (dynUiEmissiveDraw) DynUiBeginDraw(*d3d9Device);
				HRESULT hr = (*d3d9Device)->DrawPrimitiveUP(dptPrimitiveType, GetNumberOfPrimitives(dptPrimitiveType, dwVertexCount), lpVertices, targetStride);
				DynUiEndDraw(*d3d9Device);

				// Handle dwFlags
				RestoreDrawStates(hr, newVertexTypeDesc, dwFlags);

				return hr;
			}
		}

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(dwVertexTypeDesc)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
			return DDERR_INVALIDPARAMS;
		}

		// Handle dwFlags
		SetDrawStates(dwVertexTypeDesc, dwFlags, DirectXVersion);

		// [BLITQUAD]: bind the overlay texture AFTER SetDrawStates (see converted branch).
		if (IDirect3DTexture9* overlayTex = MenuBlitOverlayCurrentTex())
		{
			(*d3d9Device)->SetTexture(0, overlayTex);
		}

		// Draw primitive UP (dynamic-UI draws get the additive/emissive override)
		if (dynUiEmissiveDraw) DynUiBeginDraw(*d3d9Device);
		HRESULT hr = (*d3d9Device)->DrawPrimitiveUP(dptPrimitiveType, GetNumberOfPrimitives(dptPrimitiveType, dwVertexCount), lpVertices, GetVertexStride(dwVertexTypeDesc));
		DynUiEndDraw(*d3d9Device);

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawPrimitiveUP' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->DrawPrimitive(dptPrimitiveType, (D3DVERTEXTYPE)dwVertexTypeDesc, lpVertices, dwVertexCount, dwFlags);
	case 3:
		return GetProxyInterfaceV3()->DrawPrimitive(dptPrimitiveType, dwVertexTypeDesc, lpVertices, dwVertexCount, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawPrimitive(dptPrimitiveType, dwVertexTypeDesc, lpVertices, dwVertexCount, dwFlags);
	}
}

// ===================== [GEOMPROBE] geometry identity stability probe (v3) =====================
// Read-only. Does a DK2 draw carry a STABLE geometry identity a Remix mesh replacement could
// bind to? Positions bake to screen each frame so they're excluded. v1 (topology+UV) exploded
// because UVs index DK2's drifting runtime ATLASES. v2 (raw topology) STILL drifted (~71% of
// topologies one-shot, 14k distinct (v,i) count-pairs) => DK2 submits mostly variable-size
// per-frame BATCHES, with a recurring core (~284 meshes drawn 64+x). v3 adds two things:
//   (1) NORMALIZED topology -- subtract the min index before hashing + use the vertex SPAN
//       (max-min) not the buffer size, so the SAME mesh packed at different base offsets in a
//       shared per-frame vertex buffer hashes the SAME. Rules out base-offset FALSE drift
//       (the geometry analog of atlas-UV drift) that could be under-counting the bindable core.
//   (2) a RECURRENCE HISTOGRAM (1x / 2-9x / 10-99x / 100+x) on the normalized id -- the 100+x
//       bucket = the truly bindable core (creatures/props/repeat architecture), sized exactly.
// rawTopoIds (v2's key) is kept as a count-only A/B: rawTopoIds >> normIds == base-offset was
// splitting one mesh into many; normIds plateauing small == a real, bindable mesh set exists.
namespace {
	struct GeomProbeStat {
		uint32_t i = 0, span = 0, seen = 0;
		std::unordered_set<uint64_t> posSet;	// distinct baked-position hashes (capped)
		bool posCapped = false;
	};
	std::unordered_map<uint64_t, GeomProbeStat> g_geomNorm;	// primary: keyed by NORMALIZED topology
	std::unordered_set<uint64_t> g_geomRawSet;				// distinct raw topology ids -- A/B counter
	std::unordered_set<uint64_t> g_geomFullSet;				// distinct (topology^UV) ids -- A/B counter
	std::unordered_set<uint64_t> g_geomVISet;				// distinct (vcount,icount) pairs -- coarse floor
	unsigned long long g_geomProbeDraws = 0;
	unsigned long g_geomProbeNewLogged = 0;
	bool g_ovfNorm = false, g_ovfRaw = false, g_ovfFull = false;
	const size_t kNormMaxIds = 60000;		// backstop; a real bindable set is expected in the hundreds-low thousands
	const size_t kSetMaxIds = 500000;		// A/B counter backstop
	const size_t kGeomProbePosCap = 64;		// per-id distinct-position sample cap (>=2 is all we test)

	inline uint64_t GeomFnv(const void* data, size_t len, uint64_t h = 1469598103934665603ULL) {
		const uint8_t* p = (const uint8_t*)data;
		for (size_t k = 0; k < len; k++) { h ^= p[k]; h *= 1099511628211ULL; }
		return h;
	}

	// Called from DrawIndexedPrimitive under the DD critical section (map access is serialized).
	void GeomProbeObserve(LPVOID lpVertices, DWORD dwVertexCount, LPWORD lpwIndices, DWORD dwIndexCount,
		UINT stride, bool hasUv, UINT uvOff, float u0, float v0, float u1, float v1)
	{
		if (!lpVertices || !lpwIndices || dwVertexCount == 0 || dwIndexCount == 0) return;
		g_geomProbeDraws++;

		// Index range, to normalize away a shared-buffer base offset.
		WORD imin = lpwIndices[0], imax = lpwIndices[0];
		for (DWORD k = 1; k < dwIndexCount; k++) { WORD x = lpwIndices[k]; if (x < imin) imin = x; if (x > imax) imax = x; }
		const uint32_t span = (uint32_t)(imax - imin) + 1;	// real vertices the mesh uses (base-invariant)

		// NORMALIZED topology id: hash (index - imin) + icount + span. Base-offset invariant.
		uint64_t normH = 1469598103934665603ULL;
		for (DWORD k = 0; k < dwIndexCount; k++) { WORD d = (WORD)(lpwIndices[k] - imin); normH = GeomFnv(&d, sizeof(WORD), normH); }
		const uint64_t normId = normH ^ ((uint64_t)span << 40) ^ ((uint64_t)dwIndexCount << 8);
		// RAW topology id (v2 key): includes absolute index values + buffer vcount.
		const uint64_t rawId = GeomFnv(lpwIndices, (size_t)dwIndexCount * sizeof(WORD))
			^ ((uint64_t)dwVertexCount << 40) ^ ((uint64_t)dwIndexCount << 8);
		// UV + position hashes (strided). Neither is in the identity -- UV only for the A/B count.
		uint64_t uvH = 1469598103934665603ULL, posH = 1469598103934665603ULL;
		const uint8_t* vb = (const uint8_t*)lpVertices;
		for (UINT n = 0; n < dwVertexCount; n++) {
			const uint8_t* vtx = vb + (size_t)n * stride;
			posH = GeomFnv(vtx, 12, posH);					// xyz (skip RHW at +12)
			if (hasUv) uvH = GeomFnv(vtx + uvOff, 8, uvH);	// u,v
		}
		// A/B counters.
		if (g_geomRawSet.size() < kSetMaxIds) g_geomRawSet.insert(rawId); else g_ovfRaw = true;
		if (g_geomFullSet.size() < kSetMaxIds) g_geomFullSet.insert(rawId ^ (uvH * 0x9E3779B97F4A7C15ULL)); else g_ovfFull = true;
		g_geomVISet.insert(((uint64_t)dwVertexCount << 32) | dwIndexCount);

		auto it = g_geomNorm.find(normId);
		if (it == g_geomNorm.end()) {
			if (g_geomNorm.size() >= kNormMaxIds) { g_ovfNorm = true; return; }
			GeomProbeStat st; st.i = dwIndexCount; st.span = span; st.seen = 1;
			st.posSet.insert(posH);
			g_geomNorm.emplace(normId, std::move(st));
			if (g_geomProbeNewLogged < 600) {
				g_geomProbeNewLogged++;
				char buf[200];
				sprintf_s(buf, sizeof(buf), "[GEOMPROBE] NEW norm=%016llX span=%lu i=%lu uv=[%.4f,%.4f,%.4f,%.4f]",
					(unsigned long long)normId, (unsigned long)span, (unsigned long)dwIndexCount, u0, v0, u1, v1);
				Logging::Log() << buf;
			}
		} else {
			GeomProbeStat& st = it->second;
			st.seen++;
			if (!st.posCapped) { if (st.posSet.size() < kGeomProbePosCap) st.posSet.insert(posH); else st.posCapped = true; }
		}

		// Summary every 10k draws. THE answer: normIds vs rawIds (base-offset?), normIds plateau?,
		// and the recurrence histogram -- hist100 = the cleanly-bindable mesh core, sized exactly.
		if ((g_geomProbeDraws % 10000ULL) == 0) {
			size_t multiPos = 0, posCappedIds = 0, h1 = 0, h2 = 0, h10 = 0, h100 = 0;
			for (auto& kv : g_geomNorm) {
				const GeomProbeStat& s = kv.second;
				if (s.posCapped || s.posSet.size() >= 2) multiPos++;
				if (s.posCapped) posCappedIds++;
				if (s.seen >= 100) h100++; else if (s.seen >= 10) h10++; else if (s.seen >= 2) h2++; else h1++;
			}
			char buf[360];
			sprintf_s(buf, sizeof(buf), "[GEOMPROBE] SUMMARY draws=%llu normIds=%zu rawIds=%zu fullIds=%zu viPairs=%zu hist[1=%zu,2-9=%zu,10-99=%zu,100+=%zu] multiPosNorm=%zu posCapped=%zu ovf[N=%d,R=%d,F=%d]",
				(unsigned long long)g_geomProbeDraws, g_geomNorm.size(), g_geomRawSet.size(), g_geomFullSet.size(), g_geomVISet.size(),
				h1, h2, h10, h100, multiPos, posCappedIds, g_ovfNorm ? 1 : 0, g_ovfRaw ? 1 : 0, g_ovfFull ? 1 : 0);
			Logging::Log() << buf;
		}
	}
}
// =========================================================================================

// ===================== [DRAWCALLER] locate DK2's draw call sites (RE for the LOD routine) =====
// The game's LOD selection runs immediately before it calls our DrawIndexedPrimitive. Manual
// stack scan (FPO-proof, no frame pointers needed): the first stack dword that lands in DKII.exe
// .text (0x401000-0x64D431) AND is immediately preceded by a CALL = the game's draw call site.
// Disassemble backward from there to find LOD selection. Read-only.
namespace {
	std::unordered_map<DWORD, DWORD> g_drawCallers, g_dcIcMin, g_dcIcMax, g_dcVcMin, g_dcVcMax;
	unsigned long long g_drawCallerN = 0;
	unsigned long g_drawCallerNew = 0;
	const DWORD kTextLo = 0x00401000, kTextHi = 0x0064D431;	// DKII.exe .text

	inline DWORD GameDrawCaller() {
		DWORD* sp = (DWORD*)_AddressOfReturnAddress();
		for (int i = 0; i < 600; i++) {
			DWORD a = sp[i];
			if (a > kTextLo + 8 && a <= kTextHi) {
				const BYTE* p = (const BYTE*)a;
				if (p[-5] == 0xE8) return a;								// call rel32
				if (p[-2] == 0xFF && ((p[-1] >> 3) & 7) == 2) return a;		// call reg / [reg]   (FF /2, 2-byte)
				if (p[-3] == 0xFF && ((p[-2] >> 3) & 7) == 2) return a;		// call [reg+disp8]   (3-byte)
				if (p[-6] == 0xFF && ((p[-5] >> 3) & 7) == 2) return a;		// call [reg+disp32]  (6-byte)
			}
		}
		return 0;
	}
	void DrawCallerObserve(DWORD ic, DWORD vc) {
		DWORD a = GameDrawCaller();
		if (!a) return;
		g_drawCallerN++;
		auto it = g_drawCallers.find(a);
		if (it == g_drawCallers.end()) {
			g_drawCallers[a] = 1; g_dcIcMin[a] = ic; g_dcIcMax[a] = ic; g_dcVcMin[a] = vc; g_dcVcMax[a] = vc;
			if (g_drawCallerNew < 300) { g_drawCallerNew++; char b[96]; sprintf_s(b, sizeof(b), "[DRAWCALLER] NEW site=%08lX ic=%lu vc=%lu", (unsigned long)a, (unsigned long)ic, (unsigned long)vc); Logging::Log() << b; }
		} else {
			it->second++;
			if (ic < g_dcIcMin[a]) g_dcIcMin[a] = ic; if (ic > g_dcIcMax[a]) g_dcIcMax[a] = ic;
			if (vc < g_dcVcMin[a]) g_dcVcMin[a] = vc; if (vc > g_dcVcMax[a]) g_dcVcMax[a] = vc;
		}
		if ((g_drawCallerN % 50000ULL) == 0) {
			for (auto& kv : g_drawCallers) { char b[180]; sprintf_s(b, sizeof(b), "[DRAWCALLER] SUMMARY site=%08lX count=%lu ic=%lu-%lu vc=%lu-%lu", (unsigned long)kv.first, (unsigned long)kv.second, (unsigned long)g_dcIcMin[kv.first], (unsigned long)g_dcIcMax[kv.first], (unsigned long)g_dcVcMin[kv.first], (unsigned long)g_dcVcMax[kv.first]); Logging::Log() << b; }
		}
	}
}
// =============================================================================================

HRESULT m_IDirect3DDeviceX::DrawIndexedPrimitive(D3DPRIMITIVETYPE dptPrimitiveType, DWORD dwVertexTypeDesc, LPVOID lpVertices, DWORD dwVertexCount, LPWORD lpwIndices, DWORD dwIndexCount, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" VertexType = " << Logging::hex(dptPrimitiveType) <<
		" VertexDesc = " << Logging::hex(dwVertexTypeDesc) <<
		" Vertices = " << lpVertices <<
		" VertexCount = " << dwVertexCount <<
		" Indices = " << lpwIndices <<
		" IndexCount = " << dwIndexCount <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Version = " << DirectXVersion;

	if (Config.Dd7to9)
	{
		if (dwVertexCount == 0 || dwIndexCount == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpVertices || !lpwIndices)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (Config.DdrawDrawCallerLog) DrawCallerObserve(dwIndexCount, dwVertexCount);

		if (DirectXVersion == 2)
		{
			if (dwVertexTypeDesc != D3DVT_VERTEX && dwVertexTypeDesc != D3DVT_LVERTEX && dwVertexTypeDesc != D3DVT_TLVERTEX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: invalid Vertex type: " << dwVertexTypeDesc);
				return D3DERR_INVALIDVERTEXTYPE;
			}
			dwVertexTypeDesc = ConvertVertexTypeToFVF((D3DVERTEXTYPE)dwVertexTypeDesc);
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection(), !Config.DdrawNoDrawBufferSysLock);

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		// Update vertices for Direct3D9 (needs to be first)
		UpdateVertices(dwVertexTypeDesc, lpVertices, 0, dwVertexCount);

		// === UI detection via RHW-constancy heuristic (see DrawPrimitive for rationale) ===
		bool isUIPassthrough = false;
		float uiHeuristicMinRHW = 0.0f, uiHeuristicMaxRHW = 0.0f;
		// Phase A.10 atlas decomposition state -- populated inside the XYZRHW block,
		// consumed at the d3d9 draw call sites below. uvOffBase is the byte offset
		// of stage-0 UV in the ORIGINAL (XYZRHW) vertex; converted-buffer offset is
		// uvOffBase-4 because the conversion drops RHW.
		IDirect3DTexture9* atlasDecomposeSubTex = nullptr;
		float adRegU0 = 0.0f, adRegV0 = 0.0f, adRegDu = 1.0f, adRegDv = 1.0f;
		int adRegIdx = -1;
		UINT atlasDecomposeUVOffBase = 0;
		bool dynUiEmissiveDraw = false;
		bool waterFlattenDraw = false;
		bool orphanOverlayDraw = false;
		if ((dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW && dwVertexCount > 0 && lpVertices)
		{
			const UINT scanStride = GetVertexStride(dwVertexTypeDesc);
			const UINT8* scanVertex = (const UINT8*)lpVertices;
			uiHeuristicMinRHW = uiHeuristicMaxRHW = ((const float*)scanVertex)[3];
			for (UINT i = 1; i < dwVertexCount; i++)
			{
				scanVertex += scanStride;
				const float rhw = ((const float*)scanVertex)[3];
				if (rhw < uiHeuristicMinRHW) uiHeuristicMinRHW = rhw;
				if (rhw > uiHeuristicMaxRHW) uiHeuristicMaxRHW = rhw;
			}
			const float denom = (uiHeuristicMaxRHW > 0.0f) ? uiHeuristicMaxRHW : 1.0f;
			const bool isConstantRHW = ((uiHeuristicMaxRHW - uiHeuristicMinRHW) / denom) < 1e-4f;
			isUIPassthrough = isConstantRHW;
			// [may20-minus-mip] DISABLED: the UI-passthrough heuristic classified
			// constant-RHW torch billboards as UI -> they skipped the world inverse
			// projection -> torches vanish by angle/distance under Remix. OG (pre-May-2)
			// projected ALL XYZRHW draws. Force false to restore that baseline.
			isUIPassthrough = false;
			orphanOverlayDraw = Config.DdrawOrphanOverlayLift && isConstantRHW;

			// Phase A.8: same UV-bounds scan as DrawPrimitive site (see comment there)
			float u_min = 0, u_max = 0, v_min = 0, v_max = 0;
			bool has_uv = false;
			const DWORD numTexCoords = (dwVertexTypeDesc & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
			if (numTexCoords > 0)
			{
				UINT uvOff = 16;
				if (dwVertexTypeDesc & D3DFVF_RESERVED1) uvOff += 4;
				if (dwVertexTypeDesc & D3DFVF_DIFFUSE) uvOff += 4;
				if (dwVertexTypeDesc & D3DFVF_SPECULAR) uvOff += 4;
				const UINT8* vbase = (const UINT8*)lpVertices;
				const float* uv0 = (const float*)(vbase + uvOff);
				u_min = u_max = uv0[0];
				v_min = v_max = uv0[1];
				for (UINT i = 1; i < dwVertexCount; i++)
				{
					const float* uv = (const float*)(vbase + i * scanStride + uvOff);
					if (uv[0] < u_min) u_min = uv[0];
					if (uv[0] > u_max) u_max = uv[0];
					if (uv[1] < v_min) v_min = uv[1];
					if (uv[1] > v_max) v_max = uv[1];
				}
				has_uv = true;
				atlasDecomposeUVOffBase = uvOff;
			}

			void* uiTexPtr = (void*)CurrentTextureSurfaceX[0];
			LOG_LIMIT(20000, "DRAW_XYZRHW DrawIndexedPrimitive ui=" << (isUIPassthrough ? 1 : 0)
				<< " vcount=" << dwVertexCount << " icount=" << dwIndexCount
				<< " rhw=[" << uiHeuristicMinRHW << "," << uiHeuristicMaxRHW << "]"
				<< " tex=" << uiTexPtr
				<< (has_uv ? (std::string(" uv=[") + std::to_string(u_min) + "," + std::to_string(v_min) + "," + std::to_string(u_max) + "," + std::to_string(v_max) + "]").c_str() : ""));

			// [GEOMPROBE] read-only geometry-identity stability probe (flag-gated). Hash
			// indices+UVs (position-independent) and track distinct position-sets per id.
			if (Config.DdrawGeomProbe)
			{
				GeomProbeObserve(lpVertices, dwVertexCount, lpwIndices, dwIndexCount,
					scanStride, has_uv, atlasDecomposeUVOffBase, u_min, v_min, u_max, v_max);
			}

			// Dynamic-UI emissive: clear the per-draw classification before the
			// decompose probes (TryGetUniversalSubTextureForUV sets it).
			NamekeyResetDrawDynUi();

			// Phase A.10: probe the atlas-region map for this drawcall. Match by
			// content hash (stable across sessions) + UV bbox containment (with
			// rasterizer-rounding slack). On a hit, the synthesized sub-texture
			// will be bound after SetDrawStates and the vertex UVs will be
			// rewritten to [0,1] over the sub-texture rect.
			if (Config.DdrawAtlasDecompose && has_uv && CurrentTextureSurfaceX[0])
			{
				CurrentTextureSurfaceX[0]->TryGetSubTextureForUV(u_min, v_min, u_max, v_max,
					atlasDecomposeSubTex, adRegU0, adRegV0, adRegDu, adRegDv, adRegIdx);
			}

			// Universal UV-region decomposition (2026-05-21): for ANY drawcall whose
			// stage-0 UV bbox is a PROPER sub-rectangle of its bound texture, crop that
			// exact region into a content-hash-keyed sub-texture and bind it. Full-texture
			// draws (dirt walls, UV ~ [0,1]) fail the area gate and pass through untouched,
			// which structurally eliminates both the wall over-match AND the co-atlas flash
			// (the torch flame then samples ONLY the flame region, never flame+HUD). No
			// fingerprints, no hardcoded hashes. Reuses the A.10 UV-rewrite + SetTexture
			// override below. Mutually exclusive with the A.10 path above (only fires if
			// A.10 didn't already match). See dk2_universal_decompose_plan memory.
			constexpr float kUnivMinArea = 1e-5f;   // reject degenerate/zero-area UV bboxes
			constexpr float kUnivMaxArea = 0.5f;     // only decompose proper sub-rects (tune)
			// VERTEX-COUNT GATE (2026-05-21): only decompose small sprite/billboard-like
			// draws. Co-atlased torches are tiny meshes (~12 verts). A COMPLEX world mesh
			// (115 verts) whose UV bbox merely happens to be a sub-rect is NOT a uniform
			// quad -- rewriting all its UVs to [0,1] over the bbox + binding a cropped NPOT
			// sub-texture produced geometry that CRASHED the Remix server (0xC0000005 AV in
			// its geometry/BVH path; vcount=115 draw, see dk2_universal_decompose_plan).
			// 32 sits well above the 12-vert torch and well below the 115-vert mesh.
			constexpr UINT kUnivMaxVertexCount = 32;
			// Gate-free water classification (run 11): large water bodies are 34-340
			// vert meshes whose UV bbox sits inside ONE Water placement -- ask the
			// page recipe directly, before the gates. Drives the flatten always; with
			// DdrawWaterCropLarge it also lets these draws through the vcount gate so
			// they get the collapsed translucent crop instead of the whole page (the
			// crops here are POT 32x32 and the UVs verified-contained, unlike the
			// NPOT case in the historic 115-vert Remix AV).
			const bool waterRecipeDraw = Config.DdrawWaterFlatten && has_uv && CurrentTextureSurfaceX[0] &&
				CurrentTextureSurfaceX[0]->IsNamekeyWaterDraw(u_min, v_min, u_max, v_max);
			if (!atlasDecomposeSubTex && Config.DdrawUniversalDecompose && has_uv && CurrentTextureSurfaceX[0]
				&& (dwVertexCount <= kUnivMaxVertexCount || (waterRecipeDraw && Config.DdrawWaterCropLarge)))
			{
				const float uvArea = (u_max - u_min) * (v_max - v_min);
				if (uvArea > kUnivMinArea && uvArea < kUnivMaxArea)
				{
					CurrentTextureSurfaceX[0]->TryGetUniversalSubTextureForUV(u_min, v_min, u_max, v_max,
						atlasDecomposeSubTex, adRegU0, adRegV0, adRegDu, adRegDv, adRegIdx);
				}
			}

			// Dynamic-UI emissive: placement-level flag (set during the probe above)
			// OR recipe-level solo page (minimap-class full-page draws that never
			// enter the crop layer).
			dynUiEmissiveDraw = Config.DdrawDynamicUiEmissive &&
				(NamekeyGetDrawDynUi() || (CurrentTextureSurfaceX[0] && CurrentTextureSurfaceX[0]->IsNamekeyDynamicUiPage()));

			// Orphan-overlay lift (2026-06-12, menu-text fix) -- see the DrawPrimitive
			// site for the full rationale: constant-RHW + recipeless page = runtime-
			// composited overlay (menu text); lift to near depth + draw additive.
			orphanOverlayDraw = orphanOverlayDraw &&
				((CurrentTextureSurfaceX[0] && CurrentTextureSurfaceX[0]->IsNamekeyOrphanPage()) || MenuBlitOverlayDrawActive());
			if (orphanOverlayDraw) dynUiEmissiveDraw = true;

			// [LIFT] telemetry v2 -- see the DrawPrimitive site.
			if (Config.DdrawOrphanOverlayLift && isConstantRHW)
			{
				char recipeBrief[256];
				if (CurrentTextureSurfaceX[0]) CurrentTextureSurfaceX[0]->GetNamekeyRecipeBrief(recipeBrief, sizeof(recipeBrief));
				else strcpy_s(recipeBrief, sizeof(recipeBrief), "notex");
				LOG_LIMIT(100000, "[LIFT] DrawIndexedPrimitive " << (orphanOverlayDraw ? "LIFTED" : "skip")
					<< " tex=" << (void*)CurrentTextureSurfaceX[0]
					<< " recipe=" << recipeBrief
					<< " vcount=" << dwVertexCount
					<< " rhw=" << uiHeuristicMinRHW
					<< " srcz=" << ((const float*)lpVertices)[2]
					<< " liftz=" << g_orphanLiftZ);
			}

			// Water flatten: this draw samples a WaterN placement -> snap its world
			// heights after the inverse projection below (the game waves the verts).
			waterFlattenDraw = Config.DdrawWaterFlatten && (waterRecipeDraw || NamekeyGetDrawWater());
		}

		// Handle PositionT
		// UI passthrough draws skip this branch entirely so they emit as XYZRHW screen-space.
		if (Config.DdrawConvertHomogeneousW && (dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW && !isUIPassthrough)
		{
			static bool loggedDrawIndexedConversion = false;
			if (!loggedDrawIndexedConversion)
			{
				Logging::Log() << __FUNCTION__ << " Converting XYZRHW to world space (DrawIndexedPrimitive)";
				Logging::Log() << "  Projection matrix [0][0]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._11
					<< " [1][1]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._22
					<< " [2][2]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._33
					<< " [3][3]=" << ConvertHomogeneous.ToWorld_ProjectionMatrix._44;
				Logging::Log() << "  View matrix [3][0]=" << ConvertHomogeneous.ToWorld_ViewMatrix._41
					<< " [3][1]=" << ConvertHomogeneous.ToWorld_ViewMatrix._42
					<< " [3][2]=" << ConvertHomogeneous.ToWorld_ViewMatrix._43;
				loggedDrawIndexedConversion = true;
			}

			if (!ConvertHomogeneous.IsTransformViewSet)
			{
				D3DMATRIX Matrix = {};
				GetTransform(D3DTS_VIEW, &Matrix);
				SetTransform(D3DTS_VIEW, &Matrix);
			}

			if (!Config.DdrawConvertHomogeneousToWorld)
			{
				/*UINT8 *vertex = (UINT8*)lpVertices;
				for (UINT x = 0; x < dwVertexCount; x++)
				{
				  float *pos = (float*) vertex;
				  pos[3] = 1.0f;
				  vertex += stride;
				}*/

				// Update the FVF
				dwVertexTypeDesc = (dwVertexTypeDesc & ~D3DFVF_XYZRHW) | D3DFVF_XYZW;
			}
			else
			{
				const UINT stride = GetVertexStride(dwVertexTypeDesc);

				const UINT targetStride = stride - sizeof(float);
				const UINT restSize = stride - sizeof(float) * 4;

				ConvertHomogeneous.ToWorld_IntermediateGeometry.resize(targetStride * dwVertexCount);

				UINT8* sourceVertex = (UINT8*)lpVertices;
				UINT8* targetVertex = (UINT8*)ConvertHomogeneous.ToWorld_IntermediateGeometry.data();

				lpVertices = targetVertex;

				for (UINT x = 0; x < dwVertexCount; x++)
				{
					// Transform the vertices into world space
					float* srcpos = (float*)sourceVertex;
					float* trgtpos = (float*)targetVertex;

					DirectX::XMVECTOR xpos = DirectX::XMVectorSet(srcpos[0], srcpos[1],
						orphanOverlayDraw ? g_orphanLiftZ : srcpos[2], srcpos[3]);

					DirectX::XMVECTOR xpos_global = DirectX::XMVector3TransformCoord(xpos, ConvertHomogeneous.ToWorld_ViewMatrixInverse);

					xpos_global = DirectX::XMVectorDivide(xpos_global, DirectX::XMVectorSplatW(xpos_global));

					trgtpos[0] = DirectX::XMVectorGetX(xpos_global);
					trgtpos[1] = DirectX::XMVectorGetY(xpos_global);
					trgtpos[2] = DirectX::XMVectorGetZ(xpos_global);

					// Copy the rest
					std::memcpy(targetVertex + sizeof(float) * 3, sourceVertex + sizeof(float) * 4, restSize);

					// Move to next vertex
					sourceVertex += stride;
					targetVertex += targetStride;
				}

				// Painter's order for lifted overlays: the next lifted draw sits
				// slightly nearer so later draws (text over panels) resolve on top.
				if (orphanOverlayDraw && g_orphanLiftZ > 0.01f) g_orphanLiftZ -= 1e-5f;

				// Water flatten (2026-06-11, surface-fit era -- see the block comment
				// at the top of this file): the game waves the water verts every
				// frame; under translucent path-traced water the bobbing churns
				// refraction/reflection. Feed this draw's RAW world verts to the
				// frame accumulator (the fit must see the live sine, not our own
				// output), then snap each vert to LAST frame's fitted surface
				// evaluated at its own (x,z). Verts beyond the snap band (shore
				// skirts, stray geometry) keep their height. Cold start (first
				// water frame after a 5s gap): per-draw average for one frame.
				if (waterFlattenDraw && dwVertexCount >= 3)
				{
					UINT8* tv = (UINT8*)ConvertHomogeneous.ToWorld_IntermediateGeometry.data();
					float ySum = 0.0f;
					float yMin = ((float*)tv)[1], yMax = yMin;
					for (UINT vi = 0; vi < dwVertexCount; vi++)
					{
						const float y = ((float*)(tv + vi * targetStride))[1];
						ySum += y;
						if (y < yMin) yMin = y;
						if (y > yMax) yMax = y;
					}
					const DWORD nowWaterMs = GetTickCount();
					if (nowWaterMs - g_waterLastDrawMs > 5000)
					{
						g_waterSurfValid = false;	// loading gap / map change: stale surface
						g_waterSurfEpochSet = false;
						g_waterSurfAgeFrames = 0;
						g_waterFrameXYZ.clear();
						g_waterFrameDraws = 0;
						g_waterVertsTotal = 0;
						g_waterVertsSnapped = 0;
					}
					g_waterLastDrawMs = nowWaterMs;
					if (g_waterFrameXYZ.empty()) { g_waterFrameYMin = yMin; g_waterFrameYMax = yMax; }
					else
					{
						if (yMin < g_waterFrameYMin) g_waterFrameYMin = yMin;
						if (yMax > g_waterFrameYMax) g_waterFrameYMax = yMax;
					}
					for (UINT vi = 0; vi < dwVertexCount && g_waterFrameXYZ.size() < kWaterFrameVertCap * 3; vi++)
					{
						const float* pos = (const float*)(tv + vi * targetStride);
						g_waterFrameXYZ.push_back(pos[0]);
						g_waterFrameXYZ.push_back(pos[1]);
						g_waterFrameXYZ.push_back(pos[2]);
					}
					g_waterFrameDraws++;
					g_waterVertsTotal += dwVertexCount;
					if (g_waterSurfValid)
					{
						for (UINT vi = 0; vi < dwVertexCount; vi++)
						{
							float* pos = (float*)(tv + vi * targetStride);
							const float s = WaterSurfEval(pos[0], pos[2]);
							if (fabsf(pos[1] - s) <= g_waterSurfBand) { pos[1] = s; g_waterVertsSnapped++; }
						}
					}
					else
					{
						const float yAvg = ySum / (float)dwVertexCount;
						for (UINT vi = 0; vi < dwVertexCount; vi++)
						{
							((float*)(tv + vi * targetStride))[1] = yAvg;
						}
						g_waterVertsSnapped += dwVertexCount;
					}
					NamekeyNoteWaterFlatten();
				}

				// Set transform
				(*d3d9Device)->SetTransform(D3DTS_VIEW, &ConvertHomogeneous.ToWorld_ViewMatrix);
				(*d3d9Device)->SetTransform(D3DTS_PROJECTION, &ConvertHomogeneous.ToWorld_ProjectionMatrix);

				// Update the FVF
				const DWORD newVertexTypeDesc = (dwVertexTypeDesc & ~D3DFVF_XYZRHW) | D3DFVF_XYZ;

				// Set fixed function vertex type
				if (FAILED((*d3d9Device)->SetFVF(newVertexTypeDesc)))
				{
					LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
					return D3DERR_INVALIDVERTEXTYPE;
				}

				// Phase A.10: rewrite UVs in the converted-buffer to [0,1] over the
				// matched sub-region. RHW was dropped (XYZRHW -> XYZ), so the UV
				// offset is uvOffBase - 4 here.
				if (atlasDecomposeSubTex && atlasDecomposeUVOffBase >= 4 && adRegDu > 0.0f && adRegDv > 0.0f)
				{
					const UINT uvOffConv = atlasDecomposeUVOffBase - 4;
					UINT8* tv = (UINT8*)ConvertHomogeneous.ToWorld_IntermediateGeometry.data();
					for (UINT vi = 0; vi < dwVertexCount; vi++)
					{
						float* uv = (float*)(tv + vi * targetStride + uvOffConv);
						uv[0] = (uv[0] - adRegU0) / adRegDu;
						uv[1] = (uv[1] - adRegV0) / adRegDv;
					}
				}

				// Handle dwFlags
				SetDrawStates(newVertexTypeDesc, dwFlags, DirectXVersion);

				// Phase A.10: override the atlas binding with the synthesized
				// sub-texture. Must come AFTER SetDrawStates (which is what does
				// the original SetTexture).
				if (atlasDecomposeSubTex)
				{
					(*d3d9Device)->SetTexture(0, atlasDecomposeSubTex);
					Stage3OnRemixBind(atlasDecomposeSubTex, /*isCrop=*/true);
				}

				// Draw indexed primitive UP (dynamic-UI draws get the additive/emissive override)
				if (dynUiEmissiveDraw) DynUiBeginDraw(*d3d9Device);
				HRESULT hr = (*d3d9Device)->DrawIndexedPrimitiveUP(dptPrimitiveType, 0, dwVertexCount, GetNumberOfPrimitives(dptPrimitiveType, dwIndexCount), lpwIndices, D3DFMT_INDEX16, lpVertices, targetStride);
				DynUiEndDraw(*d3d9Device);

				// Handle dwFlags
				RestoreDrawStates(hr, newVertexTypeDesc, dwFlags);

				// NOTE: Do NOT restore transform - keep 3D camera matrices set for RTX Remix detection
				// The original code restored the matrices here which prevented RTX Remix from detecting the camera

				return hr;
			}
		}

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(dwVertexTypeDesc)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
			return DDERR_INVALIDPARAMS;
		}

		// Phase A.10: for the non-conversion path, lpVertices still points at
		// user memory -- copy to a scratch buffer before rewriting UVs.
		// Single-threaded so a static buffer is safe (matches the pattern of
		// ToWorld_IntermediateGeometry above).
		static thread_local std::vector<uint8_t> atlasDecomposeScratch;
		if (atlasDecomposeSubTex && atlasDecomposeUVOffBase >= 16 && adRegDu > 0.0f && adRegDv > 0.0f)
		{
			const UINT stride = GetVertexStride(dwVertexTypeDesc);
			const UINT total = stride * dwVertexCount;
			atlasDecomposeScratch.resize(total);
			memcpy(atlasDecomposeScratch.data(), lpVertices, total);
			UINT8* tv = atlasDecomposeScratch.data();
			for (UINT vi = 0; vi < dwVertexCount; vi++)
			{
				float* uv = (float*)(tv + vi * stride + atlasDecomposeUVOffBase);
				uv[0] = (uv[0] - adRegU0) / adRegDu;
				uv[1] = (uv[1] - adRegV0) / adRegDv;
			}
			lpVertices = tv;
		}

		// Handle dwFlags
		SetDrawStates(dwVertexTypeDesc, dwFlags, DirectXVersion);

		// Phase A.10: override atlas bind with the synthesized sub-texture.
		if (atlasDecomposeSubTex)
		{
			(*d3d9Device)->SetTexture(0, atlasDecomposeSubTex);
			Stage3OnRemixBind(atlasDecomposeSubTex, /*isCrop=*/true);
		}

		// Draw indexed primitive UP (dynamic-UI draws get the additive/emissive override)
		if (dynUiEmissiveDraw) DynUiBeginDraw(*d3d9Device);
		HRESULT hr = (*d3d9Device)->DrawIndexedPrimitiveUP(dptPrimitiveType, 0, dwVertexCount, GetNumberOfPrimitives(dptPrimitiveType, dwIndexCount), lpwIndices, D3DFMT_INDEX16, lpVertices, GetVertexStride(dwVertexTypeDesc));
		DynUiEndDraw(*d3d9Device);

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawIndexedPrimitiveUP' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->DrawIndexedPrimitive(dptPrimitiveType, (D3DVERTEXTYPE)dwVertexTypeDesc, lpVertices, dwVertexCount, lpwIndices, dwIndexCount, dwFlags);
	case 3:
		return GetProxyInterfaceV3()->DrawIndexedPrimitive(dptPrimitiveType, dwVertexTypeDesc, lpVertices, dwVertexCount, lpwIndices, dwIndexCount, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawIndexedPrimitive(dptPrimitiveType, dwVertexTypeDesc, lpVertices, dwVertexCount, lpwIndices, dwIndexCount, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::SetClipStatus(LPD3DCLIPSTATUS lpD3DClipStatus)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DClipStatus || !(lpD3DClipStatus->dwFlags & (D3DCLIPSTATUS_STATUS | D3DCLIPSTATUS_EXTENTS2 | D3DCLIPSTATUS_EXTENTS3)))
		{
			return DDERR_INVALIDPARAMS;
		}

		LOG_LIMIT(100, __FUNCTION__ << " Warning: clip status is not fully supported.");

		D3DClipStatus.dwFlags = lpD3DClipStatus->dwFlags & (D3DCLIPSTATUS_STATUS | D3DCLIPSTATUS_EXTENTS2 | D3DCLIPSTATUS_EXTENTS3);

		// Status
		if (lpD3DClipStatus->dwFlags & D3DCLIPSTATUS_STATUS)
		{
			D3DClipStatus.dwStatus = lpD3DClipStatus->dwStatus;
		}

		// Extents 2-D
		if (lpD3DClipStatus->dwFlags & (D3DCLIPSTATUS_EXTENTS2 | D3DCLIPSTATUS_EXTENTS3))
		{
			D3DClipStatus.minx = lpD3DClipStatus->minx;
			D3DClipStatus.maxx = lpD3DClipStatus->maxx;
			D3DClipStatus.miny = lpD3DClipStatus->miny;
			D3DClipStatus.maxy = lpD3DClipStatus->maxy;
		}

		// Extents 3-D
		if (lpD3DClipStatus->dwFlags & D3DCLIPSTATUS_EXTENTS3)
		{
			D3DClipStatus.minz = lpD3DClipStatus->minz;
			D3DClipStatus.maxz = lpD3DClipStatus->maxz;
		}

		return D3D_OK;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->SetClipStatus(lpD3DClipStatus);
	case 3:
		return GetProxyInterfaceV3()->SetClipStatus(lpD3DClipStatus);
	case 7:
		return GetProxyInterfaceV7()->SetClipStatus(lpD3DClipStatus);
	}
}

HRESULT m_IDirect3DDeviceX::GetClipStatus(LPD3DCLIPSTATUS lpD3DClipStatus)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DClipStatus)
		{
			return DDERR_INVALIDPARAMS;
		}

		LOG_LIMIT(100, __FUNCTION__ << " Warning: clip status is not fully supported.");

		*lpD3DClipStatus = D3DClipStatus;

		return D3D_OK;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	default:
		return DDERR_GENERIC;
	case 2:
		return GetProxyInterfaceV2()->GetClipStatus(lpD3DClipStatus);
	case 3:
		return GetProxyInterfaceV3()->GetClipStatus(lpD3DClipStatus);
	case 7:
		return GetProxyInterfaceV7()->GetClipStatus(lpD3DClipStatus);
	}
}

// ******************************
// IDirect3DDevice v3 functions
// ******************************

HRESULT m_IDirect3DDeviceX::DrawPrimitiveStrided(D3DPRIMITIVETYPE dptPrimitiveType, DWORD dwVertexTypeDesc, LPD3DDRAWPRIMITIVESTRIDEDDATA lpVertexArray, DWORD dwVertexCount, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwVertexCount == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpVertexArray)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (DirectXVersion == 2)
		{
			if (dwVertexTypeDesc != D3DVT_VERTEX && dwVertexTypeDesc != D3DVT_LVERTEX && dwVertexTypeDesc != D3DVT_TLVERTEX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: invalid Vertex type: " << dwVertexTypeDesc);
				return D3DERR_INVALIDVERTEXTYPE;
			}
			dwVertexTypeDesc = ConvertVertexTypeToFVF((D3DVERTEXTYPE)dwVertexTypeDesc);
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection(), !Config.DdrawNoDrawBufferSysLock);

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		// Update vertex desc type (FVF) before interleaving
		dwVertexTypeDesc = dwVertexTypeDesc == D3DFVF_LVERTEX ? D3DFVF_LVERTEX9 : dwVertexTypeDesc;

		// Update vertices for Direct3D9 (needs to be first)
		if (!m_IDirect3DVertexBufferX::InterleaveStridedVertexData(VertexCache, lpVertexArray, 0, dwVertexCount, dwVertexTypeDesc))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid StridedVertexData!");
			return DDERR_INVALIDPARAMS;
		}

		// Menu-text fix (2026-06-12) -- same gap as the VB path: strided XYZRHW draws
		// bypassed the XYZRHW->world conversion. Funnel through the user-pointer path
		// (LOCAL copy: the UP path's UpdateVertices may itself write into VertexCache).
		if (Config.DdrawConvertHomogeneousW && (dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW)
		{
			std::vector<BYTE> stridedCopy(VertexCache.begin(), VertexCache.begin() + dwVertexCount * GetVertexStride(dwVertexTypeDesc));
			LOG_LIMIT(100, __FUNCTION__ << " rerouting XYZRHW strided draw through UP path (menu-text fix)");
			return DrawPrimitive(dptPrimitiveType, dwVertexTypeDesc, stridedCopy.data(), dwVertexCount, dwFlags, DirectXVersion);
		}

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(dwVertexTypeDesc)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
			return DDERR_INVALIDPARAMS;
		}

		// Handle dwFlags
		SetDrawStates(dwVertexTypeDesc, dwFlags, DirectXVersion);

		// Draw primitive UP
		HRESULT hr = (*d3d9Device)->DrawPrimitiveUP(dptPrimitiveType, GetNumberOfPrimitives(dptPrimitiveType, dwVertexCount), VertexCache.data(), GetVertexStride(dwVertexTypeDesc));

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawPrimitiveUP' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->DrawPrimitiveStrided(dptPrimitiveType, dwVertexTypeDesc, lpVertexArray, dwVertexCount, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawPrimitiveStrided(dptPrimitiveType, dwVertexTypeDesc, lpVertexArray, dwVertexCount, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::DrawIndexedPrimitiveStrided(D3DPRIMITIVETYPE dptPrimitiveType, DWORD dwVertexTypeDesc, LPD3DDRAWPRIMITIVESTRIDEDDATA lpVertexArray, DWORD dwVertexCount, LPWORD lpwIndices, DWORD dwIndexCount, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwVertexCount == 0 || dwIndexCount == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpVertexArray || !lpwIndices)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (DirectXVersion == 2)
		{
			if (dwVertexTypeDesc != D3DVT_VERTEX && dwVertexTypeDesc != D3DVT_LVERTEX && dwVertexTypeDesc != D3DVT_TLVERTEX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: invalid Vertex type: " << dwVertexTypeDesc);
				return D3DERR_INVALIDVERTEXTYPE;
			}
			dwVertexTypeDesc = ConvertVertexTypeToFVF((D3DVERTEXTYPE)dwVertexTypeDesc);
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		// Update vertex desc type (FVF) before interleaving
		dwVertexTypeDesc = dwVertexTypeDesc == D3DFVF_LVERTEX ? D3DFVF_LVERTEX9 : dwVertexTypeDesc;

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(dwVertexTypeDesc)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(dwVertexTypeDesc));
			return DDERR_INVALIDPARAMS;
		}

		// Update vertices for Direct3D9 (needs to be first)
		if (!m_IDirect3DVertexBufferX::InterleaveStridedVertexData(VertexCache, lpVertexArray, 0, dwVertexCount, dwVertexTypeDesc))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid StridedVertexData!");
			return DDERR_INVALIDPARAMS;
		}

		// Menu-text fix (2026-06-12) -- same gap as the VB path: strided XYZRHW draws
		// bypassed the XYZRHW->world conversion. Funnel through the user-pointer path
		// (LOCAL copy: the UP path's UpdateVertices may itself write into VertexCache).
		if (Config.DdrawConvertHomogeneousW && (dwVertexTypeDesc & 0x0E) == D3DFVF_XYZRHW)
		{
			std::vector<BYTE> stridedCopy(VertexCache.begin(), VertexCache.begin() + dwVertexCount * GetVertexStride(dwVertexTypeDesc));
			LOG_LIMIT(100, __FUNCTION__ << " rerouting XYZRHW strided draw through UP path (menu-text fix)");
			return DrawIndexedPrimitive(dptPrimitiveType, dwVertexTypeDesc, stridedCopy.data(), dwVertexCount, lpwIndices, dwIndexCount, dwFlags, DirectXVersion);
		}

		// Handle dwFlags
		SetDrawStates(dwVertexTypeDesc, dwFlags, DirectXVersion);

		// Draw indexed primitive UP
		HRESULT hr = (*d3d9Device)->DrawIndexedPrimitiveUP(dptPrimitiveType, 0, dwVertexCount, GetNumberOfPrimitives(dptPrimitiveType, dwIndexCount), lpwIndices, D3DFMT_INDEX16, VertexCache.data(), GetVertexStride(dwVertexTypeDesc));

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawIndexedPrimitiveUP' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->DrawIndexedPrimitiveStrided(dptPrimitiveType, dwVertexTypeDesc, lpVertexArray, dwVertexCount, lpwIndices, dwIndexCount, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawIndexedPrimitiveStrided(dptPrimitiveType, dwVertexTypeDesc, lpVertexArray, dwVertexCount, lpwIndices, dwIndexCount, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::DrawPrimitiveVB(D3DPRIMITIVETYPE dptPrimitiveType, LPDIRECT3DVERTEXBUFFER7 lpd3dVertexBuffer, DWORD dwStartVertex, DWORD dwNumVertices, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" VertexType = " << Logging::hex(dptPrimitiveType) <<
		" VertexBuffer = " << lpd3dVertexBuffer <<
		" StartVertex = " << dwStartVertex <<
		" NumVertices = " << dwNumVertices <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Version = " << DirectXVersion;

	if (Config.Dd7to9)
	{
		if (dwNumVertices == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpd3dVertexBuffer)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection(), !Config.DdrawNoDrawBufferSysLock);

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		m_IDirect3DVertexBufferX* pVertexBufferX = nullptr;
		lpd3dVertexBuffer->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pVertexBufferX);
		if (!pVertexBufferX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get vertex buffer wrapper!");
			return DDERR_GENERIC;
		}

		LPDIRECT3DVERTEXBUFFER9 d3d9VertexBuffer = pVertexBufferX->GetCurrentD9VertexBuffer();
		if (!d3d9VertexBuffer)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get d3d9 vertex buffer!");
			return DDERR_GENERIC;
		}

		DWORD FVF = pVertexBufferX->GetFVF9();

		// Menu-text fix (2026-06-12): XYZRHW VB draws bypassed the XYZRHW->world
		// conversion entirely and reached Remix as raw screen-space geometry --
		// invisible in the path-traced scene (the front-end menu text draws this
		// way; orthographicIsUI=True rasterized it, proving the content is fine).
		// Reroute through the user-pointer draw path, which applies the inverse
		// projection and ALL draw classification (orphan lift, decompose, telemetry).
		// The wrapper keeps a CPU-side copy of every VB, so this reads cached memory.
		if (Config.DdrawConvertHomogeneousW && (FVF & 0x0E) == D3DFVF_XYZRHW)
		{
			const BYTE* vbData = pVertexBufferX->GetCpuVertexData();
			const UINT vbStride = GetVertexStride(FVF);
			if (vbData && pVertexBufferX->GetCpuVertexDataSize() >= (dwStartVertex + dwNumVertices) * vbStride)
			{
				LOG_LIMIT(100, __FUNCTION__ << " rerouting XYZRHW VB draw through UP path (menu-text fix)");
				return DrawPrimitive(dptPrimitiveType, FVF,
					(LPVOID)(vbData + dwStartVertex * vbStride), dwNumVertices, dwFlags, DirectXVersion);
			}
		}

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(FVF)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(FVF));
			return DDERR_INVALIDPARAMS;
		}

		// Set stream source
		(*d3d9Device)->SetStreamSource(0, d3d9VertexBuffer, 0, GetVertexStride(FVF));

		// Handle dwFlags
		SetDrawStates(FVF, dwFlags, DirectXVersion);

		// Draw primitive
		HRESULT hr = (*d3d9Device)->DrawPrimitive(dptPrimitiveType, dwStartVertex, GetNumberOfPrimitives(dptPrimitiveType, dwNumVertices));

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawPrimitive' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	if (lpd3dVertexBuffer)
	{
		lpd3dVertexBuffer->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpd3dVertexBuffer);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->DrawPrimitiveVB(dptPrimitiveType, (LPDIRECT3DVERTEXBUFFER)lpd3dVertexBuffer, dwStartVertex, dwNumVertices, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawPrimitiveVB(dptPrimitiveType, lpd3dVertexBuffer, dwStartVertex, dwNumVertices, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::DrawIndexedPrimitiveVB(D3DPRIMITIVETYPE dptPrimitiveType, LPDIRECT3DVERTEXBUFFER7 lpd3dVertexBuffer, DWORD dwStartVertex, DWORD dwNumVertices, LPWORD lpwIndices, DWORD dwIndexCount, DWORD dwFlags, DWORD DirectXVersion)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")" <<
		" VertexType = " << Logging::hex(dptPrimitiveType) <<
		" VertexBuffer = " << lpd3dVertexBuffer <<
		" StartVertex = " << dwStartVertex <<
		" NumVertices = " << dwNumVertices <<
		" Indices = " << lpwIndices <<
		" IndexCount = " << dwIndexCount <<
		" Flags = " << Logging::hex(dwFlags) <<
		" Version = " << DirectXVersion;

	if (Config.Dd7to9)
	{
		if (dwNumVertices == 0 || dwIndexCount == 0)
		{
			return D3D_OK; // Nothing to draw
		}

		if (!lpd3dVertexBuffer || !lpwIndices)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection(), !Config.DdrawNoDrawBufferSysLock);

		dwFlags = (dwFlags & D3DDP_FORCE_DWORD);

		m_IDirect3DVertexBufferX* pVertexBufferX = nullptr;
		lpd3dVertexBuffer->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pVertexBufferX);
		if (!pVertexBufferX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get vertex buffer wrapper!");
			return DDERR_INVALIDPARAMS;
		}

		LPDIRECT3DVERTEXBUFFER9 d3d9VertexBuffer = pVertexBufferX->GetCurrentD9VertexBuffer();
		if (!d3d9VertexBuffer)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get d3d9 vertex buffer!");
			return DDERR_GENERIC;
		}

		DWORD FVF = pVertexBufferX->GetFVF9();

		// Menu-text fix (2026-06-12) -- see DrawPrimitiveVB: reroute XYZRHW VB draws
		// through the user-pointer path for the world conversion + classification.
		if (Config.DdrawConvertHomogeneousW && (FVF & 0x0E) == D3DFVF_XYZRHW)
		{
			const BYTE* vbData = pVertexBufferX->GetCpuVertexData();
			const UINT vbStride = GetVertexStride(FVF);
			if (vbData && pVertexBufferX->GetCpuVertexDataSize() >= (dwStartVertex + dwNumVertices) * vbStride)
			{
				LOG_LIMIT(100, __FUNCTION__ << " rerouting XYZRHW VB draw through UP path (menu-text fix)");
				return DrawIndexedPrimitive(dptPrimitiveType, FVF,
					(LPVOID)(vbData + dwStartVertex * vbStride), dwNumVertices,
					lpwIndices, dwIndexCount, dwFlags, DirectXVersion);
			}
		}

		// Set fixed function vertex type
		if (FAILED((*d3d9Device)->SetFVF(FVF)))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid FVF type: " << Logging::hex(FVF));
			return DDERR_INVALIDPARAMS;
		}

		LPDIRECT3DINDEXBUFFER9 d3d9IndexBuffer = ddrawParent->GetIndexBuffer(lpwIndices, dwIndexCount);
		if (!d3d9IndexBuffer)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get d3d9 index buffer!");
			return DDERR_GENERIC;
		}

		// Set stream source
		(*d3d9Device)->SetStreamSource(0, d3d9VertexBuffer, 0, GetVertexStride(FVF));

		// Set Index data
		(*d3d9Device)->SetIndices(d3d9IndexBuffer);

		// Handle dwFlags
		SetDrawStates(FVF, dwFlags, DirectXVersion);

		// Draw primitive
		HRESULT hr = (*d3d9Device)->DrawIndexedPrimitive(dptPrimitiveType, dwStartVertex, 0, dwNumVertices, 0, GetNumberOfPrimitives(dptPrimitiveType, dwIndexCount));

		// Handle dwFlags
		RestoreDrawStates(hr, dwFlags, DirectXVersion);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: 'DrawIndexedPrimitive' call failed: " << (D3DERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	if (lpd3dVertexBuffer)
	{
		lpd3dVertexBuffer->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpd3dVertexBuffer);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->DrawIndexedPrimitiveVB(dptPrimitiveType, (LPDIRECT3DVERTEXBUFFER)lpd3dVertexBuffer, lpwIndices, dwIndexCount, dwFlags);
	case 7:
		return GetProxyInterfaceV7()->DrawIndexedPrimitiveVB(dptPrimitiveType, lpd3dVertexBuffer, dwStartVertex, dwNumVertices, lpwIndices, dwIndexCount, dwFlags);
	}
}

HRESULT m_IDirect3DDeviceX::ComputeSphereVisibility(LPD3DVECTOR lpCenters, LPD3DVALUE lpRadii, DWORD dwNumSpheres, DWORD dwFlags, LPDWORD lpdwReturnValues)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpCenters || !lpRadii || !dwNumSpheres || !lpdwReturnValues)
		{
			return DDERR_INVALIDPARAMS;
		}

		LOG_LIMIT(100, __FUNCTION__ << " Warning: function not fully implemented");

		// Sphere visibility is computed by back-transforming the viewing frustum to the model space, using the inverse of the combined world, view, or projection matrices.
		// If the combined matrix can't be inverted (if the determinant is 0), the method will fail, returning D3DERR_INVALIDMATRIX.
		for (UINT x = 0; x < dwNumSpheres; x++)
		{
			// If a sphere is completely visible, the corresponding entry in lpdwReturnValues is 0.
			lpdwReturnValues[x] = 0;	// Just return all is visible for now
		}

		return D3D_OK;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->ComputeSphereVisibility(lpCenters, lpRadii, dwNumSpheres, dwFlags, lpdwReturnValues);
	case 7:
		return GetProxyInterfaceV7()->ComputeSphereVisibility(lpCenters, lpRadii, dwNumSpheres, dwFlags, lpdwReturnValues);
	}
}

HRESULT m_IDirect3DDeviceX::GetTexture(DWORD dwStage, LPDIRECT3DTEXTURE2* lplpTexture)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpTexture || dwStage >= D3DHAL_TSS_MAXSTAGES)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpTexture = nullptr;

		// Get surface stage
		ComPtr<IDirectDrawSurface7> pSurface;
		HRESULT hr = GetTexture(dwStage, pSurface.GetAddressOf());

		if (FAILED(hr))
		{
			return hr;
		}

		// Get surface wrapper
		m_IDirectDrawSurfaceX* pSurfaceX = nullptr;
		pSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pSurfaceX);
		if (!pSurfaceX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface wrapper!");
			return DDERR_INVALIDPARAMS;
		}

		// Get attached texture from surface
		m_IDirect3DTextureX* pTextureX = pSurfaceX->GetAttachedTexture();
		if (!pTextureX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture!");
			return DDERR_INVALIDPARAMS;
		}

		// Add ref to texture
		pTextureX->AddRef();

		*lplpTexture = (LPDIRECT3DTEXTURE2)pTextureX->GetWrapperInterfaceX(0);

		return D3D_OK;
	}

	HRESULT hr = GetProxyInterfaceV3()->GetTexture(dwStage, lplpTexture);

	if (SUCCEEDED(hr) && lplpTexture)
	{
		*lplpTexture = ProxyAddressLookupTable.FindAddress<m_IDirect3DTexture2>(*lplpTexture, 2);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::SetTexture(DWORD dwStage, LPDIRECT3DTEXTURE2 lpTexture)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwStage >= D3DHAL_TSS_MAXSTAGES)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (!lpTexture)
		{
			return SetTexture(dwStage, (LPDIRECTDRAWSURFACE7)nullptr);
		}

		m_IDirect3DTextureX* pTextureX = nullptr;
		lpTexture->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pTextureX);
		if (!pTextureX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture wrapper!");
			return DDERR_INVALIDPARAMS;
		}

		m_IDirectDrawSurfaceX* pSurfaceX = pTextureX->GetSurface();
		if (!pSurfaceX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface!");
			return DDERR_INVALIDPARAMS;
		}

		return SetTexture(dwStage, (LPDIRECTDRAWSURFACE7)pSurfaceX->GetWrapperInterfaceX(0));
	}

	if (lpTexture)
	{
		lpTexture->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpTexture);
	}

	return GetProxyInterfaceV3()->SetTexture(dwStage, lpTexture);
}

HRESULT m_IDirect3DDeviceX::GetTextureStageState(DWORD dwStage, D3DTEXTURESTAGESTATETYPE dwState, LPDWORD lpdwValue)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwValue)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			*lpdwValue = 0;
			return DDERR_INVALIDOBJECT;
		}

		if (dwStage >= D3DHAL_TSS_MAXSTAGES || dwState == 0
			|| (ClientDirectXVersion == 3 && dwState > 23)
			|| (ClientDirectXVersion == 7 && dwState > 24))
		{
			if (dwStage < D3DHAL_TSS_MAXSTAGES && (DWORD)dwState < MaxTextureStageStates)
			{
				*lpdwValue = DeviceStates.TextureStageState[dwStage][dwState].State;
			}
			else
			{
				*lpdwValue = (DWORD)-1;
			}
			return D3D_OK;
		}

		switch ((DWORD)dwState)
		{
		case D3DTSS_ADDRESS:
		{
			DWORD ValueU = 0, ValueV = 0;
			GetD9SamplerState(dwStage, D3DSAMP_ADDRESSU, &ValueU);
			GetD9SamplerState(dwStage, D3DSAMP_ADDRESSV, &ValueV);
			if (ValueU != ValueV)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: Texture Address U and V states don't match: " << ValueU << " and " << ValueV);
			}
			*lpdwValue = ValueU;
			return D3D_OK;
		}
		case D3DTSS_ADDRESSU:
			return GetD9SamplerState(dwStage, D3DSAMP_ADDRESSU, lpdwValue);
		case D3DTSS_ADDRESSV:
			return GetD9SamplerState(dwStage, D3DSAMP_ADDRESSV, lpdwValue);
		case D3DTSS_ADDRESSW:
			return GetD9SamplerState(dwStage, D3DSAMP_ADDRESSW, lpdwValue);
		case D3DTSS_BORDERCOLOR:
			return GetD9SamplerState(dwStage, D3DSAMP_BORDERCOLOR, lpdwValue);
		case D3DTSS_MAGFILTER:
			return GetD9SamplerState(dwStage, D3DSAMP_MAGFILTER, lpdwValue);
		case D3DTSS_MINFILTER:
			return GetD9SamplerState(dwStage, D3DSAMP_MINFILTER, lpdwValue);
		case D3DTSS_MIPFILTER:
			return GetD9SamplerState(dwStage, D3DSAMP_MIPFILTER, lpdwValue);
		case D3DTSS_MIPMAPLODBIAS:
			return GetD9SamplerState(dwStage, D3DSAMP_MIPMAPLODBIAS, lpdwValue);
		case D3DTSS_MAXMIPLEVEL:
			return GetD9SamplerState(dwStage, D3DSAMP_MAXMIPLEVEL, lpdwValue);
		case D3DTSS_MAXANISOTROPY:
			return GetD9SamplerState(dwStage, D3DSAMP_MAXANISOTROPY, lpdwValue);
		}

		if (!CheckTextureStageStateType(dwState))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Texture Stage state type not implemented: " << dwState);
		}

		return GetD9TextureStageState(dwStage, dwState, lpdwValue);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->GetTextureStageState(dwStage, dwState, lpdwValue);
	case 7:
		return GetProxyInterfaceV7()->GetTextureStageState(dwStage, dwState, lpdwValue);
	}
}

HRESULT m_IDirect3DDeviceX::SetTextureStageState(DWORD dwStage, D3DTEXTURESTAGESTATETYPE dwState, DWORD dwValue)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwStage >= D3DHAL_TSS_MAXSTAGES)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		if (dwStage >= D3DHAL_TSS_MAXSTAGES || dwState == 0
			|| (ClientDirectXVersion == 3 && dwState > 23)
			|| (ClientDirectXVersion == 7 && dwState > 24))
		{
			if (dwStage < D3DHAL_TSS_MAXSTAGES && (DWORD)dwState < MaxTextureStageStates)
			{
				DeviceStates.TextureStageState[dwStage][dwState].State = dwValue;
			}
			return D3D_OK;
		}

		switch ((DWORD)dwState)
		{
		case D3DTSS_ADDRESS:
			SetD9SamplerState(dwStage, D3DSAMP_ADDRESSU, dwValue);
			return SetD9SamplerState(dwStage, D3DSAMP_ADDRESSV, dwValue);
		case D3DTSS_ADDRESSU:
			return SetD9SamplerState(dwStage, D3DSAMP_ADDRESSU, dwValue);
		case D3DTSS_ADDRESSV:
			return SetD9SamplerState(dwStage, D3DSAMP_ADDRESSV, dwValue);
		case D3DTSS_ADDRESSW:
			return SetD9SamplerState(dwStage, D3DSAMP_ADDRESSW, dwValue);
		case D3DTSS_BORDERCOLOR:
			return SetD9SamplerState(dwStage, D3DSAMP_BORDERCOLOR, dwValue);
		case D3DTSS_MAGFILTER:
			return SetD9SamplerState(dwStage, D3DSAMP_MAGFILTER, dwValue);
		case D3DTSS_MINFILTER:
			return SetD9SamplerState(dwStage, D3DSAMP_MINFILTER, dwValue);
		case D3DTSS_MIPFILTER:
			return SetD9SamplerState(dwStage, D3DSAMP_MIPFILTER, dwValue);
		case D3DTSS_MIPMAPLODBIAS:
			return SetD9SamplerState(dwStage, D3DSAMP_MIPMAPLODBIAS, dwValue);
		case D3DTSS_MAXMIPLEVEL:
			return SetD9SamplerState(dwStage, D3DSAMP_MAXMIPLEVEL, dwValue);
		case D3DTSS_MAXANISOTROPY:
			return SetD9SamplerState(dwStage, D3DSAMP_MAXANISOTROPY, dwValue);
		}

		if (!CheckTextureStageStateType(dwState))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Texture Stage state type not implemented: " << dwState);
			return D3D_OK;	// Just return OK for now!
		}

		return SetD9TextureStageState(dwStage, dwState, dwValue);
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->SetTextureStageState(dwStage, dwState, dwValue);
	case 7:
		return GetProxyInterfaceV7()->SetTextureStageState(dwStage, dwState, dwValue);
	}
}

HRESULT m_IDirect3DDeviceX::ValidateDevice(LPDWORD lpdwPasses)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwPasses)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

#ifdef ENABLE_PROFILING
		auto startTime = std::chrono::high_resolution_clock::now();
#endif

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		PrepDevice();

		DWORD FVF, Size;
		IDirect3DVertexBuffer9* vertexBuffer = ddrawParent->GetValidateDeviceVertexBuffer(FVF, Size);

		if (!vertexBuffer)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to get vertex buffer!");
			return DDERR_GENERIC;
		}

		// Bind the vertex buffer to the device
		(*d3d9Device)->SetStreamSource(0, vertexBuffer, 0, Size);

		// Set a simple FVF (Flexible Vertex Format)
		(*d3d9Device)->SetFVF(FVF);

		// Call ValidateDevice
		HRESULT hr = (*d3d9Device)->ValidateDevice(lpdwPasses);

		if (FAILED(hr))
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: ValidateDevice() function failed: " << (DDERR)hr);
		}

#ifdef ENABLE_PROFILING
		Logging::Log() << __FUNCTION__ << " (" << this << ") hr = " << (D3DERR)hr << " Timing = " << Logging::GetTimeLapseInMS(startTime);
#endif

		return hr;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->ValidateDevice(lpdwPasses);
	case 7:
		return GetProxyInterfaceV7()->ValidateDevice(lpdwPasses);
	}
}

// ******************************
// IDirect3DDevice v7 functions
// ******************************

HRESULT m_IDirect3DDeviceX::GetCaps(LPD3DDEVICEDESC7 lpD3DDevDesc)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpD3DDevDesc)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		ConvertDeviceDesc(*lpD3DDevDesc, Caps9);

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->GetCaps(lpD3DDevDesc);
}

HRESULT m_IDirect3DDeviceX::EnumTextureFormats(LPD3DENUMPIXELFORMATSCALLBACK lpd3dEnumPixelProc, LPVOID lpArg)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpd3dEnumPixelProc)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, false)))
		{
			return DDERR_INVALIDOBJECT;
		}

		LPDIRECT3D9 d3d9Object = ddrawParent->GetDirectD9Object();

		if (!d3d9Object)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: failed to get d3d9 object!");
			return DDERR_GENERIC;
		}

		// Get texture list
		std::vector<D3DFORMAT> TextureList = {
			D3DFMT_R5G6B5,
			D3DFMT_X1R5G5B5,
			D3DFMT_A1R5G5B5,
			D3DFMT_A4R4G4B4,
			//D3DFMT_R8G8B8,	// Requires emulation
			D3DFMT_X8R8G8B8,
			D3DFMT_A8R8G8B8,
			D3DFMT_V8U8,
			D3DFMT_X8L8V8U8,
			D3DFMT_L6V5U5,
			D3DFMT_DXT1,
			D3DFMT_DXT2,
			D3DFMT_DXT3,
			D3DFMT_DXT4,
			D3DFMT_DXT5,
			D3DFMT_P8,
			D3DFMT_L8,
			D3DFMT_A8,
			D3DFMT_A4L4,
			D3DFMT_A8L8 };

		// If textures are being trimmed
		if (Config.DdrawLimitTextureFormats)
		{
			// Trim texture list
			std::vector<D3DFORMAT> TrimTextureList = {
				D3DFMT_V8U8,       // May be trimmed if normal maps are unused
				D3DFMT_X8L8V8U8,   // Rare normal map format
				D3DFMT_L6V5U5,     // Uncommon format
				D3DFMT_DXT5,       // Newer texture format
				D3DFMT_P8,         // 8-bit palettized (Direct3D9 deprecated this)
				D3DFMT_A4L4 };     // Rare grayscale+alpha format

			// Remove trimmed texture from list
			for (auto it = TextureList.begin(); it != TextureList.end(); )
			{
				if (std::find(TrimTextureList.begin(), TrimTextureList.end(), *it) != TrimTextureList.end())
				{
					it = TextureList.erase(it); // Remove and update iterator
				}
				else
				{
					++it; // Move to next element
				}
			}
		}
		// Add FourCCs to texture list
		else
		{
			for (D3DFORMAT format : FourCCTypes)
			{
				TextureList.push_back(format);
			}
		}

		// Check for supported textures
		DDPIXELFORMAT ddpfPixelFormat = {};
		ddpfPixelFormat.dwSize = sizeof(DDPIXELFORMAT);

		bool IsDirectDraw8bit = (ddrawParent->GetDisplayBPP() == 8);

		for (D3DFORMAT format : TextureList)
		{
			if (!IsUnsupportedFormat(format) && ((format == D3DFMT_P8 && IsDirectDraw8bit) ||
				SUCCEEDED(d3d9Object->CheckDeviceFormat(ddrawParent->GetAdapterIndex(), D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, format))))
			{
				SetPixelDisplayFormat(format, ddpfPixelFormat);
				if (lpd3dEnumPixelProc(&ddpfPixelFormat, lpArg) == DDENUMRET_CANCEL)
				{
					return D3D_OK;
				}
			}
		}

		return D3D_OK;
	}

	switch (ProxyDirectXVersion)
	{
	case 1:
	case 2:
	default:
		return DDERR_GENERIC;
	case 3:
		return GetProxyInterfaceV3()->EnumTextureFormats(lpd3dEnumPixelProc, lpArg);
	case 7:
		return GetProxyInterfaceV7()->EnumTextureFormats(lpd3dEnumPixelProc, lpArg);
	}
}

HRESULT m_IDirect3DDeviceX::Clear(DWORD dwCount, LPD3DRECT lpRects, DWORD dwFlags, D3DCOLOR dwColor, D3DVALUE dvZ, DWORD dwStencil)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		ScopedCriticalSection ThreadLockDD(DdrawWrapper::GetDDCriticalSection());

		// Clear respects the current viewport
		(*d3d9Device)->SetViewport(&DeviceStates.Viewport.View);

		if (lpCurrentRenderTargetX && (dwFlags & D3DCLEAR_TARGET))
		{
			lpCurrentRenderTargetX->PrepareRenderTarget();

			if (ddrawParent->GetRenderTargetSurface() != lpCurrentRenderTargetX)
			{
				ddrawParent->SetRenderTargetSurface(lpCurrentRenderTargetX);
			}
		}

		return (*d3d9Device)->Clear(dwCount, lpRects, dwFlags, dwColor, dvZ, dwStencil);
	}

	return GetProxyInterfaceV7()->Clear(dwCount, lpRects, dwFlags, dwColor, dvZ, dwStencil);
}

HRESULT m_IDirect3DDeviceX::SetViewport(LPD3DVIEWPORT7 lpViewport)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpViewport)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		// Clear viewport scaling
		DeviceStates.Viewport.UseViewportScale = false;

		return SetD9Viewport((D3DVIEWPORT9*)lpViewport);
	}

	return GetProxyInterfaceV7()->SetViewport(lpViewport);
}

HRESULT m_IDirect3DDeviceX::GetViewport(LPD3DVIEWPORT7 lpViewport)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpViewport)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			*lpViewport = {};
			return DDERR_INVALIDOBJECT;
		}

		return GetD9Viewport((D3DVIEWPORT9*)lpViewport);
	}

	return GetProxyInterfaceV7()->GetViewport(lpViewport);
}

HRESULT m_IDirect3DDeviceX::SetMaterial(LPD3DMATERIAL7 lpMaterial)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpMaterial)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return SetD9Material((D3DMATERIAL9*)lpMaterial);
	}

	return GetProxyInterfaceV7()->SetMaterial(lpMaterial);
}

HRESULT m_IDirect3DDeviceX::GetMaterial(LPD3DMATERIAL7 lpMaterial)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpMaterial)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return GetD9Material((D3DMATERIAL9*)lpMaterial);
	}

	return GetProxyInterfaceV7()->GetMaterial(lpMaterial);
}

HRESULT m_IDirect3DDeviceX::SetLight(DWORD dwLightIndex, LPD3DLIGHT7 lpLight)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpLight)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: called with nullptr: " << lpLight);
			return DDERR_INVALIDPARAMS;
		}

		if (lpLight->dltType == D3DLIGHT_PARALLELPOINT || lpLight->dltType == D3DLIGHT_GLSPOT)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Warning: Light Type: " << lpLight->dltType << " Not Implemented");
			return D3D_OK;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		HRESULT hr = SetD9Light(dwLightIndex, reinterpret_cast<D3DLIGHT9*>(lpLight));

		if (SUCCEEDED(hr))
		{
#ifdef ENABLE_DEBUGOVERLAY
			if (Config.EnableImgui)
			{
				DOverlay.SetLight(dwLightIndex, lpLight);
			}
#endif
		}

		return hr;
	}

	return GetProxyInterfaceV7()->SetLight(dwLightIndex, lpLight);
}

HRESULT m_IDirect3DDeviceX::GetLight(DWORD dwLightIndex, LPD3DLIGHT7 lpLight)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpLight)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return GetD9Light(dwLightIndex, reinterpret_cast<D3DLIGHT9*>(lpLight));
	}

	return GetProxyInterfaceV7()->GetLight(dwLightIndex, lpLight);
}

HRESULT m_IDirect3DDeviceX::BeginStateBlock()
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (StateBlock.IsRecording)
		{
			return D3DERR_INBEGINSTATEBLOCK;
		}

		StateBlock.RecordingToken = StateBlock.CreateToken(D3DSBT_ALL);

		StateBlock.Data[StateBlock.RecordingToken].RecordState.emplace();

		StateBlock.IsRecording = true;

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->BeginStateBlock();
}

HRESULT m_IDirect3DDeviceX::EndStateBlock(LPDWORD lpdwBlockHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwBlockHandle)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lpdwBlockHandle = NULL;

		if (!StateBlock.IsRecording)
		{
			return D3DERR_NOTINBEGINSTATEBLOCK;
		}

		StateBlock.IsRecording = false;

		*lpdwBlockHandle = StateBlock.RecordingToken;

		StateBlock.RecordingToken = 0;

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->EndStateBlock(lpdwBlockHandle);
}

HRESULT m_IDirect3DDeviceX::PreLoad(LPDIRECTDRAWSURFACE7 lpddsTexture)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Textures are loaded as managed in Direct3D9, so there is no need to manualy preload textures
		return D3D_OK;
	}

	if (lpddsTexture)
	{
		lpddsTexture->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpddsTexture);
	}

	return GetProxyInterfaceV7()->PreLoad(lpddsTexture);
}

HRESULT m_IDirect3DDeviceX::GetTexture(DWORD dwStage, LPDIRECTDRAWSURFACE7* lplpTexture)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lplpTexture || dwStage >= D3DHAL_TSS_MAXSTAGES)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lplpTexture = nullptr;

		HRESULT hr = DDERR_GENERIC;

		if (AttachedTexture[dwStage])
		{
			AttachedTexture[dwStage]->AddRef();

			*lplpTexture = AttachedTexture[dwStage];

			hr = D3D_OK;
		}

		return hr;
	}

	HRESULT hr = GetProxyInterfaceV7()->GetTexture(dwStage, lplpTexture);

	if (SUCCEEDED(hr) && lplpTexture)
	{
		*lplpTexture = ProxyAddressLookupTable.FindAddress<m_IDirectDrawSurface7>(*lplpTexture, 7);
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::SetTexture(DWORD dwStage, LPDIRECTDRAWSURFACE7 lpSurface)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwStage >= D3DHAL_TSS_MAXSTAGES)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = nullptr;

		if (lpSurface)
		{
			lpSurface->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDSrcSurfaceX);
			if (!lpDDSrcSurfaceX)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface wrapper!");
				return DDERR_INVALIDPARAMS;
			}

			IDirect3DTexture9* pTexture9 = lpDDSrcSurfaceX->GetD3d9Texture();
			if (!pTexture9)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture!");
				return DDERR_INVALIDPARAMS;
			}

			if (lpCurrentRenderTargetX && lpCurrentRenderTargetX->IsPalette() && !lpDDSrcSurfaceX->IsPalette())
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: setting non-palette texture on a paletted render target!");
			}
		}

		AttachedTexture[dwStage] = lpSurface;
		CurrentTextureSurfaceX[dwStage] = lpDDSrcSurfaceX;

		// Log SetTexture calls for atlas analysis
		if (Config.DdrawLogTextureAtlas && lpDDSrcSurfaceX)
		{
			static DWORD setTextureLogCount = 0;
			setTextureLogCount++;
			// Log every 1000th call to avoid spam
			if (setTextureLogCount % 1000 == 1)
			{
				DWORD texWidth = 0, texHeight = 0;
				lpDDSrcSurfaceX->GetSurfaceSetSize(texWidth, texHeight);
				Logging::Log() << "SETTEXTURE #" << setTextureLogCount
					<< " stage=" << dwStage
					<< " surface=" << lpDDSrcSurfaceX
					<< " size=" << texWidth << "x" << texHeight;
			}
		}

		return D3D_OK;
	}

	if (lpSurface)
	{
		lpSurface->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpSurface);
	}

	return GetProxyInterfaceV7()->SetTexture(dwStage, lpSurface);
}

HRESULT m_IDirect3DDeviceX::ApplyStateBlock(DWORD dwBlockHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwBlockHandle == 0)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (StateBlock.IsRecording)
		{
			return D3DERR_INBEGINSTATEBLOCK;
		}

		auto it = std::find(StateBlock.Tokens.begin(), StateBlock.Tokens.end(), dwBlockHandle);
		if (it == StateBlock.Tokens.end())
		{
			return D3D_OK;
		}

		D3DSTATEBLOCKTYPE Type = StateBlock.Data[dwBlockHandle].Type;

		switch (Type)
		{
		case D3DSBT_ALL:
			if (StateBlock.Data[dwBlockHandle].FullState.has_value())
			{
				DeviceStates = StateBlock.Data[dwBlockHandle].FullState.value();
			}
			else if (StateBlock.Data[dwBlockHandle].RecordState.has_value())
			{
				const auto& RecordState = StateBlock.Data[dwBlockHandle].RecordState.value();

				// Restore states
				for (const auto& entry : RecordState.RenderState)
				{
					SetD9RenderState(entry.first, entry.second);
				}
				for (const auto& entry : RecordState.UnmappedRenderState)
				{
					DeviceStates.RenderState[entry.first].State = entry.second;
				}
				for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
				{
					for (const auto& entry : RecordState.TextureStageState[x])
					{
						SetD9TextureStageState(x, entry.first, entry.second);
					}
					for (const auto& entry : RecordState.SamplerState[x])
					{
						SetD9SamplerState(x, entry.first, entry.second);
					}
				}
				for (const auto& entry : RecordState.Light)
				{
					SetD9Light(entry.first, &entry.second);
				}
				for (const auto& entry : RecordState.LightEnable)
				{
					D9LightEnable(entry.first, entry.second);
				}
				for (const auto& entry : RecordState.ClipPlane)
				{
					SetD9ClipPlane(entry.first, reinterpret_cast<const float*>(&entry.second));
				}
				for (const auto& entry : RecordState.Viewport)
				{
					SetD9Viewport(&entry.second);
				}
				for (const auto& entry : RecordState.Material)
				{
					SetD9Material(&entry.second);
				}
				for (const auto& entry : RecordState.Matrix)
				{
					SetD9Transform(entry.first, &entry.second);
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find any state struct to apply!");
			}
			break;
		case D3DSBT_PIXELSTATE:
			if (StateBlock.Data[dwBlockHandle].PixelState.has_value())
			{
				RequiresStateRestore = true;

				BatchStates.clear();

				const auto& PixelState = StateBlock.Data[dwBlockHandle].PixelState.value();

				// Render states
				for (const auto& entry : PixelState.RenderState)
				{
					DeviceStates.RenderState[entry.first] = entry.second;
				}

				// Texture-stage states
				memcpy(DeviceStates.TextureStageState, PixelState.TextureStageState, sizeof(DeviceStates.TextureStageState));

				// Sampler state
				memcpy(DeviceStates.SamplerState, PixelState.SamplerState, sizeof(DeviceStates.SamplerState));
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find pixel state struct to apply!");
			}
			break;
		case D3DSBT_VERTEXSTATE:
			if (StateBlock.Data[dwBlockHandle].VertexState.has_value())
			{
				RequiresStateRestore = true;

				BatchStates.clear();

				const auto& VertexState = StateBlock.Data[dwBlockHandle].VertexState.value();

				// Light enable
				DeviceStates.LightEnable = VertexState.LightEnable;

				// Transformation matrices
				DeviceStates.Matrix = VertexState.Matrix;

				// Clipping planes
				memcpy(DeviceStates.ClipPlane, VertexState.ClipPlane, sizeof(DeviceStates.ClipPlane));

				// Render states
				for (const auto& entry : VertexState.RenderState)
				{
					DeviceStates.RenderState[entry.first] = entry.second;
				}

				// Texture-stage states
				for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
				{
					for (const auto& entry : VertexState.TextureStageState[x])
					{
						DeviceStates.TextureStageState[x][entry.first] = entry.second;
					}
				}
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not find vertex state struct to apply!");
			}
			break;
		}

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->ApplyStateBlock(dwBlockHandle);
}

HRESULT m_IDirect3DDeviceX::CaptureStateBlock(DWORD dwBlockHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwBlockHandle == 0)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (StateBlock.IsRecording)
		{
			return D3DERR_INBEGINSTATEBLOCK;
		}

		auto it = std::find(StateBlock.Tokens.begin(), StateBlock.Tokens.end(), dwBlockHandle);
		if (it == StateBlock.Tokens.end())
		{
			return D3D_OK;
		}

		StateBlock.Data[dwBlockHandle].RecordState.reset();
		StateBlock.Data[dwBlockHandle].PixelState.reset();
		StateBlock.Data[dwBlockHandle].VertexState.reset();
		StateBlock.Data[dwBlockHandle].FullState.reset();

		D3DSTATEBLOCKTYPE Type = StateBlock.Data[dwBlockHandle].Type;

		switch (Type)
		{
		case D3DSBT_ALL:
			StateBlock.Data[dwBlockHandle].FullState = DeviceStates;
			break;
		case D3DSBT_PIXELSTATE:
		{
			StateBlock.Data[dwBlockHandle].PixelState.emplace();
			auto& PixelState = StateBlock.Data[dwBlockHandle].PixelState.value();
			for (const auto& State : StateBlockPixelRenderStates)
			{
				PixelState.RenderState[State] = DeviceStates.RenderState[State];
			}
			memcpy(PixelState.TextureStageState, DeviceStates.TextureStageState, sizeof(PixelState.TextureStageState));
			memcpy(PixelState.SamplerState, DeviceStates.SamplerState, sizeof(PixelState.SamplerState));
			break;
		}
		case D3DSBT_VERTEXSTATE:
		{
			StateBlock.Data[dwBlockHandle].VertexState.emplace();
			auto& VertexState = StateBlock.Data[dwBlockHandle].VertexState.value();
			VertexState.LightEnable = DeviceStates.LightEnable;
			VertexState.Matrix = DeviceStates.Matrix;
			memcpy(VertexState.ClipPlane, DeviceStates.ClipPlane, sizeof(VertexState.ClipPlane));
			for (const auto& State : StateBlockVertexRenderStates)
			{
				VertexState.RenderState[State] = DeviceStates.RenderState[State];
			}
			for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
			{
				VertexState.TextureStageState[x][D3DTSS_TEXCOORDINDEX] = DeviceStates.TextureStageState[x][D3DTSS_TEXCOORDINDEX];
				VertexState.TextureStageState[x][D3DTSS_TEXTURETRANSFORMFLAGS] = DeviceStates.TextureStageState[x][D3DTSS_TEXTURETRANSFORMFLAGS];
			}
			break;
		}
		}

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->CaptureStateBlock(dwBlockHandle);
}

HRESULT m_IDirect3DDeviceX::DeleteStateBlock(DWORD dwBlockHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (dwBlockHandle == 0)
		{
			return DDERR_INVALIDPARAMS;
		}

		if (StateBlock.IsRecording)
		{
			return D3DERR_INBEGINSTATEBLOCK;
		}

		StateBlock.DeleteToken(dwBlockHandle);

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->DeleteStateBlock(dwBlockHandle);
}

HRESULT m_IDirect3DDeviceX::CreateStateBlock(D3DSTATEBLOCKTYPE d3dsbtype, LPDWORD lpdwBlockHandle)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpdwBlockHandle)
		{
			return DDERR_INVALIDPARAMS;
		}
		*lpdwBlockHandle = NULL;

		if (StateBlock.IsRecording)
		{
			return D3DERR_INBEGINSTATEBLOCK;
		}

		switch (d3dsbtype)
		{
		case D3DSBT_ALL:
		case D3DSBT_PIXELSTATE:
		case D3DSBT_VERTEXSTATE:
			break;
		default:
			LOG_LIMIT(100, __FUNCTION__ << " Error: invalid StateBlock type: " << d3dsbtype);
			return DDERR_INVALIDPARAMS;
		}

		*lpdwBlockHandle = StateBlock.CreateToken(d3dsbtype);

		CaptureStateBlock(*lpdwBlockHandle);

		return D3D_OK;
	}

	return GetProxyInterfaceV7()->CreateStateBlock(d3dsbtype, lpdwBlockHandle);
}

HRESULT m_IDirect3DDeviceX::Load(LPDIRECTDRAWSURFACE7 lpDestTex, LPPOINT lpDestPoint, LPDIRECTDRAWSURFACE7 lpSrcTex, LPRECT lprcSrcRect, DWORD dwFlags)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!lpDestTex || !lpSrcTex)
		{
			return DDERR_INVALIDPARAMS;
		}

		m_IDirectDrawSurfaceX* pDestSurfaceX = nullptr;
		lpDestTex->QueryInterface(IID_GetInterfaceX, (LPVOID*)&pDestSurfaceX);
		if (!pDestSurfaceX)
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get surface!");
			return DDERR_GENERIC;
		}

		return pDestSurfaceX->Load(lpDestTex, lpDestPoint, lpSrcTex, lprcSrcRect, dwFlags);
	}

	if (lpDestTex)
	{
		lpDestTex->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpDestTex);
	}
	if (lpSrcTex)
	{
		lpSrcTex->QueryInterface(IID_GetRealInterface, (LPVOID*)&lpSrcTex);
	}

	return GetProxyInterfaceV7()->Load(lpDestTex, lpDestPoint, lpSrcTex, lprcSrcRect, dwFlags);
}

HRESULT m_IDirect3DDeviceX::LightEnable(DWORD dwLightIndex, BOOL bEnable)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		HRESULT hr = D9LightEnable(dwLightIndex, bEnable);

		if (SUCCEEDED(hr))
		{
#ifdef ENABLE_DEBUGOVERLAY
			if (Config.EnableImgui)
			{
				DOverlay.LightEnable(dwLightIndex, bEnable);
			}
#endif
		}

		return hr;
	}

	return GetProxyInterfaceV7()->LightEnable(dwLightIndex, bEnable);
}

HRESULT m_IDirect3DDeviceX::GetLightEnable(DWORD dwLightIndex, BOOL* pbEnable)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!pbEnable)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return GetD9LightEnable(dwLightIndex, pbEnable);
	}

	return GetProxyInterfaceV7()->GetLightEnable(dwLightIndex, pbEnable);
}

HRESULT m_IDirect3DDeviceX::SetClipPlane(DWORD dwIndex, D3DVALUE* pPlaneEquation)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return SetD9ClipPlane(dwIndex, pPlaneEquation);
	}

	return GetProxyInterfaceV7()->SetClipPlane(dwIndex, pPlaneEquation);
}

HRESULT m_IDirect3DDeviceX::GetClipPlane(DWORD dwIndex, D3DVALUE* pPlaneEquation)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!pPlaneEquation)
		{
			return DDERR_INVALIDPARAMS;
		}

		// Check for device interface
		if (FAILED(CheckInterface(__FUNCTION__, true)))
		{
			return DDERR_INVALIDOBJECT;
		}

		return GetD9ClipPlane(dwIndex, pPlaneEquation);
	}

	return GetProxyInterfaceV7()->GetClipPlane(dwIndex, pPlaneEquation);
}

HRESULT m_IDirect3DDeviceX::GetInfo(DWORD dwDevInfoID, LPVOID pDevInfoStruct, DWORD dwSize)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (Config.Dd7to9)
	{
		if (!pDevInfoStruct || dwSize == 0)
		{
			return DDERR_GENERIC;
		}

#ifdef _DEBUG
		// Fill device info structures
		switch (dwDevInfoID)
		{
		case D3DDEVINFOID_TEXTUREMANAGER:
		case D3DDEVINFOID_D3DTEXTUREMANAGER:
			if (dwSize == sizeof(D3DDEVINFO_TEXTUREMANAGER))
			{
				// Simulate a default texture manager structure for a good video card
				D3DDEVINFO_TEXTUREMANAGER* pTexManagerInfo = (D3DDEVINFO_TEXTUREMANAGER*)pDevInfoStruct;
				pTexManagerInfo->bThrashing = FALSE;
				pTexManagerInfo->dwNumEvicts = 0;
				pTexManagerInfo->dwNumVidCreates = 0;
				pTexManagerInfo->dwNumTexturesUsed = 0;
				pTexManagerInfo->dwNumUsedTexInVid = 0;
				pTexManagerInfo->dwWorkingSet = 0;
				pTexManagerInfo->dwWorkingSetBytes = 0;
				pTexManagerInfo->dwTotalManaged = 0;
				pTexManagerInfo->dwTotalBytes = 0;
				pTexManagerInfo->dwLastPri = 0;
				break;
			}
			return DDERR_GENERIC;

		case D3DDEVINFOID_TEXTURING:
			if (dwSize == sizeof(D3DDEVINFO_TEXTURING))
			{
				// Simulate a default texturing activity structure for a good video card
				D3DDEVINFO_TEXTURING* pTexturingInfo = (D3DDEVINFO_TEXTURING*)pDevInfoStruct;
				pTexturingInfo->dwNumLoads = 0;
				pTexturingInfo->dwApproxBytesLoaded = 0;
				pTexturingInfo->dwNumPreLoads = 0;
				pTexturingInfo->dwNumSet = 0;
				pTexturingInfo->dwNumCreates = 0;
				pTexturingInfo->dwNumDestroys = 0;
				pTexturingInfo->dwNumSetPriorities = 0;
				pTexturingInfo->dwNumSetLODs = 0;
				pTexturingInfo->dwNumLocks = 0;
				pTexturingInfo->dwNumGetDCs = 0;
				break;
			}
			return DDERR_GENERIC;

		default:
			Logging::LogDebug() << __FUNCTION__ << " Error: Unknown DevInfoID: " << dwDevInfoID;
			return DDERR_GENERIC;
		}
#endif

		// This method is intended to be used for performance tracking and debugging during product development (on the debug version of DirectX). 
		// The method can succeed, returning S_FALSE, without retrieving device data.
		// This occurs when the retail version of the DirectX runtime is installed on the host system.
		return S_FALSE;
	}

	return GetProxyInterfaceV7()->GetInfo(dwDevInfoID, pDevInfoStruct, dwSize);
}

// ******************************
// Helper functions
// ******************************

void m_IDirect3DDeviceX::InitInterface(DWORD DirectXVersion)
{
	if (D3DInterface)
	{
		D3DInterface->AddD3DDevice(this);
	}

	if (Config.Dd7to9)
	{
		if (ddrawParent)
		{
			d3d9Device = ddrawParent->GetDirectD9Device();

			if (CurrentRenderTarget)
			{
				m_IDirectDrawSurfaceX* lpDDSrcSurfaceX = nullptr;

				CurrentRenderTarget->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpDDSrcSurfaceX);
				if (lpDDSrcSurfaceX)
				{
					CurrentRenderTarget->AddRef();

					lpCurrentRenderTargetX = lpDDSrcSurfaceX;

					RenderTargetMultiSampleType = lpDDSrcSurfaceX->GetMultiSampleType();

					DepthBitCount = lpDDSrcSurfaceX->GetAttachedStencilSurfaceZBits();

					ddrawParent->SetRenderTargetSurface(lpCurrentRenderTargetX);
				}
			}
		}

		AddRef(DirectXVersion);
	}
}

void m_IDirect3DDeviceX::ReleaseInterface()
{
	if (Config.Exiting)
	{
		return;
	}

	if (CurrentRenderTarget)
	{
		lpCurrentRenderTargetX = nullptr;

		CurrentRenderTarget->Release();
		CurrentRenderTarget = nullptr;
	}

	if (D3DInterface)
	{
		D3DInterface->ClearD3DDevice(this);
	}

	// Don't delete wrapper interface
	SaveInterfaceAddress(WrapperInterface);
	SaveInterfaceAddress(WrapperInterface2);
	SaveInterfaceAddress(WrapperInterface3);
	SaveInterfaceAddress(WrapperInterface7);

	// Clear ExecuteBuffers
	for (const auto& entry : ExecuteBufferList)
	{
		entry->ClearD3DDevice();
	}

	// Clear device from veiwports
	for (const auto& entry : AttachedViewports)
	{
		m_IDirect3DViewportX* lpViewportX = nullptr;
		if (SUCCEEDED(entry->QueryInterface(IID_GetInterfaceX, (LPVOID*)&lpViewportX)))
		{
			lpViewportX->ClearD3DDevice(this);
		}
	}
}

HRESULT m_IDirect3DDeviceX::CheckInterface(char *FunctionName, bool CheckD3DDevice)
{
	// Check ddrawParent device
	if (!ddrawParent)
	{
		LOG_LIMIT(100, FunctionName << " Error: no ddraw parent!");
		return DDERR_INVALIDOBJECT;
	}

	// Check d3d9 device
	if (CheckD3DDevice)
	{
		if (!ddrawParent->CheckD9Device(FunctionName) || !d3d9Device || !*d3d9Device)
		{
			LOG_LIMIT(100, FunctionName << " Error: d3d9 device not setup!");
			return DDERR_INVALIDOBJECT;
		}
		if (bSetDefaults)
		{
			SetDefaults();
		}
	}

	return D3D_OK;
}

void* m_IDirect3DDeviceX::GetWrapperInterfaceX(DWORD DirectXVersion)
{
	switch (DirectXVersion)
	{
	case 0:
		if (WrapperInterface7) return WrapperInterface7;
		if (WrapperInterface3) return WrapperInterface3;
		if (WrapperInterface2) return WrapperInterface2;
		if (WrapperInterface) return WrapperInterface;
		break;
	case 1:
		return GetInterfaceAddress(WrapperInterface, (LPDIRECT3DDEVICE)ProxyInterface, this);
	case 2:
		return GetInterfaceAddress(WrapperInterface2, (LPDIRECT3DDEVICE2)ProxyInterface, this);
	case 3:
		return GetInterfaceAddress(WrapperInterface3, (LPDIRECT3DDEVICE3)ProxyInterface, this);
	case 7:
		return GetInterfaceAddress(WrapperInterface7, (LPDIRECT3DDEVICE7)ProxyInterface, this);
	}
	LOG_LIMIT(100, __FUNCTION__ << " Error: wrapper interface version not found: " << DirectXVersion);
	return nullptr;
}

LPDIRECT3DDEVICE9* m_IDirect3DDeviceX::GetD3d9Device()
{
	CheckInterface(__FUNCTION__, true);

	return d3d9Device;
}

HRESULT m_IDirect3DDeviceX::SetViewport(LPD3DVIEWPORT lpViewport)
{
	if (!lpViewport)
	{
		return DDERR_INVALIDPARAMS;
	}

	D3DVIEWPORT7 Viewport7 = {};
	ConvertViewport(Viewport7, *lpViewport);

	HRESULT hr = SetViewport(&Viewport7);

	if (SUCCEEDED(hr))
	{
		DeviceStates.Viewport.UseViewportScale = true;
		DeviceStates.Viewport.ViewportScale = *lpViewport;

		// Set transform
		D3DMATRIX Matrix = {};
		if (SUCCEEDED(GetD9Transform(D3DTS_PROJECTION, &Matrix)))
		{
			SetD9Transform(D3DTS_PROJECTION, &Matrix);
		}
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::SetViewport(LPD3DVIEWPORT2 lpViewport)
{
	if (!lpViewport)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (lpViewport->dvClipWidth != 0 || lpViewport->dvClipHeight != 0 || lpViewport->dvClipX != 0 || lpViewport->dvClipY != 0)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: 'clip volume' Not Implemented: " <<
			lpViewport->dvClipWidth << "x" << lpViewport->dvClipHeight << " X: " << lpViewport->dvClipX << " Y: " << lpViewport->dvClipY);
	}

	D3DVIEWPORT7 Viewport7 = {};
	ConvertViewport(Viewport7, *lpViewport);

	return SetViewport(&Viewport7);
}

bool m_IDirect3DDeviceX::DeleteAttachedViewport(LPDIRECT3DVIEWPORT3 ViewportX)
{
	auto it = std::find_if(AttachedViewports.begin(), AttachedViewports.end(),
		[=](auto pViewport) -> bool { return pViewport == ViewportX; });

	if (it != std::end(AttachedViewports))
	{
		AttachedViewports.erase(it);
		return true;
	}
	return false;
}

void m_IDirect3DDeviceX::ClearTextureHandle(D3DTEXTUREHANDLE tHandle)
{
	if (tHandle)
	{
		TextureHandleMap.erase(tHandle);

		// If texture handle is set then clear it
		if (tHandle == DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREHANDLE].State)
		{
			SetTexture(0, (LPDIRECT3DTEXTURE2)nullptr);
		}
	}
}

HRESULT m_IDirect3DDeviceX::SetTextureHandle(D3DTEXTUREHANDLE& tHandle, m_IDirect3DTextureX* pTextureX)
{
	if (!tHandle || !pTextureX)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: NULL pointer found! " << pTextureX << " -> " << tHandle);
		return DDERR_INVALIDPARAMS;
	}

	// Ensure that the handle is unique
	while (GetTexture(tHandle))
	{
		tHandle += 4;
	}

	TextureHandleMap[tHandle] = pTextureX;

	return D3D_OK;
}

void m_IDirect3DDeviceX::ClearLight(m_IDirect3DLight* lpLight)
{
	// Find handle associated with Light
	auto it = LightIndexMap.begin();
	while (it != LightIndexMap.end())
	{
		if (it->second == lpLight)
		{
			DWORD Index = it->first;

			// Remove entry from map
			it = LightIndexMap.erase(it);

			// Disable light
			D9LightEnable(Index, FALSE);

			// Clear light
			DeviceStates.Light.erase(Index);

			// Clear light enable
			DeviceStates.LightEnable.erase(Index);

			// Clear batch state light
			BatchStates.Light.erase(Index);

			// Clear batch state light enable
			BatchStates.LightEnable.erase(Index);
		}
		else
		{
			++it;
		}
	}
}

HRESULT m_IDirect3DDeviceX::SetLight(m_IDirect3DLight* lpLightInterface, LPD3DLIGHT lpLight)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lpLightInterface || !lpLight || (lpLight->dwSize != sizeof(D3DLIGHT) && lpLight->dwSize != sizeof(D3DLIGHT2)))
	{
		return DDERR_INVALIDPARAMS;
	}

	D3DLIGHT7 Light7;

	// ToDo: the dvAttenuation members are interpreted differently in D3DLIGHT2 than they were for D3DLIGHT.

	ConvertLight(Light7, *lpLight);

	DWORD dwLightIndex = MaxActiveLights;

	// Check if Light exists in the map
	for (const auto& entry : LightIndexMap)
	{
		if (entry.second == lpLightInterface)
		{
			dwLightIndex = entry.first;
			break;
		}
	}

	// Create index and add light to the map
	if (dwLightIndex == MaxActiveLights)
	{
		for (DWORD x = 0; x < MaxActiveLights; x++)
		{
			if (LightIndexMap.find(x) != LightIndexMap.end()) continue;
			if (DeviceStates.Light.find(x) != DeviceStates.Light.end()) continue;
			if (DeviceStates.LightEnable.find(x) != DeviceStates.LightEnable.end()) continue;

			dwLightIndex = x;
			break;
		}
	}

	if (dwLightIndex >= MaxActiveLights)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to find an available Light Index");
		return DDERR_INVALIDPARAMS;
	}

	// Add light to index map
	LightIndexMap[dwLightIndex] = lpLightInterface;

	HRESULT hr = SetLight(dwLightIndex, &Light7);

	if (SUCCEEDED(hr))
	{
		if (lpLight->dwSize == sizeof(D3DLIGHT2))
		{
			LPD3DLIGHT2 lpLight2 = reinterpret_cast<LPD3DLIGHT2>(lpLight);

			BOOL Enable = (lpLight2->dwFlags & D3DLIGHT_ACTIVE) ? TRUE : FALSE;

			LightEnable(dwLightIndex, Enable);
		}
	}

	return hr;
}

HRESULT m_IDirect3DDeviceX::GetLightEnable(m_IDirect3DLight* lpLightInterface, BOOL* pbEnable)
{
	if (!lpLightInterface || !pbEnable)
	{
		return DDERR_INVALIDPARAMS;
	}

	DWORD dwLightIndex = 0;

	// Check if Light exists in the map
	for (const auto& entry : LightIndexMap)
	{
		if (entry.second == lpLightInterface)
		{
			dwLightIndex = entry.first;
			break;
		}
	}

	if (dwLightIndex == 0)
	{
		return DDERR_INVALIDPARAMS;
	}

	return GetLightEnable(dwLightIndex, pbEnable);
}

void m_IDirect3DDeviceX::ClearMaterialHandle(D3DMATERIALHANDLE mHandle)
{
	if (mHandle)
	{
		TextureHandleMap.erase(mHandle);

		// If material handle is set then clear it
		if (mHandle == DeviceStates.LightState[D3DLIGHTSTATE_MATERIAL])
		{
			SetMaterialHandle(NULL);
		}
	}
}

HRESULT m_IDirect3DDeviceX::SetMaterialHandle(D3DMATERIALHANDLE& mHandle, m_IDirect3DMaterialX* lpMaterial)
{
	if (!mHandle || !lpMaterial)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: NULL pointer found! " << lpMaterial << " -> " << mHandle);
		return DDERR_GENERIC;
	}

	// Ensure that the handle is unique
	while (GetMaterial(mHandle))
	{
		mHandle += 4;
	}

	MaterialHandleMap[mHandle] = lpMaterial;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetMaterial(LPD3DMATERIAL lpMaterial)
{
	Logging::LogDebug() << __FUNCTION__ << " (" << this << ")";

	if (!lpMaterial)
	{
		return DDERR_INVALIDPARAMS;
	}

	D3DMATERIAL7 Material7;

	ConvertMaterial(Material7, *lpMaterial);

	HRESULT hr = SetMaterial(&Material7);

	if (FAILED(hr))
	{
		LOG_LIMIT(100, __FUNCTION__ << " Error: Failed to set material: " << (D3DERR)hr);
		return hr;
	}

	if (lpMaterial->dwRampSize)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: RampSize Not Implemented: " << lpMaterial->dwRampSize);
	}

	if (lpMaterial->hTexture)
	{
		SetTextureHandle(lpMaterial->hTexture);
	}

	return D3D_OK;
}

void m_IDirect3DDeviceX::ClearSurface(m_IDirectDrawSurfaceX* lpSurfaceX)
{
	if (lpCurrentRenderTargetX == lpSurfaceX)
	{
		CurrentRenderTarget = nullptr;
		lpCurrentRenderTargetX = nullptr;
		LOG_LIMIT(100, __FUNCTION__ << " Warning: clearing current render target!");
	}
	for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
	{
		if (CurrentTextureSurfaceX[x] == lpSurfaceX)
		{
			SetTexture(x, (LPDIRECTDRAWSURFACE7)nullptr);
			AttachedTexture[x] = nullptr;
			CurrentTextureSurfaceX[x] = nullptr;
		}
	}
}

void m_IDirect3DDeviceX::AddExecuteBuffer(m_IDirect3DExecuteBuffer* lpExecuteBuffer)
{
	if (!lpExecuteBuffer)
	{
		return;
	}

	ExecuteBufferList.push_back(lpExecuteBuffer);
}

void m_IDirect3DDeviceX::ClearExecuteBuffer(m_IDirect3DExecuteBuffer* lpExecuteBuffer)
{
	// Find and remove the buffer from the list
	auto it = std::find(ExecuteBufferList.begin(), ExecuteBufferList.end(), lpExecuteBuffer);
	if (it != ExecuteBufferList.end())
	{
		ExecuteBufferList.erase(it);
	}
}

void m_IDirect3DDeviceX::CopyConvertExecuteVertex(BYTE*& DestVertex, DWORD& DestVertexCount, BYTE* SrcVertex, DWORD SrcIndex, DWORD VertexTypeDesc)
{
	// Primitive structures and related defines. Vertex offsets are to types D3DVERTEX, D3DLVERTEX, or D3DTLVERTEX.
	if (VertexTypeDesc == D3DFVF_VERTEX)
	{
		DestVertexCount++;
		*((D3DVERTEX*)DestVertex) = ((D3DVERTEX*)SrcVertex)[SrcIndex];
		DestVertex += sizeof(D3DVERTEX);
		return;
	}
	else if (VertexTypeDesc == D3DFVF_LVERTEX)
	{
		DestVertexCount++;
		*((D3DLVERTEX*)DestVertex) = ((D3DLVERTEX*)SrcVertex)[SrcIndex];
		DestVertex += sizeof(D3DLVERTEX);
		return;
	}
	else if (VertexTypeDesc == D3DFVF_TLVERTEX)
	{
		DestVertexCount++;
		*((D3DTLVERTEX*)DestVertex) = ((D3DTLVERTEX*)SrcVertex)[SrcIndex];
		DestVertex += sizeof(D3DTLVERTEX);
		return;
	}
}

HRESULT m_IDirect3DDeviceX::DrawExecutePoint(D3DPOINT* point, WORD pointCount, DWORD vertexIndexCount, BYTE* vertexBuffer, DWORD VertexTypeDesc)
{
	// Define vertices and setup vector
	std::vector<BYTE, aligned_allocator<BYTE, 4>> vertices;
	vertices.resize(sizeof(D3DTLVERTEX) * pointCount);
	BYTE* verticesData = vertices.data();
	DWORD verticesCount = 0;

	// Add vertices to vector
	for (DWORD i = 0; i < pointCount; i++)
	{
		if ((DWORD)point[i].wFirst < vertexIndexCount)
		{
			DWORD count = min(point[i].wCount, vertexIndexCount - point[i].wFirst);

			for (DWORD x = 0; x < count; x++)
			{
				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, point[i].wFirst + x, VertexTypeDesc);
			}
		}
	}

	if (verticesCount)
	{
		// Pass the vertex data to the rendering pipeline
		DrawPrimitive(D3DPT_POINTLIST, VertexTypeDesc, vertices.data(), verticesCount, 0, 1);
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::DrawExecuteSpan(D3DSPAN* span, WORD spanCount, DWORD vertexIndexCount, BYTE* vertexBuffer, DWORD VertexTypeDesc)
{
	// Define vertices and setup vector
	std::vector<BYTE, aligned_allocator<BYTE, 4>> vertices;
	vertices.resize(sizeof(D3DTLVERTEX) * vertexIndexCount);
	BYTE* verticesData = vertices.data();
	DWORD verticesCount = 0;

	// Add vertices to vector
	for (DWORD i = 0; i < spanCount; i++)
	{
		if ((DWORD)span[i].wFirst < vertexIndexCount)
		{
			DWORD count = min(span[i].wCount, vertexIndexCount - span[i].wFirst);

			for (DWORD x = 0; x < count; x++)
			{
				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, span[i].wFirst + x, VertexTypeDesc);
			}
		}
	}

	if (verticesCount)
	{
		// Pass the vertex data to the rendering pipeline
		DrawPrimitive(D3DPT_LINESTRIP, VertexTypeDesc, vertices.data(), verticesCount, 0, 1);
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::DrawExecuteLine(D3DLINE* line, WORD lineCount, DWORD vertexIndexCount, BYTE* vertexBuffer, DWORD VertexTypeDesc)
{
	// Define vertices and setup vector
	std::vector<BYTE, aligned_allocator<BYTE, 4>> vertices;
	vertices.resize(sizeof(D3DTLVERTEX) * lineCount * 2);
	BYTE* verticesData = vertices.data();
	DWORD verticesCount = 0;

	// Add vertices to vector
	for (DWORD i = 0; i < lineCount; i++)
	{
		if (line[i].v1 < vertexIndexCount && line[i].v2 < vertexIndexCount)
		{
			CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, line[i].v1, VertexTypeDesc);
			CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, line[i].v2, VertexTypeDesc);
		}
	}

	if (verticesCount)
	{
		// Pass the vertex data to the rendering pipeline
		DrawPrimitive(D3DPT_LINELIST, VertexTypeDesc, vertices.data(), verticesCount, 0, 1);
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::DrawExecuteTriangle(D3DTRIANGLE* triangle, WORD triangleCount, DWORD vertexIndexCount, BYTE* vertexBuffer, DWORD VertexTypeDesc)
{
	// Compute buffer size
	DWORD BufferSize;
	{
		bool LastRecord = false;
		DWORD Count = 0, MaxCount = 0;
		for (DWORD i = 0; i < triangleCount; i++)
		{
			bool IsStartRecord = (triangle[i].wFlags & 0x1F) < D3DTRIFLAG_STARTFLAT(30);
			if (IsStartRecord != LastRecord)
			{
				MaxCount = max(Count, MaxCount);
				Count = (IsStartRecord) ? 3 : 4;
			}
			else
			{
				Count += (IsStartRecord) ? 3 : 1;
			}
			LastRecord = IsStartRecord;
		}
		BufferSize = sizeof(D3DTLVERTEX) * max(Count, MaxCount);
	}

	std::vector<BYTE, aligned_allocator<BYTE, 4>> vertices;
	vertices.resize(BufferSize);
	BYTE* verticesData = vertices.data();
	DWORD verticesCount = 0;

	D3DPRIMITIVETYPE PrimitiveType = D3DPT_TRIANGLELIST;

	LONG LastCullMode = D3DTRIFLAG_START;
	LONG CullRecordCount = 0;

	for (DWORD i = 0; i < triangleCount; i++)
	{
		// Flags for this triangle
		WORD TriFlags = (triangle[i].wFlags & 0x1F);

		// START loads all three vertices
		if (TriFlags < D3DTRIFLAG_STARTFLAT(30))
		{
			if (triangle[i].v1 < vertexIndexCount && triangle[i].v2 < vertexIndexCount && triangle[i].v3 < vertexIndexCount)
			{
				PrimitiveType = D3DPT_TRIANGLELIST;

				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, triangle[i].v1, VertexTypeDesc);
				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, triangle[i].v2, VertexTypeDesc);
				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, triangle[i].v3, VertexTypeDesc);

				LastCullMode = D3DTRIFLAG_START;
				CullRecordCount = TriFlags;
			}
		}
		// EVEN and ODD load just v3 with even or odd culling
		else if (TriFlags == D3DTRIFLAG_EVEN || TriFlags == D3DTRIFLAG_ODD)
		{
			// Set primative type
			if (LastCullMode == D3DTRIFLAG_START)
			{
				// Even cull modes indicates a triangle fan
				if (TriFlags == D3DTRIFLAG_EVEN)
				{
					PrimitiveType = D3DPT_TRIANGLEFAN;
				}
				// Odd or mismatching cull modes indicates a triangle strip
				else
				{
					PrimitiveType = D3DPT_TRIANGLESTRIP;
				}
			}

			if (triangle[i].v3 < vertexIndexCount)
			{
				CopyConvertExecuteVertex(verticesData, verticesCount, vertexBuffer, triangle[i].v3, VertexTypeDesc);
			}

			LastCullMode = TriFlags;
			CullRecordCount--;
		}

		// Check next records
		bool AtEndOfList = !(i + 1U < triangleCount);
		LONG NextRecord = (i + 1U < triangleCount) ? ((triangle[i + 1].wFlags & 0x1F) < 30 ? D3DTRIFLAG_START : D3DTRIFLAG_EVEN) : 0;
		LONG NextNextRecord = (i + 2U < triangleCount) ? ((triangle[i + 2].wFlags & 0x1F) < 30 ? D3DTRIFLAG_START : D3DTRIFLAG_EVEN) : 0;

		// Draw primitaves once at the end of the list
		if (verticesCount &&								// There primatives to draw
			(AtEndOfList ||									// There are no more records, or
				(NextRecord == D3DTRIFLAG_START &&			// Next record is a new START
					(LastCullMode != D3DTRIFLAG_START || NextNextRecord != D3DTRIFLAG_START))))
		{
			if (CullRecordCount > 0)
			{
				LOG_LIMIT(100, __FUNCTION__ << " Warning: drawing before all records have been culled: " << CullRecordCount);
			}

			// Pass the vertex data to the rendering pipeline
			DrawPrimitive(PrimitiveType, VertexTypeDesc, vertices.data(), verticesCount, 0, 1);

			// Reset variables for next list
			verticesCount = 0;
			verticesData = vertices.data();
		}
	}

	return D3D_OK;
}

void m_IDirect3DDeviceX::ClearViewport(m_IDirect3DViewportX* lpViewportX)
{
	if (lpViewportX == lpCurrentViewportX)
	{
		lpCurrentViewport = nullptr;
		lpCurrentViewportX = nullptr;
	}
}

void m_IDirect3DDeviceX::SetD3D(m_IDirect3DX* lpD3D)
{
	if (!lpD3D)
	{
		return;
	}

	if (D3DInterface && D3DInterface != lpD3D)
	{
		Logging::Log() << __FUNCTION__ << " Warning: Direct3D interface has already been created!";
	}

	D3DInterface = lpD3D;
}

void m_IDirect3DDeviceX::ClearD3D(m_IDirect3DX* lpD3D)
{
	if (lpD3D != D3DInterface)
	{
		Logging::Log() << __FUNCTION__ << " Warning: released Direct3D interface does not match cached one!";
	}

	D3DInterface = nullptr;
}

HRESULT m_IDirect3DDeviceX::SetTextureHandle(DWORD TexHandle)
{
	DWORD tHandle = NULL;
	IDirect3DTexture2* lpTexture = nullptr;

	if (TexHandle)
	{
		m_IDirect3DTextureX* pTextureX = GetTexture(TexHandle);
		if (pTextureX)
		{
			lpTexture = (IDirect3DTexture2*)pTextureX->GetWrapperInterfaceX(0);
			if (lpTexture)
			{
				tHandle = TexHandle;
			}
			else
			{
				LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture address!");
			}
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get texture handle!");
		}
	}

	DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREHANDLE].State = tHandle;
	SetTexture(0, lpTexture);

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetMaterialHandle(DWORD MatHandle)
{
	DeviceStates.LightState[D3DLIGHTSTATE_MATERIAL] = NULL;

	D3DMATERIAL Material = {};
	Material.dwSize = sizeof(D3DMATERIAL);

	if (MatHandle)
	{
		m_IDirect3DMaterialX* pMaterialX = GetMaterial(MatHandle);
		if (pMaterialX)
		{
			if (FAILED(pMaterialX->GetMaterial(&Material)))
			{
				return DDERR_INVALIDPARAMS;
			}
		}
		else
		{
			LOG_LIMIT(100, __FUNCTION__ << " Error: could not get material handle!");
			return D3D_OK;
		}
	}

	DeviceStates.LightState[D3DLIGHTSTATE_MATERIAL] = MatHandle;

	D3DMATERIAL7 Material7;

	ConvertMaterial(Material7, Material);

	SetMaterial(&Material7);

	if (Material.hTexture)
	{
		SetTextureHandle(Material.hTexture);
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetStateBlockRenderState(D3DRENDERSTATETYPE State, DWORD Value)
{
	if ((UINT)State >= D3D_MAXRENDERSTATES)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().UnmappedRenderState[State] = Value;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9RenderState(D3DRENDERSTATETYPE State, LPDWORD lpValue) const
{
	if (!lpValue)
	{
		return DDERR_INVALIDPARAMS;
	}
	if ((UINT)State >= D3D_MAXRENDERSTATES)
	{
		*lpValue = (DWORD)-1;
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.RenderState[State].Set)
	{
		*lpValue = DeviceStates.RenderState[State].State;
	}
	else
	{
		*lpValue = DefaultRenderState[State];
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9RenderState(D3DRENDERSTATETYPE State, DWORD Value)
{
	if ((UINT)State >= D3D_MAXRENDERSTATES)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().RenderState[State] = Value;
	}

	BatchStates.RenderState[State] = Value;

	DeviceStates.RenderState[State].Set = (DefaultRenderState[State] != Value);
	DeviceStates.RenderState[State].State = Value;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9TextureStageState(DWORD Stage, D3DTEXTURESTAGESTATETYPE Type, LPDWORD lpValue) const
{
	if (!lpValue)
	{
		return DDERR_INVALIDPARAMS;
	}
	if (Stage >= D3DHAL_TSS_MAXSTAGES || (UINT)Type >= MaxTextureStageStates)
	{
		*lpValue = (DWORD)-1;
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.TextureStageState[Stage][Type].Set)
	{
		*lpValue = DeviceStates.TextureStageState[Stage][Type].State;
	}
	else
	{
		*lpValue = DefaultTextureStageState[Stage][Type];
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9TextureStageState(DWORD Stage, D3DTEXTURESTAGESTATETYPE Type, DWORD Value)
{
	if (Stage >= D3DHAL_TSS_MAXSTAGES || (UINT)Type >= MaxTextureStageStates)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().TextureStageState[Stage][Type] = Value;
	}

	BatchStates.TextureStageState[Stage][Type] = Value;

	DeviceStates.TextureStageState[Stage][Type].Set = (DefaultTextureStageState[Stage][Type] != Value);
	DeviceStates.TextureStageState[Stage][Type].State = Value;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9SamplerState(DWORD Sampler, D3DSAMPLERSTATETYPE Type, LPDWORD lpValue) const
{
	if (!lpValue)
	{
		return DDERR_INVALIDPARAMS;
	}
	if (Sampler >= D3DHAL_TSS_MAXSTAGES || (UINT)Type >= D3DHAL_TEXTURESTATEBUF_SIZE)
	{
		*lpValue = (DWORD)-1;
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.SamplerState[Sampler][Type].Set)
	{
		*lpValue = DeviceStates.SamplerState[Sampler][Type].State;
	}
	else
	{
		*lpValue = DefaultSamplerState[Sampler][Type];
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9SamplerState(DWORD Sampler, D3DSAMPLERSTATETYPE Type, DWORD Value)
{
	if (Sampler >= D3DHAL_TSS_MAXSTAGES || (UINT)Type >= D3DHAL_TEXTURESTATEBUF_SIZE)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().SamplerState[Sampler][Type] = Value;
	}

	BatchStates.SamplerState[Sampler][Type] = FixSamplerState(Type, Value);

	DeviceStates.SamplerState[Sampler][Type].Set = (DefaultSamplerState[Sampler][Type] != Value);
	DeviceStates.SamplerState[Sampler][Type].State = Value;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9Light(DWORD Index, D3DLIGHT9* lpLight) const
{
	if (!lpLight)
	{
		return DDERR_INVALIDPARAMS;
	}

	auto it = DeviceStates.Light.find(Index);
	if (it != DeviceStates.Light.end())
	{
		*lpLight = it->second;
		return D3D_OK;
	}

	return DDERR_INVALIDPARAMS;
}

HRESULT m_IDirect3DDeviceX::SetD9Light(DWORD Index, const D3DLIGHT9* lpLight)
{
	if (!lpLight)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().Light[Index] = *lpLight;
	}

	BatchStates.Light[Index] = FixLight(*lpLight);

	DeviceStates.Light[Index] = *lpLight;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9LightEnable(DWORD Index, LPBOOL lpEnable) const
{
	if (!lpEnable)
	{
		return DDERR_INVALIDPARAMS;
	}

	auto it = DeviceStates.LightEnable.find(Index);
	if (it != DeviceStates.LightEnable.end())
	{
		*lpEnable = it->second;
	}
	else
	{
		*lpEnable = FALSE;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::D9LightEnable(DWORD Index, BOOL Enable)
{
	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().LightEnable[Index] = Enable;
	}

	BatchStates.LightEnable[Index] = Enable;

	DeviceStates.LightEnable[Index] = Enable;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9ClipPlane(DWORD Index, float* lpPlane) const
{
	if (!lpPlane)
	{
		return DDERR_INVALIDPARAMS;
	}
	if (Index >= MaxClipPlaneIndex)
	{
		*(FLOAT4*)lpPlane = {};
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.ClipPlane[Index].Set)
	{
		*(FLOAT4*)lpPlane = DeviceStates.ClipPlane[Index].Plane;
	}
	else
	{
		*(FLOAT4*)lpPlane = *(FLOAT4*)&DefaultClipPlane;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9ClipPlane(DWORD Index, const float* lpPlane)
{
	if (!lpPlane || Index >= MaxClipPlaneIndex)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().ClipPlane[Index] = *(FLOAT4*)lpPlane;
	}

	BatchStates.ClipPlane[Index] = *(FLOAT4*)lpPlane;

	DeviceStates.ClipPlane[Index].Set = true;
	DeviceStates.ClipPlane[Index].Plane = *(FLOAT4*)lpPlane;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9Viewport(D3DVIEWPORT9* lpViewport) const
{
	if (!lpViewport)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.Viewport.Set)
	{
		*lpViewport = DeviceStates.Viewport.View;
	}
	else
	{
		*lpViewport = DefaultViewport;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9Viewport(const D3DVIEWPORT9* lpViewport)
{
	if (!lpViewport)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().Viewport[0] = *lpViewport;
	}

	DeviceStates.Viewport.Set = true;
	DeviceStates.Viewport.View = *lpViewport;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9Material(D3DMATERIAL9* lpMaterial) const
{
	if (!lpMaterial)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (DeviceStates.Material.Set)
	{
		*lpMaterial = DeviceStates.Material.Material;
	}
	else
	{
		*lpMaterial = DefaultMaterial;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9Material(const D3DMATERIAL9* lpMaterial)
{
	if (!lpMaterial)
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().Material[0] = *lpMaterial;
	}

	BatchStates.Material.Set = true;

	DeviceStates.Material.Set = true;
	DeviceStates.Material.Material = *lpMaterial;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::GetD9Transform(D3DTRANSFORMSTATETYPE State, D3DMATRIX* lpMatrix) const
{
	if (!lpMatrix)
	{
		return DDERR_INVALIDPARAMS;
	}
	if (!IsValidTransformState(State))
	{
		*lpMatrix = {};
		return DDERR_INVALIDPARAMS;
	}

	auto it = DeviceStates.Matrix.find(State);
	if (it != DeviceStates.Matrix.end())
	{
		*lpMatrix = it->second;
	}
	else
	{
		*lpMatrix = DefaultIdentityMatrix;
	}

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::SetD9Transform(D3DTRANSFORMSTATETYPE State, const D3DMATRIX* lpMatrix)
{
	if (!lpMatrix || !IsValidTransformState(State))
	{
		return DDERR_INVALIDPARAMS;
	}

	if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
	{
		StateBlock.Data[StateBlock.RecordingToken].RecordState.value().Matrix[State] = *lpMatrix;
	}

	BatchStates.Matrix[State] = FixMatrix(*lpMatrix, State, DeviceStates.Viewport.ViewportScale, DeviceStates.Viewport.UseViewportScale);

	DeviceStates.Matrix[State] = *lpMatrix;

	return D3D_OK;
}

HRESULT m_IDirect3DDeviceX::D9MultiplyTransform(D3DTRANSFORMSTATETYPE State, const D3DMATRIX* lpMatrix)
{
	if (!lpMatrix)
	{
		return DDERR_INVALIDPARAMS;
	}

	D3DMATRIX Matrix;
	HRESULT hr = GetD9Transform(State, &Matrix);

	if (SUCCEEDED(hr))
	{
		D3DMATRIX result = {};
		D3DXMatrixMultiply(&result, lpMatrix, &Matrix);

		if (StateBlock.IsRecording && StateBlock.Data[StateBlock.RecordingToken].RecordState.has_value())
		{
			StateBlock.Data[StateBlock.RecordingToken].RecordState.value().Matrix[State] = result;
		}

		BatchStates.Matrix[State] = FixMatrix(result, State, DeviceStates.Viewport.ViewportScale, DeviceStates.Viewport.UseViewportScale);

		DeviceStates.Matrix[State] = result;
	}

	return hr;
}

void m_IDirect3DDeviceX::PrepDevice()
{
	if (ddrawParent && (RequiresStateRestore || ddrawParent->GetLastDrawDevice() != (DWORD)this))
	{
		RestoreStates();
	}

	// Set batched states
	if (Config.Dd7to9)
	{
		for (const auto& entry : BatchStates.RenderState)
		{
			(*d3d9Device)->SetRenderState(entry.first, entry.second);
		}
		BatchStates.RenderState.clear();
		for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
		{
			for (const auto& entry : BatchStates.TextureStageState[x])
			{
				(*d3d9Device)->SetTextureStageState(x, entry.first, entry.second);
			}
			BatchStates.TextureStageState[x].clear();
			for (const auto& entry : BatchStates.SamplerState[x])
			{
				(*d3d9Device)->SetSamplerState(x, entry.first, entry.second);
			}
			BatchStates.SamplerState[x].clear();
		}
		for (const auto& entry : BatchStates.Light)
		{
			(*d3d9Device)->SetLight(entry.first, &entry.second);
		}
		BatchStates.Light.clear();
		for (const auto& entry : BatchStates.LightEnable)
		{
			(*d3d9Device)->LightEnable(entry.first, entry.second);
		}
		BatchStates.LightEnable.clear();
		for (const auto& entry : BatchStates.ClipPlane)
		{
			(*d3d9Device)->SetClipPlane(entry.first, reinterpret_cast<const float*>(&entry.second));
		}
		BatchStates.ClipPlane.clear();
		if (BatchStates.Material.Set)
		{
			(*d3d9Device)->SetMaterial(&DeviceStates.Material.Material);
			BatchStates.Material.Set = false;
		}
		for (const auto& entry : BatchStates.Matrix)
		{
			(*d3d9Device)->SetTransform(entry.first, &entry.second);
		}
		if (Config.DdrawDisableLighting)
		{
			(*d3d9Device)->SetRenderState(D3DRS_LIGHTING, FALSE);
		}
	}
}

HRESULT m_IDirect3DDeviceX::RestoreStates()
{
	if (!d3d9Device || !*d3d9Device)
	{
		RequiresStateRestore = true;
		Logging::Log() << __FUNCTION__ " Error: Failed to restore the device state!";
		return DDERR_GENERIC;
	}

	// Reset vars
	RequiresStateRestore = false;
	ddrawParent->SetLastDrawDevice((DWORD)this);

	// Reset device state
	ddrawParent->ApplyStateBlock();

	// Restore render states
	for (const auto& State : D9RenderStateList)
	{
		if (DeviceStates.RenderState[State].Set)
		{
			(*d3d9Device)->SetRenderState(State, DeviceStates.RenderState[State].State);
		}
	}

	for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
	{
		// Restore texture states
		for (const auto& State : D9TextureStateList)
		{
			if (DeviceStates.TextureStageState[x][State].Set)
			{
				(*d3d9Device)->SetTextureStageState(x, State, DeviceStates.TextureStageState[x][State].State);
			}
		}

		// Restore sampler states
		for (const auto& State : D9SamplerStateList)
		{
			if (DeviceStates.SamplerState[x][State].Set)
			{
				(*d3d9Device)->SetSamplerState(x, State, FixSamplerState(State, DeviceStates.SamplerState[x][State].State));
			}
		}
	}

	// Restore lights
	for (const auto& entry : DeviceStates.Light)
	{
		D3DLIGHT9 Light = FixLight(entry.second);
		(*d3d9Device)->SetLight(entry.first, &Light);
	}

	// Restore light enable
	for (const auto& entry : DeviceStates.LightEnable)
	{
		if (entry.second)
		{
			(*d3d9Device)->LightEnable(entry.first, entry.second);
		}
	}

	// Restore clip planes
	for (UINT Index = 0; Index < MaxClipPlaneIndex; Index++)
	{
		if (DeviceStates.ClipPlane[Index].Set)
		{
			(*d3d9Device)->SetClipPlane(Index, reinterpret_cast<float*>(&DeviceStates.ClipPlane[Index].Plane));
		}
	}

	// Restore viewport
	if (DeviceStates.Viewport.Set)
	{
		(*d3d9Device)->SetViewport(&DeviceStates.Viewport.View);
	}

	// Restore material
	if (DeviceStates.Material.Set)
	{
		(*d3d9Device)->SetMaterial(&DeviceStates.Material.Material);
	}

	// Restore transform
	for (const auto& entry : DeviceStates.Matrix)
	{
		D3DMATRIX Matrix = FixMatrix(entry.second, entry.first, DeviceStates.Viewport.ViewportScale, DeviceStates.Viewport.UseViewportScale);
		(*d3d9Device)->SetTransform(entry.first, &Matrix);
	}

	return D3D_OK;
}

void m_IDirect3DDeviceX::AfterResetDevice()
{
	// Get defaults
	(*d3d9Device)->GetViewport(&DefaultViewport);

	// If viewport isn't set then set to default
	if (!DeviceStates.Viewport.Set)
	{
		DeviceStates.Viewport.View = DefaultViewport;
	}
}

void m_IDirect3DDeviceX::ClearDdraw()
{
	ddrawParent = nullptr;
	colorkeyPixelShader = nullptr;
	fixupVertexShader = nullptr;
	d3d9Device = nullptr;
	RequiresStateRestore = true;
}

void m_IDirect3DDeviceX::SetDefaults()
{
	// Reset defaults flag
	bSetDefaults = false;
	RequiresStateRestore = true;

	// Reset Homogeneous flag
	ConvertHomogeneous.IsTransformViewSet = false;

	// Render states
	if (ClientDirectXVersion > 1)
	{
		for (auto State : { 1, 4, 5, 6, 10, 11, 30, 31, 32, 33, 39, 41, 43, 46, 47 })
		{
			DeviceStates.RenderState[State].State = FALSE;
		}
		for (UINT State = 64; State < 96; State++)
		{
			DeviceStates.RenderState[State].State = FALSE;
		}
		for (auto State : { 3, 4, 17, 18, 44, 45, 49 })
		{
			DeviceStates.RenderState[State].State = TRUE;
		}
		DeviceStates.RenderState[D3DRENDERSTATE_ROP2].State = R2_COPYPEN;
		DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREMAPBLEND].State = D3DTBLEND_MODULATE;
		DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREPERSPECTIVE].State = (ClientDirectXVersion == 2 ? FALSE : TRUE);
		DeviceStates.rsMap161 = (DWORD)-1;
		DeviceStates.rsMap162 = (DWORD)-1;
		DeviceStates.rsMap195 = (DWORD)-1;

		if (ClientDirectXVersion != 7)
		{
			DeviceStates.RenderState[D3DRENDERSTATE_NONE].State = 0;
			DeviceStates.RenderState[D3DRENDERSTATE_ANTIALIAS].State = 0;
			DeviceStates.RenderState[D3DRENDERSTATE_EDGEANTIALIAS].State = 0;

			DeviceStates.rsMap139 = (DWORD)-1;
			DeviceStates.rsMap140 = (DWORD)-1;
			DeviceStates.rsMap141 = (DWORD)-1;
		}
		else
		{
			DeviceStates.RenderState[D3DRENDERSTATE_ANTIALIAS].State = (DWORD)-1;
			DeviceStates.RenderState[D3DRENDERSTATE_EXTENTS].State = 0;
			DeviceStates.RenderState[D3DRENDERSTATE_COLORKEYBLENDENABLE].State = 0;
		}

		if (ClientDirectXVersion > 2)
		{
			for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
			{
				for (auto State : { 12, 14, 16, 18 })
				{
					DeviceStates.TextureStageState[x][State].State = TRUE;
				}
			}
		}
	}

	// Set DirectDraw defaults
	for (UINT x = 1; x < D3DHAL_TSS_MAXSTAGES; x++)
	{
		SetD9TextureStageState(x, D3DTSS_TEXCOORDINDEX, 0);
	}

	if (ClientDirectXVersion < 3)
	{
		SetRenderState(D3DRENDERSTATE_TEXTUREMAPBLEND, D3DTBLEND_MODULATE);
		SetD9RenderState(D3DRS_SPECULARENABLE, TRUE);
	}
	else
	{
		SetD9TextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
	}

	float Value = -0.0f;
	SetD9RenderState(D3DRS_DEPTHBIAS, *(DWORD*)&Value);

	// Get default structures
	(*d3d9Device)->GetDeviceCaps(&Caps9);
	ddrawParent->GetDefaultViewport(&DefaultViewport);

	// Set defaults
	DeviceStates.Viewport.View = DefaultViewport;
}

void m_IDirect3DDeviceX::SetDrawStates(DWORD dwVertexTypeDesc, DWORD& dwFlags, DWORD DirectXVersion)
{
	PrepDevice();

	// Handle dwFlags
	if (DirectXVersion < 7)
	{
		// dwFlags (D3DDP_WAIT) can be ignored safely

		if (dwFlags & D3DDP_DONOTCLIP)
		{
			DrawStates.rsClipping = TRUE;
			(*d3d9Device)->SetRenderState(D3DRS_CLIPPING, FALSE);
		}
		if ((dwFlags & D3DDP_DONOTLIGHT) || !(dwVertexTypeDesc & D3DFVF_NORMAL))
		{
			dwFlags |= D3DDP_DONOTLIGHT;
			DrawStates.rsLighting = TRUE;
			(*d3d9Device)->SetRenderState(D3DRS_LIGHTING, FALSE);
		}
		if (dwFlags & D3DDP_DONOTUPDATEEXTENTS)
		{
			// ToDo: fix Extents see SetRenderState() implementation
			//GetRenderState(D3DRENDERSTATE_EXTENTS, &DrawStates.rsExtents);
			//SetRenderState(D3DRENDERSTATE_EXTENTS, FALSE);
		}
	}

	// Prepare render target surface
	if (lpCurrentRenderTargetX)
	{
		lpCurrentRenderTargetX->PrepareRenderTarget();

		if (ddrawParent->GetRenderTargetSurface() != lpCurrentRenderTargetX)
		{
			ddrawParent->SetRenderTargetSurface(lpCurrentRenderTargetX);
		}

		if (DeviceStates.RenderState[D3DRS_ZENABLE].Set)
		{
			(*d3d9Device)->SetRenderState(D3DRS_ZENABLE, DeviceStates.RenderState[D3DRS_ZENABLE].State);
		}
	}

	// Check Multi-Sample Type
	if (RenderTargetMultiSampleType == D3DMULTISAMPLE_NONE && DeviceStates.RenderState[D3DRS_MULTISAMPLEANTIALIAS].State)
	{
		LOG_LIMIT(100, __FUNCTION__ << " Warning: MultiSample render state enabled on a non-MultiSample render target!");
	}

	// Need to always set viewport
	(*d3d9Device)->SetViewport(&DeviceStates.Viewport.View);

	if (Config.DdrawFixByteAlignment > 1)
	{
		for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
		{
			if (CurrentTextureSurfaceX[x] && CurrentTextureSurfaceX[x]->GetWasBitAlignLocked())
			{
				GetD9SamplerState(x, D3DSAMP_MINFILTER, &DrawStates.ssMinFilter[x]);
				GetD9SamplerState(x, D3DSAMP_MAGFILTER, &DrawStates.ssMagFilter[x]);

				(*d3d9Device)->SetSamplerState(x, D3DSAMP_MINFILTER, Config.DdrawFixByteAlignment == 2 ? D3DTEXF_POINT : D3DTEXF_LINEAR);
				(*d3d9Device)->SetSamplerState(x, D3DSAMP_MAGFILTER, Config.DdrawFixByteAlignment == 2 ? D3DTEXF_POINT : D3DTEXF_LINEAR);
			}
		}
	}
	if (CurrentTextureSurfaceX[0] &&
		DeviceStates.RenderState[D3DRENDERSTATE_TEXTUREMAPBLEND].State == D3DTBLEND_MODULATE &&
		DeviceStates.TextureStageState[0][D3DTSS_ALPHAOP].Set == FALSE)
	{
		// texture alpha replaces; if no texture alpha, use vertex alpha.
		if (CurrentTextureSurfaceX[0]->HasAlphaChannel())
		{
			(*d3d9Device)->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
		}
		else
		{
			(*d3d9Device)->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_MODULATE);
		}
	}
	for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
	{
		// Set textures
		if (CurrentTextureSurfaceX[x])
		{
			// Path B: collapse animation-pool members to one canonical surface
			// (no-op when DdrawCollapseAnimationPools=0; identity-returns).
			m_IDirectDrawSurfaceX* effective = CurrentTextureSurfaceX[x]->GetCanonicalForPathB();
			IDirect3DTexture9* pTexture9 = effective->GetD3d9Texture();
			if (pTexture9)
			{
				// Canonical Identity Layer: converge drifting pool/atlas-composite
				// content onto frozen canonical textures so Remix sees one stable
				// hash per content (no-op when DdrawCanonicalRebind=0).
				IDirect3DTexture9* canonTex = Stage3CanonicalResolve(pTexture9, d3d9Device);
				if (canonTex) pTexture9 = canonTex;
				(*d3d9Device)->SetTexture(x, pTexture9);
				Stage3OnRemixBind(pTexture9, /*isCrop=*/false);
				// Phase A.7 v2: capture content at first bind for static textures
				// (loaded once at init and never re-dirtied -- SetDirtyFlag hook
				// misses these). Internal dedup makes this one-shot per surface.
				// NOTE: we capture from the ORIGINAL surface (not the redirect target)
				// so the corpus still enumerates pool members.
				CurrentTextureSurfaceX[x]->CaptureForPhaseA7FirstBind();
			}
			// Generate MipMap levels
			if (DeviceStates.TextureStageState[x][D3DTSS_MIPFILTER].State != D3DTEXF_NONE && !CurrentTextureSurfaceX[x]->IsMipMapGenerated())
			{
				CurrentTextureSurfaceX[x]->GenerateMipMapLevels();
			}
		}
		else
		{
			(*d3d9Device)->SetTexture(x, nullptr);
		}
	}
	if (DeviceStates.RenderState[D3DRENDERSTATE_COLORKEYENABLE].State ||
		DeviceStates.RenderState[D3DRENDERSTATE_COLORKEYBLENDENABLE].State)
	{
		// Check for color key alpha texture
		for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
		{
			if (CurrentTextureSurfaceX[x] && CurrentTextureSurfaceX[x]->IsColorKeyTexture() && CurrentTextureSurfaceX[x]->GetD3d9DrawTexture())
			{
				dwFlags |= D3DDP_DXW_ALPHACOLORKEY;
				LPDIRECT3DTEXTURE9 ckDrawTex = CurrentTextureSurfaceX[x]->GetD3d9DrawTexture();
				(*d3d9Device)->SetTexture(x, ckDrawTex);
				Stage3OnRemixBind(ckDrawTex, /*isCrop=*/false);
			}
		}
		if (dwFlags & D3DDP_DXW_ALPHACOLORKEY)
		{
			GetD9RenderState(D3DRS_ALPHATESTENABLE, &DrawStates.rsAlphaTestEnable);
			GetD9RenderState(D3DRS_ALPHAFUNC, &DrawStates.rsAlphaFunc);
			GetD9RenderState(D3DRS_ALPHAREF, &DrawStates.rsAlphaRef);

			(*d3d9Device)->SetRenderState(D3DRS_ALPHATESTENABLE, TRUE);
			(*d3d9Device)->SetRenderState(D3DRS_ALPHAFUNC, D3DCMP_GREATER);
			(*d3d9Device)->SetRenderState(D3DRS_ALPHAREF, (DWORD)0x01);
		}
	}
	if ((dwFlags & D3DDP_DXW_COLORKEYENABLE) && ddrawParent)
	{
		if (!colorkeyPixelShader || !*colorkeyPixelShader)
		{
			colorkeyPixelShader = ddrawParent->GetColorKeyPixelShader();
		}
		if (colorkeyPixelShader && *colorkeyPixelShader)
		{
			(*d3d9Device)->SetPixelShader(*colorkeyPixelShader);
			(*d3d9Device)->SetPixelShaderConstantF(0, DrawStates.lowColorKey, 1);
			(*d3d9Device)->SetPixelShaderConstantF(1, DrawStates.highColorKey, 1);
		}
	}
	/*if ((dwVertexTypeDesc & D3DFVF_XYZRHW) && d3d9Device && *d3d9Device && ddrawParent)
	{
		if (!fixupVertexShader || !*fixupVertexShader)
		{
			fixupVertexShader = ddrawParent->GetFixupVertexShader();
		}
		if (fixupVertexShader && *fixupVertexShader)
		{
			// Set pixel center offset (-0.5 if AlternatePixelCenter is enabled)
			float pixelOffset[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			if (Config.DdrawAlternatePixelCenter)
			{
				pixelOffset[0] = -0.5f;
				pixelOffset[1] = -0.5f;
			}
			(*d3d9Device)->SetVertexShaderConstantF(1, pixelOffset, 1);

			// Set vertex shader
			(*d3d9Device)->SetVertexShader(*fixupVertexShader);
		}
	}*/
}

void m_IDirect3DDeviceX::RestoreDrawStates(HRESULT hr, DWORD dwFlags, DWORD DirectXVersion)
{
	// Handle dwFlags
	if (DirectXVersion < 7)
	{
		if (dwFlags & D3DDP_DONOTCLIP)
		{
			(*d3d9Device)->SetRenderState(D3DRS_CLIPPING, DrawStates.rsClipping);
		}
		if (dwFlags & D3DDP_DONOTLIGHT)
		{
			(*d3d9Device)->SetRenderState(D3DRS_LIGHTING, DrawStates.rsLighting);
		}
		if (dwFlags & D3DDP_DONOTUPDATEEXTENTS)
		{
			// ToDo: fix Extents see SetRenderState() implementation
			//SetRenderState(D3DRENDERSTATE_EXTENTS, DrawStates.rsExtents);
		}
	}
	if (Config.DdrawFixByteAlignment > 1)
	{
		for (UINT x = 0; x < D3DHAL_TSS_MAXSTAGES; x++)
		{
			if (CurrentTextureSurfaceX[x] && CurrentTextureSurfaceX[x]->GetWasBitAlignLocked())
			{
				SetD9SamplerState(x, D3DSAMP_MINFILTER, DrawStates.ssMinFilter[x]);
				SetD9SamplerState(x, D3DSAMP_MAGFILTER, DrawStates.ssMagFilter[x]);
			}
		}
	}
	if (dwFlags & D3DDP_DXW_ALPHACOLORKEY)
	{
		SetD9RenderState(D3DRS_ALPHATESTENABLE, DrawStates.rsAlphaTestEnable);
		SetD9RenderState(D3DRS_ALPHAFUNC, DrawStates.rsAlphaFunc);
		SetD9RenderState(D3DRS_ALPHAREF, DrawStates.rsAlphaRef);
	}
	if (dwFlags & D3DDP_DXW_COLORKEYENABLE)
	{
		(*d3d9Device)->SetPixelShader(nullptr);
	}
	/*if ((dwVertexTypeDesc & D3DFVF_XYZRHW) && d3d9Device && *d3d9Device)
	{
		(*d3d9Device)->SetVertexShader(nullptr);
	}*/

	if (SUCCEEDED(hr))
	{
		// Mark render target as dirty
		if (lpCurrentRenderTargetX)
		{
			lpCurrentRenderTargetX->SetDirtyFlag(0);
		}
	}
}

bool m_IDirect3DDeviceX::IsLightInUse(m_IDirect3DLight* pLightX)
{
	if (ClientDirectXVersion != 7)
	{
		if (lpCurrentViewportX && lpCurrentViewportX->IsLightAttached(pLightX))
		{
			return true;
		}
	}

	return false;
}

void m_IDirect3DDeviceX::GetEnabledLightList(std::vector<DXLIGHT7>& AttachedLightList)
{
	if (ClientDirectXVersion == 7)
	{
		for (const auto& entry : DeviceStates.Light)
		{
			auto it = DeviceStates.LightEnable.find(entry.first);
			if (it != DeviceStates.LightEnable.end() && it->second)
			{
				DXLIGHT7 DxLight7 = {};
				*reinterpret_cast<D3DLIGHT9*>(&DxLight7) = entry.second;
				DxLight7.dwLightVersion = 7;
				DxLight7.dwFlags = 0;

				AttachedLightList.push_back(DxLight7);
			}
		}
	}
	else if (lpCurrentViewportX)
	{
		lpCurrentViewportX->GetEnabledLightList(AttachedLightList, this);
	}
}

void m_IDirect3DDeviceX::UpdateVertices(DWORD& dwVertexTypeDesc, LPVOID& lpVertices, DWORD dwVertexStart, DWORD dwNumVertices)
{
	if (dwVertexTypeDesc == D3DFVF_LVERTEX)
	{
		VertexCache.resize((dwVertexStart + dwNumVertices) * sizeof(DXLVERTEX9));
		ConvertLVertex(reinterpret_cast<DXLVERTEX9*>(VertexCache.data() + (dwVertexStart * sizeof(DXLVERTEX9))),
			reinterpret_cast<DXLVERTEX7*>((DWORD)lpVertices + (dwVertexStart * sizeof(DXLVERTEX7))),
			dwNumVertices);

		dwVertexTypeDesc = D3DFVF_LVERTEX9;
		lpVertices = VertexCache.data();
	}
	else if (dwVertexTypeDesc & D3DFVF_XYZRHW)
	{
		if (Config.DdrawClampVertexZDepth)
		{
			DWORD stride = GetVertexStride(dwVertexTypeDesc);
			VertexCache.resize((dwVertexStart + dwNumVertices) * stride);
			memcpy(reinterpret_cast<void*>(VertexCache.data() + (dwVertexStart * stride)),
				reinterpret_cast<void*>((DWORD)lpVertices + (dwVertexStart * stride)),
				dwNumVertices * stride);

			ClampVertices(VertexCache.data(), stride, dwNumVertices);

			lpVertices = VertexCache.data();
		}
	}
}

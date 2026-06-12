// DK2 front-end (main menu) resolution patch -- see MenuRes.h for the rationale.
//
// dk2::CFrontEndComponent::launchGame() hardcodes the menu window at 640x480:
//   push 480; push 640; mov ecx, &MyWindow_instance; call MyWindow::prepareScreenEx
// The menu GUI itself is resolution-agnostic: layouts live in a 640x480 virtual
// space and the ORIGINAL exe scales them to MyWindow::dwWidth/dwHeight at render
// time (scaleAabb_2560_1920 in Flame's naming), with the same scaling applied to
// hit-testing. That is why DiaLight's whole flame:menu-res feature is this one
// override -- and why a 2-immediate byte patch ports it completely.
//
// Site located in the GOG build by byte signature (Flame v1.70 maps launchGame at
// 0x52F140 / prepareScreenEx at 0x5581B0; the GOG build is shifted -0x3A00: call
// site 0x52BB66, prepareScreenEx 0x5547B0, MyWindow_instance 0x751C28).

#include <Windows.h>
#include <cstdint>
#include <cstdio>
#include "MenuRes.h"
#include "Settings\Settings.h"
#include "Logging\Logging.h"

namespace MenuRes
{
	namespace
	{
		LONG g_installTried = 0;

		constexpr uintptr_t kImgBase = 0x400000;
		// CFrontEndComponent::launchGame's prepareScreenEx call site (GOG build).
		constexpr uintptr_t kSiteVA = 0x52BB66;

		const BYTE kExpect[20] =
		{
			0x68, 0xE0, 0x01, 0x00, 0x00,	// push 480
			0x68, 0x80, 0x02, 0x00, 0x00,	// push 640
			0xB9, 0x28, 0x1C, 0x75, 0x00,	// mov ecx, offset MyWindow_instance
			0xE8, 0x36, 0x8C, 0x02, 0x00,	// call MyWindow::prepareScreenEx
		};
	}

	void InstallOnce()
	{
		const DWORD w = Config.DdrawMenuResWidth;
		const DWORD h = Config.DdrawMenuResHeight;
		if (!w && !h) return;
		if (InterlockedExchange(&g_installTried, 1) != 0) return;

		if (w < 640 || w > 7680 || h < 480 || h > 4320)
		{
			Logging::Log() << "[MENURES] ABORTED: implausible resolution " << w << "x" << h <<
				" (need both DdrawMenuResWidth and DdrawMenuResHeight, 640x480..7680x4320)";
			return;
		}

		uintptr_t base = (uintptr_t)GetModuleHandleW(nullptr);
		if (!base)
		{
			Logging::Log() << "[MENURES] ABORTED: no module base";
			return;
		}

		BYTE* p = (BYTE*)(base + (kSiteVA - kImgBase));
		for (int i = 0; i < (int)sizeof(kExpect); ++i)
		{
			if (p[i] != kExpect[i])
			{
				char msg[160];
				sprintf_s(msg, sizeof(msg),
					"[MENURES] ABORTED: site %p byte %d = %02X expected %02X (wrong exe build?) -- menu stays 640x480",
					p, i, (unsigned)p[i], (unsigned)kExpect[i]);
				Logging::Log() << msg;
				return;
			}
		}

		DWORD oldProt = 0;
		if (!VirtualProtect(p, 10, PAGE_EXECUTE_READWRITE, &oldProt))
		{
			Logging::Log() << "[MENURES] ABORTED: VirtualProtect err=" << (DWORD)GetLastError();
			return;
		}
		*(DWORD*)(p + 1) = h;	// push 480 -> push <height>
		*(DWORD*)(p + 6) = w;	// push 640 -> push <width>
		VirtualProtect(p, 10, oldProt, &oldProt);
		FlushInstructionCache(GetCurrentProcess(), p, 10);

		char msg[160];
		sprintf_s(msg, sizeof(msg),
			"[MENURES] INSTALLED: front-end menu 640x480 -> %lux%lu (launchGame site %p)",
			(unsigned long)w, (unsigned long)h, p);
		Logging::Log() << msg;
	}
}

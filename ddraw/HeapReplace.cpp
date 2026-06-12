// DK2 CRT heap replacement -- see HeapReplace.h for the rationale.
//
// DKII.exe (2,951,680-byte GOG build, loads at 0x400000, RELOCS_STRIPPED) statically
// links the VC6 CRT. Every general-purpose allocation funnels through five entry
// points; each takes the global CRT lock(9) and runs the small-block-heap (___sbh_*)
// path that is slow under churn (scroll/heavy scenes spray small alloc/free). We
// detour all five to a single private HeapCreate() heap with the Win10/11
// low-fragmentation policy -- thread-safe (serialized heap) and dramatically faster
// under fragmentation than the 1999 sbh.
//
// Entry points (verified by disassembly 2026-06-12; offsets are absolute VAs because
// the exe never relocates):
//   __nh_malloc 0x632680  (size, mode)   -- the funnel for operator new, malloc, _malloc_1
//   _realloc    0x6324B0  (ptr, size)
//   _free       0x632CA0  (ptr)
//   _calloc     0x632FA0  (count, size)
//   __msize     0x633EA0  (ptr)
// _malloc_1 (0x632660) tail-calls __nh_malloc, and operator new enters __nh_malloc
// directly, so detouring __nh_malloc captures all allocation. The five are replaced
// atomically so every post-install pointer lives on -- and is freed from -- our heap.
//
// Install timing: DllMain PROCESS_ATTACH runs before DKII's CRT mainCRTStartup, so the
// very first DK2 allocation (C++ static ctors in _initterm) already lands on our heap.
// The only callers of these addresses are DKII.exe code (its own internal CRT), which
// has not run yet -- so there is no pre-install pointer that could later be freed here.

#include <Windows.h>
#include <cstdint>
#include <cstdio>
#include "HeapReplace.h"
#include "Settings\Settings.h"
#include "Logging\Logging.h"

namespace HeapReplace
{
	namespace
	{
		HANDLE g_heap = nullptr;
		LONG   g_installTried = 0;

		constexpr uintptr_t kImgBase = 0x400000;

		struct EntryPoint
		{
			uintptr_t va;          // absolute VA in DKII.exe
			BYTE      expect[5];   // exact original prologue bytes (verify gate)
			void*     replacement; // our __cdecl handler
		};

		// --- replacement allocators (self-contained; never call the originals) ---

		void* __cdecl repl_nh_malloc(size_t size, int /*mode*/)
		{
			if (size == 0) size = 1;
			// No HEAP_GENERATE_EXCEPTIONS: return NULL on OOM so DK2's existing
			// null checks (and the engine heap's grow-on-null logic) still fire.
			return HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size);
		}

		void* __cdecl repl_calloc(size_t count, size_t size)
		{
			size_t total = count * size;
			// Match the CRT's overflow guard (it treats > 0xFFFFFFE0 as failure).
			if (size != 0 && count > (size_t)0xFFFFFFE0 / size) return nullptr;
			if (total == 0) total = 1;
			return HeapAlloc(g_heap, HEAP_ZERO_MEMORY, total);
		}

		void* __cdecl repl_realloc(void* ptr, size_t size)
		{
			if (ptr == nullptr) return repl_nh_malloc(size, 1);
			if (size == 0)
			{
				HeapFree(g_heap, 0, ptr);
				return nullptr;
			}
			return HeapReAlloc(g_heap, HEAP_ZERO_MEMORY, ptr, size);
		}

		void __cdecl repl_free(void* ptr)
		{
			if (ptr) HeapFree(g_heap, 0, ptr);
		}

		size_t __cdecl repl_msize(void* ptr)
		{
			if (!ptr) return 0;
			SIZE_T s = HeapSize(g_heap, 0, ptr);
			return (s == (SIZE_T)-1) ? 0 : (size_t)s;
		}

		EntryPoint g_entries[] =
		{
			{ 0x632680, { 0x56, 0x8B, 0x74, 0x24, 0x08 }, (void*)&repl_nh_malloc },
			{ 0x6324B0, { 0x83, 0xEC, 0x08, 0x53, 0x55 }, (void*)&repl_realloc  },
			{ 0x632CA0, { 0x51, 0x56, 0x8B, 0x74, 0x24 }, (void*)&repl_free     },
			{ 0x632FA0, { 0x53, 0x8B, 0x5C, 0x24, 0x0C }, (void*)&repl_calloc   },
			{ 0x633EA0, { 0x83, 0xEC, 0x08, 0x56, 0x6A }, (void*)&repl_msize    },
		};
		constexpr int kNumEntries = sizeof(g_entries) / sizeof(g_entries[0]);
	}

	void InstallOnce()
	{
		if (!Config.DdrawReplaceHeap) return;
		if (InterlockedExchange(&g_installTried, 1) != 0) return;

		uintptr_t base = (uintptr_t)GetModuleHandleW(nullptr);
		if (!base)
		{
			Logging::Log() << "[HEAP] install: no module base";
			return;
		}

		// Resolve actual addresses (base is 0x400000 in practice; rebase for safety).
		uintptr_t sites[kNumEntries];
		for (int i = 0; i < kNumEntries; ++i)
			sites[i] = base + (g_entries[i].va - kImgBase);

		// VERIFY ALL prologues before touching anything (all-or-nothing fail-safe).
		for (int i = 0; i < kNumEntries; ++i)
		{
			const BYTE* p = (const BYTE*)sites[i];
			for (int b = 0; b < 5; ++b)
			{
				if (p[b] != g_entries[i].expect[b])
				{
					char msg[160];
					sprintf_s(msg, sizeof(msg),
						"[HEAP] install ABORTED: entry %d @ %p byte %d = %02X expected %02X (wrong exe build?)",
						i, (void*)sites[i], b, (unsigned)p[b], (unsigned)g_entries[i].expect[b]);
					Logging::Log() << msg;
					return; // game stays 100% on its original heap
				}
			}
		}

		// Create our private heap. Serialized (thread-safe). Request the
		// low-fragmentation policy explicitly (default-on for HeapCreate heaps on
		// modern Windows, but be explicit).
		g_heap = HeapCreate(0, 0, 0);
		if (!g_heap)
		{
			Logging::Log() << "[HEAP] install ABORTED: HeapCreate failed err=" << (DWORD)GetLastError();
			return;
		}
		ULONG lfh = 2; // HEAP_LFH
		HeapSetInformation(g_heap, HeapCompatibilityInformation, &lfh, sizeof(lfh));

		// Patch all five: overwrite the 5-byte prologue with `jmp rel32` to our handler.
		int patched = 0;
		for (int i = 0; i < kNumEntries; ++i)
		{
			BYTE* p = (BYTE*)sites[i];
			DWORD oldProt = 0;
			if (!VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &oldProt))
			{
				Logging::Log() << "[HEAP] VirtualProtect failed on entry " << i << " err=" << (DWORD)GetLastError();
				continue;
			}
			int32_t rel = (int32_t)((intptr_t)g_entries[i].replacement - (intptr_t)(sites[i] + 5));
			p[0] = 0xE9; // jmp rel32
			*(int32_t*)(p + 1) = rel;
			VirtualProtect(p, 5, oldProt, &oldProt);
			FlushInstructionCache(GetCurrentProcess(), p, 5);
			++patched;
		}

		char msg[200];
		sprintf_s(msg, sizeof(msg),
			"[HEAP] INSTALLED: %d/%d CRT allocators -> private LFH heap %p (base=%p). "
			"DK2 small-block-heap + lock(9) bypassed.",
			patched, kNumEntries, (void*)g_heap, (void*)base);
		Logging::Log() << msg;
	}
}

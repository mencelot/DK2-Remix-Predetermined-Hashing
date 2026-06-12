#pragma once
// DK2 CRT heap replacement (2026-06-12).
// Redirects DKII.exe's statically-linked VC6 CRT allocator entry points to a
// modern private Win32 heap (low-fragmentation), replacing the 1999-era small-block
// heap + global-lock(9) that serializes every alloc/free across the sound + main
// threads. Ported in spirit from DiaLight/Flame's replace_heap, but applied as an
// in-memory jmp detour (we cannot recompile the exe).
//
// Gated by Config.DdrawReplaceHeap. Install is ALL-OR-NOTHING and verify-gated:
// it checks the exact 5-byte prologue of all five entry points BEFORE patching any,
// so a mismatched/updated exe leaves the game 100% on its original heap.
//
// See memory: dk2_heap_replacement_2026_06_12.

namespace HeapReplace
{
	// Call once, as early as possible (DllMain PROCESS_ATTACH, after Config is set).
	// No-op unless Config.DdrawReplaceHeap is set. Safe to call more than once.
	void InstallOnce();
}

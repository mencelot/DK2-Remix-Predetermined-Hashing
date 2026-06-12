#pragma once
// DK2 front-end (main menu) resolution patch (2026-06-12) -- port of DiaLight/Flame's
// flame:menu-res (big_resolution_fix/screen_resolution.cpp). Rewrites the two
// push-immediates (640/480) at the CFrontEndComponent::launchGame ->
// MyWindow::prepareScreenEx call site (GOG DKII.exe VA 0x52BB66) so the menu window
// opens at DdrawMenuResWidth x DdrawMenuResHeight. The original exe already scales the
// 640x480-space GUI layout (and hit-testing) to the real window size at render time
// (scaleAabb_2560_1920), so overriding the two immediates is the entire fix.
// Verify-gated on the exact 20 site bytes; any mismatch leaves the menu at 640x480.
// Inert on "-level N" launches (launchGame never runs on those).
namespace MenuRes
{
	// No-op unless DdrawMenuResWidth/Height are set. Safe to call more than once.
	void InstallOnce();
}

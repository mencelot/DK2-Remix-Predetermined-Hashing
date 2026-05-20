@echo off
cd /d "C:\Users\Alex\Downloads\dxwrapper-rtx"

REM Create BuildNo.rc file that pre-build script fails to create
echo #define FILEVERSION 1,0,8055,0 > "d3d8\BuildNo.rc"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
MSBuild dxwrapper.sln /p:Configuration=Release /p:Platform=Win32 /t:Build

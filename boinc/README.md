# Skia BOINC wrapper (Linux x86_64, glibc >= 2.31)
- `skia_wrapper`: stock BOINC 8.2.15 samples/wrapper built in Debian 11 chroot (max GLIBC_2.29, libstdc++ static); zip support stubbed (unused).
- `job.xml`: task 1 unpacks `skia_app.tar.gz` with /bin/tar, task 2 runs `skia_app/skia_app task.zip -o result.txt`.
- Version files: skia_wrapper (main), skia_app.tar.gz (175,798,226 B), job.xml. Templates: skia_in.xml / skia_out.xml (draft, not deployed).
- Tested: slot dir with symlinks, smoke task, result sha256 2170e827...cacb identical to direct run (~1.6 min, 4 cores).
- Not done: server registration, Windows/macOS, progress reporting (weights only), licence clearance.

## Rebuilding the wrapper (Linux x86_64, glibc >= 2.31)
`skia_wrapper` is not tracked (binary). Build BOINC 8.2.15 `samples/wrapper/wrapper.cpp` inside a Debian 11 (glibc 2.31) chroot against
`libboinc_api.a` and `libboinc.a`, add `zipstub.cpp` (stubs `boinc_zip`, the wrapper's zip support is unused), and link with
`-static-libstdc++ -static-libgcc`. The app archive is unpacked by `/bin/tar` in `job.xml`; give `skia_app.tar.gz` and `job.xml`
`<copy_file/>` in the app version (BOINC otherwise hands the wrapper text link files instead of the real files).
`make_tasks.py` builds the task queue (one baseline fit per Q1 cutout, then 66-point scan bundles).

# P3.3 packaging status

No portable executable has been built. The current host is macOS arm64 with
Python 3.14.1. Neither PyInstaller nor Nuitka is installed, and the configured
Podman client has no connected local machine/container registry or cached image
available to inspect. No image or dependency was downloaded and no remote host
was accessed.

The Python `skia-app` entry point is a local development runner, not a volunteer
binary. `docs/LICENSES.md` lists dependencies that still require resolution for
a distributable payload, including LGPL components in Astropy/WCSLIB and pyxdg
transitive components and bundled notices. A build would also need demonstrated
glibc <= 2.31, dependency/license clearance, and an actual Linux runtime test.

This task is blocked on a pre-provisioned offline Linux build environment with
the approved packager and dependency artifacts, plus license review. These are
still open; P3 is not validated or ready for BOINC distribution.

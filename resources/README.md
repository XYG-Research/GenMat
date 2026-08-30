# Release resource manifests

These manifests make large, gitignored datasets and checkpoints auditable
without committing the binary assets. SHA256 values are taken from GitHub's
release asset digests and verified against the corresponding local files.

The v0.1.0 assets are retained for compatibility and regression testing. Their
presence does not imply general-chemistry model competence. A future GenMat
model should be published under a new immutable release with its own dataset
provenance, held-out benchmark report, and format-v2 checkpoint metadata.

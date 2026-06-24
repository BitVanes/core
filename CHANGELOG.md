# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased] / 0.2.0

### Added
- `examples/` — `basic_chunks`, `pii_scrub`, `custom_embedder` runnable demos.
- **Embedding-guided (semantic) chunking** — `ChunkStrategy::Semantic` merges
  adjacent spans while cosine similarity stays above a threshold, keeping
  topical units whole. See `chunk::strategy`.
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`.

### Changed
- `token_count` is now an exact BPE re-count of the emitted chunk text (was a
  sum of per-span counts, which could drift by ±1 at span junctions).

### Fixed
- Overlap drop at oversized spans: the overlap tail now carries across
  oversized-split boundaries (`0.1.1` backport).

## [0.1.1] — 2026-06-24

### Added
- `cli-pdf` feature: native PDF text extraction via `pdf-extract`.
- `parallel` feature: rayon-backed `run_pipeline_batch`.
- Re-exported `OrtEmbedder`, `run_pipeline_batch` at the crate root.
- CI workflow (fmt / clippy / test / wasm size gate).

### Changed
- Documented that BPE vocab embedding is **unconditional** (tiktoken-rs has no
  toggle) — removed the phantom `embed-vocab` feature from docs.

### Fixed
- `mean_pool` stride bug when configured `dim != hidden_dim`.
- `token_count` u16 truncation: `max_tokens` is now validated `<= 65535`.
- CSV writer panic replaced with `Result` propagation.

## [0.1.0] — 2026-06-22

Initial public release: four-stage pipeline (parse → scrub → chunk → Arrow),
six OpenAI tokenizers, seven PII patterns, zero-copy Arrow FFI for wasm.

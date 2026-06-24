# AGENTS.md

Build commands for the `bitvanes-core` workspace.

## Quick verification

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## Wasm build

```bash
wasm-pack build crates/wasm --target web --out-dir pkg
```

Check bundle size (target: reasonable for a SaaS dashboard, ~4-5 MB gzipped):

```bash
gzip -c crates/wasm/pkg/bitvanes_wasm_bg.wasm | wc -c
```

## Feature flags

- `ipc`: Arrow IPC stream writer. Native-only (CLI).
- `csv`: Arrow CSV writer. Native-only.
- `embeddings`: On-device embedding generation via ONNX Runtime (ort). Native-only.
- `parallel`: Rayon parallel batch (`run_pipeline_batch`). Native-only.
- `cli-pdf`: Native PDF text extraction via pdf-extract. Native-only (browser uses PDF.js).

BPE vocab embedding is **unconditional** (tiktoken-rs embeds it with no
feature toggle), so there is no `embed-vocab` flag — every build is offline.

Run tests with all features:

```bash
cargo test --workspace --all-features
```

## Architecture

Four-stage pipeline: parse -> scrub -> chunk -> Arrow assembly.

Entry point: `bitvanes_core::pipeline::run_pipeline(bytes, &cfg)`.

Wasm export: `bitvanes_wasm::process(config_js, bytes) -> slot_id`.

The wasm module exports Arrow data via zero-copy FFI pointers (Arrow C Data
Interface). JS reads them using `arrow-js-ffi`'s `parseRecordBatch`.

## Toolchain

- Rust stable (edition 2024), MSRV 1.85, pinned via `rust-toolchain.toml`.
- `wasm-pack` 0.13+ for wasm builds.
- `wasm32-unknown-unknown` target (auto-installed by rust-toolchain.toml).

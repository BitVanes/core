# bitvanes-core

Zero-trust ETL engine for AI/RAG workloads. Written in Rust, compiled to both
`wasm32-unknown-unknown` (for the browser) and native targets (for the CLI).

## What it does

A four-stage pipeline that transforms raw documents into Apache Arrow
columnar chunks ready for vector database ingestion:

1. **Parse** - Markdown, HTML, or plain text into structural spans with
  heading ancestry and section classification.
2. **Scrub** - PII redaction (email, SSN, phone, credit card, API keys)
  via regex + Luhn validation, with an offset-delta map for projecting
  chunk positions back to the original document.
3. **Chunk** - BPE-aware splitting at structural boundaries using any of
  six OpenAI tokenizers (`cl100k_base`, `o200k_base`, `r50k_base`,
  `p50k_base`, `p50k_edit`, `o200k_harmony`).
4. **Assemble** - Arrow `RecordBatch` with 9 columns (chunk_index, text,
  token_count, source_path, heading_path, section_kind, char offsets,
  embedding placeholder), exported via zero-copy FFI pointers.

## Zero-trust guarantees

- All vocab files are embedded at compile time via `include_str!`.
- No network calls during parsing, scrubbing, or tokenization.
- In the browser, data is processed in a Web Worker sandbox.

## Workspace layout

```
core/
  Cargo.toml              workspace manifest (2 crates)
  crates/
    core/                  bitvanes-core (pure library, no wasm-bindgen)
      src/
        schema.rs          PipelineConfig, ChunkSpec, EmbeddingConfig
        error.rs           BitVanesError
        parse/             markdown.rs, html.rs, text.rs
        scrub.rs           PII redaction + OffsetMap
        tokenize.rs        BPE wrapper (tiktoken-rs)
        chunk.rs           structural-boundary chunker
        arrow_io/          batch.rs, ffi.rs, ipc.rs, csv.rs
        pipeline.rs        full pipeline orchestration
        embed.rs           Embedder trait (API foundation)
    wasm/                  bitvanes-wasm (thin #[wasm_bindgen] wrapper)
      src/lib.rs           process(), release_batch(), array_ptr(), schema_ptr()
```

## Build

```bash
# Native (for testing and the CLI)
cargo build --workspace
cargo test --workspace --all-features

# WebAssembly (for the browser)
wasm-pack build crates/wasm --target web --out-dir pkg
```

## Usage from JavaScript

```javascript
import init, { process, array_ptr, schema_ptr, release_batch, version } from './bitvanes_wasm.js';
await init();

const config = {
  format: "markdown",
  scrub: { patterns: ["email", "ssn"], custom: [] },
  chunk: { max_tokens: 512, overlap_tokens: 0, tokenizer: "cl100k_base" },
  source_label: "doc.md",
};

const slotId = process(config, new Uint8Array(fileBytes));
// Read via arrow-js-ffi using array_ptr(slotId) and schema_ptr(slotId)
// Then:
release_batch(slotId);
```

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `embed-vocab` | yes | Embed BPE vocab files at compile time (zero telemetry) |
| `ipc` | no | Arrow IPC stream output (`StreamWriter`) for CLI piping |
| `csv` | no | Arrow CSV output for data export |
| `embeddings` | no | On-device embedding generation via ONNX Runtime (`ort`) |
| `parallel` | no | Rayon-based parallel processing (native only) |
| `cli-pdf` | no | Native PDF parsing via pdf-extract (not in wasm) |

## Output schema

| Column | Type | Nullable |
|--------|------|----------|
| `chunk_index` | `UInt32` | no |
| `text` | `Utf8` | no |
| `token_count` | `UInt16` | no |
| `source_path` | `Utf8` | no |
| `heading_path` | `List<Utf8>` | yes |
| `section_kind` | `Dictionary<Int8, Utf8>` | no |
| `char_offset_start` | `UInt32` | no |
| `char_offset_end` | `UInt32` | no |
| `embedding` | `FixedSizeList<Float32, 1536>` | yes |

## Embeddings (native, `embeddings` feature)

When the `embeddings` feature is enabled, the pipeline can generate dense
vector embeddings for each chunk using ONNX Runtime:

```rust
use bitvanes_core::{OrtEmbedder, Embedder, pipeline::run_pipeline_with_embeddings};
use std::path::Path;

let embedder = OrtEmbedder::new(
    Path::new("models/model_quantized.onnx"),
    Path::new("models/tokenizer.json"),
    384,   // dimension (MiniLM-L6-v2)
    256,   // max sequence length
)?;

let batch = run_pipeline_with_embeddings(bytes, &config, &embedder)?;
// batch.column("embedding") now has real Float32 vectors.
```

The model file (e.g. MiniLM-L6-v2 quantized, ~22 MB) is fetched separately
and cached. In the browser, use `@xenova/transformers` or `onnxruntime-web`
for client-side embeddings — the Rust wasm module delegates this to JS.

## License

Dual-licensed under MIT OR Apache-2.0.

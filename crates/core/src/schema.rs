//! Domain types for `BitVanes` pipeline configuration and chunk output.
//!
//! These types form the public API of the engine. They are deliberately
//! [`serde`]-serializable so that:
//!
//! - The web worker receives a [`PipelineConfig`] from JavaScript via
//!   `serde-wasm-bindgen` (a small JSON payload - this is the *config*
//!   bridge, never the data bridge).
//! - The `bitvanes-cli` binary loads the identical config from a JSON
//!   profile file (the Milestone 4 byte-for-byte parity guarantee).
//!
//! All *data* payloads - document bytes in, Arrow [`RecordBatch`] out -
//! bypass serde entirely via raw byte buffers and FFI pointers.
//!
//! [`RecordBatch`]: arrow::record_batch::RecordBatch

use serde::{Deserialize, Serialize};

// ===========================================================================
// Pipeline configuration
// ===========================================================================

/// Top-level configuration for a single ETL pipeline run.
///
/// Identical JSON shape is consumed by both the wasm web worker and the
/// native CLI - this is the wire format of the Milestone 4 `profile.json`
/// exported by the visual studio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PipelineConfig {
    /// Source document format. Selects the parser. Defaults to
    /// [`DocumentFormat::Markdown`] via the derived `Default` impl.
    pub format: DocumentFormat,

    /// PII scrubbing profile (built-in patterns plus user-supplied regexes).
    #[serde(default)]
    pub scrub: ScrubProfile,

    /// Chunking parameters.
    #[serde(default)]
    pub chunk: ChunkConfig,

    /// Optional display name for the source file; copied verbatim into the
    /// `source_path` column of every output chunk row.
    #[serde(default)]
    pub source_label: Option<String>,

    /// Optional embedding generation configuration. When present, the
    /// pipeline generates dense vector embeddings for each chunk and fills
    /// the `embedding` column. When `None`, the column is all-null.
    ///
    /// Requires the `embeddings` cargo feature for the actual model
    /// inference. The config type is always available so that profiles
    /// can be saved/restored regardless of build configuration.
    #[serde(default)]
    pub embeddings: Option<EmbeddingConfig>,
}

/// Configuration for on-device embedding generation.
///
/// When this is present in a [`PipelineConfig`], the pipeline optionally
/// generates embeddings for each chunk using the specified model.
///
/// # Model selection
///
/// Common models and their dimensions:
///
/// | Model | Dimension | Size (ONNX int8) |
/// |-------|-----------|-------------------|
/// | `all-MiniLM-L6-v2` | 384 | ~22 MB |
/// | `bge-small-en-v1.5` | 384 | ~33 MB |
/// | `e5-small-v2` | 384 | ~33 MB |
/// | `text-embedding-3-small` | 1536 | API only |
///
/// The model file is fetched from `model_url` on first use and cached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier (e.g., `"all-MiniLM-L6-v2"`).
    pub model: String,

    /// URL to fetch the `ONNX` model file from. Cached at runtime (`IndexedDB`
    /// in the browser, a local cache directory for the CLI).
    pub model_url: String,

    /// Embedding vector dimension (e.g., 384 for MiniLM-L6-v2).
    pub dimension: usize,
}

/// Supported source document formats.
///
/// # Note on PDF
///
/// `Pdf` is only resolvable on native targets with the `cli-pdf` feature
/// enabled. The web PDF path runs through Mozilla PDF.js in a dedicated
/// JavaScript worker *before* reaching this engine, so the wasm build
/// receives the extracted text as [`Markdown`] or [`Text`].
///
/// [`Markdown`]: DocumentFormat::Markdown
/// [`Text`]: DocumentFormat::Text
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    /// GitHub-flavored Markdown via `pulldown-cmark`.
    #[default]
    Markdown,
    /// Structural JSON (one chunk per object or per leaf value).
    Json,
    /// Plain text with paragraph-based fallback splitting.
    Text,
    /// HTML via `scraper` (html5ever). Headings, paragraphs, code blocks,
    /// and list items are classified into [`SectionKind`]s.
    Html,
    /// PDF (native only; requires the `cli-pdf` feature).
    Pdf,
}

/// Which BPE tokenizer to apply when computing chunk boundaries.
///
/// All vocab files are embedded at compile time by `tiktoken-rs` — no
/// network calls ever occur. See the `tiktoken-rs` crate for model mapping
/// details.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    /// GPT-3.5 / GPT-4 tokenizer (`cl100k_base`).
    #[default]
    Cl100kBase,
    /// GPT-5 / GPT-4.1 / GPT-4o tokenizer (`o200k_base`).
    O200kBase,
    /// GPT-3 / `davinci` tokenizer (`r50k_base`, also known as `gpt2`).
    R50kBase,
    /// Code models / `text-davinci-002` / `text-davinci-003` (`p50k_base`).
    P50kBase,
    /// Edit models / `text-davinci-edit-001` (`p50k_edit`).
    P50kEdit,
    /// `gpt-oss` models / `gpt-oss-20b` / `gpt-oss-120b` (`o200k_harmony`).
    O200kHarmony,
}

/// Chunking parameters. All fields are validated by the engine at
/// pipeline-start time; see [`crate::chunk`] for the enforcement site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target maximum token count per chunk. The chunker saturates up to
    /// but never exceeds this value. Must be greater than zero.
    pub max_tokens: u32,

    /// Number of tokens of overlap between adjacent chunks. Defaults to
    /// zero (no overlap). Must be strictly less than `max_tokens`.
    #[serde(default)]
    pub overlap_tokens: u32,

    /// Which BPE tokenizer to apply.
    #[serde(default)]
    pub tokenizer: TokenizerKind,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 0,
            tokenizer: TokenizerKind::default(),
        }
    }
}

// ===========================================================================
// PII scrubbing
// ===========================================================================

/// Selection of built-in PII patterns to apply pre-tokenization.
///
/// Patterns always run BEFORE tokenization so that PII matches cannot be
/// split across chunk boundaries. The scrubber emits an offset-delta map
/// alongside the redacted text so chunk offsets can still be projected back
/// onto the original document for UI highlighting.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScrubProfile {
    /// Built-in pattern categories to enable.
    #[serde(default)]
    pub patterns: Vec<BuiltInPattern>,

    /// User-supplied regex patterns. Each is compiled into a single
    /// `RegexSet` pass alongside the built-ins.
    #[serde(default)]
    pub custom: Vec<CustomPattern>,
}

/// Built-in PII pattern categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInPattern {
    /// Email addresses (RFC-5322-ish).
    Email,
    /// US Social Security Numbers (`XXX-XX-XXXX`).
    Ssn,
    /// E.164-format phone numbers.
    Phone,
    /// Credit-card numbers: regex candidate followed by Luhn validation.
    CreditCard,
    /// AWS access-key IDs (`AKIA...`) and secret access keys.
    AwsKey,
    /// GitHub personal access tokens (`ghp_...`, `gho_...`, etc.).
    #[serde(rename = "github_pat")]
    GitHubPat,
    /// JSON Web Tokens (three base64url segments).
    Jwt,
}

/// A user-supplied regex pattern plus its replacement string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomPattern {
    /// Rust `regex` crate syntax.
    pub regex: String,
    /// Literal replacement string. Capture-group expansion is intentionally
    /// NOT supported, to keep replacement deterministic across the wasm and
    /// native builds.
    pub replacement: String,
}

// ===========================================================================
// Output domain model (pre-Arrow)
// ===========================================================================

/// Structural classification of a chunk's source span.
///
/// Preserved into the `section_kind` Dictionary column of the output
/// [`RecordBatch`](arrow::record_batch::RecordBatch) so downstream RAG
/// pipelines can route code differently from prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionKind {
    /// Plain paragraph text.
    Paragraph,
    /// Inside a fenced or indented code block.
    Code,
    /// A heading (H1-H6) that produced its own chunk.
    Heading,
    /// A table cell or table region.
    TableCell,
    /// A list item (ordered or unordered).
    ListItem,
    /// A block quote.
    BlockQuote,
    /// Front-matter or metadata region.
    FrontMatter,
}

impl SectionKind {
    /// All variants in declaration order. Used to build the Dictionary
    /// index type for the `section_kind` column.
    #[must_use]
    pub const fn all() -> &'static [SectionKind] {
        &[
            Self::Paragraph,
            Self::Code,
            Self::Heading,
            Self::TableCell,
            Self::ListItem,
            Self::BlockQuote,
            Self::FrontMatter,
        ]
    }
}

/// A single chunk produced by the engine, before being assembled into an
/// Arrow [`RecordBatch`](arrow::record_batch::RecordBatch).
///
/// `char_offset_start` / `char_offset_end` are offsets into the
/// *post-scrubbed* document text. The scrubber's offset-delta map can
/// project these back onto the original document for UI highlighting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpec {
    /// 0-based ordinal within the pipeline output.
    pub chunk_index: u32,
    /// The chunk text (post-scrubbing, post-tokenization slicing).
    pub text: String,
    /// BPE token count of `text`. Always `<= ChunkConfig::max_tokens`.
    pub token_count: u16,
    /// Label copied verbatim from [`PipelineConfig::source_label`].
    pub source_path: String,
    /// Ancestry of enclosing headings (H1 outermost, H6 innermost).
    pub heading_path: Vec<String>,
    /// Structural classification of this chunk's source span.
    pub section_kind: SectionKind,
    /// Half-open `[start, end)` character range into the scrubbed document.
    pub char_offset_start: u32,
    pub char_offset_end: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_config_defaults_are_sensible() {
        let c = ChunkConfig::default();
        assert_eq!(c.max_tokens, 512, "default max_tokens should be 512");
        assert_eq!(c.overlap_tokens, 0, "default overlap should be zero");
        assert_eq!(
            c.tokenizer,
            TokenizerKind::Cl100kBase,
            "default tokenizer should be cl100k_base"
        );
    }

    #[test]
    fn tokenizer_default_matches_cl100k() {
        // The Default impl must agree with the explicit variant, since the
        // wasm side reads `[serde(default)]` on ChunkConfig::tokenizer.
        assert_eq!(TokenizerKind::default(), TokenizerKind::Cl100kBase);
    }

    #[test]
    fn pipeline_config_round_trips_through_json() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Markdown,
            scrub: ScrubProfile {
                patterns: vec![BuiltInPattern::Email, BuiltInPattern::Ssn],
                custom: vec![CustomPattern {
                    regex: r"\bPROJECT-\d+\b".to_string(),
                    replacement: "[PROJECT-ID]".to_string(),
                }],
            },
            chunk: ChunkConfig {
                max_tokens: 256,
                overlap_tokens: 32,
                tokenizer: TokenizerKind::O200kBase,
            },
            source_label: Some("docs/architecture.md".to_string()),
            embeddings: None,
        };

        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn pipeline_config_minimal_json_uses_defaults() {
        // A minimal config: only the required fields. Everything marked
        // `#[serde(default)]` should fall back gracefully.
        let json = r#"{
            "format": "markdown",
            "chunk": { "max_tokens": 128 }
        }"#;
        let cfg: PipelineConfig = serde_json::from_str(json)
            .expect("minimal config should deserialize");

        assert_eq!(cfg.format, DocumentFormat::Markdown);
        assert_eq!(cfg.chunk.max_tokens, 128);
        assert_eq!(cfg.chunk.overlap_tokens, 0, "overlap should default to 0");
        assert_eq!(
            cfg.chunk.tokenizer,
            TokenizerKind::Cl100kBase,
            "tokenizer should default to cl100k_base"
        );
        assert!(cfg.scrub.patterns.is_empty());
        assert!(cfg.scrub.custom.is_empty());
        assert!(cfg.source_label.is_none());
    }

    #[test]
    fn document_format_serializes_lowercase() {
        // The serde rename_all = "lowercase" is part of the public wire
        // format; renaming is a breaking change for downstream JSON.
        for (kind, expected) in [
            (DocumentFormat::Markdown, "\"markdown\""),
            (DocumentFormat::Json, "\"json\""),
            (DocumentFormat::Text, "\"text\""),
            (DocumentFormat::Pdf, "\"pdf\""),
        ] {
            let s = serde_json::to_string(&kind).expect("serialize");
            assert_eq!(s, expected);
        }
    }

    #[test]
    fn section_kind_all_covers_every_variant_without_dupes() {
        let all = SectionKind::all();
        assert_eq!(all.len(), 7, "expected exactly 7 SectionKind variants");
        // Every variant must appear exactly once.
        for variant in all {
            let count = all.iter().filter(|&&v| v == *variant).count();
            assert_eq!(count, 1, "duplicate variant in SectionKind::all()");
        }
    }
}

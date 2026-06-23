//! BPE tokenization via `tiktoken-rs`.
//!
//! Provides token counting and token-boundary-aware splitting for the
//! chunker. Both `cl100k_base` (GPT-4/3.5) and `o200k_base` (GPT-4o) are
//! supported, selected by [`TokenizerKind`].
//!
//! # Zero-telemetry
//!
//! The vocab files are embedded at compile time by `tiktoken-rs` itself
//! via `include_str!`. No network calls ever occur — this is the
//! foundation of the zero-telemetry mandate.

use tiktoken_rs::CoreBPE;

use crate::error::{BitVanesError, Result};
use crate::schema::TokenizerKind;

/// A compiled BPE tokenizer. Wraps a `&'static CoreBPE` singleton (cheap
/// to construct; the singleton is initialized once on first use).
pub struct Tokenizer {
    kind: TokenizerKind,
    bpe: &'static CoreBPE,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Returns a tokenizer of the requested kind, backed by a process-wide
    /// singleton `CoreBPE`.
    ///
    /// # Errors
    ///
    /// Returns [`BitVanesError::FeatureNotEnabled`] if the requested
    /// tokenizer's vocab data is not available in this build. With the
    /// default `embed-vocab` feature, all tokenizers are always available.
    pub fn new(kind: TokenizerKind) -> Result<Self> {
        let bpe = match kind {
            TokenizerKind::Cl100kBase => tiktoken_rs::cl100k_base_singleton(),
            TokenizerKind::O200kBase => tiktoken_rs::o200k_base_singleton(),
            TokenizerKind::R50kBase => tiktoken_rs::r50k_base_singleton(),
            TokenizerKind::P50kBase => tiktoken_rs::p50k_base_singleton(),
            TokenizerKind::P50kEdit => tiktoken_rs::p50k_edit_singleton(),
            TokenizerKind::O200kHarmony => tiktoken_rs::o200k_harmony_singleton(),
        };
        Ok(Self { kind, bpe })
    }

    /// Returns the [`TokenizerKind`] of this tokenizer.
    #[must_use]
    pub const fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// Returns the BPE token count of `text`.
    #[must_use]
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }

    /// Splits `text` at the token boundary that yields at most `max_tokens`
    /// tokens. Returns `(byte_offset, token_count)` where:
    ///
    /// - `byte_offset` is the byte position in `text` at which to cut.
    ///   Always snapped to a UTF-8 character boundary.
    /// - `token_count` is the actual number of tokens in the prefix
    ///   `[..byte_offset]` (may be less than `max_tokens` due to the
    ///   char-boundary snap).
    ///
    /// If the entire text encodes to `<= max_tokens` tokens, returns
    /// `(text.len(), token_count)`.
    ///
    /// # Errors
    ///
    /// Returns [`BitVanesError::InvalidInput`] if token decoding fails
    /// (structurally impossible for valid BPE output, but handled for
    /// safety).
    pub fn split_at_token_boundary(&self, text: &str, max_tokens: usize) -> Result<(usize, usize)> {
        let tokens = self.bpe.encode_ordinary(text);
        if tokens.len() <= max_tokens {
            return Ok((text.len(), tokens.len()));
        }

        let prefix_tokens = &tokens[..max_tokens];
        let prefix_bytes = self
            .bpe
            .decode_bytes(prefix_tokens)
            .map_err(|e| BitVanesError::InvalidInput(format!("BPE decode failed: {e}")))?;

        let offset = snap_to_char_boundary(text, prefix_bytes.len());
        // Recount tokens in the snapped prefix to get an exact count.
        let actual_tokens = self.bpe.encode_ordinary(&text[..offset]).len();
        Ok((offset, actual_tokens))
    }
}

/// Moves `offset` forward to the nearest UTF-8 character boundary in `text`.
/// This prevents splitting a multi-byte character when cutting at an
/// arbitrary byte offset.
fn snap_to_char_boundary(text: &str, mut offset: usize) -> usize {
    while offset < text.len() && !text.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cl100k() -> Tokenizer {
        Tokenizer::new(TokenizerKind::Cl100kBase).expect("cl100k_base should load")
    }

    #[test]
    fn token_count_is_nonzero_for_nonempty_text() {
        let t = cl100k();
        assert!(t.count("hello world") > 0);
    }

    #[test]
    fn token_count_grows_with_text() {
        let t = cl100k();
        let short = t.count("hello");
        let long = t.count("hello world this is a longer sentence");
        assert!(long > short, "{long} should be > {short}");
    }

    #[test]
    fn cl100k_hello_world_is_one_token_for_hello() {
        let t = cl100k();
        // "hello" is a single cl100k token (id 15339).
        assert_eq!(t.count("hello"), 1);
    }

    #[test]
    fn split_at_boundary_returns_full_text_when_under_limit() {
        let t = cl100k();
        let text = "hello";
        let (offset, count) = t.split_at_token_boundary(text, 100).unwrap();
        assert_eq!(offset, text.len());
        assert_eq!(count, 1);
    }

    #[test]
    fn split_at_boundary_caps_at_max_tokens() {
        let t = cl100k();
        let text = "hello world this is a test of the tokenizer splitting";
        let (offset, count) = t.split_at_token_boundary(text, 3).unwrap();
        assert!(count <= 3, "token count {count} should be <= 3");
        assert!(offset <= text.len());
        assert!(offset > 0, "offset should be nonzero");
        // The prefix must be valid UTF-8 and non-empty.
        assert!(!text[..offset].is_empty(), "prefix should not be empty");
    }

    #[test]
    fn split_at_boundary_snaps_to_char_boundary() {
        let t = cl100k();
        // Use a multi-byte char to verify snapping.
        let text = "héllo wörld тест тест";
        let (offset, _) = t.split_at_token_boundary(text, 2).unwrap();
        assert!(
            text.is_char_boundary(offset),
            "offset must be a char boundary"
        );
    }

    #[test]
    fn o200k_base_loads() {
        let t = Tokenizer::new(TokenizerKind::O200kBase).expect("o200k_base should load");
        assert!(t.count("hello world") > 0);
    }

    #[test]
    fn all_tokenizer_variants_load_and_count() {
        // Every variant in TokenizerKind must produce a working tokenizer.
        for kind in [
            TokenizerKind::Cl100kBase,
            TokenizerKind::O200kBase,
            TokenizerKind::R50kBase,
            TokenizerKind::P50kBase,
            TokenizerKind::P50kEdit,
            TokenizerKind::O200kHarmony,
        ] {
            let t = Tokenizer::new(kind)
                .unwrap_or_else(|e| panic!("tokenizer {kind:?} should load: {e}"));
            assert!(
                t.count("hello world") > 0,
                "tokenizer {kind:?} returned zero tokens for non-empty text"
            );
            assert_eq!(t.kind(), kind);
        }
    }

    #[test]
    fn token_count_of_empty_text_is_zero() {
        let t = cl100k();
        assert_eq!(t.count(""), 0);
    }
}

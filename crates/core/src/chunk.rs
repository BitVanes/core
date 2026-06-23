//! Structural-boundary-aware BPE chunker.
//!
//! Takes a scrubbed [`Document`] and a [`ChunkConfig`] and produces a
//! [`Vec<ChunkSpec>`] by greedily packing [`TextSpan`]s into chunks up to
//! `max_tokens`, splitting over-long spans at exact BPE token boundaries.
//!
//! # Algorithm
//!
//! 1. Walk spans in order.
//! 2. For each span, compute its BPE token count.
//! 3. If the span fits in the current chunk (current + span <= `max_tokens`),
//!    add it.
//! 4. If it doesn't fit, flush the current chunk and start a new one with
//!    this span.
//! 5. If a single span exceeds `max_tokens`, split it into sub-chunks at
//!    token boundaries (each sub-chunk respects `max_tokens`).
//! 6. Every chunk inherits the `heading_path` and `section_kind` of its
//!    first span — this is how structural context is preserved into the
//!    output `RecordBatch`.
//!
//! # Overlap
//!
//! `overlap_tokens > 0` is accepted by the config but not yet implemented.
//! The chunker asserts `overlap_tokens == 0` for now.

use crate::error::{BitVanesError, Result};
use crate::parse::{Document, offset_to_u32};
use crate::schema::{ChunkConfig, ChunkSpec, SectionKind};
use crate::tokenize::Tokenizer;

/// Chunks a scrubbed [`Document`] into [`ChunkSpec`]s according to the
/// given [`ChunkConfig`].
///
/// `source_label` is copied verbatim into every chunk's `source_path`
/// field (typically from [`crate::schema::PipelineConfig::source_label`]).
///
/// # Errors
///
/// Returns [`BitVanesError::InvalidConfig`] if the config is inconsistent
/// (e.g. `overlap` >= `max_tokens`, or `max_tokens` == 0).
pub fn chunk_document(
    doc: &Document,
    cfg: &ChunkConfig,
    source_label: Option<&str>,
) -> Result<Vec<ChunkSpec>> {
    validate_config(cfg)?;

    if doc.spans.is_empty() {
        return Ok(Vec::new());
    }

    let tokenizer = Tokenizer::new(cfg.tokenizer)?;
    let max_tokens = cfg.max_tokens as usize;
    let source = source_label.unwrap_or("").to_string();

    let mut chunks = Vec::new();
    let mut accum = ChunkAccum::default();

    for span in &doc.spans {
        let span_text = span.text(doc);
        let span_tokens = tokenizer.count(span_text);

        if span_tokens > max_tokens {
            // Flush whatever we have so far.
            if accum.is_nonempty() {
                chunks.push(accum.finish(doc, &source));
                accum = ChunkAccum::default();
            }
            // Split the over-long span into sub-chunks at token boundaries.
            split_oversized_span(
                doc,
                span,
                &tokenizer,
                max_tokens,
                &source,
                &mut chunks,
            )?;
        } else if accum.try_add(span, span_tokens, max_tokens) {
            // Span fits in current chunk.
        } else {
            // Span doesn't fit. Flush and start new.
            chunks.push(accum.finish(doc, &source));
            accum = ChunkAccum::default();
            accum.try_add(span, span_tokens, max_tokens);
        }
    }

    if accum.is_nonempty() {
        chunks.push(accum.finish(doc, &source));
    }

    // Assign sequential chunk_index values.
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = offset_to_u32(i);
    }

    Ok(chunks)
}

/// Validates that the chunk config is self-consistent.
fn validate_config(cfg: &ChunkConfig) -> Result<()> {
    if cfg.max_tokens == 0 {
        return Err(BitVanesError::InvalidConfig(
            "max_tokens must be greater than zero".to_string(),
        ));
    }
    if cfg.overlap_tokens >= cfg.max_tokens {
        return Err(BitVanesError::InvalidConfig(format!(
            "overlap_tokens ({}) must be less than max_tokens ({})",
            cfg.overlap_tokens, cfg.max_tokens
        )));
    }
    if cfg.overlap_tokens > 0 {
        return Err(BitVanesError::InvalidConfig(
            "overlap_tokens > 0 is not yet implemented".to_string(),
        ));
    }
    Ok(())
}

/// Splits an over-long span into sub-chunks, each respecting `max_tokens`.
fn split_oversized_span(
    doc: &Document,
    span: &crate::parse::TextSpan,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    source: &str,
    out: &mut Vec<ChunkSpec>,
) -> Result<()> {
    let full_text = span.text(doc);
    let mut cursor = 0usize;
    let abs_base = span.char_offset_start as usize;

    while cursor < full_text.len() {
        let remaining = &full_text[cursor..];
        let (split_at, token_count) =
            tokenizer.split_at_token_boundary(remaining, max_tokens)?;

        if split_at == 0 {
            // Safety valve: if the tokenizer returns 0 for a non-empty
            // remaining, fail rather than risk an infinite loop.
            return Err(BitVanesError::InvalidInput(format!(
                "tokenizer produced a zero-length split at byte {cursor} in a span of {} bytes",
                full_text.len()
            )));
        }

        out.push(ChunkSpec {
            chunk_index: 0, // re-indexed by caller
            text: remaining[..split_at].to_string(),
            token_count: u16::try_from(token_count).map_err(|_| {
                BitVanesError::InvalidInput(
                    "token count overflowed u16".to_string(),
                )
            })?,
            source_path: source.to_string(),
            heading_path: span.heading_path.clone(),
            section_kind: span.section_kind,
            char_offset_start: offset_to_u32(abs_base + cursor),
            char_offset_end: offset_to_u32(abs_base + cursor + split_at),
        });

        cursor += split_at;
    }

    Ok(())
}

/// Mutable accumulator for the current chunk being built.
#[derive(Default)]
struct ChunkAccum {
    start_offset: usize,
    end_offset: usize,
    token_count: usize,
    heading_path: Vec<String>,
    section_kind: Option<SectionKind>,
}

impl ChunkAccum {
    /// Tries to add a span to this chunk. Returns `true` if added, `false`
    /// if the span doesn't fit.
    fn try_add(
        &mut self,
        span: &crate::parse::TextSpan,
        span_tokens: usize,
        max_tokens: usize,
    ) -> bool {
        if self.token_count + span_tokens > max_tokens {
            return false;
        }
        if self.section_kind.is_none() {
            // First span in this chunk: adopt its metadata.
            self.start_offset = span.char_offset_start as usize;
            self.heading_path.clone_from(&span.heading_path);
            self.section_kind = Some(span.section_kind);
        }
        self.end_offset = span.char_offset_end as usize;
        self.token_count += span_tokens;
        true
    }

    const fn is_nonempty(&self) -> bool {
        self.section_kind.is_some()
    }

    fn finish(self, doc: &Document, source: &str) -> ChunkSpec {
        let text =
            doc.full_text[self.start_offset..self.end_offset].to_string();
        ChunkSpec {
            chunk_index: 0, // re-indexed by caller
            text,
            token_count: u16::try_from(self.token_count).unwrap_or(u16::MAX),
            source_path: source.to_string(),
            heading_path: self.heading_path,
            section_kind: self.section_kind.unwrap_or(SectionKind::Paragraph),
            char_offset_start: offset_to_u32(self.start_offset),
            char_offset_end: offset_to_u32(self.end_offset),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::Parser;
    use crate::parse::{Document, TextSpan};
    use crate::schema::{
        ChunkConfig, DocumentFormat, PipelineConfig, SectionKind, TokenizerKind,
    };

    fn make_doc(spans: &[(&str, SectionKind)]) -> Document {
        let mut full_text = String::new();
        let mut span_vec = Vec::new();
        for (text, kind) in spans {
            let start = full_text.len();
            full_text.push_str(text);
            let end = full_text.len();
            span_vec.push(TextSpan {
                char_offset_start: offset_to_u32(start),
                char_offset_end: offset_to_u32(end),
                heading_path: Vec::new(),
                section_kind: *kind,
            });
        }
        Document {
            full_text,
            spans: span_vec,
        }
    }

    fn cfg(max_tokens: u32) -> ChunkConfig {
        ChunkConfig {
            max_tokens,
            overlap_tokens: 0,
            tokenizer: TokenizerKind::Cl100kBase,
        }
    }

    #[test]
    fn empty_document_produces_no_chunks() {
        let doc = Document::default();
        assert!(chunk_document(&doc, &cfg(512), None).unwrap().is_empty());
    }

    #[test]
    fn single_small_span_produces_one_chunk() {
        let doc = make_doc(&[("Hello world.", SectionKind::Paragraph)]);
        let chunks = chunk_document(&doc, &cfg(512), Some("test.md")).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world.");
        assert!(chunks[0].token_count > 0);
        assert_eq!(chunks[0].source_path, "test.md");
        assert_eq!(chunks[0].section_kind, SectionKind::Paragraph);
    }

    #[test]
    fn multiple_small_spans_pack_into_one_chunk() {
        let doc = make_doc(&[
            ("First sentence. ", SectionKind::Paragraph),
            ("Second sentence. ", SectionKind::Paragraph),
            ("Third.", SectionKind::Paragraph),
        ]);
        let chunks = chunk_document(&doc, &cfg(512), None).unwrap();
        assert_eq!(chunks.len(), 1, "all three spans should fit in one chunk");
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Third"));
    }

    #[test]
    fn spans_split_across_chunks_when_exceeding_max_tokens() {
        // Each span is ~3-4 tokens. With max_tokens=5, we should get
        // multiple chunks.
        let doc = make_doc(&[
            ("The quick brown ", SectionKind::Paragraph),
            ("fox jumps over ", SectionKind::Paragraph),
            ("the lazy dog ", SectionKind::Paragraph),
            ("while a cat watches.", SectionKind::Paragraph),
        ]);
        let chunks = chunk_document(&doc, &cfg(5), None).unwrap();
        assert!(
            chunks.len() > 1,
            "should produce multiple chunks with tight token limit"
        );

        // Every chunk must respect max_tokens.
        for chunk in &chunks {
            assert!(
                chunk.token_count <= 5,
                "chunk {} has {} tokens, max is 5",
                chunk.chunk_index,
                chunk.token_count
            );
        }

        // Chunk indices must be sequential.
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index as usize, i);
        }
    }

    #[test]
    fn oversized_span_is_split_at_token_boundaries() {
        // A single very long span that far exceeds max_tokens.
        let long_text =
            "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let doc = make_doc(&[(&long_text, SectionKind::Paragraph)]);
        let chunks = chunk_document(&doc, &cfg(10), None).unwrap();

        assert!(
            chunks.len() > 1,
            "oversized span should be split into multiple chunks"
        );
        for chunk in &chunks {
            assert!(
                chunk.token_count <= 10,
                "sub-chunk has {} tokens, max is 10",
                chunk.token_count
            );
        }
    }

    #[test]
    fn heading_path_inherited_from_first_span() {
        let doc = {
            let mut full_text = String::new();
            let mut spans = Vec::new();
            // First span under "Architecture" heading.
            let s1_start = full_text.len();
            full_text.push_str("Content under architecture.");
            spans.push(TextSpan {
                char_offset_start: offset_to_u32(s1_start),
                char_offset_end: offset_to_u32(full_text.len()),
                heading_path: vec!["Architecture".to_string()],
                section_kind: SectionKind::Paragraph,
            });
            // Second span under "Architecture > Storage".
            let s2_start = full_text.len();
            full_text.push_str("Storage details here.");
            spans.push(TextSpan {
                char_offset_start: offset_to_u32(s2_start),
                char_offset_end: offset_to_u32(full_text.len()),
                heading_path: vec![
                    "Architecture".to_string(),
                    "Storage".to_string(),
                ],
                section_kind: SectionKind::Paragraph,
            });
            Document { full_text, spans }
        };

        // Use max_tokens=5 so each span fits individually (~4 tokens each)
        // but both don't fit together (~8 > 5), producing two chunks.
        let chunks = chunk_document(&doc, &cfg(5), None).unwrap();
        assert!(chunks.len() >= 2, "should produce at least 2 chunks");

        // Find the chunk containing "Storage" and verify its heading_path.
        let storage_chunk = chunks
            .iter()
            .find(|c| c.text.contains("Storage"))
            .expect("should find a chunk containing Storage text");
        assert_eq!(
            storage_chunk.heading_path,
            vec!["Architecture".to_string(), "Storage".to_string()]
        );

        // Find the chunk containing "Content under" and verify its heading_path.
        let arch_chunk = chunks
            .iter()
            .find(|c| c.text.contains("Content under"))
            .expect("should find a chunk with Content text");
        assert_eq!(arch_chunk.heading_path, vec!["Architecture".to_string()]);
    }

    #[test]
    fn section_kind_inherited_from_first_span() {
        let doc = make_doc(&[
            ("Some code here.", SectionKind::Code),
            ("More code.", SectionKind::Code),
        ]);
        let chunks = chunk_document(&doc, &cfg(512), None).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_kind, SectionKind::Code);
    }

    #[test]
    fn max_tokens_zero_is_rejected() {
        let doc = make_doc(&[("hello", SectionKind::Paragraph)]);
        let err = chunk_document(&doc, &cfg(0), None).unwrap_err();
        assert!(matches!(err, BitVanesError::InvalidConfig(_)));
    }

    #[test]
    fn overlap_not_yet_implemented_is_rejected() {
        let doc = make_doc(&[("hello", SectionKind::Paragraph)]);
        let cfg_overlap = ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 10,
            tokenizer: TokenizerKind::Cl100kBase,
        };
        let err = chunk_document(&doc, &cfg_overlap, None).unwrap_err();
        assert!(matches!(err, BitVanesError::InvalidConfig(_)));
    }

    #[test]
    fn chunk_text_round_trips_through_parser() {
        // End-to-end: parse markdown → chunk → verify chunk text is
        // a substring of the document text.
        let input =
            "# Title\n\nFirst paragraph here.\n\nSecond paragraph there.";
        let parse_cfg = PipelineConfig {
            format: DocumentFormat::Markdown,
            ..PipelineConfig::default()
        };
        let doc = crate::parse::MarkdownParser
            .parse(input, &parse_cfg)
            .expect("parse");
        let chunks = chunk_document(&doc, &cfg(5), None).unwrap();
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                doc.full_text.contains(&chunk.text),
                "chunk text must be a substring of the document"
            );
        }
    }
}

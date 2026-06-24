//! PII scrubbing: redact built-in patterns before chunking, with an offset
//! map that can project chunk positions back onto the original document.
//!
//! Run: `cargo run -p bitvanes-core --example pii_scrub`

use bitvanes_core::{
    BuiltInPattern, ChunkConfig, PipelineConfig, ScrubProfile, TokenizerKind,
    chunk::chunk_document,
    parse::{MarkdownParser, Parser},
    scrub::scrub_document,
};

fn main() -> bitvanes_core::Result<()> {
    let src = "Reach alice@example.com or 415-555-0123. File 123-45-6789 was leaked.";

    let doc = MarkdownParser.parse(src, &PipelineConfig::default())?;

    let profile = ScrubProfile {
        patterns: vec![
            BuiltInPattern::Email,
            BuiltInPattern::Phone,
            BuiltInPattern::Ssn,
        ],
        custom: vec![],
    };
    let (scrubbed, _offset_map) = scrub_document(doc, &profile)?;

    println!("scrubbed text:\n  {}\n", scrubbed.full_text);

    let chunks = chunk_document(
        &scrubbed,
        &ChunkConfig {
            max_tokens: 32,
            overlap_tokens: 0,
            tokenizer: TokenizerKind::Cl100kBase,
            ..ChunkConfig::default()
        },
        None,
    )?;
    println!(
        "{} chunk(s) emitted (PII never crosses a boundary).",
        chunks.len()
    );
    Ok(())
}

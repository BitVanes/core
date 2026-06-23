//! PII scrubbing: character-level redaction of sensitive data, executed
//! BEFORE tokenization so that PII matches cannot be split across chunk
//! boundaries.
//!
//! # Architecture
//!
//! 1. [`Scrubber::from_profile`] compiles all enabled patterns (built-in
//!    and user-supplied) into individual [`Regex`](regex::Regex) instances.
//! 2. [`Scrubber::scrub`] finds all matches via `find_iter`, applies
//!    optional post-filters (Luhn validation for credit cards), resolves
//!    overlaps via greedy interval scheduling, and builds the scrubbed
//!    text.
//! 3. [`OffsetMap`] records every replacement's position so that chunk
//!    offsets (into the scrubbed text) can be projected back onto the
//!    original document for UI highlighting.
//!
//! # Built-in patterns
//!
//! | Pattern     | Replacement     | Post-filter |
//! |-------------|-----------------|-------------|
//! | Email       | `[EMAIL]`       | none        |
//! | SSN         | `[SSN]`         | none        |
//! | Phone       | `[PHONE]`       | none        |
//! | CreditCard  | `[CREDIT_CARD]` | Luhn        |
//! | AwsKey      | `[AWS_KEY]`     | none        |
//! | GitHubPat   | `[GITHUB_PAT]`  | none        |
//! | Jwt         | `[JWT]`         | none        |

use regex::Regex;

use crate::error::{BitVanesError, Result};
use crate::parse::{Document, offset_to_u32};
use crate::schema::{BuiltInPattern, ScrubProfile};

// ===========================================================================
// Offset map
// ===========================================================================

/// A sorted list of text replacements that maps offsets between the
/// original document text and the scrubbed output.
///
/// Built by [`Scrubber::scrub`]. Supports bidirectional projection:
///
/// - [`project_forward`](Self::project_forward): original offset to scrubbed offset.
/// - [`project_inverse`](Self::project_inverse): scrubbed offset to original offset.
///
/// Both directions are O(n) in the number of edits. For typical PII
/// densities (< 100 replacements per document) this is effectively free.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OffsetMap {
    /// Sorted by `orig_start`. Each entry records one replacement region.
    edits: Vec<OffsetEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OffsetEdit {
    /// Start byte offset in the original text (inclusive).
    orig_start: usize,
    /// End byte offset in the original text (exclusive).
    orig_end: usize,
    /// Byte length of the replacement string.
    replacement_len: usize,
}

impl OffsetEdit {
    /// Net character delta: positive = text got shorter, negative = text got longer.
    #[allow(clippy::cast_possible_wrap)]
    const fn delta(&self) -> isize {
        (self.orig_end - self.orig_start) as isize
            - self.replacement_len as isize
    }
}

// Cast safety: document offsets are bounded by wasm linear memory (< 4 GiB),
// far below isize::MAX on both 32- and 64-bit targets. The sign-loss and
// wrap-around cases clippy warns about are structurally impossible here.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
impl OffsetMap {
    /// Projects an original-text byte offset to the corresponding
    /// scrubbed-text byte offset.
    ///
    /// For positions inside a replaced region, snaps to the start of the
    /// replacement text in the scrubbed output.
    #[must_use]
    pub fn project_forward(&self, orig_offset: usize) -> usize {
        let mut delta: isize = 0;
        for edit in &self.edits {
            if edit.orig_end <= orig_offset {
                delta += edit.delta();
            } else if edit.orig_start <= orig_offset {
                // Inside this edit: snap to the edit's scrubbed start.
                return (edit.orig_start as isize - delta).max(0) as usize;
            } else {
                break;
            }
        }
        (orig_offset as isize - delta).max(0) as usize
    }

    /// Projects a scrubbed-text byte offset back to the corresponding
    /// original-text byte offset.
    ///
    /// For positions inside a replacement token, maps to the start of the
    /// original PII region.
    #[must_use]
    pub fn project_inverse(&self, scrubbed_offset: usize) -> usize {
        let mut delta: isize = 0;
        let s = scrubbed_offset as isize;
        for edit in &self.edits {
            let scrub_start = edit.orig_start as isize - delta;
            let scrub_end = scrub_start + edit.replacement_len as isize;
            if s < scrub_start {
                return (s + delta).max(0) as usize;
            }
            if s < scrub_end {
                return edit.orig_start;
            }
            delta += edit.delta();
        }
        (s + delta).max(0) as usize
    }

    /// Returns `true` if no replacements were made (identity map).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.edits.is_empty()
    }

    /// Returns the number of replacements recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Returns `true` if no replacements were recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }
}

// ===========================================================================
// Scrubber
// ===========================================================================

/// A compiled PII scrubber. Built once from a [`ScrubProfile`] and applied
/// to many documents (or document chunks) without recompilation.
///
/// Cloning is cheap (regexes use `Arc` internally).
#[derive(Debug, Clone)]
pub struct Scrubber {
    patterns: Vec<CompiledPattern>,
}

#[derive(Debug, Clone)]
struct CompiledPattern {
    regex: Regex,
    replacement: String,
    validator: Validator,
}

#[derive(Debug, Clone, Copy)]
enum Validator {
    None,
    /// Extract digits from the match, validate via Luhn checksum.
    Luhn,
}

impl Validator {
    fn is_valid(self, text: &str, start: usize, end: usize) -> bool {
        match self {
            Self::None => true,
            Self::Luhn => luhn_valid(&text[start..end]),
        }
    }
}

impl Scrubber {
    /// Compiles all patterns (built-in and custom) from the profile into
    /// a single [`Scrubber`].
    ///
    /// # Errors
    ///
    /// Returns [`BitVanesError::InvalidConfig`] if any regex fails to compile.
    pub fn from_profile(profile: &ScrubProfile) -> Result<Self> {
        let mut patterns =
            Vec::with_capacity(profile.patterns.len() + profile.custom.len());

        for &kind in &profile.patterns {
            let (src, replacement, validator) = builtin_config(kind);
            let regex = Regex::new(src).map_err(|e| {
                BitVanesError::InvalidConfig(format!(
                    "built-in pattern {kind:?}: {e}"
                ))
            })?;
            patterns.push(CompiledPattern {
                regex,
                replacement: replacement.to_string(),
                validator,
            });
        }

        for custom in &profile.custom {
            let regex = Regex::new(&custom.regex).map_err(|e| {
                BitVanesError::InvalidConfig(format!(
                    "custom regex '{}': {e}",
                    custom.regex
                ))
            })?;
            patterns.push(CompiledPattern {
                regex,
                replacement: custom.replacement.clone(),
                validator: Validator::None,
            });
        }

        Ok(Self { patterns })
    }

    /// Returns `true` if no patterns are compiled (scrubbing is a no-op).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Scrubs `text`: finds all PII matches, resolves overlaps, and builds
    /// the redacted output plus an [`OffsetMap`] for offset projection.
    #[must_use]
    pub fn scrub(&self, text: &str) -> (String, OffsetMap) {
        if self.patterns.is_empty() || text.is_empty() {
            return (text.to_string(), OffsetMap::default());
        }

        let mut matches = Vec::new();
        for cp in &self.patterns {
            for m in cp.regex.find_iter(text) {
                if cp.validator.is_valid(text, m.start(), m.end()) {
                    matches.push(PiiMatch {
                        start: m.start(),
                        end: m.end(),
                        replacement: cp.replacement.clone(),
                    });
                }
            }
        }

        let resolved = resolve_overlaps(matches);
        build_scrubbed(text, &resolved)
    }
}

/// A raw PII match found by a regex, before overlap resolution.
struct PiiMatch {
    start: usize,
    end: usize,
    replacement: String,
}

/// Resolves overlapping matches using greedy interval scheduling (earliest
/// end first) to maximize the number of non-overlapping matches.
fn resolve_overlaps(mut matches: Vec<PiiMatch>) -> Vec<PiiMatch> {
    if matches.len() <= 1 {
        return matches;
    }
    // Sort by end (earliest-ending first), then by start for determinism.
    matches.sort_by(|a, b| a.end.cmp(&b.end).then(a.start.cmp(&b.start)));

    let mut accepted: Vec<PiiMatch> = Vec::with_capacity(matches.len());
    for m in matches {
        let overlaps = accepted.last().is_some_and(|last| last.end > m.start);
        if !overlaps {
            accepted.push(m);
        }
    }
    // Re-sort by start for the sequential text-building pass.
    accepted.sort_by_key(|m| m.start);
    accepted
}

/// Interleaves original text with replacement tokens to produce the
/// scrubbed string and records each replacement in an [`OffsetMap`].
fn build_scrubbed(text: &str, matches: &[PiiMatch]) -> (String, OffsetMap) {
    let mut scrubbed = String::with_capacity(text.len());
    let mut edits = Vec::with_capacity(matches.len());
    let mut cursor = 0usize;

    for m in matches {
        debug_assert!(
            m.start >= cursor,
            "matches must be sorted by start and non-overlapping"
        );
        if m.start > cursor {
            scrubbed.push_str(&text[cursor..m.start]);
        }
        edits.push(OffsetEdit {
            orig_start: m.start,
            orig_end: m.end,
            replacement_len: m.replacement.len(),
        });
        scrubbed.push_str(&m.replacement);
        cursor = m.end;
    }
    if cursor < text.len() {
        scrubbed.push_str(&text[cursor..]);
    }

    (scrubbed, OffsetMap { edits })
}

// ===========================================================================
// Built-in pattern definitions
// ===========================================================================

/// Returns `(regex_source, replacement_token, validator)` for a built-in kind.
fn builtin_config(
    kind: BuiltInPattern,
) -> (&'static str, &'static str, Validator) {
    match kind {
        BuiltInPattern::Email => (
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            "[EMAIL]",
            Validator::None,
        ),
        BuiltInPattern::Ssn => {
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", Validator::None)
        }
        BuiltInPattern::Phone => (r"\+1\d{10}\b", "[PHONE]", Validator::None),
        BuiltInPattern::CreditCard => (
            r"\b(?:\d[ -]?){12,18}\d\b",
            "[CREDIT_CARD]",
            Validator::Luhn,
        ),
        BuiltInPattern::AwsKey => {
            (r"\bAKIA[0-9A-Z]{16}\b", "[AWS_KEY]", Validator::None)
        }
        BuiltInPattern::GitHubPat => (
            r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
            "[GITHUB_PAT]",
            Validator::None,
        ),
        BuiltInPattern::Jwt => (
            r"\beyJ[A-Za-z0-9_\-]*\.eyJ[A-Za-z0-9_\-]*\.[A-Za-z0-9_\-]*\b",
            "[JWT]",
            Validator::None,
        ),
    }
}

// ===========================================================================
// Luhn checksum
// ===========================================================================

/// Validates a string of digits (possibly mixed with separators) using the
/// Luhn algorithm. Returns `false` if fewer than 13 digits are present.
fn luhn_valid(text: &str) -> bool {
    let digits: Vec<u8> = text
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b - b'0')
        .collect();
    if digits.len() < 13 {
        return false;
    }
    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut n = u32::from(d);
        if double {
            n *= 2;
            if n > 9 {
                n -= 9;
            }
        }
        sum += n;
        double = !double;
    }
    sum % 10 == 0
}

// ===========================================================================
// High-level entry points
// ===========================================================================

/// Scrubs a plain-text string. Convenience wrapper around
/// [`Scrubber::from_profile`] + [`Scrubber::scrub`].
///
/// # Errors
///
/// Returns [`BitVanesError::InvalidConfig`] if any regex in the profile
/// fails to compile.
pub fn scrub_text(
    text: &str,
    profile: &ScrubProfile,
) -> Result<(String, OffsetMap)> {
    let scrubber = Scrubber::from_profile(profile)?;
    Ok(scrubber.scrub(text))
}

/// Scrubs a [`Document`]'s `full_text` and projects all span offsets into
/// the scrubbed text's coordinate space.
///
/// This is the primary entry point called by the pipeline between parsing
/// and chunking.
///
/// # Errors
///
/// Returns [`BitVanesError::InvalidConfig`] if any regex fails to compile.
pub fn scrub_document(
    doc: Document,
    profile: &ScrubProfile,
) -> Result<(Document, OffsetMap)> {
    let scrubber = Scrubber::from_profile(profile)?;
    let (scrubbed_text, map) = scrubber.scrub(&doc.full_text);

    let spans = doc
        .spans
        .into_iter()
        .map(|mut span| {
            let new_start =
                map.project_forward(span.char_offset_start as usize);
            let new_end = map.project_forward(span.char_offset_end as usize);
            span.char_offset_start = offset_to_u32(new_start);
            span.char_offset_end = offset_to_u32(new_end);
            span
        })
        .collect();

    Ok((
        Document {
            full_text: scrubbed_text,
            spans,
        },
        map,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::Parser;
    use crate::schema::{CustomPattern, DocumentFormat, PipelineConfig};

    fn scrub_with(
        text: &str,
        patterns: &[BuiltInPattern],
    ) -> (String, OffsetMap) {
        let scrubber = Scrubber::from_profile(&ScrubProfile {
            patterns: patterns.to_vec(),
            custom: vec![],
        })
        .expect("built-in patterns should compile");
        scrubber.scrub(text)
    }

    // ----- Built-in pattern tests -----

    #[test]
    fn email_is_redacted() {
        let (out, map) = scrub_with(
            "Contact alice@example.com for details.",
            &[BuiltInPattern::Email],
        );
        assert_eq!(out, "Contact [EMAIL] for details.");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn multiple_emails_in_one_text() {
        let (out, map) = scrub_with(
            "From a@x.com and b@y.org to c@z.io.",
            &[BuiltInPattern::Email],
        );
        assert_eq!(out, "From [EMAIL] and [EMAIL] to [EMAIL].");
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn ssn_is_redacted() {
        let (out, _) = scrub_with("SSN: 123-45-6789.", &[BuiltInPattern::Ssn]);
        assert_eq!(out, "SSN: [SSN].");
    }

    #[test]
    fn phone_e164_is_redacted() {
        let (out, _) =
            scrub_with("Call +15551234567.", &[BuiltInPattern::Phone]);
        assert_eq!(out, "Call [PHONE].");
    }

    #[test]
    fn aws_key_is_redacted() {
        let (out, _) =
            scrub_with("key = AKIAIOSFODNN7EXAMPLE", &[BuiltInPattern::AwsKey]);
        assert_eq!(out, "key = [AWS_KEY]");
    }

    #[test]
    fn github_pat_is_redacted() {
        let pat = "ghp_abcdefghijklmnopqrstuvwxyz0123456789AB";
        let (out, _) =
            scrub_with(&format!("token: {pat}"), &[BuiltInPattern::GitHubPat]);
        assert_eq!(out, "token: [GITHUB_PAT]");
    }

    #[test]
    fn jwt_is_redacted() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let (out, _) = scrub_with(
            &format!("Authorization: Bearer {jwt}"),
            &[BuiltInPattern::Jwt],
        );
        assert!(out.contains("[JWT]"), "JWT should be redacted: {out}");
        assert!(
            !out.contains("eyJ"),
            "raw JWT header should not survive: {out}"
        );
    }

    // ----- Credit card + Luhn -----

    #[test]
    fn valid_credit_card_passes_luhn() {
        // 4111 1111 1111 1111 is a classic test card (passes Luhn).
        let (out, map) = scrub_with(
            "Card: 4111 1111 1111 1111 done.",
            &[BuiltInPattern::CreditCard],
        );
        assert_eq!(out, "Card: [CREDIT_CARD] done.");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn invalid_credit_card_fails_luhn_and_is_not_redacted() {
        // Same length but invalid checksum.
        let (out, map) = scrub_with(
            "Card: 4111 1111 1111 1112 done.",
            &[BuiltInPattern::CreditCard],
        );
        assert_eq!(
            out, "Card: 4111 1111 1111 1112 done.",
            "invalid-Luhn sequence should NOT be redacted"
        );
        assert!(map.is_empty());
    }

    #[test]
    fn credit_card_without_separators() {
        // 4111111111111111 — unseparated, valid Luhn.
        let (out, _) =
            scrub_with("4111111111111111", &[BuiltInPattern::CreditCard]);
        assert_eq!(out, "[CREDIT_CARD]");
    }

    // ----- OffsetMap projection -----

    #[test]
    fn offset_map_forward_round_trip() {
        let text = "Contact alice@example.com please.";
        let (scrubbed, map) = scrub_with(text, &[BuiltInPattern::Email]);
        assert_eq!(scrubbed, "Contact [EMAIL] please.");

        // [EMAIL] occupies scrubbed positions [8..15). Positions OUTSIDE the
        // replacement must round-trip exactly: forward(inverse(s)) == s.
        for s in 0..=7 {
            let back = map.project_forward(map.project_inverse(s));
            assert_eq!(back, s, "pre-replacement round-trip failed at {s}");
        }
        for s in 15..=scrubbed.len() {
            let back = map.project_forward(map.project_inverse(s));
            assert_eq!(back, s, "post-replacement round-trip failed at {s}");
        }
    }

    #[test]
    fn offset_map_identity_when_no_matches() {
        let (out, map) = scrub_with("No PII here.", &[BuiltInPattern::Email]);
        assert!(map.is_identity());
        assert_eq!(out, "No PII here.");
        // Forward projection of any offset is the offset itself.
        for i in 0..=out.len() {
            assert_eq!(map.project_forward(i), i);
            assert_eq!(map.project_inverse(i), i);
        }
    }

    // ----- scrub_document -----

    #[test]
    fn scrub_document_projects_span_offsets() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Text,
            ..PipelineConfig::default()
        };
        let doc = crate::parse::TextParser
            .parse("Contact alice@test.com.\n\nEmail bob@test.org too.", &cfg)
            .expect("parse");

        assert_eq!(doc.spans.len(), 2);

        let profile = ScrubProfile {
            patterns: vec![BuiltInPattern::Email],
            custom: vec![],
        };
        let (scrubbed_doc, map) =
            scrub_document(doc.clone(), &profile).expect("scrub");

        assert_eq!(map.len(), 2, "two emails should be scrubbed");
        assert!(scrubbed_doc.full_text.contains("[EMAIL]"));
        assert!(!scrubbed_doc.full_text.contains("alice@test.com"));

        // Span offsets must be valid indices into the scrubbed text.
        for span in &scrubbed_doc.spans {
            let e = span.char_offset_end as usize;
            assert!(
                e <= scrubbed_doc.full_text.len(),
                "span end {e} exceeds scrubbed text len {}",
                scrubbed_doc.full_text.len()
            );
            let text = span.text(&scrubbed_doc);
            assert!(
                !text.is_empty(),
                "span should not be empty after scrubbing"
            );
        }

        // Contiguity invariant must still hold after projection.
        scrubbed_doc.assert_spans_contiguous();
    }

    #[test]
    fn scrub_document_with_empty_profile_is_noop() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Text,
            ..PipelineConfig::default()
        };
        let doc = crate::parse::TextParser
            .parse("Just some text.\n\nNo PII.", &cfg)
            .expect("parse");

        let (scrubbed_doc, map) =
            scrub_document(doc.clone(), &ScrubProfile::default())
                .expect("scrub");
        assert!(map.is_identity());
        assert_eq!(scrubbed_doc, doc);
    }

    // ----- Custom patterns -----

    #[test]
    fn custom_pattern_redacts_matches() {
        let scrubber = Scrubber::from_profile(&ScrubProfile {
            patterns: vec![],
            custom: vec![CustomPattern {
                regex: r"PROJECT-\d{4}".to_string(),
                replacement: "[PROJ]".to_string(),
            }],
        })
        .expect("custom regex should compile");

        let (out, _) = scrubber.scrub("See PROJECT-1234 and PROJECT-5678.");
        assert_eq!(out, "See [PROJ] and [PROJ].");
    }

    #[test]
    fn invalid_custom_regex_returns_error() {
        let err = Scrubber::from_profile(&ScrubProfile {
            patterns: vec![],
            custom: vec![CustomPattern {
                regex: "[invalid".to_string(),
                replacement: "X".to_string(),
            }],
        })
        .unwrap_err();
        assert!(
            matches!(err, BitVanesError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );
    }

    // ----- Overlap resolution -----

    #[test]
    fn overlapping_matches_are_resolved() {
        // Two emails that overlap (contrived): the earlier-ending one wins.
        let scrubber = Scrubber::from_profile(&ScrubProfile {
            patterns: vec![BuiltInPattern::Email],
            custom: vec![CustomPattern {
                regex: r"alice@example\.com\.[a-z]+".to_string(),
                replacement: "[ALICE_FULL]".to_string(),
            }],
        })
        .expect("compile");

        // The custom pattern is longer and overlaps the email match.
        let (out, map) = scrubber.scrub("Contact alice@example.com.org now.");
        // Greedy scheduling: earliest-ending wins. Email ends at
        // "alice@example.com" which is earlier than the custom match
        // "alice@example.com.org".
        assert!(
            out.contains("[EMAIL]") || out.contains("[ALICE_FULL]"),
            "one pattern should win: {out}"
        );
        assert_eq!(map.len(), 1, "exactly one match should survive overlap");
    }

    // ----- Luhn unit tests -----

    #[test]
    fn luhn_known_valid_cards() {
        assert!(luhn_valid("4111111111111111")); // Visa test
        assert!(luhn_valid("4111 1111 1111 1111")); // with spaces
        assert!(luhn_valid("5500000000000004")); // Mastercard test
        assert!(luhn_valid("4012888888881881")); // Visa test 2
    }

    #[test]
    fn luhn_known_invalid_cards() {
        assert!(!luhn_valid("4111111111111112"));
        assert!(!luhn_valid("1234567890123"));
        assert!(!luhn_valid("49927398717")); // classic Luhn-fail example
    }

    #[test]
    fn luhn_rejects_short_sequences() {
        assert!(!luhn_valid("12345"));
        assert!(!luhn_valid(""));
    }
}

//! Apache Arrow columnar output: schema, batch assembly, FFI export, IPC
//! streaming.
//!
//! This module is the contract between Rust and JavaScript: it defines the
//! [`Schema`] that `arrow-js-ffi` reads off the wasm heap, the release
//! registry that owns the lifetime of every exported [`RecordBatch`], and
//! the batch builder that converts [`ChunkSpec`]s into Arrow columns.
//!
//! # Schema stability
//!
//! Adding, removing, or reordering columns in [`output_schema`] is a
//! **breaking change** to both the npm and crates.io published artifacts
//! and MUST be gated by a semver-major version bump.
//!
//! [`RecordBatch`]: arrow::record_batch::RecordBatch
//! [`Schema`]: arrow::datatypes::Schema

pub mod batch;
pub mod ffi;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// The default embedding dimension (matches `OpenAI`
/// `text-embedding-3-small` / `text-embedding-ada-002`). Models that
/// produce a different dimension (e.g. MiniLM-L6-v2 = 384) should use
/// [`output_schema_with_dim`] instead.
/// is chosen over `List` so downstream vector DBs can mmap the column
/// directly without an offset buffer.
pub const EMBEDDING_DIM: usize = 1536;

/// Returns the canonical output [`SchemaRef`] for a fully-processed
/// pipeline run, using the default embedding dimension (1536).
#[must_use]
pub fn output_schema() -> SchemaRef {
    output_schema_with_dim(EMBEDDING_DIM)
}

/// Returns the output schema with a specific embedding column dimension.
/// Use when generating embeddings with a non-default model (e.g. `MiniLM`
/// produces 384-dim vectors).
#[must_use]
pub fn output_schema_with_dim(embedding_dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("token_count", DataType::UInt16, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new_list(
            "heading_path",
            Field::new("item", DataType::Utf8, true),
            true,
        ),
        Field::new(
            "section_kind",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Utf8),
            ),
            false,
        ),
        Field::new("char_offset_start", DataType::UInt32, false),
        Field::new("char_offset_end", DataType::UInt32, false),
        Field::new_fixed_size_list(
            "embedding",
            Field::new("item", DataType::Float32, true),
            i32::try_from(embedding_dim).expect("embedding dim fits in i32"),
            true,
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_schema_columns_match_contract() {
        let s = output_schema();
        assert_eq!(
            s.fields().len(),
            9,
            "column count must match the documented contract"
        );

        let names: Vec<&str> =
            s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            [
                "chunk_index",
                "text",
                "token_count",
                "source_path",
                "heading_path",
                "section_kind",
                "char_offset_start",
                "char_offset_end",
                "embedding",
            ]
        );
    }

    #[test]
    fn output_schema_nullability_invariants() {
        let s = output_schema();

        // Document text and structural fields must never be null.
        for required in [
            "chunk_index",
            "text",
            "token_count",
            "source_path",
            "section_kind",
            "char_offset_start",
            "char_offset_end",
        ] {
            assert!(
                !s.field_with_name(required).unwrap().is_nullable(),
                "{required} must be non-nullable"
            );
        }

        // heading_path may be null (top-level paragraphs have no ancestry).
        assert!(s.field_with_name("heading_path").unwrap().is_nullable());
        // embedding is populated downstream; nullable in v1.
        assert!(s.field_with_name("embedding").unwrap().is_nullable());
    }

    #[test]
    fn output_schema_section_kind_is_int8_dictionary() {
        let s = output_schema();
        let f = s.field_with_name("section_kind").unwrap();
        assert!(
            matches!(f.data_type(), DataType::Dictionary(k, v)
                if **k == DataType::Int8 && **v == DataType::Utf8),
            "section_kind must be Dictionary<Int8, Utf8>, got {:?}",
            f.data_type()
        );
    }

    #[test]
    fn output_schema_embedding_has_correct_dimension() {
        let s = output_schema();
        let f = s.field_with_name("embedding").unwrap();
        match f.data_type() {
            DataType::FixedSizeList(_, dim) => {
                assert_eq!(
                    *dim,
                    i32::try_from(EMBEDDING_DIM)
                        .expect("EMBEDDING_DIM fits in i32")
                );
            }
            other => panic!("embedding must be FixedSizeList, got {other:?}"),
        }
    }

    #[test]
    fn output_schema_is_consistent() {
        // Two calls must return structurally identical schemas.
        let a = output_schema();
        let b = output_schema();
        assert_eq!(a.as_ref(), b.as_ref());
    }

    #[test]
    fn output_schema_with_dim_uses_correct_dimension() {
        let s = output_schema_with_dim(384);
        let f = s.field_with_name("embedding").unwrap();
        match f.data_type() {
            DataType::FixedSizeList(_, dim) => assert_eq!(*dim, 384),
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
    }
}

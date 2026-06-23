//! Assembly of [`RecordBatch`] from [`ChunkSpec`] data.
//!
//! This is the last step before FFI export: the chunker produces
//! `Vec<ChunkSpec>`, this module converts them into Arrow columns matching
//! [`crate::arrow_io::output_schema`], and the FFI module exports the
//! resulting batch to JavaScript via zero-copy pointers.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, ListArray, ListBuilder,
    RecordBatch, StringArray, StringDictionaryBuilder, UInt16Array,
    UInt32Array,
};
use arrow::datatypes::Int8Type;

use crate::arrow_io::{EMBEDDING_DIM, output_schema};
use crate::error::Result;
use crate::schema::{ChunkSpec, SectionKind};

/// Converts a slice of [`ChunkSpec`]s into an Arrow [`RecordBatch`] whose
/// schema matches [`output_schema`].
///
/// Column layout (positional):
/// `chunk_index`, `text`, `token_count`, `source_path`, `heading_path`,
/// `section_kind`, `char_offset_start`, `char_offset_end`, `embedding`.
///
/// The `embedding` column is all-null in v1 (populated downstream by the
/// user's model, not by this engine).
///
/// # Errors
///
/// Returns [`crate::error::BitVanesError::Arrow`] if the column count or
/// types don't match the schema (structurally impossible if the schema is
/// unchanged, but handled for safety).
pub fn chunks_to_batch(chunks: &[ChunkSpec]) -> Result<RecordBatch> {
    let schema = output_schema();

    let chunk_index =
        UInt32Array::from_iter_values(chunks.iter().map(|c| c.chunk_index));
    let text: StringArray =
        chunks.iter().map(|c| Some(c.text.as_str())).collect();
    let token_count =
        UInt16Array::from_iter_values(chunks.iter().map(|c| c.token_count));
    let source_path: StringArray = chunks
        .iter()
        .map(|c| Some(c.source_path.as_str()))
        .collect();

    // heading_path: List<Utf8>, nullable.
    let heading_path = build_heading_path(chunks);

    // section_kind: Dictionary<Int8, Utf8>.
    let section_kind = build_section_kind(chunks);

    let char_offset_start = UInt32Array::from_iter_values(
        chunks.iter().map(|c| c.char_offset_start),
    );
    let char_offset_end =
        UInt32Array::from_iter_values(chunks.iter().map(|c| c.char_offset_end));

    // embedding: all-null placeholder (FixedSizeList<Float32, 1536>).
    let embedding = build_null_embedding(chunks.len());

    let columns: Vec<ArrayRef> = vec![
        Arc::new(chunk_index),
        Arc::new(text),
        Arc::new(token_count),
        Arc::new(source_path),
        Arc::new(heading_path),
        Arc::new(section_kind),
        Arc::new(char_offset_start),
        Arc::new(char_offset_end),
        Arc::new(embedding),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Like [`chunks_to_batch`] but fills the `embedding` column with real
/// `Float32` vectors instead of nulls.
///
/// `embeddings` must have exactly one `Vec<f32>` per chunk, each of length
/// `dim`. The schema is built with [`output_schema_with_dim`] to match.
///
/// # Errors
///
/// Returns [`crate::error::BitVanesError`] if the embedding count doesn't
/// match the chunk count, or if Arrow assembly fails.
pub fn chunks_to_batch_with_embeddings(
    chunks: &[ChunkSpec],
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<RecordBatch> {
    if embeddings.len() != chunks.len() {
        return Err(crate::error::BitVanesError::InvalidInput(format!(
            "embeddings count ({}) does not match chunks count ({})",
            embeddings.len(),
            chunks.len()
        )));
    }

    let schema = crate::arrow_io::output_schema_with_dim(dim);
    let chunk_index =
        UInt32Array::from_iter_values(chunks.iter().map(|c| c.chunk_index));
    let text: StringArray =
        chunks.iter().map(|c| Some(c.text.as_str())).collect();
    let token_count =
        UInt16Array::from_iter_values(chunks.iter().map(|c| c.token_count));
    let source_path: StringArray = chunks
        .iter()
        .map(|c| Some(c.source_path.as_str()))
        .collect();
    let heading_path = build_heading_path(chunks);
    let section_kind = build_section_kind(chunks);
    let char_offset_start = UInt32Array::from_iter_values(
        chunks.iter().map(|c| c.char_offset_start),
    );
    let char_offset_end =
        UInt32Array::from_iter_values(chunks.iter().map(|c| c.char_offset_end));

    // embedding: real Float32 vectors (non-null).
    let embedding = build_real_embedding(embeddings, dim);

    let columns: Vec<ArrayRef> = vec![
        Arc::new(chunk_index),
        Arc::new(text),
        Arc::new(token_count),
        Arc::new(source_path),
        Arc::new(heading_path),
        Arc::new(section_kind),
        Arc::new(char_offset_start),
        Arc::new(char_offset_end),
        Arc::new(embedding),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Builds the `heading_path` column: `List<Utf8>`, nullable.
fn build_heading_path(chunks: &[ChunkSpec]) -> ListArray {
    let mut builder = ListBuilder::new(arrow::array::StringBuilder::new());
    for chunk in chunks {
        for heading in &chunk.heading_path {
            builder.values().append_value(heading);
        }
        builder.append(!chunk.heading_path.is_empty());
    }
    builder.finish()
}

/// Builds the `section_kind` column: `Dictionary<Int8, Utf8>`.
fn build_section_kind(
    chunks: &[ChunkSpec],
) -> arrow::array::DictionaryArray<Int8Type> {
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for chunk in chunks {
        builder
            .append(section_kind_str(chunk.section_kind))
            .expect("dictionary append should not fail for valid SectionKind");
    }
    builder.finish()
}

/// Returns the string key used in the `section_kind` dictionary.
/// Must match the serde `rename_all = "snake_case"` on [`SectionKind`].
const fn section_kind_str(kind: SectionKind) -> &'static str {
    match kind {
        SectionKind::Paragraph => "paragraph",
        SectionKind::Code => "code",
        SectionKind::Heading => "heading",
        SectionKind::TableCell => "table_cell",
        SectionKind::ListItem => "list_item",
        SectionKind::BlockQuote => "block_quote",
        SectionKind::FrontMatter => "front_matter",
    }
}

/// Builds an all-null `FixedSizeList<Float32, EMBEDDING_DIM>` column for the
/// embedding placeholder. Each row is null (populated downstream by the
/// user's model, not by this engine).
fn build_null_embedding(num_rows: usize) -> arrow::array::FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(
        Float32Builder::new(),
        i32::try_from(EMBEDDING_DIM).expect("EMBEDDING_DIM fits in i32"),
    );
    for _ in 0..num_rows {
        for _ in 0..EMBEDDING_DIM {
            builder.values().append_value(0.0);
        }
        builder.append(false);
    }
    builder.finish()
}

/// Builds a `FixedSizeList<Float32, dim>` column with real (non-null)
/// embedding values from `embeddings`.
fn build_real_embedding(
    embeddings: &[Vec<f32>],
    dim: usize,
) -> arrow::array::FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(
        Float32Builder::new(),
        i32::try_from(dim).expect("embedding dim fits in i32"),
    );
    for emb in embeddings {
        for &val in emb {
            builder.values().append_value(val);
        }
        builder.append(true); // non-null
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_io::output_schema;
    use arrow::array::Array;

    fn sample_chunks() -> Vec<ChunkSpec> {
        vec![
            ChunkSpec {
                chunk_index: 0,
                text: "First chunk.".to_string(),
                token_count: 3,
                source_path: "doc.md".to_string(),
                heading_path: vec!["Intro".to_string()],
                section_kind: SectionKind::Paragraph,
                char_offset_start: 0,
                char_offset_end: 12,
            },
            ChunkSpec {
                chunk_index: 1,
                text: "Second chunk with code.".to_string(),
                token_count: 5,
                source_path: "doc.md".to_string(),
                heading_path: vec!["Intro".to_string(), "Setup".to_string()],
                section_kind: SectionKind::Code,
                char_offset_start: 12,
                char_offset_end: 35,
            },
            ChunkSpec {
                chunk_index: 2,
                text: "Third.".to_string(),
                token_count: 1,
                source_path: "doc.md".to_string(),
                heading_path: vec![],
                section_kind: SectionKind::ListItem,
                char_offset_start: 35,
                char_offset_end: 41,
            },
        ]
    }

    #[test]
    fn batch_has_correct_schema_and_row_count() {
        let chunks = sample_chunks();
        let batch = chunks_to_batch(&chunks).expect("batch should build");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 9);
        assert_eq!(batch.schema().as_ref(), output_schema().as_ref());
    }

    #[test]
    fn chunk_index_column_is_correct() {
        let batch = chunks_to_batch(&sample_chunks()).unwrap();
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("chunk_index should be UInt32");
        assert_eq!(col.value(0), 0);
        assert_eq!(col.value(1), 1);
        assert_eq!(col.value(2), 2);
    }

    #[test]
    fn text_column_preserves_content() {
        let batch = chunks_to_batch(&sample_chunks()).unwrap();
        let col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text should be Utf8");
        assert_eq!(col.value(0), "First chunk.");
        assert_eq!(col.value(1), "Second chunk with code.");
        assert_eq!(col.value(2), "Third.");
    }

    #[test]
    fn heading_path_handles_empty_and_nested() {
        let batch = chunks_to_batch(&sample_chunks()).unwrap();
        let col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("heading_path should be List");

        // Row 0: ["Intro"] — non-null
        assert!(!col.is_null(0));
        // Row 2: [] — null (empty heading_path)
        assert!(col.is_null(2));
    }

    #[test]
    fn section_kind_dictionary_round_trips() {
        let batch = chunks_to_batch(&sample_chunks()).unwrap();
        let col = batch.column(5);
        assert_eq!(
            col.data_type(),
            &arrow::datatypes::DataType::Dictionary(
                Box::new(arrow::datatypes::DataType::Int8),
                Box::new(arrow::datatypes::DataType::Utf8),
            ),
        );
    }

    #[test]
    fn embedding_column_is_all_null() {
        let batch = chunks_to_batch(&sample_chunks()).unwrap();
        let col = batch.column(8);
        assert_eq!(col.null_count(), 3);
    }

    #[test]
    fn empty_chunks_produces_empty_batch() {
        let batch = chunks_to_batch(&[]).expect("empty batch should build");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 9);
    }
}

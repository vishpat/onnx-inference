use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Axis, Ix3};
use ort::{
    Error, inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokenizers::Tokenizer;

fn cosine_similarity(embedding1: &[f32], embedding2: &[f32]) -> f32 {
    let dot_product = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>();
    let norm1 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm1 * norm2)
}

struct EmbeddingGenerator {
    tokenizer: Tokenizer,
    session: Session,
}

impl EmbeddingGenerator {
    async fn new() -> Result<Self> {
        let tokenizer_path = "./tokenizer.json".to_string();
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(1)?
            .commit_from_file("./model.onnx".to_string())?;

        Ok(Self { tokenizer, session })
    }

    fn generate_embeddings(&mut self, text: &[String]) -> Result<()> {
        let encodings = self
            .tokenizer
            .encode_batch(text.to_vec(), false)
            .map_err(|e| Error::new(e.to_string()))?;

        let padded_token_length = encodings[0].len();

        // Get our token IDs & mask as a flattened array.
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let token_type_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
            .collect();

        // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
        let a_ids = TensorRef::from_array_view(([text.len(), padded_token_length], &*ids))?;
        let a_mask = TensorRef::from_array_view(([text.len(), padded_token_length], &*mask))?;
        let token_type_ids =
            TensorRef::from_array_view(([text.len(), padded_token_length], &*token_type_ids))?;

        let outputs = self.session.run(inputs![a_ids, a_mask, token_type_ids])?;
        let embeddings = outputs[0]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()
            .unwrap();
        let query = embeddings.index_axis(Axis(0), 0);
        for embedding in embeddings.axis_iter(Axis(0)).skip(1) {
            let vec1 = query.iter().copied().collect::<Vec<f32>>();
            let vec2 = embedding.iter().copied().collect::<Vec<f32>>();
            println!("similarity: {:?}", cosine_similarity(&vec1, &vec2));
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing embedding generator...");
    let mut generator = EmbeddingGenerator::new()
        .await
        .context("Failed to initialize embedding generator")?;

    let sample_texts = vec![
        "The fox ran into the jungle"
            .to_string(),
        "The fox towards the forest"
            .to_string(),
        "Little Miss Muffet sat on a tuffet, eating her curds and whey. Along came a spider and sat down beside her and frightened Miss Muffet away."
            .to_string(),
    ];

    generator
        .generate_embeddings(&sample_texts)
        .context("Failed to generate embeddings")?;
    Ok(())
}

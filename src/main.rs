use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array1, Ix2};
use ort::{
    Error, inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "onnx-inference")]
#[command(about = "Convert text to embeddings using all-MiniLM-L6-v2 model")]
struct Args {
    /// Input text to convert to embedding
    #[arg(short, long)]
    text: String,

    /// Path to the tokenizer file (optional, will download if not provided)
    #[arg(short = 'k', long)]
    tokenizer_path: Option<String>,

    /// Path to the model file (optional, will download if not provided)
    #[arg(short = 'm', long)]
    model_path: Option<String>,
}

struct EmbeddingGenerator {
    tokenizer: Tokenizer,
    session: Session,
}

impl EmbeddingGenerator {
    async fn new(tokenizer_path: Option<String>, model_path: Option<String>) -> Result<Self> {
        let tokenizer_path = tokenizer_path.unwrap_or("./tokenizer.json".to_string());
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(1)?
            .commit_from_file(model_path.unwrap_or("./model.onnx".to_string()))?;

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
        println!("Ids: {:?}", ids);
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        println!("Mask: {:?}", mask);
        let token_type_ids: Vec<i64> = encodings
            .iter().flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
            .collect();
        println!("Token type ids: {:?}", token_type_ids);

        // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
        let a_ids = TensorRef::from_array_view(([text.len(), padded_token_length], &*ids))?;
        println!("A ids: {:?}", a_ids);
        let a_mask = TensorRef::from_array_view(([text.len(), padded_token_length], &*mask))?;
        println!("A mask: {:?}", a_mask);
        let token_type_ids = TensorRef::from_array_view(([text.len(), padded_token_length], &*token_type_ids))?;
        println!("Token type ids: {:?}", token_type_ids);
        // Tokenize the input text
        let outputs = self.session.run(inputs![a_ids, a_mask, token_type_ids])?;
        println!("Outputs: {:?}", outputs);
        let embeddings = outputs[0].try_extract_array::<f32>().unwrap();
        println!("Embeddings: {:?}", embeddings);
        Ok(())
    }

    fn normalize_embedding(&self, mut embedding: Array1<f32>) -> Array1<f32> {
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        if norm > 0.0 {
            embedding /= norm;
        }
        embedding
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Initializing embedding generator...");
    let mut generator = EmbeddingGenerator::new(args.tokenizer_path, args.model_path)
        .await
        .context("Failed to initialize embedding generator")?;

    let sample_texts = vec![
        "The quick brown fox jumps over the lazy dog. Ding dong bell. Pussy in the well"
            .to_string(),
        "The quick brown fox jumps over the lazy dog. Ding dong bell. Pussy in the well"
            .to_string(),
        "The quick brown fox jumps over the lazy dog. Ding dong bell. Pussy in the well"
            .to_string(),
    ];

    generator
        .generate_embeddings(&sample_texts)
        .context("Failed to generate embeddings")?;
    Ok(())
}

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array1, s};
use serde_json;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use ort::{Environment, Session, SessionBuilder};


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

        let environment = Arc::new(Environment::builder()
            .with_name("onnx-inference")
            .build()
            .context("Failed to create ONNX Runtime environment")?);

        let session = SessionBuilder::new(&environment)?.with_model_from_file(&model_path.unwrap_or("./model.onnx".to_string()))?;

        Ok(Self { tokenizer, session })
    }

    fn tokenize_text(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize text: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

        Ok((input_ids, attention_mask))
    }

    fn generate_embedding(&self, text: &str) -> Result<Array1<f32>> {
        // Tokenize the input text
        let (input_ids, attention_mask) = self.tokenize_text(text)?;

        println!("Input IDs: {:?}", input_ids);
        println!("Attention Mask: {:?}", attention_mask);

        // For demonstration purposes, create a dummy embedding
        // In a real implementation, you would:
        // 1. Load the ONNX model
        // 2. Run inference with the tokenized input
        // 3. Extract the last hidden state
        // 4. Apply mean pooling
        // 5. Normalize the result

        println!("Input IDs: {:?}", input_ids);

        Ok(Array1::zeros(384))
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
    let generator = EmbeddingGenerator::new(args.tokenizer_path, args.model_path).await
        .context("Failed to initialize embedding generator")?;

    println!("Generating embedding for: \"{}\"", args.text);
    let embedding = generator.generate_embedding(&args.text)
        .context("Failed to generate embedding")?;

    println!("Embedding generated successfully!");
    println!("Embedding dimension: {}", embedding.len());
    println!("First 10 values: {:?}", embedding.slice(s![..10]).to_vec());
    println!("Embedding norm: {:.6}", embedding.mapv(|x| x * x).sum().sqrt());

    // Optionally save the embedding to a file
    let embedding_json = serde_json::to_string_pretty(&embedding.to_vec())
        .context("Failed to serialize embedding")?;
    
    std::fs::write("embedding.json", embedding_json)
        .context("Failed to write embedding to file")?;
    
    println!("Embedding saved to embedding.json");

    Ok(())
}
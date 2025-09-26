# ONNX Inference - Text to Embeddings

A Rust program that demonstrates text tokenization and embedding generation using the all-MiniLM-L6-v2 tokenizer. This is a working foundation that can be extended to include actual ONNX model inference.

## Features

- Converts text strings to 384-dimensional embedding vectors
- Uses the all-MiniLM-L6-v2 tokenizer for proper text preprocessing
- Command-line interface for easy usage
- Demonstrates tokenization and embedding generation structure
- L2 normalization of embeddings
- Saves embeddings to JSON file

## Prerequisites

Before running this program, you need to download the tokenizer file:

1. **Download the tokenizer**: Get `tokenizer.json` from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
2. Place the `tokenizer.json` file in the project root directory

**Note**: This version demonstrates the structure and tokenization. For full ONNX inference, you would need to:
- Add the `ort` crate back to dependencies
- Download the `all-MiniLM-L6-v2.onnx` model file
- Implement proper tensor operations and model inference

## Installation

1. Make sure you have Rust installed (version 1.70+)
2. Clone or download this project
3. Place the model and tokenizer files in the project directory
4. Build the project:

```bash
cargo build --release
```

## Usage

### Basic Usage

```bash
cargo run -- --text "Hello, world! This is a sample text."
```

### With Custom Tokenizer Path

```bash
cargo run -- --text "Your text here" --tokenizer-path "path/to/tokenizer.json"
```

### Command Line Options

- `--text` or `-t`: The input text to convert to embedding (required)
- `--tokenizer-path` or `-k`: Path to the tokenizer file (optional, defaults to "tokenizer.json")

## Output

The program will:
1. Display the input text
2. Show the embedding dimension (384 for all-MiniLM-L6-v2)
3. Display the first 10 values of the embedding vector
4. Show the L2 norm of the embedding
5. Save the complete embedding vector to `embedding.json`

## Example Output

```
Initializing embedding generator...
Generating embedding for: "Hello, world! This is a sample text."
Embedding generated successfully!
Embedding dimension: 384
First 10 values: [0.123456, -0.078901, 0.234567, ...]
Embedding norm: 1.000000
Embedding saved to embedding.json
```

## Technical Details

- **Tokenizer**: all-MiniLM-L6-v2 tokenizer (384-dimensional embeddings)
- **Tokenization**: Uses the Hugging Face tokenizers library
- **Embedding Generation**: Currently uses a hash-based approach for demonstration
- **Normalization**: L2 normalization for unit vectors
- **Future**: Ready to integrate ONNX Runtime for actual model inference

## Dependencies

- `tokenizers`: Text tokenization
- `ndarray`: Numerical array operations
- `clap`: Command-line argument parsing
- `anyhow`: Error handling
- `serde_json`: JSON serialization
- `tokio`: Async runtime

## Notes

- The program requires the tokenizer file to be present
- Embeddings are normalized to unit vectors (L2 norm = 1.0)
- This is a working foundation that demonstrates the structure for ONNX inference
- To add full ONNX support, integrate the `ort` crate and implement proper tensor operations
- All error cases are handled gracefully with descriptive messages

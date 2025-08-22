A semantic search system that embeds Quora questions with a transformer model, stores them in Pinecone, and retrieves the most similar questions for any user query.

Dataset Loading & Preprocessing

- Uses Hugging Face’s `load_dataset` to fetch a subset (50k rows) of the Quora dataset.

- Extracts all text questions from records, deduplicates them using set, and prepares them for embedding.

- This ensures the input corpus is clean and not overly redundant before storage.

Embedding Model Setup

- Loads the `all-MiniLM-L6-v2` `SentenceTransformer` model, a compact and efficient transformer for sentence embeddings.

- Determines runtime device using torch (`cuda` if GPU available, otherwise `CPU`).

- Converts both dataset questions and user queries into numerical `vectors` (embeddings) that capture semantic meaning.

Pinecone Vector Database Initialization

- Retrieves the `Pinecone API key` from environment variables via a utility class.

- Creates a serverless `Pinecone` index with appropriate settings: embedding dimension (matching the model), cosine similarity metric, and AWS us-west-2 region.

- If an index with the same name exists, it deletes and recreates it to avoid conflicts.

Batch Embedding & Upsert to Pinecone

- Iterates through the questions in manageable batches (200 at a time, up to 10k total).

- For each batch, generates embeddings with the model, builds metadata dictionaries ({'text': question}), and assigns unique IDs.

- Uploads (upserts) these vectors into the Pinecone index, allowing semantic search to run efficiently at scale.

Semantic Search via Query Function

- A helper function `run_query(query)` embeds a user’s search query into a vector.

- Submits the embedding to `Pinecone` for similarity search (index.query), retrieving the top 10 closest matching questions based on `cosine similarity`.

- Displays results ranked by `semantic closeness`, showing how natural‑language queries return meaningful related questions.

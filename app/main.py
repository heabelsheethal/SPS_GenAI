
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.word_embeddings import WordEmbeddings
import numpy as np

app = FastAPI()

# ------------------------- Root -------------------------
@app.get("/")
def read_root():
    return {"Hello": "World"}


# ------------------------- Bigram Model -------------------------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus, frequency_threshold=5)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

# ------------------------- Bigram Endpoints -------------------------
@app.post("/bigram")
def generate_text(request: TextGenerationRequest):
    return {"generated_text": bigram_model.generate_text(request.start_word, request.length)}


# ------------------------- Gaussian Endpoints -------------------------
@app.get("/gaussian")
def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> list[float]:
    std_dev = np.sqrt(variance)
    return np.random.normal(mean, std_dev, size).tolist()


# ------------------------- Word Embeddings -------------------------
try:
    embeddings = WordEmbeddings(model_name="en_core_web_lg")
except RuntimeError as e:
    embeddings = None
    embeddings_error_msg = str(e)

class EmbedTextRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str
    infos: list[str]

def _check_embeddings_available():
    if embeddings is None:
        raise HTTPException(status_code=500, detail=embeddings_error_msg)

# Word vector endpoint
@app.get("/embeddings/word")
def embed_word(word: str):
    _check_embeddings_available()
    try:
        return {"word": word, "vector": embeddings.calculate_embedding(word)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Word similarity endpoint
@app.get("/embeddings/word/similarity")
def embed_similarity(a: str, b: str):
    _check_embeddings_available()
    return {"a": a, "b": b, "similarity": embeddings.calculate_similarity(a, b)}

# Sentence embedding endpoint
@app.post("/embeddings/sentence")
def embed_sentence(request: EmbedTextRequest):
    _check_embeddings_available()
    return {"text": request.text, "vector": embeddings.get_sentence_embedding(request.text)}

# Query similarity endpoint
@app.post("/embeddings/sentences/similarity")
def embed_query(request: QueryRequest):
    _check_embeddings_available()
    results = embeddings.sentence_similarity(request.query, request.infos)
    return {"query": request.query, "results": results}




# ------------------------- Run the app -------------------------
# To run the app, use the command: 

# Option 1: with uvicorn
# uvicorn app.main:app --reload

# Option 2: with uv (if you’re using uv as your package manager)
# uv run fastapi dev app/main.py


# ------------------------- HTTP Endpoints -------------------------
# root JSON ->  http://127.0.0.1:8000/

# interactive API  ->  http://127.0.0.1:8000/docs
#                  ->  http://127.0.0.1:8000/redoc


# ------------------------- curl command in terminal (while FastAPI is running) -------------------------

# Root Endpoint (GET)
"""
curl http://127.0.0.1:8000/
"""

# 2) Bigram Text Generation (POST)
"""
curl -X POST "http://127.0.0.1:8000/bigram" \
     -H "Content-Type: application/json" \
     -d '{
           "start_word": "the",
           "length": 10
         }'
"""

# 3) Gaussian Sampling (GET)
"""
curl "http://127.0.0.1:8000/gaussian?mean=5&variance=2&size=3"
"""

# 4) Word Embedding (GET)
"""
curl "http://127.0.0.1:8000/embeddings/word?word=hello"
"""

# 5) Word Similarity (GET)
"""
curl "http://127.0.0.1:8000/embeddings/word/similarity?a=hello&b=hi"
"""

# 6) Sentence Embedding (POST)
"""
curl -X POST "http://127.0.0.1:8000/embeddings/sentence" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "This is a test sentence."
         }'
"""

# 7) Query Similarity (POST)
"""
curl -X POST "http://127.0.0.1:8000/embeddings/sentences/similarity" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "I like apples",
           "infos": ["I love apples", "I hate oranges"]
         }'
"""


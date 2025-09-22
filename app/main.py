from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.word_embeddings import WordEmbeddings
import numpy as np

app = FastAPI()

# ------------------------- Bigram Model -------------------------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus, frequency_threshold=5)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

# ------------------------- Root -------------------------
@app.get("/")
def read_root():
    return {"Hello": "World"}

# ------------------------- Bigram Endpoints -------------------------
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    return {"generated_text": bigram_model.generate_text(request.start_word, request.length)}

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
@app.get("/embed/word")
def embed_word(word: str):
    _check_embeddings_available()
    try:
        return {"word": word, "vector": embeddings.calculate_embedding(word)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Word similarity endpoint
@app.get("/embed/similarity")
def embed_similarity(a: str, b: str):
    _check_embeddings_available()
    return {"a": a, "b": b, "similarity": embeddings.calculate_similarity(a, b)}

# Sentence embedding endpoint
@app.post("/embed/sentence")
def embed_sentence(request: EmbedTextRequest):
    _check_embeddings_available()
    return {"text": request.text, "vector": embeddings.get_sentence_embedding(request.text)}

# Query similarity endpoint
@app.post("/embed/query")
def embed_query(request: QueryRequest):
    _check_embeddings_available()
    results = embeddings.sentence_similarity(request.query, request.infos)
    return {"query": request.query, "results": results}




# ------------------------- Run the app -------------------------
# To run the app, use the command: 
# uvicorn app.main:app --reload

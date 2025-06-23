import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    INDEX_NAME = "stem-tutor-index"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    QA_MODEL = "deepset/roberta-base-squad2"
    CHUNK_SIZE = 768  # Increased for complex subjects
    CHUNK_OVERLAP = 100
    TOP_K = 5
    REPETITION_INTERVALS = [1, 3, 7, 14, 30]
    SIMILARITY_THRESHOLD = 0.85
    DEVICE = None  # Will be set later
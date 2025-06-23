import chromadb
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from ai_tutor_bot.utils.config import Config
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        
        self.collection = self.client.get_or_create_collection(
            name=Config.INDEX_NAME,
            embedding_function=self.sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )

    async def async_upsert(self, vectors: List[Tuple[str, List[float], Dict]]):
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        
        for vector_id, embedding, metadata in vectors:
            ids.append(vector_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Vector upsert failed: {e}")

    async def async_query(self, vector: List[float], filter: Optional[Dict] = None) -> List[Dict]:
        # Convert to numpy array
        query_embedding = np.array(vector).reshape(1, -1).tolist()[0]
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=Config.TOP_K,
                where=filter,
                include=["metadatas", "distances"]
            )
            
            # Format results
            matches = []
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i]
                })
            
            return matches
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
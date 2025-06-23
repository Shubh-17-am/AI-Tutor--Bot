import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from ai_tutor_bot.utils.config import Config
from ai_tutor_bot.utils.text_processor import TextProcessor
from ai_tutor_bot.db.vector_db import VectorDBManager
from ai_tutor_bot.utils.adaptive_learning import AdaptiveLearningSystem
import logging

logger = logging.getLogger(__name__)

class TutorAgent:
    def __init__(self):
        # Set device in Config
        Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device set to use {Config.DEVICE}")
        
        self.embed_model = self._load_embedding_model()
        self.qa_pipeline = self._load_qa_model()
        self.db_manager = VectorDBManager()
        self.learning_system = AdaptiveLearningSystem()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.QA_MODEL)

    def _load_embedding_model(self) -> SentenceTransformer:
        return SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)

    def _load_qa_model(self) -> Any:
        return pipeline(
            'question-answering',
            model=Config.QA_MODEL,
            tokenizer=Config.QA_MODEL,
            device=0 if Config.DEVICE == "cuda" else -1
        )

    async def ingest_documents(self, documents: List[Dict[str, str]]):
        vectors = []
        for doc in documents:
            text = doc.get('text', '')
            if not text or not isinstance(text, str):
                logger.warning(f"Skipping document {doc.get('id')} with invalid text")
                continue
                
            chunks = TextProcessor.chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
            
            # Filter out invalid chunks
            chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
            
            if not chunks:
                logger.warning(f"No valid chunks found for document {doc.get('id')}")
                continue
                
            # Encode chunks with error handling
            try:
                embeddings = self.embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                continue
                
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    metadata = {
                        "source": doc.get("source", ""),
                        "doc_id": doc.get("id", ""),
                        "chunk_id": str(idx),
                        "text": chunk,
                        **TextProcessor.extract_metadata(chunk)
                    }
                    vector_id = f"{doc['id']}_{idx}"
                    vectors.append((vector_id, embedding.tolist(), metadata))
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
        
        if vectors:
            await self.db_manager.async_upsert(vectors)
            logger.info(f"Ingested {len(vectors)} vectors from {len(documents)} documents")
        else:
            logger.warning("No vectors to ingest")

    async def generate_response(self, user_id: str, query: str) -> Dict[str, Any]:
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "answer": "Please provide a valid question",
                "context": "",
                "concepts": [],
                "relevance_score": 0,
                "sources": []
            }
            
        try:
            query_embed = self.embed_model.encode([query])[0].tolist()
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return {
                "answer": "I couldn't process your question",
                "context": "",
                "concepts": [],
                "relevance_score": 0,
                "sources": []
            }
        
        # Get user's concepts for filtering
        user_concepts = self.learning_system.user_progress.get(user_id, {}).keys()
        
        # Create concept filter if any concepts exist
        filter = None
        if user_concepts:
            # Build OR filter to match any of the user's concepts
            filter = {"$or": [{"concepts": {"$contains": concept}} for concept in user_concepts]}
        
        # Retrieve relevant chunks
        try:
            results = await self.db_manager.async_query(query_embed, filter)
            
            # Fallback to unfiltered search if no results
            if not results:
                logger.info("No results with concept filter, trying unfiltered search")
                results = await self.db_manager.async_query(query_embed, None)
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            results = []
        
        # If still no results, return empty response
        if not results:
            return {
                "answer": "I couldn't find relevant information to answer your question",
                "context": "",
                "concepts": [],
                "relevance_score": 0,
                "sources": []
            }
        
        # Extract concepts from results
        context_concepts = set()
        for res in results:
            concepts = res["metadata"].get('concepts', '')
            if concepts:
                context_concepts.update(concepts.split(", "))
        
        # Prioritize concepts based on learning progress
        prioritized_concepts = self.learning_system.get_learning_context(
            user_id, list(context_concepts))
        
        # Build context from retrieved chunks
        context = "\n".join([res["metadata"]['text'] for res in results])
        
        # Generate answer using Q&A pipeline
        try:
            answer = self.qa_pipeline(question=query, context=context)['answer']
        except Exception as e:
            logger.error(f"QA pipeline failed: {e}")
            answer = "I couldn't generate an answer for that question. Please try again."
        
        # Update learning system with top concepts
        for concept in prioritized_concepts[:2]:
            self.learning_system.update_progress(user_id, concept)
        
        # Calculate relevance score
        try:
            answer_embed = self.embed_model.encode([answer])[0].reshape(1, -1)
            query_embed_2d = np.array(query_embed).reshape(1, -1)
            relevance_score = cosine_similarity(query_embed_2d, answer_embed)[0][0]
        except Exception as e:
            logger.error(f"Relevance score calculation failed: {e}")
            relevance_score = 0.0
        
        return {
            "answer": answer,
            "context": context,
            "concepts": prioritized_concepts,
            "relevance_score": relevance_score,
            "sources": list(set(res["metadata"]['source'] for res in results))
        }

    async def handle_query(self, user_id: str, query: str) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            response = await self.generate_response(user_id, query)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            response = {
                "answer": "I encountered an error processing your request",
                "context": "",
                "concepts": [],
                "relevance_score": 0,
                "sources": []
            }
            
        response['latency'] = (datetime.now() - start_time).total_seconds()
        response['user_id'] = user_id
        return response
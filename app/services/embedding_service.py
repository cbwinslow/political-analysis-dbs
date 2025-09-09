import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.openai_client = None
        self.localai_url = os.getenv("LOCALAI_URL", "http://localhost:8080")
        
        # Initialize sentence transformer as fallback
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using available service"""
        if not text or not text.strip():
            return None
        
        # Try LocalAI first
        try:
            embedding = await self._generate_localai_embedding(text)
            if embedding:
                return embedding
        except Exception as e:
            print(f"LocalAI embedding failed: {e}")
        
        # Try OpenAI if API key is available
        try:
            if os.getenv("OPENAI_API_KEY"):
                embedding = await self._generate_openai_embedding(text)
                if embedding:
                    return embedding
        except Exception as e:
            print(f"OpenAI embedding failed: {e}")
        
        # Fallback to local sentence transformer
        try:
            if self.model:
                embedding = self.model.encode(text, convert_to_tensor=False)
                # Pad or truncate to 1536 dimensions to match OpenAI format
                if len(embedding) < 1536:
                    padding = np.zeros(1536 - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                elif len(embedding) > 1536:
                    embedding = embedding[:1536]
                return embedding.tolist()
        except Exception as e:
            print(f"Local embedding failed: {e}")
        
        return None
    
    async def _generate_localai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using LocalAI"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.localai_url}/v1/embeddings",
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
        
        return None
    
    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI"""
        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        response = await self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        return response.data[0].embedding
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            if candidate:
                similarity = self.cosine_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
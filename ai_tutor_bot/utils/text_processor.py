import re
import string
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Robust text chunking that preserves mathematical expressions"""
        if not text or not isinstance(text, str):
            return []
            
        # Split on sentence boundaries, preserving mathematical expressions
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        sentences = re.split(f'({sentence_endings}|\\n\\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length <= chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                overlap = []
                overlap_length = 0
                while current_chunk and overlap_length < chunk_overlap:
                    popped = current_chunk.pop(0)
                    popped_words = popped.split()
                    overlap.append(popped)
                    overlap_length += len(popped_words)
                current_chunk = overlap + [sentence]
                current_length = overlap_length + sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        metadata = {}
        try:
            # Subject detection
            subjects = {
                "math": ["equation", "theorem", "calculate", "solve", "derivative", "integral"],
                "physics": ["force", "energy", "velocity", "quantum", "electron"],
                "chemistry": ["atom", "molecule", "reaction", "bond", "pH"],
                "biology": ["cell", "dna", "protein", "photosynthesis", "gene"]
            }
            
            # Detect subject
            subject = "general"
            for sub, keywords in subjects.items():
                if any(keyword in text.lower() for keyword in keywords):
                    subject = sub
                    break
                    
            metadata['subject'] = subject
            
            # Extract key terms and formulas
            concepts = []
            # Capture mathematical expressions
            formulas = re.findall(r'[a-zA-Z0-9_]+[\s]*[=+\-*/^][\s]*[a-zA-Z0-9_]+', text)
            concepts.extend(formulas)
            
            # Capture theorem names
            theorems = re.findall(r'[A-Z][a-z]+(\'s)? [A-Z][a-z]+ (Theorem|Law|Identity)', text)
            concepts.extend([t[0] for t in theorems])
            
            # Capture important terms
            terms = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*', text)
            concepts.extend(terms)
            
            # Filter and limit concepts
            if concepts:
                unique_concepts = list(set(concepts))[:5]
                metadata['concepts'] = ", ".join(unique_concepts)
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            
        return metadata
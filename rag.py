import os
import re
import json
import math
import hashlib
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables
load_dotenv()

# Set up Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CACHE_FILE = ".embeddings_cache.json"

def tokenize(text):
    """Tokenizes text into a list of words for lexical matching."""
    return re.findall(r'\w+', text.lower())

def get_hash(text):
    """Generates an MD5 hash of text for caching embeddings."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_lengths = [len(tokenize(c['content'])) for c in chunks]
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1
        self.doc_freqs = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for chunk in self.chunks:
            words = set(tokenize(chunk['content']))
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        
        for word, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query):
        query_words = tokenize(query)
        scores = [0.0] * self.corpus_size
        for q_word in query_words:
            if q_word not in self.idf:
                continue
            idf = self.idf[q_word]
            for i, chunk in enumerate(self.chunks):
                content_words = tokenize(chunk['content'])
                freq = content_words.count(q_word)
                doc_len = self.doc_lengths[i]
                
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[i] += idf * (numerator / denominator)
        return scores

class RAGPipeline:
    def __init__(self, docs_dir="AboutMLSC"):
        self.docs_dir = docs_dir
        self.chunks = []
        self.bm25 = None
        self.embedding_cache = {}
        self.load_documents()
        self.load_embedding_cache()
        self.build_retrievers()

    def load_documents(self):
        """Loads and chunks documents from the specified directory."""
        if not os.path.exists(self.docs_dir):
            print(f"Warning: Document directory '{self.docs_dir}' not found.")
            return

        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".txt") and not filename.startswith("._"):
                filepath = os.path.join(self.docs_dir, filename)
                self.chunks.extend(self.chunk_file(filepath))
        
        print(f"Loaded {len(self.chunks)} chunks from {self.docs_dir}.")

    def chunk_file(self, filepath):
        """Chunks a single file by double newline/paragraphs, retaining line numbers."""
        filename = os.path.basename(filepath)
        
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        chunks = []
        current_chunk_lines = []
        chunk_start_line = 1
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            if stripped == '':
                if current_chunk_lines:
                    content = '\n'.join(current_chunk_lines).strip()
                    if content:
                        chunks.append({
                            'source': filename,
                            'content': content,
                            'start_line': chunk_start_line,
                            'end_line': line_num - 1,
                            'hash': get_hash(content)
                        })
                    current_chunk_lines = []
                chunk_start_line = line_num + 1
            else:
                if not current_chunk_lines:
                    chunk_start_line = line_num
                current_chunk_lines.append(line.rstrip('\r\n'))
                
        # Handle trailing lines
        if current_chunk_lines:
            content = '\n'.join(current_chunk_lines).strip()
            if content:
                chunks.append({
                    'source': filename,
                    'content': content,
                    'start_line': chunk_start_line,
                    'end_line': len(lines),
                    'hash': get_hash(content)
                })
                
        return chunks

    def load_embedding_cache(self):
        """Loads dense embeddings from local JSON cache."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                print(f"Loaded {len(self.embedding_cache)} embeddings from cache.")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.embedding_cache = {}

    def save_embedding_cache(self):
        """Saves current embedding cache to local JSON file."""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving embedding cache: {e}")

    def build_retrievers(self):
        """Builds retrievers: BM25 (lexical) and Gemini (dense)."""
        if not self.chunks:
            return
        
        # Build Lexical
        self.bm25 = BM25Retriever(self.chunks)
        
        # Build/Load Dense Embeddings if API key is set
        if GEMINI_API_KEY:
            missing_hashes = []
            missing_texts = []
            for chunk in self.chunks:
                h = chunk['hash']
                if h not in self.embedding_cache:
                    missing_hashes.append(h)
                    missing_texts.append(chunk['content'])
            
            if missing_texts:
                print(f"Generating embeddings for {len(missing_texts)} new chunks...")
                try:
                    # Embed in batches to prevent hitting API limits
                    batch_size = 50
                    for i in range(0, len(missing_texts), batch_size):
                        batch_texts = missing_texts[i:i+batch_size]
                        batch_hashes = missing_hashes[i:i+batch_size]
                        
                        response = genai.embed_content(
                            model="models/text-embedding-004",
                            content=batch_texts,
                            task_type="retrieval_document"
                        )
                        embeddings = response['embeddings']
                        for h, emb in zip(batch_hashes, embeddings):
                            self.embedding_cache[h] = emb
                    
                    self.save_embedding_cache()
                    print("Embeddings generation complete.")
                except Exception as e:
                    print(f"Error generating embeddings: {e}")
        else:
            print("GEMINI_API_KEY not set. Dense retriever is running in dry mode (Lexical only).")

    def retrieve(self, query, top_k=3, k_rrf=60):
        """Performs hybrid retrieval using BM25, Gemini embeddings, and RRF."""
        if not self.chunks:
            return []

        # 1. Sparse Search
        bm25_scores = self.bm25.get_scores(query)
        
        # 2. Dense Search
        dense_scores = [0.0] * len(self.chunks)
        if GEMINI_API_KEY and self.embedding_cache:
            try:
                # Embed query
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                query_emb = response['embedding']
                
                # Compute Cosine Similarity
                q_vec = np.array(query_emb)
                for i, chunk in enumerate(self.chunks):
                    h = chunk['hash']
                    if h in self.embedding_cache:
                        c_vec = np.array(self.embedding_cache[h])
                        dot = np.dot(q_vec, c_vec)
                        norm_q = np.linalg.norm(q_vec)
                        norm_c = np.linalg.norm(c_vec)
                        if norm_q > 0 and norm_c > 0:
                            dense_scores[i] = float(dot / (norm_q * norm_c))
            except Exception as e:
                print(f"Error in dense retrieval: {e}. Falling back to lexical only.")
                # If dense retrieval fails, we just rely on lexical scores

        # 3. Reciprocal Rank Fusion (RRF)
        # Lexical rank mapping
        lexical_ranks = np.argsort(bm25_scores)[::-1]
        lex_rank_map = {idx: rank + 1 for rank, idx in enumerate(lexical_ranks)}
        
        # Dense rank mapping
        dense_ranks = np.argsort(dense_scores)[::-1]
        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}
        
        # Combine using RRF
        rrf_scores = {}
        for idx in range(len(self.chunks)):
            score = 0.0
            if idx in lex_rank_map and bm25_scores[idx] > 0:
                score += 1.0 / (k_rrf + lex_rank_map[idx])
            if idx in dense_rank_map and dense_scores[idx] > 0:
                score += 1.0 / (k_rrf + dense_rank_map[idx])
            rrf_scores[idx] = score
            
        # Sort and select top_k
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        retrieved = []
        for idx in sorted_indices[:top_k]:
            # Make sure we only retrieve chunks with positive matching signal,
            # or if it's the very top result to avoid empty context when there is a trace score
            if rrf_scores[idx] > 0 or len(retrieved) == 0:
                retrieved.append({
                    'chunk': self.chunks[idx],
                    'lexical_score': bm25_scores[idx],
                    'dense_score': dense_scores[idx],
                    'rrf_score': rrf_scores[idx]
                })
                
        return retrieved

    def generate_answer(self, query, retrieved_contexts):
        """Generates answer using gemini-2.5-flash with strict context grounding."""
        if not GEMINI_API_KEY:
            # Local fallback mock mode
            return self._generate_fallback(query, retrieved_contexts)

        # Build prompt
        context_str = ""
        for i, ctx in enumerate(retrieved_contexts):
            chunk = ctx['chunk']
            context_str += f"[{i+1}] (Source: {chunk['source']}, lines {chunk['start_line']}-{chunk['end_line']}):\n{chunk['content']}\n\n"

        prompt = f"""You are the MLSC Knowledge Assistant, an AI helper for the Microsoft Learn Student Community (MLSC) VIT Pune.
Your job is to answer user questions using ONLY the provided knowledge base segments.

STRICT INSTRUCTIONS:
1. Base your answer strictly and only on the text segments provided below.
2. If the answer to the question cannot be directly and fully found in the provided segments, you must answer exactly with this phrase and nothing else:
"I am sorry, but the provided knowledge base does not contain information to answer this question."
3. Do not make up facts, extrapolate information, or use external knowledge about MLSC, VIT, or anything else.
4. If the segments contain some information but not enough to fully answer, state what you can answer from the text and that the rest is unavailable.
5. In your answer, cite the source files using [Source: filename.txt] notation where relevant.

Provided Segments:
{context_str}

User Question: {query}
Answer:"""

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_fallback(self, query, retrieved_contexts):
        """Helper to generate local fallback responses when API key is missing."""
        # Simple rule-based mock logic for demonstration
        if not retrieved_contexts or all(r['rrf_score'] == 0 for r in retrieved_contexts):
            return "I am sorry, but the provided knowledge base does not contain information to answer this question. (Reason: Offline Fallback - No API Key & No Matches)"
        
        # Build brief answer from retrieved content
        source_files = set(r['chunk']['source'] for r in retrieved_contexts)
        best_content = retrieved_contexts[0]['chunk']['content']
        
        answer = f"[OFFLINE FALLBACK - NO GEMINI API KEY SET]\n\n"
        answer += f"Retrieved source: {', '.join(source_files)}\n"
        answer += f"Matching content segment:\n\"{best_content}\""
        return answer

    def answer_query(self, query):
        """Main interface to process query: retrieve then generate."""
        retrieved = self.retrieve(query)
        answer = self.generate_answer(query, retrieved)
        return {
            'query': query,
            'answer': answer,
            'sources': [r['chunk'] for r in retrieved if r['rrf_score'] > 0]
        }

if __name__ == "__main__":
    # Test execution
    pipeline = RAGPipeline("AboutMLSC")
    test_q = "What technical domains exist in MLSC?"
    print(f"\nTesting Query: '{test_q}'")
    res = pipeline.answer_query(test_q)
    print("\nGenerated Answer:\n", res['answer'])
    print("\nSources Cited:")
    for s in res['sources']:
        print(f"- {s['source']} (Lines {s['start_line']}-{s['end_line']})")

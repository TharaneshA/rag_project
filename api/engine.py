import os
import sys
import logging
import time
import pandas as pd
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


class QueryEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Loading local embedding model...")
        self.embeddings = LocalEmbeddings("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded!")
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.df = None
        self.rag_chain = None
        self.vectorstore = None
        self.all_docs = []  # Store docs for BM25
        self._initialized = True
        
        self.processing = False
        self.processing_error = None
        self.processing_progress = 0
        self.total_rows = 0
        self.status_message = ""
    
    def ingest(self, file, text_column=None):
        if file.name.endswith('.xlsx'):
            self.df = pd.read_excel(file)
        else:
            self.df = pd.read_csv(file)
        
        self.total_rows = len(self.df)
        self.processing = True
        self.processing_error = None
        self.processing_progress = 0
        self.rag_chain = None
        self.status_message = "Starting..."
        
        thread = threading.Thread(target=self._process_embeddings_batched)
        thread.daemon = True
        thread.start()
        
        return len(self.df), list(self.df.columns)
    
    def _process_embeddings_batched(self):
        try:
            BATCH_SIZE = 500
            NUM_WORKERS = 4
            start_time = time.time()

            self.status_message = "Clearing old data..."
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")

            self.status_message = "Reading rows..."
            all_docs = []
            all_texts = []
            total = len(self.df)
            total_chars = 0

            for idx, row in self.df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_text.strip():
                    all_docs.append(Document(page_content=row_text, metadata={"row_index": idx}))
                    all_texts.append(row_text)
                    total_chars += len(row_text)

                if idx % 500 == 0:
                    self.processing_progress = idx
                    self.status_message = f"Reading rows: {idx}/{total}"

            total_docs = len(all_docs)
            num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"Rows: {total} | Chunks: {total_docs} | Text size: {total_chars/1024:.1f}KB")
            logger.info(f"Embedding model: all-MiniLM-L6-v2 | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS} | Batches: {num_batches}")

            self.status_message = "Computing embeddings (parallel)..."

            # Split texts into batches for parallel processing
            batches = [all_texts[i:i + BATCH_SIZE] for i in range(0, len(all_texts), BATCH_SIZE)]
            all_embeddings = [None] * len(batches)
            completed = 0

            def embed_batch(batch_idx, texts):
                return batch_idx, self.embeddings.model.encode(texts, show_progress_bar=False).tolist()

            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(embed_batch, i, batch): i for i, batch in enumerate(batches)}

                for future in as_completed(futures):
                    batch_idx, embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                    completed += 1
                    self.processing_progress = int((completed / len(batches)) * total_docs * 0.7)
                    self.status_message = f"Embedding batch {completed}/{len(batches)}"
                    logger.info(f"Embedded batch {completed}/{len(batches)}")

            # Flatten embeddings list
            flat_embeddings = []
            for batch_emb in all_embeddings:
                flat_embeddings.extend(batch_emb)

            logger.info(f"All embeddings computed: {len(flat_embeddings)}")

            # Add to ChromaDB with pre-computed embeddings
            self.status_message = "Storing in vector database..."
            logger.info("Adding to ChromaDB...")

            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )

            CHROMA_BATCH = 1000
            for i in range(0, total_docs, CHROMA_BATCH):
                batch_docs = all_docs[i:i + CHROMA_BATCH]
                batch_texts = [doc.page_content for doc in batch_docs]
                batch_embeddings = flat_embeddings[i:i + CHROMA_BATCH]
                batch_metadatas = [doc.metadata for doc in batch_docs]
                batch_ids = [f"doc_{i+j}" for j in range(len(batch_docs))]

                self.vectorstore._collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                self.processing_progress = int(total_docs * 0.7 + (i / total_docs) * total_docs * 0.3)
                self.status_message = f"Storing: {min(i + CHROMA_BATCH, total_docs)}/{total_docs}"

            self.processing_progress = total_docs

            logger.info("Creating hybrid retriever (BM25 + Semantic)...")
            self.status_message = "Creating hybrid retriever..."

            # Store docs for BM25
            self.all_docs = all_docs

            # Create BM25 retriever 
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 10

            # Create semantic retriever
            semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

            # Combine with ensemble (50% BM25, 50% semantic)
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )

            logger.info("Creating RAG chain...")
            self.status_message = "Finalizing..."

            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=hybrid_retriever
            )
            
            self.processing = False
            self.status_message = "Ready!"
            total_time = time.time() - start_time
            docs_per_sec = total_docs / total_time if total_time > 0 else 0
            logger.info(f"Processing complete! Total time: {total_time:.1f}s | Rate: {docs_per_sec:.1f} docs/sec")

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            logger.error(f"Processing failed: {err}")
            self.processing_error = str(e)
            self.processing = False
    
    def get_status(self):
        return {
            "processing": self.processing,
            "progress": self.processing_progress,
            "total": self.total_rows,
            "error": self.processing_error,
            "ready": self.rag_chain is not None,
            "message": self.status_message
        }
    
    def query(self, question):
        if self.df is None:
            return "No data uploaded yet.", "error"
        if self.processing:
            return f"{self.status_message}", "processing"
        if self.processing_error:
            return f"Error: {self.processing_error}", "error"
        
        route = self._classify(question)
        
        try:
            if route == "structured":
                prefix = f"""You are a data analyst working with a pandas DataFrame called `df`.
DataFrame has {len(self.df)} rows and {len(self.df.columns)} columns.
Columns: {', '.join(self.df.columns.tolist())}

IMPORTANT:
- Always query the FULL dataframe, never use sample or head
- Execute code and return actual numbers/results
- For aggregations, show complete results with all values"""

                agent = create_pandas_dataframe_agent(
                    self.llm, self.df, verbose=True, allow_dangerous_code=True,
                    agent_type="tool-calling",
                    prefix=prefix
                )
                result = agent.invoke(question)
                return result["output"], "text-to-code"
            else:
                if self.rag_chain is None:
                    return "RAG not ready.", "error"
                result = self.rag_chain.invoke({"query": question})
                return result["result"], "rag"
        except Exception as e:
            return f"Error: {str(e)}", "error"
    
    def _classify(self, query):
        """Use LLM to decide if query needs structured (SQL/pandas) or semantic (RAG) approach."""
        columns_info = ", ".join(self.df.columns.tolist()) if self.df is not None else "unknown"

        prompt = f"""You are a query router for a tabular dataset. Decide the best approach:

STRUCTURED (use pandas/SQL operations) - Choose for:
- Counts, totals, sums, averages ("how many", "total", "count", "average")
- Filtering by column values ("where status is open", "tickets from email")
- Grouping/aggregations ("breakdown by", "group by", "per category")
- Sorting/ranking ("top 10", "highest", "lowest", "most common")
- Lookups by ID or specific value ("ticket 190", "find customer John")
- Finding ALL matching records ("find all", "list all", "show all tickets with")
- Comparisons ("how many X vs Y", "more than", "less than")
- Date/time queries ("tickets from last week", "created in January")
- Column-based questions ("what channels", "what types", "unique values")

SEMANTIC (use RAG/similarity search) - Choose for:
- Understanding content/meaning ("what are people saying about", "sentiment")
- Finding similar themes ("issues like", "complaints about", "problems with")
- Summarizing patterns ("common themes", "main concerns")
- Open-ended exploration ("tell me about", "describe", "explain")
- Questions about text content not filterable by exact column values

KEY RULE: If the question can be answered by filtering/counting DataFrame columns, choose STRUCTURED.
Only choose SEMANTIC when you need to understand the meaning of text content.

Dataset columns: {columns_info}

Question: {query}

Reply with exactly one word: STRUCTURED or SEMANTIC"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            route = "structured" if "STRUCTURED" in answer else "semantic"
            logger.info(f"Router decision: {route} (LLM said: {answer})")
            return route
        except Exception as e:
            logger.warning(f"Router error, defaulting to semantic: {e}")
            return "semantic"


engine = QueryEngine()

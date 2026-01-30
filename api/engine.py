import os
import pandas as pd
import threading
import shutil
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


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
        
        print("Loading local embedding model...")
        self.embeddings = LocalEmbeddings("all-MiniLM-L6-v2")
        print("Embedding model loaded!")
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.df = None
        self.rag_chain = None
        self.vectorstore = None
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
            BATCH_SIZE = 500  # Larger batches = faster
            
            self.status_message = "Clearing old data..."
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            
            self.status_message = "Reading rows..."
            print("Starting document creation...")
            
            # Create documents - update progress here
            all_docs = []
            total = len(self.df)
            for idx, row in self.df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_text.strip():
                    all_docs.append(Document(page_content=row_text, metadata={"row_index": idx}))
                
                # Update progress every 100 rows
                if idx % 100 == 0:
                    self.processing_progress = idx
                    self.status_message = f"Reading rows: {idx}/{total}"
            
            self.processing_progress = total
            print(f"Created {len(all_docs)} documents")
            
            # Now embed in batches
            self.status_message = "Embedding..."
            total_docs = len(all_docs)
            self.vectorstore = None
            
            print(f"Starting embedding of {total_docs} documents in batches of {BATCH_SIZE}...")
            
            for i in range(0, total_docs, BATCH_SIZE):
                batch = all_docs[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
                
                self.status_message = f"Embedding batch {batch_num}/{total_batches}"
                self.processing_progress = i
                print(f"Embedding batch {batch_num}/{total_batches}...")
                
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(
                        batch, self.embeddings, persist_directory="./chroma_db"
                    )
                else:
                    self.vectorstore.add_documents(batch)
                
                self.processing_progress = min(i + BATCH_SIZE, total_docs)
            
            print("Creating RAG chain...")
            self.status_message = "Finalizing..."
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
            self.processing = False
            self.status_message = "Ready!"
            print("Processing complete!")
            
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"ERROR: {err}")
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
                agent = create_pandas_dataframe_agent(
                    self.llm, self.df, verbose=True, allow_dangerous_code=True,
                    agent_type="tool-calling"
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

        prompt = f"""You are a query router. Given a user question about a dataset, decide the best approach:

STRUCTURED: Use for questions that need exact data operations like:
- Counts, totals, averages, sums (e.g., "how many", "total number")
- Filtering by specific values (e.g., "ticket number 190", "where status is open")
- Grouping/aggregations (e.g., "breakdown by category", "most common")
- Sorting/ranking (e.g., "top 10", "highest rated")
- Lookups by ID or specific field value

SEMANTIC: Use for questions that need meaning-based search like:
- Finding similar content (e.g., "reviews about battery issues")
- Understanding sentiment or themes (e.g., "what are customers complaining about")
- Open-ended exploration (e.g., "tell me about shipping problems")

Dataset columns: {columns_info}

Question: {query}

Reply with exactly one word: STRUCTURED or SEMANTIC"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            route = "structured" if "STRUCTURED" in answer else "semantic"
            print(f"Router decision: {route} (LLM said: {answer})")
            return route
        except Exception as e:
            print(f"Router error, defaulting to semantic: {e}")
            return "semantic"


engine = QueryEngine()

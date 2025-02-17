import os
from typing import List, Dict, Optional
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever

from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    relevant_chunks: List[str] = Field(default_factory=list)
    generated_queries: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None

class QuestionRequest(BaseModel):
    question: str

query_generation_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an expert at generating alternative search queries based on a user's original question. Generate diverse search queries that could help find the most relevant information.

Original question: {question}

Generate 2 alternative queries that:
1. Rephrase the original question using different words
2. Break down complex questions into simpler ones
3. Add relevant context or synonyms

Format your response as a list, one query per line. Keep each query concise.

Alternative queries:"""
)

class RAGApplication:
    def __init__(
        self, 
        docs_dir: str = "docs", 
        embedding_model: str = "textembedding-gecko@003",
        llm_model: str = "gemini-1.5-flash",
        collection_name: str = "effigo_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.docs_dir = docs_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = VertexAIEmbeddings(model_name=embedding_model)
        self.llm = ChatVertexAI(
            model=llm_model,
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40
        )
        self.query_llm = ChatVertexAI(
            model=llm_model,
            temperature=0.4,
            max_output_tokens=1024
        )
        
        self.qa_prompt = PromptTemplate(
            template="""You are an expert assistant tasked with providing precise and accurate answers based on the given context.

            Context: 
            {context}

            Guidelines:
            - Answer ONLY based on the provided context
            - If the information in the context is insufficient or unclear, explicitly state that
            - If the answer requires making assumptions, clearly state those assumptions
            - Maintain a neutral, professional tone
            - Format the response for clarity and readability
            - If there are contradictions in the sources, acknowledge them
            - Provide concrete examples from the context when relevant
            - **Do not include any reference-like phrases (e.g., "(See X.Y.X)","Section X.Y.Z") in your response.**

            Question: {question}

            Response (be direct, clear, and specific):""",
            input_variables=["context", "question"]
        )
        
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        logger.info("Initializing Qdrant vector database...")
        
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Dimension for textembedding-gecko@003
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,
                    )
                )
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
            
        self.vectordb = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        
        base_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': 6,
                'fetch_k': 20,
                'lambda_mult': 0.7
            }
        )
        
        # Initialize MultiQueryRetriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.query_llm,
            prompt=query_generation_prompt,
            parser_key="alternative queries"
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )

        # Check if we need to ingest documents
        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            self._ingest_documents()
        else:
            logger.info(f"Collection already contains {collection_info.points_count} points.")

    def _ingest_documents(self):
        logger.info(f"Loading documents from {self.docs_dir}")
        
        try:
            loader = DirectoryLoader(
                self.docs_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"No documents found in {self.docs_dir}")

            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                raise ValueError("No text chunks generated from documents")

            # Add metadata
            for text in texts:
                text.metadata["ingested_at"] = datetime.now().isoformat()

            logger.info(f"Adding {len(texts)} chunks to vector database...")
            self.vectordb.add_documents(texts)
            logger.info("Document ingestion complete.")
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            raise

    def generate_alternative_queries(self, question: str) -> List[str]:
        """Generate alternative queries using the LLM directly."""
        try:
            # Format the prompt
            prompt = f"""You are an expert at generating alternative search queries. 
            For the question: "{question}"
            
            Generate 1-2 alternative ways to ask this question that:
            1. Rephrase using different words
            2. Make the question more specific
            3. Break down complex aspects
            4. Include relevant synonyms or related concepts
            
            Return only the alternative queries, one per line."""

            # Get response from LLM
            response = self.query_llm.invoke(prompt)
            
            # Process the response into a list of queries
            queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            queries = [q.strip('- ') for q in queries]
            
            # Add original question and return unique queries
            all_queries = [question] + queries
            return list(dict.fromkeys(all_queries))
            
        except Exception as e:
            logger.warning(f"Error generating alternative queries: {str(e)}")
            return [question]

    def query(self, question: str) -> Dict:
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate alternative queries
            generated_queries = self.generate_alternative_queries(question)
            logger.info(f"Generated queries: {generated_queries}")
            
            all_docs = []
            
            # Query for each generated query
            for query in generated_queries:
                result = self.qa_chain.invoke({"query": query})
                all_docs.extend(result["source_documents"])
            
            # Remove duplicate documents
            seen_contents = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
            
            # Use the unique documents for the final answer
            result = self.qa_chain.invoke({
                "query": question,
                "source_documents": unique_docs[:6]
            })
            
            relevant_chunks = []
            for doc in unique_docs[:6]:
                chunk_text = doc.page_content
                source = os.path.basename(doc.metadata["source"])
                page = doc.metadata.get("page", "N/A")
                relevant_chunks.append(f"Source: {source} (Page {page})\nContent: {chunk_text}\n")
            
            # Get unique sources
            sources = []
            for doc in unique_docs[:6]:
                source = os.path.basename(doc.metadata["source"])
                page = doc.metadata.get("page", "N/A")
                sources.append(f"{source} (Page {page})")
            
            logger.info("Query processed successfully.")
            return {
                "answer": result["result"],
                "sources": list(set(sources)),
                "relevant_chunks": relevant_chunks,
                "generated_queries": generated_queries
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
# FastAPI App
app = FastAPI(
    title="RAG Document Query Service",
    description="A RAG system using Google Vertex AI for document querying",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
rag_app = RAGApplication()

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QuestionRequest):
    try:
        result = rag_app.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "collection_name": rag_app.collection_name,
        "document_count": rag_app.client.get_collection(rag_app.collection_name).points_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
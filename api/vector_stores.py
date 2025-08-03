
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.stores import InMemoryStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoresManager():
    def __init__(self, MODE:str, 
                loaded_data, 
                chunk_config = {"enabled": True, "params": {"chunk_size": 1000, "chunk_overlap": 250}},
                embeddings_model_name = "text-embedding-3-small", 
                chat_model = "gpt-4.1-nano",
                collection_name = "Rag Loaded Data Baseline"):

        self.loaded_data = loaded_data
        self.MODE = MODE
        self.chunk_config = chunk_config
        self.embeddings_model_name = embeddings_model_name
        self.embeddings_model = OpenAIEmbeddings(model=self.embeddings_model_name)
        self.chat_model = ChatOpenAI(model=chat_model)
        self.vectorstore = None
        self.parent_document_vectorstore = None
        self.in_memory_store = None
        self.child_splitter = None

        if MODE == "baseline":
            # ===============================
            # Baseline Retrieval
            # ===============================
            if self.chunk_config["enabled"]:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = self.chunk_config["params"]["chunk_size"],
                    chunk_overlap = self.chunk_config["params"]["chunk_overlap"]
                )

                self.loan_data_chunks = text_splitter.split_documents(self.loaded_data)
            else:
                self.loan_data_chunks = self.loaded_data

            self.vectorstore = Qdrant.from_documents(
                self.loan_data_chunks,
                self.embeddings_model,
                location=":memory:",
                collection_name="Rag Loaded Data Baseline"
            )
            
        elif MODE == "semantic":
            # ===============================
            # Semantic Retrieval
            # ===============================

            semantic_chunker = SemanticChunker(
                self.embeddings_model,
                breakpoint_threshold_type="percentile"
            )
            semantic_data = semantic_chunker.split_documents(loaded_data)
            
            semantic_vectorstore = Qdrant.from_documents(
                semantic_data,
                self.embeddings_model,
                location=":memory:",
                collection_name="Rag Loaded Data Semantic"
            )
            self.vectorstore = semantic_vectorstore
        else:
            raise ValueError(f"Invalid mode: {MODE}")

        # Create the retriever - parent document retrieval
        self.parent_docs = self.loaded_data
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=750) # TODO: make this dynamic

        self.client_qdrant = QdrantClient(location=":memory:")

        self.client_qdrant.create_collection(
            collection_name="full_documents",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

        self.parent_document_vectorstore = QdrantVectorStore(
            collection_name="full_documents", 
            embedding=self.embeddings_model, 
            client=self.client_qdrant
        )

        self.in_memory_store = InMemoryStore()


    def get_base_vectorstore(self):
        return self.vectorstore

    def get_parent_document_vectorstore(self):
        return self.parent_document_vectorstore

    def get_in_memory_store(self):
        return self.in_memory_store

    def get_child_splitter(self):
        return self.child_splitter

    def get_chunked_loaded_data(self):
        return self.loan_data_chunks

    def get_loaded_data(self):
        return self.loaded_data
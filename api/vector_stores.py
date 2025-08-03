
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
                embeddings_model = "text-embedding-3-small", 
                chat_model = "gpt-4.1-nano",
                collection_name = "Rag Loaded Data Baseline"):

        self.loaded_data = loaded_data
        self.MODE = MODE
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.chat_model = ChatOpenAI(model=chat_model)
        self.vectorstore = None
        self.parent_document_vectorstore = None
        self.in_memory_store = None
        self.child_splitter = None

        if MODE == "baseline":
            # ===============================
            # Naive Retrieval
            # ===============================

            self.vectorstore = Qdrant.from_documents(
                self.loaded_data,
                self.embeddings,
                location=":memory:",
                collection_name="Rag Loaded Data Baseline"
            )
            
        elif MODE == "semantic":
            # ===============================
            # Semantic Retrieval
            # ===============================

            semantic_chunker = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile"
            )
            loan_complaint_data = semantic_chunker.split_documents(loaded_data)
            semantic_vectorstore = Qdrant.from_documents(
                loan_complaint_data,
                self.embeddings,
                location=":memory:",
                collection_name="Rag Loaded Data Semantic"
            )
            self.vectorstore = semantic_vectorstore
        else:
            raise ValueError(f"Invalid mode: {MODE}")

        # Create the retriever - parent document retrieval
        self.parent_docs = loan_complaint_data
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

        self.client_qdrant = QdrantClient(location=":memory:")
        self.client_qdrant.create_collection(
            collection_name="full_documents",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

        self.parent_document_vectorstore = QdrantVectorStore(
            collection_name="full_documents", 
            embedding=OpenAIEmbeddings(model=self.embeddings_model), 
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
    
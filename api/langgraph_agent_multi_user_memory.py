from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.runnables import RunnableConfig
from langsmith.utils import traceable
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import BaseMessage


class LangGraphChatAgent:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embeddings_model: Any,
        llm_model: Any,
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model

        # Per-user LangChain memory
        self.memory_by_user: Dict[str, ConversationBufferMemory] = {}

        # Init vectorstore and graph
        self.vectorstore = self._init_vectorstore()
        self.graph = self._build_graph()

    def _init_vectorstore(self) -> Qdrant:
        client = QdrantClient(url=self.qdrant_url)
        return Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embeddings_model,
        )

    def _build_graph(self) -> Pregel:
        builder = StateGraph(name="chat-agent-graph")

        builder.add_node("retrieve", self._retrieve_docs)
        builder.add_node("generate", self._generate_response)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    def _get_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self.memory_by_user:
            self.memory_by_user[user_id] = ConversationBufferMemory(
                return_messages=True
            )
        return self.memory_by_user[user_id]

    @traceable(name="Document Retrieval")
    def _retrieve_docs(self, state: Dict, config: RunnableConfig) -> Dict:
        query = state["input"]
        user_id = config.get("configurable", {}).get("user_id", "anonymous")

        docs = self.vectorstore.similarity_search(query, k=4)
        return {
            "input": query,
            "docs": docs,
            "history": state.get("history", []),
            "user_id": user_id,
        }

    @traceable(name="LLM Generation")
    def _generate_response(self, state: Dict, config: RunnableConfig) -> Dict:
        query = state["input"]
        docs = state["docs"]
        history = state.get("history", [])
        user_id = config.get("configurable", {}).get("user_id", "anonymous")

        context = "\n".join(doc.page_content for doc in docs)

        history_str = "\n".join(
            [f"{msg.type.capitalize()}: {msg.content}" for msg in history]
        )

        prompt = f"""You are a helpful assistant. Here is the conversation so far:
{history_str}

Here is some relevant context:
{context}

Now answer the user's latest question:
User: {query}
Assistant:"""

        response = self.llm_model.invoke(prompt)

        return {
            "response": response,
            "user_id": user_id,
            "input": query,
            "output": response,
        }

    def chat(self, user_input: str, user_id: Optional[str] = "anonymous") -> str:
        # Get user-specific memory
        memory = self._get_memory(user_id)
        history_messages: list[BaseMessage] = memory.load_memory_variables({})["history"]

        # Build state and config
        state = {
            "input": user_input,
            "history": history_messages,
        }
        config: RunnableConfig = {"configurable": {"user_id": user_id}}

        # Run graph
        result = self.graph.invoke(state, config=config)
        response = result["response"]

        # Update memory
        memory.save_context({"input": user_input}, {"output": response})

        return response

'''
agent = LangGraphChatAgent(
    qdrant_url="http://localhost:6333",
    collection_name="my_docs",
    embeddings_model=OpenAIEmbeddings(),
    llm_model=ChatOpenAI(),
)

print(agent.chat("What's LangGraph?", user_id="alice"))
print(agent.chat("Tell me about Qdrant", user_id="bob"))
print(agent.chat("How do they work together?", user_id="alice"))


----

Local qdrant:

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Initialize Qdrant client with local path (folder will be created if not exists)
client = QdrantClient(path="qdrant_local_data")

# Define collection schema (dimension depends on your embedding model)
embedding_model = OpenAIEmbeddings()
vector_size = 1536  # 1536 for OpenAI Ada, change if using another model

collection_name = "my_local_collection"

# Create collection (recreate_collection is safe to run repeatedly)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

'''

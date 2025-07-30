from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.runnables import RunnableConfig
from langsmith.utils import traceable
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document


class LangGraphChatAgent:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embeddings_model: Any,
        llm_model: Any,
        user_id: Optional[str] = None,
    ):
        self.user_id = user_id or "default_user"
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model

        # Store simple in-memory conversation history
        self.chat_history: List[Dict[str, str]] = []

        # Initialize vector store and graph
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

    @traceable(name="Document Retrieval")
    def _retrieve_docs(self, state: Dict) -> Dict:
        query = state["input"]
        docs = self.vectorstore.similarity_search(query, k=4)
        return {"input": query, "docs": docs, "history": state.get("history", [])}

    @traceable(name="LLM Generation")
    def _generate_response(self, state: Dict) -> Dict:
        query = state["input"]
        docs = state["docs"]
        history = state.get("history", [])

        context = "\n".join(doc.page_content for doc in docs)

        # Include memory in the prompt
        history_str = "\n".join(
            [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history]
        )

        prompt = f"""You are a helpful assistant. Here is the conversation so far:
                    {history_str}

                    Here is some relevant context:
                    {context}

                    Now answer the user's latest question:
                    User: {query}
                    Assistant:"""

        response = self.llm_model.invoke(prompt)
        return {"response": response, "history": history}

    def chat(self, user_input: str) -> str:
        state = {
            "input": user_input,
            "history": self.chat_history.copy(),
        }

        config: RunnableConfig = {
            "configurable": {"user_id": self.user_id}
        }

        result = self.graph.invoke(state, config=config)
        response = result["response"]

        # Save to history
        self.chat_history.append({"user": user_input, "assistant": response})
        return response

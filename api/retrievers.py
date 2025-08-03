
from operator import itemgetter
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langsmith import traceable


def get_retrieval_chains_and_wrappers_for_evals(retrievers_config, loaded_data, rag_prompt, chat_model, MODE):

    vectorstore = retrievers_config["base"]["vectorstore"]
    parent_document_vectorstore = retrievers_config["parent_document"]["vectorstore"]
    in_memory_store = retrievers_config["parent_document"]["in_memory_store"]
    child_splitter = retrievers_config["parent_document"]["child_splitter"]

    # Create the retriever - base retrieval
    base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

    base_retrieval_chain = (
        {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # ===============================
    # BM25 Retrieval
    # ===============================
    # Create the retriever - BM25 retrieval
    bm25_retriever = BM25Retriever.from_documents(loaded_data, )

    bm25_retrieval_chain = (
        {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # ===============================
    # Contextual Compression Retrieval
    # ===============================
    # Create the retriever - contextual compression retrieval
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    contextual_compression_retrieval_chain = (
        {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # ===============================
    # Multi-Query Retrieval
    # ===============================
    # Create the retriever - multi-query retrieval
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=chat_model
    )

    multi_query_retrieval_chain = (
        {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # Example of invoking the multi-query retrieval chain:
    # ===============================
    # Parent Document Retrieval
    # ===============================

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore = parent_document_vectorstore,
        docstore=in_memory_store,
        child_splitter=child_splitter,
    )

    parent_document_retriever.add_documents(loaded_data, ids=None)

    parent_document_retrieval_chain = (
        {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # ===============================
    # Ensemble Retrieval
    # ===============================
    # Create the retriever - ensemble retrieval
    retriever_list = [bm25_retriever, base_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
    equal_weighting = [1/len(retriever_list)] * len(retriever_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=equal_weighting
    )

    ensemble_retrieval_chain = (
        {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

    # ===============================
    # Create the chains and wrapers
    # ===============================
    chains = {"base_retrieval_chain": base_retrieval_chain,
              "bm25_retrieval_chain": bm25_retrieval_chain,
              "contextual_compression_retrieval_chain": contextual_compression_retrieval_chain,
              "multi_query_retrieval_chain": multi_query_retrieval_chain,
              "parent_document_retrieval_chain": parent_document_retrieval_chain,
              "ensemble_retrieval_chain": ensemble_retrieval_chain}

    @traceable(name=f"RAG base_retrieval_chain - {MODE}")
    def run_base_retrieval_chain(question):
        return base_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG bm25_retrieval_chain - {MODE}")
    def run_bm25_retrieval_chain(question):
        return bm25_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG contextual_compression_retrieval_chain - {MODE}")
    def run_contextual_compression_retrieval_chain(question):
        return contextual_compression_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG multi_query_retrieval_chain - {MODE}")
    def run_multi_query_retrieval_chain(question):
        return multi_query_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG parent_document_retrieval_chain - {MODE}")
    def run_parent_document_retrieval_chain(question):
        return parent_document_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG ensemble_retrieval_chain - {MODE}")
    def run_ensemble_retrieval_chain(question):
        return ensemble_retrieval_chain.invoke({"question": question})

    wrappers = {"base_retrieval_chain": run_base_retrieval_chain,
              "bm25_retrieval_chain": run_bm25_retrieval_chain,
              "contextual_compression_retrieval_chain": run_contextual_compression_retrieval_chain,
              "multi_query_retrieval_chain": run_multi_query_retrieval_chain,
              "parent_document_retrieval_chain": run_parent_document_retrieval_chain,
              "ensemble_retrieval_chain": run_ensemble_retrieval_chain}

    return chains, wrappers  


def get_retrieval_chains_and_wrappers(retrievers_config, loaded_data, chat_model, MODE):

    vectorstore = retrievers_config["base"]["vectorstore"]
    parent_document_vectorstore = retrievers_config["parent_document"]["vectorstore"]
    in_memory_store = retrievers_config["parent_document"]["in_memory_store"]
    child_splitter = retrievers_config["parent_document"]["child_splitter"]

    # Create the retriever - base retrieval
    base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

    base_retrieval_chain = (
        {"context": itemgetter("question") | base_retriever}
    )

    # ===============================
    # BM25 Retrieval
    # ===============================
    # Create the retriever - BM25 retrieval
    bm25_retriever = BM25Retriever.from_documents(loaded_data, )

    bm25_retrieval_chain = (
        {"context": itemgetter("question") | bm25_retriever}
    )

    # ===============================
    # Contextual Compression Retrieval
    # ===============================
    # Create the retriever - contextual compression retrieval
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    contextual_compression_retrieval_chain = (
        {"context": itemgetter("question") | compression_retriever}
    )

    # ===============================
    # Multi-Query Retrieval
    # ===============================
    # Create the retriever - multi-query retrieval
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=chat_model
    )

    multi_query_retrieval_chain = (
        {"context": itemgetter("question") | multi_query_retriever}
    )

    # Example of invoking the multi-query retrieval chain:
    # ===============================
    # Parent Document Retrieval
    # ===============================

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore = parent_document_vectorstore,
        docstore=in_memory_store,
        child_splitter=child_splitter,
    )

    parent_document_retriever.add_documents(loaded_data, ids=None)

    parent_document_retrieval_chain = (
        {"context": itemgetter("question") | parent_document_retriever}
    )

    # ===============================
    # Ensemble Retrieval
    # ===============================
    # Create the retriever - ensemble retrieval
    retriever_list = [bm25_retriever, base_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
    equal_weighting = [1/len(retriever_list)] * len(retriever_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=equal_weighting
    )

    ensemble_retrieval_chain = (
        {"context": itemgetter("question") | ensemble_retriever}
    )

    # ===============================
    # Create the chains and wrapers
    # ===============================
    chains = {"base_retrieval_chain": base_retrieval_chain,
              "bm25_retrieval_chain": bm25_retrieval_chain,
              "contextual_compression_retrieval_chain": contextual_compression_retrieval_chain,
              "multi_query_retrieval_chain": multi_query_retrieval_chain,
              "parent_document_retrieval_chain": parent_document_retrieval_chain,
              "ensemble_retrieval_chain": ensemble_retrieval_chain}

    @traceable(name=f"RAG base_retrieval_chain prod - {MODE}")
    def run_base_retrieval_chain(question):
        return base_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG bm25_retrieval_chain prod - {MODE}")
    def run_bm25_retrieval_chain(question):
        return bm25_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG contextual_compression_retrieval_chain prod - {MODE}")
    def run_contextual_compression_retrieval_chain(question):
        return contextual_compression_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG multi_query_retrieval_chain prod - {MODE}")
    def run_multi_query_retrieval_chain(question):
        return multi_query_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG parent_document_retrieval_chain prod - {MODE}")
    def run_parent_document_retrieval_chain(question):
        return parent_document_retrieval_chain.invoke({"question": question})

    @traceable(name=f"RAG ensemble_retrieval_chain prod - {MODE}")
    def run_ensemble_retrieval_chain(question):
        return ensemble_retrieval_chain.invoke({"question": question})

    wrappers = {"base_retrieval_chain": run_base_retrieval_chain,
              "bm25_retrieval_chain": run_bm25_retrieval_chain,
              "contextual_compression_retrieval_chain": run_contextual_compression_retrieval_chain,
              "multi_query_retrieval_chain": run_multi_query_retrieval_chain,
              "parent_document_retrieval_chain": run_parent_document_retrieval_chain,
              "ensemble_retrieval_chain": run_ensemble_retrieval_chain}

    return chains, wrappers  

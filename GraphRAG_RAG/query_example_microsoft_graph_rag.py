# query_example.py
import asyncio
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

async def main():
    # Setup LLM
    llm = ChatOpenAI(
        api_key="your_openai_api_key",
        model="gpt-4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    # Load indexed data
    entities = read_indexer_entities("./output/artifacts", entity_table="create_final_entities", entity_key_col="title")
    relationships = read_indexer_relationships("./output/artifacts")
    reports = read_indexer_reports("./output/artifacts", community_table="create_final_communities")
    text_units = read_indexer_text_units("./output/artifacts")

    # Setup vector store
    vector_store = LanceDBVectorStore(collection_name="default-entity-description")
    vector_store.connect(db_uri="./output/lancedb")

    # Setup search engine
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=vector_store,
        embedding_vectorstore_key=EntityVectorStoreKey.TITLE,
        text_embedder=None,  # We'll use the default
    )

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=None,  # We'll use the default
        response_type="multiple paragraphs",
    )

    # Perform search
    result = await search_engine.asearch("What are the main topics discussed in the documents?")
    print(result.response)

if __name__ == "__main__":
    asyncio.run(main())



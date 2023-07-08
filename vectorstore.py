from typing import Any, List

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import Client


class CustomSupabaseVectorStore(SupabaseVectorStore):
    def __init__(
        self,
        client: Client,
        embedding: OpenAIEmbeddings,
        table_name: str,
    ):
        super().__init__(client, embedding, table_name)

    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Document]:
        vectors = self._embedding.embed_documents([query])
        query_embedding = vectors[0]

        res = self._client.rpc(
            self.query_name,
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter": {"blockId": "k3u46gu4bg"},
            },
        ).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        documents = [doc for doc, _ in match_result]

        return documents

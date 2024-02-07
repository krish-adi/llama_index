"""DuckDB vector store."""

import logging
import math
import json
from typing import Any, Dict, Generator, List, Optional, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.utils import truncate_text
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
import_err_msg = "`duckdb` package not found, please run `pip install duckdb`"


class DuckDBVectorStore(BasePydanticVectorStore):
    """DuckDB vector store.

    In this vector store, embeddings are stored within a DuckDB database.

    During query time, the index uses DuckDB to query for the top
    k most similar nodes.

    """

    stores_text: bool = True
    flat_metadata: bool = True

    database_name: Optional[str]
    table_name: Optional[str]
    schema_name: Optional[str]
    embed_dim: Optional[int]
    hybrid_search: Optional[bool]
    text_search_config: Optional[dict]
    persist_dir: Optional[str]

    _conn: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        database_name: Optional[str] = ":memory:",
        table_name: Optional[str] = "documents",
        schema_name: Optional[str] = "main",
        embed_dim: Optional[int] = 1536,
        hybrid_search: Optional[bool] = True,
        # https://duckdb.org/docs/extensions/full_text_search
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": "(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": False,
        },
        persist_dir: Optional[str] = "./storage",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        self._conn = duckdb.connect(database_name)
        self._conn.install_extension("json")
        self._conn.load_extension("json")
        self._conn.install_extension("fts")
        self._conn.load_extension("fts")

        self._is_initialized = False

        super().__init__(
            database_name=database_name,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            persist_dir=persist_dir,
        )

    @classmethod
    def from_local(cls, database: str) -> "DuckDBVectorStore":
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        # TODO: load a local database based on it's path and persist dir.
        # extract persist directory from database str
        # extract name of the database from the database str

        return cls(database_name=database)

    @classmethod
    def from_params(
        cls,
        database_name: Optional[str] = ":memory:",
        table_name: Optional[str] = "documents",
        schema_name: Optional[str] = "main",
        embed_dim: Optional[int] = 1536,
        hybrid_search: Optional[bool] = True,
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": "(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": False,
        },
        persist_dir: Optional[str] = "./storage",
        **kwargs: Any,
    ) -> "DuckDBVectorStore":
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        return cls(
            database_name=database_name,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            persist_dir=persist_dir,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DuckDBVectorStore"

    @property
    def client(self) -> Any:
        """Return client."""
        return self._conn

    def _initialize(self) -> None:
        if not self._conn:
            raise ValueError("DuckDB connection not initialized!")
        if not self._is_initialized:
            # TODO: check if the table, schema exists
            # if not, create the table, schema
            self._conn.execute(
                f"""
                CREATE TABLE {self.table_name} (
                    node_id VARCHAR,
                    text TEXT,
                    embedding FLOAT[{self.embed_dim}],
                    metadata_ JSON
                    );
                """
            )
            self._is_initialized = True

    def _node_to_table_row(self, node: BaseNode) -> Any:
        return (
            node.node_id,
            node.get_content(metadata_mode=MetadataMode.NONE),
            node.get_embedding(),
            node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """

        self._initialize()

        ids = []
        _table = self._conn.table(self.table_name)
        for node in nodes:
            ids.append(node.node_id)
            _row = self._node_to_table_row(node)
            _table.insert(_row)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._conn.execute(
            f"""
            DELETE FROM {self.table_name}
            WHERE metadata_->>'ref_doc_id' = '{ref_doc_id}';
            """
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        # query.query_embedding,
        # query.similarity_top_k,
        # query.filters,
        if query.filters is not None:
            # Implement metadata filter queries
            pass
        else:
            where = kwargs.pop("where", {})

        # TODO: results from the metadata filter query
        # filtered_set = []

        nodes = []
        similarities = []
        ids = []
        _final_results = self._conn.execute(
            f"""
            SELECT node_id, text, embedding, metadata_, score
            FROM (
                SELECT *, list_cosine_similarity(embedding, {query.query_embedding}) AS score
                FROM {self.table_name}
            ) sq
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT {query.similarity_top_k};
            """
        ).fetchall()

        for _row in _final_results:
            node = TextNode(
                id_=_row[0],
                text=_row[1],
                embedding=_row[2],
                metadata=json.loads(_row[3]),
            )
            nodes.append(node)
            similarities.append(_row[4])
            ids.append(_row[0])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

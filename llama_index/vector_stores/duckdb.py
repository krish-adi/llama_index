"""DuckDB vector store."""

import logging
import math
import json
from typing import Any, Dict, Generator, List, Optional, cast
import os
from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.utils import truncate_text
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict

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
    # schema_name: Optional[str] # TODO: support schema
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
        # schema_name: Optional[str] = "main",
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

        self._is_initialized = False

        if database_name == ":memory:":
            self._conn = duckdb.connect(database_name)
            self._conn.install_extension("json")
            self._conn.load_extension("json")
            self._conn.install_extension("fts")
            self._conn.load_extension("fts")
        else:
            # check if persist dir exists
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)

            with duckdb.connect(os.path.join(persist_dir, database_name)) as _conn:
                _conn.install_extension("json")
                _conn.load_extension("json")
                _conn.install_extension("fts")
                _conn.load_extension("fts")

            self._conn = None

        super().__init__(
            database_name=database_name,
            table_name=table_name,
            # schema_name=schema_name,
            embed_dim=embed_dim,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            persist_dir=persist_dir,
        )

    @classmethod
    def from_local(
        cls, database_path: str, table_name: str = "documents"
    ) -> "DuckDBVectorStore":
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        # TODO: load a local database based on it's path and persist dir.
        # extract persist directory from database str
        # extract name of the database from the database str

        # check if database_path exists and is a path
        if not os.path.exists(database_path):
            raise ValueError(f"Database path {database_path} does not exist.")

        if not os.path.isfile(database_path):
            raise ValueError(f"Database path {database_path} is not a valid file.")

        # check if table_name exists in the database
        with duckdb.connect(database_path) as _conn:
            _conn.install_extension("json")
            _conn.load_extension("json")
            _conn.install_extension("fts")
            _conn.load_extension("fts")
            try:
                _table_info = _conn.execute(f"SHOW {table_name};").fetchall()
            except Exception as e:
                raise ValueError(f"Index table {table_name} not found in the database.")

            _std = {
                "text": "VARCHAR",
                "node_id": "VARCHAR",
                "embedding": "FLOAT[]",
                "metadata_": "JSON",
            }
            _ti = {_i[0]: _i[1] for _i in _table_info}
            if _std != _ti:
                raise ValueError(
                    f"Index table {table_name} does not have the correct schema."
                )

        _cls = cls(
            database_name=os.path.basename(database_path),
            table_name=table_name,
            persist_dir=os.path.dirname(database_path),
        )
        _cls._is_initialized = True

        return _cls

    @classmethod
    def from_params(
        cls,
        database_name: Optional[str] = ":memory:",
        table_name: Optional[str] = "documents",
        # schema_name: Optional[str] = "main",
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
            # schema_name=schema_name,
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
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        if not self._is_initialized:
            # TODO: schema.table also.
            # Check if table and type is present
            # if not, create table
            if self.database_name == ":memory:":
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
            else:
                with duckdb.connect(
                    os.path.join(self.persist_dir, self.database_name)
                ) as _conn:
                    _conn.install_extension("json")
                    _conn.load_extension("json")
                    _conn.install_extension("fts")
                    _conn.load_extension("fts")
                    _conn.execute(
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

        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        self._initialize()

        ids = []

        if self.database_name == ":memory:":
            _table = self._conn.table(self.table_name)
            for node in nodes:
                ids.append(node.node_id)
                _row = self._node_to_table_row(node)
                _table.insert(_row)
        else:
            with duckdb.connect(
                os.path.join(self.persist_dir, self.database_name)
            ) as _conn:
                _conn.install_extension("json")
                _conn.load_extension("json")
                _conn.install_extension("fts")
                _conn.load_extension("fts")
                _table = _conn.table(self.table_name)
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
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        _ddb_query = f"""
            DELETE FROM {self.table_name}
            WHERE json_extract_string(metadata_, '$.ref_doc_id') = '{ref_doc_id}';
            """
        if self.database_name == ":memory:":
            self._conn.execute(_ddb_query)
        else:
            with duckdb.connect(
                os.path.join(self.persist_dir, self.database_name)
            ) as _conn:
                _conn.execute(_ddb_query)

    @staticmethod
    def _build_metadata_filter_condition(
        standard_filters: MetadataFilters,
    ) -> dict:
        """Translate standard metadata filters to DuckDB SQL specification."""

        filters_list = []
        # condition = standard_filters.condition or "and"  ## and/or as strings.
        condition = "AND"
        _filters_condition_list = []

        for filter in standard_filters.filters:
            if filter.operator:
                if filter.operator in [
                    "<",
                    ">",
                    "<=",
                    ">=",
                    "<>",
                    "!=",
                ]:
                    filters_list.append((filter.key, filter.operator, filter.value))
                elif filter.operator in ["=="]:
                    filters_list.append((filter.key, "=", filter.value))
                else:
                    raise ValueError(
                        f"Filter operator {filter.operator} not supported."
                    )
            else:
                filters_list.append((filter.key, "=", filter.value))

        for _fc in filters_list:
            if isinstance(_fc[2], str):
                _filters_condition_list.append(
                    f"json_extract_string(metadata_, '$.{_fc[0]}') {_fc[1]} '{_fc[2]}'"
                )
            else:
                _filters_condition_list.append(
                    f"json_extract(metadata_, '$.{_fc[0]}') {_fc[1]} {_fc[2]}"
                )

        return f" {condition} ".join(_filters_condition_list)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query.query_embedding (List[float]): query embedding
            query.similarity_top_k (int): top k most similar nodes

        """

        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        nodes = []
        similarities = []
        ids = []

        if query.filters is not None:
            # TODO: results from the metadata filter query
            _filter_string = self._build_metadata_filter_condition(query.filters)
            _ddb_query = f"""
            SELECT node_id, text, embedding, metadata_, score
            FROM (
                SELECT *, list_cosine_similarity(embedding, {query.query_embedding}) AS score
                FROM {self.table_name}
                WHERE {_filter_string}
            ) sq            
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT {query.similarity_top_k};
            """
        else:
            _ddb_query = f"""
            SELECT node_id, text, embedding, metadata_, score
            FROM (
                SELECT *, list_cosine_similarity(embedding, {query.query_embedding}) AS score
                FROM {self.table_name}
            ) sq
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT {query.similarity_top_k};
            """

        if self.database_name == ":memory:":
            _final_results = self._conn.execute(_ddb_query).fetchall()
        else:
            with duckdb.connect(
                os.path.join(self.persist_dir, self.database_name)
            ) as _conn:
                _conn.install_extension("json")
                _conn.load_extension("json")
                _conn.install_extension("fts")
                _conn.load_extension("fts")
                _final_results = _conn.execute(_ddb_query).fetchall()

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

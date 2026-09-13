"""pgvector backend.

The interesting one, because it can be forced exact: drop the HNSW index and
let Postgres sequential-scan. That gives an exact reference on realistic
infrastructure rather than only in memory, which matters for the crossover
argument — a reviewer can object that in-process brute force is not a fair
comparison against a deployed database, and this backend answers that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from .base import BaseIndex, require_module


class PgVectorIndex(BaseIndex):
    """pgvector backend. Set `use_index=False` for an exact sequential scan."""

    def __init__(
        self,
        dsn: str,
        table: str = "evidentia_chunks",
        use_index: bool = True,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 40,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        require_module("psycopg", "pgvector")
        super().__init__(seed=seed)
        self.dsn = dsn
        self.table = table
        self.use_index = use_index
        self.exact = not use_index
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.score_precision = score_precision

    @property
    def spec(self) -> str:
        if not self.use_index:
            return "pgvector/v1(seqscan,exact=true)"
        return (
            f"pgvector/v1(hnsw,m={self.m},efC={self.ef_construction},"
            f"efS={self.ef_search},seed={self.seed})"
        )

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        matrix = self._register(vectors, ids)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"DROP TABLE IF EXISTS {self.table}")
            cursor.execute(
                f"CREATE TABLE {self.table} "
                f"(chunk_id text PRIMARY KEY, embedding vector({self.dimension}))"
            )
            cursor.executemany(
                f"INSERT INTO {self.table} (chunk_id, embedding) VALUES (%s, %s)",
                [(chunk_id, matrix[i].tolist()) for i, chunk_id in enumerate(self.ids)],
            )
            if self.use_index:
                cursor.execute(
                    f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {self.m}, ef_construction = {self.ef_construction})"
                )
            connection.commit()

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        vector = np.asarray(query, dtype=np.float64).reshape(-1).tolist()
        with self._connect() as connection, connection.cursor() as cursor:
            if self.use_index:
                cursor.execute(f"SET hnsw.ef_search = {self.ef_search}")
            else:
                cursor.execute("SET enable_indexscan = off")
            cursor.execute(
                f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, k),
            )
            rows = cursor.fetchall()
        pairs = [(row[0], round(float(row[1]), self.score_precision)) for row in rows]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs

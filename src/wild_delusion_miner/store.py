from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from wild_delusion_miner.records import UserMessageRecord


class DedupeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=30000000000")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
              message_hash TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              text_len INTEGER NOT NULL,
              first_source TEXT NOT NULL,
              first_split TEXT NOT NULL,
              first_row_offset INTEGER NOT NULL,
              first_conversation_id TEXT NOT NULL,
              first_message_index INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS refs (
              message_hash TEXT NOT NULL,
              source TEXT NOT NULL,
              split TEXT NOT NULL,
              row_offset INTEGER NOT NULL,
              conversation_id TEXT NOT NULL,
              message_index INTEGER NOT NULL,
              PRIMARY KEY (message_hash, source, split, row_offset, message_index)
            );
            CREATE INDEX IF NOT EXISTS idx_refs_hash ON refs(message_hash);
            CREATE INDEX IF NOT EXISTS idx_refs_conversation
              ON refs(source, split, conversation_id, row_offset);
            CREATE INDEX IF NOT EXISTS idx_refs_source_split_offset
              ON refs(source, split, row_offset);
            """
        )
        self.conn.commit()

    def add_many(self, records: Iterable[UserMessageRecord]) -> tuple[int, int]:
        rows = list(records)
        if not rows:
            return 0, 0
        before = self.message_count()
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO messages (
                  message_hash, text, text_len, first_source, first_split,
                  first_row_offset, first_conversation_id, first_message_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.message_hash,
                        row.text,
                        len(row.text),
                        row.source,
                        row.split,
                        row.row_offset,
                        row.conversation_id,
                        row.message_index,
                    )
                    for row in rows
                ],
            )
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO refs (
                  message_hash, source, split, row_offset, conversation_id, message_index
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.message_hash,
                        row.source,
                        row.split,
                        row.row_offset,
                        row.conversation_id,
                        row.message_index,
                    )
                    for row in rows
                ],
            )
        return self.message_count() - before, len(rows)

    def message_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

    def max_row_offset(self, *, source: str, split: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(row_offset) FROM refs WHERE source = ? AND split = ?",
            (source, split),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def iter_messages(
        self,
        *,
        batch_size: int = 10_000,
        start_after_rowid: int = 0,
    ) -> Iterator[list[tuple[int, str, str]]]:
        last_rowid = start_after_rowid
        while True:
            rows = self.conn.execute(
                """
                SELECT rowid, message_hash, text
                FROM messages
                WHERE rowid > ?
                ORDER BY rowid
                LIMIT ?
                """,
                (last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            last_rowid = int(rows[-1][0])
            yield [(int(rowid), str(message_hash), str(text)) for rowid, message_hash, text in rows]

    def get_messages(self, hashes: Sequence[str]) -> dict[str, str]:
        if not hashes:
            return {}
        result: dict[str, str] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT message_hash, text FROM messages WHERE message_hash IN ({placeholders})",
                list(batch),
            ).fetchall()
            result.update({str(row[0]): str(row[1]) for row in rows})
        return result

    def get_first_refs(self, hashes: Sequence[str]) -> dict[str, tuple[str, str, int, str, int]]:
        if not hashes:
            return {}
        result: dict[str, tuple[str, str, int, str, int]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"""
                SELECT message_hash, first_source, first_split, first_row_offset,
                       first_conversation_id, first_message_index
                FROM messages
                WHERE message_hash IN ({placeholders})
                """,
                list(batch),
            ).fetchall()
            for row in rows:
                result[str(row[0])] = (str(row[1]), str(row[2]), int(row[3]), str(row[4]), int(row[5]))
        return result

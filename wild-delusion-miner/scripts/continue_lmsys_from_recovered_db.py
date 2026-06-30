from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from wild_delusion_miner.config import DatasetSpec
from wild_delusion_miner.datasets import iter_user_messages, stream_conversations
from wild_delusion_miner.records import UserMessageRecord

DB_PATH = Path("data/dedupe.sqlite")
RECOVERED_TMP_PATH = Path("data/dedupe.recovered.sqlite.tmp")
FLUSH_EVERY = 5_000


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}", flush=True)


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=30000000000")
    return conn


def promote_recovered_db() -> None:
    if not RECOVERED_TMP_PATH.exists():
        log(f"no recovered temp DB to promote at {RECOVERED_TMP_PATH}")
        return
    backup_path = DB_PATH.with_name(f"dedupe.corrupt_{int(time.time())}.sqlite")
    DB_PATH.replace(backup_path)
    RECOVERED_TMP_PATH.replace(DB_PATH)
    for suffix in ("-wal", "-shm"):
        Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
    log(f"promoted recovered DB; corrupt backup={backup_path}")


def last_lmsys_offset(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT first_source, first_row_offset
        FROM messages
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return -1
    source, row_offset = str(row[0]), int(row[1])
    if source != "lmsys_chat_1m":
        raise RuntimeError(f"last copied source is {source}, expected lmsys_chat_1m")
    return row_offset


def max_rowid(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT rowid FROM messages ORDER BY rowid DESC LIMIT 1").fetchone()
    return int(row[0]) if row else 0


def insert_records(conn: sqlite3.Connection, records: list[UserMessageRecord]) -> int:
    before = conn.total_changes
    with conn:
        conn.executemany(
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
                for row in records
            ],
        )
        after_messages = conn.total_changes
        conn.executemany(
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
                for row in records
            ],
        )
    return after_messages - before


def iter_lmsys_conversations(spec: DatasetSpec, *, start_offset: int):
    yield from tqdm(
        stream_conversations(spec, start_offset=start_offset),
        desc="extracting lmsys_chat_1m continuation",
        initial=start_offset,
    )


def continue_lmsys() -> None:
    conn = connect(DB_PATH)
    try:
        copied_offset = last_lmsys_offset(conn)
        start_offset = copied_offset + 1
        start_rowid = max_rowid(conn)
        log(f"continuing lmsys from row_offset={start_offset}; start_rowid={start_rowid}")
        spec = DatasetSpec(name="lmsys/lmsys-chat-1m", source="lmsys_chat_1m", split="train")
        buffer: list[UserMessageRecord] = []
        seen = 0
        added = 0
        conversations = iter_lmsys_conversations(spec, start_offset=start_offset)
        for conversation in conversations:
            seen += 1
            buffer.extend(iter_user_messages(conversation))
            if len(buffer) >= FLUSH_EVERY:
                new_count = insert_records(conn, buffer)
                added += new_count
                tqdm.write(
                    f"flush seen={seen} source_offset={conversation.row_offset} "
                    f"added_unique={added} last_rowid={start_rowid + added}"
                )
                buffer = []
        if buffer:
            new_count = insert_records(conn, buffer)
            added += new_count
            log(f"final flush added_unique={added} last_rowid={start_rowid + added}")
        log(f"lmsys continuation done; added_unique={added} final_rowid={max_rowid(conn)}")
    finally:
        conn.close()


def main() -> None:
    promote_recovered_db()
    continue_lmsys()


if __name__ == "__main__":
    main()

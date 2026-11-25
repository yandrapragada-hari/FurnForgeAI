import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.config import DB_PATH

class Database:
    """SQLite Database manager for FurnForge AI generations and gallery persistence."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    material TEXT NOT NULL,
                    color TEXT NOT NULL,
                    style TEXT NOT NULL,
                    finish TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    original_image_url TEXT NOT NULL,
                    preprocessed_edge_url TEXT NOT NULL,
                    result_image_url TEXT NOT NULL,
                    variations_json TEXT NOT NULL,
                    cnn_features_json TEXT NOT NULL,
                    pipeline_mode TEXT NOT NULL,
                    execution_time REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    model_3d_json TEXT
                )
            """)
            cols = {
                row["name"]
                for row in cursor.execute("PRAGMA table_info(generations)").fetchall()
            }
            if "model_3d_json" not in cols:
                cursor.execute("ALTER TABLE generations ADD COLUMN model_3d_json TEXT")
            conn.commit()

    def save_generation(self, data: Dict[str, Any]) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO generations (
                    id, category, material, color, style, finish, prompt,
                    original_image_url, preprocessed_edge_url, result_image_url,
                    variations_json, cnn_features_json, pipeline_mode, execution_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"],
                data["category"],
                data["material"],
                data["color"],
                data["style"],
                data["finish"],
                data["prompt"],
                data["original_image_url"],
                data["preprocessed_edge_url"],
                data["result_image_url"],
                json.dumps(data.get("variations", [])),
                json.dumps(data.get("cnn_features", {})),
                data["pipeline_mode"],
                data.get("execution_time", 0.0),
                data.get("created_at", datetime.now().isoformat())
            ))
            conn.commit()
            return data["id"]

    def get_all(self, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if category and category.lower() != "all":
                cursor.execute("""
                    SELECT * FROM generations 
                    WHERE LOWER(category) = LOWER(?)
                    ORDER BY created_at DESC LIMIT ?
                """, (category, limit))
            else:
                cursor.execute("""
                    SELECT * FROM generations 
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = self._deserialize_row(dict(r))
                results.append(item)
            return results

    @staticmethod
    def _deserialize_row(item: Dict[str, Any]) -> Dict[str, Any]:
        for json_key, out_key, default in (
            ("variations_json", "variations", []),
            ("cnn_features_json", "cnn_features", {}),
            ("model_3d_json", "model_3d", None),
        ):
            raw = item.get(json_key)
            if not raw:
                item[out_key] = default
                continue
            try:
                item[out_key] = json.loads(raw)
            except Exception:
                item[out_key] = default
        return item

    def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM generations WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._deserialize_row(dict(row))

    def update_3d_model(self, item_id: str, model_3d: Dict[str, Any]) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE generations SET model_3d_json = ? WHERE id = ?",
                (json.dumps(model_3d), item_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_by_id(self, item_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM generations WHERE id = ?", (item_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM generations")
            return cursor.fetchone()[0]

db = Database()


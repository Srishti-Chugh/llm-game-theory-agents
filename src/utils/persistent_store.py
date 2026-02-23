import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver

def get_checkpointer(db_path="scua_memory.sqlite"):
    """Returns a LangGraph SQLite saver for persistent memory"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)

class FeedbackLoopStore:
    """Manual store for game-specific historical payoffs"""
    def __init__(self, db_path="game_history.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS history 
                             (round_id INTEGER PRIMARY KEY, data TEXT)''')
        self.conn.commit()

    def save_round(self, round_data: dict):
        self.cursor.execute("INSERT INTO history (data) VALUES (?)", 
                           (json.dumps(round_data),))
        self.conn.commit()

    def get_history(self):
        self.cursor.execute("SELECT data FROM history ORDER BY round_id DESC LIMIT 10")
        return [json.loads(row[0]) for row in self.cursor.fetchall()]

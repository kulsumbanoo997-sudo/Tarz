import json
import sqlite3
import time
import uuid

from . import config


def _cx():
    cx = sqlite3.connect(config.DB_PATH)
    cx.row_factory = sqlite3.Row
    return cx


def init():
    with _cx() as cx:
        cx.executescript("""
        CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, created REAL, data TEXT, photo_path TEXT);
        CREATE TABLE IF NOT EXISTS tryons(id TEXT PRIMARY KEY, created REAL, profile_id TEXT, product_id TEXT, backend TEXT, result_url TEXT);
        CREATE TABLE IF NOT EXISTS stylist_logs(id TEXT PRIMARY KEY, created REAL, profile_id TEXT, product_id TEXT, model TEXT, verdict TEXT, total REAL);
        """)


def save_profile(profile: dict, photo_path: str | None = None) -> str:
    pid = uuid.uuid4().hex[:12]
    with _cx() as cx:
        cx.execute("INSERT INTO profiles VALUES(?,?,?,?)", (pid, time.time(), json.dumps(profile), photo_path))
    return pid


def get_profile(pid: str):
    with _cx() as cx:
        r = cx.execute("SELECT * FROM profiles WHERE id=?", (pid,)).fetchone()
    if not r:
        return None, None
    return json.loads(r["data"]), r["photo_path"]


def delete_profile(pid: str) -> str | None:
    with _cx() as cx:
        r = cx.execute("SELECT photo_path FROM profiles WHERE id=?", (pid,)).fetchone()
        cx.execute("DELETE FROM profiles WHERE id=?", (pid,))
    return r["photo_path"] if r else None


def log_tryon(profile_id, product_id, backend, url):
    with _cx() as cx:
        cx.execute("INSERT INTO tryons VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex[:12], time.time(), profile_id, product_id, backend, url))


def log_stylist(profile_id, product_id, model, verdict, total):
    with _cx() as cx:
        cx.execute("INSERT INTO stylist_logs VALUES(?,?,?,?,?,?,?)", (uuid.uuid4().hex[:12], time.time(), profile_id, product_id, model, verdict, total))

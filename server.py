import asyncio
import collections
import json
import logging
import os
import secrets
import sqlite3
import hashlib
import subprocess
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Form, Cookie
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-proxy")

SERVER_HOST = os.environ.get("SERVER_HOST", "submit01.ubelix.unibe.ch")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7535"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    logger.warning("API_KEY not set! Using default: change-me")
    API_KEY = "change-me"

SUPERUSER_EMAIL = os.environ.get("SUPERUSER_EMAIL", "silas.kieser@gmail.com")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "llm_proxy.db"))
DASHBOARD_PASSWORD = "JustDoIt!"

LLAMA_BIN = os.environ.get("LLAMA_BIN", "/rs_scratch/users/sk25f059/llama.cpp/build/bin/llama-server")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "/rs_scratch/users/sk25f059/models/Qwen3.6-27B-UD-Q4_K_XL.gguf")
SLURM_SCRIPT = os.environ.get("SLURM_SCRIPT", os.path.join(os.path.dirname(__file__), "llama_worker.sh"))
DEFAULT_GPU = os.environ.get("DEFAULT_GPU", "rtx4090:1")
DEFAULT_TIME = os.environ.get("DEFAULT_TIME", "00:20:00")
DEFAULT_MEM = os.environ.get("DEFAULT_MEM", "16G")

sessions: dict[str, dict] = {}

pool_workers: dict[str, dict] = {}
pool_pending: set[str] = set()
_pool_create_lock = asyncio.Lock()
POOL_MAX_PENDING = 4
POOL_NP = 1
POOL_IDLE_TIMEOUT = 600
POOL_RENEW_LEAD = 300
POOL_SCALE_UP = 0.7
POOL_HEALTH_RETRIES = 2
POOL_MIN_SPARE = 1  # keep at least 1 idle worker
POOL_SPARE_WINDOW = 900  # 15 min window for demand tracking
POOL_SPARE_THRESHOLD = 2  # requests in window to keep spare

session_routes: dict[str, str] = {}  # opencode_session_id -> worker_session_id
SESSION_PIN_TIMEOUT = 300  # 5 min idle before releasing session pin
SESSION_PIN_STALE_TIMEOUT = 600  # 10 min before releasing pin to a dead worker

REQUEST_HISTORY: collections.deque = collections.deque(maxlen=10000)

STATS_LOG = os.environ.get("STATS_LOG", os.path.join(os.path.dirname(__file__), "llm_proxy_stats.jsonl"))

dashboard_sessions: dict[str, dict] = {}

security = HTTPBearer(auto_error=False)


def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            key_value TEXT UNIQUE NOT NULL,
            prefix TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS usage_by_key (
            api_key_id INTEGER PRIMARY KEY,
            requests INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        );
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt.hex() + ':' + key.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(':', 1)
        salt = bytes.fromhex(salt_hex)
        expected = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return expected.hex() == key_hex
    except Exception:
        return False


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=403, detail="Invalid API key")
    token = credentials.credentials
    if token == API_KEY:
        return {"user_id": None, "key_id": None, "is_master": True}
    conn = get_db()
    row = conn.execute("""
        SELECT ak.id as key_id, ak.user_id, u.role, u.email
        FROM api_keys ak JOIN users u ON ak.user_id = u.id
        WHERE ak.key_value = ? AND ak.active = 1
    """, (token,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return dict(row) | {"is_master": False}


class CreateSession(BaseModel):
    model: str = "Qwen3.6-27B-MTP"
    time: str = DEFAULT_TIME
    gpu_type: str = DEFAULT_GPU


class RegisterBody(BaseModel):
    session_id: str
    hostname: str
    port: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    app.state.pool_task = asyncio.create_task(_pool_maintenance())
    yield
    app.state.pool_task.cancel()
    await app.state.http_client.aclose()


app = FastAPI(title="LLM Slurm Proxy", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc) or "Internal server error", "type": "server_error", "code": 500}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "server_error", "code": exc.status_code}},
    )


def _get_client() -> httpx.AsyncClient:
    return app.state.http_client


def _check_job_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
    return ""


def _parse_slurm_time(t: str) -> int:
    try:
        t = t.strip()
        days = 0
        if "-" in t:
            days, rest = t.split("-", 1)
            days = int(days)
        else:
            rest = t
        parts = rest.split(":")
        if len(parts) == 3:
            return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return days * 86400 + int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return 0


def _require_dashboard_session(session: str | None = Cookie(default=None)):
    if not session or session not in dashboard_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    sess = dashboard_sessions[session]
    if time.time() - sess["created_at"] > 86400:
        del dashboard_sessions[session]
        raise HTTPException(status_code=401, detail="Session expired")
    return sess


def _track_usage_response(request: Request, path: str, resp):
    if "chat/completions" not in path:
        return
    ct = resp.headers.get("content-type", "")
    if "application/json" not in ct:
        return
    try:
        data = resp.json()
        usage = data.get("usage")
        if not usage:
            return
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        if not token or token == API_KEY:
            return
        conn = get_db()
        row = conn.execute("SELECT id FROM api_keys WHERE key_value = ? AND active = 1", (token,)).fetchone()
        if not row:
            conn.close()
            return
        api_key_id = row["id"]
        conn.execute(
            "INSERT INTO usage_by_key (api_key_id, requests, prompt_tokens, completion_tokens) VALUES (?, 1, ?, ?) ON CONFLICT(api_key_id) DO UPDATE SET requests = requests + 1, prompt_tokens = prompt_tokens + ?, completion_tokens = completion_tokens + ?",
            (api_key_id, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _log_stats_event(event: dict):
    try:
        with open(STATS_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


_STYLE = """* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }
.nav { background: #1e293b; border-bottom: 1px solid #334155; padding: 0.75rem 1.5rem; display: flex; align-items: center; justify-content: space-between; }
.nav h2 { font-size: 1.25rem; font-weight: 700; }
.nav h2 span { color: #ea580c; }
.nav-links { display: flex; align-items: center; gap: 1.25rem; }
.nav-links a { color: #94a3b8; text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: color 0.15s; }
.nav-links a:hover { color: #ea580c; }
.nav-links a.active { color: #ea580c; }
.nav-right { display: flex; align-items: center; gap: 1rem; font-size: 0.875rem; color: #94a3b8; }
.nav-right form { display: inline; }
.logout-btn { padding: 0.375rem 0.75rem; background: transparent; border: 1px solid #475569; border-radius: 6px; color: #cbd5e1; font-size: 0.8125rem; cursor: pointer; transition: all 0.15s; }
.logout-btn:hover { background: #334155; border-color: #ea580c; color: #ea580c; }
.container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; }
.stat-card .label { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.375rem; }
.stat-card .value { font-size: 1.75rem; font-weight: 700; }
.stat-card .value.green { color: #22c55e; }
.stat-card .value.yellow { color: #eab308; }
.stat-card .value.orange { color: #ea580c; }
.stat-card .value.blue { color: #3b82f6; }
.stat-card .value.purple { color: #a855f7; }
.section { margin-bottom: 1.5rem; }
.section h3 { font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; color: #e2e8f0; display: flex; align-items: center; gap: 0.5rem; }
.badge { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 999px; font-size: 0.6875rem; font-weight: 600; }
.badge-ready { background: #14532d; color: #4ade80; }
.badge-pending { background: #451a03; color: #fb923c; }
.badge-queued { background: #1e3a5f; color: #60a5fa; }
.badge-starting { background: #5b21b6; color: #c4b5fd; }
.badge-completed { background: #1e3a5f; color: #60a5fa; }
.badge-cancelled { background: #450a0a; color: #f87171; }
.badge-failed { background: #450a0a; color: #fca5a5; }
table { width: 100%; border-collapse: collapse; background: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }
th { text-align: left; padding: 0.75rem 1rem; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; background: #0f172a; border-bottom: 1px solid #334155; }
td { padding: 0.75rem 1rem; font-size: 0.8125rem; border-bottom: 1px solid #1e293b; color: #cbd5e1; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #334155; }
.empty-state { text-align: center; padding: 2.5rem 1rem; color: #64748b; font-size: 0.875rem; }
.mono { font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 0.75rem; }
.refresh-note { text-align: center; font-size: 0.75rem; color: #475569; padding: 1rem; }
.info-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; }
.info-box h4 { font-size: 0.9375rem; font-weight: 600; margin-bottom: 0.75rem; color: #e2e8f0; cursor: pointer; user-select: none; }
.info-box h4:hover { color: #ea580c; }
.info-box h4 .arrow { display: inline-block; transition: transform 0.2s; margin-right: 0.5rem; }
.info-box h4 .arrow.open { transform: rotate(90deg); }
.info-content { display: none; font-size: 0.8125rem; color: #94a3b8; }
.info-content.open { display: block; }
.code-block { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem 1rem; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.75rem; color: #e2e8f0; overflow-x: auto; white-space: pre-wrap; word-break: break-all; margin-bottom: 0.75rem; user-select: all; }
.limits-table td:first-child { color: #64748b; white-space: nowrap; }"""


def _nav_html(email: str, role: str, active: str) -> str:
    admin_link = '<a href="/admin/users" class="nav-link">Users</a>' if role == "superuser" and active != "users" else ''
    if role == "superuser" and active == "users":
        admin_link = '<a href="/admin/users" class="nav-link active">Users</a>'
    dash_active = 'active' if active == 'dashboard' else ''
    keys_active = 'active' if active == 'keys' else ''
    return f"""<div class="nav">
<h2>Polynterpret <span>LLM</span></h2>
<div class="nav-links">
<a href="/" class="nav-link {dash_active}">Dashboard</a>
<a href="/keys" class="nav-link {keys_active}">API Keys</a>
{admin_link}
</div>
<div class="nav-right">
<span>{email}</span>
<form action="/logout" method="POST">
<button type="submit" class="logout-btn">Sign Out</button>
</form>
</div>
</div>"""


def _login_html(error: str = "") -> str:
    err_block = f'<div class="error" id="error">{error}</div>' if error else '<div class="error" id="error" style="display:none"></div>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Dashboard — Polynterpret</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
.login-card {{ background: #1e293b; border-radius: 16px; padding: 2.5rem; width: 380px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }}
.login-card h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }}
.login-card p {{ color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem; }}
.form-group {{ margin-bottom: 1rem; }}
.form-group label {{ display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.375rem; color: #cbd5e1; }}
.form-group input {{ width: 100%; padding: 0.625rem 0.75rem; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; font-size: 0.875rem; outline: none; transition: border-color 0.15s; }}
.form-group input:focus {{ border-color: #ea580c; }}
.submit-btn {{ width: 100%; padding: 0.75rem; background: #ea580c; color: white; border: none; border-radius: 8px; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: background 0.15s; }}
.submit-btn:hover {{ background: #d97706; }}
.error {{ background: #7f1d1d; color: #fca5a5; padding: 0.625rem; border-radius: 8px; font-size: 0.8125rem; margin-bottom: 1rem; }}
.logo {{ font-size: 1.75rem; font-weight: 800; margin-bottom: 1.5rem; }}
.logo span {{ color: #ea580c; }}
.footer-link {{ text-align: center; margin-top: 1rem; font-size: 0.8125rem; color: #64748b; }}
.footer-link a {{ color: #ea580c; text-decoration: none; }}
.footer-link a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="login-card">
<div class="logo">Polynterpret <span>LLM</span></div>
<h1>Dashboard Login</h1>
<p>Sign in to view job status and usage</p>
{err_block}
<form action="/login" method="POST">
<div class="form-group">
<label for="email">Email</label>
<input type="email" id="email" name="email" placeholder="you@example.com" required>
</div>
<div class="form-group">
<label for="password">Password</label>
<input type="password" id="password" name="password" placeholder="••••••••" required>
</div>
<button type="submit" class="submit-btn">Sign In</button>
</form>
<div class="footer-link">Don't have an account? <a href="/signup">Sign up</a></div>
</div>
</body>
</html>"""


def _signup_html(error: str = "") -> str:
    err_block = f'<div class="error" id="error">{error}</div>' if error else '<div class="error" id="error" style="display:none"></div>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign Up — Polynterpret LLM</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
.login-card {{ background: #1e293b; border-radius: 16px; padding: 2.5rem; width: 380px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }}
.login-card h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }}
.login-card p {{ color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem; }}
.form-group {{ margin-bottom: 1rem; }}
.form-group label {{ display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.375rem; color: #cbd5e1; }}
.form-group input {{ width: 100%; padding: 0.625rem 0.75rem; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; font-size: 0.875rem; outline: none; transition: border-color 0.15s; }}
.form-group input:focus {{ border-color: #ea580c; }}
.submit-btn {{ width: 100%; padding: 0.75rem; background: #ea580c; color: white; border: none; border-radius: 8px; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: background 0.15s; }}
.submit-btn:hover {{ background: #d97706; }}
.error {{ background: #7f1d1d; color: #fca5a5; padding: 0.625rem; border-radius: 8px; font-size: 0.8125rem; margin-bottom: 1rem; }}
.logo {{ font-size: 1.75rem; font-weight: 800; margin-bottom: 1.5rem; }}
.logo span {{ color: #ea580c; }}
.footer-link {{ text-align: center; margin-top: 1rem; font-size: 0.8125rem; color: #64748b; }}
.footer-link a {{ color: #ea580c; text-decoration: none; }}
.footer-link a:hover {{ text-decoration: underline; }}
.info-box {{ background: #1a3a5f; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.8125rem; color: #93c5fd; margin-bottom: 1.25rem; }}
</style>
</head>
<body>
<div class="login-card">
<div class="logo">Polynterpret <span>LLM</span></div>
<h1>Create Account</h1>
<p>Register for API access</p>
{err_block}
<div class="info-box">&#x2139;&#xFE0F; Your temporary password is: <strong style="color:#e2e8f0;user-select:all;">{DASHBOARD_PASSWORD}</strong><br>You will be asked to change it after first login.</div>
<form action="/signup" method="POST">
<div class="form-group">
<label for="email">Email</label>
<input type="email" id="email" name="email" placeholder="you@example.com" required>
</div>
<button type="submit" class="submit-btn">Create Account</button>
</form>
<div class="footer-link">Already have an account? <a href="/">Sign in</a></div>
</div>
</body>
</html>"""


def _change_password_html(email: str, forced: bool, error: str = "") -> str:
    err_block = f'<div class="error" id="error">{error}</div>' if error else '<div class="error" id="error" style="display:none"></div>'
    note = '<div class="info-box">&#x26A0;&#xFE0F; You are using the default password. Please set a new password to continue.</div>' if forced else ''
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Change Password — Polynterpret LLM</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
.card {{ background: #1e293b; border-radius: 16px; padding: 2.5rem; width: 400px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }}
.card h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }}
.card p {{ color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem; }}
.form-group {{ margin-bottom: 1rem; }}
.form-group label {{ display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.375rem; color: #cbd5e1; }}
.form-group input {{ width: 100%; padding: 0.625rem 0.75rem; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; font-size: 0.875rem; outline: none; transition: border-color 0.15s; }}
.form-group input:focus {{ border-color: #ea580c; }}
.submit-btn {{ width: 100%; padding: 0.75rem; background: #ea580c; color: white; border: none; border-radius: 8px; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: background 0.15s; }}
.submit-btn:hover {{ background: #d97706; }}
.error {{ background: #7f1d1d; color: #fca5a5; padding: 0.625rem; border-radius: 8px; font-size: 0.8125rem; margin-bottom: 1rem; }}
.logo {{ font-size: 1.75rem; font-weight: 800; margin-bottom: 1.5rem; }}
.logo span {{ color: #ea580c; }}
.info-box {{ background: #451a03; border: 1px solid #ea580c; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.8125rem; color: #fdba74; margin-bottom: 1.25rem; }}
</style>
</head>
<body>
<div class="card">
<div class="logo">Polynterpret <span>LLM</span></div>
<h1>Change Password</h1>
<p>{email}</p>
{note}
{err_block}
<form action="/change-password" method="POST">
<div class="form-group">
<label for="new_password">New Password</label>
<input type="text" id="new_password" name="new_password" placeholder="At least 6 characters" minlength="6" required>
</div>
<div class="form-group">
<label for="confirm_password">Confirm Password</label>
<input type="text" id="confirm_password" name="confirm_password" placeholder="Repeat password" minlength="6" required>
</div>
<button type="submit" class="submit-btn">Change Password</button>
</form>
</div>
</body>
</html>"""


def _dashboard_html(email: str, role: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Dashboard — Polynterpret</title>
<style>{_STYLE}
.info-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; }}
.info-box h4 {{ font-size: 0.9375rem; font-weight: 600; margin-bottom: 0.75rem; color: #e2e8f0; cursor: pointer; user-select: none; }}
.info-box h4:hover {{ color: #ea580c; }}
.info-box h4 .arrow {{ display: inline-block; transition: transform 0.2s; margin-right: 0.5rem; }}
.info-box h4 .arrow.open {{ transform: rotate(90deg); }}
.info-content {{ display: none; font-size: 0.8125rem; color: #94a3b8; }}
.info-content.open {{ display: block; }}
.code-block {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem 1rem; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.75rem; color: #e2e8f0; overflow-x: auto; white-space: pre-wrap; word-break: break-all; margin-bottom: 0.75rem; user-select: all; }}
.limits-table td:first-child {{ color: #64748b; white-space: nowrap; }}
</style>
</head>
<body>
{_nav_html(email, role, 'dashboard')}
<div class="container">
<div class="stats" id="stats-cards">
<div class="stat-card"><div class="label">Sessions Total</div><div class="value blue" id="stat-sessions">-</div></div>
<div class="stat-card"><div class="label">Ready</div><div class="value green" id="stat-ready">-</div></div>
<div class="stat-card"><div class="label">Pending</div><div class="value yellow" id="stat-pending">-</div></div>
<div class="stat-card"><div class="label">Pinned</div><div class="value purple" id="stat-pinned">-</div></div>
<div class="stat-card"><div class="label">Cache Hit Rate</div><div class="value green" id="stat-cache-rate">-</div></div>
<div class="stat-card"><div class="label">API Requests</div><div class="value purple" id="stat-requests">-</div></div>
</div>
<div class="section">
<h3>&#x1f4cb; Running Jobs
<button onclick="cancelAllPending()" style="float:right;padding:0.25rem 0.75rem;background:#7f1d1d;border:1px solid #f87171;border-radius:6px;color:#f87171;font-size:0.75rem;font-weight:600;cursor:pointer;">Cancel All Pending</button>
</h3>
<div id="sessions-table"><div class="empty-state">Loading...</div></div>
</div>
<div class="section">
<h3>&#x1f4ca; LLM Usage by API Key</h3>
<div id="usage-table"><div class="empty-state">Loading...</div></div>
</div>
<div class="section">
<h3>&#x1f4ca; KV Cache & Session Stats (last 5 min)</h3>
<div id="live-stats"><div class="empty-state">Loading...</div></div>
</div>
<div class="section">
<h3>&#x26A0;&#xFE0F; Error messages</h3>
<table><thead><tr><th>Code</th><th>Meaning</th></tr></thead><tbody>
<tr><td style="color:#f87171;">403</td><td>Invalid or missing API key. Check the <code>Authorization: Bearer</code> header.</td></tr>
<tr><td style="color:#fb923c;">503</td><td>No GPU worker ready yet. Slurm job is queued / starting. Retry in ~30s.</td></tr>
<tr><td style="color:#60a5fa;">502</td><td>Worker unreachable or crashed. The pool will replace it automatically.</td></tr>
<tr><td style="color:#f87171;">404</td><td>Session ID (X-LLM-Worker-Id) not found. It may have expired or been cancelled.</td></tr>
</tbody></table>
</div>
<div class="section">
<h3 onclick="this.querySelector('.arrow').classList.toggle('open');this.nextElementSibling.classList.toggle('open')"><span class="arrow">&#x25B6;</span> &#x1f4cb; KV Cache Sessions — How It Works</h3>
<div class="info-content">
<p style="margin-bottom:0.75rem;">OpenCode conversations span multiple API calls. Each call sends the full history. If successive calls hit different <code>llama-server</code> workers, there is <strong>zero KV cache reuse</strong> — every call recomputes all tokens from scratch.</p>
<p style="margin-bottom:0.75rem;">This server supports <strong>session-pinned routing</strong> via the <code>X-Session-ID</code> header. Requests with the same session ID are routed to the same worker, preserving its in-memory KV cache across turns.</p>

<p style="margin-bottom:0.5rem;"><strong>Using this service in OpenCode:</strong></p>
<ol style="padding-left:1.25rem;margin-bottom:0.75rem;">
<li>Add the provider to <code>~/.config/opencode/opencode.json</code>:</li>
</ol>
<div class="code-block">"provider": {{
  "my-hpc-llm": {{
    "npm": "@ai-sdk/openai-compatible",
    "name": "HPC LLM",
    "options": {{
      "baseURL": "http://submit01:7535/v1",
      "apiKey": "&lt;your-api-key&gt;"
    }},
    "models": {{
      "Qwen3.6-27B-MTP": {{
        "name": "Qwen 3.6 27B",
        "limit": {{ "context": 131072, "output": 8192 }}
      }}
    }}
  }}
}}</div>
<ol start="2" style="padding-left:1.25rem;margin-bottom:0.75rem;">
<li>Install the session plugin for KV cache reuse:</li>
</ol>
<div class="code-block">cd ~/.config/opencode
npm install opencode-helicone-session</div>
<ol start="3" style="padding-left:1.25rem;margin-bottom:0.75rem;">
<li>Add to <code>opencode.json</code>:</li>
</ol>
<div class="code-block">"plugin": ["opencode-helicone-session"]</div>

<p style="margin-bottom:0.5rem;"><strong>How session pinning works:</strong></p>
<ul style="padding-left:1.25rem;margin-bottom:0.5rem;">
<li>On first request with <code>X-Session-ID</code>, a GPU worker is allocated and <strong>pinned</strong> to that session</li>
<li>Subsequent requests with the same ID are routed to the <strong>same worker</strong> — KV cache is preserved</li>
<li>After <strong>5 minutes idle</strong>, the pin is released. Worker goes back to the pool for other sessions</li>
<li>If the original session returns later, it gets a (possibly different) worker and starts fresh</li>
</ul>
</div>
</div>
<div class="refresh-note">Auto-refreshes every 10 seconds</div>
</div>
<script>
const fmt = (n) => n.toLocaleString();
const age = (ts) => {{ const s = Math.floor((Date.now()/1000 - ts)); if (s < 60) return s + 's ago'; if (s < 3600) return Math.floor(s/60) + 'm ago'; if (s < 86400) return Math.floor(s/3600) + 'h ago'; return Math.floor(s/86400) + 'd ago'; }};
const fmtDuration = (secs) => {{ if (secs < 60) return Math.floor(secs) + 's'; if (secs < 3600) return Math.floor(secs/60) + 'm ' + Math.floor(secs%60) + 's'; return Math.floor(secs/3600) + 'h ' + Math.floor((secs%3600)/60) + 'm'; }};
const badge = (st) => `<span class="badge badge-${{st}}">${{st}}</span>`;
async function cancelAllPending() {{
  if (!confirm('Cancel all pending SLURM jobs?')) return;
  try {{
    const r = await fetch('/admin/cancel-pending', {{method: 'POST'}});
    const data = await r.json();
    alert('Cancelled ' + data.cancelled + ' pending jobs.');
    load();
  }} catch(e) {{ alert('Error: ' + e); }}
}}
async function load() {{
try {{
const [sessions, usage, liveStats] = await Promise.all([
  fetch('/admin/sessions').then(r=>r.json()),
  fetch('/admin/usage').then(r=>r.json()),
  fetch('/admin/live-stats').then(r=>r.json()),
]);
const ready = sessions.filter(s => s.status === 'ready').length;
const pending = sessions.filter(s => s.status === 'pending').length;
const pinned = liveStats.summary?.pinned_now || 0;
const cacheRate = liveStats.summary?.cache_hit_rate ?? '-';
document.getElementById('stat-sessions').textContent = sessions.length;
document.getElementById('stat-ready').textContent = ready;
document.getElementById('stat-pending').textContent = pending;
document.getElementById('stat-pinned').textContent = pinned;
document.getElementById('stat-cache-rate').textContent = typeof cacheRate === 'number' ? cacheRate + '%' : '-';
document.getElementById('stat-requests').textContent = fmt(usage.total?.requests || 0);
let lhtml = '';
const recent = (liveStats.events || []).filter(e => e.event === 'request').slice(-20).reverse();
if (recent.length === 0) {{ lhtml = '<div class="empty-state">No requests in recent log</div>'; }}
else {{
lhtml = '<table><thead><tr><th>Time</th><th>Session</th><th>Cache</th><th>Idle</th><th>Tokens</th><th>Status</th></tr></thead><tbody>';
for (const e of recent) {{
const t = new Date(e.ts * 1000).toLocaleTimeString();
const sess = (e.opencode_session || '').slice(0,12);
const cache = e.cache_hit ? '<span style="color:#22c55e;">HIT</span>' : '<span style="color:#f87171;">MISS</span>';
const idle = e.worker_idle_before ? fmtDuration(e.worker_idle_before) : '-';
const tokens = (e.prompt_tokens || 0) + (e.completion_tokens || 0);
lhtml += `<tr><td style="font-size:0.75rem;">${{t}}</td><td class="mono" style="font-size:0.75rem;">${{sess}}</td><td>${{cache}}</td><td style="font-size:0.75rem;">${{idle}}</td><td style="font-size:0.75rem;">${{tokens || '-'}}</td><td>${{e.status}}</td></tr>`;
}}
lhtml += '</tbody></table>';
}}
document.getElementById('live-stats').innerHTML = lhtml;
let shtml = '';
if (sessions.length === 0) {{ shtml = '<div class="empty-state">No sessions</div>'; }}
else {{
shtml = '<table><thead><tr><th>Session ID</th><th>Status</th><th>Model</th><th>Slurm Job</th><th>Worker</th><th>Uptime</th><th>Created</th></tr></thead><tbody>';
for (const s of sessions) {{
const uptime = s.uptime ? fmtDuration(s.uptime) : (s.status === 'pending' ? 'queued' : '-');
                    const statusLabel = s.status === 'pending' && s.slurm_state === 'RUNNING' ? 'starting' : s.status === 'pending' && s.slurm_state === 'PENDING' ? 'queued' : s.status;
                    shtml += `<tr><td class="mono">${{s.session_id.slice(0,8)}}&hellip;</td><td>${{badge(statusLabel)}}</td><td>${{s.model}}</td><td class="mono">${{s.slurm_job_id || '-'}}</td><td class="mono">${{s.worker_url ? s.worker_url.split('//')[1] : '-'}}</td><td>${{uptime}}</td><td>${{age(s.created_at)}}</td></tr>`;
}}
shtml += '</tbody></table>';
}}
document.getElementById('sessions-table').innerHTML = shtml;
let uhtml = '';
if ((usage.usage_by_key || []).length === 0) {{ uhtml = '<div class="empty-state">No usage data yet</div>'; }}
else {{
uhtml = '<table><thead><tr><th>Key Name</th><th>User</th><th>Prefix</th><th>Requests</th><th>Prompt Tokens</th><th>Completion Tokens</th><th>Total Tokens</th></tr></thead><tbody>';
for (const row of usage.usage_by_key) {{
uhtml += `<tr><td>${{row.key_name}}</td><td>${{row.user_email}}</td><td class="mono">${{row.key_prefix}}&hellip;</td><td>${{fmt(row.requests)}}</td><td>${{fmt(row.prompt_tokens)}}</td><td>${{fmt(row.completion_tokens)}}</td><td>${{fmt(row.prompt_tokens + row.completion_tokens)}}</td></tr>`;
}}
uhtml += '</tbody></table>';
}}
document.getElementById('usage-table').innerHTML = uhtml;
}} catch(e) {{ document.getElementById('sessions-table').innerHTML = '<div class="empty-state">Error loading data</div>'; }}
}}
load();
setInterval(load, 10000);
</script>
</body>
</html>"""


def _keys_html(email: str, role: str, keys: list, new_key: str | None = None, new_key_name: str = "") -> str:
    new_key_block = ""
    if new_key:
        new_key_block = f"""<div id="new-key-banner" style="background:#1a3a2a;border:1px solid #22c55e;border-radius:12px;padding:1.25rem;margin-bottom:1.5rem;">
<div style="display:flex;align-items:start;justify-content:space-between;">
<div>
<h4 style="color:#4ade80;margin-bottom:0.75rem;">&#x2705; Key Created: {new_key_name}</h4>
<p style="color:#94a3b8;font-size:0.875rem;margin-bottom:0.5rem;">Copy this key now. You won't be able to see it again.</p>
</div>
<button onclick="document.getElementById('new-key-banner').style.display='none'" style="background:transparent;border:none;color:#94a3b8;font-size:1.25rem;cursor:pointer;padding:0;line-height:1;">&times;</button>
</div>
<div style="display:flex;gap:0.5rem;align-items:stretch;">
<div style="flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:0.75rem 1rem;font-family:monospace;font-size:0.8125rem;color:#e2e8f0;word-break:break-all;user-select:all;">{new_key}</div>
<button onclick="navigator.clipboard.writeText('{new_key}');this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',2000)" style="padding:0.5rem 1rem;background:#22c55e;color:#052e16;border:none;border-radius:8px;font-size:0.8125rem;font-weight:600;cursor:pointer;white-space:nowrap;">Copy</button>
</div>
</div>"""
    keys_rows = ""
    if keys:
        for k in keys:
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(k["created_at"]))
            keys_rows += f"""<tr><td>{k["name"]}</td><td class="mono">{k["prefix"]}&hellip;</td><td style="color:#64748b;font-size:0.75rem;">{dt}</td>
<td><form action="/keys/{k["id"]}/delete" method="POST" onsubmit="return confirm('Delete key \\"{k["name"]}\\"?')"><button type="submit" style="padding:0.25rem 0.625rem;background:transparent;border:1px solid #7f1d1d;border-radius:6px;color:#f87171;font-size:0.75rem;cursor:pointer;">Delete</button></form></td></tr>"""
    else:
        keys_rows = '<tr><td colspan="4" class="empty-state">No API keys yet</td></tr>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Keys — Polynterpret LLM</title>
<style>{_STYLE}</style>
</head>
<body>
{_nav_html(email, role, 'keys')}
<div class="container">
{new_key_block}
<div class="section">
<h3>&#x1f511; Generate New API Key</h3>
<form action="/keys" method="POST" style="display:flex;gap:0.75rem;align-items:end;">
<div style="flex:1;">
<label for="name" style="display:block;font-size:0.75rem;font-weight:500;color:#64748b;margin-bottom:0.375rem;">Key Name</label>
<input type="text" id="name" name="name" placeholder="e.g. my-app-key" required style="width:100%;padding:0.5rem 0.75rem;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:0.875rem;outline:none;">
</div>
<button type="submit" style="padding:0.5rem 1.25rem;background:#ea580c;color:white;border:none;border-radius:8px;font-size:0.875rem;font-weight:600;cursor:pointer;white-space:nowrap;">Generate Key</button>
</form>
</div>
<div class="section">
<h3>&#x1f512; Your API Keys</h3>
<table><thead><tr><th>Name</th><th>Key</th><th>Created</th><th></th></tr></thead><tbody>{keys_rows}</tbody></table>
</div>
<div class="info-box">
<h4 onclick="this.querySelector('.arrow').classList.toggle('open');this.nextElementSibling.classList.toggle('open')"><span class="arrow">&#x25B6;</span> How to use</h4>
<div class="info-content">
<p style="margin-bottom:0.5rem;"><strong>Base URL:</strong> <code style="color:#e2e8f0;">http://submit01:7535/v1</code> (no trailing slash)</p>
<p style="margin-bottom:0.5rem;"><strong>curl example:</strong></p>
<div class="code-block">export API_KEY=&lt;your-api-key&gt;

curl -X POST http://submit01:7535/v1/chat/completions \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{
  "model": "Qwen3.6-27B-MTP",
  "messages": [{{"role": "user", "content": "Hello!"}}]
}}'</div>
<p style="margin-bottom:0.5rem;"><strong>Limits:</strong></p>
<table class="limits-table"><thead><tr><th>Limit</th><th>Value</th></tr></thead><tbody>
<tr><td>Context window</td><td>32,768 tokens</td></tr>
<tr><td>Max output</td><td>8,192 tokens</td></tr>
<tr><td>Concurrent requests</td><td>4 per worker (auto-scales)</td></tr>
</tbody></table>
</div>
</div>
</div>
</body>
</html>"""


def _admin_users_html(email: str, role: str, users: list) -> str:
    users_rows = ""
    for u in users:
        dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(u["created_at"]))
        role_badge = '<span class="badge badge-ready" style="background:#451a03;color:#fb923c;">superuser</span>' if u["role"] == "superuser" else '<span class="badge badge-pending" style="background:#1e3a5f;color:#60a5fa;">user</span>'
        conn = get_db()
        key_count = conn.execute("SELECT COUNT(*) as cnt FROM api_keys WHERE user_id = ? AND active = 1", (u["id"],)).fetchone()["cnt"]
        conn.close()
        users_rows += f"<tr><td>{u['email']}</td><td>{role_badge}</td><td>{key_count}</td><td style='color:#64748b;font-size:0.75rem;'>{dt}</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Users — Polynterpret LLM</title>
<style>{_STYLE}</style>
</head>
<body>
{_nav_html(email, role, 'users')}
<div class="container">
<div class="section">
<h3>&#x1f465; Registered Users</h3>
<table><thead><tr><th>Email</th><th>Role</th><th>API Keys</th><th>Registered</th></tr></thead><tbody>{users_rows}</tbody></table>
</div>
</div>
</body>
</html>"""


async def _submit_job(session: dict) -> str | None:
    env = {
        "SESSION_ID": session["id"],
        "SERVER_URL": SERVER_URL,
        "API_KEY": API_KEY,
        "LLAMA_BIN": LLAMA_BIN,
        "LLAMA_MODEL": LLAMA_MODEL,
        "GPU_TYPE": session.get("gpu_type", DEFAULT_GPU),
    }
    export_str = ",".join(f"{k}={v}" for k, v in env.items())
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_base = os.path.join(log_dir, f"worker_{session['id'][:8]}")
    cmd = [
        "sbatch",
        "--job-name", f"llama-{session['id'][:8]}",
        "--output", f"{log_base}_%j.out",
        "--error", f"{log_base}_%j.err",
        f"--export={export_str}",
        "--parsable",
    ]
    cmd += [
        "--qos=job_gpu_preemptable", "--partition=gpu-invest",
        "--nodes=1",
        f"--gres=gpu:{session.get('gpu_type', DEFAULT_GPU)}",
        f"--time={session.get('walltime', DEFAULT_TIME)}",
        f"--mem={session.get('memory', DEFAULT_MEM)}",
    ]
    cmd.append(SLURM_SCRIPT)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"sbatch failed: {result.stderr}")
            return None
        job_id = result.stdout.strip()
        logger.info(f"Submitted preemptable job {job_id} for session {session['id']}")
        return job_id
    except subprocess.TimeoutExpired:
        logger.error("sbatch timed out")
        return None


async def _pool_get_worker() -> dict | str | None:
    async with _pool_create_lock:
        for sid, s in list(sessions.items()):
            if s["status"] == "ready" and s.get("worker_url") and sid not in pool_workers and sid not in pool_pending:
                _pool_add(sid)
                s["pool"] = True
                logger.info(f"Lazy-adopted session {sid[:8]} into pool")

        ready = []
        for sid, pw in list(pool_workers.items()):
            s = sessions.get(sid)
            if not s or s["status"] != "ready":
                pool_workers.pop(sid, None)
                continue
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    r = await c.get(f"{s['worker_url']}/v1/models")
                if r.status_code >= 500 and r.status_code != 503:
                    continue
            except Exception:
                continue
            pw["last_active"] = time.time()
            load = pw["active_requests"] / POOL_NP
            pinned = 1 if s.get("pinned_for_session") else 0
            ready.append((pinned, load, sid, s))

        if ready:
            ready.sort(key=lambda x: (x[0], x[1]))  # unpinned first, then least-loaded
            sid, s = ready[0][2], ready[0][3]
            pool_workers[sid]["active_requests"] += 1
            return s

        return await _pool_create_worker()


async def _pool_create_worker() -> str | None:
    if len(pool_pending) >= POOL_MAX_PENDING:
        return None
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id, "status": "pending",
        "model": "Qwen3.6-27B-MTP",
        "slurm_job_id": None, "worker_url": None,
        "created_at": time.time(), "last_active": time.time(),
        "gpu_type": DEFAULT_GPU, "walltime": DEFAULT_TIME,
        "memory": DEFAULT_MEM, "mode": "gpu", "restart_count": 0,
        "auto": True, "pool": True,
    }
    sessions[session_id] = session
    pool_pending.add(session_id)
    job_id = await _submit_job(session)
    if job_id:
        session["slurm_job_id"] = job_id
        logger.info(f"Pool created session {session_id[:8]} -> Slurm job {job_id}")
        _log_stats_event({
            "event": "worker_created", "ts": time.time(),
            "worker_session": session_id, "slurm_job_id": job_id,
            "source": "pool", "model": "Qwen3.6-27B-MTP",
        })
        return session_id
    else:
        session["status"] = "failed"
        pool_pending.discard(session_id)
        logger.error(f"Pool sbatch failed for {session_id[:8]}")
        return None


def _pool_add(session_id: str):
    pool_workers[session_id] = {
        "active_requests": 0,
        "last_active": time.time(),
        "created_at": time.time(),
    }


async def _pool_release(session_id: str):
    pw = pool_workers.get(session_id)
    if pw:
        pw["active_requests"] = max(0, pw["active_requests"] - 1)


async def _pool_maintenance():
    while True:
        await asyncio.sleep(30)
        try:
            now = time.time()

            for sid in list(pool_pending):
                s = sessions.get(sid)
                if not s or s["status"] != "pending":
                    pool_pending.discard(sid)
                    continue
                job_id = s.get("slurm_job_id")
                if job_id:
                    state = _check_job_state(job_id)
                    s["slurm_state"] = state
                    if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
                        logger.info(f"Pending pool worker {sid[:8]} job {job_id} {state}, removing")
                        _log_stats_event({
                            "event": "worker_removed", "ts": time.time(),
                            "worker_session": sid, "reason": f"slurm_{state.lower()}",
                        })
                        pool_pending.discard(sid)
                        s.pop("pinned_for_session", None)
                        s["status"] = "completed"

            for sid, pw in list(pool_workers.items()):
                s = sessions.get(sid)
                if not s:
                    pool_workers.pop(sid, None)
                    continue

                alive = False
                for attempt in range(POOL_HEALTH_RETRIES):
                    try:
                        async with httpx.AsyncClient(timeout=3.0) as c:
                            r = await c.get(f"{s['worker_url']}/v1/models")
                        if r.status_code == 200:
                            alive = True
                            break
                        if r.status_code == 503:
                            alive = True
                            break
                    except Exception:
                        if attempt < POOL_HEALTH_RETRIES - 1:
                            await asyncio.sleep(1)
                if not alive:
                    evictions = pw.setdefault("evictions", 0) + 1
                    pw["evictions"] = evictions
                    if evictions >= 3:
                        logger.info(f"Pool worker {sid[:8]} unreachable for {evictions} cycles, removing")
                        _log_stats_event({
                            "event": "worker_removed", "ts": time.time(),
                            "worker_session": sid, "reason": "unreachable",
                        })
                        for oc_sid, w_sid in list(session_routes.items()):
                            if w_sid == sid:
                                del session_routes[oc_sid]
                        pool_workers.pop(sid, None)
                        s.pop("pinned_for_session", None)
                        s["status"] = "completed"
                        if s.get("slurm_job_id"):
                            subprocess.run(["scancel", s["slurm_job_id"]], capture_output=True, timeout=10)
                    continue
                pw["evictions"] = 0

                if s.get("pinned_for_session"):
                    if pw["active_requests"] == 0 and now - s["last_active"] > SESSION_PIN_TIMEOUT:
                        for oc_sid, w_sid in list(session_routes.items()):
                            if w_sid == sid:
                                del session_routes[oc_sid]
                        s.pop("pinned_for_session", None)
                        logger.info(f"Session pin released for worker {sid[:8]} after idle timeout")
                        _log_stats_event({
                            "event": "pin_released", "ts": time.time(),
                            "worker_session": sid, "reason": "idle_timeout",
                        })
                    continue

                if pw["active_requests"] == 0 and now - s["last_active"] > POOL_IDLE_TIMEOUT:
                    # Demand-based spare: keep at least 1 idle worker if recent activity
                    recent = sum(1 for t in REQUEST_HISTORY if now - t < POOL_SPARE_WINDOW)
                    idle_count = sum(1 for pw2 in pool_workers.values() if pw2["active_requests"] == 0)
                    if recent >= POOL_SPARE_THRESHOLD and idle_count <= POOL_MIN_SPARE:
                        logger.debug(f"Keeping spare worker {sid[:8]} ({recent} reqs in {POOL_SPARE_WINDOW}s)")
                    else:
                        logger.info(f"Pool worker {sid[:8]} idle 10min, killing")
                        _log_stats_event({
                            "event": "worker_removed", "ts": time.time(),
                            "worker_session": sid, "reason": "idle_timeout",
                        })
                        for oc_sid, w_sid in list(session_routes.items()):
                            if w_sid == sid:
                                del session_routes[oc_sid]
                        pool_workers.pop(sid, None)
                        s.pop("pinned_for_session", None)
                        s["status"] = "completed"
                        if s.get("slurm_job_id"):
                            subprocess.run(["scancel", s["slurm_job_id"]], capture_output=True, timeout=10)
                    continue

                job_id = s.get("slurm_job_id")
                if job_id:
                    try:
                        result = subprocess.run(
                            ["sacct", "-j", job_id, "--format=Elapsed,TimeLimit", "--noheader", "-P"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            parts = result.stdout.strip().split("\n")[0].split("|")
                            if len(parts) == 2:
                                elapsed = _parse_slurm_time(parts[0])
                                limit = _parse_slurm_time(parts[1])
                                remaining = limit - elapsed
                                if 0 < remaining < POOL_RENEW_LEAD and pw["active_requests"] == 0:
                                    logger.info(f"Pool worker {sid[:8]} expiring in {remaining}s, replacing")
                                    _log_stats_event({
                                        "event": "worker_removed", "ts": time.time(),
                                        "worker_session": sid, "reason": "slurm_expiry",
                                    })
                                    for oc_sid, w_sid in list(session_routes.items()):
                                        if w_sid == sid:
                                            del session_routes[oc_sid]
                                    s.pop("pinned_for_session", None)
                                    s["status"] = "completed"
                                    pool_workers.pop(sid, None)
                                    asyncio.create_task(_pool_create_worker())
                    except Exception:
                        pass

            # Clean up stale pins: pinned sessions whose worker is gone
            for oc_sid, w_sid in list(session_routes.items()):
                ws = sessions.get(w_sid)
                if ws is None or ws.get("status") in ("completed", "failed", "cancelled"):
                    del session_routes[oc_sid]
                    logger.info(f"Cleaned up stale pin {oc_sid[:8]} -> dead worker {w_sid[:8]}")

            total_active = sum(pw["active_requests"] for pw in pool_workers.values())
            total_cap = len(pool_workers) * POOL_NP
            pinned_count = sum(1 for sid, pw in pool_workers.items() if sessions.get(sid, {}).get("pinned_for_session"))
            _log_stats_event({
                "event": "pool_status", "ts": now,
                "active_workers": len(pool_workers),
                "pinned_workers": pinned_count,
                "pending": len(pool_pending),
                "total_sessions": len(sessions),
                "pool_capacity": total_cap,
                "pool_load": total_active,
            })
            # Scale up on load threshold
            if pool_workers and total_active >= total_cap * POOL_SCALE_UP and len(pool_pending) < POOL_MAX_PENDING:
                logger.info(f"Pool at {total_active}/{total_cap} ({total_active/total_cap:.0%}), spawning worker")
                asyncio.create_task(_pool_create_worker())
            # Recovery: pool empty but there's demand (pinned or pending)
            if not pool_workers and (pinned_count > 0 or pool_pending) and len(pool_pending) < POOL_MAX_PENDING:
                logger.info(f"Pool empty with {pinned_count} pinned + {len(pool_pending)} pending, spawning recovery worker")
                asyncio.create_task(_pool_create_worker())
            # Demand-based pre-scaling: spawn spare if recent activity warrants it
            recent = sum(1 for t in REQUEST_HISTORY if now - t < POOL_SPARE_WINDOW)
            if pool_workers and recent >= POOL_SPARE_THRESHOLD and len(pool_pending) < POOL_MAX_PENDING:
                idle_count = sum(1 for pw2 in pool_workers.values() if pw2["active_requests"] == 0)
                if idle_count < POOL_MIN_SPARE:
                    logger.info(f"Recent activity ({recent} reqs), spawning spare worker")
                    asyncio.create_task(_pool_create_worker())
        except Exception as e:
            logger.error(f"Pool maintenance error: {e}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, session: str | None = Cookie(default=None)):
    if session and session in dashboard_sessions:
        sess = dashboard_sessions[session]
        if time.time() - sess["created_at"] <= 86400:
            if sess.get("needs_password_change"):
                return RedirectResponse(url="/change-password", status_code=303)
            return _dashboard_html(sess["email"], sess.get("role", "user"))
    return _login_html()


@app.get("/signup", response_class=HTMLResponse)
async def signup_form():
    return _signup_html()


@app.post("/signup", response_class=HTMLResponse)
async def signup(email: str = Form(...)):
    conn = get_db()
    role = "superuser" if email == SUPERUSER_EMAIL else "user"
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (email, hash_password(DASHBOARD_PASSWORD), role, time.time())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return _signup_html(error="Email already registered")
    conn.close()
    token = secrets.token_hex(32)
    dashboard_sessions[token] = {"email": email, "role": role, "user_id": None, "created_at": time.time(), "needs_password_change": True}
    redirect = RedirectResponse(url="/change-password", status_code=303)
    redirect.set_cookie(key="session", value=token, httponly=True, max_age=86400, path="/")
    return redirect


@app.post("/login", response_class=HTMLResponse)
async def login(email: str = Form(...), password: str = Form(...)):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user:
        if password == DASHBOARD_PASSWORD:
            role = "superuser" if email == SUPERUSER_EMAIL else "user"
            conn.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (email, hash_password(DASHBOARD_PASSWORD), role, time.time())
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        else:
            conn.close()
            return _login_html(error="Invalid email or password")

    if not verify_password(password, user["password_hash"]):
        conn.close()
        return _login_html(error="Invalid email or password")

    needs_change = verify_password(DASHBOARD_PASSWORD, user["password_hash"])
    conn.close()

    token = secrets.token_hex(32)
    dashboard_sessions[token] = {"email": email, "role": user["role"], "user_id": user["id"], "created_at": time.time(), "needs_password_change": needs_change}

    if needs_change:
        redirect = RedirectResponse(url="/change-password", status_code=303)
    else:
        redirect = RedirectResponse(url="/", status_code=303)
    redirect.set_cookie(key="session", value=token, httponly=True, max_age=86400, path="/")
    return redirect


@app.post("/logout")
async def logout(session: str | None = Cookie(default=None)):
    if session and session in dashboard_sessions:
        del dashboard_sessions[session]
    redirect = RedirectResponse(url="/", status_code=303)
    redirect.delete_cookie(key="session", path="/")
    return redirect


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(sess=Depends(_require_dashboard_session)):
    return _change_password_html(sess["email"], sess.get("needs_password_change", False))


@app.post("/change-password", response_class=HTMLResponse)
async def change_password(new_password: str = Form(...), confirm_password: str = Form(...), sess=Depends(_require_dashboard_session)):
    if len(new_password) < 6:
        return _change_password_html(sess["email"], True, error="New password must be at least 6 characters")
    if new_password != confirm_password:
        return _change_password_html(sess["email"], True, error="Passwords do not match")
    conn = get_db()
    conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hash_password(new_password), sess["email"]))
    conn.commit()
    conn.close()
    sess["needs_password_change"] = False
    return RedirectResponse(url="/", status_code=303)


@app.get("/keys", response_class=HTMLResponse)
async def keys_page(sess=Depends(_require_dashboard_session)):
    if sess.get("needs_password_change"):
        return RedirectResponse(url="/change-password", status_code=303)
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (sess["email"],)).fetchone()
    if not user:
        conn.close()
        return RedirectResponse(url="/", status_code=303)
    keys = conn.execute(
        "SELECT id, name, prefix, created_at FROM api_keys WHERE user_id = ? AND active = 1 ORDER BY created_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()
    return _keys_html(sess["email"], sess.get("role", "user"), [dict(k) for k in keys])


@app.post("/keys", response_class=HTMLResponse)
async def create_key(name: str = Form(...), sess=Depends(_require_dashboard_session)):
    if not name.strip():
        return RedirectResponse(url="/keys", status_code=303)
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (sess["email"],)).fetchone()
    if not user:
        conn.close()
        return RedirectResponse(url="/keys", status_code=303)
    key_value = secrets.token_hex(32)
    prefix = key_value[:8]
    conn.execute(
        "INSERT INTO api_keys (user_id, name, key_value, prefix, created_at) VALUES (?, ?, ?, ?, ?)",
        (user["id"], name.strip(), key_value, prefix, time.time())
    )
    conn.commit()
    keys = conn.execute(
        "SELECT id, name, prefix, created_at FROM api_keys WHERE user_id = ? AND active = 1 ORDER BY created_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()
    return _keys_html(sess["email"], sess.get("role", "user"), [dict(k) for k in keys], new_key=key_value, new_key_name=name.strip())


@app.post("/keys/{key_id}/delete")
async def delete_key(key_id: int, sess=Depends(_require_dashboard_session)):
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (sess["email"],)).fetchone()
    if user:
        conn.execute("UPDATE api_keys SET active = 0 WHERE id = ? AND user_id = ?", (key_id, user["id"]))
        conn.commit()
    conn.close()
    return RedirectResponse(url="/keys", status_code=303)


@app.get("/admin/sessions")
async def admin_sessions(_=Depends(_require_dashboard_session)):
    now = time.time()
    return [{
        "session_id": s["id"],
        "status": s["status"],
        "mode": s.get("mode", "gpu"),
        "model": s["model"],
        "slurm_job_id": s.get("slurm_job_id"),
        "worker_url": s.get("worker_url"),
        "auto": s.get("auto", False),
        "created_at": s["created_at"],
        "last_active": s.get("last_active", 0),
        "uptime": now - s.get("last_active", s["created_at"]) if s["status"] == "ready" and s.get("worker_url") else 0,
    } for s in sessions.values()]


@app.get("/admin/usage")
async def admin_usage(_=Depends(_require_dashboard_session)):
    conn = get_db()
    rows = conn.execute("""
        SELECT ak.name as key_name, ak.prefix as key_prefix, u.email as user_email,
               COALESCE(ubk.requests, 0) as requests,
               COALESCE(ubk.prompt_tokens, 0) as prompt_tokens,
               COALESCE(ubk.completion_tokens, 0) as completion_tokens,
               ak.created_at
        FROM usage_by_key ubk
        JOIN api_keys ak ON ak.id = ubk.api_key_id
        JOIN users u ON u.id = ak.user_id
        WHERE ak.active = 1
        ORDER BY ak.created_at DESC
    """).fetchall()
    total = conn.execute("""
        SELECT COALESCE(SUM(requests), 0) as requests,
               COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) as completion_tokens
        FROM usage_by_key
    """).fetchone()
    conn.close()
    return {"usage_by_key": [dict(r) for r in rows], "total": dict(total)}


@app.get("/admin/live-stats")
async def admin_live_stats(_=Depends(_require_dashboard_session)):
    if not os.path.exists(STATS_LOG):
        return {"events": [], "summary": {}}
    try:
        with open(STATS_LOG) as f:
            lines = f.readlines()
        events = [json.loads(l) for l in lines[-500:] if l.strip()]
        requests = [e for e in events if e.get("event") == "request"]
        total = len(requests)
        hits = sum(1 for e in requests if e.get("cache_hit"))
        recent = [e for e in requests if e.get("ts", 0) > time.time() - 300]
        return {
            "events": events[-100:],
            "summary": {
                "total_requests": total,
                "cache_hits": hits,
                "cache_misses": total - hits,
                "cache_hit_rate": round(hits / total * 100, 1) if total else 0,
                "requests_5min": len(recent),
                "cache_hits_5min": sum(1 for e in recent if e.get("cache_hit")),
                "pinned_now": sum(1 for s in sessions.values() if s.get("pinned_for_session")),
                "workers_ready": sum(1 for s in sessions.values() if s["status"] == "ready"),
            },
        }
    except Exception:
        return {"events": [], "summary": {}}


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(sess=Depends(_require_dashboard_session)):
    if sess.get("needs_password_change"):
        return RedirectResponse(url="/change-password", status_code=303)
    if sess.get("role") != "superuser":
        raise HTTPException(status_code=403, detail="Superuser only")
    conn = get_db()
    users = conn.execute("SELECT id, email, role, created_at FROM users ORDER BY created_at ASC").fetchall()
    conn.close()
    return _admin_users_html(sess["email"], sess["role"], [dict(u) for u in users])


@app.post("/admin/cancel-pending")
async def cancel_pending(sess=Depends(_require_dashboard_session)):
    cancelled = 0
    for sid, s in list(sessions.items()):
        if s["status"] == "pending" and s.get("slurm_job_id"):
            subprocess.run(["scancel", s["slurm_job_id"]], capture_output=True, timeout=10)
            s["status"] = "cancelled"
            pool_pending.discard(sid)
            for oc_sid, w_sid in list(session_routes.items()):
                if w_sid == sid:
                    del session_routes[oc_sid]
            _log_stats_event({
                "event": "worker_removed", "ts": time.time(),
                "worker_session": sid, "reason": "user_cancelled",
            })
            cancelled += 1
    return {"cancelled": cancelled}


@app.get("/sessions")
async def list_sessions(_=Depends(require_auth)):
    return [{"session_id": s["id"], "status": s["status"], "mode": s.get("mode", "gpu"), "model": s["model"], "slurm_job_id": s.get("slurm_job_id"), "slurm_state": s.get("slurm_state"), "auto": s.get("auto", False)} for s in sessions.values()]


@app.post("/sessions")
async def create_session(body: CreateSession, _=Depends(require_auth)):
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "status": "pending",
        "model": body.model,
        "slurm_job_id": None,
        "worker_url": None,
        "created_at": time.time(),
        "last_active": time.time(),
        "gpu_type": body.gpu_type,
        "walltime": body.time,
        "memory": DEFAULT_MEM,
        "mode": "gpu",
        "restart_count": 0,
        "auto": False,
    }
    sessions[session_id] = session
    job_id = await _submit_job(session)
    if not job_id:
        session["status"] = "failed"
        raise HTTPException(status_code=500, detail="sbatch failed")
    session["slurm_job_id"] = job_id
    _log_stats_event({
        "event": "worker_created", "ts": time.time(),
        "worker_session": session_id, "slurm_job_id": job_id,
        "source": "explicit", "model": body.model,
    })
    return {"session_id": session_id, "status": "pending", "slurm_job_id": job_id}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, _=Depends(require_auth)):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s["status"] == "pending" and s.get("slurm_job_id"):
        state = _check_job_state(s["slurm_job_id"])
        if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
            s["status"] = "completed"
    return {"session_id": s["id"], "status": s["status"], "mode": s.get("mode", "gpu"), "model": s["model"], "worker_url": s.get("worker_url"), "slurm_job_id": s.get("slurm_job_id"), "created_at": s["created_at"], "auto": s.get("auto", False)}


@app.post("/register")
async def register_worker(body: RegisterBody):
    s = sessions.get(body.session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    s["worker_url"] = f"http://{body.hostname}:{body.port}"
    s["status"] = "ready"
    pool_pending.discard(body.session_id)
    _log_stats_event({
        "event": "worker_ready", "ts": time.time(),
        "worker_session": body.session_id,
        "startup_seconds": time.time() - s["created_at"],
        "source": "pool" if s.get("pool") else "explicit",
    })
    if s.get("pool"):
        _pool_add(body.session_id)
        logger.info(f"Pool worker registered: {s['worker_url']} for session {body.session_id[:8]}")
    else:
        logger.info(f"Worker registered: {s['worker_url']} for session {body.session_id}")
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _=Depends(require_auth)):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.get("slurm_job_id"):
        subprocess.run(["scancel", s["slurm_job_id"]], capture_output=True, timeout=10)
    s["status"] = "cancelled"
    pool_workers.pop(session_id, None)
    pool_pending.discard(session_id)
    for oc_sid, w_sid in list(session_routes.items()):
        if w_sid == session_id:
            del session_routes[oc_sid]
    _log_stats_event({
        "event": "worker_removed", "ts": time.time(),
        "worker_session": session_id, "reason": "user_cancelled",
    })
    return {"ok": True}


async def _proxy(request: Request, path: str):
    opencode_session = (
        request.headers.get("X-Session-ID") or
        request.headers.get("Helicone-Session-Id")
    )
    session_id = request.headers.get("X-LLM-Worker-Id")
    from_pool = False

    cache_hit = False
    worker_idle_before = 0.0

    REQUEST_HISTORY.append(time.time())

    if opencode_session:
        worker_id = session_routes.get(opencode_session)
        if worker_id:
            s = sessions.get(worker_id)
            if s and s["status"] == "ready":
                session_id = worker_id
                cache_hit = True
                s["pinned_for_session"] = opencode_session
            elif s and s["status"] == "pending":
                return JSONResponse(status_code=503, content={"error": {"message": "GPU worker still starting. Retry in ~30s.", "type": "server_error", "code": 503}})
            else:
                session_routes.pop(opencode_session, None)
                worker_id = None

        if not session_id:
            result = await _pool_get_worker()
            if result is None:
                return JSONResponse(status_code=503, content={"error": {"message": "Server at capacity, try again later.", "type": "server_error", "code": 503}})
            if isinstance(result, str):
                session_routes[opencode_session] = result
                return JSONResponse(status_code=503, content={"error": {"message": "Allocating GPU resources. Retry in ~30s.", "type": "server_error", "code": 503}})
            s = result
            session_id = s["id"]
            session_routes[opencode_session] = session_id
            s["pinned_for_session"] = opencode_session
        else:
            s = sessions.get(session_id)
            if not s or s["status"] != "ready":
                return JSONResponse(status_code=503, content={"error": {"message": f"Session status: {s.get('status', 'not found')}", "type": "server_error", "code": 503}})

    elif session_id:
        s = sessions.get(session_id)
        if not s:
            return JSONResponse(status_code=404, content={"error": {"message": "Session not found", "type": "not_found", "code": 404}})
        if s["status"] != "ready":
            return JSONResponse(status_code=503, content={"error": {"message": f"Session status: {s['status']}", "type": "server_error", "code": 503}})

    else:
        result = await _pool_get_worker()
        if result is None:
            return JSONResponse(status_code=503, content={"error": {"message": "Server at capacity, try again later.", "type": "server_error", "code": 503}})
        if isinstance(result, str):
            return JSONResponse(status_code=503, content={"error": {"message": "Allocating GPU resources. Retry in ~30s.", "type": "server_error", "code": 503}})
        s = result
        session_id = s["id"]
        from_pool = True

    target_url = f"{s['worker_url']}/{path.lstrip('/')}"
    now = time.time()
    worker_idle_before = now - s.get("last_active", s["created_at"])
    s["last_active"] = now

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("x-session-id", None)
    headers.pop("x-llm-worker-id", None)
    headers.pop("helicone-session-id", None)
    headers.pop("authorization", None)

    client = _get_client()
    try:
        resp = await client.request(method=request.method, url=target_url, headers=headers, content=body, follow_redirects=True)
        resp_body = resp.json() if "chat/completions" in path and "application/json" in resp.headers.get("content-type", "") else None
        _track_usage_response(request, path, resp)
        usage = resp_body.get("usage") if resp_body else None
        _log_stats_event({
            "event": "request", "ts": time.time(),
            "opencode_session": opencode_session, "worker_session": session_id,
            "cache_hit": cache_hit, "path": path,
            "status": resp.status_code, "worker_idle_before": worker_idle_before,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
        })
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except httpx.RequestError as e:
        logger.error(f"Proxy error for {session_id}: {e}")
        _log_stats_event({
            "event": "request_error", "ts": time.time(),
            "opencode_session": opencode_session, "worker_session": session_id,
            "path": path, "worker_idle_before": worker_idle_before, "error": str(e),
        })
        return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "proxy_error", "code": 502}})
    finally:
        if from_pool:
            await _pool_release(session_id)
        elif opencode_session:
            await _pool_release(session_id)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_v1(request: Request, path: str, _=Depends(require_auth)):
    return await _proxy(request, f"v1/{path}")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_api(request: Request, path: str, _=Depends(require_auth)):
    return await _proxy(request, f"api/{path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)

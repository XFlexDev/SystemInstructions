import asyncio
import json
import uuid
import aiosqlite
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import websockets
from typing import Optional, List
import uvicorn

DB_PATH = "/root/cluster_chats.db"

# Worker pool state
workers = {}
worker_paths = {}
pending_requests = {}
worker_queue: List[str] = []

async def ws_handler(websocket):
    email = None
    try:
        async for raw_msg in websocket:
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            # Worker registration on startup
            if msg_type == "REGISTER":
                email = data.get("email")
                workers[email] = websocket
                worker_paths[email] = data.get("path")
                if email not in worker_queue:
                    worker_queue.append(email)
                print(f"🟢 Worker registered: [{email}] | Active: {len(workers)}/4", flush=True)
                await websocket.send(json.dumps({"status": "REGISTERED"}))

            elif msg_type == "READY_STATE":
                worker_paths[email] = data.get("path")

            elif msg_type == "RESPONSE":
                req_id = data.get("id")
                if req_id in pending_requests and not pending_requests[req_id].done():
                    pending_requests[req_id].set_result(data)

            elif msg_type == "ERROR":
                req_id = data.get("id")
                if req_id in pending_requests and not pending_requests[req_id].done():
                    pending_requests[req_id].set_exception(Exception(data.get("error")))

    except Exception as e:
        print(f"⚠️ Worker [{email}] disconnected: {e}", flush=True)
    finally:
        if email in workers:
            del workers[email]
        if email in worker_paths:
            del worker_paths[email]
        if email in worker_queue:
            worker_queue.remove(email)
        print(f"Active workers remaining: {list(workers.keys())}", flush=True)

async def run_websocket_hub():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        print("🚀 WebSocket hub listening on ws://0.0.0.0:8765", flush=True)
        await asyncio.Future()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite table
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                last_account TEXT NOT NULL,
                prompt_url TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    hub_task = asyncio.create_task(run_websocket_hub())
    yield
    hub_task.cancel()

app = FastAPI(title="AI Studio 4-Worker Cluster", lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    if not workers:
        raise HTTPException(status_code=503, detail="No AI Studio browser workers connected!")

    # 1. Look up session in SQLite
    target_account = None
    target_path = "/prompts/new_chat?model=gemini-3.5-flash-lite"

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT last_account, prompt_url FROM sessions WHERE session_id = ?", (req.session_id,)) as cur:
            row = await cur.fetchone()
            if row:
                target_account, target_path = row[0], row[1]

    # Prioritize the account that already holds this chat thread
    candidate_accounts = [acc for acc in worker_queue if acc in workers]
    if target_account and target_account in candidate_accounts:
        candidate_accounts.remove(target_account)
        candidate_accounts.insert(0, target_account)

    last_err = None

    # 2. Failover Loop across accounts
    for account_email in candidate_accounts:
        ws = workers.get(account_email)
        if not ws:
            continue

        req_id = str(uuid.uuid4())
        fut = asyncio.get_event_loop().create_future()
        pending_requests[req_id] = fut

        try:
            print(f"👉 Routing prompt to [{account_email}] for session [{req.session_id}]...", flush=True)

            await ws.send(json.dumps({
                "id": req_id,
                "action": "EXECUTE",
                "prompt": req.prompt,
                "expected_path": target_path if account_email == target_account else "/prompts/new_chat?model=gemini-3.5-flash-lite"
            }))

            res = await asyncio.wait_for(fut, timeout=90.0)
            reply_text = res.get("response", "")

            # Check for permission or quota errors
            if "permission denied" in reply_text.lower() or ("error" in reply_text.lower() and len(reply_text) < 60):
                raise Exception(f"AI Studio error: {reply_text}")

            new_path = res.get("path")

            # 3. Persist session mapping in SQLite
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("""
                    INSERT INTO sessions (session_id, last_account, prompt_url)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET last_account = excluded.last_account, prompt_url = excluded.prompt_url
                """, (req.session_id, account_email, new_path))
                await db.commit()

            # Rotate queue to distribute load
            worker_queue.remove(account_email)
            worker_queue.append(account_email)

            return {
                "session_id": req.session_id,
                "account_used": account_email,
                "prompt_url": new_path,
                "response": reply_text
            }

        except Exception as err:
            err_str = str(err)
            print(f"❌ Worker [{account_email}] failed: {err_str}", flush=True)
            last_err = err_str
            # Move failing worker to back of the queue
            if account_email in worker_queue:
                worker_queue.remove(account_email)
                worker_queue.append(account_email)
            print("🔄 Auto-failing over to next available account...", flush=True)
            continue
        finally:
            pending_requests.pop(req_id, None)

    raise HTTPException(status_code=500, detail=f"All workers failed. Last error: {last_err}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, ws="none")

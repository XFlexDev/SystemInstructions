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
MAX_MESSAGES_PER_CHAT = 500  # Roll over to fresh chat after 40 messages (20 turns)

# Worker state
workers = {}       # { "u0": websocket, "u1": websocket, ... }
worker_info = {}   # { "u0": { "email": "...", "path": "..." } }
worker_queue: List[str] = [] # Account rotation queue
pending_requests = {}
nav_futures = {}   # { "u0": Future } to await SPA page navigations

def get_u_index(account_key: str) -> int:
    try:
        return int(account_key.split("_")[0].replace("u", ""))
    except Exception:
        return 0

async def ws_handler(websocket):
    account_key = None
    try:
        async for raw_msg in websocket:
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "REGISTER":
                account_key = data.get("account_id", "u0") # e.g. "u0"
                workers[account_key] = websocket
                worker_info[account_key] = {
                    "email": data.get("email"),
                    "path": data.get("path")
                }
                if account_key not in worker_queue:
                    worker_queue.append(account_key)
                print(f"🟢 Worker registered: [{account_key}] ({data.get('email')}) | Path: {data.get('path')}", flush=True)
                await websocket.send(json.dumps({"status": "REGISTERED"}))

                # Fulfill navigation wait if waiting for this worker to reload
                if account_key in nav_futures and not nav_futures[account_key].done():
                    nav_futures[account_key].set_result(data.get("path"))

            elif msg_type == "READY_STATE":
                if account_key in worker_info:
                    worker_info[account_key]["path"] = data.get("path")
                if account_key in nav_futures and not nav_futures[account_key].done():
                    nav_futures[account_key].set_result(data.get("path"))

            elif msg_type == "RESPONSE":
                req_id = data.get("id")
                if req_id in pending_requests and not pending_requests[req_id].done():
                    pending_requests[req_id].set_result(data)

            elif msg_type == "ERROR":
                req_id = data.get("id")
                if req_id in pending_requests and not pending_requests[req_id].done():
                    pending_requests[req_id].set_exception(Exception(data.get("error")))

    except Exception as e:
        print(f"⚠️ Worker [{account_key}] disconnected: {e}", flush=True)
    finally:
        if account_key in workers:
            del workers[account_key]
        if account_key in worker_info:
            del worker_info[account_key]
        if account_key in worker_queue:
            worker_queue.remove(account_key)
        print(f"Active workers remaining: {list(workers.keys())}", flush=True)

async def run_websocket_hub():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        print("🚀 WebSocket hub listening on ws://0.0.0.0:8765", flush=True)
        await asyncio.Future()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Relational Database
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                prompt_url TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            )
        """)
        await db.commit()
    hub_task = asyncio.create_task(run_websocket_hub())
    yield
    hub_task.cancel()

app = FastAPI(title="AI Studio 4-Worker Cluster", lifespan=lifespan)

# Request Models
class NewChatRequest(BaseModel):
    chat_id: Optional[str] = None
    account: Optional[str] = None # e.g. "u0", "u1" or None for auto-balance

class ChatRequest(BaseModel):
    chat_id: Optional[str] = None
    session_id: Optional[str] = None # Backward compatibility
    prompt: str

# Helper: navigate worker to URL and wait for reload
async def navigate_worker(account_key: str, target_url: str):
    ws = workers.get(account_key)
    if not ws:
        raise Exception(f"Worker {account_key} offline")

    fut = asyncio.get_event_loop().create_future()
    nav_futures[account_key] = fut

    print(f"🔄 Navigating [{account_key}] to: {target_url}...", flush=True)
    await ws.send(json.dumps({
        "action": "NAVIGATE",
        "url": target_url
    }))

    try:
        await asyncio.wait_for(fut, timeout=20.0)
    finally:
        nav_futures.pop(account_key, None)

# 1. NEW CHAT ENDPOINT
@app.post("/v1/chat/new")
async def create_new_chat(req: NewChatRequest):
    if not workers:
        raise HTTPException(status_code=503, detail="No AI Studio browser workers connected!")

    chat_id = req.chat_id or f"chat_{uuid.uuid4().hex[:10]}"

    # Pick account
    assigned_account = req.account
    if not assigned_account or assigned_account not in workers:
        assigned_account = worker_queue[0]
        worker_queue.remove(assigned_account)
        worker_queue.append(assigned_account)

    u_idx = get_u_index(assigned_account)
    new_chat_url = f"https://aistudio.google.com/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"
    db_prompt_path = f"/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"

    # Save to SQLite
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO chats (chat_id, account_id, prompt_url, message_count)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(chat_id) DO UPDATE SET 
                account_id = excluded.account_id,
                prompt_url = excluded.prompt_url,
                message_count = 0,
                updated_at = CURRENT_TIMESTAMP
        """, (chat_id, assigned_account, db_prompt_path))
        await db.commit()

    # Direct worker to new chat URL
    await navigate_worker(assigned_account, new_chat_url)

    return {
        "chat_id": chat_id,
        "account_assigned": assigned_account,
        "prompt_url": db_prompt_path,
        "message": "New chat created and initialized."
    }

# 2. CHAT & EXECUTE ENDPOINT
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    if not workers:
        raise HTTPException(status_code=503, detail="No AI Studio browser workers connected!")

    chat_id = req.chat_id or req.session_id or f"chat_{uuid.uuid4().hex[:10]}"

    # Check if chat exists in SQLite
    chat_record = None
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT account_id, prompt_url, message_count FROM chats WHERE chat_id = ?", (chat_id,)) as cur:
            chat_record = await cur.fetchone()

    # If new chat, initialize it
    if not chat_record:
        assigned_account = worker_queue[0]
        worker_queue.remove(assigned_account)
        worker_queue.append(assigned_account)
        u_idx = get_u_index(assigned_account)
        prompt_url = f"/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"
        message_count = 0

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO chats (chat_id, account_id, prompt_url, message_count)
                VALUES (?, ?, ?, 0)
            """, (chat_id, assigned_account, prompt_url))
            await db.commit()
    else:
        assigned_account, prompt_url, message_count = chat_record

    # AUTO-ROLLOVER: If conversation has grown too long, start a fresh Google chat thread
    if message_count >= MAX_MESSAGES_PER_CHAT:
        print(f"⚠️ Chat [{chat_id}] reached {message_count} messages. Rolling over to a fresh Google chat thread...", flush=True)
        u_idx = get_u_index(assigned_account)
        fresh_url = f"https://aistudio.google.com/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"
        await navigate_worker(assigned_account, fresh_url)
        prompt_url = f"/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"
        message_count = 0

    # Build priority list for execution and failover
    candidate_accounts = [acc for acc in worker_queue if acc in workers]
    if assigned_account in candidate_accounts:
        candidate_accounts.remove(assigned_account)
        candidate_accounts.insert(0, assigned_account)

    last_err = None

    for account_key in candidate_accounts:
        ws = workers.get(account_key)
        if not ws:
            continue

        # If switching accounts on failover, adjust URL index
        if account_key != assigned_account:
            u_idx = get_u_index(account_key)
            current_target_url = f"/u/{u_idx}/prompts/new_chat?model=gemini-3.5-flash-lite"
        else:
            current_target_url = prompt_url

        # Check if browser worker needs to navigate to the chat's URL
        current_worker_path = worker_info.get(account_key, {}).get("path", "")
        if current_target_url not in current_worker_path and "new_chat" not in current_target_url:
            print(f"Navigating [{account_key}] to resume chat: {current_target_url}...", flush=True)
            await navigate_worker(account_key, f"https://aistudio.google.com{current_target_url}")

        req_id = str(uuid.uuid4())
        fut = asyncio.get_event_loop().create_future()
        pending_requests[req_id] = fut

        try:
            print(f"👉 Dispatching prompt to [{account_key}] for chat_id [{chat_id}]...", flush=True)
            await ws.send(json.dumps({
                "id": req_id,
                "action": "EXECUTE",
                "prompt": req.prompt
            }))

            res = await asyncio.wait_for(fut, timeout=90.0)
            reply_text = res.get("response", "")

            if "permission denied" in reply_text.lower() or ("error" in reply_text.lower() and len(reply_text) < 60):
                raise Exception(f"AI Studio error: {reply_text}")

            new_path = res.get("path")

            # Store turn in SQLite
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)", (chat_id, req.prompt))
                await db.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, 'model', ?)", (chat_id, reply_text))
                await db.execute("""
                    UPDATE chats 
                    SET account_id = ?, prompt_url = ?, message_count = message_count + 2, updated_at = CURRENT_TIMESTAMP
                    WHERE chat_id = ?
                """, (account_key, new_path, chat_id))
                await db.commit()

            # Rotate queue
            worker_queue.remove(account_key)
            worker_queue.append(account_key)

            return {
                "chat_id": chat_id,
                "account_used": account_key,
                "prompt_url": new_path,
                "message_count": message_count + 2,
                "response": reply_text
            }

        except Exception as err:
            err_str = str(err)
            print(f"❌ Worker [{account_key}] failed: {err_str}", flush=True)
            last_err = err_str
            if account_key in worker_queue:
                worker_queue.remove(account_key)
                worker_queue.append(account_key)
            print("🔄 Failing over to next account...", flush=True)
            continue
        finally:
            pending_requests.pop(req_id, None)

    raise HTTPException(status_code=500, detail=f"All workers failed. Last error: {last_err}")

# 3. GET CHAT TRANSCRIPT ENDPOINT
@app.get("/v1/chat/{chat_id}")
async def get_chat_history(chat_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT account_id, prompt_url, message_count, updated_at FROM chats WHERE chat_id = ?", (chat_id,)) as cur:
            chat_info = await cur.fetchone()
            if not chat_info:
                raise HTTPException(status_code=404, detail="Chat ID not found")

        async with db.execute("SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)) as cur:
            messages = [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in await cur.fetchall()]

    return {
        "chat_id": chat_id,
        "account_id": chat_info[0],
        "prompt_url": chat_info[1],
        "message_count": chat_info[2],
        "last_updated": chat_info[3],
        "messages": messages
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, ws="none")

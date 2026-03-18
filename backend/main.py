import os
import time
import json
import shutil
import uuid
import threading
import asyncio
import numpy as np
import faiss
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks, Depends, status, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import Database modules
from database import SessionLocal, init_db, User as UserModel, Project as ProjectModel
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = os.environ.get("ALGORITHM")
# Sửa lại thời gian mặc định cho an toàn nếu thiếu biến môi trường
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)) 
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n-cls.pt")

app = FastAPI(title="NeuroSearch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS_DIR = "projects_data"
UPLOADS_DIR = "uploads"
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserSession:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.image_paths: List[str] = []
        self.current_project: Optional[Dict] = None

user_sessions: Dict[str, UserSession] = {}
model: Optional[YOLO] = None
indexing_states: Dict[str, Dict] = {}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        # Lưu trữ danh sách kết nối WS theo username
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, username: str):
        await websocket.accept()
        if username not in self.active_connections:
            self.active_connections[username] = []
        self.active_connections[username].append(websocket)

    def disconnect(self, websocket: WebSocket, username: str):
        if username in self.active_connections:
            self.active_connections[username].remove(websocket)
            if not self.active_connections[username]:
                del self.active_connections[username]

    async def send_personal_message(self, message: dict, username: str):
        if username in self.active_connections:
            for connection in self.active_connections[username]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"WS send error for {username}: {e}")

manager = ConnectionManager()
app_loop = None # Lưu event loop chính để background thread gọi về

# --- Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        raise credentials_exception
    
    if username not in user_sessions:
        user_sessions[username] = UserSession()
        
    return user

async def get_current_user_ws(token: str, db: Session):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: return None
        user = db.query(UserModel).filter(UserModel.username == username).first()
        return user
    except:
        return None

def load_model():
    global model
    print(f"Loading YOLO model: {YOLO_MODEL_PATH} on CPU...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        model.to('cpu')
    except Exception as e:
        print(f"Error loading model: {e}. Fallback to 'yolov8n-cls.pt'")
        try:
            model = YOLO('yolov8n-cls.pt')
            model.to('cpu')
        except:
            print("CRITICAL: Failed to load model")
            model = None

def get_embedding(source):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = model.embed(source, device='cpu')
    return results[0].cpu().numpy().astype('float32')

def load_project_data(username: str, project: ProjectModel):
    session = user_sessions[username]
    if os.path.exists(project.index_file) and os.path.exists(project.metadata_file):
        try:
            session.index = faiss.read_index(project.index_file)
            with open(project.metadata_file, "r") as f:
                session.image_paths = json.load(f)
            session.current_project = {
                "id": project.id, "name": project.name, "train_path": project.train_path,
                "index_file": project.index_file, "metadata_file": project.metadata_file
            }
            return True
        except Exception as e:
            print(f"Error loading project data: {e}")
            return False
    else:
        session.current_project = {
            "id": project.id, "name": project.name, "train_path": project.train_path,
            "index_file": project.index_file, "metadata_file": project.metadata_file
        }
        session.index = None
        session.image_paths = []
        return False

# --- Background Task ---
def process_build_index(project_id: str, train_path: str, index_file: str, metadata_file: str, username: str):
    global indexing_states, app_loop
    
    state = {
        "is_indexing": True, "progress": 0, "total_files": 0,
        "processed_files": 0, "current_step": "Scanning directory..."
    }
    indexing_states[project_id] = state

    def broadcast(event_type: str, data: dict = None):
        if app_loop and not app_loop.is_closed():
            msg = {"type": event_type, "project_id": project_id}
            if data: msg["data"] = data
            asyncio.run_coroutine_threadsafe(manager.send_personal_message(msg, username), app_loop)

    try:
        broadcast("indexing_update", state)
        
        files = []
        for root, _, filenames in os.walk(train_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    files.append(os.path.relpath(os.path.join(root, filename), train_path))
        
        if not files: raise Exception("No images found")

        state["total_files"] = len(files)
        state["current_step"] = "Extracting embeddings..."
        broadcast("indexing_update", state)
        
        vectors, valid_paths = [], []
        
        for idx, f in enumerate(files):
            try:
                vec = get_embedding(os.path.join(train_path, f))
                vectors.append(vec)
                valid_paths.append(f)
            except Exception as e:
                print(f"Error embedding {f}: {e}")
            
            state["processed_files"] = idx + 1
            state["progress"] = int(((idx + 1) / len(files)) * 90)
            
            # Chỉ broadcast mỗi 5 file hoặc file cuối cùng để tránh spam socket quá nhiều
            if (idx + 1) % 5 == 0 or (idx + 1) == len(files):
                broadcast("indexing_update", state)
        
        if not vectors: raise Exception("No vectors extracted")

        state["current_step"] = "Building Index..."
        broadcast("indexing_update", state)
        
        dataset_vectors = np.array(vectors)
        new_index = faiss.IndexFlatL2(dataset_vectors.shape[1])
        new_index.add(dataset_vectors)
        
        state["current_step"] = "Saving data..."
        broadcast("indexing_update", state)
        
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        faiss.write_index(new_index, index_file)
        with open(metadata_file, "w") as f:
            json.dump(valid_paths, f)

        if username in user_sessions:
            session = user_sessions[username]
            if session.current_project and session.current_project['id'] == project_id:
                session.index = new_index
                session.image_paths = valid_paths

    except Exception as e:
        state["current_step"] = f"Error: {str(e)}"
        broadcast("indexing_update", state)
    finally:
        state["progress"] = 100
        state["is_indexing"] = False
        broadcast("indexing_complete")

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    global app_loop
    app_loop = asyncio.get_running_loop() # Lấy loop để chạy threadsafe WS
    init_db()
    load_model()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...), db: Session = Depends(get_db)):
    user = await get_current_user_ws(token, db)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await manager.connect(websocket, user.username)
    try:
        while True:
            # Giữ kết nối mở, client không cần gửi gì, chỉ nhận
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, user.username)

@app.post("/auth/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(UserModel).filter(UserModel.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = UserModel(username=username, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    return {"status": "success", "message": "User created"}

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer", "username": user.username}

@app.post("/auth/change-password")
async def change_password(old_password: str = Form(...), new_password: str = Form(...), current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
    if not verify_password(old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    return {"status": "success"}

@app.get("/users/me")
async def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return {"username": current_user.username}

@app.get("/projects")
def get_projects(current_user: UserModel = Depends(get_current_user)):
    projects = []
    for p in current_user.projects:
        is_idx = indexing_states.get(p.id, {}).get("is_indexing", False)
        prog = indexing_states.get(p.id, {}).get("progress", 0)
        projects.append({
            "id": p.id, "name": p.name, "train_path": p.train_path,
            "index_file": p.index_file, "metadata_file": p.metadata_file,
            "created_at": p.created_at, "is_default": p.is_default,
            "owner": current_user.username, "is_indexing": is_idx, "indexing_progress": prog
        })
    return projects

@app.post("/projects/create")
async def create_project(name: str = Form(...), files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None, current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    username = current_user.username
    project_id = str(uuid.uuid4())[:8]
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
    if not safe_name: safe_name = project_id
    
    project_train_path = os.path.join(UPLOADS_DIR, f"{username}_{safe_name}_{project_id}")
    index_file = os.path.join(PROJECTS_DIR, project_id, "vector.index")
    metadata_file = os.path.join(PROJECTS_DIR, project_id, "paths.json")

    try:
        os.makedirs(project_train_path, exist_ok=True)
        saved_count = 0
        ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in ALLOWED:
                with open(os.path.join(project_train_path, os.path.basename(file.filename)), "wb+") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_count += 1
                
        if saved_count == 0: raise Exception("No valid images uploaded")

        is_first = len(current_user.projects) == 0
        new_project = ProjectModel(
            id=project_id, name=name, train_path=os.path.abspath(project_train_path),
            index_file=index_file, metadata_file=metadata_file, created_at=time.time(),
            is_default=is_first, owner_id=current_user.id
        )
        db.add(new_project)
        db.commit()
        db.refresh(new_project)

        if is_first:
            session = user_sessions[username]
            session.current_project = { "id": new_project.id, "name": new_project.name, "train_path": new_project.train_path, "index_file": new_project.index_file, "metadata_file": new_project.metadata_file }
            session.index = None
            session.image_paths = []

        background_tasks.add_task(process_build_index, project_id, new_project.train_path, index_file, metadata_file, username)
        
        # Trả về data format giống hệt hàm cũ
        project_dict = {
            "id": new_project.id,
            "name": new_project.name,
            "train_path": new_project.train_path,
            "index_file": new_project.index_file,
            "metadata_file": new_project.metadata_file,
            "created_at": new_project.created_at,
            "is_default": new_project.is_default,
            "owner": username
        }
        
        return {"status": "started", "project": project_dict, "file_count": saved_count}
    except Exception as e:
        db.rollback()
        if os.path.exists(project_train_path): shutil.rmtree(project_train_path)
        raise HTTPException(status_code=400, detail=f"Failed: {str(e)}")

@app.post("/projects/switch")
def switch_project(project_id: str = Form(...), current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    target = db.query(ProjectModel).filter(ProjectModel.id == project_id, ProjectModel.owner_id == current_user.id).first()
    if not target: raise HTTPException(status_code=404, detail="Not found")
    if indexing_states.get(project_id, {}).get("is_indexing", False): raise HTTPException(status_code=400, detail="Indexing")
    success = load_project_data(current_user.username, target)
    
    project_dict = {
        "id": target.id,
        "name": target.name,
        "is_default": target.is_default
    }
    return {
        "status": "success" if success else "needs_indexing", 
        "project": project_dict,
        "message": "Switched successfully"
    }

@app.post("/projects/set_default")
def set_default_project(project_id: str = Form(...), current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(ProjectModel).filter(ProjectModel.owner_id == current_user.id).update({ProjectModel.is_default: False})
    target = db.query(ProjectModel).filter(ProjectModel.id == project_id, ProjectModel.owner_id == current_user.id).first()
    target.is_default = True
    db.commit()
    return {"status": "success"}

@app.post("/projects/delete")
def delete_project(project_id: str = Form(...), current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    target = db.query(ProjectModel).filter(ProjectModel.id == project_id, ProjectModel.owner_id == current_user.id).first()
    if not target: raise HTTPException(status_code=404, detail="Not found")
    if indexing_states.get(project_id, {}).get("is_indexing", False): raise HTTPException(status_code=400, detail="Indexing")

    try:
        if os.path.exists(target.train_path): shutil.rmtree(target.train_path)
        if os.path.exists(os.path.join(PROJECTS_DIR, project_id)): shutil.rmtree(os.path.join(PROJECTS_DIR, project_id))
        db.delete(target)
        db.commit()
        
        if current_user.username in user_sessions:
            s = user_sessions[current_user.username]
            if s.current_project and s.current_project['id'] == project_id:
                s.current_project, s.index, s.image_paths = None, None, []
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexing_status")
def get_indexing_status(current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if session and session.current_project:
        return indexing_states.get(session.current_project["id"], {"is_indexing": False})
    return {"is_indexing": False}

@app.get("/status")
def get_status(current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session: return {"status": "loading", "index_size": 0, "current_project": "None"}
    
    proj_id = session.current_project["id"] if session.current_project else None
    status_text = "ready"
    if proj_id and indexing_states.get(proj_id, {}).get("is_indexing", False): status_text = "indexing"
    elif session.index is None: status_text = "loading" if session.current_project else "no_project"
        
    return {
        "status": status_text, "index_size": session.index.ntotal if session.index else 0,
        "current_project": session.current_project["name"] if session.current_project else "None",
        "train_path": session.current_project["train_path"] if session.current_project else ""
    }

@app.get("/serve_image/{filename:path}")
def serve_image(filename: str):
    if os.path.exists(filename): return FileResponse(filename)
    if os.path.exists(os.path.join(UPLOADS_DIR, filename)): return FileResponse(os.path.join(UPLOADS_DIR, filename))
    raise HTTPException(status_code=404)

@app.get("/database")
def get_database(request: Request, current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session or not session.image_paths or not session.current_project: return []
    # Trả về kết quả hoàn chỉnh y hệt bản gốc
    base_url = str(request.base_url)
    train_path = session.current_project["train_path"]
    results = []
    for filename in session.image_paths:
        full_path = os.path.join(train_path, filename)
        results.append({
            "filename": filename,
            "url": f"{base_url}serve_image/{full_path}" 
        })
    return results

@app.get("/visualize")
def get_visualization(request: Request, current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session or not session.current_project: return {"error": "No project"}
    if session.index is None or session.index.ntotal < 3: return {"error": "Min 3 images"}

    vectors = session.index.reconstruct_n(0, session.index.ntotal)
    # Re-added the mean calculation that was in the original code
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projection = np.dot(centered, Vt[:3].T)
    except: return {"error": "PCA failed"}

    points = []
    base_url = str(request.base_url)
    train_path = session.current_project["train_path"]
    
    for i, path in enumerate(session.image_paths):
        full_path = os.path.join(train_path, path)
        points.append({
            "filename": path,
            "url": f"{base_url}serve_image/{full_path}",
            "x": float(projection[i, 0]),
            "y": float(projection[i, 1]),
            "z": float(projection[i, 2])
        })
        
    return {"points": points}

@app.post("/add")
async def add_to_index(file: UploadFile = File(...), current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session or not session.index: raise HTTPException(status_code=503)
    
    save_path = os.path.join(session.current_project["train_path"], file.filename)
    with open(save_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

    vec = get_embedding(save_path).reshape(1, -1)
    session.index.add(vec)
    session.image_paths.append(file.filename)
    faiss.write_index(session.index, session.current_project["index_file"])
    with open(session.current_project["metadata_file"], "w") as f: json.dump(session.image_paths, f)
    return {"status": "success", "index_size": session.index.ntotal}

@app.post("/delete")
def delete_image(filename: str = Form(...), current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session or not session.index or filename not in session.image_paths: raise HTTPException(status_code=404)

    idx = session.image_paths.index(filename)
    try: os.remove(os.path.join(session.current_project["train_path"], filename))
    except: pass

    session.index.remove_ids(np.array([idx], dtype='int64'))
    session.image_paths.pop(idx)
    faiss.write_index(session.index, session.current_project["index_file"])
    with open(session.current_project["metadata_file"], "w") as f: json.dump(session.image_paths, f)
    return {"status": "deleted", "index_size": session.index.ntotal}

@app.post("/search")
async def search_image(request: Request, k: int = Form(5), file: UploadFile = File(...), current_user: UserModel = Depends(get_current_user)):
    session = user_sessions.get(current_user.username)
    if not session or not session.index: raise HTTPException(status_code=503)

    start_time = time.time()
    query_vector = get_embedding(Image.open(file.file)).reshape(1, -1)
    distances, indices = session.index.search(query_vector, k=min(k, len(session.image_paths)))

    results = []
    base_url = str(request.base_url)
    train_path = session.current_project["train_path"]

    for i, idx in enumerate(indices[0]):
        if idx != -1:
            fname = session.image_paths[idx]
            full_path = os.path.join(train_path, fname)
            results.append({
                "filename": fname, 
                "filepath": full_path, # Fixed missing field
                "url": f"{base_url}serve_image/{full_path}", 
                "distance": float(distances[0][i])
            })
            
    return {"results": results, "query_time": time.time() - start_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
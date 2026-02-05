# import os
# import time
# import json
# import shutil
# import uuid
# import threading
# import numpy as np
# import faiss
# from typing import List, Optional, Dict
# from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from ultralytics import YOLO
# from PIL import Image
# from passlib.context import CryptContext
# from jose import JWTError, jwt
# from pydantic import BaseModel

# from dotenv import load_dotenv

# load_dotenv()

# # --- Configuration ---
# SECRET_KEY = os.environ.get("SECRET_KEY")
# ALGORITHM = os.environ.get("ALGORITHM")
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")) # 1 day
# YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH")

# # Initialize FastAPI
# app = FastAPI(title="NeuroSearch API")

# # CORS Setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Global Configuration & Paths ---
# PROJECTS_FILE = "projects.json"
# USERS_FILE = "users.json"
# PROJECTS_DIR = "projects_data"
# UPLOADS_DIR = "uploads"

# # Ensure directories exist
# os.makedirs(PROJECTS_DIR, exist_ok=True)
# os.makedirs(UPLOADS_DIR, exist_ok=True)

# # --- Auth Setup ---
# # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# pwd_context = CryptContext(
#     schemes=["argon2"],
#     deprecated="auto",
# )

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# # --- Models ---
# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class User(BaseModel):
#     username: str

# class UserInDB(User):
#     hashed_password: str

# # --- State Management (Per User) ---
# class UserSession:
#     def __init__(self):
#         self.index: Optional[faiss.IndexFlatL2] = None
#         self.image_paths: List[str] = []
#         self.current_project: Optional[Dict] = None

# # Global dictionary to hold session state for each active user
# # Key: username, Value: UserSession instance
# user_sessions: Dict[str, UserSession] = {}

# # Global Model (Shared)
# model: Optional[YOLO] = None

# # Indexing State (Simplified for multi-user: we track by project_id)
# # Key: project_id, Value: Status Dict
# indexing_states: Dict[str, Dict] = {}

# # --- Helper Functions ---

# def load_users():
#     if not os.path.exists(USERS_FILE):
#         return []
#     try:
#         with open(USERS_FILE, "r") as f:
#             return json.load(f)
#     except:
#         return []

# def save_users(users):
#     with open(USERS_FILE, "w") as f:
#         json.dump(users, f, indent=2)

# def get_user(username: str):
#     users = load_users()
#     for user in users:
#         if user["username"] == username:
#             return user
#     return None

# # def verify_password(plain_password, hashed_password):
# #     return pwd_context.verify(plain_password, hashed_password)

# # def get_password_hash(password):
# #     return pwd_context.hash(password)

# import hashlib

# def normalize_password(password: str) -> str:
#     return hashlib.sha256(password.encode("utf-8")).hexdigest()  # 64 chars

# def get_password_hash(password: str) -> str:
#     normalized = normalize_password(password)
#     return pwd_context.hash(normalized)

# def verify_password(password: str, hashed: str) -> bool:
#     normalized = normalize_password(password)
#     return pwd_context.verify(normalized, hashed)

# def create_access_token(data: dict):
#     to_encode = data.copy()
#     expire = time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
    
#     user = get_user(username)
#     if user is None:
#         raise credentials_exception
    
#     # Initialize session for user if not exists
#     if username not in user_sessions:
#         user_sessions[username] = UserSession()
        
#     return user

# def load_projects_list():
#     if not os.path.exists(PROJECTS_FILE):
#         return []
#     try:
#         with open(PROJECTS_FILE, "r") as f:
#             projects = json.load(f)
#             if not isinstance(projects, list): return []
#             return projects
#     except:
#         return []

# def save_projects_list(projects):
#     with open(PROJECTS_FILE, "w") as f:
#         json.dump(projects, f, indent=2)

# def initialize_projects():
#     if not os.path.exists(PROJECTS_FILE):
#         save_projects_list([])

# def load_model():
#     global model
#     model_path = os.environ.get("YOLO_MODEL_PATH", "YOLO_MODEL_PATH")
#     print(f"Loading YOLO model: {model_path}...")
#     try:
#         model = YOLO(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}. Fallback to 'yolov8n-cls.pt'")
#         try:
#             model = YOLO('yolov8n-cls.pt')
#         except:
#             print("CRITICAL: Failed to load model")
#             model = None

# def get_embedding(source):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#     results = model.embed(source)
#     return results[0].cpu().numpy().astype('float32')

# def load_project_data(username: str, project: dict):
#     session = user_sessions[username]
#     print(f"Loading project '{project['name']}' for user '{username}'")
    
#     if os.path.exists(project['index_file']) and os.path.exists(project['metadata_file']):
#         try:
#             session.index = faiss.read_index(project['index_file'])
#             with open(project['metadata_file'], "r") as f:
#                 session.image_paths = json.load(f)
#             session.current_project = project
#             return True
#         except Exception as e:
#             print(f"Error loading project data: {e}")
#             return False
#     else:
#         session.current_project = project
#         session.index = None
#         session.image_paths = []
#         return False

# # --- Background Task ---
# def process_build_index(project_id: str, train_path: str, index_file: str, metadata_file: str, username: str):
#     global indexing_states
    
#     state = {
#         "is_indexing": True,
#         "progress": 0,
#         "total_files": 0,
#         "processed_files": 0,
#         "current_step": "Scanning directory..."
#     }
#     indexing_states[project_id] = state

#     try:
#         print(f"Starting index build for {project_id} (User: {username})")
        
#         files = []
#         for root, _, filenames in os.walk(train_path):
#             for filename in filenames:
#                 if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
#                     relative_path = os.path.relpath(os.path.join(root, filename), train_path)
#                     files.append(relative_path)
        
#         if not files: raise Exception("No images found")

#         state["total_files"] = len(files)
#         state["current_step"] = "Extracting embeddings..."
        
#         vectors = []
#         valid_paths = []
        
#         for idx, f in enumerate(files):
#             full_path = os.path.join(train_path, f)
#             try:
#                 vec = get_embedding(full_path)
#                 vectors.append(vec)
#                 valid_paths.append(f)
#             except Exception as e:
#                 print(f"Error embedding {f}: {e}")
            
#             processed = idx + 1
#             state["processed_files"] = processed
#             state["progress"] = int((processed / len(files)) * 90)
        
#         if not vectors: raise Exception("No vectors extracted")

#         state["current_step"] = "Building Index..."
#         dataset_vectors = np.array(vectors)
#         new_index = faiss.IndexFlatL2(dataset_vectors.shape[1])
#         new_index.add(dataset_vectors)
        
#         state["current_step"] = "Saving data..."
#         os.makedirs(os.path.dirname(index_file), exist_ok=True)
#         os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

#         faiss.write_index(new_index, index_file)
#         with open(metadata_file, "w") as f:
#             json.dump(valid_paths, f)
            
#         print("Index build complete.")

#         # If the user is currently looking at this project, reload it into their session
#         if username in user_sessions:
#             session = user_sessions[username]
#             if session.current_project and session.current_project['id'] == project_id:
#                 session.index = new_index
#                 session.image_paths = valid_paths

#     except Exception as e:
#         print(f"Indexing Failed: {e}")
#         state["current_step"] = f"Error: {str(e)}"
#     finally:
#         state["progress"] = 100
#         state["is_indexing"] = False

# # --- Auth Endpoints ---

# @app.post("/auth/register")
# async def register(username: str = Form(...), password: str = Form(...)):
#     users = load_users()
#     if any(u["username"] == username for u in users):
#         raise HTTPException(status_code=400, detail="Username already registered")
    
#     hashed_password = get_password_hash(password)
#     new_user = {"username": username, "hashed_password": hashed_password}
#     users.append(new_user)
#     save_users(users)
#     return {"status": "success", "message": "User created"}

# @app.post("/auth/login")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = get_user(form_data.username)
#     if not user or not verify_password(form_data.password, user["hashed_password"]):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token = create_access_token(data={"sub": user["username"]})
#     return {"access_token": access_token, "token_type": "bearer", "username": user["username"]}

# @app.get("/users/me")
# async def read_users_me(current_user: dict = Depends(get_current_user)):
#     return {"username": current_user["username"]}

# # --- Project Endpoints ---

# @app.on_event("startup")
# async def startup_event():
#     load_model()
#     initialize_projects()

# @app.get("/projects")
# def get_projects(current_user: dict = Depends(get_current_user)):
#     all_projects = load_projects_list()
#     # Filter projects owned by user
#     user_projects = [p for p in all_projects if p.get("owner") == current_user["username"]]
#     return user_projects

# @app.post("/projects/create")
# async def create_project(
#     name: str = Form(...), 
#     files: List[UploadFile] = File(...),
#     background_tasks: BackgroundTasks = None,
#     current_user: dict = Depends(get_current_user)
# ):
#     username = current_user["username"]
    
#     # Check if user has an active indexing job (Simplified check)
#     # Ideally iterate indexing_states and check ownership, but simple check is okay
    
#     projects = load_projects_list()
#     user_projects = [p for p in projects if p.get("owner") == username]
#     is_first = len(user_projects) == 0

#     project_id = str(uuid.uuid4())[:8]
#     safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
#     if not safe_name: safe_name = project_id
        
#     project_train_path = os.path.join(UPLOADS_DIR, f"{username}_{safe_name}_{project_id}")
#     os.makedirs(project_train_path, exist_ok=True)
    
#     saved_count = 0
#     ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
#     for file in files:
#         ext = os.path.splitext(file.filename)[1].lower()
#         if ext in ALLOWED:
#             file_path = os.path.join(project_train_path, os.path.basename(file.filename))
#             with open(file_path, "wb+") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             saved_count += 1
            
#     if saved_count == 0:
#         shutil.rmtree(project_train_path)
#         raise HTTPException(status_code=400, detail="No valid images")

#     index_file = os.path.join(PROJECTS_DIR, project_id, "vector.index")
#     metadata_file = os.path.join(PROJECTS_DIR, project_id, "paths.json")

#     new_project = {
#         "id": project_id,
#         "name": name,
#         "owner": username,
#         "train_path": os.path.abspath(project_train_path),
#         "index_file": index_file,
#         "metadata_file": metadata_file,
#         "created_at": time.time(),
#         "is_default": is_first
#     }

#     projects.append(new_project)
#     save_projects_list(projects)

#     # Auto switch if it's the first project
#     if is_first:
#         # We need to set it in the session, but wait for indexing
#         # Session logic handles uninitialized index gracefully
#         session = user_sessions[username]
#         session.current_project = new_project
#         session.index = None
#         session.image_paths = []

#     background_tasks.add_task(process_build_index, project_id, new_project["train_path"], index_file, metadata_file, username)

#     return {"status": "started", "project": new_project, "file_count": saved_count}

# @app.post("/projects/switch")
# def switch_project(project_id: str = Form(...), current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     projects = load_projects_list()
    
#     # Ensure project belongs to user
#     target = next((p for p in projects if p["id"] == project_id and p.get("owner") == username), None)
    
#     if not target:
#         raise HTTPException(status_code=404, detail="Project not found")
    
#     # Check if indexing
#     if project_id in indexing_states and indexing_states[project_id]["is_indexing"]:
#          raise HTTPException(status_code=400, detail="Project is currently indexing")

#     success = load_project_data(username, target)
    
#     return {
#         "status": "success" if success else "needs_indexing", 
#         "project": target,
#         "message": "Switched successfully"
#     }

# @app.post("/projects/set_default")
# def set_default_project(project_id: str = Form(...), current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     projects = load_projects_list()
    
#     target = next((p for p in projects if p["id"] == project_id and p.get("owner") == username), None)
#     if not target: raise HTTPException(status_code=404, detail="Project not found")

#     for p in projects:
#         if p.get("owner") == username:
#             p["is_default"] = (p["id"] == project_id)
        
#     save_projects_list(projects)
#     return {"status": "success", "message": f"Project '{target['name']}' set as default."}

# @app.get("/indexing_status")
# def get_indexing_status(current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if session and session.current_project:
#         proj_id = session.current_project["id"]
#         if proj_id in indexing_states:
#             return indexing_states[proj_id]
            
#     return {"is_indexing": False}

# @app.get("/status")
# def get_status(current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session:
#         return {"status": "loading", "index_size": 0, "current_project": "None"}
        
#     proj_id = session.current_project["id"] if session.current_project else None
    
#     status_text = "ready"
#     if proj_id and proj_id in indexing_states and indexing_states[proj_id]["is_indexing"]:
#         status_text = "indexing"
#     elif session.index is None:
#         status_text = "loading" if session.current_project else "no_project"
        
#     return {
#         "status": status_text,
#         "index_size": session.index.ntotal if session.index else 0,
#         "current_project": session.current_project["name"] if session.current_project else "None",
#         "train_path": session.current_project["train_path"] if session.current_project else ""
#     }

# @app.get("/serve_image/{filename:path}")
# def serve_image(filename: str):
#     # This endpoint remains open for image tags to load, but requires obfuscation in real world.
#     # For now, we assume the filename contains enough entropy or we just check existence.
#     # Note: To secure this properly, we'd need a token in the URL or cookies.
#     # We will check if the file exists in UPLOADS_DIR (searching recursively is expensive, 
#     # but we rely on the fact that we stored absolute path or relative known structure).
    
#     # Security simplification: We won't validate user ownership here for <img> tag simplicity,
#     # as <img> tags don't easily send Bearer headers.
#     # In a strict environment, use a signed URL or Cookie Auth.
    
#     if os.path.exists(filename): # Absolute path check
#         return FileResponse(filename)
    
#     # Try finding in uploads dir if path is relative
#     possible_path = os.path.join(UPLOADS_DIR, filename) 
#     if os.path.exists(possible_path):
#         return FileResponse(possible_path)
        
#     raise HTTPException(status_code=404, detail="File not found")

# @app.get("/database")
# def get_database(request: Request, current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session or not session.image_paths or not session.current_project:
#         return []
    
#     base_url = str(request.base_url)
#     results = []
#     # session.image_paths stores relative paths inside the train_path
#     train_path = session.current_project["train_path"]
    
#     for filename in session.image_paths:
#         full_path = os.path.join(train_path, filename)
#         # We serve using the absolute path for simplicity in serve_image
#         results.append({
#             "filename": filename,
#             "url": f"{base_url}serve_image/{full_path}" 
#         })
#     return results

# @app.get("/visualize")
# def get_visualization(request: Request, current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session or not session.current_project:
#          return {"error": "No project loaded."}
    
#     if session.index is None or session.index.ntotal < 3:
#         return {"error": "Not enough data (min 3 images)."}

#     vectors = session.index.reconstruct_n(0, session.index.ntotal)
#     mean = np.mean(vectors, axis=0)
#     centered = vectors - mean
    
#     try:
#         U, S, Vt = np.linalg.svd(centered, full_matrices=False)
#         components = Vt[:3]
#         projection = np.dot(centered, components.T)
#     except:
#         return {"error": "PCA failed"}

#     points = []
#     base_url = str(request.base_url)
#     train_path = session.current_project["train_path"]
    
#     for i, path in enumerate(session.image_paths):
#         full_path = os.path.join(train_path, path)
#         points.append({
#             "filename": path,
#             "url": f"{base_url}serve_image/{full_path}",
#             "x": float(projection[i, 0]),
#             "y": float(projection[i, 1]),
#             "z": float(projection[i, 2])
#         })
        
#     return {"points": points}

# @app.post("/add")
# async def add_to_index(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session or not session.index or not session.current_project:
#         raise HTTPException(status_code=503, detail="Project not ready")

#     filename = file.filename
#     save_path = os.path.join(session.current_project["train_path"], filename)
    
#     try:
#         with open(save_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Save failed: {e}")

#     try:
#         vec = get_embedding(save_path)
#         vec = vec.reshape(1, -1)
#         session.index.add(vec)
#         session.image_paths.append(filename)
#     except Exception as e:
#          raise HTTPException(status_code=500, detail=f"Index update failed: {e}")

#     # Persistence
#     faiss.write_index(session.index, session.current_project["index_file"])
#     with open(session.current_project["metadata_file"], "w") as f:
#         json.dump(session.image_paths, f)

#     return {"status": "success", "index_size": session.index.ntotal}

# @app.post("/delete")
# def delete_image(filename: str = Form(...), current_user: dict = Depends(get_current_user)):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session or not session.index:
#         raise HTTPException(status_code=503, detail="Not ready")
        
#     if filename not in session.image_paths:
#         raise HTTPException(status_code=404, detail="Image not found")

#     idx = session.image_paths.index(filename)
#     session.index.remove_ids(np.array([idx], dtype='int64'))
#     session.image_paths.pop(idx)

#     faiss.write_index(session.index, session.current_project["index_file"])
#     with open(session.current_project["metadata_file"], "w") as f:
#         json.dump(session.image_paths, f)

#     return {"status": "deleted", "index_size": session.index.ntotal}

# @app.post("/search")
# async def search_image(
#     request: Request, 
#     k: int = Form(5), 
#     file: UploadFile = File(...),
#     current_user: dict = Depends(get_current_user)
# ):
#     username = current_user["username"]
#     session = user_sessions.get(username)
    
#     if not session or not session.current_project:
#         raise HTTPException(status_code=503, detail="No project selected")
        
#     if not session.index or not session.image_paths:
#         raise HTTPException(status_code=503, detail="Index empty")

#     start_time = time.time()
#     try:
#         contents = await file.read()
#         import io
#         image_data = Image.open(io.BytesIO(contents))
#         query_vector = get_embedding(image_data).reshape(1, -1)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Processing failed: {e}")

#     search_k = min(k, len(session.image_paths))
#     distances, indices = session.index.search(query_vector, k=search_k)

#     results = []
#     base_url = str(request.base_url)
#     train_path = session.current_project["train_path"]

#     for i, idx in enumerate(indices[0]):
#         if idx == -1: continue
#         filename = session.image_paths[idx]
#         full_path = os.path.join(train_path, filename)
#         results.append({
#             "filename": filename,
#             "filepath": full_path,
#             "url": f"{base_url}serve_image/{full_path}",
#             "distance": float(distances[0][i])
#         })

#     return {"results": results, "query_time": time.time() - start_time}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import time
import json
import shutil
import uuid
import threading
import numpy as np
import faiss
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks, Depends, status
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
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")) # 1 day
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH")

# Initialize FastAPI
app = FastAPI(title="NeuroSearch API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configuration & Paths ---
PROJECTS_DIR = "projects_data"
UPLOADS_DIR = "uploads"

# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Auth Setup ---
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Models (Pydantic) ---
class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

class UserSchema(BaseModel):
    username: str

# --- State Management (Per User - In Memory) ---
# Note: Session state (loaded index in RAM) remains in memory as it's not persistent in DB
class UserSession:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.image_paths: List[str] = []
        self.current_project: Optional[Dict] = None

# Global dictionary to hold session state for each active user
# Key: username, Value: UserSession instance
user_sessions: Dict[str, UserSession] = {}

# Global Model (Shared)
model: Optional[YOLO] = None

# Indexing State (Simplified for multi-user: we track by project_id)
# Key: project_id, Value: Status Dict
indexing_states: Dict[str, Dict] = {}

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
    
    # Query User from DB
    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        raise credentials_exception
    
    # Initialize session for user if not exists
    if username not in user_sessions:
        user_sessions[username] = UserSession()
        
    return user

def load_model():
    global model
    model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n-cls.pt")
    print(f"Loading YOLO model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Fallback to 'yolov8n-cls.pt'")
        try:
            model = YOLO('yolov8n-cls.pt')
        except:
            print("CRITICAL: Failed to load model")
            model = None

def get_embedding(source):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = model.embed(source)
    return results[0].cpu().numpy().astype('float32')

def load_project_data(username: str, project: ProjectModel):
    session = user_sessions[username]
    print(f"Loading project '{project.name}' for user '{username}'")
    
    if os.path.exists(project.index_file) and os.path.exists(project.metadata_file):
        try:
            session.index = faiss.read_index(project.index_file)
            with open(project.metadata_file, "r") as f:
                session.image_paths = json.load(f)
            # Convert SQLAlchemy model to dict for session usage
            session.current_project = {
                "id": project.id,
                "name": project.name,
                "train_path": project.train_path,
                "index_file": project.index_file,
                "metadata_file": project.metadata_file
            }
            return True
        except Exception as e:
            print(f"Error loading project data: {e}")
            return False
    else:
        session.current_project = {
            "id": project.id,
            "name": project.name,
            "train_path": project.train_path,
            "index_file": project.index_file,
            "metadata_file": project.metadata_file
        }
        session.index = None
        session.image_paths = []
        return False

# --- Background Task ---
def process_build_index(project_id: str, train_path: str, index_file: str, metadata_file: str, username: str):
    global indexing_states
    
    state = {
        "is_indexing": True,
        "progress": 0,
        "total_files": 0,
        "processed_files": 0,
        "current_step": "Scanning directory..."
    }
    indexing_states[project_id] = state

    try:
        print(f"Starting index build for {project_id} (User: {username})")
        
        files = []
        for root, _, filenames in os.walk(train_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    relative_path = os.path.relpath(os.path.join(root, filename), train_path)
                    files.append(relative_path)
        
        if not files: raise Exception("No images found")

        state["total_files"] = len(files)
        state["current_step"] = "Extracting embeddings..."
        
        vectors = []
        valid_paths = []
        
        for idx, f in enumerate(files):
            full_path = os.path.join(train_path, f)
            try:
                vec = get_embedding(full_path)
                vectors.append(vec)
                valid_paths.append(f)
            except Exception as e:
                print(f"Error embedding {f}: {e}")
            
            processed = idx + 1
            state["processed_files"] = processed
            state["progress"] = int((processed / len(files)) * 90)
        
        if not vectors: raise Exception("No vectors extracted")

        state["current_step"] = "Building Index..."
        dataset_vectors = np.array(vectors)
        new_index = faiss.IndexFlatL2(dataset_vectors.shape[1])
        new_index.add(dataset_vectors)
        
        state["current_step"] = "Saving data..."
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

        faiss.write_index(new_index, index_file)
        with open(metadata_file, "w") as f:
            json.dump(valid_paths, f)
            
        print("Index build complete.")

        # If the user is currently looking at this project, reload it into their session
        if username in user_sessions:
            session = user_sessions[username]
            if session.current_project and session.current_project['id'] == project_id:
                session.index = new_index
                session.image_paths = valid_paths

    except Exception as e:
        print(f"Indexing Failed: {e}")
        state["current_step"] = f"Error: {str(e)}"
    finally:
        state["progress"] = 100
        state["is_indexing"] = False

# --- Auth Endpoints ---

@app.post("/auth/register")
async def register(
    username: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    existing_user = db.query(UserModel).filter(UserModel.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(password)
    new_user = UserModel(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"status": "success", "message": "User created"}

@app.post("/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(UserModel).filter(UserModel.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

@app.post("/auth/change-password")
async def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Re-fetch user from DB to ensure attached to session (get_current_user usually attaches, but being safe)
    user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
    
    if not verify_password(old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
        
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    
    return {"status": "success", "message": "Password updated successfully"}

@app.get("/users/me")
async def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return {"username": current_user.username}

# --- Project Endpoints ---

@app.on_event("startup")
async def startup_event():
    # Initialize DB Tables
    init_db()
    load_model()

@app.get("/projects")
def get_projects(current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    # Filter projects owned by user via Relationship
    # Convert DB objects to list of dicts/schemas
    projects = []
    for p in current_user.projects:
        projects.append({
            "id": p.id,
            "name": p.name,
            "train_path": p.train_path,
            "index_file": p.index_file,
            "metadata_file": p.metadata_file,
            "created_at": p.created_at,
            "is_default": p.is_default,
            "owner": current_user.username
        })
    return projects

@app.post("/projects/create")
async def create_project(
    name: str = Form(...), 
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    username = current_user.username
    
    # Check if first project
    is_first = len(current_user.projects) == 0

    project_id = str(uuid.uuid4())[:8]
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
    if not safe_name: safe_name = project_id
        
    project_train_path = os.path.join(UPLOADS_DIR, f"{username}_{safe_name}_{project_id}")
    os.makedirs(project_train_path, exist_ok=True)
    
    saved_count = 0
    ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in ALLOWED:
            file_path = os.path.join(project_train_path, os.path.basename(file.filename))
            with open(file_path, "wb+") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
            
    if saved_count == 0:
        shutil.rmtree(project_train_path)
        raise HTTPException(status_code=400, detail="No valid images")

    index_file = os.path.join(PROJECTS_DIR, project_id, "vector.index")
    metadata_file = os.path.join(PROJECTS_DIR, project_id, "paths.json")

    # Save to Database
    new_project = ProjectModel(
        id=project_id,
        name=name,
        train_path=os.path.abspath(project_train_path),
        index_file=index_file,
        metadata_file=metadata_file,
        created_at=time.time(),
        is_default=is_first,
        owner_id=current_user.id
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    # Auto switch if it's the first project (In Memory)
    if is_first:
        session = user_sessions[username]
        session.current_project = {
            "id": new_project.id,
            "name": new_project.name,
            "train_path": new_project.train_path,
            "index_file": new_project.index_file,
            "metadata_file": new_project.metadata_file
        }
        session.index = None
        session.image_paths = []

    background_tasks.add_task(process_build_index, project_id, new_project.train_path, index_file, metadata_file, username)

    # Convert to dict for response
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

@app.post("/projects/switch")
def switch_project(
    project_id: str = Form(...), 
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    username = current_user.username
    
    target = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, 
        ProjectModel.owner_id == current_user.id
    ).first()
    
    if not target:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if indexing
    if project_id in indexing_states and indexing_states[project_id]["is_indexing"]:
         raise HTTPException(status_code=400, detail="Project is currently indexing")

    success = load_project_data(username, target)
    
    # Convert to dict for response
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
def set_default_project(
    project_id: str = Form(...), 
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Set all user's projects default to false
    db.query(ProjectModel).filter(ProjectModel.owner_id == current_user.id).update({ProjectModel.is_default: False})
    
    # Set target to true
    target = db.query(ProjectModel).filter(ProjectModel.id == project_id, ProjectModel.owner_id == current_user.id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Project not found")
    
    target.is_default = True
    db.commit()
    
    return {"status": "success", "message": f"Project '{target.name}' set as default."}

@app.get("/indexing_status")
def get_indexing_status(current_user: UserModel = Depends(get_current_user)):
    username = current_user.username
    session = user_sessions.get(username)
    
    if session and session.current_project:
        proj_id = session.current_project["id"]
        if proj_id in indexing_states:
            return indexing_states[proj_id]
            
    return {"is_indexing": False}

@app.get("/status")
def get_status(current_user: UserModel = Depends(get_current_user)):
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session:
        return {"status": "loading", "index_size": 0, "current_project": "None"}
        
    proj_id = session.current_project["id"] if session.current_project else None
    
    status_text = "ready"
    if proj_id and proj_id in indexing_states and indexing_states[proj_id]["is_indexing"]:
        status_text = "indexing"
    elif session.index is None:
        status_text = "loading" if session.current_project else "no_project"
        
    return {
        "status": status_text,
        "index_size": session.index.ntotal if session.index else 0,
        "current_project": session.current_project["name"] if session.current_project else "None",
        "train_path": session.current_project["train_path"] if session.current_project else ""
    }

@app.get("/serve_image/{filename:path}")
def serve_image(filename: str):
    if os.path.exists(filename): # Absolute path check
        return FileResponse(filename)
    
    # Try finding in uploads dir if path is relative
    possible_path = os.path.join(UPLOADS_DIR, filename) 
    if os.path.exists(possible_path):
        return FileResponse(possible_path)
        
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/database")
def get_database(request: Request, current_user: UserModel = Depends(get_current_user)):
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session or not session.image_paths or not session.current_project:
        return []
    
    base_url = str(request.base_url)
    results = []
    # session.image_paths stores relative paths inside the train_path
    train_path = session.current_project["train_path"]
    
    for filename in session.image_paths:
        full_path = os.path.join(train_path, filename)
        # We serve using the absolute path for simplicity in serve_image
        results.append({
            "filename": filename,
            "url": f"{base_url}serve_image/{full_path}" 
        })
    return results

@app.get("/visualize")
def get_visualization(request: Request, current_user: UserModel = Depends(get_current_user)):
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session or not session.current_project:
         return {"error": "No project loaded."}
    
    if session.index is None or session.index.ntotal < 3:
        return {"error": "Not enough data (min 3 images)."}

    vectors = session.index.reconstruct_n(0, session.index.ntotal)
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:3]
        projection = np.dot(centered, components.T)
    except:
        return {"error": "PCA failed"}

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
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session or not session.index or not session.current_project:
        raise HTTPException(status_code=503, detail="Project not ready")

    filename = file.filename
    save_path = os.path.join(session.current_project["train_path"], filename)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

    try:
        vec = get_embedding(save_path)
        vec = vec.reshape(1, -1)
        session.index.add(vec)
        session.image_paths.append(filename)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Index update failed: {e}")

    # Persistence
    faiss.write_index(session.index, session.current_project["index_file"])
    with open(session.current_project["metadata_file"], "w") as f:
        json.dump(session.image_paths, f)

    return {"status": "success", "index_size": session.index.ntotal}

@app.post("/delete")
def delete_image(filename: str = Form(...), current_user: UserModel = Depends(get_current_user)):
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session or not session.index:
        raise HTTPException(status_code=503, detail="Not ready")
        
    if filename not in session.image_paths:
        raise HTTPException(status_code=404, detail="Image not found")

    idx = session.image_paths.index(filename)
    session.index.remove_ids(np.array([idx], dtype='int64'))
    session.image_paths.pop(idx)

    faiss.write_index(session.index, session.current_project["index_file"])
    with open(session.current_project["metadata_file"], "w") as f:
        json.dump(session.image_paths, f)

    return {"status": "deleted", "index_size": session.index.ntotal}

@app.post("/search")
async def search_image(
    request: Request, 
    k: int = Form(5), 
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    username = current_user.username
    session = user_sessions.get(username)
    
    if not session or not session.current_project:
        raise HTTPException(status_code=503, detail="No project selected")
        
    if not session.index or not session.image_paths:
        raise HTTPException(status_code=503, detail="Index empty")

    start_time = time.time()
    try:
        contents = await file.read()
        import io
        image_data = Image.open(io.BytesIO(contents))
        query_vector = get_embedding(image_data).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")

    search_k = min(k, len(session.image_paths))
    distances, indices = session.index.search(query_vector, k=search_k)

    results = []
    base_url = str(request.base_url)
    train_path = session.current_project["train_path"]

    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        filename = session.image_paths[idx]
        full_path = os.path.join(train_path, filename)
        results.append({
            "filename": filename,
            "filepath": full_path,
            "url": f"{base_url}serve_image/{full_path}",
            "distance": float(distances[0][i])
        })

    return {"results": results, "query_time": time.time() - start_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

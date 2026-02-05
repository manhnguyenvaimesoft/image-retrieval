import os
import time
import json
import shutil
import uuid
import threading
import numpy as np
import faiss
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image

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

# --- Global State & Configuration ---
# REMOVED: CONFIG_PATH = "config.yaml"
PROJECTS_FILE = "projects.json"
PROJECTS_DIR = "projects_data"  # Folder to store indices for new projects
UPLOADS_DIR = "uploads" # Folder to store user uploaded images

# Global Variables
index: Optional[faiss.IndexFlatL2] = None
image_paths: List[str] = []
model: Optional[YOLO] = None
current_project: Optional[Dict] = None

YOLO_MODEL_PATH = "./weights/yolo26x-cls.pt"

# Indexing State for Progress Bar
indexing_state = {
    "is_indexing": False,
    "progress": 0,
    "total_files": 0,
    "processed_files": 0,
    "current_step": "idle" # idle, loading_images, embedding, saving, complete
}

# Ensure directories exist
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def load_projects_list():
    if not os.path.exists(PROJECTS_FILE):
        return []
    try:
        with open(PROJECTS_FILE, "r") as f:
            projects = json.load(f)
            # Ensure compatibility if file exists but is empty or malformed
            if not isinstance(projects, list):
                return []
            return projects
    except Exception:
        return []

def save_projects_list(projects):
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=2)

def initialize_projects():
    """Ensures projects.json exists"""
    if not os.path.exists(PROJECTS_FILE):
        save_projects_list([])
    return load_projects_list()

def load_model():
    global model
    # Hardcoded default or use Environment Variable
    model_path = os.environ.get("YOLO_MODEL_PATH", YOLO_MODEL_PATH)

    print(f"Loading YOLO model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}. Trying generic 'yolov8n-cls.pt'")
        try:
            model = YOLO('yolov8n-cls.pt')
        except Exception as e2:
            print(f"CRITICAL: Failed to load any YOLO model: {e2}")
            model = None

def get_embedding(source):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = model.embed(source)
    return results[0].cpu().numpy().astype('float32')

# --- Background Task for Indexing ---
def process_build_index(project_id: str, train_path: str, index_file: str, metadata_file: str):
    global indexing_state, index, image_paths, current_project

    try:
        indexing_state["is_indexing"] = True
        indexing_state["progress"] = 0
        indexing_state["current_step"] = "Scanning directory..."
        
        print(f"Starting background index build for: {train_path}")
        
        if not os.path.exists(train_path):
            raise Exception(f"Train path does not exist: {train_path}")

        # 1. Scan Files
        files = []
        for root, _, filenames in os.walk(train_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    relative_path = os.path.relpath(os.path.join(root, filename), train_path)
                    files.append(relative_path)
        
        if not files:
            raise Exception("No images found in the specified directory.")

        indexing_state["total_files"] = len(files)
        indexing_state["current_step"] = "Extracting embeddings..."
        
        vectors = []
        valid_paths = []
        
        # 2. Extract Embeddings
        for idx, f in enumerate(files):
            full_path = os.path.join(train_path, f)
            try:
                vec = get_embedding(full_path)
                vectors.append(vec)
                valid_paths.append(f)
            except Exception as e:
                print(f"Error embedding {f}: {e}")
            
            # Update progress
            processed = idx + 1
            indexing_state["processed_files"] = processed
            indexing_state["progress"] = int((processed / len(files)) * 90) # Up to 90%
        
        if not vectors:
            raise Exception("Failed to extract vectors from any image.")

        # 3. Build FAISS
        indexing_state["current_step"] = "Building Index..."
        dataset_vectors = np.array(vectors)
        dimension = dataset_vectors.shape[1]
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(dataset_vectors)
        
        # 4. Save
        indexing_state["current_step"] = "Saving data..."
        
        # Ensure directory for index file exists
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

        faiss.write_index(new_index, index_file)
        with open(metadata_file, "w") as f:
            json.dump(valid_paths, f)
            
        print("Index build complete.")

        # Auto-load if this is the current project
        if current_project and current_project['id'] == project_id:
            index = new_index
            image_paths = valid_paths

    except Exception as e:
        print(f"Indexing Failed: {e}")
        indexing_state["current_step"] = f"Error: {str(e)}"
    finally:
        indexing_state["progress"] = 100
        indexing_state["is_indexing"] = False

def load_project_data(project):
    global index, image_paths, current_project
    
    print(f"Loading project: {project['name']}")
    
    # Verify train path
    if not os.path.exists(project['train_path']):
         print(f"Warning: Project path {project['train_path']} missing.")
    
    if os.path.exists(project['index_file']) and os.path.exists(project['metadata_file']):
        try:
            index = faiss.read_index(project['index_file'])
            with open(project['metadata_file'], "r") as f:
                image_paths = json.load(f)
            current_project = project
            print(f"Project loaded. {index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"Error loading project data: {e}")
            return False
    else:
        print("Project index missing. Needs building.")
        current_project = project
        index = None
        image_paths = []
        return False

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    load_model()
    projects = initialize_projects()
    
    # Logic: Only load if there is a project marked as default
    default_project = next((p for p in projects if p.get("is_default") is True), None)
    
    if default_project:
        print(f"Startup: Loading default project '{default_project['name']}'")
        load_project_data(default_project)
    else:
        print("Startup: No default project set. Waiting for user selection.")

@app.get("/projects")
def get_projects():
    return load_projects_list()

@app.post("/projects/create")
async def create_project(
    name: str = Form(...), 
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    if indexing_state["is_indexing"]:
        raise HTTPException(status_code=400, detail="Another indexing process is running.")

    projects = load_projects_list()
    
    # Check if this is the first project ever
    is_first_project = len(projects) == 0

    project_id = str(uuid.uuid4())[:8]
    
    # Create a safe directory name for the project
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
    if not safe_name:
        safe_name = project_id
        
    project_train_path = os.path.join(UPLOADS_DIR, safe_name)
    
    # Ensure project upload directory exists
    if not os.path.exists(project_train_path):
        os.makedirs(project_train_path, exist_ok=True)
    
    # Save uploaded files
    saved_count = 0
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    try:
        for file in files:
            # Check extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue

            # We use just the filename to flatten structure for simplicity
            file_path = os.path.join(project_train_path, os.path.basename(file.filename))
            with open(file_path, "wb+") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

    if saved_count == 0:
        try:
            os.rmdir(project_train_path)
        except:
            pass
        raise HTTPException(status_code=400, detail="No valid image files found in the selection.")

    # Define storage paths inside PROJECTS_DIR (metadata/indices)
    index_file = os.path.join(PROJECTS_DIR, project_id, "vector.index")
    metadata_file = os.path.join(PROJECTS_DIR, project_id, "paths.json")

    new_project = {
        "id": project_id,
        "name": name,
        "train_path": os.path.abspath(project_train_path),
        "index_file": index_file,
        "metadata_file": metadata_file,
        "created_at": time.time(),
        "is_default": is_first_project # Set as default if it's the only one
    }

    # Save to list
    projects.append(new_project)
    save_projects_list(projects)

    # Automatically load this project if it's the first one or if no project is currently loaded
    global current_project
    if is_first_project or current_project is None:
        current_project = new_project
        # Note: Index will be loaded/created by the background task
        
    # Start Indexing in Background
    background_tasks.add_task(process_build_index, project_id, new_project["train_path"], index_file, metadata_file)

    return {"status": "started", "project": new_project, "file_count": saved_count}

@app.post("/projects/switch")
def switch_project(project_id: str = Form(...)):
    if indexing_state["is_indexing"]:
        raise HTTPException(status_code=400, detail="Cannot switch while indexing.")

    projects = load_projects_list()
    target = next((p for p in projects if p["id"] == project_id), None)
    
    if not target:
        raise HTTPException(status_code=404, detail="Project not found")
    
    success = load_project_data(target)
    
    return {
        "status": "success" if success else "needs_indexing", 
        "project": target,
        "message": "Switched successfully" if success else "Project switched but index missing"
    }

@app.post("/projects/set_default")
def set_default_project(project_id: str = Form(...)):
    projects = load_projects_list()
    target = next((p for p in projects if p["id"] == project_id), None)
    
    if not target:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update default flags
    for p in projects:
        p["is_default"] = (p["id"] == project_id)
        
    save_projects_list(projects)
    
    return {"status": "success", "message": f"Project '{target['name']}' set as default."}

@app.get("/indexing_status")
def get_indexing_status():
    return indexing_state

@app.get("/status")
def get_status():
    status = "ready"
    if indexing_state["is_indexing"]:
        status = "indexing"
    elif index is None:
        status = "loading" # or empty
        
    return {
        "status": status,
        "index_size": index.ntotal if index else 0,
        "current_project": current_project["name"] if current_project else "None",
        "train_path": current_project["train_path"] if current_project else ""
    }

# Dynamic Image Serving
@app.get("/serve_image/{filename:path}")
def serve_image(filename: str):
    if not current_project:
        raise HTTPException(status_code=503, detail="No project loaded. Please select a project.")
    
    file_path = os.path.join(current_project["train_path"], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/database")
def get_database(request: Request):
    """Return all images in the database"""
    if not image_paths or not current_project:
        return []
    
    base_url = str(request.base_url)
    results = []
    for filename in image_paths:
        results.append({
            "filename": filename,
            "url": f"{base_url}serve_image/{filename}"
        })
    return results

@app.get("/visualize")
def get_visualization(request: Request):
    global index, image_paths, current_project
    
    if not current_project:
         return {"error": "No project loaded. Please select a project."}

    if index is None or index.ntotal < 3:
        return {"error": "Not enough data to visualize (need at least 3 images)"}

    # Reconstruct vectors
    vectors = index.reconstruct_n(0, index.ntotal)
    
    # PCA
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:3]
        projection = np.dot(centered, components.T)
    except Exception as e:
        return {"error": f"Failed to compute visualization: {e}"}

    points = []
    base_url = str(request.base_url)
    
    for i, path in enumerate(image_paths):
        points.append({
            "filename": path,
            "url": f"{base_url}serve_image/{path}",
            "x": float(projection[i, 0]),
            "y": float(projection[i, 1]),
            "z": float(projection[i, 2])
        })
        
    return {"points": points}

@app.post("/add")
async def add_to_index(file: UploadFile = File(...)):
    global index, image_paths, current_project

    if index is None or not current_project:
        raise HTTPException(status_code=503, detail="Index/Project not initialized")

    filename = file.filename
    # Check for duplicate filenames roughly? For now just overwrite
    save_path = os.path.join(current_project["train_path"], filename)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        vec = get_embedding(save_path)
        vec = vec.reshape(1, -1)
    except Exception as e:
        if os.path.exists(save_path): os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        index.add(vec)
        image_paths.append(filename)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to update index: {e}")

    # Persistence
    try:
        faiss.write_index(index, current_project["index_file"])
        with open(current_project["metadata_file"], "w") as f:
            json.dump(image_paths, f)
    except Exception as e:
        print(f"Warning: Failed to save index to disk: {e}")

    return {
        "status": "success", 
        "filename": filename, 
        "index_size": index.ntotal,
        "message": "Image added to database successfully"
    }

@app.post("/delete")
def delete_image(filename: str = Form(...)):
    global index, image_paths, current_project
    
    if not current_project:
        raise HTTPException(status_code=503, detail="No project loaded")
        
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    if filename not in image_paths:
        raise HTTPException(status_code=404, detail="Image not found in index")

    try:
        # 1. Find index
        idx = image_paths.index(filename)

        # 2. Remove from FAISS
        # Note: removing from flat index shifts subsequent IDs
        index.remove_ids(np.array([idx], dtype='int64'))

        # 3. Remove from Metadata List
        image_paths.pop(idx)

        # 4. Remove File from Disk -> SKIPPED based on user request to keep physical file
        # full_path = os.path.join(current_project["train_path"], filename)
        # if os.path.exists(full_path):
        #    os.remove(full_path)

        # 5. Persist Changes
        faiss.write_index(index, current_project["index_file"])
        with open(current_project["metadata_file"], "w") as f:
            json.dump(image_paths, f)

        return {"status": "deleted", "index_size": index.ntotal}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.post("/search")
async def search_image(request: Request, k: int = Form(5), file: UploadFile = File(...)):
    global index, image_paths, current_project
    
    if not current_project:
        raise HTTPException(status_code=503, detail="No project selected.")
        
    if index is None or not image_paths:
        # It's possible to have a project but no index yet (indexing or empty)
        raise HTTPException(status_code=503, detail="Index is empty or not ready.")

    start_time = time.time()
    try:
        contents = await file.read()
        import io
        image_data = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        query_vector = get_embedding(image_data)
        query_vector = query_vector.reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

    search_k = min(k, len(image_paths))
    distances, indices = index.search(query_vector, k=search_k)

    results = []
    base_url = str(request.base_url)

    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        
        filename = image_paths[idx]
        results.append({
            "filename": filename,
            "filepath": os.path.join(current_project["train_path"], filename),
            # Use new serve_image endpoint
            "url": f"{base_url}serve_image/{filename}",
            "distance": float(distances[0][i])
        })

    elapsed = time.time() - start_time
    
    return {
        "results": results,
        "query_time": elapsed
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

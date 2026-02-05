import os
import time
import json
import yaml
import shutil
import asyncio
import numpy as np
import faiss
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import concurrent.futures

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

# Configuration and Paths
PROJECTS_ROOT = "projects"
PROJECTS_CONFIG_FILE = os.path.join(PROJECTS_ROOT, "projects.json")
CONFIG_PATH = "config.yaml"

if not os.path.exists(PROJECTS_ROOT):
    os.makedirs(PROJECTS_ROOT, exist_ok=True)

# Global Variables
index: Optional[faiss.IndexFlatL2] = None
image_paths: List[str] = []
model: Optional[YOLO] = None
active_project_name: Optional[str] = None
active_project_info: Dict = {}
build_progress: Dict[str, Dict] = {} # project_name -> {current, total, status, message}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        except:
            pass
    return {}

def load_projects_metadata():
    if os.path.exists(PROJECTS_CONFIG_FILE):
        try:
            with open(PROJECTS_CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"active_project": "default", "projects": {}}

def save_projects_metadata(metadata):
    with open(PROJECTS_CONFIG_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def get_project_paths(project_name: str):
    project_dir = os.path.join(PROJECTS_ROOT, project_name)
    return {
        "dir": project_dir,
        "index": os.path.join(project_dir, "vector.index"),
        "metadata": os.path.join(project_dir, "image_paths.json"),
        "config": os.path.join(project_dir, "config.json")
    }

def load_model():
    global model
    config = load_config()
    # Priority for model path
    model_path = config.get("model", {}).get("path", "./weights/yolov8n-cls.pt")
    if not os.path.exists(model_path):
        # Check backend/weights if running from root
        alt_path = os.path.join("backend", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    print(f"Loading YOLO model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Trying generic 'yolov8n-cls.pt'")
        try:
            model = YOLO('yolov8n-cls.pt')
        except:
            print("Failed to load any model.")

def get_embedding(source):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = model.embed(source)
    return results[0].cpu().numpy().astype('float32')

def build_index_sync(project_name: str, train_path: str):
    """Synchronous indexing function to be run in a thread."""
    global index, image_paths, active_project_name, active_project_info
    
    paths = get_project_paths(project_name)
    os.makedirs(paths["dir"], exist_ok=True)
    
    build_progress[project_name] = {"current": 0, "total": 0, "status": "scanning", "message": "Scanning directory..."}
    
    files = []
    for root, _, filenames in os.walk(train_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                relative_path = os.path.relpath(os.path.join(root, filename), train_path)
                files.append(relative_path)

    total = len(files)
    if total == 0:
        build_progress[project_name] = {"current": 0, "total": 0, "status": "failed", "message": "No images found in path."}
        return

    build_progress[project_name] = {"current": 0, "total": total, "status": "processing", "message": f"Embedding {total} images..."}

    vectors = []
    valid_paths = []
    
    for idx, f in enumerate(files):
        full_path = os.path.join(train_path, f)
        try:
            vec = get_embedding(full_path)
            vectors.append(vec)
            valid_paths.append(f)
        except Exception as e:
            print(f"Error processing {f}: {e}")
        
        build_progress[project_name]["current"] = idx + 1
        # No need to sleep in a separate thread, but we can if we want to yield
        # time.sleep(0.001)

    if not vectors:
        build_progress[project_name]["status"] = "failed"
        build_progress[project_name]["message"] = "Failed to extract any vectors."
        return

    dataset_vectors = np.array(vectors)
    dimension = dataset_vectors.shape[1]
    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(dataset_vectors)

    faiss.write_index(new_index, paths["index"])
    with open(paths["metadata"], "w") as f:
        json.dump(valid_paths, f)
    
    with open(paths["config"], "w") as f:
        json.dump({"train_path": train_path}, f)

    build_progress[project_name]["status"] = "completed"
    build_progress[project_name]["message"] = f"Successfully indexed {len(valid_paths)} images."

    # If this is the active project, reload it into memory
    if active_project_name == project_name:
        index = new_index
        image_paths = valid_paths
        active_project_info["train_path"] = train_path

def switch_project(project_name: str):
    global index, image_paths, active_project_name, active_project_info
    
    metadata = load_projects_metadata()
    if project_name not in metadata["projects"]:
        raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
    
    paths = get_project_paths(project_name)
    
    # Update active name immediately so UI knows what's selected
    active_project_name = project_name
    metadata["active_project"] = project_name
    save_projects_metadata(metadata)
    
    if os.path.exists(paths["index"]) and os.path.exists(paths["metadata"]):
        try:
            index = faiss.read_index(paths["index"])
            with open(paths["metadata"], "r") as f:
                image_paths = json.load(f)
            
            if os.path.exists(paths["config"]):
                with open(paths["config"], "r") as f:
                    active_project_info = json.load(f)
            else:
                active_project_info = {"train_path": metadata["projects"][project_name].get("train_path", "")}
            return True
        except Exception as e:
            print(f"Error loading index for {project_name}: {e}")
            index = None
            image_paths = []
            return False
    else:
        index = None
        image_paths = []
        if os.path.exists(paths["config"]):
            with open(paths["config"], "r") as f:
                active_project_info = json.load(f)
        else:
             active_project_info = {"train_path": metadata["projects"][project_name].get("train_path", "")}
        return False

@app.on_event("startup")
async def startup_event():
    load_model()
    metadata = load_projects_metadata()
    
    # Initialize default project if none exists
    if not metadata["projects"]:
        config = load_config()
        # Try to find a reasonable default path
        default_train_path = config.get("dataset", {}).get("train_path", "./train_data")
        if not os.path.isabs(default_train_path):
             # Try relative to script
             script_dir = os.path.dirname(os.path.abspath(__file__))
             abs_path = os.path.join(script_dir, default_train_path)
             if os.path.exists(abs_path):
                  default_train_path = abs_path

        metadata["projects"]["default"] = {
            "name": "default",
            "train_path": default_train_path,
            "created_at": time.time()
        }
        metadata["active_project"] = "default"
        save_projects_metadata(metadata)
    
    active = metadata.get("active_project", "default")
    if active not in metadata["projects"]:
         active = list(metadata["projects"].keys())[0]

    switch_project(active)

@app.get("/projects")
def list_projects():
    metadata = load_projects_metadata()
    return {
        "projects": metadata["projects"],
        "active_project": active_project_name,
        "build_progress": build_progress
    }

@app.post("/projects")
async def create_project(background_tasks: BackgroundTasks, name: str = Form(...), train_path: str = Form(...)):
    metadata = load_projects_metadata()
    if name in metadata["projects"]:
        raise HTTPException(status_code=400, detail="Project name already exists")
    
    if not os.path.exists(train_path):
         raise HTTPException(status_code=400, detail=f"Path does not exist: {train_path}")

    metadata["projects"][name] = {
        "name": name,
        "train_path": train_path,
        "created_at": time.time()
    }
    save_projects_metadata(metadata)
    
    # Run in background thread to avoid blocking
    background_tasks.add_task(build_index_sync, name, train_path)
    
    return {"message": f"Project {name} created", "project": metadata["projects"][name]}

@app.post("/projects/select")
def select_project(name: str = Form(...)):
    if switch_project(name):
        return {"message": f"Switched to {name}", "status": "ready"}
    else:
        return {"message": f"Switched to {name}, indexing in progress", "status": "indexing"}

@app.delete("/projects/{name}")
def delete_project(name: str):
    metadata = load_projects_metadata()
    if name not in metadata["projects"]:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default project")
        
    paths = get_project_paths(name)
    if os.path.exists(paths["dir"]):
        shutil.rmtree(paths["dir"])
    
    del metadata["projects"][name]
    if name in build_progress:
        del build_progress[name]
    
    if active_project_name == name:
        metadata["active_project"] = "default"
        save_projects_metadata(metadata)
        switch_project("default")
    else:
        save_projects_metadata(metadata)
        
    return {"message": f"Project {name} deleted"}

@app.get("/images/{project_name}/{filename:path}")
def serve_image(project_name: str, filename: str):
    metadata = load_projects_metadata()
    if project_name not in metadata["projects"]:
         raise HTTPException(status_code=404, detail="Project not found")
    
    train_path = metadata["projects"][project_name].get("train_path", "")
    # Try to get from config.json if available for more accuracy
    paths = get_project_paths(project_name)
    if os.path.exists(paths["config"]):
        try:
            with open(paths["config"], "r") as f:
                train_path = json.load(f).get("train_path", train_path)
        except: pass

    full_path = os.path.join(train_path, filename)
    if not os.path.exists(full_path):
        # Fallback: check if it's relative to backend
        alt_path = os.path.join(os.path.dirname(__file__), train_path, filename)
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    return FileResponse(full_path)

@app.get("/status")
def get_status():
    return {
        "status": "ready" if index is not None else "loading",
        "active_project": active_project_name,
        "index_size": index.ntotal if index else 0,
        "train_path": active_project_info.get("train_path", ""),
        "indexing_active": any(p.get("status") in ["scanning", "processing"] for p in build_progress.values())
    }

@app.get("/database")
def get_database(request: Request):
    if not image_paths or not active_project_name:
        return []
    
    base_url = str(request.base_url)
    return [{"filename": f, "url": f"{base_url}images/{active_project_name}/{f}"} for f in image_paths]

@app.get("/visualize")
def get_visualization(request: Request):
    global index, image_paths
    if index is None or index.ntotal < 3:
        return {"error": "Not enough data (min 3 images)"}

    vectors = index.reconstruct_n(0, index.ntotal)
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projection = np.dot(centered, Vt[:3].T)
    except Exception as e:
        return {"error": str(e)}

    base_url = str(request.base_url)
    points = []
    for i, path in enumerate(image_paths):
        points.append({
            "filename": path,
            "url": f"{base_url}images/{active_project_name}/{path}",
            "x": float(projection[i, 0]),
            "y": float(projection[i, 1]),
            "z": float(projection[i, 2])
        })
    return {"points": points}

@app.post("/add")
async def add_to_index(file: UploadFile = File(...)):
    global index, image_paths
    if index is None:
        raise HTTPException(status_code=503, detail="Index not ready")

    train_path = active_project_info.get("train_path")
    if not train_path or not os.path.exists(train_path):
        raise HTTPException(status_code=500, detail="Invalid train path")

    filename = file.filename
    save_path = os.path.join(train_path, filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        vec = get_embedding(save_path).reshape(1, -1)
        index.add(vec)
        image_paths.append(filename)
        
        paths = get_project_paths(active_project_name)
        faiss.write_index(index, paths["index"])
        with open(paths["metadata"], "w") as f:
            json.dump(image_paths, f)
            
        return {"status": "success", "index_size": index.ntotal}
    except Exception as e:
        if os.path.exists(save_path): os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_image(request: Request, k: int = Form(5), file: UploadFile = File(...)):
    global index, image_paths
    if index is None or not image_paths:
        raise HTTPException(status_code=503, detail="Index not ready")

    try:
        contents = await file.read()
        import io
        image_data = Image.open(io.BytesIO(contents))
        query_vector = get_embedding(image_data).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")

    search_k = min(k, len(image_paths))
    distances, indices = index.search(query_vector, k=search_k)

    base_url = str(request.base_url)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        filename = image_paths[idx]
        results.append({
            "filename": filename,
            "url": f"{base_url}images/{active_project_name}/{filename}",
            "distance": float(distances[0][i])
        })
    return {"results": results}

@app.get("/")
def home():
    return {"message": "NeuroSearch API Active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

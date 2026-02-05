import os
import time
import json
import yaml
import shutil
import numpy as np
import faiss
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image

# Initialize FastAPI
app = FastAPI(title="NeuroSearch API")

# CORS Setup (Allow Frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Configuration
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    # Fallback default if file missing
    config = {
        "dataset": {"train_path": "./train_data"},
        "storage": {"index_file": "vector.index", "metadata_file": "image_paths.json"},
        "model": {"name": "yolov8n-cls.pt"}
    }
else:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        # Storage is always set to default.
        project_base_path = config.get("project", {}).get("base_path", "./projects/default_project")
        config["storage"] = {"index_file": f"{project_base_path}/vector.index", "metadata_file": f"{project_base_path}/image_paths.json"}

# Global Variables
index: Optional[faiss.IndexFlatL2] = None
image_paths: List[str] = []
model: Optional[YOLO] = None

# Ensure Dataset Directory Exists for Serving
TRAIN_PATH = config["dataset"]["train_path"]
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH, exist_ok=True)

# Mount static files to serve images to frontend
# Access images via http://localhost:8000/images/filename.png
app.mount("/images", StaticFiles(directory=TRAIN_PATH), name="images")

def load_model():
    global model
    print(f"Loading YOLO model: {config['model']['path']}...")
    try:
        model = YOLO(config['model']['path'])
    except Exception as e:
        print(f"Error loading model: {e}. Trying generic 'yolov8n-cls.pt'")
        model = YOLO('./weights/yolov8n-cls.pt')

def get_embedding(source):
    """
    Extract feature vector using YOLO. 
    Source can be a file path or a PIL Image.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # YOLO embed returns a list of tensors
    results = model.embed(source)
    # Return first result as numpy array (float32 for FAISS)
    return results[0].cpu().numpy().astype('float32')

def build_index():
    global index, image_paths
    
    print("Building new index from scratch...")
    if not os.path.exists(TRAIN_PATH):
        print(f"Warning: Train path {TRAIN_PATH} does not exist.")
        return

    # files = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # Instead of just browsing within a folder, use recursive browsing to retrieve all images in subfolders.
    files = []
    for root, _, filenames in os.walk(TRAIN_PATH):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Get relative path to TRAIN_PATH
                relative_path = os.path.relpath(os.path.join(root, filename), TRAIN_PATH)
                files.append(relative_path)

    if not files:
        print("No images found in train path.")
        return

    vectors = []
    valid_paths = []
    
    total = len(files)
    print(f"Processing {total} images...")

    for idx, f in enumerate(files):
        full_path = os.path.join(TRAIN_PATH, f)
        try:
            vec = get_embedding(full_path)
            vectors.append(vec)
            valid_paths.append(f) # Store filename relative to TRAIN_PATH
        except Exception as e:
            print(f"Error processing {f}: {e}")
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total}")

    if not vectors:
        print("Failed to extract any vectors.")
        return

    dataset_vectors = np.array(vectors)
    
    # Create FAISS Index
    dimension = dataset_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(dataset_vectors)
    image_paths = valid_paths

    # Save to disk
    print("Saving index and metadata...")
    faiss.write_index(index, config["storage"]["index_file"])
    with open(config["storage"]["metadata_file"], "w") as f:
        json.dump(image_paths, f)
    print("Index built and saved successfully.")

@app.on_event("startup")
async def startup_event():
    load_model()
    global index, image_paths
    
    index_file = config["storage"]["index_file"]
    meta_file = config["storage"]["metadata_file"]

    # Check if index exists
    if os.path.exists(index_file) and os.path.exists(meta_file):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_file)
        with open(meta_file, "r") as f:
            image_paths = json.load(f)
        print(f"Loaded {index.ntotal} vectors.")
    else:
        print("Index not found. Triggering build process...")
        build_index()

@app.get("/status")
def get_status():
    return {
        "status": "ready" if index is not None else "loading",
        "index_size": index.ntotal if index else 0,
        "train_path": TRAIN_PATH
    }

@app.get("/database")
def get_database(request: Request):
    """Return all images in the database"""
    if not image_paths:
        return []
    
    base_url = str(request.base_url)
    results = []
    for filename in image_paths:
        results.append({
            "filename": filename,
            "url": f"{base_url}images/{filename}"
        })
    # Reverse to show newest first if strictly appended, 
    # but build_index sorts by OS, so strictly speaking not chronological.
    return results

@app.get("/visualize")
def get_visualization(request: Request):
    """
    Reduce vector dimensions to 3D using PCA (SVD) for visualization.
    Returns x, y, z coordinates for every image.
    """
    global index, image_paths
    
    if index is None or index.ntotal < 3:
        return {"error": "Not enough data to visualize (need at least 3 images)"}

    # 1. Reconstruct all vectors from FAISS
    # IndexFlatL2 stores raw vectors, so we can reconstruct them.
    vectors = index.reconstruct_n(0, index.ntotal) # Shape: (N, D)
    
    # 2. Perform PCA using NumPy (SVD)
    # Center the data
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    # SVD
    # U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project to top 3 components
    # Projection = centered @ Vt.T[:, :3]
    try:
        # Using numpy's SVD is robust. 
        # We only need the top 3 principal components (columns of V, or rows of Vt)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:3] # Shape (3, D)
        projection = np.dot(centered, components.T) # Shape (N, 3)
        
        # Normalize to typical plotting range (-1 to 1 or similar) for cleaner charts if needed
        # But raw PCA scores are usually fine.
    except Exception as e:
        print(f"PCA Error: {e}")
        return {"error": f"Failed to compute visualization: {e}"}

    # 3. Format output
    points = []
    base_url = str(request.base_url)
    
    for i, path in enumerate(image_paths):
        points.append({
            "filename": path,
            "url": f"{base_url}images/{path}",
            "x": float(projection[i, 0]),
            "y": float(projection[i, 1]),
            "z": float(projection[i, 2])
        })
        
    return {"points": points}

@app.post("/add")
async def add_to_index(file: UploadFile = File(...)):
    """
    Upload a new image, save it to dataset, embed it, and add to FAISS index.
    """
    global index, image_paths

    if index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    # 1. Save file to TRAIN_PATH
    filename = file.filename
    # Handle duplicates by prepending timestamp if needed, but simple overwrite for now
    if not filename:
         raise HTTPException(status_code=400, detail="Filename missing")
         
    save_path = os.path.join(TRAIN_PATH, filename)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 2. Extract Embedding
    try:
        # Load the saved image to ensure it's valid and get embedding
        # We assume get_embedding handles path string
        vec = get_embedding(save_path)
        vec = vec.reshape(1, -1) # Reshape for FAISS (1, dim)
    except Exception as e:
        # If embedding fails, remove the file to keep state consistent
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3. Update Index in Memory
    try:
        index.add(vec)
        image_paths.append(filename)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to update index: {e}")

    # 4. Save Index and Metadata to Disk (Persistence)
    try:
        faiss.write_index(index, config["storage"]["index_file"])
        with open(config["storage"]["metadata_file"], "w") as f:
            json.dump(image_paths, f)
    except Exception as e:
        print(f"Warning: Failed to save index to disk: {e}")
        # We don't fail the request here, but data might be lost on restart

    return {
        "status": "success", 
        "filename": filename, 
        "index_size": index.ntotal,
        "message": "Image added to database successfully"
    }

@app.post("/search")
async def search_image(request: Request, k: int = Form(5), file: UploadFile = File(...)):
    global index, image_paths
    
    if index is None or not image_paths:
        raise HTTPException(status_code=503, detail="Index is not ready or empty.")

    # Read uploaded image
    start_time = time.time()
    try:
        contents = await file.read()
        # Save temp file or stream to PIL
        # Using PIL directly from bytes
        import io
        image_data = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Get query embedding
    try:
        query_vector = get_embedding(image_data)
        query_vector = query_vector.reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

    # Search
    # k might be larger than dataset
    search_k = min(k, len(image_paths))
    distances, indices = index.search(query_vector, k=search_k)

    results = []
    base_url = str(request.base_url) # Get base URL from request (e.g., http://192.168.1.35:8000/)

    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # FAISS padding
        
        filename = image_paths[idx]
        results.append({
            "filename": filename,
            "filepath": os.path.join(TRAIN_PATH, filename),
            # Use dynamic base_url so images load on LAN devices
            "url": f"{base_url}images/{filename}",
            "distance": float(distances[0][i])
        })

    elapsed = time.time() - start_time
    
    return {
        "results": results,
        "query_time": elapsed
    }

@app.get("/")
def home():
    return {"message": "Welcome to the NeuroSearch API"}

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" allows access from other devices on the network
    uvicorn.run(app, host="0.0.0.0", port=8000)

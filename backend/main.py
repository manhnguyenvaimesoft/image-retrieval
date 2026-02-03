import os
import time
import json
import yaml
import shutil
import numpy as np
import faiss
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
    print(f"Loading YOLO model: {config['model']['name']}...")
    try:
        model = YOLO(config['model']['name'])
    except Exception as e:
        print(f"Error loading model: {e}. Trying generic 'yolov8n-cls.pt'")
        model = YOLO('yolov8n-cls.pt')

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

    files = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
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

@app.post("/search")
async def search_image(k: int = Form(5), file: UploadFile = File(...)):
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
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # FAISS padding
        
        filename = image_paths[idx]
        results.append({
            "filename": filename,
            "filepath": os.path.join(TRAIN_PATH, filename),
            "url": f"http://localhost:8000/images/{filename}",
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

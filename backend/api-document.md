# NeuroSearch API Documentation

This API provides endpoints for managing image datasets, indexing them using YOLO embeddings, and performing semantic similarity searches.

**Base URL**: `http://localhost:8000`

---

## 1. Project Management

### Get All Projects
Retrieves a list of all available projects.

- **URL**: `/projects`
- **Method**: `GET`
- **Response**: Array of project objects.
```json
[
  {
    "id": "a1b2c3d4",
    "name": "My Dataset",
    "train_path": "/absolute/path/to/uploads/My_Dataset",
    "index_file": "projects_data/a1b2c3d4/vector.index",
    "metadata_file": "projects_data/a1b2c3d4/paths.json",
    "created_at": 1709823456.789,
    "is_default": true
  }
]
```

### Create Project
Uploads a folder of images, creates a new project, and starts the background indexing process.

- **URL**: `/projects/create`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `name` (Form String): Name of the project.
  - `files` (Form Files): List of image files to upload.
- **Response**:
```json
{
  "status": "started",
  "project": { ...project_details... },
  "file_count": 50
}
```

### Switch Project
Switches the active project and loads its FAISS index into memory.

- **URL**: `/projects/switch`
- **Method**: `POST`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Parameters**:
  - `project_id` (Form String): The ID of the project to switch to.
- **Response**:
```json
{
  "status": "success",
  "project": { ...project_details... },
  "message": "Switched successfully"
}
```

### Set Default Project
Marks a project as the default one to be loaded automatically on server startup.

- **URL**: `/projects/set_default`
- **Method**: `POST`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Parameters**:
  - `project_id` (Form String): The ID of the project to set as default.
- **Response**:
```json
{
  "status": "success",
  "message": "Project 'My Dataset' set as default."
}
```

---

## 2. Search

### Search Similar Images
Uploads a query image and returns the top K most similar images from the current project.

- **URL**: `/search`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (Form File): The query image.
  - `k` (Form Integer): Number of results to return (Default: 5).
- **Response**:
```json
{
  "results": [
    {
      "filename": "image_01.jpg",
      "filepath": "/path/to/image_01.jpg",
      "url": "http://localhost:8000/serve_image/image_01.jpg",
      "distance": 0.1234
    }
  ],
  "query_time": 0.045
}
```

---

## 3. Data Management

### Get Database Gallery
Returns a list of all images in the currently loaded project.

- **URL**: `/database`
- **Method**: `GET`
- **Response**:
```json
[
  {
    "filename": "cat.jpg",
    "url": "http://localhost:8000/serve_image/cat.jpg"
  }
]
```

### Add Image to Index
Uploads a single image, extracts its embedding, and adds it to the active FAISS index and metadata.

- **URL**: `/add`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (Form File): The image to add.
- **Response**:
```json
{
  "status": "success",
  "filename": "new_dog.jpg",
  "index_size": 150,
  "message": "Image added to database successfully"
}
```

### Delete Image
Removes an image from the FAISS index and metadata list (File remains on disk).

- **URL**: `/delete`
- **Method**: `POST`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Parameters**:
  - `filename` (Form String): The filename to remove (e.g., "cat.jpg").
- **Response**:
```json
{
  "status": "deleted",
  "index_size": 149
}
```

---

## 4. Visualization

### Get Vector Space (3D)
Returns PCA-reduced 3D coordinates (x, y, z) for all images in the index for visualization.

- **URL**: `/visualize`
- **Method**: `GET`
- **Response**:
```json
{
  "points": [
    {
      "filename": "cat.jpg",
      "url": "http://localhost:8000/serve_image/cat.jpg",
      "x": 1.2,
      "y": -0.5,
      "z": 0.3
    }
  ]
}
```

### Serve Image
Serves the raw image file from the server's storage.

- **URL**: `/serve_image/{filename}`
- **Method**: `GET`
- **Parameters**:
  - `filename` (Path String): The name of the file to retrieve.
- **Response**: Binary image file.

---

## 5. System Status

### General Status
Returns the current state of the backend (loading, indexing, ready) and loaded project info.

- **URL**: `/status`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "ready",
  "index_size": 150,
  "current_project": "My Dataset",
  "train_path": "/absolute/path/to/uploads/My_Dataset"
}
```

### Indexing Progress
Returns real-time progress of the background indexing task.

- **URL**: `/indexing_status`
- **Method**: `GET`
- **Response**:
```json
{
  "is_indexing": true,
  "progress": 45,
  "total_files": 100,
  "processed_files": 45,
  "current_step": "Extracting embeddings..."
}
```

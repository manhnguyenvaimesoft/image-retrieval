export interface SearchResult {
  filename: string;
  filepath: string;
  distance: number;
  url: string; // The serving URL from backend
}

export interface SearchResponse {
  results: SearchResult[];
  query_time: number;
}

export interface SystemStatus {
  status: 'loading' | 'ready' | 'error' | 'indexing';
  index_size: number;
  message: string;
  current_project?: string;
}

export interface Project {
  id: string;
  name: string;
  train_path: string;
  index_file: string;
  metadata_file: string;
  created_at: number;
}

export interface IndexingStatus {
  is_indexing: boolean;
  progress: number; // 0 to 100
  current_step: string;
  total_files: number;
  processed_files: number;
}
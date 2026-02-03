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
  status: 'loading' | 'ready' | 'error';
  index_size: number;
  message: string;
}
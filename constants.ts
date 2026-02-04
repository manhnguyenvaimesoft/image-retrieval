// Dynamically set the API URL based on the browser's hostname.
// This enables the app to connect to the backend whether accessed via 'localhost' 
// or a local network IP (e.g., '192.168.1.35'), assuming backend is on port 8000.
const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
export const API_BASE_URL = `http://${hostname}:8000`;

export const DEFAULT_K = 5;
export const MAX_K = 50;
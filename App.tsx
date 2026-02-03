import React, { useState, useEffect, useRef } from 'react';
import { Upload, Search, Image as ImageIcon, Settings, Loader2, AlertCircle } from 'lucide-react';
import { API_BASE_URL, DEFAULT_K, MAX_K } from './constants';
import { SearchResponse, SearchResult, SystemStatus } from './types';

// Utility for class names
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(DEFAULT_K);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check Backend Status on Mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/status`);
        if (!res.ok) throw new Error("Failed to connect to backend");
        const data = await res.json();
        setSystemStatus(data);
      } catch (err) {
        setSystemStatus({ status: 'error', index_size: 0, message: 'Backend unreachable' });
      }
    };
    checkStatus();
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]); // Clear previous results
      setError(null);
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) return;

    setIsSearching(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('k', topK.toString());

    try {
      const res = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Search failed: ${res.statusText}`);
      }

      const data: SearchResponse = await res.json();
      setResults(data.results);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch results. Ensure backend is running.");
    } finally {
      setIsSearching(false);
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="min-h-screen bg-dark text-slate-200 font-sans selection:bg-primary selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-dark/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary/20 p-2 rounded-lg">
              <Search className="w-6 h-6 text-primary" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-400">
              NeuroSearch
            </h1>
          </div>
          
          <div className="flex items-center gap-4 text-sm">
            {systemStatus?.status === 'ready' ? (
              <div className="flex items-center gap-2 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Index Active ({systemStatus.index_size} images)
              </div>
            ) : systemStatus?.status === 'error' ? (
              <div className="flex items-center gap-2 text-rose-400">
                 <AlertCircle className="w-4 h-4" />
                 Backend Offline
              </div>
            ) : (
              <div className="flex items-center gap-2 text-amber-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                Loading Index...
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Controls & Input */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Upload Card */}
          <div className="bg-surface rounded-2xl p-6 border border-slate-700 shadow-xl">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-slate-400" />
              Query Image
            </h2>

            <div 
              onClick={triggerFileUpload}
              className={cn(
                "border-2 border-dashed rounded-xl h-64 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 group relative overflow-hidden",
                previewUrl ? "border-primary/50 bg-dark" : "border-slate-600 hover:border-primary hover:bg-slate-800/50"
              )}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="image/*" 
                onChange={handleFileSelect} 
              />
              
              {previewUrl ? (
                <img 
                  src={previewUrl} 
                  alt="Query" 
                  className="w-full h-full object-contain p-2 z-10" 
                />
              ) : (
                <div className="text-center p-4 z-10">
                  <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                    <Upload className="w-6 h-6 text-slate-300" />
                  </div>
                  <p className="text-slate-300 font-medium">Click to upload image</p>
                  <p className="text-slate-500 text-xs mt-1">PNG, JPG support</p>
                </div>
              )}
            </div>
          </div>

          {/* Settings Card */}
          <div className="bg-surface rounded-2xl p-6 border border-slate-700 shadow-xl">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-slate-400" />
              Parameters
            </h2>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm text-slate-400">Top K Results</label>
                  <span className="text-sm font-mono text-primary font-bold">{topK}</span>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max={MAX_K} 
                  value={topK} 
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary hover:accent-secondary"
                />
              </div>

              <button
                onClick={handleSearch}
                disabled={!selectedFile || isSearching || systemStatus?.status !== 'ready'}
                className="w-full py-3 px-4 bg-primary hover:bg-secondary disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-semibold text-white transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary/25"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Find Similar Images
                  </>
                )}
              </button>
              
              {error && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  {error}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Results</h2>
            {results.length > 0 && (
              <span className="text-slate-400 text-sm">Found {results.length} matches</span>
            )}
          </div>

          {results.length === 0 ? (
            <div className="h-[500px] border border-slate-800 rounded-2xl flex flex-col items-center justify-center text-slate-500 bg-surface/30">
              <Search className="w-16 h-16 mb-4 opacity-20" />
              <p>Upload an image and hit search to see results</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-3 gap-6">
              {results.map((result, idx) => (
                <div 
                  key={idx} 
                  className="group bg-surface border border-slate-700 rounded-xl overflow-hidden shadow-lg hover:shadow-primary/10 transition-all duration-300 hover:-translate-y-1"
                >
                  <div className="aspect-square relative overflow-hidden bg-black">
                    <img 
                      src={result.url} 
                      alt={result.filename}
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'https://placehold.co/400x400?text=Image+Not+Found';
                      }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                      <p className="text-white text-sm font-medium truncate">{result.filename}</p>
                    </div>
                  </div>
                  <div className="p-3 border-t border-slate-700 bg-slate-800/50">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-400">Distance (L2)</span>
                      <span className="font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">
                        {result.distance.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
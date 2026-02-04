import React, { useState, useEffect, useRef } from 'react';
import { Upload, Search, Image as ImageIcon, Settings, Loader2, AlertCircle, X, ZoomIn, Database, Plus, CheckCircle2, Box, Grid, LayoutDashboard } from 'lucide-react';
import { API_BASE_URL, DEFAULT_K, MAX_K } from './constants';
import { SearchResponse, SearchResult, SystemStatus } from './types';

// Utility for class names
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

// Define tabs
type Tab = 'search' | 'gallery' | 'visualization';

// --- Components ---

// 3D Visualization Component using Plotly via global window object
const VectorSpace = () => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const plotDivRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/visualize`)
      .then(res => {
        if (!res.ok) throw new Error(`Server Error: ${res.statusText}`);
        return res.json();
      })
      .then(resData => {
        if (resData.error) throw new Error(resData.error);
        setData(resData.points || []);
      })
      .catch(err => {
        console.error("Visualization fetch error:", err);
        setError("Could not load visualization. Is the backend running?");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!loading && data.length > 0 && plotDivRef.current && (window as any).Plotly) {
      const Plotly = (window as any).Plotly;
      
      const trace = {
        x: data.map(p => p.x),
        y: data.map(p => p.y),
        z: data.map(p => p.z),
        mode: 'markers',
        type: 'scatter3d',
        text: data.map(p => p.filename), // Hover text
        hoverinfo: 'text',
        marker: {
          size: 5,
          color: data.map(p => p.z), // Color by Z axis for depth effect
          colorscale: 'Viridis',
          opacity: 0.8
        }
      };

      const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        scene: {
          xaxis: { title: 'PCA 1', color: 'white', gridcolor: '#334155' },
          yaxis: { title: 'PCA 2', color: 'white', gridcolor: '#334155' },
          zaxis: { title: 'PCA 3', color: 'white', gridcolor: '#334155' },
          bgcolor: 'rgba(0,0,0,0)' // Transparent background
        },
        showlegend: false,
        autosize: true
      };

      const config = { responsive: true, displayModeBar: false };

      Plotly.newPlot(plotDivRef.current, [trace], layout, config);
      
      // Cleanup
      return () => Plotly.purge(plotDivRef.current);
    }
  }, [loading, data]);

  if (loading) return <div className="h-96 flex items-center justify-center text-primary"><Loader2 className="w-8 h-8 animate-spin" /></div>;
  if (error) return <div className="h-96 flex items-center justify-center text-rose-400"><AlertCircle className="w-6 h-6 mr-2" /> {error}</div>;
  if (data.length === 0) return <div className="h-96 flex items-center justify-center text-slate-500">Not enough data to visualize.</div>;

  return (
    <div className="w-full h-[600px] border border-slate-700 rounded-2xl overflow-hidden bg-black/40 relative">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
         <h3 className="text-white font-bold text-lg bg-black/50 px-2 rounded">Semantic Vector Space</h3>
         <p className="text-slate-400 text-xs bg-black/50 px-2 rounded">PCA Projection (3D)</p>
      </div>
      <div ref={plotDivRef} className="w-full h-full" />
    </div>
  );
};

const GalleryGrid = ({ onZoom }: { onZoom: (item: any) => void }) => {
  const [images, setImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string|null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/database`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch gallery");
        return res.json();
      })
      .then(setData => setImages(setData))
      .catch(err => {
        console.error(err);
        setError("Failed to load images");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="py-20 flex justify-center"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>;
  if (error) return <div className="py-20 flex justify-center text-rose-400">{error}</div>;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {images.map((img, idx) => (
        <div 
          key={idx}
          onClick={() => onZoom(img)}
          className="aspect-square rounded-xl overflow-hidden border border-slate-800 relative group cursor-zoom-in bg-surface"
        >
          <img 
            src={img.url} 
            loading="lazy" 
            alt={img.filename} 
            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110" 
          />
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
             <p className="text-xs text-white px-2 text-center truncate w-full">{img.filename}</p>
          </div>
        </div>
      ))}
    </div>
  );
};


export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('search');

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(DEFAULT_K);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // State for lightbox
  const [zoomedImage, setZoomedImage] = useState<SearchResult | null>(null);

  // State for Upload/Add Modal
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  // Check Backend Status on Mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/status`);
        if (!res.ok) throw new Error("Failed to connect to backend");
        const data = await res.json();
        setSystemStatus(data);
      } catch (err) {
        console.error("Backend status check failed:", err);
        setSystemStatus({ status: 'error', index_size: 0, message: 'Backend unreachable' });
      }
    };
    checkStatus();
  }, [activeTab]); // Re-check when switching tabs sometimes helps

  // Handle Escape key to close modals
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setZoomedImage(null);
        if (uploadStatus !== 'success' && !isUploading) {
            setIsUploadModalOpen(false);
        }
      } 
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [uploadStatus, isUploading]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]); // Clear previous results
      setError(null);
    }
  };

  const handleUploadSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadFile(file);
      setUploadPreview(URL.createObjectURL(file));
      setUploadStatus('idle');
    }
  };

  const handleAddToDatabase = async () => {
    if (!uploadFile) return;

    setIsUploading(true);
    setUploadStatus('idle');

    const formData = new FormData();
    formData.append('file', uploadFile);

    try {
      const res = await fetch(`${API_BASE_URL}/add`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      
      const data = await res.json();
      
      // Update system status locally
      setSystemStatus(prev => prev ? ({...prev, index_size: data.index_size}) : null);
      setUploadStatus('success');
      
      // Clear selection after short delay
      setTimeout(() => {
          setIsUploadModalOpen(false);
          setUploadFile(null);
          setUploadPreview(null);
          setUploadStatus('idle');
          // If on gallery view, we might want to refresh, but simple state change is okay for now
      }, 1500);

    } catch (err) {
      console.error(err);
      setUploadStatus('error');
    } finally {
      setIsUploading(false);
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
      setError("Failed to fetch results. Ensure backend is running at " + API_BASE_URL);
    } finally {
      setIsSearching(false);
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="min-h-screen bg-dark text-slate-200 font-sans selection:bg-primary selection:text-white pb-10">
      {/* Header */}
      <header className="border-b border-slate-800 bg-dark/50 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary/20 p-2 rounded-lg">
              <Search className="w-6 h-6 text-primary" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-400">
              NeuroSearch
            </h1>
          </div>
          
          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-surface/50 p-1 rounded-xl border border-slate-700/50">
            <button 
              onClick={() => setActiveTab('search')}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'search' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50"
              )}
            >
              <Search className="w-4 h-4" /> Search
            </button>
            <button 
              onClick={() => setActiveTab('gallery')}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'gallery' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50"
              )}
            >
              <Grid className="w-4 h-4" /> Gallery
            </button>
            <button 
              onClick={() => setActiveTab('visualization')}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'visualization' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50"
              )}
            >
              <Box className="w-4 h-4" /> 3D Space
            </button>
          </nav>

          <div className="flex items-center gap-4 text-sm">
             {/* Add Button */}
             <button 
                onClick={() => setIsUploadModalOpen(true)}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors text-xs font-medium"
             >
                <Plus className="w-4 h-4" />
                Add Image
             </button>

            <div className="h-4 w-px bg-slate-700 mx-2 hidden sm:block"></div>

            {systemStatus?.status === 'ready' ? (
              <div className="flex items-center gap-2 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="hidden sm:inline">Index Active</span> 
                <span className="font-mono bg-emerald-400/10 px-1.5 rounded">
                  {systemStatus.index_size}
                </span>
              </div>
            ) : systemStatus?.status === 'error' ? (
              <div className="flex items-center gap-2 text-rose-400" title={systemStatus.message}>
                 <AlertCircle className="w-4 h-4" />
                 Backend Offline
              </div>
            ) : (
              <div className="flex items-center gap-2 text-amber-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                Loading...
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Mobile Nav (visible only on small screens) */}
      <div className="md:hidden flex p-2 justify-center gap-2 border-b border-slate-800 bg-surface">
         <button onClick={() => setActiveTab('search')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'search' ? 'bg-slate-700 text-white' : 'text-slate-400')}>Search</button>
         <button onClick={() => setActiveTab('gallery')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'gallery' ? 'bg-slate-700 text-white' : 'text-slate-400')}>Gallery</button>
         <button onClick={() => setActiveTab('visualization')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'visualization' ? 'bg-slate-700 text-white' : 'text-slate-400')}>3D Space</button>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {/* VIEW: SEARCH */}
        {activeTab === 'search' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-500">
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
                      onClick={() => setZoomedImage(result)}
                      className="group bg-surface border border-slate-700 rounded-xl overflow-hidden shadow-lg hover:shadow-primary/10 transition-all duration-300 hover:-translate-y-1 cursor-zoom-in relative"
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
                        
                        {/* Hover Overlay with Zoom Icon */}
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center z-10">
                            <ZoomIn className="text-white w-8 h-8 drop-shadow-md transform scale-75 group-hover:scale-100 transition-transform duration-300" />
                        </div>

                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4 z-20 pointer-events-none">
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
          </div>
        )}

        {/* VIEW: GALLERY */}
        {activeTab === 'gallery' && (
           <div className="animate-in fade-in duration-500">
              <div className="flex items-center justify-between mb-6">
                 <div>
                    <h2 className="text-2xl font-bold">Database Gallery</h2>
                    <p className="text-slate-400 text-sm">Browse all indexed images</p>
                 </div>
              </div>
              <GalleryGrid onZoom={(img) => setZoomedImage({...img, distance: 0})} />
           </div>
        )}

        {/* VIEW: VISUALIZATION */}
        {activeTab === 'visualization' && (
           <div className="animate-in fade-in duration-500">
              <div className="flex items-center justify-between mb-6">
                 <div>
                    <h2 className="text-2xl font-bold">Vector Space</h2>
                    <p className="text-slate-400 text-sm">3D visualization of image embeddings using PCA</p>
                 </div>
              </div>
              <VectorSpace />
           </div>
        )}

      </main>

      {/* Lightbox / Modal */}
      {zoomedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setZoomedImage(null)}
        >
          {/* Close Button */}
          <button 
            onClick={() => setZoomedImage(null)}
            className="absolute top-6 right-6 p-2 rounded-full bg-slate-800/50 hover:bg-slate-700 text-white/80 hover:text-white transition-colors z-50"
          >
            <X className="w-8 h-8" />
          </button>
          
          <div 
            className="relative flex flex-col items-center max-w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={zoomedImage.url} 
              alt={zoomedImage.filename}
              className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl border border-slate-800/50"
            />
            
            <div className="mt-6 bg-slate-900/90 backdrop-blur-md px-8 py-4 rounded-2xl border border-slate-700 text-center shadow-xl">
              <p className="text-white text-lg font-medium tracking-wide">{zoomedImage.filename}</p>
              {zoomedImage.distance > 0 && (
                <div className="flex items-center justify-center gap-2 mt-1">
                  <span className="text-slate-400 text-sm">Similarity Distance:</span>
                  <span className="text-primary font-mono font-bold bg-primary/10 px-2 rounded">
                    {zoomedImage.distance.toFixed(6)}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Upload/Add Modal */}
      {isUploadModalOpen && (
         <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in zoom-in-95 duration-200">
             <div className="bg-surface border border-slate-700 w-full max-w-md rounded-2xl p-6 shadow-2xl relative">
                 <button 
                   onClick={() => setIsUploadModalOpen(false)}
                   className="absolute top-4 right-4 text-slate-400 hover:text-white"
                   disabled={isUploading}
                 >
                   <X className="w-5 h-5" />
                 </button>

                 <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                   <Database className="w-5 h-5 text-primary" />
                   Add to Database
                 </h3>
                 
                 <div 
                   onClick={() => !isUploading && uploadInputRef.current?.click()}
                   className={cn(
                     "border-2 border-dashed rounded-xl h-48 flex flex-col items-center justify-center cursor-pointer transition-all mb-4 relative overflow-hidden",
                     uploadPreview ? "border-primary/50 bg-dark" : "border-slate-600 hover:border-primary hover:bg-slate-800/50",
                     isUploading && "pointer-events-none opacity-50"
                   )}
                 >
                   <input 
                     type="file" 
                     ref={uploadInputRef} 
                     className="hidden" 
                     accept="image/*" 
                     onChange={handleUploadSelect} 
                   />

                   {uploadPreview ? (
                     <img src={uploadPreview} alt="Upload Preview" className="w-full h-full object-contain p-2" />
                   ) : (
                     <div className="text-center p-4">
                       <Plus className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                       <p className="text-sm text-slate-300">Select image to index</p>
                     </div>
                   )}
                 </div>

                 {uploadStatus === 'error' && (
                    <p className="text-rose-400 text-sm mb-4 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" /> Upload failed. Try again.
                    </p>
                 )}

                 <button
                    onClick={handleAddToDatabase}
                    disabled={!uploadFile || isUploading || uploadStatus === 'success'}
                    className={cn(
                        "w-full py-2.5 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all",
                        uploadStatus === 'success' 
                            ? "bg-emerald-500 text-white" 
                            : "bg-primary hover:bg-secondary text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    )}
                 >
                    {isUploading ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Processing...
                        </>
                    ) : uploadStatus === 'success' ? (
                        <>
                            <CheckCircle2 className="w-5 h-5" />
                            Added Successfully
                        </>
                    ) : (
                        "Upload & Index"
                    )}
                 </button>
             </div>
         </div>
      )}
    </div>
  );
}
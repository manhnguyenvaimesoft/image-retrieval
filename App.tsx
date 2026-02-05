import React, { useState, useEffect, useRef } from 'react';
import { Upload, Search, Image as ImageIcon, Settings, Loader2, AlertCircle, X, ZoomIn, Database, Plus, CheckCircle2, Box, Grid, Folder, Trash2, LayoutDashboard } from 'lucide-react';
import { API_BASE_URL, DEFAULT_K, MAX_K } from './constants';
import { SearchResponse, SearchResult, SystemStatus } from './types';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

type Tab = 'search' | 'gallery' | 'visualization' | 'projects';

// --- Components ---

const VectorSpace = ({ activeProject }: { activeProject: string | null }) => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const plotDivRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
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
  }, [activeProject]);

  useEffect(() => {
    if (!loading && data.length > 0 && plotDivRef.current && (window as any).Plotly) {
      const Plotly = (window as any).Plotly;
      
      const trace = {
        x: data.map(p => p.x),
        y: data.map(p => p.y),
        z: data.map(p => p.z),
        mode: 'markers',
        type: 'scatter3d',
        text: data.map(p => p.filename),
        hoverinfo: 'text',
        marker: {
          size: 5,
          color: data.map(p => p.z),
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
          bgcolor: 'rgba(0,0,0,0)'
        },
        showlegend: false,
        autosize: true
      };

      const config = { responsive: true, displayModeBar: false };
      Plotly.newPlot(plotDivRef.current, [trace], layout, config);
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

const GalleryGrid = ({ activeProject, onZoom }: { activeProject: string | null, onZoom: (item: any) => void }) => {
  const [images, setImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string|null>(null);

  useEffect(() => {
    setLoading(true);
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
  }, [activeProject]);

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
  const [zoomedImage, setZoomedImage] = useState<SearchResult | null>(null);

  // Projects State
  const [projects, setProjects] = useState<any>({});
  const [activeProject, setActiveProject] = useState<string | null>(null);
  const [buildProgress, setBuildProgress] = useState<any>({});
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectPath, setNewProjectPath] = useState('');

  // Upload Modal
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/status`);
      const data = await res.json();
      setSystemStatus(data);
      setActiveProject(data.active_project);
    } catch (err) {
      setSystemStatus({ status: 'error', index_size: 0, message: 'Backend unreachable' });
    }
  };

  const fetchProjects = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/projects`);
      const data = await res.json();
      setProjects(data.projects);
      setBuildProgress(data.build_progress);
      setActiveProject(data.active_project);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchProjects();
    const interval = setInterval(fetchProjects, 3000); // Polling for build progress
    return () => clearInterval(interval);
  }, []);

  const handleSelectProject = async (name: string) => {
    const formData = new FormData();
    formData.append('name', name);
    try {
      await fetch(`${API_BASE_URL}/projects/select`, { method: 'POST', body: formData });
      fetchStatus();
      fetchProjects();
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', newProjectName);
    formData.append('train_path', newProjectPath);
    try {
      await fetch(`${API_BASE_URL}/projects`, { method: 'POST', body: formData });
      setIsCreateModalOpen(false);
      setNewProjectName('');
      setNewProjectPath('');
      fetchProjects();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteProject = async (name: string) => {
    if (!confirm(`Delete project "${name}"? This will remove the index but not your source images.`)) return;
    try {
      await fetch(`${API_BASE_URL}/projects/${name}`, { method: 'DELETE' });
      fetchProjects();
      fetchStatus();
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
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
      const res = await fetch(`${API_BASE_URL}/search`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data: SearchResponse = await res.json();
      setResults(data.results);
    } catch (err) {
      setError("Search failed. Ensure backend is running.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleAddToDatabase = async () => {
    if (!uploadFile) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', uploadFile);
    try {
      const res = await fetch(`${API_BASE_URL}/add`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error("Upload failed");
      setUploadStatus('success');
      fetchStatus();
      setTimeout(() => {
          setIsUploadModalOpen(false);
          setUploadFile(null);
          setUploadPreview(null);
          setUploadStatus('idle');
      }, 1500);
    } catch (err) {
      setUploadStatus('error');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark text-slate-200 font-sans pb-10">
      <header className="border-b border-slate-800 bg-dark/50 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary/20 p-2 rounded-lg"><Search className="w-6 h-6 text-primary" /></div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-400">NeuroSearch</h1>
          </div>
          
          <nav className="hidden md:flex items-center gap-1 bg-surface/50 p-1 rounded-xl border border-slate-700/50">
            {[
              { id: 'search', label: 'Search', icon: Search },
              { id: 'gallery', label: 'Gallery', icon: Grid },
              { id: 'visualization', label: '3D Space', icon: Box },
              { id: 'projects', label: 'Projects', icon: Folder },
            ].map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id as Tab)}
                className={cn(
                  "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                  activeTab === tab.id ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                )}
              >
                <tab.icon className="w-4 h-4" /> {tab.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-4 text-sm">
             <div className="flex flex-col items-end mr-2">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Active Project</span>
                <span className="text-primary font-medium">{activeProject || 'None'}</span>
             </div>

             <button 
                onClick={() => setIsUploadModalOpen(true)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors text-xs font-medium flex items-center gap-2"
             >
                <Plus className="w-4 h-4" /> Add Image
             </button>

            <div className="h-4 w-px bg-slate-700 mx-2 hidden sm:block"></div>

            {systemStatus?.status === 'ready' ? (
              <div className="flex items-center gap-2 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="font-mono bg-emerald-400/10 px-1.5 rounded">{systemStatus.index_size}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-amber-400"><Loader2 className="w-3 h-3 animate-spin" /></div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {activeTab === 'search' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-4 space-y-6">
              <div className="bg-surface rounded-2xl p-6 border border-slate-700 shadow-xl">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><ImageIcon className="w-5 h-5 text-slate-400" /> Query Image</h2>
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "border-2 border-dashed rounded-xl h-64 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 relative overflow-hidden",
                    previewUrl ? "border-primary/50 bg-dark" : "border-slate-600 hover:border-primary hover:bg-slate-800/50"
                  )}
                >
                  <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileSelect} />
                  {previewUrl ? <img src={previewUrl} alt="Query" className="w-full h-full object-contain p-2" /> : (
                    <div className="text-center p-4">
                      <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center mx-auto mb-3"><Upload className="w-6 h-6 text-slate-300" /></div>
                      <p className="text-slate-300 font-medium">Click to upload image</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-surface rounded-2xl p-6 border border-slate-700 shadow-xl">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Settings className="w-5 h-5 text-slate-400" /> Parameters</h2>
                <div className="space-y-4">
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-slate-400">Top K Results</label>
                    <span className="text-sm font-mono text-primary font-bold">{topK}</span>
                  </div>
                  <input type="range" min="1" max={MAX_K} value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary" />
                  <button onClick={handleSearch} disabled={!selectedFile || isSearching || systemStatus?.status !== 'ready'} className="w-full py-3 px-4 bg-primary hover:bg-secondary disabled:opacity-50 rounded-xl font-semibold text-white flex items-center justify-center gap-2">
                    {isSearching ? <><Loader2 className="w-5 h-5 animate-spin" /> Searching...</> : <><Search className="w-5 h-5" /> Find Similar</>}
                  </button>
                </div>
              </div>
            </div>

            <div className="lg:col-span-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Results</h2>
                {results.length > 0 && <span className="text-slate-400 text-sm">Found {results.length} matches</span>}
              </div>
              {results.length === 0 ? (
                <div className="h-[500px] border border-slate-800 rounded-2xl flex flex-col items-center justify-center text-slate-500 bg-surface/30">
                  <Search className="w-16 h-16 mb-4 opacity-20" />
                  <p>Upload an image and hit search</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                  {results.map((result, idx) => (
                    <div key={idx} onClick={() => setZoomedImage(result)} className="group bg-surface border border-slate-700 rounded-xl overflow-hidden hover:shadow-primary/10 transition-all cursor-zoom-in">
                      <div className="aspect-square relative overflow-hidden bg-black">
                        <img src={result.url} alt={result.filename} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><ZoomIn className="text-white w-8 h-8" /></div>
                      </div>
                      <div className="p-3 border-t border-slate-700 bg-slate-800/50 flex justify-between items-center text-xs">
                        <span className="text-slate-400 truncate max-w-[100px]">{result.filename}</span>
                        <span className="font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">{result.distance.toFixed(4)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'gallery' && <GalleryGrid activeProject={activeProject} onZoom={(img) => setZoomedImage({...img, distance: 0})} />}
        {activeTab === 'visualization' && <VectorSpace activeProject={activeProject} />}

        {activeTab === 'projects' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold">Projects</h2>
                <p className="text-slate-400">Manage image databases and indexing</p>
              </div>
              <button 
                onClick={() => setIsCreateModalOpen(true)}
                className="bg-primary hover:bg-secondary text-white px-4 py-2 rounded-xl font-semibold flex items-center gap-2"
              >
                <Plus className="w-5 h-5" /> New Project
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.values(projects).map((project: any) => {
                const progress = buildProgress[project.name];
                const isActive = activeProject === project.name;
                
                return (
                  <div key={project.name} className={cn("bg-surface border rounded-2xl p-6 transition-all", isActive ? "border-primary shadow-lg shadow-primary/10" : "border-slate-700")}>
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-lg font-bold flex items-center gap-2">
                           {project.name}
                           {isActive && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                        </h3>
                        <p className="text-xs text-slate-500 font-mono mt-1 truncate max-w-[200px]">{project.train_path}</p>
                      </div>
                      <div className="flex gap-2">
                         {!isActive && (
                           <button 
                            onClick={() => handleSelectProject(project.name)}
                            className="p-2 text-slate-400 hover:text-primary transition-colors"
                            title="Select Project"
                           >
                             <CheckCircle2 className="w-5 h-5" />
                           </button>
                         )}
                         {project.name !== 'default' && (
                            <button 
                              onClick={() => handleDeleteProject(project.name)}
                              className="p-2 text-slate-400 hover:text-rose-400 transition-colors"
                            >
                              <Trash2 className="w-5 h-5" />
                            </button>
                         )}
                      </div>
                    </div>

                    {progress && progress.status !== 'completed' && (
                      <div className="space-y-2 mt-4">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400 uppercase">{progress.status}</span>
                          <span className="font-mono">{progress.current} / {progress.total}</span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                           <div 
                            className="h-full bg-primary transition-all duration-300" 
                            style={{ width: `${(progress.current / (progress.total || 1)) * 100}%` }} 
                           />
                        </div>
                        <p className="text-[10px] text-slate-500 italic">{progress.message}</p>
                      </div>
                    )}

                    {progress?.status === 'completed' && (
                        <div className="mt-4 flex items-center gap-2 text-xs text-emerald-400">
                           <CheckCircle2 className="w-3 h-3" /> Indexed Successfully
                        </div>
                    )}
                    
                    {!progress && (
                         <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
                            <Box className="w-3 h-3" /> Ready
                         </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </main>

      {/* Lightbox */}
      {zoomedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-sm p-4" onClick={() => setZoomedImage(null)}>
          <button className="absolute top-6 right-6 p-2 text-white/80 hover:text-white"><X className="w-8 h-8" /></button>
          <div className="relative flex flex-col items-center max-w-full" onClick={(e) => e.stopPropagation()}>
            <img src={zoomedImage.url} alt={zoomedImage.filename} className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl" />
            <div className="mt-6 bg-slate-900/90 backdrop-blur-md px-8 py-4 rounded-2xl border border-slate-700 text-center">
              <p className="text-white text-lg font-medium">{zoomedImage.filename}</p>
            </div>
          </div>
        </div>
      )}

      {/* Create Project Modal */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-surface border border-slate-700 w-full max-w-md rounded-2xl p-6 shadow-2xl">
            <h3 className="text-xl font-bold mb-4">Create New Project</h3>
            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Project Name</label>
                <input required type="text" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} className="w-full bg-dark border border-slate-700 rounded-xl px-4 py-2 text-white focus:border-primary outline-none" placeholder="e.g. MyDataset" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Source Path (Absolute path to folder)</label>
                <input required type="text" value={newProjectPath} onChange={(e) => setNewProjectPath(e.target.value)} className="w-full bg-dark border border-slate-700 rounded-xl px-4 py-2 text-white focus:border-primary outline-none font-mono text-xs" placeholder="/path/to/your/images" />
              </div>
              <div className="flex gap-3 pt-2">
                <button type="button" onClick={() => setIsCreateModalOpen(false)} className="flex-1 py-2 bg-slate-800 hover:bg-slate-700 rounded-xl font-semibold">Cancel</button>
                <button type="submit" className="flex-1 py-2 bg-primary hover:bg-secondary text-white rounded-xl font-semibold">Create & Index</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Add Image Modal */}
      {isUploadModalOpen && (
         <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
             <div className="bg-surface border border-slate-700 w-full max-w-md rounded-2xl p-6 shadow-2xl relative">
                 <button onClick={() => setIsUploadModalOpen(false)} className="absolute top-4 right-4 text-slate-400 hover:text-white" disabled={isUploading}><X className="w-5 h-5" /></button>
                 <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><Database className="w-5 h-5 text-primary" /> Add to {activeProject}</h3>
                 <div onClick={() => !isUploading && uploadInputRef.current?.click()} className={cn("border-2 border-dashed rounded-xl h-48 flex flex-col items-center justify-center cursor-pointer transition-all mb-4 relative overflow-hidden", uploadPreview ? "border-primary/50 bg-dark" : "border-slate-600 hover:border-primary hover:bg-slate-800/50", isUploading && "pointer-events-none opacity-50")}>
                   <input type="file" ref={uploadInputRef} className="hidden" accept="image/*" onChange={(e) => { if (e.target.files?.[0]) { setUploadFile(e.target.files[0]); setUploadPreview(URL.createObjectURL(e.target.files[0])); } }} />
                   {uploadPreview ? <img src={uploadPreview} alt="Upload Preview" className="w-full h-full object-contain p-2" /> : (
                     <div className="text-center p-4">
                       <Plus className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                       <p className="text-sm text-slate-300">Select image</p>
                     </div>
                   )}
                 </div>
                 <button onClick={handleAddToDatabase} disabled={!uploadFile || isUploading || uploadStatus === 'success'} className={cn("w-full py-2.5 rounded-xl font-semibold flex items-center justify-center gap-2", uploadStatus === 'success' ? "bg-emerald-500 text-white" : "bg-primary hover:bg-secondary text-white disabled:opacity-50")}>
                    {isUploading ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</> : uploadStatus === 'success' ? <><CheckCircle2 className="w-5 h-5" /> Added</> : "Upload & Index"}
                 </button>
             </div>
         </div>
      )}
    </div>
  );
}

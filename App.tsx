import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Search, Image as ImageIcon, Settings, Loader2, AlertCircle, X, ZoomIn, Database, Plus, CheckCircle2, Box, Grid, FolderPlus, ChevronDown, Layers, Trash2, Star, Layout, LogOut, User, Lock, ArrowRight } from 'lucide-react';
import { API_BASE_URL, DEFAULT_K, MAX_K } from './constants';
import { SearchResponse, SearchResult, SystemStatus, Project, IndexingStatus } from './types';

// Utility for class names
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

// Define tabs
type Tab = 'search' | 'gallery' | 'visualization';

// Extend Project type locally to include new field if backend sends it
interface ExtendedProject extends Project {
    is_default?: boolean;
}

// Helper for Fetch with Auth Header
const authFetch = async (url: string, options: RequestInit = {}) => {
    const token = localStorage.getItem('access_token');
    const headers = new Headers(options.headers || {});
    if (token) {
        headers.append('Authorization', `Bearer ${token}`);
    }
    
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
        // Just clear token, don't trigger full reload loop inside fetch
        localStorage.removeItem('access_token');
        localStorage.removeItem('username');
        throw new Error("Unauthorized");
    }
    return res;
};

// --- Auth Components ---

const AuthPage = ({ onLogin }: { onLogin: () => void }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("username", username);
            formData.append("password", password);

            if (isLogin) {
                const res = await fetch(`${API_BASE_URL}/auth/login`, {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error("Invalid username or password");
                
                const data = await res.json();
                localStorage.setItem("access_token", data.access_token);
                localStorage.setItem("username", data.username);
                onLogin();
            } else {
                const res = await fetch(`${API_BASE_URL}/auth/register`, {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Registration failed");
                }
                setIsLogin(true); // Switch to login after success
                setError("Account created! Please log in.");
                return; // Don't trigger onLogin yet
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-dark flex items-center justify-center p-4">
            <div className="bg-surface border border-slate-700 w-full max-w-md p-8 rounded-2xl shadow-2xl animate-in zoom-in-95 fade-in">
                <div className="flex justify-center mb-6">
                    <div className="bg-primary/20 p-3 rounded-xl">
                        <Search className="w-8 h-8 text-primary" />
                    </div>
                </div>
                <h1 className="text-2xl font-bold text-center text-white mb-2">NeuroSearch</h1>
                <p className="text-slate-400 text-center text-sm mb-8">
                    {isLogin ? "Sign in to access your projects" : "Create an account to get started"}
                </p>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-1 uppercase">Username</label>
                        <div className="relative">
                            <User className="absolute left-3 top-2.5 w-5 h-5 text-slate-500" />
                            <input 
                                type="text" 
                                required
                                value={username}
                                onChange={e => setUsername(e.target.value)}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-white focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                                placeholder="Enter username"
                            />
                        </div>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-1 uppercase">Password</label>
                        <div className="relative">
                            <Lock className="absolute left-3 top-2.5 w-5 h-5 text-slate-500" />
                            <input 
                                type="password" 
                                required
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-white focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                                placeholder="••••••••"
                            />
                        </div>
                    </div>

                    {error && (
                        <div className={`text-sm p-3 rounded-lg flex items-center gap-2 ${error.includes("created") ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                            {error.includes("created") ? <CheckCircle2 className="w-4 h-4"/> : <AlertCircle className="w-4 h-4" />}
                            {error}
                        </div>
                    )}

                    <button 
                        type="submit" 
                        disabled={loading}
                        className="w-full bg-primary hover:bg-secondary text-white font-semibold py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary/25"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : (
                            <>
                                {isLogin ? "Sign In" : "Create Account"}
                                <ArrowRight className="w-4 h-4" />
                            </>
                        )}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <button 
                        onClick={() => { setIsLogin(!isLogin); setError(null); }}
                        className="text-sm text-slate-400 hover:text-white transition-colors"
                    >
                        {isLogin ? "Don't have an account? Sign up" : "Already have an account? Log in"}
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Components ---

// 3D Visualization Component using Plotly via global window object
const VectorSpace = ({ versionKey, onPointClick }: { versionKey: number, onPointClick: (item: any) => void }) => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const plotDivRef = useRef<HTMLDivElement>(null);
  const onPointClickRef = useRef(onPointClick);

  useEffect(() => {
    onPointClickRef.current = onPointClick;
  }, [onPointClick]);

  useEffect(() => {
    setLoading(true);
    authFetch(`${API_BASE_URL}/visualize`)
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
        setError("Could not load visualization. Select a project.");
      })
      .finally(() => setLoading(false));
  }, [versionKey]);

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
        autosize: true,
        hovermode: 'closest',
        uirevision: 'true' 
      };

      const config = { responsive: true, displayModeBar: false };
      Plotly.react(plotDivRef.current, [trace], layout, config);
      
      const plotEl = plotDivRef.current as any;
      plotEl.removeAllListeners('plotly_click');
      plotEl.on('plotly_click', (dataEvent: any) => {
          if (dataEvent.points && dataEvent.points.length > 0) {
              const pointIndex = dataEvent.points[0].pointNumber;
              const pointData = data[pointIndex];
              if (pointData) {
                  onPointClickRef.current({
                      filename: pointData.filename,
                      url: pointData.url,
                      distance: 0, 
                      filepath: pointData.filename 
                  });
              }
          }
      });
    }
  }, [loading, data]);

  if (loading) return <div className="h-96 flex items-center justify-center text-primary"><Loader2 className="w-8 h-8 animate-spin" /></div>;
  if (error) return <div className="h-96 flex items-center justify-center text-rose-400"><AlertCircle className="w-6 h-6 mr-2" /> {error}</div>;
  if (data.length === 0) return <div className="h-96 flex items-center justify-center text-slate-500">Not enough data to visualize (min 3 images).</div>;

  return (
    <div className="w-full h-[600px] border border-slate-700 rounded-2xl overflow-hidden bg-black/40 relative">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
         <h3 className="text-white font-bold text-lg bg-black/50 px-2 rounded">Semantic Vector Space</h3>
         <p className="text-slate-400 text-xs bg-black/50 px-2 rounded">PCA Projection (3D) - Click point to view image</p>
      </div>
      <div ref={plotDivRef} className="w-full h-full cursor-pointer" />
    </div>
  );
};

const GalleryGrid = ({ onZoom, versionKey }: { onZoom: (item: any) => void, versionKey: number }) => {
  const [images, setImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string|null>(null);

  useEffect(() => {
    setLoading(true);
    authFetch(`${API_BASE_URL}/database`)
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
  }, [versionKey]);

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
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem("access_token"));
  const [currentUser, setCurrentUser] = useState(localStorage.getItem("username") || "");

  const [activeTab, setActiveTab] = useState<Tab>('search');

  // --- Project State ---
  const [projects, setProjects] = useState<ExtendedProject[]>([]);
  const [isProjectDropdownOpen, setIsProjectDropdownOpen] = useState(false);
  
  // --- Create Project State ---
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectFiles, setNewProjectFiles] = useState<FileList | null>(null);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  
  // --- Change Password State ---
  const [isChangePasswordModalOpen, setIsChangePasswordModalOpen] = useState(false);
  const [oldPass, setOldPass] = useState("");
  const [newPass, setNewPass] = useState("");
  const [isChangingPass, setIsChangingPass] = useState(false);
  const [changePassMsg, setChangePassMsg] = useState<{type: 'success'|'error', text: string} | null>(null);

  // --- Indexing/Loading State ---
  const [indexingStatus, setIndexingStatus] = useState<IndexingStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [dataVersion, setDataVersion] = useState(0);

  // --- Search State ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(DEFAULT_K);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // --- Lightbox ---
  const [zoomedImage, setZoomedImage] = useState<SearchResult | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // --- Upload/Add Modal ---
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const handleLogout = () => {
      // Confirm with user
      if (!window.confirm("Are you sure you want to sign out?")) return;

      localStorage.removeItem("access_token");
      localStorage.removeItem("username");
      
      // Clear ALL state to prevent data leakage
      setIsAuthenticated(false);
      setCurrentUser("");
      setProjects([]);
      setSystemStatus(null);
      setIndexingStatus(null);
      setResults([]);
      setSelectedFile(null);
      setPreviewUrl(null);
      setZoomedImage(null);
      setDataVersion(0);
      setActiveTab('search');
      
      // Close any open modals
      setIsProjectDropdownOpen(false);
      setIsCreateModalOpen(false);
      setIsUploadModalOpen(false);
      setIsChangePasswordModalOpen(false);
  };

  const onLoginSuccess = () => {
      setIsAuthenticated(true);
      setCurrentUser(localStorage.getItem("username") || "");
  };

  const handleChangePassword = async (e: React.FormEvent) => {
      e.preventDefault();
      setIsChangingPass(true);
      setChangePassMsg(null);
      
      try {
          const formData = new FormData();
          formData.append("old_password", oldPass);
          formData.append("new_password", newPass);
          
          const res = await authFetch(`${API_BASE_URL}/auth/change-password`, {
              method: 'POST',
              body: formData
          });
          
          if(!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || "Failed to update password");
          }
          
          setChangePassMsg({type: 'success', text: "Password changed successfully."});
          setOldPass("");
          setNewPass("");
          
          setTimeout(() => {
              setIsChangePasswordModalOpen(false);
              setChangePassMsg(null);
          }, 1500);
          
      } catch(err: any) {
          setChangePassMsg({type: 'error', text: err.message});
      } finally {
          setIsChangingPass(false);
      }
  };

  // --- Poll for System Status (Only when authenticated) ---
  useEffect(() => {
    if (!isAuthenticated) return;

    fetchProjects();
    const interval = setInterval(async () => {
       try {
         // Check System Status
         const resStatus = await authFetch(`${API_BASE_URL}/status`);
         if(resStatus.ok) {
             const statusData = await resStatus.json();
             setSystemStatus(statusData);

             // Check Indexing Process
             if (statusData.status === 'indexing') {
                const resIndex = await authFetch(`${API_BASE_URL}/indexing_status`);
                const indexData = await resIndex.json();
                setIndexingStatus(indexData);
             } else {
                // If we were indexing and now we are not, refresh data
                if (indexingStatus?.is_indexing) {
                    setIndexingStatus(null);
                    fetchProjects();
                    setDataVersion(v => v + 1); // Refresh gallery
                }
             }
         }
       } catch (e: any) {
         if (e.message === "Unauthorized") {
             handleLogout(); // Force clean logout on auth fail
         }
         console.error("Polling error", e);
       }
    }, 1000);
    return () => clearInterval(interval);
  }, [indexingStatus?.is_indexing, isAuthenticated]);

  const fetchProjects = async () => {
    try {
        const res = await authFetch(`${API_BASE_URL}/projects`);
        if(res.ok) {
            const data = await res.json();
            setProjects(data);
        }
    } catch (e) { console.error("Failed to load projects"); }
  };

  const handleCreateProject = async () => {
    if (!newProjectName || !newProjectFiles || newProjectFiles.length === 0) return;
    
    setIsCreatingProject(true);
    try {
        const formData = new FormData();
        formData.append('name', newProjectName);
        for (let i = 0; i < newProjectFiles.length; i++) {
            formData.append('files', newProjectFiles[i]);
        }
        
        const res = await authFetch(`${API_BASE_URL}/projects/create`, { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to create project");
        }
        
        fetchProjects();
        setIsCreateModalOpen(false);
        setNewProjectName("");
        setNewProjectFiles(null);
    } catch (err: any) {
        alert(err.message);
    } finally {
        setIsCreatingProject(false);
    }
  };

  const handleSwitchProject = async (projectId: string) => {
    setIsProjectDropdownOpen(false);
    try {
        const formData = new FormData();
        formData.append('project_id', projectId);
        await authFetch(`${API_BASE_URL}/projects/switch`, { method: 'POST', body: formData });
        
        setResults([]);
        setSelectedFile(null);
        setPreviewUrl(null);
        setDataVersion(v => v + 1);
    } catch (e) { console.error(e); }
  };

  const handleSetDefault = async (projectId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
          const formData = new FormData();
          formData.append('project_id', projectId);
          const res = await authFetch(`${API_BASE_URL}/projects/set_default`, { method: 'POST', body: formData });
          if(res.ok) fetchProjects();
      } catch(e) { console.error(e); }
  };

  const handleSearch = async () => {
    if (!selectedFile) return;
    if (systemStatus?.current_project === "None" || !systemStatus?.current_project) {
        setError("Please select a project first.");
        return;
    }

    setIsSearching(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('k', topK.toString());

    try {
      const res = await authFetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data: SearchResponse = await res.json();
      setResults(data.results);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch results.");
    } finally {
      setIsSearching(false);
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
      const res = await authFetch(`${API_BASE_URL}/add`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error("Upload failed");
      
      const data = await res.json();
      setSystemStatus(prev => prev ? ({...prev, index_size: data.index_size}) : null);
      setUploadStatus('success');
      setDataVersion(v => v + 1);
      
      setTimeout(() => {
          setIsUploadModalOpen(false);
          setUploadFile(null);
          setUploadPreview(null);
          setUploadStatus('idle');
      }, 1500);
    } catch (err) {
      console.error(err);
      setUploadStatus('error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteImage = async () => {
      if (!zoomedImage) return;
      if (!window.confirm(`Are you sure you want to remove "${zoomedImage.filename}" from the database?`)) return;

      setIsDeleting(true);
      const formData = new FormData();
      formData.append('filename', zoomedImage.filename);

      try {
          const res = await authFetch(`${API_BASE_URL}/delete`, { method: 'POST', body: formData });
          if (!res.ok) throw new Error("Delete failed");

          const data = await res.json();
          setSystemStatus(prev => prev ? ({...prev, index_size: data.index_size}) : null);
          setDataVersion(v => v + 1);
          setZoomedImage(null);
          setResults(prev => prev.filter(r => r.filename !== zoomedImage.filename));
      } catch (err) {
          alert("Failed to delete image");
      } finally {
          setIsDeleting(false);
      }
  };

  const handlePointClick = useCallback((img: any) => {
    setZoomedImage(img);
  }, []);

  if (!isAuthenticated) {
      return <AuthPage onLogin={onLoginSuccess} />;
  }

  const isProjectLoaded = systemStatus?.current_project && systemStatus.current_project !== "None";

  return (
    <div className="min-h-screen bg-dark text-slate-200 font-sans selection:bg-primary selection:text-white pb-10">
      
      {/* Indexing Overlay */}
      {systemStatus?.status === 'indexing' && indexingStatus && (
        <div className="fixed inset-0 z-[60] bg-black/80 backdrop-blur-sm flex items-center justify-center">
            <div className="bg-surface border border-slate-700 p-8 rounded-2xl max-w-lg w-full shadow-2xl animate-in zoom-in-95 fade-in">
                <h3 className="text-xl font-bold mb-2 flex items-center gap-2">
                    <Loader2 className="w-6 h-6 animate-spin text-primary" />
                    Creating Database...
                </h3>
                <p className="text-slate-400 mb-6">{indexingStatus.current_step}</p>
                <div className="w-full bg-slate-700 rounded-full h-4 mb-2 overflow-hidden">
                    <div 
                        className="bg-primary h-full transition-all duration-300 ease-out" 
                        style={{ width: `${indexingStatus.progress}%` }}
                    />
                </div>
                <div className="flex justify-between text-sm text-slate-400">
                    <span>{indexingStatus.processed_files} / {indexingStatus.total_files} files</span>
                    <span>{indexingStatus.progress}%</span>
                </div>
            </div>
        </div>
      )}

      {/* Header */}
      <header className="border-b border-slate-800 bg-dark/50 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
                <div className="bg-primary/20 p-2 rounded-lg">
                <Search className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-400 hidden sm:block">
                NeuroSearch
                </h1>
            </div>

            {/* Project Switcher Dropdown */}
            <div className="relative">
                <button 
                    onClick={() => setIsProjectDropdownOpen(!isProjectDropdownOpen)}
                    className={cn(
                        "flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors",
                        isProjectLoaded 
                            ? "bg-slate-800 hover:bg-slate-700 border-slate-700" 
                            : "bg-rose-500/10 border-rose-500/30 text-rose-400 animate-pulse"
                    )}
                >
                    <Layers className="w-4 h-4" />
                    <span className="max-w-[150px] truncate">
                        {isProjectLoaded ? systemStatus?.current_project : "Select Project"}
                    </span>
                    <ChevronDown className="w-3 h-3 opacity-70" />
                </button>

                {isProjectDropdownOpen && (
                    <div className="absolute top-full left-0 mt-2 w-72 bg-surface border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50 animate-in fade-in zoom-in-95 duration-100">
                        <div className="p-2 border-b border-slate-700/50">
                            <span className="text-xs font-semibold text-slate-500 px-2 uppercase tracking-wider">Projects</span>
                        </div>
                        <div className="max-h-64 overflow-y-auto">
                            {projects.length === 0 ? (
                                <div className="p-4 text-center text-xs text-slate-500">
                                    No projects found. Create one to start.
                                </div>
                            ) : (
                                projects.map(p => (
                                    <div
                                        key={p.id}
                                        className={cn(
                                            "w-full px-4 py-2 text-sm flex items-center justify-between group cursor-pointer transition-colors border-l-2",
                                            systemStatus?.current_project === p.name 
                                                ? "bg-primary/5 border-primary text-white" 
                                                : "border-transparent text-slate-300 hover:bg-slate-700/50"
                                        )}
                                        onClick={() => handleSwitchProject(p.id)}
                                    >
                                        <div className="flex items-center gap-2 overflow-hidden">
                                            {p.is_default && <Star className="w-3 h-3 text-amber-400 fill-amber-400" />}
                                            <span className="truncate">{p.name}</span>
                                        </div>
                                        
                                        <div className="flex items-center gap-2">
                                            {systemStatus?.current_project === p.name && <CheckCircle2 className="w-4 h-4 text-primary" />}
                                            <button 
                                                title="Set as Default"
                                                onClick={(e) => handleSetDefault(p.id, e)}
                                                className={cn(
                                                    "p-1 rounded hover:bg-slate-600 transition-colors",
                                                    p.is_default ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                                                )}
                                            >
                                                <Star className={cn("w-3 h-3", p.is_default ? "text-amber-400 fill-amber-400" : "text-slate-500")} />
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                        <div className="p-2 border-t border-slate-700/50 bg-slate-800/50">
                            <button 
                                onClick={() => { setIsProjectDropdownOpen(false); setIsCreateModalOpen(true); }}
                                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-primary hover:text-white hover:bg-primary/20 rounded-lg transition-colors"
                            >
                                <Plus className="w-3 h-3" /> New Project
                            </button>
                        </div>
                    </div>
                )}
            </div>
          </div>
          
          <nav className="hidden md:flex items-center gap-1 bg-surface/50 p-1 rounded-xl border border-slate-700/50">
            <button 
              onClick={() => setActiveTab('search')}
              disabled={!isProjectLoaded}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'search' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50",
                !isProjectLoaded && "opacity-50 cursor-not-allowed"
              )}
            >
              <Search className="w-4 h-4" /> Search
            </button>
            <button 
              onClick={() => setActiveTab('gallery')}
              disabled={!isProjectLoaded}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'gallery' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50",
                !isProjectLoaded && "opacity-50 cursor-not-allowed"
              )}
            >
              <Grid className="w-4 h-4" /> Gallery
            </button>
            <button 
              onClick={() => setActiveTab('visualization')}
              disabled={!isProjectLoaded}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'visualization' ? "bg-primary text-white shadow-lg" : "text-slate-400 hover:text-white hover:bg-slate-700/50",
                !isProjectLoaded && "opacity-50 cursor-not-allowed"
              )}
            >
              <Box className="w-4 h-4" /> 3D Space
            </button>
          </nav>

          <div className="flex items-center gap-4 text-sm">
             <div className="flex items-center gap-2 bg-surface border border-slate-700 rounded-full px-3 py-1">
                 <User className="w-4 h-4 text-primary" />
                 <span className="text-xs font-medium hidden sm:inline">{currentUser}</span>
             </div>

             <button 
                onClick={() => setIsChangePasswordModalOpen(true)}
                className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-700/50 transition-colors"
                title="Change Password"
             >
                <Lock className="w-5 h-5" />
             </button>

             <button 
                onClick={handleLogout}
                className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-700/50 transition-colors"
                title="Sign Out"
             >
                <LogOut className="w-5 h-5" />
             </button>
          </div>
        </div>
      </header>

      {/* Mobile Nav */}
      <div className="md:hidden flex p-2 justify-center gap-2 border-b border-slate-800 bg-surface">
         <button onClick={() => setActiveTab('search')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'search' ? 'bg-slate-700 text-white' : 'text-slate-400')}>Search</button>
         <button onClick={() => setActiveTab('gallery')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'gallery' ? 'bg-slate-700 text-white' : 'text-slate-400')}>Gallery</button>
         <button onClick={() => setActiveTab('visualization')} className={cn("flex-1 py-2 text-xs font-medium rounded-lg text-center", activeTab === 'visualization' ? 'bg-slate-700 text-white' : 'text-slate-400')}>3D Space</button>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {!isProjectLoaded ? (
            <div className="min-h-[60vh] flex flex-col items-center justify-center animate-in fade-in zoom-in-95">
                <div className="w-24 h-24 bg-slate-800 rounded-3xl flex items-center justify-center mb-6 border border-slate-700 shadow-xl">
                    <FolderPlus className="w-12 h-12 text-primary" />
                </div>
                <h2 className="text-2xl font-bold mb-2">Welcome, {currentUser}</h2>
                <p className="text-slate-400 max-w-md text-center mb-8">
                    To get started, please create a new project with your image dataset or select an existing one from the menu.
                </p>
                <div className="flex gap-4">
                    <button 
                        onClick={() => setIsCreateModalOpen(true)}
                        className="px-6 py-3 bg-primary hover:bg-secondary text-white rounded-xl font-semibold shadow-lg shadow-primary/25 transition-all flex items-center gap-2"
                    >
                        <Plus className="w-5 h-5" /> Create Project
                    </button>
                    {projects.length > 0 && (
                        <button 
                            onClick={() => setIsProjectDropdownOpen(true)}
                            className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-semibold border border-slate-700 transition-all flex items-center gap-2"
                        >
                            <Layers className="w-5 h-5" /> Select Existing
                        </button>
                    )}
                </div>
            </div>
        ) : (
            <>
                {activeTab === 'search' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-500">
                    <div className="lg:col-span-4 space-y-6">
                    <div className="bg-surface rounded-2xl p-6 border border-slate-700 shadow-xl">
                        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <ImageIcon className="w-5 h-5 text-slate-400" />
                        Query Image
                        </h2>
                        <div 
                        onClick={() => fileInputRef.current?.click()}
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
                            <img src={previewUrl} alt="Query" className="w-full h-full object-contain p-2 z-10" />
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
                            {isSearching ? <><Loader2 className="w-5 h-5 animate-spin" /> Searching...</> : <><Search className="w-5 h-5" /> Find Similar Images</>}
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
                    <div className="lg:col-span-8">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-2xl font-bold">Results</h2>
                        {results.length > 0 && <span className="text-slate-400 text-sm">Found {results.length} matches</span>}
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
                                onError={(e) => { (e.target as HTMLImageElement).src = 'https://placehold.co/400x400?text=Image+Not+Found'; }}
                                />
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
                                <span className="font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">{result.distance.toFixed(4)}</span>
                                </div>
                            </div>
                            </div>
                        ))}
                        </div>
                    )}
                    </div>
                </div>
                )}
                {activeTab === 'gallery' && (
                <div className="animate-in fade-in duration-500">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-2xl font-bold">Database Gallery</h2>
                            <p className="text-slate-400 text-sm">Browse all indexed images</p>
                        </div>
                        <button 
                            onClick={() => setIsUploadModalOpen(true)}
                            className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors text-xs font-medium flex items-center gap-2"
                        >
                            <Plus className="w-4 h-4" /> Add Image
                        </button>
                    </div>
                    <GalleryGrid versionKey={dataVersion} onZoom={(img) => setZoomedImage({...img, distance: 0})} />
                </div>
                )}
                {activeTab === 'visualization' && (
                <div className="animate-in fade-in duration-500">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-2xl font-bold">Vector Space</h2>
                            <p className="text-slate-400 text-sm">3D visualization of image embeddings using PCA</p>
                        </div>
                    </div>
                    <VectorSpace versionKey={dataVersion} onPointClick={handlePointClick} />
                </div>
                )}
            </>
        )}

      </main>

      {/* Lightbox / Modal */}
      {zoomedImage && (
        <div 
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/95 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setZoomedImage(null)}
        >
          <div className="absolute top-6 right-6 flex items-center gap-3 z-50">
             <button
               onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteImage();
               }}
               disabled={isDeleting}
               className="p-2 rounded-full bg-rose-500/10 hover:bg-rose-500 text-rose-400 hover:text-white transition-colors border border-rose-500/20"
               title="Delete Image"
             >
               {isDeleting ? <Loader2 className="w-6 h-6 animate-spin"/> : <Trash2 className="w-6 h-6" />}
             </button>
             <button 
                onClick={() => setZoomedImage(null)}
                className="p-2 rounded-full bg-slate-800/50 hover:bg-slate-700 text-white/80 hover:text-white transition-colors border border-slate-700"
             >
               <X className="w-6 h-6" />
             </button>
          </div>
          <div className="relative flex flex-col items-center max-w-full" onClick={(e) => e.stopPropagation()}>
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

      {/* Change Password Modal */}
      {isChangePasswordModalOpen && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in zoom-in-95">
            <div className="bg-surface border border-slate-700 w-full max-w-sm rounded-2xl p-6 shadow-2xl relative">
                <button 
                  onClick={() => setIsChangePasswordModalOpen(false)} 
                  className="absolute top-4 right-4 text-slate-400 hover:text-white"
                >
                  <X className="w-5 h-5"/>
                </button>
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><Lock className="w-5 h-5 text-primary"/> Change Password</h3>
                
                <form onSubmit={handleChangePassword} className="space-y-4">
                    <div>
                        <label className="block text-xs text-slate-400 mb-1">Old Password</label>
                        <input 
                            type="password" 
                            required
                            className="w-full bg-dark border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-primary outline-none"
                            value={oldPass}
                            onChange={e => setOldPass(e.target.value)}
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-slate-400 mb-1">New Password</label>
                        <input 
                            type="password" 
                            required
                            className="w-full bg-dark border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-primary outline-none"
                            value={newPass}
                            onChange={e => setNewPass(e.target.value)}
                        />
                    </div>
                    
                    {changePassMsg && (
                        <div className={cn("text-xs p-2 rounded flex items-center gap-2", changePassMsg.type === 'success' ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400")}>
                             {changePassMsg.type === 'success' ? <CheckCircle2 className="w-3 h-3"/> : <AlertCircle className="w-3 h-3"/>}
                             {changePassMsg.text}
                        </div>
                    )}

                    <button 
                        type="submit"
                        disabled={!oldPass || !newPass || isChangingPass}
                        className="w-full py-2 bg-primary hover:bg-secondary rounded-xl font-semibold text-white mt-2 disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                        {isChangingPass ? <Loader2 className="w-4 h-4 animate-spin"/> : "Update Password"}
                    </button>
                </form>
            </div>
        </div>
      )}

      {/* New Project Modal */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in zoom-in-95">
            <div className="bg-surface border border-slate-700 w-full max-w-md rounded-2xl p-6 shadow-2xl relative">
                <button 
                  onClick={() => !isCreatingProject && setIsCreateModalOpen(false)} 
                  className="absolute top-4 right-4 text-slate-400 hover:text-white"
                  disabled={isCreatingProject}
                >
                  <X className="w-5 h-5"/>
                </button>
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><FolderPlus className="w-5 h-5 text-primary"/> New Project</h3>
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-1">Project Name</label>
                        <input 
                            type="text" 
                            className="w-full bg-dark border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-primary outline-none"
                            placeholder="My Awesome Dataset"
                            value={newProjectName}
                            onChange={e => setNewProjectName(e.target.value)}
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-1">Select Image Folder</label>
                        <input 
                            type="file" 
                            // @ts-ignore
                            webkitdirectory=""
                            directory=""
                            multiple
                            className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-slate-700 file:text-white hover:file:bg-slate-600 cursor-pointer"
                            onChange={e => setNewProjectFiles(e.target.files)}
                        />
                        <p className="text-xs text-slate-500 mt-1">Select a folder to upload all images within it.</p>
                        {newProjectFiles && (
                           <p className="text-xs text-primary mt-1">{newProjectFiles.length} files found</p>
                        )}
                    </div>
                    <button 
                        onClick={handleCreateProject}
                        disabled={!newProjectName || !newProjectFiles || isCreatingProject}
                        className="w-full py-2.5 bg-primary hover:bg-secondary rounded-xl font-semibold text-white mt-4 disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                        {isCreatingProject ? <><Loader2 className="w-4 h-4 animate-spin"/> Uploading...</> : "Create & Index"}
                    </button>
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
                        uploadStatus === 'success' ? "bg-emerald-500 text-white" : "bg-primary hover:bg-secondary text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    )}
                 >
                    {isUploading ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</> : uploadStatus === 'success' ? <><CheckCircle2 className="w-5 h-5" /> Added Successfully</> : "Upload & Index"}
                 </button>
             </div>
         </div>
      )}
    </div>
  );
}
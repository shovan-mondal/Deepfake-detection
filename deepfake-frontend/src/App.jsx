import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  History,
  Settings,
  Info,
  ShieldCheck,
  Activity,
  UploadCloud,
  XCircle,
  CheckCircle,
  Brain,
  BarChart3,
  Github,
  PieChart,
  X
} from "lucide-react";

// --- IMPORT IMAGES (Must match your folder structure exactly) ---
import chart1 from './datavisual_pics/chart1.jpeg';
import chart2 from './datavisual_pics/chart2.jpeg';
import chart3 from './datavisual_pics/chart3.jpeg';
import chart4 from './datavisual_pics/chart4.jpeg';
import chart5 from './datavisual_pics/chart5.jpeg';
import predictionGrid from './datavisual_pics/prediction_grid.jpeg';
import srmAnaly from './datavisual_pics/srm_analy.jpeg';

// --- 1. Visualizations Modal Component ---
const VisualizationsModal = ({ onClose }) => {
  
  const visualData = {
    charts: [
      { title: "Confusion Matrix", src: chart1 },
      { title: "Loss Graph",       src: chart2 },
      { title: "Accuracy Curve",   src: chart3 },
      { title: "ROC Curve",        src: chart4 },
      { title: "F1 Score Trend",   src: chart5 },
    ],
    prediction: {
      title: "Prediction Confidence Grid",
      src: predictionGrid
    },
    srm: {
      title: "SRM Noise Residual Analysis",
      src: srmAnaly
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.95, y: 20 }}
        className="bg-[#0f172a] border border-white/10 rounded-2xl p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto shadow-2xl relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors z-10"
        >
          <X size={24} />
        </button>

        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3 border-b border-white/10 pb-4">
          <PieChart className="text-pink-500" /> Data Visualizations
        </h3>

        {/* --- SECTION 1: Charts & Graphs --- */}
        <div className="mb-10">
          <h4 className="text-xl font-semibold text-indigo-400 mb-4 flex items-center gap-2">
            <span className="bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded text-sm">01</span>
            Analysis Charts & Graphs
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {visualData.charts.map((img, index) => (
              <div key={index} className="space-y-2 group">
                {/* UPDATED: Changed h-48 to h-56 and object-cover to object-contain */}
                <div className="rounded-xl overflow-hidden border border-white/10 bg-black/40 relative h-56 flex items-center justify-center p-2">
                  <img 
                    src={img.src} 
                    alt={img.title} 
                    className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-500"
                  />
                </div>
                <p className="text-center text-xs text-slate-400 font-mono mt-2">{img.title}</p>
              </div>
            ))}
          </div>
        </div>

        {/* --- SECTION 2: Prediction Grid --- */}
        <div className="mb-10">
          <h4 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center gap-2">
            <span className="bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded text-sm">02</span>
            Prediction Grid
          </h4>
          <div className="rounded-xl overflow-hidden border border-white/10 bg-black/40 p-2">
            <img 
              src={visualData.prediction.src} 
              alt={visualData.prediction.title} 
              className="w-full h-auto max-h-[400px] object-contain mx-auto"
            />
          </div>
          <p className="mt-2 text-sm text-slate-400 text-center">{visualData.prediction.title}</p>
        </div>

        {/* --- SECTION 3: SRM Noise Residual --- */}
        <div className="mb-4">
          <h4 className="text-xl font-semibold text-emerald-400 mb-4 flex items-center gap-2">
             <span className="bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded text-sm">03</span>
             SRM Noise Residual
          </h4>
          <div className="rounded-xl overflow-hidden border border-white/10 bg-black/40 p-2">
            <img 
              src={visualData.srm.src} 
              alt={visualData.srm.title} 
              className="w-full h-auto max-h-[400px] object-contain mx-auto"
            />
          </div>
          <p className="mt-2 text-sm text-slate-400 text-center">{visualData.srm.title}</p>
        </div>

      </motion.div>
    </motion.div>
  );
};

// --- 2. Sidebar Button Component ---
const SidebarButton = ({ icon: Icon, label, value, activePage, setPage }) => (
  <button
    onClick={() => setPage(value)}
    className={`relative flex items-center gap-3 px-5 py-3 rounded-xl transition-all duration-300 w-full text-left font-medium 
      ${activePage === value ? "text-white" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}
  >
    <Icon size={20} />
    <span className="text-base">{label}</span>
    
    {activePage === value && (
      <motion.div
        layoutId="active-pill"
        className="absolute inset-0 bg-gradient-to-r from-indigo-600/20 to-cyan-500/20 border border-indigo-500/30 rounded-xl -z-10"
        initial={false}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      />
    )}
  </button>
);

// --- 3. Result Card Component ---
const ResultCard = ({ result }) => {
  const isHighConfidence = result.confidence > 80;
  const barColor = isHighConfidence ? "bg-green-500" : "bg-red-500";
  const textColor = isHighConfidence ? "text-green-400" : "text-red-400";
  const isReal = result.prediction === "REAL";
  const badgeColor = isReal ? "bg-green-500/20 text-green-400 border-green-500/30" : "bg-red-500/20 text-red-400 border-red-500/30";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card h-full flex flex-col"
    >
      <h3 className="card-title flex items-center gap-2">
        <Activity size={20} className="text-indigo-400"/> Analysis Result
      </h3>

      <div className="space-y-6 mt-6">
        <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg border border-white/5">
          <span className="text-slate-400">Verdict</span>
          <span className={`px-4 py-1.5 rounded-full text-sm font-bold border ${badgeColor} flex items-center gap-2`}>
            {isReal ? <CheckCircle size={14}/> : <XCircle size={14}/>}
            {result.prediction}
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Confidence Score</span>
            <span className={`font-mono font-bold ${textColor}`}>
              {result.confidence.toFixed(2)}%
            </span>
          </div>
          
          <div className="h-4 w-full bg-slate-900 rounded-full overflow-hidden border border-white/5 p-[2px]">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${result.confidence}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className={`h-full rounded-full shadow-[0_0_12px_rgba(0,0,0,0.5)] ${barColor}`}
            />
          </div>
          
          <p className="text-xs text-slate-500 mt-2 text-right">
            {isHighConfidence 
              ? "AI is highly certain of this result." 
              : "Low confidence. Please verify manually."}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

// --- 4. Dashboard Page ---
const Dashboard = ({ file, setFile, preview, setPreview, handleUpload, loading, result }) => (
  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
    <div className="card flex flex-col">
      <h3 className="card-title flex items-center gap-2 mb-4">
        <UploadCloud size={20} className="text-indigo-400"/> Upload Image
      </h3>

      <div className="h-64 flex flex-col justify-center items-center bg-black/20 border-2 border-dashed border-slate-700 rounded-xl p-2 relative hover:border-indigo-500/50 transition-colors">
        {preview ? (
          <img 
            src={preview} 
            alt="Preview" 
            className="w-full h-full object-contain rounded-lg" 
          />
        ) : (
          <div className="text-center p-6">
            <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-3 text-slate-500">
              <UploadCloud size={24} />
            </div>
            <p className="text-slate-400 font-medium text-sm">Drag & drop or click to upload</p>
            <p className="text-slate-600 text-xs mt-1">Supports JPG, PNG</p>
          </div>
        )}
        
        <input
          type="file"
          accept="image/*"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={(e) => {
            const selected = e.target.files[0];
            if (selected) {
              setFile(selected);
              setPreview(URL.createObjectURL(selected));
            }
          }}
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={loading || !file}
        className={`primary-btn w-full mt-6 py-3 text-base flex justify-center items-center gap-3
          ${!file ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02] active:scale-[0.98]'}`}
      >
        {loading ? <div className="loader" /> : "Run Detection"}
      </button>
    </div>

    {result ? (
      <ResultCard result={result} />
    ) : (
      <div className="card h-full min-h-[300px] flex flex-col justify-center items-center text-center p-12 opacity-50 border-dashed">
        <Activity size={48} className="text-slate-700 mb-4"/>
        <p className="text-slate-500 font-medium">Run a detection to see analysis results here.</p>
      </div>
    )}
  </div>
);

// --- 5. History Page ---
const HistoryList = ({ history }) => (
  <div className="space-y-6 max-w-4xl mx-auto pb-10">
    <div className="flex justify-between items-end mb-6">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        <History className="text-indigo-400"/> Recent Analysis
      </h2>
      <span className="text-sm text-slate-500">{history.length} items</span>
    </div>

    {history.length === 0 ? (
      <div className="text-center py-20 text-slate-500 bg-black/20 rounded-2xl border border-white/5">
        <History size={48} className="mx-auto mb-4 opacity-20"/>
        <p>No history available yet.</p>
      </div>
    ) : (
      <div className="grid gap-4">
        {history.map((item) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="history-card group hover:bg-slate-800/50 transition-colors"
          >
            <img src={item.image} alt="History thumbnail" className="border border-white/10" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3">
                <span className={`text-sm font-bold px-2 py-0.5 rounded ${item.prediction === 'REAL' ? 'text-green-400 bg-green-500/10' : 'text-red-400 bg-red-500/10'}`}>
                  {item.prediction}
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {new Date(item.id).toLocaleTimeString()}
                </span>
              </div>
              <div className="w-full bg-slate-900 h-1.5 mt-2 rounded-full overflow-hidden opacity-50 group-hover:opacity-100 transition-opacity">
                 <div className={`h-full ${item.confidence > 80 ? 'bg-green-500' : 'bg-red-500'}`} style={{width: `${item.confidence}%`}} />
              </div>
            </div>
            <span className="font-mono text-slate-400 font-bold">{item.confidence.toFixed(1)}%</span>
          </motion.div>
        ))}
      </div>
    )}
  </div>
);

// --- 6. About Page ---
const AboutPage = () => {
  const [showVisuals, setShowVisuals] = useState(false);

  return (
    <div className="max-w-5xl mx-auto pb-10">
      
      {/* Modal Popup */}
      <AnimatePresence>
        {showVisuals && <VisualizationsModal onClose={() => setShowVisuals(false)} />}
      </AnimatePresence>

      <div className="card p-8 md:p-10 border border-white/10 bg-[#0f172a]/50 backdrop-blur-md">
        
        <div className="flex items-center gap-4 mb-2">
          <Brain className="text-pink-500" size={32} />
          <h2 className="text-3xl font-bold text-white">Deepfake Detector</h2>
        </div>
        
        <div className="text-xs font-mono text-slate-500 mb-8 pl-1">
          Version 3.0.0 • Stable • State-of-the-Art
        </div>

        <p className="text-slate-300 leading-relaxed mb-10 text-lg max-w-3xl">
          This system detects AI-generated deepfake images using a dual-stream deep learning architecture optimized for real-time inference.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <div className="p-6 rounded-xl bg-[#020617] border border-white/5 hover:border-indigo-500/30 transition-colors">
            <h4 className="text-slate-400 text-sm font-bold uppercase tracking-wider mb-2">Model</h4>
            <p className="text-white font-medium text-lg">Deepfake Detector V3</p>
          </div>
          <div className="p-6 rounded-xl bg-[#020617] border border-white/5 hover:border-indigo-500/30 transition-colors">
            <h4 className="text-slate-400 text-sm font-bold uppercase tracking-wider mb-2">Architecture</h4>
            <p className="text-white font-medium text-lg">MobileNetV2 + SRM + Gated Fusion</p>
          </div>
          <div className="p-6 rounded-xl bg-[#020617] border border-white/5 hover:border-indigo-500/30 transition-colors">
            <h4 className="text-slate-400 text-sm font-bold uppercase tracking-wider mb-2">Datasets</h4>
            <p className="text-slate-300">CIFAKE, FaceForensics++, Celeb-DF</p>
          </div>
          <div className="p-6 rounded-xl bg-[#020617] border border-white/5 hover:border-indigo-500/30 transition-colors">
            <h4 className="text-slate-400 text-sm font-bold uppercase tracking-wider mb-2">Inference</h4>
            <p className="text-slate-300">TensorFlow Lite</p>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="mb-10">
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="text-indigo-400" size={24} />
            <h3 className="text-xl font-bold text-white">Performance Metrics</h3>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            <div className="p-4 rounded-lg bg-[#020617] border border-white/5">
              <p className="text-slate-500 text-xs font-bold uppercase mb-1">AUC</p>
              <p className="text-green-400 font-bold text-xl">99.13%</p>
            </div>
            <div className="p-4 rounded-lg bg-[#020617] border border-white/5">
              <p className="text-slate-500 text-xs font-bold uppercase mb-1">Accuracy</p>
              <p className="text-white font-bold text-xl">94.78%</p>
            </div>
            <div className="p-4 rounded-lg bg-[#020617] border border-white/5">
              <p className="text-slate-500 text-xs font-bold uppercase mb-1">Precision</p>
              <p className="text-white font-bold text-xl">93.46%</p>
            </div>
            <div className="p-4 rounded-lg bg-[#020617] border border-white/5">
              <p className="text-slate-500 text-xs font-bold uppercase mb-1">Recall</p>
              <p className="text-white font-bold text-xl">96.21%</p>
            </div>
            <div className="p-4 rounded-lg bg-[#020617] border border-white/5">
              <p className="text-slate-500 text-xs font-bold uppercase mb-1">F1 Score</p>
              <p className="text-white font-bold text-xl">94.81%</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-6 border-t border-white/5">
          <a 
            href="https://github.com/shovan-mondal/Deepfake-detection/releases/tag/v3.0.0" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-indigo-400 hover:text-indigo-300 text-sm transition-colors"
          >
            <Github size={16} />
            <span className="hover:underline">View Full Release Notes on GitHub</span>
          </a>

          <button 
            onClick={() => setShowVisuals(true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 text-sm font-medium transition-all"
          >
            <PieChart size={16} />
            Data Visualizations
          </button>
        </div>

      </div>
    </div>
  );
};

// --- 7. Main App Structure ---
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [page, setPage] = useState("dashboard");

  const handleUpload = async () => {
    if (!file) return alert("Please select an image");
    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server Error");
      const data = await response.json();
      setResult(data);

      setHistory((prev) => [
        {
          id: Date.now(),
          prediction: data.prediction,
          confidence: data.confidence,
          image: preview,
        },
        ...prev.slice(0, 9), 
      ]);
    } catch (e) {
      alert("Backend not reachable or Error occurred 😢");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-white flex font-sans selection:bg-indigo-500/30">
      
      {/* Sidebar */}
      <aside className="sidebar fixed h-screen z-10 hidden md:flex w-[280px]">
        <div>
          <div className="logo">
            <ShieldCheck className="text-indigo-500" size={32} />
            <span className="tracking-tight">DeepScan<span className="text-indigo-500">.AI</span></span>
          </div>

          <nav className="sidebar-nav mt-8">
            <SidebarButton icon={LayoutDashboard} label="Dashboard" value="dashboard" activePage={page} setPage={setPage} />
            <SidebarButton icon={History} label="History" value="history" activePage={page} setPage={setPage} />
            <SidebarButton icon={Settings} label="Settings" value="settings" activePage={page} setPage={setPage} />
            <SidebarButton icon={Info} label="About" value="about" activePage={page} setPage={setPage} />
          </nav>
        </div>

        <div className="sidebar-footer">
          <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-slate-900/50 border border-white/5 w-full">
            <div className="relative">
              <Activity size={18} className="text-green-400" />
              <span className="absolute inset-0 bg-green-400 blur-[8px] opacity-40 animate-pulse"></span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-white">System Online</span>
              <span className="text-[10px] text-slate-400">v3.0.0 Stable</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 md:ml-[280px] p-6 md:p-10 relative">
        <div className="absolute top-0 left-0 w-full h-[300px] bg-gradient-to-b from-indigo-900/10 to-transparent -z-10 pointer-events-none" />

        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              {page.charAt(0).toUpperCase() + page.slice(1)}
            </h1>
            <p className="text-slate-400 text-sm mt-1">Real-time deepfake detection engine</p>
          </div>
          <span className="badge-online flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-white animate-pulse"/> LIVE
          </span>
        </div>

        {/* Page Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{ duration: 0.25 }}
            className="min-h-[calc(100vh-140px)]"
          >
            {page === "dashboard" && (
              <Dashboard 
                file={file} 
                setFile={setFile} 
                preview={preview} 
                setPreview={setPreview}
                handleUpload={handleUpload}
                loading={loading}
                result={result}
              />
            )}
            
            {page === "history" && <HistoryList history={history} />}
            
            {page === "settings" && (
               <div className="card p-10 text-center opacity-60">
                 <Settings size={48} className="mx-auto mb-4"/>
                 <h3 className="text-xl font-bold">Settings</h3>
                 <p>Configuration options coming in next update.</p>
               </div>
            )}
            
            {page === "about" && <AboutPage />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
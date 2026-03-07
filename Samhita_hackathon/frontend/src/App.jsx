import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import UploadPage from './pages/Upload';
import Dashboard from './pages/Dashboard';
import NullAnalysis from './pages/NullAnalysis';
import Duplicates from './pages/Duplicates';
import Outliers from './pages/Outliers';
import Consistency from './pages/Consistency';
import Distribution from './pages/Distribution';
import Insights from './pages/Insights';

export default function App() {
  const [datasetId, setDatasetId] = useState(null);
  const [datasetMeta, setDatasetMeta] = useState(null);

  const handleDatasetLoaded = (id, meta) => {
    setDatasetId(id);
    setDatasetMeta(meta);
  };

  return (
    <Router>
      <div className="flex min-h-screen">
        <Sidebar datasetId={datasetId} />
        <main className="flex-1 ml-64 p-8">
          {/* Top bar */}
          {datasetMeta && (
            <div className="mb-6 flex items-center gap-3 text-xs">
              <span className="bg-brand-50 text-brand-600 px-2.5 py-1 rounded-full font-semibold">
                {datasetMeta.filename || 'Dataset'}
              </span>
              <span className="text-text-muted">
                {(datasetMeta.rows || 0).toLocaleString()} rows × {(datasetMeta.columns || datasetMeta.column_names?.length || 0)} cols
              </span>
              <button
                onClick={() => { setDatasetId(null); setDatasetMeta(null); }}
                className="ml-auto text-xs text-red-500 hover:text-red-700 transition-colors"
              >
                Clear Dataset
              </button>
            </div>
          )}

          <Routes>
            <Route path="/" element={<UploadPage onDatasetLoaded={handleDatasetLoaded} />} />
            <Route path="/dashboard" element={datasetId ? <Dashboard datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/nulls" element={datasetId ? <NullAnalysis datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/duplicates" element={datasetId ? <Duplicates datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/outliers" element={datasetId ? <Outliers datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/consistency" element={datasetId ? <Consistency datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/distribution" element={datasetId ? <Distribution datasetId={datasetId} /> : <Navigate to="/" />} />
            <Route path="/insights" element={datasetId ? <Insights datasetId={datasetId} /> : <Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

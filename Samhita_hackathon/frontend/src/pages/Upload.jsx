import { useState } from 'react';
import { Upload as UploadIcon, Sparkles, FileSpreadsheet, Loader2 } from 'lucide-react';
import { uploadDataset, generateDataset, getPreview } from '../services/api';

export default function UploadPage({ onDatasetLoaded }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [preview, setPreview] = useState(null);
    const [dragOver, setDragOver] = useState(false);

    const handleFile = async (file) => {
        setLoading(true);
        setError(null);
        try {
            const { data } = await uploadDataset(file);
            const previewRes = await getPreview(data.dataset_id, 20);
            setPreview({ ...data, rows: previewRes.data.rows });
            onDatasetLoaded(data.dataset_id, data);
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
        }
        setLoading(false);
    };

    const handleGenerate = async () => {
        setLoading(true);
        setError(null);
        try {
            const { data } = await generateDataset(20000);
            const previewRes = await getPreview(data.dataset_id, 20);
            setPreview({ ...data, rows: previewRes.data.rows });
            onDatasetLoaded(data.dataset_id, data);
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
        }
        setLoading(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    };

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Upload Dataset</h2>
            <p className="text-text-muted text-sm mb-6">Upload a CSV/Excel file or generate synthetic data to analyze.</p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Upload zone */}
                <div
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`glass-card p-8 flex flex-col items-center justify-center cursor-pointer
            transition-all duration-300 min-h-[240px]
            ${dragOver ? 'border-brand-400 bg-brand-50/50 scale-[1.01]' : 'hover:border-brand-300'}`}
                    onClick={() => document.getElementById('file-input').click()}
                >
                    <UploadIcon className={`w-10 h-10 mb-4 ${dragOver ? 'text-brand-400' : 'text-brand-300'}`} />
                    <p className="text-sm font-semibold text-text-strong mb-1">
                        {dragOver ? 'Drop your file here' : 'Click to upload or drag & drop'}
                    </p>
                    <p className="text-xs text-text-muted">CSV, XLSX (max 100k rows)</p>
                    <input
                        id="file-input"
                        type="file"
                        accept=".csv,.xlsx,.xls"
                        className="hidden"
                        onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                    />
                </div>

                {/* Generate synthetic */}
                <div
                    onClick={handleGenerate}
                    className="glass-card p-8 flex flex-col items-center justify-center cursor-pointer
            hover:border-accent transition-all duration-300 min-h-[240px] group"
                >
                    <Sparkles className="w-10 h-10 mb-4 text-accent group-hover:scale-110 transition-transform" />
                    <p className="text-sm font-semibold text-text-strong mb-1">Generate Test Data</p>
                    <p className="text-xs text-text-muted">20,000 rows of realistic sample data</p>
                </div>
            </div>

            {loading && (
                <div className="flex items-center gap-2 text-brand-500 mb-4">
                    <Loader2 className="w-4 h-4 animate-spin" /> <span className="text-sm">Processing...</span>
                </div>
            )}
            {error && (
                <div className="bg-red-50 text-red-700 px-4 py-3 rounded-xl text-sm mb-4">{error}</div>
            )}

            {preview && (
                <div className="animate-fade-in">
                    <div className="flex items-center gap-2 mb-3">
                        <FileSpreadsheet className="w-4 h-4 text-brand-500" />
                        <h3 className="text-sm font-semibold text-text-strong">
                            Dataset Loaded: {preview.rows?.length ? 'Showing first 20 rows' : ''}
                        </h3>
                        <span className="text-xs bg-brand-50 text-brand-600 px-2 py-0.5 rounded-full font-medium">
                            {preview.rows_count || preview.rows?.length || 0} rows × {preview.columns?.length || 0} cols
                        </span>
                    </div>
                    <div className="overflow-x-auto glass-card">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="border-b border-border">
                                    {preview.column_names?.map((col) => (
                                        <th key={col} className="px-3 py-2 text-left font-semibold text-text-muted whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {preview.rows?.slice(0, 20).map((row, i) => (
                                    <tr key={i} className="border-b border-border/50 hover:bg-brand-50/30">
                                        {preview.column_names?.map((col) => (
                                            <td key={col} className="px-3 py-1.5 whitespace-nowrap text-text-strong">
                                                {row[col] === null ? (
                                                    <span className="text-red-400 italic">null</span>
                                                ) : (
                                                    String(row[col]).slice(0, 40)
                                                )}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}

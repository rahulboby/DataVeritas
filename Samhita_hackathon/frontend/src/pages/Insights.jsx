import { useState, useEffect } from 'react';
import { generateInsights } from '../services/api';
import { Loader2, Sparkles } from 'lucide-react';
import InsightPanel from '../components/InsightPanel';

export default function Insights({ datasetId }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        generateInsights(datasetId).then(r => setData(r.data)).finally(() => setLoading(false));
    }, [datasetId]);

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">AI Insights</h2>
            <p className="text-text-muted text-sm mb-6">AI-generated data quality insights ranked by severity.</p>

            {/* Summary badges */}
            <div className="flex gap-3 mb-6 flex-wrap">
                {data.by_severity.critical > 0 && (
                    <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-red-50 text-red-700 border border-red-200">
                        {data.by_severity.critical} Critical
                    </span>
                )}
                {data.by_severity.warning > 0 && (
                    <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-amber-50 text-amber-700 border border-amber-200">
                        {data.by_severity.warning} Warnings
                    </span>
                )}
                {data.by_severity.info > 0 && (
                    <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-blue-50 text-blue-700 border border-blue-200">
                        {data.by_severity.info} Info
                    </span>
                )}
                {data.by_severity.good > 0 && (
                    <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                        {data.by_severity.good} Good
                    </span>
                )}
            </div>

            <div className="glass-card p-6">
                <div className="flex items-center gap-2 mb-4">
                    <Sparkles className="w-4 h-4 text-purple-500" />
                    <h3 className="text-sm font-semibold text-text-strong">{data.total} Insights Generated</h3>
                </div>
                <InsightPanel insights={data.insights} />
            </div>
        </div>
    );
}

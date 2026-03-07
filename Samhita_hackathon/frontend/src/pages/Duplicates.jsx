import { useState, useEffect } from 'react';
import { computeDuplicates, computeEntityGraph } from '../services/api';
import { Loader2, Copy, Network } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

export default function Duplicates({ datasetId }) {
    const [data, setData] = useState(null);
    const [graph, setGraph] = useState(null);
    const [loading, setLoading] = useState(true);
    const [graphLoading, setGraphLoading] = useState(false);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        computeDuplicates(datasetId).then(r => setData(r.data)).finally(() => setLoading(false));
    }, [datasetId]);

    const runEntityGraph = async () => {
        setGraphLoading(true);
        try {
            const r = await computeEntityGraph(datasetId, null, 80);
            setGraph(r.data);
        } catch (e) {
            console.error(e);
        }
        setGraphLoading(false);
    };

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    const pieData = [
        { name: 'Unique Rows', value: data.unique_rows, color: '#10b981' },
        { name: 'Duplicate Rows', value: data.duplicate_rows, color: '#ef4444' },
    ];

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Duplicate Analysis</h2>
            <p className="text-text-muted text-sm mb-6">Exact duplicate detection and AI-powered entity resolution.</p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8 stagger-children">
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Uniqueness Score</p>
                    <p className="text-2xl font-bold text-brand-600">{(data.uniqueness_score * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Total Rows</p>
                    <p className="text-2xl font-bold text-text-strong">{data.total_rows.toLocaleString()}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Duplicate Rows</p>
                    <p className="text-2xl font-bold text-red-500">{data.duplicate_rows.toLocaleString()}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Redundant Rows</p>
                    <p className="text-2xl font-bold text-amber-500">{data.redundant_rows.toLocaleString()}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Pie chart */}
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <Copy className="w-4 h-4 text-brand-400" /> Unique vs Duplicate
                    </h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                            <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={3} dataKey="value">
                                {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                            </Pie>
                            <Tooltip formatter={(v) => v.toLocaleString()} />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Entity Graph */}
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <Network className="w-4 h-4 text-purple-500" /> Entity Resolution
                    </h3>
                    {!graph ? (
                        <div className="flex flex-col items-center justify-center h-48">
                            <p className="text-xs text-text-muted mb-3">Run fuzzy entity matching to find near-duplicates</p>
                            <button
                                onClick={runEntityGraph}
                                disabled={graphLoading}
                                className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors flex items-center gap-2"
                            >
                                {graphLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Network className="w-4 h-4" />}
                                Run Entity Graph
                            </button>
                        </div>
                    ) : (
                        <div>
                            <div className="grid grid-cols-2 gap-2 mb-4">
                                <div className="bg-purple-50 rounded-lg p-3 text-center">
                                    <p className="text-xs text-text-muted">Clusters</p>
                                    <p className="text-xl font-bold text-purple-600">{graph.total_clusters}</p>
                                </div>
                                <div className="bg-purple-50 rounded-lg p-3 text-center">
                                    <p className="text-xs text-text-muted">Records in Clusters</p>
                                    <p className="text-xl font-bold text-purple-600">{graph.total_records_in_clusters}</p>
                                </div>
                            </div>
                            <p className="text-xs text-text-muted">
                                Matched on: {graph.columns_used?.join(', ')}
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

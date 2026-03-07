import { useState, useEffect } from 'react';
import { computeNulls } from '../services/api';
import { Loader2, Droplets } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function NullAnalysis({ datasetId }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        computeNulls(datasetId).then(r => setData(r.data)).finally(() => setLoading(false));
    }, [datasetId]);

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    const chartData = data.per_column
        .filter(c => c.null_count > 0)
        .slice(0, 15)
        .map(c => ({ name: c.column.slice(0, 15), nulls: c.null_count, pct: c.null_pct }));

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Null & Completeness Analysis</h2>
            <p className="text-text-muted text-sm mb-6">Detailed breakdown of missing values across all columns.</p>

            {/* KPI row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8 stagger-children">
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Null Score</p>
                    <p className="text-2xl font-bold text-brand-600">{(data.null_score * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Completeness</p>
                    <p className="text-2xl font-bold text-brand-600">{(data.completeness_score * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Total Nulls</p>
                    <p className="text-2xl font-bold text-red-500">{data.total_nulls.toLocaleString()}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Rows with Nulls</p>
                    <p className="text-2xl font-bold text-amber-500">{data.null_rows.toLocaleString()} ({data.null_rows_pct}%)</p>
                </div>
            </div>

            {/* Chart */}
            {chartData.length > 0 && (
                <div className="glass-card p-6 mb-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <Droplets className="w-4 h-4 text-blue-400" />
                        Columns with Missing Values
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={chartData} layout="vertical" barSize={16}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis type="number" tick={{ fontSize: 11, fill: '#516477' }} />
                            <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11, fill: '#516477' }} />
                            <Tooltip formatter={(v) => [v, 'Nulls']} contentStyle={{ borderRadius: 12 }} />
                            <Bar dataKey="nulls" radius={[0, 6, 6, 0]}>
                                {chartData.map((_, i) => (
                                    <Cell key={i} fill={i < 3 ? '#ef4444' : i < 7 ? '#f59e0b' : '#3b82f6'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Per-column table */}
            <div className="glass-card overflow-hidden">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="bg-gray-50 border-b border-border">
                            <th className="px-4 py-2.5 text-left font-semibold text-text-muted text-xs">Column</th>
                            <th className="px-4 py-2.5 text-left font-semibold text-text-muted text-xs">Type</th>
                            <th className="px-4 py-2.5 text-right font-semibold text-text-muted text-xs">Null Count</th>
                            <th className="px-4 py-2.5 text-right font-semibold text-text-muted text-xs">Null %</th>
                            <th className="px-4 py-2.5 text-left font-semibold text-text-muted text-xs">Health</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.per_column.map((col, i) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-brand-50/30">
                                <td className="px-4 py-2 font-medium text-text-strong text-xs">{col.column}</td>
                                <td className="px-4 py-2 text-text-muted text-xs">{col.dtype}</td>
                                <td className="px-4 py-2 text-right text-xs">{col.null_count}</td>
                                <td className="px-4 py-2 text-right text-xs">{col.null_pct}%</td>
                                <td className="px-4 py-2">
                                    <div className="w-24 h-1.5 bg-gray-100 rounded-full">
                                        <div
                                            className="h-full rounded-full"
                                            style={{
                                                width: `${100 - col.null_pct}%`,
                                                backgroundColor: col.null_pct > 20 ? '#ef4444' : col.null_pct > 5 ? '#f59e0b' : '#10b981'
                                            }}
                                        />
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

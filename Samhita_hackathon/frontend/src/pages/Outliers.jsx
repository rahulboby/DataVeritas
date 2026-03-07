import { useState, useEffect } from 'react';
import { computeOutliers } from '../services/api';
import { Loader2, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Outliers({ datasetId }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        computeOutliers(datasetId).then(r => setData(r.data)).finally(() => setLoading(false));
    }, [datasetId]);

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    const chartData = data.per_column
        .filter(c => c.outlier_count > 0)
        .slice(0, 10)
        .map(c => ({ name: c.column.slice(0, 12), count: c.outlier_count, pct: c.outlier_pct }));

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Outlier Analysis</h2>
            <p className="text-text-muted text-sm mb-6">Isolation Forest-based anomaly detection across numeric columns.</p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8 stagger-children">
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Outlier Score</p>
                    <p className="text-2xl font-bold text-brand-600">{(data.outlier_score * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Outlier Rows</p>
                    <p className="text-2xl font-bold text-red-500">{data.total_outlier_rows.toLocaleString()}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Numeric Columns</p>
                    <p className="text-2xl font-bold text-text-strong">{data.numeric_columns}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Affected Columns</p>
                    <p className="text-2xl font-bold text-amber-500">{data.columns_with_outliers}</p>
                </div>
            </div>

            {chartData.length > 0 && (
                <div className="glass-card p-6 mb-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                        Outliers by Column
                    </h3>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={chartData} barSize={30}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#516477' }} />
                            <YAxis tick={{ fontSize: 11, fill: '#516477' }} />
                            <Tooltip formatter={(v) => [v, 'Outliers']} contentStyle={{ borderRadius: 12 }} />
                            <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                                {chartData.map((_, i) => (
                                    <Cell key={i} fill={i === 0 ? '#ef4444' : i < 3 ? '#f59e0b' : '#3b82f6'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Stats table */}
            <div className="glass-card overflow-hidden">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="bg-gray-50 border-b border-border">
                            <th className="px-4 py-2.5 text-left font-semibold text-text-muted text-xs">Column</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Outliers</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Min</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Q1</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Median</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Q3</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.per_column.map((col, i) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-brand-50/30">
                                <td className="px-4 py-2 font-medium text-text-strong text-xs">{col.column}</td>
                                <td className="px-4 py-2 text-right text-xs">
                                    <span className={col.outlier_count > 0 ? 'text-red-500 font-semibold' : ''}>
                                        {col.outlier_count}
                                    </span>
                                </td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.min}</td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.q1}</td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.median}</td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.q3}</td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.max}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

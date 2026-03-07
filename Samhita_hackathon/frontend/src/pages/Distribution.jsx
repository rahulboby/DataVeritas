import { useState, useEffect } from 'react';
import { computeDistribution, computeProfile } from '../services/api';
import { Loader2, PieChart as PieIcon, Search } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const TYPE_COLORS = {
    Numeric: '#3b82f6', Categorical: '#8b5cf6', Datetime: '#f59e0b',
    Boolean: '#10b981', Constant: '#6b7280', Identifier: '#ec4899'
};

export default function Distribution({ datasetId }) {
    const [data, setData] = useState(null);
    const [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        Promise.all([
            computeDistribution(datasetId).then(r => setData(r.data)),
            computeProfile(datasetId).then(r => setProfile(r.data)).catch(() => { }),
        ]).finally(() => setLoading(false));
    }, [datasetId]);

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    const pieData = Object.entries(data.column_types)
        .filter(([_, v]) => v > 0)
        .map(([name, value]) => ({ name, value, color: TYPE_COLORS[name] || '#94a3b8' }));

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Column Distribution</h2>
            <p className="text-text-muted text-sm mb-6">Type breakdown and per-column statistics.</p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Type pie chart */}
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <PieIcon className="w-4 h-4 text-brand-400" /> Column Types
                    </h3>
                    <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                            <Pie data={pieData} cx="50%" cy="50%" outerRadius={90} paddingAngle={3} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                                {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Profile summary */}
                {profile && (
                    <div className="glass-card p-6">
                        <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                            <Search className="w-4 h-4 text-accent" /> Schema Profile
                        </h3>
                        <div className="space-y-3">
                            <div className="flex justify-between text-sm">
                                <span className="text-text-muted">Total Rows</span>
                                <span className="font-semibold">{profile.total_rows.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-text-muted">Total Columns</span>
                                <span className="font-semibold">{profile.total_columns}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-text-muted">Memory</span>
                                <span className="font-semibold">{profile.memory_mb} MB</span>
                            </div>
                            {profile.primary_key_candidates?.length > 0 && (
                                <div className="mt-3 p-3 bg-emerald-50 rounded-lg">
                                    <p className="text-xs font-semibold text-emerald-700 mb-1">Primary Key Candidates</p>
                                    <p className="text-xs text-emerald-600">{profile.primary_key_candidates.join(', ')}</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Column details */}
            <div className="glass-card overflow-hidden">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="bg-gray-50 border-b border-border">
                            <th className="px-4 py-2.5 text-left text-xs font-semibold text-text-muted">Column</th>
                            <th className="px-4 py-2.5 text-left text-xs font-semibold text-text-muted">Type</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Unique</th>
                            <th className="px-4 py-2.5 text-right text-xs font-semibold text-text-muted">Nulls</th>
                            <th className="px-4 py-2.5 text-left text-xs font-semibold text-text-muted">Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.columns.map((col, i) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-brand-50/30">
                                <td className="px-4 py-2 font-medium text-text-strong text-xs">{col.column}</td>
                                <td className="px-4 py-2">
                                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded"
                                        style={{ backgroundColor: `${TYPE_COLORS[col.type] || '#94a3b8'}15`, color: TYPE_COLORS[col.type] || '#94a3b8' }}>
                                        {col.type}
                                    </span>
                                </td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.unique_count}</td>
                                <td className="px-4 py-2 text-right text-xs text-text-muted">{col.null_count}</td>
                                <td className="px-4 py-2 text-xs text-text-muted">
                                    {col.stats && `μ=${col.stats.mean}, σ=${col.stats.std}`}
                                    {col.top_values && Object.keys(col.top_values).slice(0, 3).join(', ')}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

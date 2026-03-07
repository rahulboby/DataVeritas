import { useState, useEffect } from 'react';
import { computeConsistency, discoverRules } from '../services/api';
import { Loader2, ShieldCheck, Wand2 } from 'lucide-react';

export default function Consistency({ datasetId }) {
    const [data, setData] = useState(null);
    const [discovered, setDiscovered] = useState(null);
    const [loading, setLoading] = useState(true);
    const [discoverLoading, setDiscoverLoading] = useState(false);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        computeConsistency(datasetId).then(r => setData(r.data)).finally(() => setLoading(false));
    }, [datasetId]);

    const handleDiscover = async () => {
        setDiscoverLoading(true);
        try {
            const r = await discoverRules(datasetId);
            setDiscovered(r.data);
        } catch (e) { console.error(e); }
        setDiscoverLoading(false);
    };

    if (loading) return <div className="flex justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-brand-400" /></div>;
    if (!data) return <p className="text-text-muted">No data.</p>;

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Consistency Analysis</h2>
            <p className="text-text-muted text-sm mb-6">Business rule validation and automatic rule discovery.</p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-8 stagger-children">
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Consistency Score</p>
                    <p className="text-2xl font-bold text-brand-600">{(data.consistency_score * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Total Violations</p>
                    <p className="text-2xl font-bold text-red-500">{data.total_violations.toLocaleString()}</p>
                </div>
                <div className="metric-card">
                    <p className="text-xs text-text-muted mb-1">Violation Rate</p>
                    <p className="text-2xl font-bold text-amber-500">{data.violation_pct}%</p>
                </div>
            </div>

            {/* Violations table */}
            {data.violations && data.violations.length > 0 && (
                <div className="glass-card overflow-hidden mb-6">
                    <div className="px-4 py-3 bg-gray-50 border-b border-border flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-brand-400" />
                        <h3 className="text-sm font-semibold text-text-strong">Rule Violations</h3>
                    </div>
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-border">
                                <th className="px-4 py-2 text-left font-semibold text-text-muted text-xs">Reason</th>
                                <th className="px-4 py-2 text-right font-semibold text-text-muted text-xs">Count</th>
                                <th className="px-4 py-2 text-right font-semibold text-text-muted text-xs">% of Rows</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.violations.map((v, i) => (
                                <tr key={i} className="border-b border-border/50 hover:bg-brand-50/30">
                                    <td className="px-4 py-2 text-xs text-text-strong">{v.reason}</td>
                                    <td className="px-4 py-2 text-right text-xs text-red-500 font-semibold">{v.count}</td>
                                    <td className="px-4 py-2 text-right text-xs text-text-muted">{v.pct}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Rule Discovery */}
            <div className="glass-card p-6">
                <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                    <Wand2 className="w-4 h-4 text-purple-500" />
                    AI Rule Discovery
                </h3>
                {!discovered ? (
                    <div className="text-center py-6">
                        <p className="text-xs text-text-muted mb-3">Automatically discover data quality rules from patterns</p>
                        <button
                            onClick={handleDiscover}
                            disabled={discoverLoading}
                            className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors flex items-center gap-2 mx-auto"
                        >
                            {discoverLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                            Discover Rules
                        </button>
                    </div>
                ) : (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        <p className="text-xs text-text-muted mb-2">{discovered.total_rules} rules discovered</p>
                        {discovered.discovered_rules.map((rule, i) => (
                            <div key={i} className="flex items-start gap-3 text-xs p-3 bg-purple-50 rounded-lg border border-purple-200">
                                <span className="font-mono text-purple-600 flex-shrink-0">{rule.rule_type}</span>
                                <span className="text-text-strong flex-1">{rule.name}</span>
                                <span className="text-purple-500 font-semibold whitespace-nowrap">
                                    {(rule.confidence * 100).toFixed(0)}% conf
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

import { useState, useEffect } from 'react';
import { computeScore, computeDynamicScore, predictQuality } from '../services/api';
import TrustScoreGauge from '../components/TrustScoreGauge';
import ScoreCard from '../components/ScoreCard';
import { Loader2, TrendingUp, Brain, Zap } from 'lucide-react';
import InteractiveRadarChart from '../components/InteractiveRadarChart';
import { getTrustColor } from '../lib/utils';

export default function Dashboard({ datasetId }) {
    const [scores, setScores] = useState(null);
    const [dynamicScores, setDynamicScores] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!datasetId) return;
        setLoading(true);
        Promise.all([
            computeScore(datasetId).then(r => setScores(r.data)),
            computeDynamicScore(datasetId).then(r => setDynamicScores(r.data)).catch(() => { }),
            predictQuality(datasetId).then(r => setPrediction(r.data)).catch(() => { }),
        ]).finally(() => setLoading(false));
    }, [datasetId]);

    if (loading) return (
        <div className="flex items-center justify-center h-64">
            <Loader2 className="w-6 h-6 animate-spin text-brand-400" />
        </div>
    );

    if (!scores) return <p className="text-text-muted">No data available.</p>;

    const radarMetrics = [
        { name: 'Null Score', score: +(scores.null_score * 100).toFixed(1) },
        { name: 'Completeness', score: +(scores.completeness_score * 100).toFixed(1) },
        { name: 'Uniqueness', score: +(scores.uniqueness_score * 100).toFixed(1) },
        { name: 'Outlier Score', score: +(scores.outlier_score * 100).toFixed(1) },
        { name: 'Consistency', score: +(scores.consistency_score * 100).toFixed(1) },
    ];

    return (
        <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-text-strong mb-1">Trust Score Dashboard</h2>
            <p className="text-text-muted text-sm mb-6">Overall data quality assessment and sub-score breakdown.</p>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                {/* Hero gauge */}
                <div className="glass-card p-6 flex flex-col items-center lg:col-span-1">
                    <TrustScoreGauge score={scores.dq_score} size={200} />
                    <div className="mt-4 text-center">
                        <span className={`text-sm font-bold px-3 py-1 rounded-full ${scores.trust_level === 'Excellent' ? 'bg-emerald-50 text-emerald-700' :
                            scores.trust_level === 'Moderate' ? 'bg-amber-50 text-amber-700' :
                                'bg-red-50 text-red-700'
                            }`}>
                            {scores.trust_level}
                        </span>
                    </div>
                </div>

                {/* Sub-score cards */}
                <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-3 stagger-children">
                    <ScoreCard label="Null Score" value={scores.null_score} help="Rows with zero nulls" />
                    <ScoreCard label="Completeness" value={scores.completeness_score} help="Non-null cell ratio" />
                    <ScoreCard label="Uniqueness" value={scores.uniqueness_score} help="Deduplicated row ratio" />
                    <ScoreCard label="Outlier Score" value={scores.outlier_score} help="Outlier-free ratio" />
                    <ScoreCard label="Consistency" value={scores.consistency_score} help="Rule violations" />
                    {prediction && (
                        <div className="metric-card">
                            <div className="flex items-center gap-1.5 mb-2">
                                <Brain className="w-3.5 h-3.5 text-purple-500" />
                                <span className="text-xs font-medium text-text-muted">AI Predicted</span>
                            </div>
                            <span className="text-2xl font-bold text-purple-600">
                                {(prediction.predicted_quality_score * 100).toFixed(1)}%
                            </span>
                            <p className="text-[11px] text-text-muted mt-1">Random Forest model</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Interactive Radar Chart */}
            <div className="glass-card p-6 mb-6">
                <h3 className="text-sm font-semibold text-text-strong mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-brand-400" />
                    Quality Trust Profile
                </h3>
                <p className="text-xs text-text-muted mb-4">Hover over nodes to explore each quality dimension.</p>
                <InteractiveRadarChart metrics={radarMetrics} size={420} />
            </div>

            {/* Dynamic vs Static */}
            {dynamicScores && (
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-text-strong mb-4 flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-500" />
                        Dynamic vs Static Scoring
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-4 bg-brand-50 rounded-xl">
                            <p className="text-xs text-text-muted mb-1">Static Score</p>
                            <p className="text-2xl font-bold" style={{ color: getTrustColor(dynamicScores.static_score) }}>
                                {(dynamicScores.static_score * 100).toFixed(1)}%
                            </p>
                            <p className="text-[10px] text-text-muted mt-1">Fixed weights</p>
                        </div>
                        <div className="text-center p-4 bg-purple-50 rounded-xl">
                            <p className="text-xs text-text-muted mb-1">Dynamic Score</p>
                            <p className="text-2xl font-bold" style={{ color: getTrustColor(dynamicScores.dynamic_score) }}>
                                {(dynamicScores.dynamic_score * 100).toFixed(1)}%
                            </p>
                            <p className="text-[10px] text-text-muted mt-1">Entropy-based weights</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

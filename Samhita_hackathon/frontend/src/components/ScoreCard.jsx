import { getTrustColor } from '../lib/utils';

export default function ScoreCard({ label, value, help, threshold = 0.9 }) {
    const pct = (value * 100).toFixed(1);
    const color = getTrustColor(value);
    const status = value >= threshold ? 'Good' : 'Review';

    return (
        <div className="metric-card group">
            <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-text-muted">{label}</span>
                <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${value >= threshold
                            ? 'bg-emerald-50 text-emerald-700'
                            : 'bg-amber-50 text-amber-700'
                        }`}
                >
                    {status}
                </span>
            </div>
            <div className="flex items-end gap-2">
                <span className="text-2xl font-bold" style={{ color }}>
                    {pct}%
                </span>
            </div>
            {/* Progress bar */}
            <div className="mt-3 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                />
            </div>
            {help && (
                <p className="mt-2 text-[11px] text-text-muted">{help}</p>
            )}
        </div>
    );
}

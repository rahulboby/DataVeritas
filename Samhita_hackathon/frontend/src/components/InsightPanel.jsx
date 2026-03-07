import { AlertTriangle, CheckCircle, Info, XCircle } from 'lucide-react';

const severityConfig = {
    critical: { icon: XCircle, className: 'severity-critical', iconColor: 'text-red-500' },
    warning: { icon: AlertTriangle, className: 'severity-warning', iconColor: 'text-amber-500' },
    info: { icon: Info, className: 'severity-info', iconColor: 'text-blue-500' },
    good: { icon: CheckCircle, className: 'severity-good', iconColor: 'text-emerald-500' },
};

export default function InsightPanel({ insights }) {
    if (!insights || insights.length === 0) {
        return <p className="text-text-muted text-sm">No insights generated yet.</p>;
    }

    return (
        <div className="space-y-2 stagger-children">
            {insights.map((insight, idx) => {
                const config = severityConfig[insight.severity] || severityConfig.info;
                const Icon = config.icon;

                return (
                    <div
                        key={idx}
                        className={`flex items-start gap-3 px-4 py-3 rounded-xl border ${config.className}`}
                    >
                        <Icon className={`w-4 h-4 mt-0.5 flex-shrink-0 ${config.iconColor}`} />
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium">{insight.message}</p>
                            <span className="text-[10px] font-semibold uppercase tracking-wider opacity-60">
                                {insight.category}
                            </span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

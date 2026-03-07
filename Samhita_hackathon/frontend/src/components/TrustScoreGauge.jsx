import { getTrustColor, formatPct } from '../lib/utils';

export default function TrustScoreGauge({ score, size = 180, label = 'DQ Score' }) {
    const color = getTrustColor(score);
    const radius = (size - 20) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - score * circumference;

    return (
        <div className="score-gauge" style={{ width: size, height: size }}>
            <svg width={size} height={size}>
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="10"
                />
                {/* Score arc */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    className="animate-score-reveal"
                    style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
                />
            </svg>
            {/* Center text */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-extrabold" style={{ color }}>
                    {formatPct(score)}
                </span>
                <span className="text-xs text-text-muted font-medium mt-1">{label}</span>
            </div>
        </div>
    );
}

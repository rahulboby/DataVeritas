import { useState, useEffect, useRef } from 'react';

const METRIC_DESCRIPTIONS = {
    'Null Score': 'Ratio of rows with zero null values',
    'Completeness': 'Non-null cell ratio across the dataset',
    'Uniqueness': 'Ratio of deduplicated rows to total rows',
    'Consistency': 'Percentage of rows passing business rules',
    'Outlier Score': 'Ratio of outlier-free rows (Isolation Forest)',
};

function getNodeColor(score) {
    if (score >= 90) return '#10b981';   // emerald-500
    if (score >= 60) return '#f59e0b';   // amber-500
    return '#ef4444';                     // red-500
}

function getNodeGlow(score) {
    if (score >= 90) return '#10b98140';
    if (score >= 60) return '#f59e0b40';
    return '#ef444440';
}

export default function InteractiveRadarChart({ metrics, size = 400 }) {
    const [animProgress, setAnimProgress] = useState(0);
    const [hoveredIdx, setHoveredIdx] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
    const svgRef = useRef(null);

    // Animate on mount
    useEffect(() => {
        let frame;
        let start = null;
        const duration = 1200;
        const animate = (ts) => {
            if (!start) start = ts;
            const elapsed = ts - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            setAnimProgress(eased);
            if (progress < 1) frame = requestAnimationFrame(animate);
        };
        frame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(frame);
    }, []);

    const cx = size / 2;
    const cy = size / 2;
    const maxR = size * 0.36;
    const levels = [25, 50, 75, 100];
    const n = metrics.length;
    const angleStep = (2 * Math.PI) / n;
    const startAngle = -Math.PI / 2; // start from top

    // Compute point position
    const getPoint = (idx, value) => {
        const angle = startAngle + idx * angleStep;
        const r = (value / 100) * maxR;
        return {
            x: cx + r * Math.cos(angle),
            y: cy + r * Math.sin(angle),
        };
    };

    // Axis endpoint (100%)
    const getAxisEnd = (idx) => getPoint(idx, 100);

    // Polygon points (animated)
    const polygonPoints = metrics
        .map((m, i) => {
            const p = getPoint(i, m.score * animProgress);
            return `${p.x},${p.y}`;
        })
        .join(' ');

    // Find the weak metric for highlight
    const minScore = Math.min(...metrics.map(m => m.score));

    const handleMouseMove = (e, idx) => {
        const svg = svgRef.current;
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        setTooltipPos({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top - 16,
        });
        setHoveredIdx(idx);
    };

    // Unique gradient ID
    const gradientId = 'radar-gradient-fill';
    const glowFilterId = 'node-glow';

    return (
        <div className="relative" style={{ width: size, height: size + 20, margin: '0 auto' }}>
            <svg
                ref={svgRef}
                width={size}
                height={size}
                viewBox={`0 0 ${size} ${size}`}
                className="overflow-visible"
                onMouseLeave={() => setHoveredIdx(null)}
            >
                <defs>
                    {/* Gradient fill for the polygon */}
                    <radialGradient id={gradientId} cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#1f5a7a" stopOpacity="0.35" />
                        <stop offset="70%" stopColor="#0f8b8d" stopOpacity="0.18" />
                        <stop offset="100%" stopColor="#10b981" stopOpacity="0.08" />
                    </radialGradient>

                    {/* Glow filter for nodes */}
                    <filter id={glowFilterId} x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Background grid circles */}
                {levels.map((level) => {
                    const r = (level / 100) * maxR;
                    return (
                        <circle
                            key={level}
                            cx={cx}
                            cy={cy}
                            r={r}
                            fill="none"
                            stroke="#d7e1e8"
                            strokeWidth={level === 100 ? 1.5 : 0.8}
                            strokeDasharray={level === 100 ? 'none' : '4 3'}
                        />
                    );
                })}

                {/* Percentage labels on the right axis */}
                {levels.map((level) => {
                    const r = (level / 100) * maxR;
                    return (
                        <text
                            key={`label-${level}`}
                            x={cx + 6}
                            y={cy - r + 4}
                            fontSize="9"
                            fill="#94a3b8"
                            fontWeight="500"
                            fontFamily="Inter, sans-serif"
                        >
                            {level}%
                        </text>
                    );
                })}

                {/* Axis lines */}
                {metrics.map((_, i) => {
                    const end = getAxisEnd(i);
                    return (
                        <line
                            key={`axis-${i}`}
                            x1={cx}
                            y1={cy}
                            x2={end.x}
                            y2={end.y}
                            stroke="#d7e1e8"
                            strokeWidth="1"
                        />
                    );
                })}

                {/* Filled polygon */}
                <polygon
                    points={polygonPoints}
                    fill={`url(#${gradientId})`}
                    stroke="#1f5a7a"
                    strokeWidth="2"
                    strokeLinejoin="round"
                    style={{
                        transition: 'all 0.3s ease',
                        filter: hoveredIdx !== null ? 'brightness(1.08)' : 'none',
                    }}
                />

                {/* Data points (glowing nodes) */}
                {metrics.map((m, i) => {
                    const p = getPoint(i, m.score * animProgress);
                    const color = getNodeColor(m.score);
                    const isWeak = m.score === minScore && m.score < 90;
                    const isHovered = hoveredIdx === i;
                    const nodeR = isHovered ? 8 : isWeak ? 7 : 5.5;

                    return (
                        <g key={`node-${i}`}>
                            {/* Outer glow */}
                            <circle
                                cx={p.x}
                                cy={p.y}
                                r={nodeR + 6}
                                fill={getNodeGlow(m.score)}
                                style={{
                                    opacity: animProgress,
                                    transition: 'r 0.2s ease',
                                }}
                            />
                            {/* Pulsing ring for weak metrics */}
                            {isWeak && (
                                <circle
                                    cx={p.x}
                                    cy={p.y}
                                    r={nodeR + 10}
                                    fill="none"
                                    stroke="#ef444450"
                                    strokeWidth="1.5"
                                    style={{
                                        animation: 'pulse-ring 2s ease-out infinite',
                                        transformOrigin: `${p.x}px ${p.y}px`,
                                    }}
                                />
                            )}
                            {/* Core dot */}
                            <circle
                                cx={p.x}
                                cy={p.y}
                                r={nodeR}
                                fill={color}
                                stroke="white"
                                strokeWidth="2.5"
                                filter={`url(#${glowFilterId})`}
                                style={{
                                    cursor: 'pointer',
                                    opacity: animProgress,
                                    transition: 'r 0.2s ease',
                                }}
                                onMouseMove={(e) => handleMouseMove(e, i)}
                                onMouseEnter={() => setHoveredIdx(i)}
                            />
                        </g>
                    );
                })}

                {/* Axis labels */}
                {metrics.map((m, i) => {
                    const angle = startAngle + i * angleStep;
                    const labelR = maxR + 28;
                    const lx = cx + labelR * Math.cos(angle);
                    const ly = cy + labelR * Math.sin(angle);

                    // Determine anchor based on position
                    let anchor = 'middle';
                    if (Math.cos(angle) > 0.3) anchor = 'start';
                    else if (Math.cos(angle) < -0.3) anchor = 'end';

                    const color = getNodeColor(m.score);

                    return (
                        <g key={`label-${i}`}>
                            <text
                                x={lx}
                                y={ly}
                                textAnchor={anchor}
                                dominantBaseline="middle"
                                fontSize="12"
                                fontWeight="600"
                                fontFamily="Inter, sans-serif"
                                fill="#13293d"
                            >
                                {m.name}
                            </text>
                            <text
                                x={lx}
                                y={ly + 15}
                                textAnchor={anchor}
                                dominantBaseline="middle"
                                fontSize="11"
                                fontWeight="700"
                                fontFamily="Inter, sans-serif"
                                fill={color}
                            >
                                {m.score}%
                            </text>
                        </g>
                    );
                })}
            </svg>

            {/* Tooltip */}
            {hoveredIdx !== null && (
                <div
                    className="absolute pointer-events-none z-50 transition-all duration-150"
                    style={{
                        left: tooltipPos.x,
                        top: tooltipPos.y,
                        transform: 'translate(-50%, -100%)',
                    }}
                >
                    <div className="bg-gray-900/95 backdrop-blur-sm text-white px-4 py-3 rounded-xl shadow-2xl border border-white/10 min-w-[180px]">
                        <p className="text-xs font-bold mb-1" style={{ color: getNodeColor(metrics[hoveredIdx].score) }}>
                            {metrics[hoveredIdx].name}
                        </p>
                        <p className="text-lg font-extrabold mb-1">
                            {metrics[hoveredIdx].score}%
                        </p>
                        <p className="text-[11px] text-white/60 leading-snug">
                            {METRIC_DESCRIPTIONS[metrics[hoveredIdx].name] || ''}
                        </p>
                    </div>
                    {/* Arrow */}
                    <div className="flex justify-center -mt-px">
                        <div className="w-2.5 h-2.5 bg-gray-900/95 rotate-45 border-b border-r border-white/10" />
                    </div>
                </div>
            )}

            {/* CSS for pulse animation */}
            <style>{`
        @keyframes pulse-ring {
          0% { r: ${7 + 10}; opacity: 0.6; }
          100% { r: ${7 + 22}; opacity: 0; }
        }
      `}</style>
        </div>
    );
}

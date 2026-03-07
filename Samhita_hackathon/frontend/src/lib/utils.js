import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

export function formatPct(value) {
    return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value) {
    return new Intl.NumberFormat().format(value);
}

export function getTrustColor(score) {
    if (score > 0.9) return '#10b981'; // emerald
    if (score > 0.7) return '#f59e0b'; // amber
    if (score > 0.5) return '#f97316'; // orange
    return '#ef4444'; // red
}

export function getTrustLabel(score) {
    if (score > 0.9) return 'Excellent';
    if (score > 0.7) return 'Moderate';
    if (score > 0.5) return 'Low';
    return 'Critical';
}

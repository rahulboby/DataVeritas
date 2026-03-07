import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard, Upload, BarChart3, Copy, AlertTriangle,
    ShieldCheck, PieChart, Sparkles, Database
} from 'lucide-react';

const navItems = [
    { to: '/', icon: Upload, label: 'Upload Data' },
    { to: '/dashboard', icon: LayoutDashboard, label: 'Trust Score' },
    { to: '/nulls', icon: BarChart3, label: 'Null Analysis' },
    { to: '/duplicates', icon: Copy, label: 'Duplicates' },
    { to: '/outliers', icon: AlertTriangle, label: 'Outliers' },
    { to: '/consistency', icon: ShieldCheck, label: 'Consistency' },
    { to: '/distribution', icon: PieChart, label: 'Distribution' },
    { to: '/insights', icon: Sparkles, label: 'AI Insights' },
];

export default function Sidebar({ datasetId }) {
    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-gradient-to-b from-brand-700 to-brand-600 flex flex-col z-40">
            {/* Logo */}
            <div className="px-6 py-5 border-b border-white/10">
                <div className="flex items-center gap-2.5">
                    <Database className="w-7 h-7 text-brand-200" />
                    <div>
                        <h1 className="text-lg font-bold text-white tracking-tight">DataVeritas</h1>
                        <p className="text-[11px] text-white/50 font-medium tracking-wide uppercase">AI Data Intelligence</p>
                    </div>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
                <p className="px-3 text-[10px] font-semibold text-white/40 uppercase tracking-widest mb-2">
                    Navigation
                </p>
                {navItems.map(({ to, icon: Icon, label }) => {
                    const isDisabled = to !== '/' && !datasetId;
                    return (
                        <NavLink
                            key={to}
                            to={isDisabled ? '#' : to}
                            onClick={(e) => isDisabled && e.preventDefault()}
                            className={({ isActive }) =>
                                `sidebar-btn ${isActive && !isDisabled ? 'sidebar-btn-active' : 'sidebar-btn-inactive'} ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}`
                            }
                        >
                            <Icon className="w-4 h-4 flex-shrink-0" />
                            <span>{label}</span>
                        </NavLink>
                    );
                })}
            </nav>

            {/* Footer */}
            <div className="px-4 py-3 border-t border-white/10">
                <p className="text-[10px] text-white/30 text-center">DataVeritas v2.0</p>
            </div>
        </aside>
    );
}

import { Link, useLocation } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/',          label: 'Chat',          icon: '💬' },
  { path: '/leads',     label: 'Leads',         icon: '👥' },
  { path: '/pipeline',  label: 'Pipeline',      icon: '📊' },
  { path: '/knowledge', label: 'Knowledge Base', icon: '📚' },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <div className="w-56 bg-gray-900 text-white flex flex-col h-screen fixed left-0 top-0">

      {/* Logo */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-sm font-bold">
            AI
          </div>
          <div>
            <p className="font-semibold text-sm">CRM Platform</p>
            <p className="text-xs text-gray-400">AI Powered</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {NAV_ITEMS.map(item => (
          <Link
            key={item.path}
            to={item.path}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              location.pathname === item.path
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <p className="text-xs text-gray-500">AI CRM v1.0.0</p>
        <p className="text-xs text-gray-500">Built with LangChain</p>
      </div>
    </div>
  );
}

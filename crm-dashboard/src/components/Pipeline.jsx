import { useState, useEffect } from 'react';
import { api } from '../config/api';

const STAGES = [
  { key: 'new',         label: 'New',         color: 'bg-blue-500' },
  { key: 'contacted',   label: 'Contacted',   color: 'bg-yellow-500' },
  { key: 'qualified',   label: 'Qualified',   color: 'bg-green-500' },
  { key: 'unqualified', label: 'Unqualified', color: 'bg-red-400' },
  { key: 'closed',      label: 'Closed',      color: 'bg-gray-500' },
];

export default function Pipeline() {
  const [stats, setStats]     = useState(null);
  const [leads, setLeads]     = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, allLeadsRes] = await Promise.all([
          api.get('/leads/stats/overview'),
          api.get('/leads/?limit=100'),
        ]);

        setStats(statsRes.data);

        const grouped = {};
        STAGES.forEach(s => { grouped[s.key] = []; });
        allLeadsRes.data.leads.forEach(lead => {
          if (grouped[lead.status]) grouped[lead.status].push(lead);
        });
        setLeads(grouped);

      } catch (err) {
        console.error('Failed to fetch pipeline:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return (
    <div className="p-6 text-center text-gray-400 mt-20">Loading pipeline...</div>
  );

  const total = stats?.total_leads || 0;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Pipeline</h1>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {STAGES.map(stage => {
          const count = stats?.by_status?.[stage.key] || 0;
          const pct   = total > 0 ? Math.round((count / total) * 100) : 0;
          return (
            <div key={stage.key} className="bg-white rounded-xl p-4 shadow-sm border">
              <div className={`w-3 h-3 rounded-full ${stage.color} mb-2`}></div>
              <p className="text-2xl font-bold text-gray-900">{count}</p>
              <p className="text-sm text-gray-500">{stage.label}</p>
              <p className="text-xs text-gray-400 mt-1">{pct}% of total</p>
            </div>
          );
        })}
      </div>

      {/* Progress Bar */}
      <div className="bg-white rounded-xl p-6 shadow-sm border mb-8">
        <h2 className="font-semibold mb-4 text-gray-700">Pipeline Overview</h2>
        <div className="flex rounded-full overflow-hidden h-6">
          {STAGES.map(stage => {
            const count = stats?.by_status?.[stage.key] || 0;
            const pct   = total > 0 ? (count / total) * 100 : 0;
            return pct > 0 ? (
              <div key={stage.key}
                className={`${stage.color} flex items-center justify-center text-white text-xs font-medium`}
                style={{ width: `${pct}%` }}
                title={`${stage.label}: ${count}`}>
                {pct > 8 ? `${Math.round(pct)}%` : ''}
              </div>
            ) : null;
          })}
          {total === 0 && (
            <div className="bg-gray-200 w-full flex items-center justify-center text-gray-400 text-sm">
              No leads yet
            </div>
          )}
        </div>
        <div className="flex gap-4 mt-3">
          {STAGES.map(stage => (
            <div key={stage.key} className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${stage.color}`}></div>
              <span className="text-xs text-gray-500">{stage.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Kanban Columns */}
      <div className="grid grid-cols-5 gap-4">
        {STAGES.map(stage => (
          <div key={stage.key} className="bg-gray-100 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-3">
              <div className={`w-2 h-2 rounded-full ${stage.color}`}></div>
              <h3 className="font-medium text-sm text-gray-700">{stage.label}</h3>
              <span className="ml-auto text-xs bg-white text-gray-500 px-2 py-0.5 rounded-full">
                {leads[stage.key]?.length || 0}
              </span>
            </div>
            <div className="space-y-2">
              {(leads[stage.key] || []).map(lead => (
                <div key={lead.id} className="bg-white rounded-lg p-3 shadow-sm">
                  <p className="font-medium text-sm text-gray-900">{lead.name}</p>
                  {lead.company && <p className="text-xs text-gray-400 mt-0.5">{lead.company}</p>}
                  {lead.plan_interest && (
                    <span className="mt-1 inline-block text-xs bg-purple-50 text-purple-600 px-1.5 py-0.5 rounded">
                      {lead.plan_interest}
                    </span>
                  )}
                </div>
              ))}
              {(leads[stage.key] || []).length === 0 && (
                <p className="text-xs text-gray-400 text-center py-4">Empty</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

import { useState, useEffect, useCallback } from 'react';
import { api } from '../config/api';

const STATUS_COLORS = {
  new:         'bg-blue-100 text-blue-700',
  contacted:   'bg-yellow-100 text-yellow-700',
  qualified:   'bg-green-100 text-green-700',
  unqualified: 'bg-red-100 text-red-700',
  closed:      'bg-gray-100 text-gray-700',
};

export default function LeadsTable() {
  const [leads, setLeads]         = useState([]);
  const [loading, setLoading]     = useState(true);
  const [total, setTotal]         = useState(0);
  const [statusFilter, setStatus] = useState('');
  const [showForm, setShowForm]   = useState(false);
  const [form, setForm] = useState({
    name: '', email: '', phone: '', company: '', plan_interest: '', notes: ''
  });

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    try {
      const params = statusFilter ? `?status=${statusFilter}` : '';
      const res = await api.get(`/leads/${params}`);
      setLeads(res.data.leads);
      setTotal(res.data.total);
    } catch (err) {
      console.error('Failed to fetch leads:', err);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => { fetchLeads(); }, [fetchLeads]);

  const createLead = async (e) => {
    e.preventDefault();
    try {
      await api.post('/leads/', form);
      setForm({ name:'', email:'', phone:'', company:'', plan_interest:'', notes:'' });
      setShowForm(false);
      fetchLeads();
    } catch (err) {
      alert('Failed to create lead: ' + err.message);
    }
  };

  const updateStatus = async (leadId, newStatus) => {
    try {
      await api.patch(`/leads/${leadId}`, { status: newStatus });
      setLeads(prev => prev.map(l => l.id === leadId ? {...l, status: newStatus} : l));
    } catch (err) {
      console.error('Failed to update:', err);
    }
  };

  const deleteLead = async (leadId) => {
    if (!confirm('Delete this lead?')) return;
    try {
      await api.delete(`/leads/${leadId}`);
      setLeads(prev => prev.filter(l => l.id !== leadId));
      setTotal(prev => prev - 1);
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  };

  const exportCSV = () => {
    const csv = [
      'id,name,email,company,status,plan_interest',
      ...leads.map(l =>
        `${l.id},${l.name},${l.email||''},${l.company||''},${l.status},${l.plan_interest||''}`
      )
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'leads.csv'; a.click();
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Leads</h1>
          <p className="text-gray-500 text-sm">{total} total leads</p>
        </div>
        <div className="flex gap-2">
          <button onClick={exportCSV}
            className="border text-gray-600 px-4 py-2 rounded-lg hover:bg-gray-50 text-sm">
            Export CSV
          </button>
          <button onClick={() => setShowForm(!showForm)}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm">
            + Add Lead
          </button>
        </div>
      </div>

      {/* Add Lead Form */}
      {showForm && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <h2 className="font-semibold mb-4">New Lead</h2>
          <form onSubmit={createLead} className="grid grid-cols-2 gap-4">
            <input required placeholder="Name *" value={form.name}
              onChange={e => setForm({...form, name: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" />
            <input placeholder="Email" value={form.email}
              onChange={e => setForm({...form, email: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" />
            <input placeholder="Phone" value={form.phone}
              onChange={e => setForm({...form, phone: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" />
            <input placeholder="Company" value={form.company}
              onChange={e => setForm({...form, company: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" />
            <select value={form.plan_interest}
              onChange={e => setForm({...form, plan_interest: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none">
              <option value="">Plan Interest</option>
              <option value="starter">Starter</option>
              <option value="growth">Growth</option>
              <option value="enterprise">Enterprise</option>
            </select>
            <input placeholder="Notes" value={form.notes}
              onChange={e => setForm({...form, notes: e.target.value})}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" />
            <div className="col-span-2 flex gap-3">
              <button type="submit" className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700">
                Create Lead
              </button>
              <button type="button" onClick={() => setShowForm(false)}
                className="text-gray-600 px-6 py-2 rounded-lg border hover:bg-gray-50">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-2 mb-4">
        {['', 'new', 'contacted', 'qualified', 'unqualified', 'closed'].map(s => (
          <button key={s} onClick={() => setStatus(s)}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              statusFilter === s ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 border hover:bg-gray-50'
            }`}>
            {s || 'All'}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-gray-400">Loading leads...</div>
        ) : leads.length === 0 ? (
          <div className="p-12 text-center text-gray-400">No leads found. Add your first lead!</div>
        ) : (
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Lead</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Company</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Plan</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Created</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {leads.map(lead => (
                <tr key={lead.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <p className="font-medium text-gray-900 text-sm">{lead.name}</p>
                    <p className="text-xs text-gray-400">{lead.email || '—'}</p>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">{lead.company || '—'}</td>
                  <td className="px-4 py-3">
                    {lead.plan_interest && (
                      <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full capitalize">
                        {lead.plan_interest}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <select value={lead.status} onChange={e => updateStatus(lead.id, e.target.value)}
                      className={`text-xs px-2 py-1 rounded-full border-0 font-medium cursor-pointer ${STATUS_COLORS[lead.status] || 'bg-gray-100'}`}>
                      {['new','contacted','qualified','unqualified','closed'].map(s => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-400">
                    {lead.created_at ? new Date(lead.created_at).toLocaleDateString() : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <button onClick={() => deleteLead(lead.id)}
                      className="text-xs text-red-500 hover:text-red-700">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

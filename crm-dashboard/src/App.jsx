import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import LeadsTable from './components/LeadsTable';
import Pipeline from './components/Pipeline';
import KnowledgeBase from './components/KnowledgeBase';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex">
        <Sidebar />
        <main className="ml-56 flex-1 min-h-screen bg-gray-50">
          <Routes>
            <Route path="/"          element={<ChatInterface />} />
            <Route path="/leads"     element={<LeadsTable />} />
            <Route path="/pipeline"  element={<Pipeline />} />
            <Route path="/knowledge" element={<KnowledgeBase />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

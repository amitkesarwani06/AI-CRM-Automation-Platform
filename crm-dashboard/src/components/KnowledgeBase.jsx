import { useState } from 'react';
import { api } from '../config/api';

const SAMPLE_QUESTIONS = [
  "What is your refund policy?",
  "What does the Growth plan include?",
  "What are your support hours?",
  "Is there a free trial?",
  "How do I import leads?",
];

export default function KnowledgeBase() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer]     = useState('');
  const [loading, setLoading]   = useState(false);
  const [history, setHistory]   = useState([]);

  const askQuestion = async (q = question) => {
    if (!q.trim() || loading) return;
    setLoading(true);
    setAnswer('');

    try {
      const res = await api.post('/chat/rag', {
        message:    q,
        session_id: 'kb_session',
      });
      const ans = res.data.response;
      setAnswer(ans);
      setHistory(prev => [{question: q, answer: ans}, ...prev].slice(0, 5));
      setQuestion('');
    } catch (err) {
      setAnswer(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Knowledge Base</h1>
      <p className="text-gray-500 mb-8">Ask questions about our CRM platform — answers from real docs</p>

      {/* Question Input */}
      <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
        <label className="text-sm font-medium text-gray-700 mb-2 block">Your Question</label>
        <div className="flex gap-3">
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && askQuestion()}
            placeholder="e.g. What is your refund policy?"
            className="flex-1 border rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
          />
          <button onClick={() => askQuestion()}
            disabled={loading || !question.trim()}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium">
            {loading ? 'Searching...' : 'Ask'}
          </button>
        </div>

        {/* Sample Questions */}
        <div className="mt-4">
          <p className="text-xs text-gray-400 mb-2">Try these:</p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_QUESTIONS.map((q, i) => (
              <button key={i} onClick={() => askQuestion(q)}
                className="text-xs bg-gray-100 text-gray-600 px-3 py-1.5 rounded-full hover:bg-blue-50 hover:text-blue-600">
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Answer */}
      {(answer || loading) && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center text-white text-xs">AI</div>
            <span className="text-sm font-medium text-gray-700">Answer</span>
          </div>
          {loading ? (
            <div className="flex gap-1 items-center text-gray-400">
              <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{animationDelay:'150ms'}}></div>
              <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{animationDelay:'300ms'}}></div>
              <span className="text-sm ml-2">Searching knowledge base...</span>
            </div>
          ) : (
            <p className="text-gray-800 text-sm leading-relaxed">{answer}</p>
          )}
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div>
          <h2 className="font-semibold text-gray-700 mb-3">Recent Questions</h2>
          <div className="space-y-3">
            {history.map((item, i) => (
              <div key={i} className="bg-white rounded-xl border p-4">
                <p className="text-sm font-medium text-gray-900 mb-2">Q: {item.question}</p>
                <p className="text-sm text-gray-600">A: {item.answer.slice(0, 150)}...</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

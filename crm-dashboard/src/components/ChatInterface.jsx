import { useState, useRef, useEffect } from 'react';
import { api } from '../config/api';

const SESSION_ID = `session_${Date.now()}`;

export default function ChatInterface() {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      content: "Hi! I'm your AI CRM assistant. I can help you manage leads, answer product questions, and show analytics. What would you like to do?",
      agents: [],
    }
  ]);
  const [input, setInput]     = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef         = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    setMessages(prev => [...prev, { role: 'human', content: userMessage, agents: [] }]);

    try {
      const response = await api.post('/chat/', {
        message:    userMessage,
        session_id: SESSION_ID,
      });

      const { response: aiResponse, agents_used } = response.data;

      setMessages(prev => [...prev, {
        role:    'ai',
        content: aiResponse,
        agents:  agents_used || [],
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        role:    'ai',
        content: `Sorry, something went wrong. ${error.message}`,
        agents:  [],
        error:   true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickActions = [
    "Show pipeline stats",
    "What is your refund policy?",
    "Add a new lead",
    "List all leads",
  ];

  return (
    <div className="flex flex-col h-screen bg-gray-50">

      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex items-center gap-3">
        <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold">
          AI
        </div>
        <div>
          <h1 className="font-semibold text-gray-900">CRM AI Assistant</h1>
          <p className="text-xs text-green-500">● Online</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'human' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl rounded-2xl px-4 py-3 ${
              msg.role === 'human'
                ? 'bg-blue-600 text-white'
                : msg.error
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-white text-gray-800 shadow-sm border'
            }`}>
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              {msg.agents && msg.agents.length > 0 && (
                <div className="mt-2 flex gap-1 flex-wrap">
                  {msg.agents.map((agent, j) => (
                    <span key={j} className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                      {agent.replace('_agent', '')}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border">
              <div className="flex gap-1 items-center">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                <span className="text-xs text-gray-400 ml-2">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div className="px-4 pb-2 flex gap-2 overflow-x-auto">
        {quickActions.map((action, i) => (
          <button
            key={i}
            onClick={() => setInput(action)}
            className="text-xs bg-white border rounded-full px-3 py-1.5 text-gray-600 hover:bg-blue-50 hover:border-blue-300 whitespace-nowrap"
          >
            {action}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="bg-white border-t px-4 py-4">
        <div className="flex gap-3 items-end max-w-4xl mx-auto">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message AI CRM Assistant... (Enter to send)"
            rows={2}
            className="flex-1 border rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white rounded-xl px-4 py-3 font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

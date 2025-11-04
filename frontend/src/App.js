import React, { useState } from 'react';
import './App.css';
import Input from './Input';
import ChatHistory from './ChatHistory';

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  return (
    <div className="h-screen bg-slate-600 flex flex-col justify-center items-center">
      <h1 className="text-2xl font-bold text-white mb-6">하이봇</h1>
      <div className='flex flex-col w-[80%] gap-4'>
        <ChatHistory chatHistory={chatHistory} />
        <Input setChatHistory={setChatHistory} setLoading={setLoading} />
      </div>
    </div>
  );
}

export default App;
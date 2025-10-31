import React, { useState } from 'react';
import './App.css';
import Input from './input';

function App() {
  // 전체 대화 내역 (배열)
  const [chatHistory, setChatHistory] = useState([]);

  

  return (
    <div className="App">
      <header className="bg-white">
        <h1>FAQ 챗봇</h1>
        <div>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
            </div>
          ))}
        </div>

        <Input setChatHistory={setChatHistory} />
      </header>
    </div>
  );
}

export default App;
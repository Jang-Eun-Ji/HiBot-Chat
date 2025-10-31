import React, { useState } from 'react';
import './App.css';

function App() {
  // 현재 입력창의 메시지
  const [currentMessage, setCurrentMessage] = useState('');
  // 전체 대화 내역 (배열)
  const [chatHistory, setChatHistory] = useState([]);

  // 메시지 전송 처리 함수
  const handleSubmit = async (e) => {
    e.preventDefault(); // 폼 전송 시 페이지 새로고침 방지
    
    const userMessage = currentMessage.trim();
    if (!userMessage) return; // 빈 메시지는 전송하지 않음

    // 1. 사용자 메시지를 대화 내역에 추가
    setChatHistory(prevHistory => [
      ...prevHistory,
      { sender: 'user', text: userMessage }
    ]);

    // 2. 입력창 비우기
    setCurrentMessage('');

    try {
      // 3. 백엔드 API에 사용자 메시지 전송 (POST)
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }), // JSON 형태로 전송
      });

      const data = await response.json();
      
      // 4. 백엔드에서 받은 봇의 응답을 대화 내역에 추가
      setChatHistory(prevHistory => [
        ...prevHistory,
        { sender: 'bot', text: data.response }
      ]);

    } catch (error) {
      console.error('챗봇 응답 오류:', error);
      // 5. 오류 발생 시
      setChatHistory(prevHistory => [
        ...prevHistory,
        { sender: 'bot', text: '오류가 발생했습니다. 서버를 확인해주세요.' }
      ]);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>FAQ 챗봇</h1>
        
        {/* 채팅 내역이 표시될 창 */}
        <div>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit}>
          <input
            className=''
            type="text"
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            placeholder="질문을 입력하세요..."
          />
          <button type="submit">전송</button>
        </form>
      </header>
    </div>
  );
}

export default App;
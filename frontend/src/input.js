import { useState } from "react";

function Input({ setChatHistory, isLoading, setIsLoading }) {
    const [currentMessage, setCurrentMessage] = useState('');
    const notAllowed = isLoading || !currentMessage;
    // 메시지 전송 처리 함수
    const handleSubmit = async (e) => {
        e.preventDefault(); // 폼 전송 시 페이지 새로고침 방지
        setIsLoading(true);
        const userMessage = currentMessage.trim();
        if (!userMessage) return; // 빈 메시지는 전송하지 않음

        // 1. 사용자 메시지를 대화 내역에 추가
        setChatHistory(prevHistory => [
            ...prevHistory,
            { sender: 'user', text: userMessage }
        ]);

        // 2. 입력창 비우기
        setCurrentMessage('');

        setChatHistory(prevHistory => [
            ...prevHistory,
            { sender: 'bot', text: "", isLoading: true }
        ]);
        try {
            // 3. 백엔드 API에 사용자 메시지 전송 (POST)
            const response = await fetch('https://hibot-chat-production.up.railway.app/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }), // JSON 형태로 전송
            });

            const data = await response.json();

            // 4. 백엔드에서 받은 봇의 응답을 대화 내역에 추가
            setChatHistory(prevHistory => {
                const newHistory = [...prevHistory];
                newHistory[newHistory.length - 1] = { sender: 'bot', text: data.response };
                return newHistory;
            });
            setIsLoading(false);
        } catch (error) {
            console.error('챗봇 응답 오류:', error);
            // 5. 오류 발생 시
            setChatHistory(prevHistory => {
                const newHistory = [...prevHistory];
                newHistory[newHistory.length - 1] = { sender: 'bot', text: '오류가 발생했습니다. 서버를 확인해주세요.' };
                return newHistory;
            });
            setIsLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="w-full">
            <div class="relative">
                <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    placeholder="질문을 입력하세요..."
                    class="w-full border border-gray-300 rounded-lg py-2 px-3 pr-16 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                    class={`absolute top-2 right-2 text-sm px-3 py-1 rounded-md transition ${notAllowed
                            ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                            : "bg-blue-500 text-white hover:bg-blue-600"
                        }`}
                    disabled={notAllowed}
                >
                    전송
                </button>
            </div>
        </form>
    )
};

export default Input;
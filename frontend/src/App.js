import React, { useState } from "react";
import "./App.css";
import Input from "./Input";
import ChatHistory from "./ChatHistory";
import FAQList from "./FAQList";
import FAQButton from "./FAQButton";

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isFAQOpen, setIsFAQOpen] = useState(false);

  return (
    <div className="h-screen bg-slate-600 flex flex-col justify-center items-center relative">
      <h1 className="text-2xl font-bold text-white mb-6">하이봇</h1>

      <div className="flex flex-col xl:flex-row w-[80%] min-w-[600px] gap-4 relative">
        <div className="flex-1 flex flex-col gap-4">
          <ChatHistory chatHistory={chatHistory} />
          <Input
            setChatHistory={setChatHistory}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>

        <div className="hidden xl:block">
          <FAQList
            setChatHistory={setChatHistory}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>

        {/* ✅ 좁은 화면: 아이콘 버튼 + 팝업 FAQ */}
        <div className="block xl:hidden fixed bottom-6 right-6 z-50">
          <div className="relative flex flex-col items-end">
            {/* 팝업 리스트 — 아이콘 바로 위 */}
            {isFAQOpen && (
              <div className="absolute bottom-full mb-3 bg-white rounded-lg shadow-lg w-64 p-4 z-40">
                <FAQList
                  setChatHistory={setChatHistory}
                  isLoading={isLoading}
                  setIsLoading={setIsLoading}
                />
              </div>
            )}

            {/* 아이콘 버튼 */}
            <FAQButton setIsFAQOpen={setIsFAQOpen} isFAQOpen={isFAQOpen} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

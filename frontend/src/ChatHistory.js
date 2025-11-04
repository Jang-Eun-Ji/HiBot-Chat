import { useEffect, useRef } from "react";

function ChatHistory({ chatHistory }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  return (
    <div className="w-full min-w-md h-[60vh] overflow-y-auto bg-slate-700 rounded-lg p-8 pb-0 shadow-md flex flex-col gap-8">
      {chatHistory.map((msg, index) => (
        <div
          key={index}
          className={`mb-2 flex ${
            msg.sender === "user" ? "justify-end" : "justify-start"
          }`}
        >
          <div
            className={`max-w-[70%] px-3 py-2 rounded-lg ${
              msg.sender === "user"
                ? "bg-blue-500 text-white self-end"
                : "bg-gray-300 text-gray-900 self-start"
            }`}
          >
            {msg.isLoading ? (
              // 🟡 로딩 중일 때
              <div className="flex space-x-1">
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
              </div>
            ) : (
              msg.text
            )}
          </div>
        </div>
      ))}

      <div ref={scrollRef} />
    </div>
  );
}

export default ChatHistory;
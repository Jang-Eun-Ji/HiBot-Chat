import { useEffect, useRef } from "react";

function ChatHistory({ chatHistory }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  return (
    <div className="w-full min-w-md h-[60vh] overflow-y-auto bg-slate-700 rounded-lg p-8 pb-0 shadow-md flex flex-col gap-8 font-momo">
      {chatHistory.map((msg, index) => (
        <div
          key={index}
          className={`mb-2 flex transition-all duration-500 ease-out transform ${msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
        >
          <div
            className={`max-w-[70%] px-4 py-3 rounded-xl opacity-0 animate-[fadeInUp_0.5s_ease-out_forwards] ${msg.sender === "user"
                ? "bg-blue-500 text-white self-end"
                : "bg-gray-300 text-gray-900 self-start"
              }`}
          >
            {msg.isLoading ? (
              <div className="flex items-center justify-center gap-2 w-12">
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s] [animation-duration:0.6s]" />
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s] [animation-duration:0.6s]" />
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-duration:0.6s]" />
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
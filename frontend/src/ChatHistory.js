import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function ChatHistory({ chatHistory }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  return (
    <div className="w-full h-[60vh] overflow-y-auto bg-white shadow-xl rounded-lg p-8 pb-0 flex flex-col gap-8 font-batang font-medium">
      {chatHistory.map((msg, index) => (
        <div
          key={index}
          className={`mb-2 flex transition-all duration-500 ease-out transform 
            ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-[90%] px-4 py-3 rounded-xl opacity-0 
              animate-[fadeInUp_0.5s_ease-out_forwards]
              text-sm sm:text-[16px]
              ${msg.sender === "user"
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
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ node, ...props }) => {
                    const rawText = node.children
                      ?.map(child => child.value || "")
                      .join("")
                      .trim();

                    if (rawText && rawText.includes("출처")) {
                      return (
                        <p className="text-blue-700 text-xs font-semibold bg-blue-100 px-3 py-2 rounded-md mt-3">
                          {rawText}
                        </p>
                      );
                    }

                    return (
                      <p className="leading-relaxed" {...props}>
                        {props.children}
                      </p>
                    );
                  },

                  ul: ({ node, ...props }) => (
                    <ul className="list-disc ml-4 space-y-1" {...props} />
                  ),

                  li: ({ node, ...props }) => (
                    <li className="leading-relaxed" {...props} />
                  ),

                  strong: ({ node, ...props }) => (
                    <strong className="font-semibold" {...props} />
                  ),
                }}
              >
                {msg.text}
              </ReactMarkdown>
            )}
          </div>
        </div>
      ))}

      <div ref={scrollRef} />
    </div>
  );
}

export default ChatHistory;
function ChatHistory({ chatHistory }) {
  return (
    <div className="w-full min-w-md h-[60vh] overflow-y-auto bg-slate-700 rounded-lg p-8 shadow-md flex flex-col gap-8">
      {chatHistory.map((msg, index) => (
        <div
          key={index}
          className={`mb-2 flex ${
            msg.sender === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-[70%] px-3 py-2 rounded-lg ${
              msg.sender === 'user'
                ? 'bg-blue-500 text-white self-end'
                : 'bg-gray-300 text-gray-900 self-start'
            }`}
          >
            {msg.text}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ChatHistory;

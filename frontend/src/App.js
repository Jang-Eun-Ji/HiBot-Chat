import { useState } from "react";
import "./App.css";
import Input from "./input";
import ChatHistory from "./ChatHistory";
import FAQList from "./FAQList";
import FAQPopUp from "./FAQPopUp";
import Logo from "./Logo";

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isFAQOpen, setIsFAQOpen] = useState(false);

  return (
    <div className="h-screen bg-slate-200 flex justify-center items-center relative">
      <Logo />
      <div className="flex items-center justify-center flex-row w-[90%] sm:w-[85%] gap-4 relative">
        <div className="w-[80%] flex-1 flex flex-col gap-4">
          <ChatHistory chatHistory={chatHistory} />
          <Input
            setChatHistory={setChatHistory}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>
        <div className="hidden xl:block">
          <div className="flex flex-col items-center gap-3">
            <h1 className=" font-bold text-3xl text-blue-500">FAQ</h1>
            <FAQList
              setChatHistory={setChatHistory}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          </div>
        </div>
        <FAQPopUp
          setIsFAQOpen={setIsFAQOpen}
          isFAQOpen={isFAQOpen}
          setChatHistory={setChatHistory}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />
      </div>
    </div>
  );
}

export default App;
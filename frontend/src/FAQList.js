import { faqList } from "./constants";

function FAQList({ setChatHistory, isLoading, setIsLoading }) {
    const handleFAQClick = (question) => {
        if (isLoading) return;
        setChatHistory((prev) => [
            ...prev,
            { sender: "user", text: question },
        ]);
        setIsLoading(true);
        setChatHistory(prevHistory => [
            ...prevHistory,
            { sender: 'bot', text: "", isLoading: true }
        ]);
        // 예시로 2초 후 응답 추가 (실제는 API 요청)
        setTimeout(() => {
            setChatHistory((prev) => {
                const newHistory = [...prev];
                newHistory[newHistory.length - 1] = { sender: "bot", text: `“${question}”에 대한 답변입니다!` };
                return newHistory;
            });
            setIsLoading(false);
        }, 2000);
    };

    return (
        < div className="flex flex-col gap-4">
            {
                faqList.map((faq, idx) => (
                    <button
                        key={idx}
                        onClick={() => handleFAQClick(faq)}
                        className="bg-white text-slate-700 font-semibold px-3 py-2 rounded-xl shadow hover:bg-slate-100 transition"
                    >
                        {faq}
                    </button>
                ))
            }
        </div >
    )
};

export default FAQList;
import { faqList } from "./constants";

function FAQList({ setChatHistory, isLoading, setIsLoading }) {
    const handleFAQClick = async (question, num) => {
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

        try {
            const response = await fetch('https://hibot-chat-production.up.railway.app/api/faq', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ faq_number: num }), // faq각각에 해당하는 번호를 전송
            });

            const data = await response.json();

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
        < div className="flex flex-col gap-4">
            {
                faqList.map((faq, idx) => (
                    <button
                        key={idx}
                        onClick={() => handleFAQClick(faq, idx)}
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
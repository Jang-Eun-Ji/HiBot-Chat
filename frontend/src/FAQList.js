// ✅ FAQ 리스트
const faqList = [
    "회원가입은 어떻게 하나요?",
    "비밀번호를 잊어버렸어요.",
    "서비스 요금은 얼마인가요?",
    "고객센터 운영시간이 궁금해요.",
];

function FAQList({setChatHistory, isLoading, setIsLoading}) {
    const handleFAQClick = (question) => {
        if(isLoading) return;
        setChatHistory((prev) => [
            ...prev,
            { sender: "user", text: question },
            { sender: "bot", text: "응답을 불러오는 중이에요..." },
        ]);
        setIsLoading(true);
        // 예시로 2초 후 응답 추가 (실제는 API 요청)
        setTimeout(() => {
            setChatHistory((prev) => [
                ...prev.slice(0, -1),
                { sender: "bot", text: `“${question}”에 대한 답변입니다!` },
            ]);
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
function FAQPopUp({isFAQOpen}) {
    return (
        < div className = "block xl:hidden fixed bottom-6 right-6 z-50" >
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

                <FAQButton setIsFAQOpen={setIsFAQOpen} isFAQOpen={isFAQOpen} />
            </div>
        </div >
    )
}

export default FAQPopUp;
function FAQButton({ setIsFAQOpen, isFAQOpen }) {
    return(
        <button
            onClick={() => setIsFAQOpen((isOpen) => !isOpen)}
            className="bg-white rounded-full shadow-lg hover:scale-105 transition transform"
        >
            {!isFAQOpen ? (
                <img
                    src="/KHIS_007.png"
                    alt="FAQ Icon"
                    className="w-12 h-12 rounded-full object-cover"
                />
            ) : (
                <div className="text-2xl font-bold text-slate-700 rounded-full w-12 h-12 flex justify-center items-center">
                    ✖
                </div>
            )}
        </button>
    )
}

export default FAQButton;
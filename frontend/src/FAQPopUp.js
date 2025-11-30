import { motion, AnimatePresence } from "framer-motion";
import FAQButton from "./FAQButton";
import FAQList from "./FAQList";

function FAQPopUp({ isFAQOpen, setIsFAQOpen, setChatHistory, isLoading, setIsLoading }) {
    return (
        <div className="block xl:hidden fixed bottom-6 right-6 z-50">
            <div className="relative flex flex-col items-end">
                <AnimatePresence>
                    {isFAQOpen && (
                        <motion.div
                            key="faq-popup"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                            transition={{ duration: 0.3, ease: "easeOut" }}
                            className="absolute bottom-full mb-3 bg-white rounded-lg shadow-lg w-52 sm:w-72 p-4 z-40"
                        >
                            <FAQList
                                setChatHistory={setChatHistory}
                                isLoading={isLoading}
                                setIsLoading={setIsLoading}
                            />
                        </motion.div>
                    )}
                </AnimatePresence>
                <FAQButton setIsFAQOpen={setIsFAQOpen} isFAQOpen={isFAQOpen} />
            </div>
        </div>
    );
}

export default FAQPopUp;
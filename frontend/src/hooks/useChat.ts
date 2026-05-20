import { useState } from "react";
import { Message } from "@/models";
import { api } from "@/services/api";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await api.chat(userMessage);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: response.response || response.error || "No response received." },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to the server." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    messages,
    input,
    setInput,
    isLoading,
    handleSend,
  };
}

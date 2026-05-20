"use client";

import { useState } from "react";
import { sendQuery } from "@/services/api";

export default function ChatModule() {
  const [messages, setMessages] = useState<{ role: "user" | "bot"; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendQuery(userMessage);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: response.response || response.error || "No response received." },
      ]);
    } catch (_error) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to the server." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto border border-gray-700 rounded-lg overflow-hidden bg-gray-900 text-gray-100">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-10">Start a conversation with L-SRAG...</div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`max-w-[80%] p-3 rounded-lg ${
                msg.role === "user"
                  ? "bg-blue-600 text-white self-end ml-auto"
                  : "bg-gray-800 text-gray-200 self-start mr-auto"
              }`}
            >
              {msg.text}
            </div>
          ))
        )}
        {isLoading && (
          <div className="bg-gray-800 text-gray-200 self-start mr-auto max-w-[80%] p-3 rounded-lg animate-pulse">
            Thinking...
          </div>
        )}
      </div>
      <div className="p-4 bg-gray-800 border-t border-gray-700 flex gap-2">
        <input
          type="text"
          className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 outline-none focus:ring-2 focus:ring-blue-500 transition-all"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask something about the knowledge base..."
          disabled={isLoading}
        />
        <button
          onClick={handleSend}
          disabled={isLoading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}

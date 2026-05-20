"use client";

import { useChat } from "@/hooks/useChat";
import MessageBubble from "./MessageBubble";

export default function ChatModule() {
  const { messages, input, setInput, isLoading, handleSend } = useChat();

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto border border-gray-700 rounded-lg overflow-hidden bg-gray-900 text-gray-100">
      <div className="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-10">Start a conversation with L-SRAG...</div>
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble key={idx} msg={msg} />
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
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 cursor-pointer"
        >
          Send
        </button>
      </div>
    </div>
  );
}

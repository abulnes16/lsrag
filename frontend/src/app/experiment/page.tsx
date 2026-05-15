"use client";

import { useState } from "react";
import { sendQuery } from "@/services/api";

type Message = { role: "user" | "bot"; text: string };

const MessageBubble = ({ msg }: { msg: Message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const limit = 400;
  const isLong = msg.text.length > limit;
  const displayText = isExpanded ? msg.text : msg.text.substring(0, limit) + (isLong ? "..." : "");

  return (
    <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] px-4 py-2 rounded-lg text-sm ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-200"}`}>
        <p className="whitespace-pre-wrap">{displayText}</p>
        {isLong && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs mt-1 text-blue-400 hover:text-blue-300 font-medium underline block"
          >
            {isExpanded ? "See less" : "See more"}
          </button>
        )}
      </div>
    </div>
  );
};

export default function ExperimentPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("mix");

  const [naiveMessages, setNaiveMessages] = useState<Message[]>([]);
  const [lightMessages, setLightMessages] = useState<Message[]>([]);
  const [isPending, setIsPending] = useState(false);

  const handleSend = async () => {
    if (!query.trim()) return;

    const currentQuery = query;
    setQuery("");

    setNaiveMessages((prev) => [...prev, { role: "user", text: currentQuery }]);
    setLightMessages((prev) => [...prev, { role: "user", text: currentQuery }]);
    setIsPending(true);

    try {
      // Fire both requests in parallel
      const [naiveRes, lightRes] = await Promise.allSettled([
        sendQuery(currentQuery, "naiverag"),
        sendQuery(currentQuery, "lightrag", mode),
      ]);

      if (naiveRes.status === "fulfilled") {
        setNaiveMessages((prev) => [...prev, { role: "bot", text: naiveRes.value.response }]);
      } else {
        setNaiveMessages((prev) => [...prev, { role: "bot", text: "Error fetching NaiveRAG response." }]);
      }

      if (lightRes.status === "fulfilled") {
        setLightMessages((prev) => [...prev, { role: "bot", text: lightRes.value.response }]);
      } else {
        setLightMessages((prev) => [...prev, { role: "bot", text: "Error fetching LightRAG response." }]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsPending(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 p-4 md:p-8 space-y-8">
      {/* Top Controls */}
      <div className="flex items-center space-x-4 bg-gray-900 p-4 rounded-lg shadow-lg">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          className="flex-1 px-4 py-2 bg-gray-800 text-gray-100 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="px-4 py-2 bg-gray-800 text-gray-100 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="mix">Mix Mode</option>
          <option value="hybrid">Hybrid Mode</option>
          <option value="local">Local Mode</option>
          <option value="global">Global Mode</option>
        </select>
        <button
          onClick={handleSend}
          disabled={isPending}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded font-medium transition-colors"
        >
          {isPending ? "Testing..." : "Send"}
        </button>
      </div>

      {/* Split Chat UI */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-8 h-[50vh]">

        {/* NaiveRAG Column */}
        <div className="flex flex-col bg-gray-900 rounded-lg shadow-lg border border-gray-800 overflow-hidden">
          <div className="p-3 bg-gray-800 border-b border-gray-700 text-center font-bold text-gray-200">
            NaiveRAG
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[500px]">
            {naiveMessages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {isPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-800 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  NaiveRAG is thinking...
                </div>
              </div>
            )}
          </div>
        </div>

        {/* LightRAG Column */}
        <div className="flex flex-col bg-gray-900 rounded-lg shadow-lg border border-gray-800 overflow-hidden">
          <div className="p-3 bg-gray-800 border-b border-gray-700 text-center font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            LightRAG ({mode})
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[500px]">
            {lightMessages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {isPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-800 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  LSRAG is thinking...
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="bg-gray-900 rounded-lg shadow-lg border border-gray-800 overflow-hidden">
        <table className="w-full text-left text-sm text-gray-300">
          <thead className="bg-gray-800 text-gray-100 uppercase font-medium">
            <tr>
              <th className="px-6 py-4">Metrics</th>
              <th className="px-6 py-4">NaiveRAG</th>
              <th className="px-6 py-4">LSRAG</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            <tr className="hover:bg-gray-800/50 transition-colors">
              <td className="px-6 py-4 font-medium text-white">Faithfulness</td>
              <td className="px-6 py-4">-</td>
              <td className="px-6 py-4">-</td>
            </tr>
            <tr className="hover:bg-gray-800/50 transition-colors">
              <td className="px-6 py-4 font-medium text-white">Context Recall</td>
              <td className="px-6 py-4">-</td>
              <td className="px-6 py-4">-</td>
            </tr>
            <tr className="hover:bg-gray-800/50 transition-colors">
              <td className="px-6 py-4 font-medium text-white">Answer Relevancy</td>
              <td className="px-6 py-4">-</td>
              <td className="px-6 py-4">-</td>
            </tr>
            <tr className="hover:bg-gray-800/50 transition-colors">
              <td className="px-6 py-4 font-medium text-white">NDCG</td>
              <td className="px-6 py-4">-</td>
              <td className="px-6 py-4">-</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

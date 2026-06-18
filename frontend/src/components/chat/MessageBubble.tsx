"use client";

import { useState } from "react";
import { Message } from "@/models";

interface MessageBubbleProps {
  msg: Message;
}

export default function MessageBubble({ msg }: MessageBubbleProps) {
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
            className="text-xs mt-1 text-blue-400 hover:text-blue-300 font-medium underline block cursor-pointer"
          >
            {isExpanded ? "See less" : "See more"}
          </button>
        )}
      </div>
    </div>
  );
}

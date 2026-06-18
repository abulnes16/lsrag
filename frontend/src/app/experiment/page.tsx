"use client";

import { useExperiment } from "@/hooks/useExperiment";
import ExperimentControls from "@/components/experiment/ExperimentControls";
import MessageBubble from "@/components/chat/MessageBubble";
import MetricsTable from "@/components/experiment/MetricsTable";

export default function ExperimentPage() {
  const {
    query,
    setQuery,
    reference,
    setReference,
    mode,
    setMode,
    naiveMessages,
    lightMessages,
    isPending,
    evalPending,
    metrics,
    handleSend,
    handleEvaluate,
  } = useExperiment();

  return (
    <div className="flex flex-col flex-1 p-4 md:p-8 space-y-8">
      {/* Top Controls */}
      <ExperimentControls
        query={query}
        setQuery={setQuery}
        reference={reference}
        setReference={setReference}
        mode={mode}
        setMode={setMode}
        isPending={isPending}
        evalPending={evalPending}
        handleSend={handleSend}
        handleEvaluate={handleEvaluate}
      />

      {/* Split Chat UI */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-8 min-h-[350px]">
        {/* NaiveRAG Column */}
        <div className="flex flex-col bg-gray-900 rounded-xl shadow-lg border border-gray-800 overflow-hidden">
          <div className="p-3 bg-gray-800 border-b border-gray-700 text-center font-bold text-gray-200 uppercase tracking-wider text-xs">
            NaiveRAG
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[450px]">
            {naiveMessages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-500 text-sm italic">
                No messages yet.
              </div>
            ) : (
              naiveMessages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} />
              ))
            )}
            {isPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-800 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  NaiveRAG is thinking...
                </div>
              </div>
            )}
            {evalPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-800 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  Evaluating NaiveRAG response & metrics...
                </div>
              </div>
            )}
          </div>
        </div>

        {/* LightRAG Column */}
        <div className="flex flex-col bg-gray-900 rounded-xl shadow-lg border border-gray-800 overflow-hidden">
          <div className="p-3 bg-gray-800 border-b border-gray-700 text-center font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 uppercase tracking-wider text-xs">
            LSRAG (LightRAG - {mode})
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[450px]">
            {lightMessages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-500 text-sm italic">
                No messages yet.
              </div>
            ) : (
              lightMessages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} />
              ))
            )}
            {isPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-800 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  LSRAG is thinking...
                </div>
              </div>
            )}
            {evalPending && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-gray-850 text-gray-400 px-4 py-2 rounded-lg text-xs italic">
                  Evaluating LSRAG response & metrics...
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Table */}
      <MetricsTable metrics={metrics} mode={mode} />
    </div>
  );
}

"use client";

import { EvaluationMetrics } from "@/models";

interface MetricsTableProps {
  metrics: EvaluationMetrics | null;
  mode: string;
}

export default function MetricsTable({ metrics, mode }: MetricsTableProps) {
  const formatMetricValue = (val: unknown) => {
    if (val === null || val === undefined) return "-";
    const num = (typeof val === "object" && val !== null && "_value" in val)
      ? (val as { _value: unknown })._value
      : val;
    if (typeof num === "number") {
      return num.toFixed(4);
    }
    return "-";
  };

  return (
    <div className="bg-gray-900 rounded-xl shadow-2xl border border-gray-800 overflow-hidden">
      <div className="px-6 py-4 bg-gray-855 border-b border-gray-800 flex items-center justify-between">
        <h3 className="font-bold text-gray-200 text-sm uppercase tracking-wider">Evaluation Metrics Comparison</h3>
        {metrics && (
          <span className="text-xs bg-purple-950 text-purple-300 border border-purple-800 px-2.5 py-1 rounded-full font-medium animate-pulse">
            Live Scores Loaded
          </span>
        )}
      </div>
      <table className="w-full text-left text-sm text-gray-300">
        <thead className="bg-gray-955 text-gray-400 uppercase font-semibold text-xs tracking-wider border-b border-gray-855">
          <tr>
            <th className="px-6 py-4">Metric</th>
            <th className="px-6 py-4">NaiveRAG Score</th>
            <th className="px-6 py-4">LSRAG Score ({mode})</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-855">
          <tr className="hover:bg-gray-850/30 transition-colors">
            <td className="px-6 py-4">
              <div className="font-medium text-white">Faithfulness</div>
              <div className="text-xs text-gray-500">Measures factual consistency of generated answer with retrieved contexts.</div>
            </td>
            <td className="px-6 py-4 text-blue-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.naive?.faithfulness)}</td>
            <td className="px-6 py-4 text-purple-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.lightrag?.faithfulness)}</td>
          </tr>
          <tr className="hover:bg-gray-855/30 transition-colors">
            <td className="px-6 py-4">
              <div className="font-medium text-white">Context Recall</div>
              <div className="text-xs text-gray-500">Measures alignment between ground-truth reference and retrieved contexts.</div>
            </td>
            <td className="px-6 py-4 text-blue-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.naive?.recall)}</td>
            <td className="px-6 py-4 text-purple-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.lightrag?.recall)}</td>
          </tr>
          <tr className="hover:bg-gray-855/30 transition-colors">
            <td className="px-6 py-4">
              <div className="font-medium text-white">Answer Relevancy</div>
              <div className="text-xs text-gray-500">{"Measures semantic relevance of the generated response to the user's query."}</div>
            </td>
            <td className="px-6 py-4 text-blue-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.naive?.relevancy)}</td>
            <td className="px-6 py-4 text-purple-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.lightrag?.relevancy)}</td>
          </tr>
          <tr className="hover:bg-gray-855/30 transition-colors">
            <td className="px-6 py-4">
              <div className="font-medium text-white">NDCG (Ranking)</div>
              <div className="text-xs text-gray-500">Measures the ranking quality of retrieved contexts using cosine similarity relevance.</div>
            </td>
            <td className="px-6 py-4 text-blue-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.naive?.ndcg)}</td>
            <td className="px-6 py-4 text-purple-400 font-mono font-semibold text-base">{formatMetricValue(metrics?.lightrag?.ndcg)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

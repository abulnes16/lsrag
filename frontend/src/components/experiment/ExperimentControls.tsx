"use client";

interface ExperimentControlsProps {
  query: string;
  setQuery: (val: string) => void;
  reference: string;
  setReference: (val: string) => void;
  mode: string;
  setMode: (val: string) => void;
  isPending: boolean;
  evalPending: boolean;
  handleSend: () => void;
  handleEvaluate: () => void;
}

export default function ExperimentControls({
  query,
  setQuery,
  reference,
  setReference,
  mode,
  setMode,
  isPending,
  evalPending,
  handleSend,
  handleEvaluate,
}: ExperimentControlsProps) {
  const isAnyPending = isPending || evalPending;

  return (
    <div className="flex flex-col space-y-4 bg-gray-900/80 backdrop-blur-md p-6 rounded-xl border border-gray-800 shadow-2xl">
      <div className="flex flex-col md:flex-row md:space-x-4 space-y-3 md:space-y-0">
        <div className="flex-1 flex flex-col space-y-1">
          <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Question</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question (e.g. why did Stalin want control of Eastern Europe?)..."
            onKeyDown={(e) => e.key === "Enter" && (reference ? handleEvaluate() : handleSend())}
            className="w-full px-4 py-3 bg-gray-955 border border-gray-800 focus:border-blue-500 text-gray-100 rounded-lg focus:outline-none transition-colors text-sm"
          />
        </div>
        <div className="flex-1 flex flex-col space-y-1">
          <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Ground Truth Reference (Required for Metrics)</label>
          <input
            type="text"
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            placeholder="Enter the golden answer reference..."
            onKeyDown={(e) => e.key === "Enter" && handleEvaluate()}
            className="w-full px-4 py-3 bg-gray-955 border border-gray-800 focus:border-purple-500 text-gray-100 rounded-lg focus:outline-none transition-colors text-sm"
          />
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between pt-2 space-y-3 sm:space-y-0">
        <div className="flex items-center space-x-3 w-full sm:w-auto">
          <span className="text-sm text-gray-400 font-medium">LSRAG Mode:</span>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="px-4 py-2 bg-gray-955 border border-gray-800 text-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer text-sm"
          >
            <option value="mix">Mix Mode (Default)</option>
            <option value="hybrid">Hybrid Mode</option>
            <option value="local">Local Mode</option>
            <option value="global">Global Mode</option>
          </select>
        </div>

        <div className="flex items-center space-x-4 w-full sm:w-auto justify-end">
          <button
            onClick={handleSend}
            disabled={isAnyPending || !query.trim()}
            className="px-5 py-2.5 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-200 rounded-lg font-semibold transition-all border border-gray-700 w-full sm:w-auto text-sm cursor-pointer"
          >
            {isPending ? "Asking..." : "Ask Chat (No Metrics)"}
          </button>
          <button
            onClick={handleEvaluate}
            disabled={isAnyPending || !query.trim() || !reference.trim()}
            className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-40 text-white rounded-lg font-semibold shadow-lg shadow-purple-500/20 transition-all w-full sm:w-auto text-sm cursor-pointer"
          >
            {evalPending ? "Evaluating..." : "Run Evaluation"}
          </button>
        </div>
      </div>
    </div>
  );
}

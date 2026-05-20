import { useState } from "react";
import { Message, EvaluationMetrics } from "@/models";
import { api } from "@/services/api";

export function useExperiment() {
  const [query, setQuery] = useState("");
  const [reference, setReference] = useState("");
  const [mode, setMode] = useState("mix");

  const [naiveMessages, setNaiveMessages] = useState<Message[]>([]);
  const [lightMessages, setLightMessages] = useState<Message[]>([]);
  const [isPending, setIsPending] = useState(false);
  const [evalPending, setEvalPending] = useState(false);
  
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);

  const handleSend = async () => {
    if (!query.trim()) return;

    const currentQuery = query;
    setQuery("");

    setNaiveMessages((prev) => [...prev, { role: "user", text: currentQuery }]);
    setLightMessages((prev) => [...prev, { role: "user", text: currentQuery }]);
    setIsPending(true);

    try {
      const [naiveRes, lightRes] = await Promise.allSettled([
        api.chat(currentQuery, "naiverag"),
        api.chat(currentQuery, "lightrag", mode),
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

  const handleEvaluate = async () => {
    if (!query.trim() || !reference.trim()) return;

    const currentQuery = query;
    const currentRef = reference;
    setQuery("");
    setReference("");

    setNaiveMessages((prev) => [...prev, { role: "user", text: `${currentQuery}\n\n[Expected Reference: ${currentRef}]` }]);
    setLightMessages((prev) => [...prev, { role: "user", text: `${currentQuery}\n\n[Expected Reference: ${currentRef}]` }]);
    setEvalPending(true);

    try {
      const evalRes = await api.evaluate(currentQuery, currentRef, mode);

      const naiveData = evalRes.naive;
      const lightData = evalRes.lightrag;

      setNaiveMessages((prev) => [...prev, { role: "bot", text: naiveData?.response || "No response received from NaiveRAG." }]);
      setLightMessages((prev) => [...prev, { role: "bot", text: lightData?.response || "No response received from LSRAG." }]);

      setMetrics({
        naive: naiveData?.metrics || {},
        lightrag: lightData?.metrics || {}
      });

    } catch (err) {
      console.error("Evaluation error:", err);
      setNaiveMessages((prev) => [...prev, { role: "bot", text: "Error running NaiveRAG evaluation." }]);
      setLightMessages((prev) => [...prev, { role: "bot", text: "Error running LSRAG evaluation." }]);
    } finally {
      setEvalPending(false);
    }
  };

  return {
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
  };
}

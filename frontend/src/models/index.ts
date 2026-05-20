export interface Message {
  role: "user" | "bot";
  text: string;
}

export interface QueryRequest {
  query: string;
  rag_type?: string;
  lightrag_mode?: string;
}

export interface EvaluateRequest {
  query: string;
  reference: string;
  lightrag_mode?: string;
}

export interface MetricDetail {
  _value: number | null;
  reason?: string | null;
  traces?: unknown;
}

export interface ServiceMetrics {
  faithfulness?: MetricDetail | number | null;
  recall?: MetricDetail | number | null;
  relevancy?: MetricDetail | number | null;
  ndcg?: number | null;
}

export interface EvaluationMetrics {
  naive?: ServiceMetrics;
  lightrag?: ServiceMetrics;
}

export interface QueryResponse {
  response: string;
  error?: string;
}

export interface EvaluateResponse {
  naive?: {
    response: string;
    metrics: ServiceMetrics;
  };
  lightrag?: {
    response: string;
    metrics: ServiceMetrics;
  };
}

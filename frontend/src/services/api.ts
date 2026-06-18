import { QueryResponse, EvaluateResponse } from "@/models";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async post<T>(endpoint: string, data: Record<string, unknown>): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API error on ${endpoint}: ${response.statusText}`);
    }

    return response.json() as Promise<T>;
  }

  public async chat(query: string, ragType?: string, lightragMode?: string): Promise<QueryResponse> {
    const payload: Record<string, unknown> = { query };
    if (ragType) payload.rag_type = ragType;
    if (lightragMode) payload.lightrag_mode = lightragMode;
    return this.post<QueryResponse>("/chat", payload);
  }

  public async evaluate(query: string, reference: string, lightragMode?: string): Promise<EvaluateResponse> {
    const payload: Record<string, unknown> = { query, reference };
    if (lightragMode) payload.lightrag_mode = lightragMode;
    return this.post<EvaluateResponse>("/evaluate", payload);
  }
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const api = new ApiClient(API_URL);

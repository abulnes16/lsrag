export async function sendQuery(query: string) {
  // Cuando se ejecuta en el cliente, NEXT_PUBLIC_API_URL es útil, 
  // O podemos usar un proxy /api si evitamos CORS, pero para este caso usaremos directo el endpoint expuesto.
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error communicating with the backend:", error);
    throw error;
  }
}

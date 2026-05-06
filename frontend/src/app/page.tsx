import ChatModule from "@/components/chat/ChatModule";

export default function Home() {
  return (
    <div className="min-h-screen bg-black flex flex-col font-[family-name:var(--font-geist-sans)]">
      <header className="p-6 text-center border-b border-gray-800">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          L-SRAG System
        </h1>
        <p className="text-gray-400 mt-2 text-sm">Lightweight Semantic Retrieval-Augmented Generation</p>
      </header>
      
      <main className="flex-1 p-4 md:p-8">
        <div className="h-[80vh]">
          <ChatModule />
        </div>
      </main>
    </div>
  );
}

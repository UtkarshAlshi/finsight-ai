import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="flex flex-col h-screen max-w-4xl mx-auto">
      <header className="px-4 py-3 border-b border-gray-200 bg-white flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-900">FinSight AI</h1>
          <p className="text-xs text-gray-500">
            Multi-agent financial intelligence
          </p>
        </div>
        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium">
          Beta
        </span>
      </header>
      <div className="flex-1 min-h-0">
        <ChatInterface />
      </div>
    </main>
  );
}

import ChatModule from "@/components/chat/ChatModule";

export default function Home() {
  return (
    <div className="flex-1 p-4 md:p-8">
      <div className="h-[80vh]">
        <ChatModule />
      </div>
    </div>
  );
}

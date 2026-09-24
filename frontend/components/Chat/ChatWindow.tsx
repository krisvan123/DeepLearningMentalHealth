import React, { useRef, useEffect } from "react";
import { Message } from "@/lib/types";
import { ChatMessage } from "./ChatMessage";
import { EmptyState } from "./EmptyState";

interface ChatWindowProps {
  messages: Message[];
  onSelectPrompt: (prompt: string) => void;
  developerMode?: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  onSelectPrompt,
  developerMode = false,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 sm:p-8 overflow-y-auto">
        <EmptyState onSelectPrompt={onSelectPrompt} />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
      <div className="max-w-[850px] w-full mx-auto space-y-5 sm:space-y-6">
        {messages.map((msg) => (
          <ChatMessage
            key={msg.id}
            message={msg}
            developerMode={developerMode}
          />
        ))}
        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
};

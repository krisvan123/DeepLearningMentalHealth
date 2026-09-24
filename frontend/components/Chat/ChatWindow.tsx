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
      <div className="flex-1 flex items-center justify-center">
        <EmptyState onSelectPrompt={onSelectPrompt} />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-2 sm:px-4 py-4 space-y-2">
      {messages.map((msg) => (
        <ChatMessage
          key={msg.id}
          message={msg}
          developerMode={developerMode}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

"use client";

import React, { useState, useEffect } from "react";
import { Menu } from "lucide-react";
import { Message, ModelStatus } from "@/lib/types";
import { getModelStatus, sendMessage } from "@/lib/api";
import { APP_NAME, APP_SUBTITLE } from "@/lib/config";
import { StatusIndicator } from "@/components/UI/StatusIndicator";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { ChatWindow } from "@/components/Chat/ChatWindow";
import { ChatInput } from "@/components/Chat/ChatInput";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [developerMode, setDeveloperMode] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  // Fetch model health and configuration status on load
  useEffect(() => {
    async function checkStatus() {
      try {
        const status = await getModelStatus();
        setModelStatus(status);
      } catch {
        setModelStatus({
          status: "unavailable",
          model_ready: false,
          model_error: "Inference server offline",
          architecture: "Bi-LSTM",
          task: "Multi-class Classification",
          num_classes: 8,
          classes: [],
          confidence_threshold: 0.35,
        });
      }
    }
    checkStatus();
  }, []);

  const handleSendMessage = async (userText: string) => {
    if (!userText.trim() || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const newUserMessage: Message = {
      id: userMessageId,
      role: "user",
      content: userText,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      // Find the last assistant response to avoid immediate repetition in response manager
      const lastAssistant = [...messages]
        .reverse()
        .find((m) => m.role === "assistant")?.content;

      const result = await sendMessage(userText, lastAssistant);

      const assistantMessage: Message = {
        id: `asst-${Date.now()}`,
        role: "assistant",
        content: result.response,
        timestamp: new Date().toISOString(),
        predicted_class: result.predicted_class,
        confidence: result.confidence,
        is_high_risk: result.is_high_risk,
        is_ood: result.is_ood,
        latency_ms: result.latency_ms,
        probabilities: result.probabilities,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      const fallbackMessage: Message = {
        id: `asst-${Date.now()}`,
        role: "assistant",
        content: "Something went wrong while processing your message. Please try again.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewConversation = () => {
    setMessages([]);
  };

  const currentStatus = modelStatus?.model_ready
    ? "ready"
    : modelStatus?.status === "connecting"
    ? "connecting"
    : "unavailable";

  return (
    <div className="flex h-screen overflow-hidden bg-white text-mono-900 font-sans">
      {/* Sidebar Component */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewConversation={handleNewConversation}
        modelStatus={modelStatus}
        developerMode={developerMode}
        onToggleDeveloperMode={setDeveloperMode}
      />

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col h-full lg:pl-72">
        {/* Header */}
        <header className="flex items-center justify-between px-4 sm:px-6 py-3.5 border-b border-mono-200 bg-white/90 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-lg text-mono-700 hover:text-black hover:bg-mono-100 lg:hidden"
              aria-label="Open sidebar menu"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-sm sm:text-base font-bold tracking-widest text-black uppercase">
                {APP_NAME}
              </h1>
              <p className="text-[11px] text-mono-500 font-normal">
                {APP_SUBTITLE}
              </p>
            </div>
          </div>

          <StatusIndicator status={currentStatus} />
        </header>

        {/* Scrollable Chat Area */}
        <main className="flex-1 flex flex-col overflow-hidden max-w-chat mx-auto w-full">
          <ChatWindow
            messages={messages}
            onSelectPrompt={handleSendMessage}
            developerMode={developerMode}
          />

          {/* Sticky Bottom Input Area */}
          <div className="p-3 sm:p-5 bg-white border-t border-mono-100">
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

import React from "react";
import { APP_NAME, QUICK_PROMPTS } from "@/lib/config";

interface EmptyStateProps {
  onSelectPrompt: (prompt: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onSelectPrompt }) => {
  return (
    <div className="flex flex-col items-center justify-center text-center max-w-xl mx-auto py-12 px-4">
      {/* Brand & Subtitle */}
      <h1 className="text-3xl font-extrabold tracking-widest text-black uppercase mb-2">
        {APP_NAME}
      </h1>
      <h2 className="text-lg font-normal text-mono-800 mb-3">
        A space to express what you&apos;re feeling.
      </h2>
      <p className="text-sm text-mono-600 max-w-md mx-auto leading-relaxed mb-8">
        You don&apos;t need the perfect words.
        <br />
        Start wherever feels natural.
      </p>

      {/* Quick Prompts */}
      <div className="w-full">
        <p className="text-xs font-semibold uppercase tracking-wider text-mono-400 mb-3">
          Example starting points
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full">
          {QUICK_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => onSelectPrompt(prompt)}
              className="text-left text-xs sm:text-sm font-normal py-2.5 px-4 rounded-xl border border-mono-300 bg-white text-mono-900 hover:border-black hover:bg-mono-50 transition-all duration-150 shadow-sm"
            >
              &quot;{prompt}&quot;
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

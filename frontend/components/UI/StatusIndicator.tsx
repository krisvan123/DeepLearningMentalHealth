import React from "react";

interface StatusIndicatorProps {
  status: "ready" | "connecting" | "unavailable";
  label?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
}) => {
  const getStatusDisplay = () => {
    switch (status) {
      case "ready":
        return {
          text: label || "MODEL READY",
          dotClass: "bg-black",
          containerClass: "border-mono-300 bg-mono-100 text-mono-900",
        };
      case "connecting":
        return {
          text: label || "CONNECTING",
          dotClass: "bg-mono-500 animate-pulse",
          containerClass: "border-mono-200 bg-mono-50 text-mono-700",
        };
      case "unavailable":
      default:
        return {
          text: label || "MODEL UNAVAILABLE",
          dotClass: "border-2 border-mono-500 bg-transparent",
          containerClass: "border-mono-200 bg-mono-50 text-mono-600",
        };
    }
  };

  const current = getStatusDisplay();

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1 text-xs font-semibold tracking-wider uppercase rounded-full border transition-all ${current.containerClass}`}
      role="status"
    >
      <span className={`w-2 h-2 rounded-full ${current.dotClass}`} />
      <span>{current.text}</span>
    </div>
  );
};

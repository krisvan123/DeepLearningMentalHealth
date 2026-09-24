import React from "react";
import { ModelStatus as ModelStatusType } from "@/lib/types";

interface ModelStatusProps {
  status: ModelStatusType | null;
  loading?: boolean;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({ status, loading }) => {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-xs pb-1 border-b border-mono-200">
        <span className="text-mono-500">Architecture</span>
        <span className="font-semibold text-mono-900">
          {status?.architecture || "Bi-LSTM"}
        </span>
      </div>
      <div className="flex justify-between items-center text-xs pb-1 border-b border-mono-200">
        <span className="text-mono-500">Task</span>
        <span className="font-semibold text-mono-900">
          {status?.task || "Multi-class Classification"}
        </span>
      </div>
      <div className="flex justify-between items-center text-xs pb-1 border-b border-mono-200">
        <span className="text-mono-500">Classes</span>
        <span className="font-semibold text-mono-900">
          {status?.num_classes || 8} Categories
        </span>
      </div>
      <div className="flex justify-between items-center text-xs pb-1 border-b border-mono-200">
        <span className="text-mono-500">Inference</span>
        <span className="font-semibold text-mono-900">
          {status?.model_ready ? "Remote / Local API" : "Offline"}
        </span>
      </div>
      <div className="flex justify-between items-center text-xs pb-1 border-b border-mono-200">
        <span className="text-mono-500">OOD Threshold</span>
        <span className="font-semibold text-mono-900">
          {status?.confidence_threshold ? `${(status.confidence_threshold * 100).toFixed(0)}%` : "35%"}
        </span>
      </div>
    </div>
  );
};

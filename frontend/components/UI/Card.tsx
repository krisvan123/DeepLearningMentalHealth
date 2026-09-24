import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  bordered?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = "",
  bordered = true,
}) => {
  return (
    <div
      className={`bg-white rounded-xl ${
        bordered ? "border border-mono-200" : ""
      } p-4 ${className}`}
    >
      {children}
    </div>
  );
};

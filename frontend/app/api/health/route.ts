import { NextResponse } from "next/server";
import { DEFAULT_CLASSES } from "@/lib/config";

export async function GET() {
  const backendUrl =
    process.env.MODEL_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://127.0.0.1:8000";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const res = await fetch(`${backendUrl}/health`, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend offline / not reachable
  }

  // Graceful offline status response
  return NextResponse.json({
    status: "unavailable",
    model_ready: false,
    model_error: "Inference server offline or not configured",
    architecture: "Bi-LSTM (Bidirectional Long Short-Term Memory)",
    task: "Multi-class Text Classification",
    num_classes: DEFAULT_CLASSES.length,
    classes: DEFAULT_CLASSES,
    confidence_threshold: 0.35,
  });
}

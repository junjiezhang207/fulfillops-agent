export type StatusTone = "ok" | "warn" | "danger" | "info" | "neutral";

export function riskTone(level?: string): StatusTone {
  const normalized = (level || "").toUpperCase();
  if (normalized === "CRITICAL") return "danger";
  if (normalized === "HIGH") return "danger";
  if (normalized === "MEDIUM") return "warn";
  if (normalized === "LOW") return "ok";
  return "neutral";
}

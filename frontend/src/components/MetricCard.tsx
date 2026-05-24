import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { StatusPill } from "./StatusPill";
import type { StatusTone } from "../lib/status";

type MetricCardProps = {
  icon: LucideIcon;
  label: string;
  value: ReactNode;
  description: string;
  trend?: string;
  tone?: StatusTone;
};

export function MetricCard({
  icon: Icon,
  label,
  value,
  description,
  trend,
  tone = "info",
}: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <Icon size={22} />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{description}</small>
      {trend && <StatusPill tone={tone}>{trend}</StatusPill>}
    </article>
  );
}

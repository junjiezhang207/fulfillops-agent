import clsx from "clsx";
import type { ReactNode } from "react";

import type { StatusTone } from "../lib/status";

type StatusPillProps = {
  tone?: StatusTone;
  children: ReactNode;
};

export function StatusPill({ tone = "neutral", children }: StatusPillProps) {
  return <span className={clsx("status-pill", `status-pill--${tone}`)}>{children}</span>;
}

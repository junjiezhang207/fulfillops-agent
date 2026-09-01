import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { HybridRunResult, InterruptEvent } from "../lib/api";

export type PendingReview = {
  threadId: string;
  orderId: string;
  question: string;
  interrupt: InterruptEvent;
  sourceResult: HybridRunResult;
  createdAt: string;
};

type ReviewState = {
  pendingReviews: PendingReview[];
  activeThreadId: string | null;
  addPendingReview: (review: PendingReview) => void;
  removePendingReview: (threadId: string) => void;
  setActiveThreadId: (threadId: string | null) => void;
  getReview: (threadId: string) => PendingReview | undefined;
};

export const useReviewStore = create<ReviewState>()(
  persist(
    (set, get) => ({
      pendingReviews: [],
      activeThreadId: null,
      addPendingReview: (review) =>
        set((state) => ({
          pendingReviews: [
            review,
            ...state.pendingReviews.filter((item) => item.threadId !== review.threadId),
          ].slice(0, 20),
          activeThreadId: review.threadId,
        })),
      removePendingReview: (threadId) =>
        set((state) => ({
          pendingReviews: state.pendingReviews.filter((item) => item.threadId !== threadId),
          activeThreadId:
            state.activeThreadId === threadId ? null : state.activeThreadId,
        })),
      setActiveThreadId: (threadId) => set({ activeThreadId: threadId }),
      getReview: (threadId) =>
        get().pendingReviews.find((item) => item.threadId === threadId),
    }),
    {
      name: "fulfillops-review-workbench",
      partialize: (state) => ({
        pendingReviews: state.pendingReviews,
        activeThreadId: state.activeThreadId,
      }),
    },
  ),
);

import { Navigate, Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { DataImportPage } from "./pages/DataImportPage";
import { OperationsDashboard } from "./pages/OperationsDashboard";
import { ReviewQueuePage, StandaloneReviewPage } from "./pages/ReviewPage";
import { SystemPage } from "./pages/SystemPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Shell />}>
        <Route index element={<OperationsDashboard />} />
        <Route path="data" element={<DataImportPage />} />
        <Route path="reviews" element={<ReviewQueuePage />} />
        <Route path="system" element={<SystemPage />} />
      </Route>
      <Route path="/review/:threadId" element={<StandaloneReviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

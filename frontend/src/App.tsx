import { Navigate, Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { DataImportPage } from "./pages/DataImportPage";
import { OperationsDashboard } from "./pages/OperationsDashboard";
import { TraceCenterPage } from "./pages/TraceCenterPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Shell />}>
        <Route index element={<OperationsDashboard />} />
        <Route path="data" element={<DataImportPage />} />
        <Route path="traces" element={<TraceCenterPage />} />
        <Route path="traces/:traceId" element={<TraceCenterPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

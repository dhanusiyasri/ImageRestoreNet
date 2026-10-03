import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import AIRestoration from "./pages/AIRestoration";
import EvaluationMetrics from "./pages/EvaluationMetrics";
import InspectionQueue from "./pages/InspectionQueue";
import ModelStatus from "./pages/ModelStatus";
import Reports from "./pages/Reports";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/restoration" element={<AIRestoration />} />
          <Route path="/metrics" element={<EvaluationMetrics />} />
          <Route path="/queue" element={<InspectionQueue />} />
          <Route path="/model-status" element={<ModelStatus />} />
          <Route path="/reports" element={<Reports />} />
        </Routes>
      </div>
    </div>
  );
}

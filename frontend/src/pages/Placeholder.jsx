import TopBar from "../components/TopBar";

export default function Placeholder({ title, blurb }) {
  return (
    <div>
      <TopBar title="Inspection System" />
      <div className="page-heading">
        <h2>{title}</h2>
        <p>{blurb}</p>
      </div>
      <div className="card">
        <p style={{ color: "var(--text-secondary)" }}>
          Not built out yet — add this page's data source and wire it up the
          same way Dashboard.jsx and EvaluationMetrics.jsx call the backend.
        </p>
      </div>
    </div>
  );
}

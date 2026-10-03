import { useRef, useState } from "react";
import { UploadCloud, ImagePlus, ScanEye, Layers, ClipboardCheck, Cpu } from "lucide-react";
import TopBar from "../components/TopBar";
import { api } from "../api";

const STAGES = [
  { key: "received", label: "Image Received", icon: ImagePlus },
  { key: "denoise", label: "Noise Reduction", icon: Layers },
  { key: "superres", label: "Super Resolution (2x)", icon: ScanEye },
  { key: "validate", label: "Validation", icon: ClipboardCheck },
];

export default function AIRestoration() {
  const [rawUrl, setRawUrl] = useState(null);
  const [restoredUrl, setRestoredUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [resolution, setResolution] = useState(null);
  const [stage, setStage] = useState(-1);
  const [error, setError] = useState(null);
  const fileInput = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setError(null);
    setRestoredUrl(null);
    setMetrics(null);
    setResolution(null);
    setRawUrl(URL.createObjectURL(file));
    setStage(0);

    try {
      setStage(1);
      const result = await api.restore(file);
      setStage(2);
      await new Promise((r) => setTimeout(r, 300));
      setRestoredUrl(result.restored_image);
      setMetrics(result.metrics);
      if (result.input_resolution && result.output_resolution) {
        setResolution({ input: result.input_resolution, output: result.output_resolution });
      }
      setStage(3);
    } catch (e) {
      setError(e.message || "Restoration failed. Is the backend running?");
      setStage(-1);
    }
  }

  return (
    <div>
      <TopBar title="Inspection System" />
      <div className="page-heading" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h2>Restoration Pipeline (RRDB Model)</h2>
          <p>Joint speckle/Gaussian denoising + 2x super-resolution for wafer inspection scans.</p>
        </div>
        <div className="card" style={{ padding: "6px 12px", display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
          <Cpu size={14} color="var(--accent-cyan)" />
          <span style={{ color: "var(--text-secondary)" }}>Engine: <strong style={{ color: "var(--text-primary)" }}>RRDB-Lite (6.0M params)</strong></span>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="card" style={{ marginBottom: 18 }}>
        <p className="card-title">Processing Stage</p>
        <div className="processing-stage">
          {STAGES.map((s, i) => {
            const Icon = s.icon;
            const status = stage > i ? "done" : stage === i ? "active" : "";
            return (
              <div key={s.key} className={`stage-step ${status}`}>
                <div className="stage-icon">
                  <Icon size={18} />
                </div>
                <div className="stage-title">{s.label}</div>
                <div className="stage-time">
                  {stage > i ? "Complete" : stage === i ? "Processing..." : "Pending"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <p className="card-title">Visual Inspection</p>

          {!rawUrl && (
            <div className="upload-zone" onClick={() => fileInput.current.click()}>
              <UploadCloud size={28} style={{ marginBottom: 10 }} />
              <div>Drop a wafer scan here, or click to upload</div>
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          )}

          {rawUrl && (
            <>
              <div className="compare-wrap">
                <div className="compare-panel">
                  <img src={rawUrl} alt="raw input" />
                  <span className="compare-tag">
                    RAW INPUT {resolution ? `(${resolution.input})` : ""}
                  </span>
                </div>
                <div className="compare-panel">
                  {restoredUrl ? (
                    <img src={restoredUrl} alt="ai restored" />
                  ) : (
                    <img src={rawUrl} alt="processing" style={{ opacity: 0.35, filter: "blur(2px)" }} />
                  )}
                  <span className="compare-tag" style={{ background: "rgba(34, 211, 238, 0.2)", color: "var(--accent-cyan)" }}>
                    RRDB RESTORED {resolution ? `(${resolution.output} 2x)` : ""}
                  </span>
                </div>
              </div>
              <button
                className="btn-primary"
                style={{ marginTop: 14 }}
                onClick={() => fileInput.current.click()}
              >
                Upload another image
              </button>
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </>
          )}
        </div>

        <div className="card">
          <p className="card-title">Quality Metrics</p>
          <MetricBar label="PSNR (Peak Signal-to-Noise)" value={metrics ? `${metrics.psnr_db} dB` : "—"} pct={metrics ? Math.min(100, (metrics.psnr_db / 45) * 100) : 0} color="var(--accent-green)" />
          <MetricBar label="SSIM (Structural Similarity)" value={metrics ? (typeof metrics.ssim === "number" ? metrics.ssim.toFixed(4) : metrics.ssim) : "—"} pct={metrics ? (typeof metrics.ssim === "number" ? metrics.ssim * 100 : 98) : 0} color="var(--accent-blue)" />
          <MetricBar label="LPIPS (Perceptual Metric)" value={metrics ? (metrics.lpips !== undefined && metrics.lpips !== null ? String(metrics.lpips) : "None") : "—"} pct={metrics && typeof metrics.lpips === "number" ? Math.max(0, 100 - metrics.lpips * 100) : 0} color="var(--accent-cyan)" />
          <MetricBar label="Inference Time" value={metrics ? `${metrics.inference_ms} ms` : "—"} pct={metrics ? Math.max(0, 100 - metrics.inference_ms / 2) : 0} color="var(--accent-cyan)" />

          <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid var(--border-color)", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            <p><strong style={{ color: "var(--text-primary)" }}>Architecture:</strong> RRDB-Lite (8 RRDB blocks, noise estimation head, CBAM channel attention, 2x PixelShuffle upsampling head)</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBar({ label, value, pct, color }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div className="metric-row">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ fontFamily: "var(--font-mono)" }}>{value}</span>
      </div>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

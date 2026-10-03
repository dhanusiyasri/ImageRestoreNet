import { useEffect, useState } from "react";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from "recharts";
import TopBar from "../components/TopBar";
import { api } from "../api";

export default function EvaluationMetrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    api.evaluationMetrics().then(setMetrics).catch(() => {});
  }, []);

  const radarData = metrics && metrics.radar
    ? Object.entries(metrics.radar).map(([k, v]) => ({ subject: k, value: v }))
    : [];

  return (
    <div>
      <TopBar title="Inspection System" />
      <div className="page-heading">
        <h2>RRDB Model Evaluation</h2>
        <p>Validation benchmark performance across {metrics ? metrics.eval_samples : 321} test scans.</p>
      </div>

      {metrics && (
        <>
          <div className="grid cols-5" style={{ marginBottom: 18 }}>
            <StatCard label="MODEL PSNR" value={`${metrics.model_psnr} dB`} delta={`+${metrics.psnr_gain} dB`} up accent />
            <StatCard label="MODEL SSIM" value={`${metrics.model_ssim}`} delta={`+${metrics.ssim_gain}`} up />
            <StatCard label="PERCEPTUAL (LPIPS)" value={metrics.lpips || "None"} sub="Optional AlexNet metric" />
            <StatCard label="BICUBIC BASELINE" value={`${metrics.bicubic_psnr} dB`} sub="No AI baseline" />
            <StatCard label="EVALUATION DATASET" value={`${metrics.eval_samples} images`} sub="Held-out validation split" />
          </div>

          <div className="grid cols-2">
            <div className="card">
              <p className="card-title">Performance Radar</p>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1f2530" />
                  <PolarAngleAxis dataKey="subject" stroke="#8892a0" fontSize={12} />
                  <Radar dataKey="value" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.25} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <p className="card-title">RRDB Model vs. Bicubic Baseline Benchmark</p>
              <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10, fontSize: 14 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #1f2530", textAlign: "left", color: "var(--text-secondary)" }}>
                    <th style={{ padding: "8px 0" }}>Metric</th>
                    <th style={{ padding: "8px 0" }}>Bicubic Baseline</th>
                    <th style={{ padding: "8px 0" }}>RRDB Model</th>
                    <th style={{ padding: "8px 0", color: "var(--accent-cyan)" }}>Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #141922" }}>
                    <td style={{ padding: "12px 0", fontWeight: 500 }}>PSNR (Fidelity)</td>
                    <td style={{ color: "var(--text-secondary)" }}>{metrics.bicubic_psnr} dB</td>
                    <td style={{ fontWeight: 600, color: "var(--accent-green)" }}>{metrics.model_psnr} dB</td>
                    <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>+{metrics.psnr_gain} dB</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #141922" }}>
                    <td style={{ padding: "12px 0", fontWeight: 500 }}>SSIM (Structure)</td>
                    <td style={{ color: "var(--text-secondary)" }}>{metrics.bicubic_ssim}</td>
                    <td style={{ fontWeight: 600, color: "var(--accent-green)" }}>{metrics.model_ssim}</td>
                    <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>+{metrics.ssim_gain}</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #141922" }}>
                    <td style={{ padding: "12px 0", fontWeight: 500 }}>LPIPS (Perceptual Loss)</td>
                    <td style={{ color: "var(--text-secondary)" }}>N/A</td>
                    <td style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>{metrics.lpips || "None"}</td>
                    <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>Optional metric</td>
                  </tr>
                  <tr>
                    <td style={{ padding: "12px 0", fontWeight: 500 }}>Super Resolution</td>
                    <td style={{ color: "var(--text-secondary)" }}>Interpolated</td>
                    <td style={{ fontWeight: 600, color: "var(--accent-green)" }}>2x PixelShuffle</td>
                    <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>Sub-pixel detail</td>
                  </tr>
                </tbody>
              </table>

              <div style={{ marginTop: 20, fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                Metrics validated via <code>RRDB_modal/utils/metrics.py</code> using skimage.metrics (fixed data_range=1.0) on joint speckle/Gaussian noise degraded wafer scan inputs.
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, delta, up, sub, accent }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">
        <span>{label}</span>
        {delta && <span className={"delta " + (up ? "up" : "down")}>{delta}</span>}
        {sub && <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-secondary)" }}>{sub}</span>}
      </div>
      <div className="stat-value" style={accent ? { color: "var(--accent-cyan)" } : {}}>{value}</div>
    </div>
  );
}

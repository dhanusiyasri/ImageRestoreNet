import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, PieChart, Pie, Cell } from "recharts";
import { Download, FileText, CheckCircle, AlertTriangle, Shield, BarChart3, TrendingUp } from "lucide-react";
import TopBar from "../components/TopBar";
import { api } from "../api";

const PIE_COLORS = ["#22d3ee", "#3b82f6", "#f59e0b", "#10b981"];

export default function Reports() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.reportsData().then(setData).catch(() => {});
  }, []);

  const defectData = data && data.defect_counts
    ? Object.entries(data.defect_counts).map(([k, v]) => ({
        name: k.replace(/_/g, " ").toUpperCase(),
        value: v,
      }))
    : [];

  return (
    <div>
      <TopBar title="Inspection System" />
      <div className="page-heading" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h2>Inspection & Quality Yield Reports</h2>
          <p>Exportable performance summaries and PSNR distribution analytics.</p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <a href={api.reportsDownloadUrl("csv")} download className="btn-primary" style={{ display: "inline-flex", alignItems: "center", gap: 6, textDecoration: "none" }}>
            <Download size={14} /> Download CSV Report
          </a>
          <a href={api.reportsDownloadUrl("json")} download className="btn-primary" style={{ display: "inline-flex", alignItems: "center", gap: 6, textDecoration: "none", background: "#1f2530" }}>
            <FileText size={14} /> Download JSON Summary
          </a>
        </div>
      </div>

      {data && (
        <>
          <div className="grid cols-5" style={{ marginBottom: 18 }}>
            <Card title="PASS YIELD RATE" value={`${data.pass_yield_rate}%`} sub="Evaluated Threshold" accent icon={CheckCircle} />
            <Card title="EVALUATED SCANS" value={data.total_scans_evaluated} sub={`Batch: ${data.batch_id}`} icon={BarChart3} />
            <Card title="AVERAGE PSNR" value={`${data.average_psnr_db} dB`} sub={`Gain: +${data.psnr_gain} dB`} accent icon={Shield} />
            <Card title="AVERAGE SSIM" value={data.average_ssim} sub={`Gain: +${data.ssim_gain}`} icon={TrendingUp} />
            <Card title="PERCEPTUAL LPIPS" value={data.lpips || "None"} sub="AlexNet Perceptual" icon={Shield} />
          </div>

          <div className="grid cols-2" style={{ marginBottom: 18 }}>
            <div className="card">
              <p className="card-title">PSNR Distribution Across Dataset</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data.psnr_distribution}>
                  <CartesianGrid stroke="#1c222c" vertical={false} />
                  <XAxis dataKey="range" stroke="#8892a0" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8892a0" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "#12161d", border: "1px solid #1f2530" }} />
                  <Bar dataKey="count" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <p className="card-title">Defect Category Breakdown</p>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={defectData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={(e) => `${e.name}: ${e.value}`}>
                    {defectData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#12161d", border: "1px solid #1f2530" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <p className="card-title">Model vs. Bicubic Baseline Benchmark Summary</p>
            <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse", marginTop: 10 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1f2530", textAlign: "left", color: "var(--text-secondary)" }}>
                  <th style={{ padding: "8px 0" }}>Inspection Metric</th>
                  <th style={{ padding: "8px 0" }}>Bicubic Baseline</th>
                  <th style={{ padding: "8px 0" }}>RRDB-Lite Model</th>
                  <th style={{ padding: "8px 0", color: "var(--accent-cyan)" }}>Delta Improvement</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid #141922" }}>
                  <td style={{ padding: "10px 0", fontWeight: 600 }}>Peak Signal-to-Noise Ratio (PSNR)</td>
                  <td style={{ color: "var(--text-secondary)" }}>{data.bicubic_psnr} dB</td>
                  <td style={{ color: "var(--accent-green)", fontWeight: 700 }}>{data.average_psnr_db} dB</td>
                  <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>+{data.psnr_gain} dB</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #141922" }}>
                  <td style={{ padding: "10px 0", fontWeight: 600 }}>Structural Similarity Index (SSIM)</td>
                  <td style={{ color: "var(--text-secondary)" }}>{data.bicubic_ssim}</td>
                  <td style={{ color: "var(--accent-green)", fontWeight: 700 }}>{data.average_ssim}</td>
                  <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>+{data.ssim_gain}</td>
                </tr>
                <tr>
                  <td style={{ padding: "10px 0", fontWeight: 600 }}>Perceptual Loss (LPIPS)</td>
                  <td style={{ color: "var(--text-secondary)" }}>N/A</td>
                  <td style={{ color: "var(--accent-green)", fontWeight: 700 }}>{data.lpips || "None"}</td>
                  <td style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>Optional metric</td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function Card({ title, value, sub, accent, icon: Icon }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Icon size={14} /> {title}
        </span>
      </div>
      <div className="stat-value" style={accent ? { color: "var(--accent-green)" } : {}}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{sub}</div>}
    </div>
  );
}

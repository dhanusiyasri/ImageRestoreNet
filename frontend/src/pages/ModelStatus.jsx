import { useEffect, useState } from "react";
import { Cpu, HardDrive, ShieldCheck, Layers, Flame, Zap, Activity } from "lucide-react";
import TopBar from "../components/TopBar";
import { api } from "../api";

export default function ModelStatus() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    api.modelStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <div>
      <TopBar title="Inspection System" />
      <div className="page-heading">
        <h2>Model Architecture & Status</h2>
        <p>Deployment runtime, parameter specifications, and checkpoint state for RRDB-Lite.</p>
      </div>

      {status && (
        <>
          <div className="grid cols-4" style={{ marginBottom: 18 }}>
            <Card
              title="MODEL STATUS"
              value={status.loaded ? "ONLINE" : "OFFLINE"}
              sub={status.device_detail || `Device: ${status.device}`}
              accent={status.loaded}
              icon={Activity}
            />
            <Card
              title="WEIGHTS CHECKPOINT"
              value={`${status.weights_size_mb} MB`}
              sub={status.weights_file}
              icon={HardDrive}
            />
            <Card
              title="TOTAL PARAMETERS"
              value={status.parameters}
              sub={`${status.blocks}x RRDB + CBAM Attention`}
              icon={Cpu}
            />
            <Card
              title="VALIDATION PSNR"
              value={`${status.val_psnr} dB`}
              sub={`SSIM: ${status.val_ssim} | LPIPS: ${status.lpips || "None"}`}
              accent
              icon={Flame}
            />
          </div>

          <div className="grid cols-2">
            <div className="card">
              <p className="card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Layers size={16} color="var(--accent-cyan)" /> Network Architecture Details
              </p>
              <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse", marginTop: 10 }}>
                <tbody>
                  <Row label="Architecture Family" value={status.architecture} />
                  <Row label="Model Version" value={status.version} />
                  <Row label="Residual Blocks" value={`${status.blocks} RRDB Blocks (3 RDBs per block)`} />
                  <Row label="Channel Attention" value="CBAM Squeeze-and-Excitation (Every 3 blocks)" />
                  <Row label="Noise Estimation Head" value="CBDNet-style per-pixel noise map branch" />
                  <Row label="Upscaling Head" value="2x Sub-pixel Convolution (PixelShuffle)" />
                  <Row label="Base Channels / Growth" value={`${status.channels} channels / ${status.growth} growth`} />
                  <Row label="Total Trainable Parameters" value={status.parameters} />
                  <Row label="Best Checkpoint Epoch" value={`Epoch ${status.trained_epoch}`} />
                  <Row label="Validation Benchmark PSNR" value={`${status.val_psnr} dB`} />
                  <Row label="Validation Benchmark SSIM" value={`${status.val_ssim}`} />
                  <Row label="Perceptual Distance (LPIPS)" value={status.lpips || "None"} />
                  <Row label="Inference Execution Device" value={status.device_detail || status.device} />
                </tbody>
              </table>
            </div>

            <div className="card">
              <p className="card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <ShieldCheck size={16} color="var(--accent-green)" /> Calibration & Loss Setup
              </p>
              <p style={{ color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6, marginBottom: 14 }}>
                The RRDB model was trained on degraded semiconductor wafer scans (joint speckle + Gaussian noise, 128x128 to 256x256).
              </p>

              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {status.loss_functions.map((loss, idx) => (
                  <div key={loss} style={{ display: "flex", alignItems: "center", gap: 10, background: "#12161d", padding: "8px 12px", borderRadius: 6, fontSize: 13 }}>
                    <Zap size={14} color="var(--accent-cyan)" />
                    <span><strong>Stage {idx < 1 ? "1 Pretrain" : "2 Fine-tune"}:</strong> {loss}</span>
                  </div>
                ))}
              </div>
            </div>
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

function Row({ label, value }) {
  return (
    <tr style={{ borderBottom: "1px solid #141922" }}>
      <td style={{ padding: "10px 0", color: "var(--text-secondary)" }}>{label}</td>
      <td style={{ padding: "10px 0", fontWeight: 600, textAlign: "right", color: "var(--text-primary)" }}>{value}</td>
    </tr>
  );
}

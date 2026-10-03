import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { ImageIcon, Hourglass, Wand2, Gauge, Activity } from "lucide-react";
import TopBar from "../components/TopBar";
import { api } from "../api";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [volume, setVolume] = useState([]);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.dashboardStats().then((s) => !cancelled && setStats(s)).catch(() => {});
      api.dashboardVolume().then((v) => !cancelled && setVolume(v.series)).catch(() => {});
    };
    load();
    const id = setInterval(load, 3000); // 3-second live refresh
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const cards = stats
    ? [
        { label: "Total Inspected", value: formatK(stats.total_inspected), delta: "+14.2%", up: true, icon: ImageIcon },
        { label: "Images Waiting", value: stats.images_waiting.toLocaleString(), delta: "In Queue", up: false, icon: Hourglass },
        { label: "AI Restored", value: formatK(stats.ai_restored), delta: "RRDB-Lite", up: true, icon: Wand2 },
        { label: "Avg Inference Time", value: `${stats.avg_inference_ms} ms`, delta: "Benchmark", up: true, icon: Gauge },
      ]
    : [];

  return (
    <div>
      <TopBar title="Inspection System" />

      <div className="page-heading" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h2>Live Telemetry</h2>
          <p>Real-time inference and restoration monitoring for wafer inspection.</p>
        </div>
        {stats && (
          <div className="card" style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: 8 }}>
            <Activity size={14} color="var(--accent-green)" />
            <span style={{ color: "var(--text-secondary)", fontSize: 12, fontFamily: "var(--font-mono)" }}>
              BATCH ID <strong style={{ color: "var(--text-primary)" }}>{stats.batch_id}</strong>
            </span>
          </div>
        )}
      </div>

      <div className="grid cols-4" style={{ marginBottom: 18 }}>
        {cards.map(({ label, value, delta, up, icon: Icon }) => (
          <div className="card stat-card" key={label}>
            <div className="stat-label">
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Icon size={14} /> {label}
              </span>
              <span className={"delta " + (up ? "up" : "down")}>{delta}</span>
            </div>
            <div className="stat-value">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid cols-2">
        <div className="card">
          <p className="card-title">Inspection Throughput (24h Volume)</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={volume}>
              <CartesianGrid stroke="#1c222c" vertical={false} />
              <XAxis dataKey="hour" stroke="#8892a0" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#8892a0" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: "#12161d", border: "1px solid #1f2530" }} />
              <Line type="monotone" dataKey="value" stroke="#6d8bf5" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="card-title">AI Restoration Yield (% Nominal)</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={volume}>
              <CartesianGrid stroke="#1c222c" vertical={false} />
              <XAxis dataKey="hour" stroke="#8892a0" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#8892a0" fontSize={11} domain={[90, 100]} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: "#12161d", border: "1px solid #1f2530" }} />
              <Line type="monotone" dataKey="yield" stroke="#22d3ee" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function formatK(n) {
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
}

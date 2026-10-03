import { NavLink } from "react-router-dom";
import { LayoutGrid, Microscope, Wand2, BarChart3, Cpu, FileText, Settings, PlayCircle, UserCircle2 } from "lucide-react";

const links = [
  { to: "/", label: "Dashboard", icon: LayoutGrid },
  { to: "/queue", label: "Inspection Queue", icon: Microscope },
  { to: "/restoration", label: "AI Restoration", icon: Wand2 },
  { to: "/metrics", label: "Evaluation Metrics", icon: BarChart3 },
  { to: "/model-status", label: "Model Status", icon: Cpu },
  { to: "/reports", label: "Reports", icon: FileText },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <Cpu size={22} color="var(--accent-cyan)" />
        <div className="brand-title">SEMICONDUCTOR AI</div>
      </div>
      <div className="brand-sub">Precision Yield Control</div>

      <nav>
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          >
            <Icon size={17} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div style={{ borderTop: "1px solid var(--panel-border)", margin: "8px 0" }} />
        <button className="run-diagnostics-btn">
          <PlayCircle size={16} /> Run Diagnostics
        </button>
        <div className="nav-link" style={{ marginTop: 4 }}>
          <UserCircle2 size={17} /> User Profile
        </div>
      </div>
    </aside>
  );
}

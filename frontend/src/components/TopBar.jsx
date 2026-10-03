import { useEffect, useState } from "react";
import { Search, Bell, HelpCircle } from "lucide-react";
import { api } from "../api";

export default function TopBar({ title }) {
  const [online, setOnline] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      api
        .health()
        .then((r) => !cancelled && setOnline(r.status === "online"))
        .catch(() => !cancelled && setOnline(false));
    };
    check();
    const id = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="topbar">
      <h1>{title}</h1>
      <div className="search-box">
        <Search size={14} />
        <span>Search parameters...</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <span className={"status-pill" + (online ? "" : " offline")}>
          <span className="dot" />
          {online === null ? "CHECKING..." : online ? "MODEL ONLINE" : "MODEL OFFLINE"}
        </span>
        <Bell size={18} color="var(--text-secondary)" />
        <HelpCircle size={18} color="var(--text-secondary)" />
      </div>
    </div>
  );
}

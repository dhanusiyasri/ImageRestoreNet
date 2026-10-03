const BASE = "/api";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body = null) {
  const options = { method: "POST" };
  if (body) {
    if (body instanceof FormData) {
      options.body = body;
    } else {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
  }
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get("/health"),
  modelStatus: () => get("/model/status"),
  dashboardStats: () => get("/dashboard/stats"),
  dashboardVolume: () => get("/dashboard/volume"),
  evaluationMetrics: () => get("/metrics"),
  
  // Inspection Queue API
  queueList: (page = 1, limit = 20, status = "all") =>
    get(`/queue?page=${page}&limit=${limit}&status=${status}`),
  queuePreview: (filename) => get(`/queue/preview/${filename}`),
  queueRestore: (filename) => post(`/queue/restore/${filename}`),
  batchRestore: (count = 10) => post(`/queue/batch-restore?count=${count}`),

  // Reports API
  reportsData: () => get("/reports"),
  reportsDownloadUrl: (format = "csv") => `${BASE}/reports/download?format=${format}`,

  // Image Upload API
  restore: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return post("/restore", form);
  },
};

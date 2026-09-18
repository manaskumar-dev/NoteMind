const API_BASE = "http://localhost:8000";

const Api = {
  token: null,

  setToken(token) {
    this.token = token;
    if (token) localStorage.setItem("token", token);
    else localStorage.removeItem("token");
  },

  loadToken() {
    this.token = localStorage.getItem("token");
    return this.token;
  },

  async request(path, { method = "GET", body, isForm = false } = {}) {
    const headers = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    if (body && !isForm) headers["Content-Type"] = "application/json";

    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: isForm ? body : body ? JSON.stringify(body) : undefined,
    });

    if (res.status === 204) return null;

    let data = null;
    try { data = await res.json(); } catch (_) { /* empty body */ }

    if (!res.ok) {
      const msg = (data && data.detail) || `Request failed (${res.status})`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  },

  // ---- Auth ----
  register(email, password) {
    return this.request("/auth/register", { method: "POST", body: { email, password } });
  },

  async login(email, password) {
    const form = new URLSearchParams();
    form.append("username", email);
    form.append("password", password);
    const res = await fetch(`${API_BASE}/auth/login`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");
    return data;
  },

  // ---- Notes ----
  listNotes(q = "", tag = "") {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (tag) params.set("tag", tag);
    return this.request(`/notes?${params.toString()}`);
  },
  getNote(id) { return this.request(`/notes/${id}`); },
  renderNote(id) { return this.request(`/notes/${id}/render`); },
  createNote(payload) { return this.request("/notes", { method: "POST", body: payload }); },
  updateNote(id, payload) { return this.request(`/notes/${id}`, { method: "PUT", body: payload }); },
  deleteNote(id) { return this.request(`/notes/${id}`, { method: "DELETE" }); },

  // ---- Tags ----
  listTags() { return this.request("/tags"); },

  // ---- Reminders ----
  startReminder(noteId) { return this.request(`/notes/${noteId}/reminder/start`, { method: "POST" }); },
  completeReminder(noteId) { return this.request(`/notes/${noteId}/reminder/complete`, { method: "POST" }); },

  // ---- AI ----
  summarize(noteId) { return this.request("/ai/summarize", { method: "POST", body: { note_id: noteId } }); },
  refine(noteId) { return this.request("/ai/refine", { method: "POST", body: { note_id: noteId } }); },

  // ---- Upload ----
  uploadPdf(file) {
    const form = new FormData();
    form.append("file", file);
    return this.request("/upload/pdf", { method: "POST", body: form, isForm: true });
  },
};

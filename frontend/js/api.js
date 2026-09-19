const API_BASE = "https://notemind-1.onrender.com";

const Api = {
  token: null,

  setToken(token, remember = true) {
    this.token = token;
    const store = remember ? localStorage : sessionStorage;
    const other = remember ? sessionStorage : localStorage;
    if (token) { store.setItem("token", token); other.removeItem("token"); }
    else { localStorage.removeItem("token"); sessionStorage.removeItem("token"); }
  },

  set userEmail(value) {
    this._userEmail = value;
    localStorage.removeItem("userEmail");
    sessionStorage.removeItem("userEmail");
    if (value) {
      const store = localStorage.getItem("token") ? localStorage : sessionStorage;
      store.setItem("userEmail", value);
    }
  },
  get userEmail() { return this._userEmail; },

  loadToken() {
    const localToken = localStorage.getItem("token");
    const sessionToken = sessionStorage.getItem("token");
    this.token = localToken || sessionToken;
    const store = localToken ? localStorage : sessionStorage;
    this._userEmail = store.getItem("userEmail");
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
  dueNotes() { return this.request("/notes/due/list"); },

  // ---- Tags ----
  listTags() { return this.request("/tags"); },

  // ---- Reminders ----
  startReminder(noteId) { return this.request(`/notes/${noteId}/reminder/start`, { method: "POST" }); },
  completeReminder(noteId) { return this.request(`/notes/${noteId}/reminder/complete`, { method: "POST" }); },
  cancelReminder(noteId) { return this.request(`/notes/${noteId}/reminder`, { method: "DELETE" }); },

  // ---- AI ----
  summarize(noteId) { return this.request("/ai/summarize", { method: "POST", body: { note_id: noteId } }); },
  refine(noteId) { return this.request("/ai/refine", { method: "POST", body: { note_id: noteId } }); },
  explainSimply(noteId) { return this.request("/ai/explain", { method: "POST", body: { note_id: noteId } }); },
  explainTopic(topic) { return this.request("/ai/explain-topic", { method: "POST", body: { topic } }); },

  // ---- Upload (XHR so we can report real progress) ----
  uploadPdf(file, onProgress) {
    const form = new FormData();
    form.append("file", file);
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/upload/pdf`);
      if (this.token) xhr.setRequestHeader("Authorization", `Bearer ${this.token}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (_) { /* empty */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new Error((data && data.detail) || `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(form);
    });
  },
};
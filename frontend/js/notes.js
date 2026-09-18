const Notes = {
  currentId: null,
  autosaveTimer: null,

  init() {
    document.getElementById("new-note-btn").addEventListener("click", () => this.newNote());
    document.getElementById("save-btn").addEventListener("click", () => this.save());
    document.getElementById("delete-btn").addEventListener("click", () => this.remove());
    document.getElementById("summarize-btn").addEventListener("click", () => this.runAi("summarize"));
    document.getElementById("refine-btn").addEventListener("click", () => this.runAi("refine"));
    document.getElementById("reminder-btn").addEventListener("click", () => this.startReminder());
    document.getElementById("ai-apply-btn").addEventListener("click", () => this.applyAiResult());
    document.getElementById("ai-dismiss-btn").addEventListener("click", () => this.hideAiResult());
    document.getElementById("pdf-input").addEventListener("change", (e) => this.uploadPdf(e));

    document.getElementById("search-input").addEventListener("input", debounce(() => this.refreshList(), 300));
    document.getElementById("tag-filter").addEventListener("change", () => this.refreshList());
    document.getElementById("note-content").addEventListener("input", () => this.onEdit());
    document.getElementById("note-title").addEventListener("input", () => this.onEdit());
    document.getElementById("note-tags").addEventListener("input", () => this.onEdit());
  },

  async refreshList() {
    const q = document.getElementById("search-input").value.trim();
    const tag = document.getElementById("tag-filter").value;
    const notes = await Api.listNotes(q, tag);
    this.renderList(notes);
    await this.refreshTagFilter();
  },

  async refreshTagFilter() {
    const tags = await Api.listTags();
    const select = document.getElementById("tag-filter");
    const current = select.value;
    select.innerHTML = '<option value="">All tags</option>' +
      tags.map((t) => `<option value="${escapeHtml(t.name)}">${escapeHtml(t.name)}</option>`).join("");
    select.value = current;
  },

  renderList(notes) {
    const list = document.getElementById("notes-list");
    if (notes.length === 0) {
      list.innerHTML = '<p style="color:var(--muted);padding:0.5rem;">No notes yet.</p>';
      return;
    }
    list.innerHTML = notes.map((n) => `
      <div class="note-card ${n.id === this.currentId ? "active" : ""}" data-id="${n.id}">
        <h3>${escapeHtml(n.title)}</h3>
        <p>${new Date(n.updated_at).toLocaleDateString()} · ${escapeHtml(n.source)}</p>
        <div class="tags">${n.tags.map((t) => `<span class="tag-pill">${escapeHtml(t.name)}</span>`).join("")}</div>
      </div>
    `).join("");

    list.querySelectorAll(".note-card").forEach((card) => {
      card.addEventListener("click", () => this.open(Number(card.dataset.id)));
    });
  },

  async open(id) {
    const note = await Api.getNote(id);
    this.currentId = note.id;
    document.getElementById("editor-panel").classList.remove("hidden");
    document.getElementById("note-title").value = note.title;
    document.getElementById("note-content").value = note.content_md;
    document.getElementById("note-tags").value = note.tags.map((t) => t.name).join(", ");
    this.renderPreview(note.content_md);
    this.renderReminderInfo(note.reminder);
    this.hideAiResult();
    document.querySelectorAll(".note-card").forEach((c) => c.classList.toggle("active", Number(c.dataset.id) === id));
  },

  newNote() {
    this.currentId = null;
    document.getElementById("editor-panel").classList.remove("hidden");
    document.getElementById("note-title").value = "";
    document.getElementById("note-content").value = "";
    document.getElementById("note-tags").value = "";
    this.renderPreview("");
    this.renderReminderInfo(null);
    this.hideAiResult();
    document.querySelectorAll(".note-card").forEach((c) => c.classList.remove("active"));
    document.getElementById("note-title").focus();
  },

  onEdit() {
    this.renderPreview(document.getElementById("note-content").value);
    this.setStatus("Editing...");
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = setTimeout(() => this.save(true), 1500);
  },

  renderPreview(md) {
    // Lightweight client preview; the server-sanitized version is used for sharing/export.
    document.getElementById("note-preview").innerHTML = simpleMarkdownPreview(md);
  },

  renderReminderInfo(reminder) {
    const el = document.getElementById("reminder-info");
    const btn = document.getElementById("reminder-btn");
    if (reminder) {
      el.textContent = `Review stage ${reminder.stage + 1} · next review ${new Date(reminder.next_review_at).toLocaleDateString()}`;
      el.classList.remove("hidden");
      btn.textContent = "Mark reviewed";
      btn.onclick = () => this.completeReminder();
    } else {
      el.classList.add("hidden");
      btn.textContent = "Start review reminder";
      btn.onclick = () => this.startReminder();
    }
  },

  getFormPayload() {
    return {
      title: document.getElementById("note-title").value.trim() || "Untitled",
      content_md: document.getElementById("note-content").value,
      tags: document.getElementById("note-tags").value.split(",").map((t) => t.trim()).filter(Boolean),
    };
  },

  async save(isAutosave = false) {
    const payload = this.getFormPayload();
    let note;
    if (this.currentId) {
      note = await Api.updateNote(this.currentId, payload);
    } else {
      note = await Api.createNote(payload);
      this.currentId = note.id;
    }
    this.setStatus(isAutosave ? "Autosaved" : "Saved");
    await this.refreshList();
  },

  async remove() {
    if (!this.currentId) return;
    if (!confirm("Delete this note?")) return;
    await Api.deleteNote(this.currentId);
    this.currentId = null;
    document.getElementById("editor-panel").classList.add("hidden");
    await this.refreshList();
  },

  async startReminder() {
    if (!this.currentId) { alert("Save the note first"); return; }
    const reminder = await Api.startReminder(this.currentId);
    this.renderReminderInfo(reminder);
  },

  async completeReminder() {
    if (!this.currentId) return;
    const reminder = await Api.completeReminder(this.currentId);
    this.renderReminderInfo(reminder);
  },

  async runAi(kind) {
    if (!this.currentId) { alert("Save the note first"); return; }
    this.setStatus(kind === "summarize" ? "Summarizing..." : "Refining...");
    try {
      const { result } = kind === "summarize" ? await Api.summarize(this.currentId) : await Api.refine(this.currentId);
      document.getElementById("ai-result-label").textContent = kind === "summarize" ? "AI Summary" : "AI Grammar Refinement";
      document.getElementById("ai-result-text").textContent = result;
      document.getElementById("ai-result").classList.remove("hidden");
      this.setStatus("");
    } catch (err) {
      this.setStatus("");
      alert(err.message);
    }
  },

  applyAiResult() {
    const text = document.getElementById("ai-result-text").textContent;
    document.getElementById("note-content").value = text;
    this.onEdit();
    this.hideAiResult();
  },

  hideAiResult() {
    document.getElementById("ai-result").classList.add("hidden");
  },

  async uploadPdf(e) {
    const file = e.target.files[0];
    if (!file) return;
    this.setStatus("Uploading PDF...");
    try {
      const note = await Api.uploadPdf(file);
      await this.refreshList();
      await this.open(note.id);
      this.setStatus("");
    } catch (err) {
      this.setStatus("");
      alert(err.message);
    } finally {
      e.target.value = "";
    }
  },

  setStatus(text) {
    document.getElementById("autosave-status").textContent = text;
  },
};

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function simpleMarkdownPreview(md) {
  // Minimal client-side preview only (escaped first, so no HTML injection).
  let html = escapeHtml(md);
  html = html
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/\n/g, "<br>");
  return html;
}

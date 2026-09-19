const Pages = {
  currentEditorId: null,
  autosaveTimer: null,
  selectedTagFilter: "",
  reminderTab: "all",

  // ---------------- NOTES LIST ----------------
  async notesList(container, opts = {}) {
    if (opts.presetTag !== undefined) this.selectedTagFilter = opts.presetTag;

    container.innerHTML = `
      <div class="list-toolbar">
        <div class="input-icon search-box"><span>🔍</span><input type="search" id="notes-search" placeholder="Search notes..."></div>
        <select id="notes-sort">
          <option value="updated_desc">Sort: Latest</option>
          <option value="updated_asc">Sort: Oldest</option>
          <option value="title_asc">Sort: Title A-Z</option>
        </select>
        ${this.selectedTagFilter ? `<span class="active-filter">#${escapeHtml(this.selectedTagFilter)} <button id="clear-tag-filter">×</button></span>` : ""}
      </div>
      <div id="notes-list-body"></div>
    `;

    document.getElementById("notes-search").addEventListener("input", debounce(() => this.loadNotesList(), 300));
    document.getElementById("notes-sort").addEventListener("change", () => this.loadNotesList());
    const clearBtn = document.getElementById("clear-tag-filter");
    if (clearBtn) clearBtn.addEventListener("click", () => { this.selectedTagFilter = ""; this.notesList(container); });

    await this.loadNotesList();

    if (opts.newNote) this.noteEditor(null);
    if (opts.openId) this.noteEditor(opts.openId);
  },

  async loadNotesList() {
    const body = document.getElementById("notes-list-body");
    if (!body) return;
    body.innerHTML = `<div class="state-loading">${Spinner.html()}<p>Fetching your notes...</p></div>`;

    const q = document.getElementById("notes-search").value.trim();
    const sort = document.getElementById("notes-sort").value;

    let notes;
    try {
      notes = await Api.listNotes(q, this.selectedTagFilter);
    } catch (err) {
      body.innerHTML = ErrorState.html(err.message, "Pages.loadNotesList()");
      return;
    }

    notes.sort((a, b) => {
      if (sort === "updated_asc") return new Date(a.updated_at) - new Date(b.updated_at);
      if (sort === "title_asc") return a.title.localeCompare(b.title);
      return new Date(b.updated_at) - new Date(a.updated_at);
    });

    if (notes.length === 0) {
      body.innerHTML = EmptyState.html("No notes yet", "Start by creating your first note.", "+ Create Note", "Pages.noteEditor(null)");
      return;
    }

    body.innerHTML = `<div class="notes-table">${notes.map((n) => `
      <div class="note-line" data-id="${n.id}">
        <div class="note-line-main">
          <span class="note-line-title">${escapeHtml(n.title)}</span>
          <span class="tags">${n.tags.map((t) => `<span class="tag-pill" data-filter-tag="${escapeAttr(t.name)}">#${escapeHtml(t.name)}</span>`).join("")}</span>
        </div>
        <span class="note-line-date">${timeAgo(n.updated_at)}</span>
        <div class="row-menu">
          <button class="row-menu-btn" data-menu="${n.id}">⋮</button>
          <div class="row-menu-dropdown hidden" id="menu-${n.id}">
            <button data-edit="${n.id}">Edit</button>
            <button data-delete="${n.id}" class="danger">Delete</button>
          </div>
        </div>
      </div>`).join("")}</div>`;

    body.querySelectorAll(".note-line").forEach((row) => {
      row.addEventListener("click", (e) => {
        if (e.target.closest(".row-menu") || e.target.closest(".tag-pill")) return;
        this.noteEditor(Number(row.dataset.id));
      });
    });
    body.querySelectorAll("[data-filter-tag]").forEach((pill) => {
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        App.navigate("notes", { presetTag: pill.dataset.filterTag });
      });
    });
    body.querySelectorAll(".row-menu-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const dropdown = document.getElementById(`menu-${btn.dataset.menu}`);
        document.querySelectorAll(".row-menu-dropdown").forEach((d) => { if (d !== dropdown) d.classList.add("hidden"); });
        dropdown.classList.toggle("hidden");
      });
    });
    body.querySelectorAll("[data-edit]").forEach((btn) => btn.addEventListener("click", (e) => { e.stopPropagation(); this.noteEditor(Number(btn.dataset.edit)); }));
    body.querySelectorAll("[data-delete]").forEach((btn) => btn.addEventListener("click", (e) => { e.stopPropagation(); this.deleteNoteFromList(Number(btn.dataset.delete)); }));

    document.addEventListener("click", () => document.querySelectorAll(".row-menu-dropdown").forEach((d) => d.classList.add("hidden")), { once: true });
  },

  async deleteNoteFromList(id) {
    if (!confirm("Delete this note? This cannot be undone.")) return;
    try {
      await Api.deleteNote(id);
      Toast.success("Note deleted.");
      this.loadNotesList();
    } catch (err) {
      Toast.error(err.message);
    }
  },

  // ---------------- NOTE EDITOR ----------------
  async noteEditor(id) {
    this.currentEditorId = id;
    this.aiMode = null;
    this.aiResultText = null;
    const container = document.getElementById("content-area");
    let note = { title: "", content_md: "", tags: [], reminder: null };

    if (id) {
      container.innerHTML = `<div class="state-loading">${Spinner.html()}<p>Loading note...</p></div>`;
      try { note = await Api.getNote(id); }
      catch (err) { container.innerHTML = ErrorState.html(err.message, `Pages.noteEditor(${id})`); return; }
    }

    container.innerHTML = `
      <div class="editor-layout">
        <div class="editor-main card">
          <div class="editor-top">
            <input type="text" id="note-title" placeholder="Untitled note" value="${escapeAttr(note.title)}">
            <span id="autosave-status" class="autosave-status"></span>
            <button class="btn-outline" id="editor-back">← Back</button>
            <button class="btn-primary" id="editor-save">${id ? "Save Changes" : "Create Note"}</button>
            ${id ? '<button class="btn-danger-outline" id="editor-delete">Delete Note</button>' : ""}
          </div>

          <label class="field-label">Your note</label>
          <div class="rt-toolbar">
            <button type="button" data-cmd="strong" title="Bold"><b>B</b></button>
            <button type="button" data-cmd="em" title="Italic"><i>I</i></button>
            <button type="button" data-cmd="u" title="Underline"><u>U</u></button>
            <span class="rt-sep"></span>
            <button type="button" class="hl-swatch hl-yellow" data-hl="hl-yellow" title="Highlight yellow"></button>
            <button type="button" class="hl-swatch hl-green" data-hl="hl-green" title="Highlight green"></button>
            <button type="button" class="hl-swatch hl-pink" data-hl="hl-pink" title="Highlight pink"></button>
            <button type="button" class="hl-swatch hl-blue" data-hl="hl-blue" title="Highlight blue"></button>
            <button type="button" class="hl-swatch hl-erase" id="rt-eraser-btn" title="Remove highlight"></button>
          </div>
          <div id="note-content" class="rt-editor" contenteditable="true" data-placeholder="Write your note..."></div>
          <span id="editor-wordcount" class="hint"></span>

          <label class="field-label">Tags</label>
          <div id="tag-pills" class="tag-pills"></div>
          <input type="text" id="tag-input" placeholder="Type a tag and press Enter">
        </div>

        <div class="editor-side">
          <div class="card ai-pane">
            <div class="ai-pane-header">
              <h3 id="ai-pane-title">Preview</h3>
              <div class="ai-buttons">
                <button class="btn-outline" id="ai-summarize-btn" ${id ? "" : "disabled"}>Summarize</button>
                <button class="btn-outline" id="ai-refine-btn" ${id ? "" : "disabled"}>Fix Grammar</button>
                <button class="btn-outline" id="ai-explain-btn" ${id ? "" : "disabled"}>Explain Simply</button>
              </div>
            </div>
            ${id ? "" : '<p class="hint">Save the note first to use AI tools.</p>'}
            <div id="ai-pane-body" class="preview"></div>
            <div id="ai-pane-actions" class="ai-result-actions hidden">
              <button class="btn-primary" id="ai-apply-btn"></button>
              <button class="btn-outline" id="ai-dismiss-btn"></button>
            </div>
          </div>

          <div class="card">
            <h3>Upload PDF</h3>
            <label class="dropzone" id="editor-dropzone">
              <input type="file" id="editor-pdf-input" accept="application/pdf" hidden>
              <span>⬆️ Drag &amp; drop a PDF or click to upload</span>
              <span class="dropzone-hint">Max size: 5MB · PDF only</span>
            </label>
            <div id="editor-upload-progress" class="upload-progress hidden">
              <div class="upload-file-row"><span id="upload-filename"></span><span id="upload-pct">0%</span></div>
              <div class="progress-track"><div id="upload-bar" class="progress-bar"></div></div>
            </div>
          </div>

          <div class="card">
            <h3>Reminder</h3>
            <div id="reminder-body"></div>
          </div>
        </div>
      </div>
    `;

    document.getElementById("note-content").innerHTML = markdownToEditableHtml(note.content_md);
    this.renderTagPills(note.tags.map((t) => t.name));
    this.renderReminder(note.reminder, id);
    this.wireEditorEvents(id);
    this.renderAiPane();
    this.updateWordCount();
  },

  renderTagPills(names) {
    const box = document.getElementById("tag-pills");
    box.innerHTML = names.map((n) => `<span class="tag-pill editable" data-tag="${escapeAttr(n)}">#${escapeHtml(n)}<button type="button">×</button></span>`).join("");
    box.querySelectorAll(".tag-pill button").forEach((btn) => {
      btn.addEventListener("click", () => { btn.parentElement.remove(); this.scheduleAutosave(); });
    });
  },

  getCurrentTags() {
    return [...document.querySelectorAll("#tag-pills .tag-pill")].map((el) => el.dataset.tag);
  },

  renderReminder(reminder, noteId) {
    const box = document.getElementById("reminder-body");
    if (!noteId) { box.innerHTML = '<p class="hint">Save the note first to set a reminder.</p>'; return; }
    if (reminder) {
      box.innerHTML = `
        <p class="reminder-line">Stage ${reminder.stage + 1} of 4 · Next review <strong>${new Date(reminder.next_review_at).toLocaleDateString()}</strong></p>
        <button class="btn-block btn-primary" id="reminder-complete-btn">Mark as Reviewed</button>
        <button class="btn-block btn-danger-outline" id="reminder-cancel-btn">Cancel Reminder</button>`;
      document.getElementById("reminder-complete-btn").addEventListener("click", async () => {
        const updated = await Api.completeReminder(noteId);
        this.renderReminder(updated, noteId);
        Toast.success("Marked as reviewed — next review scheduled.");
      });
      document.getElementById("reminder-cancel-btn").addEventListener("click", async () => {
        await Api.cancelReminder(noteId);
        this.renderReminder(null, noteId);
        Toast.success("Reminder cancelled.");
      });
    } else {
      box.innerHTML = '<button class="btn-block btn-outline" id="reminder-start-btn">Start Review Reminder</button>';
      document.getElementById("reminder-start-btn").addEventListener("click", async () => {
        const created = await Api.startReminder(noteId);
        this.renderReminder(created, noteId);
        Toast.success("Reminder scheduled.");
      });
    }
  },

  wireEditorEvents(id) {
    document.getElementById("editor-back").addEventListener("click", () => App.navigate("notes"));
    document.getElementById("editor-save").addEventListener("click", () => this.saveNote(false));
    if (id) document.getElementById("editor-delete").addEventListener("click", () => this.deleteNote(id));

    document.getElementById("note-title").addEventListener("input", () => this.scheduleAutosave());

    document.querySelectorAll(".rt-toolbar [data-cmd]").forEach((btn) => {
      btn.addEventListener("mousedown", (e) => e.preventDefault());
      btn.addEventListener("click", () => wrapSelection(btn.dataset.cmd));
    });
    document.querySelectorAll(".hl-swatch[data-hl]").forEach((btn) => {
      btn.addEventListener("mousedown", (e) => e.preventDefault());
      btn.addEventListener("click", () => wrapSelection("mark", btn.dataset.hl));
    });
    document.getElementById("rt-eraser-btn").addEventListener("mousedown", (e) => e.preventDefault());
    document.getElementById("rt-eraser-btn").addEventListener("click", () => {
      eraseHighlight();
      this.renderAiPane();
      this.scheduleAutosave();
    });

    const editor = document.getElementById("note-content");
    editor.addEventListener("input", () => { this.renderAiPane(); this.scheduleAutosave(); this.updateWordCount(); });
    editor.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); document.execCommand("insertLineBreak"); }
    });
    editor.addEventListener("paste", (e) => {
      e.preventDefault();
      const text = (e.clipboardData || window.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text);
    });

    document.addEventListener("keydown", (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (e.key === "s") { e.preventDefault(); this.saveNote(false); }
      if (e.key === "b") { e.preventDefault(); wrapSelection("strong"); }
      if (e.key === "i") { e.preventDefault(); wrapSelection("em"); }
      if (e.key === "u") { e.preventDefault(); wrapSelection("u"); }
    });

    const tagInput = document.getElementById("tag-input");
    tagInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && tagInput.value.trim()) {
        e.preventDefault();
        const name = tagInput.value.trim().toLowerCase();
        if (!this.getCurrentTags().includes(name)) { this.renderTagPills([...this.getCurrentTags(), name]); this.scheduleAutosave(); }
        tagInput.value = "";
      }
    });

    document.getElementById("ai-summarize-btn").addEventListener("click", () => this.runAi("summarize"));
    document.getElementById("ai-refine-btn").addEventListener("click", () => this.runAi("refine"));
    document.getElementById("ai-explain-btn").addEventListener("click", () => this.runAi("explain"));

    const dropzone = document.getElementById("editor-dropzone");
    const pdfInput = document.getElementById("editor-pdf-input");
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
    dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("drag"); if (e.dataTransfer.files[0]) this.uploadPdf(e.dataTransfer.files[0]); });
    pdfInput.addEventListener("change", (e) => { if (e.target.files[0]) this.uploadPdf(e.target.files[0]); });
  },

  renderAiPane() {
    const title = document.getElementById("ai-pane-title");
    const body = document.getElementById("ai-pane-body");
    const actions = document.getElementById("ai-pane-actions");
    const titles = { summarize: "AI Summary", refine: "AI Grammar Fix", explain: "Simple Explanation" };
    const applyLabels = { summarize: "Use This Summary", refine: "Apply Grammar Fix", explain: "Add Explanation" };
    const dismissLabels = { summarize: "Discard Summary", refine: "Discard Corrections", explain: "Discard Explanation" };

    if (this.aiMode && this.aiResultText !== null) {
      title.textContent = titles[this.aiMode];
      body.innerHTML = simpleMarkdownPreview(this.aiResultText);

      const applyBtn = document.getElementById("ai-apply-btn");
      const dismissBtn = document.getElementById("ai-dismiss-btn");
      applyBtn.textContent = applyLabels[this.aiMode];
      dismissBtn.textContent = dismissLabels[this.aiMode];
      applyBtn.onclick = () => this.acceptAiResult();
      dismissBtn.onclick = () => this.dismissAiResult();
      actions.classList.remove("hidden");
    } else {
      title.textContent = "Preview";
      const content = serializeEditableToMarkdown(document.getElementById("note-content"));
      body.innerHTML = content.trim() ? simpleMarkdownPreview(content) : '<p class="hint">Nothing to preview yet.</p>';
      actions.classList.add("hidden");
    }
  },

  updateWordCount() {
    const text = document.getElementById("note-content").innerText.trim();
    const words = text ? text.split(/\s+/).length : 0;
    document.getElementById("editor-wordcount").textContent = `${words} words · ${text.length} characters`;
  },

  scheduleAutosave() {
    if (!this.currentEditorId) return;
    document.getElementById("autosave-status").textContent = "Editing...";
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = setTimeout(() => this.saveNote(true), 1200);
  },

  getFormPayload() {
    return {
      title: document.getElementById("note-title").value.trim() || "Untitled",
      content_md: serializeEditableToMarkdown(document.getElementById("note-content")),
      tags: this.getCurrentTags(),
    };
  },

  async saveNote(isAutosave) {
    const payload = this.getFormPayload();
    const status = document.getElementById("autosave-status");
    try {
      if (this.currentEditorId) {
        await Api.updateNote(this.currentEditorId, payload);
      } else {
        const created = await Api.createNote(payload);
        this.currentEditorId = created.id;
        this.noteEditor(created.id);
        Toast.success("Note created.");
        return;
      }
      status.textContent = isAutosave ? "Autosaved ✓" : "Saved ✓";
      if (!isAutosave) Toast.success("Note saved.");
    } catch (err) {
      status.textContent = "";
      Toast.error(err.message);
    }
  },

  async deleteNote(id) {
    if (!confirm("Delete this note? This cannot be undone.")) return;
    try { await Api.deleteNote(id); Toast.success("Note deleted."); App.navigate("notes"); }
    catch (err) { Toast.error(err.message); }
  },

  async runAi(kind) {
    if (!this.currentEditorId) return;
    const labels = { summarize: "Summarizing...", refine: "Refining grammar...", explain: "Explaining..." };
    Toast.info(labels[kind]);
    try {
      const { result } = kind === "summarize" ? await Api.summarize(this.currentEditorId) :
        kind === "refine" ? await Api.refine(this.currentEditorId) :
        await Api.explainSimply(this.currentEditorId);
      this.aiMode = kind;
      this.aiResultText = result;
      this.renderAiPane();
    } catch (err) {
      Toast.error(err.message);
    }
  },

  async acceptAiResult() {
    const editor = document.getElementById("note-content");
    const mode = this.aiMode;

    if (mode === "refine") {
      editor.innerHTML = markdownToEditableHtml(this.aiResultText);
    } else {
      const heading = mode === "summarize" ? "Summary" : "Simple Explanation";
      const sectionHtml = markdownToEditableHtml(this.aiResultText);
      editor.innerHTML += `<div class="md-line"><strong>${heading}</strong></div>${sectionHtml}`;
    }

    this.aiMode = null;
    this.aiResultText = null;
    await this.saveNote(false);
    const messages = { summarize: "Summary added to your notes.", refine: "Grammar corrections applied.", explain: "Explanation added to your notes." };
    Toast.success(messages[mode]);
    this.updateWordCount();
    this.renderAiPane();
  },

  dismissAiResult() {
    this.aiMode = null;
    this.aiResultText = null;
    this.renderAiPane();
  },

  async uploadPdf(file) {
    const progressBox = document.getElementById("editor-upload-progress");
    progressBox.classList.remove("hidden");
    document.getElementById("upload-filename").textContent = file.name;
    const bar = document.getElementById("upload-bar");
    const pct = document.getElementById("upload-pct");
    bar.style.width = "0%";
    pct.textContent = "Uploading...";
    try {
      const note = await Api.uploadPdf(file, (percent) => {
        bar.style.width = percent + "%";
        pct.textContent = percent < 100 ? `${percent}%` : "Extracting text...";
      });
      bar.style.width = "100%";
      pct.textContent = "100%";
      Toast.success("PDF uploaded — note created.");
      setTimeout(() => this.noteEditor(note.id), 400);
    } catch (err) {
      progressBox.classList.add("hidden");
      Toast.error(err.message);
    }
  },

  // ---------------- REMINDERS ----------------
  async reminders(container) {
    container.innerHTML = `<div class="state-loading">${Spinner.html()}<p>Loading reminders...</p></div>`;
    let notes;
    try { notes = await Api.listNotes(); }
    catch (err) { container.innerHTML = ErrorState.html(err.message, "App.navigate('reminders')"); return; }

    const withReminders = notes.filter((n) => n.reminder);
    this.renderReminderPage(container, withReminders);
  },

  renderReminderPage(container, withReminders) {
    const today = new Date().toDateString();
    const filtered = withReminders.filter((n) => {
      const due = new Date(n.reminder.next_review_at);
      if (this.reminderTab === "today") return due <= new Date() || due.toDateString() === today;
      if (this.reminderTab === "upcoming") return due > new Date() && due.toDateString() !== today;
      if (this.reminderTab === "completed") return n.reminder.completed;
      return true;
    });

    container.innerHTML = `
      <div class="reminders-toolbar">
        <div class="tabs-pill">
          ${["all", "today", "upcoming", "completed"].map((t) => `<button class="pill-tab ${this.reminderTab === t ? "active" : ""}" data-rtab="${t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}
        </div>
        <button class="btn-outline" id="notify-toggle-btn">${Notify.isEnabled() ? "🔔 Notifications On" : "🔕 Enable Notifications"}</button>
        <button class="btn-primary" id="add-reminder-btn">+ Add Reminder</button>
      </div>
      <div id="add-reminder-box" class="card hidden"></div>
      <div id="reminder-list">${filtered.length ? `<div class="notes-table">${filtered.map((n) => this.reminderRow(n)).join("")}</div>` : EmptyState.html("Nothing here", "No reminders in this view yet.", null, "")}</div>
    `;

    container.querySelectorAll("[data-rtab]").forEach((btn) => {
      btn.addEventListener("click", () => { this.reminderTab = btn.dataset.rtab; this.renderReminderPage(container, withReminders); });
    });

    document.getElementById("add-reminder-btn").addEventListener("click", () => this.showAddReminderBox(container));
    document.getElementById("notify-toggle-btn").addEventListener("click", async () => {
      if (Notify.isEnabled()) {
        Notify.disable();
        Toast.info("Notifications turned off.");
      } else {
        const ok = await Notify.enable();
        if (ok) { Notify.start(); Toast.success("Notifications enabled."); }
      }
      this.renderReminderPage(container, withReminders);
    });

    container.querySelectorAll("[data-check]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        await Api.completeReminder(Number(cb.dataset.check));
        Toast.success("Marked as reviewed.");
        this.reminders(container);
      });
    });
    container.querySelectorAll(".row-menu-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const dropdown = document.getElementById(`rmenu-${btn.dataset.menu}`);
        document.querySelectorAll(".row-menu-dropdown").forEach((d) => { if (d !== dropdown) d.classList.add("hidden"); });
        dropdown.classList.toggle("hidden");
      });
    });
    container.querySelectorAll("[data-cancel]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await Api.cancelReminder(Number(btn.dataset.cancel));
        Toast.success("Reminder cancelled.");
        this.reminders(container);
      });
    });
    container.querySelectorAll("[data-open]").forEach((btn) => {
      btn.addEventListener("click", (e) => { e.stopPropagation(); App.navigate("notes", { openId: Number(btn.dataset.open) }); });
    });
    document.addEventListener("click", () => document.querySelectorAll(".row-menu-dropdown").forEach((d) => d.classList.add("hidden")), { once: true });
  },

  reminderRow(n) {
    return `
      <div class="note-line reminder-row">
        <label class="checkbox reminder-check">
          <input type="checkbox" data-check="${n.id}" ${n.reminder.completed ? "checked disabled" : ""}>
          <span class="note-line-title">${escapeHtml(n.title)}</span>
        </label>
        <span class="note-line-date">${new Date(n.reminder.next_review_at).toLocaleDateString()}</span>
        <div class="row-menu">
          <button class="row-menu-btn" data-menu="${n.id}">⋮</button>
          <div class="row-menu-dropdown hidden" id="rmenu-${n.id}">
            <button data-open="${n.id}">Open note</button>
            <button data-cancel="${n.id}" class="danger">Cancel reminder</button>
          </div>
        </div>
      </div>`;
  },

  async showAddReminderBox(container) {
    const box = document.getElementById("add-reminder-box");
    box.classList.remove("hidden");
    box.innerHTML = `<div class="state-loading">${Spinner.html()}</div>`;
    const notes = await Api.listNotes();
    const candidates = notes.filter((n) => !n.reminder);
    if (candidates.length === 0) {
      box.innerHTML = `<p class="hint">Every note already has an active reminder.</p>`;
      return;
    }
    box.innerHTML = `
      <label class="field-label">Pick a note to schedule a review for</label>
      <select id="add-reminder-select">${candidates.map((n) => `<option value="${n.id}">${escapeHtml(n.title)}</option>`).join("")}</select>
      <button class="btn-primary" id="add-reminder-confirm">Set Reminder</button>`;
    document.getElementById("add-reminder-confirm").addEventListener("click", async () => {
      const noteId = Number(document.getElementById("add-reminder-select").value);
      await Api.startReminder(noteId);
      Toast.success("Reminder scheduled.");
      this.reminders(container);
    });
  },

  // ---------------- PROFILE ----------------
  profile(container) {
    const email = Api.userEmail || "unknown@example.com";
    const initial = email[0].toUpperCase();
    container.innerHTML = `
      <div class="profile-layout">
        <div class="card profile-card">
          <div class="avatar-lg">${initial}</div>
          <h3>${escapeHtml(email.split("@")[0])}</h3>
          <p class="hint">${escapeHtml(email)}</p>
          <button class="btn-outline btn-block" id="profile-settings-btn">Settings</button>
          <button class="btn-danger-outline btn-block" id="profile-logout-btn">Logout</button>
        </div>
        <div class="card">
          <h3>Settings</h3>
          ${this.settingsRows()}
        </div>
      </div>`;
    document.getElementById("profile-settings-btn").addEventListener("click", () => App.navigate("settings"));
    document.getElementById("profile-logout-btn").addEventListener("click", () => Auth.logout());
    this.wireSettingsRows(container);
  },

  // ---------------- EXPLORE (topic search) ----------------
explore(container) {
  container.innerHTML = `
    <div class="card">
      <h3>Ask about any topic</h3>
      <p class="hint">Type a topic or question and get a simple, plain-language explanation.</p>
      <div class="input-icon search-box">
        <span>🔎</span>
        <input type="text" id="explore-input" placeholder="e.g. What is machine learning?">
      </div>
      <button class="btn-primary" id="explore-btn">Explain</button>
      <div id="explore-result" class="preview" style="margin-top:16px;"></div>
    </div>
  `;

  const input = document.getElementById("explore-input");
  const btn = document.getElementById("explore-btn");
  const result = document.getElementById("explore-result");

  const run = async () => {
    const topic = input.value.trim();
    if (!topic) return;
    btn.disabled = true;
    result.innerHTML = `<div class="state-loading">${Spinner.html()}<p>Explaining...</p></div>`;
    try {
      const { result: text } = await Api.explainTopic(topic);
      result.innerHTML = simpleMarkdownPreview(text);
    } catch (err) {
      result.innerHTML = ErrorState.html(err.message, "");
    } finally {
      btn.disabled = false;
    }
  };

  btn.addEventListener("click", run);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
},

  // ---------------- SETTINGS ----------------
  settings(container) {
    container.innerHTML = `<div class="card settings-card"><h3>Settings</h3>${this.settingsRows()}</div>`;
    this.wireSettingsRows(container);
  },

  wireSettingsRows(container) {
    container.querySelectorAll("[data-setting]").forEach((btn) => {
      if (btn.dataset.setting === "appearance") {
        btn.addEventListener("click", () => {
          const isDark = document.documentElement.dataset.theme === "dark";
          document.documentElement.dataset.theme = isDark ? "light" : "dark";
          localStorage.setItem("theme", isDark ? "light" : "dark");
        });
      } else {
        btn.addEventListener("click", () => Toast.info(`${btn.textContent.trim()} isn't part of this demo build.`));
      }
    });
  },
};

// ---------------- shared helpers ----------------
function timeAgo(dateStr) {
  const diff = (Date.now() - new Date(dateStr).getTime()) / 1000;
  if (diff < 60) return "Just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(dateStr).toLocaleDateString();
}

function debounce(fn, delay) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function escapeAttr(str) { return escapeHtml(str).replace(/"/g, "&quot;"); }

const HL_CLASSES = ["hl-yellow", "hl-green", "hl-pink", "hl-blue"];

function unwrapMark(mark) {
  const parent = mark.parentNode;
  while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
  parent.removeChild(mark);
}

// Removes highlighting from the current selection. If the cursor is just
// sitting inside a highlight (nothing selected), it clears that whole mark.
// If text is selected, it clears every highlight the selection touches.
function eraseHighlight() {
  const sel = window.getSelection();
  if (!sel.rangeCount) return;
  const range = sel.getRangeAt(0);

  if (range.collapsed) {
    const node = sel.anchorNode;
    const mark = node && (node.nodeType === Node.TEXT_NODE ? node.parentElement : node).closest?.("mark");
    if (mark) unwrapMark(mark);
    return;
  }

  const container = range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
    ? range.commonAncestorContainer
    : range.commonAncestorContainer.parentElement;

  container.querySelectorAll("mark").forEach((mark) => {
    if (range.intersectsNode(mark)) unwrapMark(mark);
  });

  const startMark = range.startContainer.nodeType === Node.TEXT_NODE
    ? range.startContainer.parentElement.closest("mark")
    : null;
  if (startMark) unwrapMark(startMark);

  sel.removeAllRanges();
}

function wrapSelection(tagName, className) {
  const sel = window.getSelection();
  if (!sel.rangeCount || sel.isCollapsed) return;
  const range = sel.getRangeAt(0);
  const el = document.createElement(tagName);
  if (className) el.className = className;
  try {
    range.surroundContents(el);
  } catch (_) {
    const frag = range.extractContents();
    el.appendChild(frag);
    range.insertNode(el);
  }
  sel.removeAllRanges();
  document.getElementById("note-content")?.dispatchEvent(new Event("input", { bubbles: true }));
}

function serializeEditableToMarkdown(root) {
  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE) return "";
    const inner = [...node.childNodes].map(walk).join("");
    switch (node.tagName) {
      case "STRONG": case "B": return `**${inner}**`;
      case "EM": case "I": return `*${inner}*`;
      case "U": return `<u>${inner}</u>`;
      case "MARK": {
        const cls = HL_CLASSES.includes(node.className) ? node.className : "hl-yellow";
        return `<mark class="${cls}">${inner}</mark>`;
      }
      case "BR": return "\n";
      default: return inner;
    }
  }
  return walk(root).replace(/\n{3,}/g, "\n\n").trim();
}

function markdownToEditableHtml(md) {
  let html = escapeHtml(normalizeListBreaks(md));
  html = html
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/&lt;u&gt;(.*?)&lt;\/u&gt;/g, "<u>$1</u>")
    .replace(/&lt;mark class="(hl-[a-z]+)"&gt;(.*?)&lt;\/mark&gt;/g, '<mark class="$1">$2</mark>')
    .replace(/^(\d+\.\s.*)\n?/gm, '<div class="md-line">$1</div>')
    .replace(/^(-\s.*)\n?/gm, '<div class="md-line md-bullet">$1</div>')
    .replace(/\*{1,}/g, "")
    .replace(/\n/g, "<br>");
  return html;
}

function simpleMarkdownPreview(md) {
  let html = escapeHtml(normalizeListBreaks(md));
  html = html
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/&lt;u&gt;(.*?)&lt;\/u&gt;/g, "<u>$1</u>")
    .replace(/&lt;mark class="(hl-[a-z]+)"&gt;(.*?)&lt;\/mark&gt;/g, '<mark class="$1">$2</mark>')
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/^> (.*)$/gm, "<blockquote>$1</blockquote>")
    .replace(/\n/g, "<br>");
  return html;
}

function normalizeListBreaks(text) {
  return text
    .replace(/(\w)(\*\*)/g, "$1 $2")
    .replace(/(\S)(\s*)(\d+\.\s)/g, "$1\n$3")
    .replace(/(\S)(\s*)(-\s)/g, "$1\n$3")
    .replace(/\n{2,}/g, "\n");
}
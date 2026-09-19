const Dashboard = {
  async render(container) {
    container.innerHTML = `<div class="state-loading">${Spinner.html()}<p>Loading dashboard...</p></div>`;

    const name = localStorage.getItem("displayName");
    document.getElementById("page-title").textContent = `Welcome back, ${name}! 👋`;
    document.getElementById("page-subtitle").textContent = "Here's a quick overview of your notes.";

    let notes, due, tags;
    try {
      [notes, due, tags] = await Promise.all([Api.listNotes(), Api.dueNotes(), Api.listTags()]);
    } catch (err) {
      container.innerHTML = ErrorState.html(err.message, "Dashboard.retry()");
      return;
    }

    const activeReminders = notes.filter((n) => n.reminder && !n.reminder.completed).length;
    const recent = [...notes].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)).slice(0, 5);

    container.innerHTML = `
      <div class="stat-grid stat-grid-3">
        ${this.statCard("📄", notes.length, "Total Notes", "blue")}
        ${this.statCard("🔔", activeReminders, "Upcoming Reminders", "green")}
        ${this.statCard("🏷️", tags.length, "Tags Used", "purple")}
      </div>

      <div class="card recent-notes">
        <div class="card-header">
          <h3>Recent Notes</h3>
          <a href="#" id="view-all-link">View all</a>
        </div>
        ${recent.length ? recent.map((n) => this.noteRow(n)).join("") : EmptyState.html("No notes yet", "Start by creating your first note.", "+ Create Note", "Pages.noteEditor(null)")}
      </div>
    `;

    container.querySelectorAll(".note-row").forEach((row) => {
      row.addEventListener("click", () => App.navigate("notes", { openId: Number(row.dataset.id) }));
    });
    document.getElementById("view-all-link").addEventListener("click", (e) => { e.preventDefault(); App.navigate("notes"); });
  },

  statCard(icon, value, label, color) {
    return `<div class="stat-card stat-${color}"><span class="stat-icon">${icon}</span><div><div class="stat-value">${value}</div><div class="stat-label">${label}</div></div></div>`;
  },

  noteRow(n) {
    const snippet = escapeHtml((n.content_md || "").slice(0, 70));
    return `
      <div class="note-row" data-id="${n.id}">
        <span class="note-row-icon">📄</span>
        <div class="note-row-body">
          <div class="note-row-title">${escapeHtml(n.title)}</div>
          <div class="note-row-snippet">${snippet}${n.content_md.length > 70 ? "..." : ""}</div>
          <div class="tags">${n.tags.map((t) => `<span class="tag-pill">#${escapeHtml(t.name)}</span>`).join("")}</div>
        </div>
        <div class="note-row-date">${timeAgo(n.updated_at)}</div>
      </div>`;
  },

  retry() { App.navigate("dashboard"); },
};

const Spinner = { html: () => '<div class="spinner"></div>' };

const EmptyState = {
  html(title, subtitle, btnLabel, onClick) {
    return `<div class="state-empty">
      <div class="state-icon">📄</div>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(subtitle)}</p>
      ${btnLabel ? `<button class="btn-primary" onclick="${onClick}">${escapeHtml(btnLabel)}</button>` : ""}
    </div>`;
  },
};

const ErrorState = {
  html(message, retryCall) {
    return `<div class="state-error">
      <div class="state-icon">⚠️</div>
      <h3>Something went wrong</h3>
      <p>${escapeHtml(message || "Please try again.")}</p>
      <button class="btn-primary" onclick="${retryCall}">Retry</button>
    </div>`;
  },
};
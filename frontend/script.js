/* ═══════════════════════════════════════════════════
   NoteMind — script.js
   Full app logic: notes, AI simulation, PDF, markdown,
   reminders, search, keyboard shortcuts, dark mode
   ═══════════════════════════════════════════════════ */

'use strict';

// ── STATE ──────────────────────────────────────────
const state = {
  notes: [],
  reminders: [],
  activeNoteId: null,
  activeTag: 'all',
  markdownMode: false,
  autosaveTimer: null,
  improvedText: '',
  selectedReminderDays: null,
};

// ── UTILS ──────────────────────────────────────────
const uid = () => Math.random().toString(36).slice(2, 10);

const fmt = (d) => {
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const fmtDate = (dateStr) =>
  new Date(dateStr).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });

const persist = () => {
  localStorage.setItem('notemind_notes', JSON.stringify(state.notes));
  localStorage.setItem('notemind_reminders', JSON.stringify(state.reminders));
};

const load = () => {
  try {
    state.notes = JSON.parse(localStorage.getItem('notemind_notes') || '[]');
    state.reminders = JSON.parse(localStorage.getItem('notemind_reminders') || '[]');
  } catch { state.notes = []; state.reminders = []; }
};

// ── TOAST ──────────────────────────────────────────
const toast = (() => {
  const el = document.getElementById('toast');
  let timer;
  return (msg, type = 'info') => {
    el.textContent = msg;
    el.className = `toast show ${type}`;
    clearTimeout(timer);
    timer = setTimeout(() => el.classList.remove('show'), 3000);
  };
})();

// ── MARKDOWN PARSER (lightweight) ──────────────────
const parseMarkdown = (text) => {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // bold/italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    // inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // blockquote
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // unordered list
    .replace(/^\* (.+)$/gm, '<li>$1</li>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    // ordered list
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // hr
    .replace(/^---$/gm, '<hr>')
    // paragraphs
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
};

// ── AI SIMULATION ──────────────────────────────────
// Simulates AI responses locally (no backend needed)

const aiSummarize = async (text) => {
  try {
    const res = await fetch("http://127.0.0.1:9000/summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        content: text
      })
    });

    const data = await res.json();
    return data.summary || "No summary generated.";

  } catch (err) {
    console.error(err);
    return "⚠️ AI server not responding.";
  }
};

const aiImprove = async (text) => {
  try {
    const res = await fetch("http://127.0.0.1:9000/improve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        content: text
      })
    });

    const data = await res.json();
    return data.improved || "No improvement generated.";

  } catch (err) {
    console.error(err);
    return "⚠️ AI server not responding.";
  }
};

const aiExtractPDF = (filename) => new Promise((resolve) => {
  // Simulates text extraction from a PDF (since we have no backend)
  setTimeout(() => {
    const samples = [
      `# ${filename.replace('.pdf', '')}\n\nThis document has been processed by NoteMind's PDF extraction engine.\n\nThe text content from your PDF would appear here after backend processing. For now, this simulated extraction demonstrates how the feature integrates with the editor.\n\nYou can edit this text, add notes, and use AI features like summarization or improvement on this content.\n\n## Key Points\n- PDF content is extracted page by page\n- Images are described when detected\n- Tables are converted to text format\n- Formatting is preserved where possible`,
      `# Extracted: ${filename.replace('.pdf', '')}\n\nDocument successfully parsed.\n\n## Introduction\nThe contents of your uploaded PDF appear in this editor. This allows you to annotate, summarize, and study the material directly within NoteMind.\n\n## Main Content\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n\nUt enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\n\n## Conclusion\nPDF extraction complete. Use AI Summary to quickly digest this content.`,
    ];
    resolve(samples[Math.floor(Math.random() * samples.length)]);
  }, 1200 + Math.random() * 800);
});

// ── NOTES CRUD ─────────────────────────────────────
const createNote = (data = {}) => ({
  id: uid(),
  title: data.title || '',
  body: data.body || '',
  tag: data.tag || '',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});

const saveActiveNote = () => {
  if (!state.activeNoteId) return;
  const idx = state.notes.findIndex(n => n.id === state.activeNoteId);
  if (idx === -1) return;
  const title = document.getElementById('noteTitleInput').value;
  const body = document.getElementById('noteBodyInput').value;
  const tag = document.getElementById('noteTagSelect').value;
  state.notes[idx] = { ...state.notes[idx], title, body, tag, updatedAt: new Date().toISOString() };
  persist();
  renderNotesList();
  updateAutosave('saved');
  updateCounts(body);
};

const deleteNote = (id) => {
  state.notes = state.notes.filter(n => n.id !== id);
  if (state.activeNoteId === id) {
    state.activeNoteId = null;
    showEmptyState();
  }
  persist();
  renderNotesList();
  toast('Note deleted', 'info');
};

// ── EDITOR ─────────────────────────────────────────
const showEmptyState = () => {
  document.getElementById('emptyState').style.display = '';
  document.getElementById('editorActive').style.display = 'none';
};

const loadNoteIntoEditor = (note) => {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('editorActive').style.display = 'flex';
  document.getElementById('noteTitleInput').value = note.title;
  document.getElementById('noteBodyInput').value = note.body;
  document.getElementById('noteTagSelect').value = note.tag || '';

  const created = new Date(note.createdAt);
  const updated = new Date(note.updatedAt);
  document.getElementById('editorMeta').textContent =
    `Created ${fmt(created)} · Edited ${fmt(updated)}`;

  updateCounts(note.body);
  updateAutosave('saved');

  // reset markdown mode
  if (state.markdownMode) toggleMarkdownMode(false);
};

const updateCounts = (text) => {
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;
  document.getElementById('wordCount').textContent = `${words} word${words !== 1 ? 's' : ''}`;
  document.getElementById('charCount').textContent = `${chars} char${chars !== 1 ? 's' : ''}`;
};

const updateAutosave = (status) => {
  const pill = document.getElementById('autosavePill');
  const text = document.getElementById('autosaveText');
  pill.className = `autosave-pill ${status}`;
  text.textContent = status === 'saving' ? 'Saving…' : 'Saved';
};

const triggerAutosave = () => {
  updateAutosave('saving');
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(saveActiveNote, 800);
};

// ── RENDER SIDEBAR NOTES ────────────────────────────
const renderNotesList = () => {
  const list = document.getElementById('notesList');
  const count = document.getElementById('noteCount');

  let notes = [...state.notes].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

  if (state.activeTag !== 'all') {
    notes = notes.filter(n => n.tag === state.activeTag);
  }

  count.textContent = notes.length;

  if (notes.length === 0) {
    list.innerHTML = `<div style="padding:12px 4px;font-size:0.78rem;color:var(--text-muted);text-align:center">No notes yet</div>`;
    return;
  }

  list.innerHTML = notes.map(note => `
    <div class="note-item ${note.id === state.activeNoteId ? 'active' : ''}" data-id="${note.id}">
      <div class="note-item-title">${escapeHtml(note.title) || '<em style="opacity:0.4">Untitled</em>'}</div>
      <div class="note-item-meta">
        <span>${fmt(new Date(note.updatedAt))}</span>
        ${note.tag ? `<span class="note-item-tag ${note.tag}">${note.tag}</span>` : ''}
      </div>
    </div>
  `).join('');

  list.querySelectorAll('.note-item').forEach(el => {
    el.addEventListener('click', () => {
      const note = state.notes.find(n => n.id === el.dataset.id);
      if (!note) return;
      state.activeNoteId = note.id;
      loadNoteIntoEditor(note);
      renderNotesList();
    });
  });
};

const escapeHtml = (str) =>
  str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ── REMINDERS ──────────────────────────────────────
const renderReminders = () => {
  const list = document.getElementById('remindersList');
  if (state.reminders.length === 0) {
    list.innerHTML = `<div class="no-reminders">No reminders set yet. Open a note and click "Remind".</div>`;
    return;
  }
  const sorted = [...state.reminders].sort((a, b) => new Date(a.date) - new Date(b.date));
  list.innerHTML = sorted.map(r => `
    <div class="reminder-card" data-rid="${r.id}">
      <div class="reminder-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
          <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
        </svg>
      </div>
      <div class="reminder-info">
        <div class="reminder-note-title">${escapeHtml(r.noteTitle || 'Untitled')}</div>
        <div class="reminder-date">Review on ${fmtDate(r.date)}</div>
      </div>
      <span class="reminder-badge">${r.label}</span>
      <button class="reminder-delete-btn" data-rid="${r.id}" title="Remove reminder">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');

  list.querySelectorAll('.reminder-delete-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      state.reminders = state.reminders.filter(r => r.id !== btn.dataset.rid);
      persist();
      renderReminders();
      toast('Reminder removed', 'info');
    });
  });
};

// ── OVERLAYS ───────────────────────────────────────
const openOverlay = (id) => document.getElementById(id).classList.add('active');
const closeOverlay = (id) => document.getElementById(id).classList.remove('active');

// ── MARKDOWN TOGGLE ────────────────────────────────
const toggleMarkdownMode = (force) => {
  state.markdownMode = typeof force === 'boolean' ? force : !state.markdownMode;
  const textarea = document.getElementById('noteBodyInput');
  const preview = document.getElementById('markdownPreview');
  const btn = document.getElementById('markdownToggle');

  if (state.markdownMode) {
    preview.innerHTML = parseMarkdown(textarea.value);
    preview.style.display = '';
    textarea.style.display = 'none';
    btn.classList.add('md-active');
    btn.title = 'Back to editing';
  } else {
    preview.style.display = 'none';
    textarea.style.display = '';
    btn.classList.remove('md-active');
    btn.title = 'Toggle Markdown Preview';
  }
};

// ── SEARCH ─────────────────────────────────────────
const renderSearchResults = (query) => {
  const results = document.getElementById('searchResults');
  if (!query.trim()) {
    results.innerHTML = `<div class="search-empty">Start typing to search your notes…</div>`;
    return;
  }
  const q = query.toLowerCase();
  const matches = state.notes.filter(n =>
    n.title.toLowerCase().includes(q) || n.body.toLowerCase().includes(q)
  );
  if (matches.length === 0) {
    results.innerHTML = `<div class="search-empty">No notes found for "<strong>${escapeHtml(query)}</strong>"</div>`;
    return;
  }
  results.innerHTML = matches.slice(0, 8).map(n => {
    const preview = n.body.replace(/\n/g, ' ').slice(0, 80) + '…';
    return `
      <div class="search-result-item" data-id="${n.id}">
        <div class="search-result-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="search-result-info">
          <div class="search-result-title">${escapeHtml(n.title) || 'Untitled'}</div>
          <div class="search-result-preview">${escapeHtml(preview)}</div>
        </div>
      </div>
    `;
  }).join('');

  results.querySelectorAll('.search-result-item').forEach(el => {
    el.addEventListener('click', () => {
      const note = state.notes.find(n => n.id === el.dataset.id);
      if (!note) return;
      state.activeNoteId = note.id;
      loadNoteIntoEditor(note);
      renderNotesList();
      closeOverlay('searchOverlay');
      // switch to editor view
      switchView('editor');
    });
  });
};

// ── VIEW SWITCHING ─────────────────────────────────
const switchView = (view) => {
  document.getElementById('editorView').style.display = view === 'editor' ? '' : 'none';
  document.getElementById('plannerView').style.display = view === 'planner' ? '' : 'none';
  document.querySelectorAll('.pill-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.view === view);
  });
  if (view === 'planner') renderReminders();
};

// ── KEYBOARD SHORTCUTS ─────────────────────────────
document.addEventListener('keydown', (e) => {
  // ⌘K / Ctrl+K → search
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    openOverlay('searchOverlay');
    setTimeout(() => document.getElementById('searchInput').focus(), 50);
  }
  // ⌘N / Ctrl+N → new note
  if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
    e.preventDefault();
    handleNewNote();
  }
  // ⌘S / Ctrl+S → save
  if ((e.metaKey || e.ctrlKey) && e.key === 's') {
    e.preventDefault();
    saveActiveNote();
    toast('Note saved ✓', 'success');
  }
  // Esc → close overlays
  if (e.key === 'Escape') {
    ['searchOverlay', 'summaryOverlay', 'improveOverlay', 'reminderOverlay'].forEach(closeOverlay);
  }
  // ⌘M → markdown toggle
  if ((e.metaKey || e.ctrlKey) && e.key === 'm') {
    e.preventDefault();
    if (state.activeNoteId) toggleMarkdownMode();
  }
});

// ── NEW NOTE ───────────────────────────────────────
const handleNewNote = () => {
  saveActiveNote(); // save current before switching
  const note = createNote();
  state.notes.unshift(note);
  state.activeNoteId = note.id;
  persist();
  renderNotesList();
  loadNoteIntoEditor(note);
  switchView('editor');
  setTimeout(() => document.getElementById('noteTitleInput').focus(), 50);
};

// ── INIT ───────────────────────────────────────────
const init = () => {
  load();

  // Theme
  const savedTheme = localStorage.getItem('notemind_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  renderNotesList();

  // If notes exist, load first one
  if (state.notes.length > 0) {
    state.activeNoteId = state.notes[0].id;
    loadNoteIntoEditor(state.notes[0]);
    renderNotesList();
  }

  // ── EVENT LISTENERS ──

  // New note buttons
  document.getElementById('newNoteBtn').addEventListener('click', handleNewNote);
  document.getElementById('emptyNewBtn').addEventListener('click', handleNewNote);

  // Theme toggle
  document.getElementById('themeToggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('notemind_theme', next);
  });

  // View switching
  document.querySelectorAll('.pill-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  // Note title input
  document.getElementById('noteTitleInput').addEventListener('input', triggerAutosave);

  // Note body input
  document.getElementById('noteBodyInput').addEventListener('input', (e) => {
    updateCounts(e.target.value);
    triggerAutosave();
    if (state.markdownMode) {
      document.getElementById('markdownPreview').innerHTML = parseMarkdown(e.target.value);
    }
  });

  // Tag select
  document.getElementById('noteTagSelect').addEventListener('change', triggerAutosave);

  // Markdown toggle
  document.getElementById('markdownToggle').addEventListener('click', toggleMarkdownMode);

  // Delete note
  document.getElementById('deleteNoteBtn').addEventListener('click', () => {
    if (!state.activeNoteId) return;
    if (confirm('Delete this note?')) deleteNote(state.activeNoteId);
  });

  // Tag filter chips
  document.querySelectorAll('.tag-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.tag-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.activeTag = chip.dataset.tag;
      renderNotesList();
    });
  });

  // Search
  document.getElementById('searchToggle').addEventListener('click', () => {
    openOverlay('searchOverlay');
    setTimeout(() => document.getElementById('searchInput').focus(), 50);
  });
  document.getElementById('searchInput').addEventListener('input', (e) => {
    renderSearchResults(e.target.value);
  });
  document.getElementById('searchOverlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('searchOverlay')) closeOverlay('searchOverlay');
  });
  // init search display
  renderSearchResults('');

  // ── AI SUMMARY ──
  document.getElementById('summaryBtn').addEventListener('click', async () => {
    const body = document.getElementById('noteBodyInput').value;
    if (!body.trim()) { toast('Write something first!', 'info'); return; }
    openOverlay('summaryOverlay');
    document.getElementById('summaryLoading').style.display = '';
    document.getElementById('summaryContent').style.display = 'none';
    const result = await aiSummarize(body);
    document.getElementById('summaryLoading').style.display = 'none';
    const sc = document.getElementById('summaryContent');
    sc.style.display = '';
    sc.innerHTML = parseMarkdown(result);
  });
  document.getElementById('closeSummary').addEventListener('click', () => closeOverlay('summaryOverlay'));
  document.getElementById('summaryOverlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('summaryOverlay')) closeOverlay('summaryOverlay');
  });

  // ── AI IMPROVE ──
  document.getElementById('improveBtn').addEventListener('click', async () => {
    const body = document.getElementById('noteBodyInput').value;
    if (!body.trim()) { toast('Write something first!', 'info'); return; }
    openOverlay('improveOverlay');
    document.getElementById('improveLoading').style.display = '';
    document.getElementById('improveContent').style.display = 'none';
    const result = await aiImprove(body);
    state.improvedText = result;
    document.getElementById('improveLoading').style.display = 'none';
    document.getElementById('improveContent').style.display = '';
    document.getElementById('improveOriginal').textContent = body;
    document.getElementById('improveResult').textContent = result;
  });
  document.getElementById('closeImprove').addEventListener('click', () => closeOverlay('improveOverlay'));
  document.getElementById('cancelImprove').addEventListener('click', () => closeOverlay('improveOverlay'));
  document.getElementById('applyImprove').addEventListener('click', () => {
    if (!state.activeNoteId) return;
    document.getElementById('noteBodyInput').value = state.improvedText;
    updateCounts(state.improvedText);
    triggerAutosave();
    closeOverlay('improveOverlay');
    toast('Improvement applied ✓', 'success');
  });
  document.getElementById('improveOverlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('improveOverlay')) closeOverlay('improveOverlay');
  });

  // ── REMINDER MODAL ──
  document.getElementById('reminderBtn').addEventListener('click', () => {
    if (!state.activeNoteId) { toast('Select a note first', 'info'); return; }
    state.selectedReminderDays = null;
    document.querySelectorAll('.reminder-chip').forEach(c => c.classList.remove('selected'));
    document.getElementById('customReminderDate').value = '';
    openOverlay('reminderOverlay');
  });
  document.querySelectorAll('.reminder-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.reminder-chip').forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      state.selectedReminderDays = parseInt(chip.dataset.days);
      document.getElementById('customReminderDate').value = '';
    });
  });
  document.getElementById('closeReminder').addEventListener('click', () => closeOverlay('reminderOverlay'));
  document.getElementById('cancelReminder').addEventListener('click', () => closeOverlay('reminderOverlay'));
  document.getElementById('reminderOverlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('reminderOverlay')) closeOverlay('reminderOverlay');
  });
  document.getElementById('saveReminder').addEventListener('click', () => {
    const note = state.notes.find(n => n.id === state.activeNoteId);
    if (!note) return;

    let date, label;
    const custom = document.getElementById('customReminderDate').value;

    if (custom) {
      date = custom;
      label = fmtDate(custom);
    } else if (state.selectedReminderDays) {
      const d = new Date();
      d.setDate(d.getDate() + state.selectedReminderDays);
      date = d.toISOString().split('T')[0];
      const labels = { 1: 'Tomorrow', 3: 'In 3 days', 7: 'In 1 week', 14: 'In 2 weeks' };
      label = labels[state.selectedReminderDays] || `${state.selectedReminderDays} days`;
    } else {
      toast('Please select a reminder time', 'info'); return;
    }

    state.reminders.push({ id: uid(), noteId: note.id, noteTitle: note.title || 'Untitled', date, label });
    persist();
    closeOverlay('reminderOverlay');
    toast(`Reminder set for ${fmtDate(date)} ✓`, 'success');
  });

  // ── PDF UPLOAD ──
document.getElementById('pdfUpload').addEventListener('change', async (e) => {

  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:9000/upload-pdf", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  const note = createNote({
    title: data.title,
    body: data.content
  });

  state.notes.unshift(note);
  state.activeNoteId = note.id;

  persist();
  renderNotesList();
  loadNoteIntoEditor(note);

});

  // ── DRAG & DROP on editor for PDF ──
  const editorView = document.getElementById('editorView');
  editorView.addEventListener('dragover', (e) => { e.preventDefault(); editorView.style.outline = '2px dashed var(--accent)'; });
  editorView.addEventListener('dragleave', () => { editorView.style.outline = ''; });
  editorView.addEventListener('drop', (e) => {
    e.preventDefault();
    editorView.style.outline = '';
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.pdf')) {
      // Trigger the same logic
      document.getElementById('pdfUpload').dispatchEvent(
        Object.assign(new Event('change'), { target: { files: [file] } })
      );
    } else if (file) {
      toast('Only PDF files are supported', 'error');
    }
  });

  // ── DEMO NOTES (first run) ──
  if (state.notes.length === 0) {
    const demos = [
      {
        title: 'Welcome to NoteMind 🧠',
        body: `# Welcome to NoteMind\n\nNoteMind is your **AI-powered study companion** built for serious learners.\n\n## What you can do\n\n- Write and organize notes with *rich formatting*\n- Use **AI Summary** to distill key ideas\n- **Improve** your writing with one click\n- Upload PDFs and extract content\n- Set **revision reminders** to retain knowledge\n\n## Keyboard Shortcuts\n\n- \`⌘K\` — Search notes\n- \`⌘N\` — New note\n- \`⌘S\` — Save note\n- \`⌘M\` — Markdown preview\n\n> Start by clicking **New Note** or try the AI features on this note!`,
        tag: 'important',
      },
      {
        title: 'Quantum Computing Basics',
        body: `## What is Quantum Computing?\n\nQuantum computing uses quantum-mechanical phenomena such as **superposition** and **entanglement** to perform computation.\n\n## Key Concepts\n\n- **Qubit**: The basic unit of quantum information. Unlike classical bits (0 or 1), qubits can exist in superposition of both states simultaneously.\n\n- **Superposition**: A qubit can represent 0, 1, or any combination until measured.\n\n- **Entanglement**: Two qubits can be correlated so the state of one instantly influences the other, regardless of distance.\n\n- **Quantum Gates**: Operations that manipulate qubits, analogous to classical logic gates.\n\n## Applications\n\n1. Cryptography and security\n2. Drug discovery and molecular simulation\n3. Financial modeling and optimization\n4. Machine learning acceleration`,
        tag: 'lecture',
      },
      {
        title: 'Research: Neural Networks',
        body: `## Neural Network Architecture\n\nA neural network is a series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.\n\n### Layers\n\n- **Input Layer**: Receives raw data\n- **Hidden Layers**: Process and transform information\n- **Output Layer**: Produces final predictions\n\n### Activation Functions\n\n- ReLU (Rectified Linear Unit)\n- Sigmoid\n- Tanh\n- Softmax\n\n### Training Process\n\nNeural networks learn through **backpropagation** — adjusting weights based on the error gradient to minimize loss.\n\n*Key insight*: More data and deeper networks generally improve performance but require more compute resources.`,
        tag: 'research',
      },
    ];

    demos.forEach(d => {
      const n = createNote(d);
      state.notes.push(n);
    });

    // Set a sample reminder
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    state.reminders.push({
      id: uid(),
      noteId: state.notes[1].id,
      noteTitle: state.notes[1].title,
      date: tomorrow.toISOString().split('T')[0],
      label: 'Tomorrow',
    });

    persist();
    state.activeNoteId = state.notes[0].id;
    loadNoteIntoEditor(state.notes[0]);
    renderNotesList();
  }
};

// ── START ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
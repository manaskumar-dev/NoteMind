const Notify = {
  STORAGE_KEY: "reminder_notify_enabled",
  notified: new Set(), // note ids already notified this session — prevents repeat spam
  timer: null,

  isEnabled() {
    return localStorage.getItem(this.STORAGE_KEY) === "1";
  },

  async enable() {
    if (!("Notification" in window)) {
      Toast.error("This browser doesn't support notifications.");
      return false;
    }
    const perm = await Notification.requestPermission();
    if (perm !== "granted") {
      Toast.error("Notification permission was not granted.");
      return false;
    }
    localStorage.setItem(this.STORAGE_KEY, "1");
    return true;
  },

  disable() {
    localStorage.setItem(this.STORAGE_KEY, "0");
    this.stop();
  },

  async checkDue() {
    if (!this.isEnabled() || Notification.permission !== "granted") return;
    try {
      const notes = await Api.listNotes();
      const now = new Date();
      notes
        .filter((n) => n.reminder && !n.reminder.completed && new Date(n.reminder.next_review_at) <= now)
        .forEach((n) => {
          if (this.notified.has(n.id)) return;
          this.notified.add(n.id);
          const notif = new Notification(`Review due: ${n.title}`, {
            body: "It's time to review this note.",
            tag: `note-${n.id}`, // same tag replaces rather than stacking duplicates
          });
          notif.onclick = () => { window.focus(); Pages.noteEditor(n.id); };
        });
    } catch (_) {
      // A missed check silently retries next interval — not worth interrupting the user.
    }
  },

  start() {
    this.checkDue();
    this.timer = setInterval(() => this.checkDue(), 5 * 60 * 1000); // every 5 min
  },

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  },
};
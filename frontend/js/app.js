const PAGE_TITLES = {
  dashboard: null,
  notes: "My Notes",
  reminders: "Reminders",
  explore: "Explore a Topic",
  profile: "Profile",
  settings: "Settings",
};

const App = {
  currentPage: "dashboard",

  init() {
    this.wireProfileMenu();
    this.wireSettingsModal();
    document.documentElement.dataset.theme = localStorage.getItem("theme") || "light";
    Auth.init();

    document.querySelectorAll(".nav-item[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => this.navigate(btn.dataset.page));
    });

    document.getElementById("topbar-new-note").addEventListener("click", () => {
      this.navigate("notes", { newNote: true });
    });

    document.getElementById("global-search").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.value.trim()) {
        const query = e.target.value.trim();
        this.navigate("notes").then(() => {
          const input = document.getElementById("notes-search");
          if (input) { input.value = query; Pages.loadNotesList(); }
        });
      }
    });

    if (Api.loadToken()) this.enterApp();
    else this.enterAuth();
  },

  enterApp() {
    document.getElementById("auth-view").classList.add("hidden");
    document.getElementById("app-view").classList.remove("hidden");
    if (Notify.isEnabled()) Notify.start();
    this.navigate("dashboard");
  },

  enterAuth() {
    document.getElementById("app-view").classList.add("hidden");
    document.getElementById("auth-view").classList.remove("hidden");
  },

  wireProfileMenu() {
    const btn = document.getElementById("profile-menu-btn");
    const popover = document.getElementById("profile-popover");
    btn.addEventListener("click", (e) => { e.stopPropagation(); popover.classList.toggle("hidden"); });
    document.addEventListener("click", (e) => {
      if (!popover.classList.contains("hidden") && !popover.contains(e.target) && e.target !== btn) popover.classList.add("hidden");
    });
    document.getElementById("popover-settings-btn").addEventListener("click", () => {
      popover.classList.add("hidden");
      this.openSettingsModal();
    });
    document.getElementById("popover-logout-btn").addEventListener("click", () => {
      popover.classList.add("hidden");
      Auth.logout();
    });
  },

  wireSettingsModal() {
    const overlay = document.getElementById("settings-modal-overlay");
    document.getElementById("settings-modal-close").addEventListener("click", () => this.closeSettingsModal());
    document.getElementById("settings-modal-cancel").addEventListener("click", () => this.closeSettingsModal());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) this.closeSettingsModal(); });

    document.querySelectorAll(".appearance-option").forEach((opt) => {
      opt.addEventListener("click", () => {
        document.querySelectorAll(".appearance-option").forEach((o) => o.classList.remove("active"));
        opt.classList.add("active");
        const theme = opt.dataset.themeOption;
        document.documentElement.dataset.theme = theme;
        localStorage.setItem("theme", theme);
        this.updateThemeButton();
      });
    });

    document.getElementById("settings-display-name").addEventListener("input", (e) => {
      document.getElementById("settings-avatar").textContent = (e.target.value.trim()[0] || "?").toUpperCase();
    });

    document.getElementById("settings-modal-save").addEventListener("click", () => {
      localStorage.setItem("displayName", document.getElementById("settings-display-name").value.trim());
      Toast.success("Settings saved.");
      this.closeSettingsModal();
      if (this.currentPage === "dashboard") Dashboard.render(document.getElementById("content-area"));
    });
  },

  updateThemeButton() {
    const button = document.getElementById("theme-toggle-btn");
    if (button) button.textContent = document.documentElement.dataset.theme === "dark" ? "Light mode" : "Dark mode";
  },

  openSettingsModal() {
    const name = localStorage.getItem("displayName") || "";
    document.getElementById("settings-display-name").value = name;
    document.getElementById("settings-avatar").textContent = (name.trim()[0] || "?").toUpperCase();
    document.getElementById("settings-signed-in-email").textContent = Api.userEmail || "";
    const currentTheme = document.documentElement.dataset.theme || "light";
    document.querySelectorAll(".appearance-option").forEach((opt) => {
      opt.classList.toggle("active", opt.dataset.themeOption === currentTheme);
    });
    document.getElementById("settings-modal-overlay").classList.remove("hidden");
  },

  closeSettingsModal() {
    document.getElementById("settings-modal-overlay").classList.add("hidden");
  },

  async navigate(page, opts = {}) {
    this.currentPage = page;
    document.querySelectorAll(".nav-item[data-page]").forEach((btn) => btn.classList.toggle("active", btn.dataset.page === page));

    const title = PAGE_TITLES[page];
    if (title !== null) {
      document.getElementById("page-title").textContent = title || "";
      document.getElementById("page-subtitle").textContent = "";
    }

    const container = document.getElementById("content-area");
    try {
      switch (page) {
        case "dashboard": await Dashboard.render(container); break;
        case "notes": await Pages.notesList(container, opts); break;
        case "explore": Pages.explore(container); break;
        case "reminders": await Pages.reminders(container); break;
        case "profile": Pages.profile(container); break;
        case "settings": Pages.settings(container); break;
        default: container.innerHTML = "";
      }
    } catch (err) {
      if (err.message && err.message.includes("401")) this.enterAuth();
      else container.innerHTML = ErrorState.html(err.message, `App.navigate('${page}')`);
    }
  },
};

document.addEventListener("DOMContentLoaded", () => App.init());
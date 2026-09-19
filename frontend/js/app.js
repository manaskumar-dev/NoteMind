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
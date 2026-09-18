const App = {
  init() {
    Auth.init();
    Notes.init();

    if (Api.loadToken()) {
      this.enterApp();
    } else {
      this.enterAuth();
    }
  },

  enterApp() {
    document.getElementById("auth-view").classList.add("hidden");
    document.getElementById("app-view").classList.remove("hidden");
    Notes.refreshList().catch((err) => {
      if (err.message.includes("401")) this.enterAuth();
    });
  },

  enterAuth() {
    document.getElementById("app-view").classList.add("hidden");
    document.getElementById("auth-view").classList.remove("hidden");
  },
};

document.addEventListener("DOMContentLoaded", () => App.init());

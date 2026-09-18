const Auth = {
  init() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => this.switchTab(btn.dataset.tab));
    });

    document.getElementById("login-form").addEventListener("submit", (e) => this.handleLogin(e));
    document.getElementById("register-form").addEventListener("submit", (e) => this.handleRegister(e));
    document.getElementById("logout-btn").addEventListener("click", () => this.logout());
  },

  switchTab(tab) {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.getElementById("login-form").classList.toggle("hidden", tab !== "login");
    document.getElementById("register-form").classList.toggle("hidden", tab !== "register");
    this.clearError();
  },

  showError(msg) {
    const el = document.getElementById("auth-error");
    el.textContent = msg;
    el.classList.remove("hidden");
  },

  clearError() {
    document.getElementById("auth-error").classList.add("hidden");
  },

  async handleLogin(e) {
    e.preventDefault();
    this.clearError();
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;
    try {
      const { access_token } = await Api.login(email, password);
      Api.setToken(access_token);
      App.enterApp();
    } catch (err) {
      this.showError(err.message);
    }
  },

  async handleRegister(e) {
    e.preventDefault();
    this.clearError();
    const email = document.getElementById("register-email").value;
    const password = document.getElementById("register-password").value;
    try {
      await Api.register(email, password);
      const { access_token } = await Api.login(email, password);
      Api.setToken(access_token);
      App.enterApp();
    } catch (err) {
      this.showError(err.message);
    }
  },

  logout() {
    Api.setToken(null);
    App.enterAuth();
  },
};

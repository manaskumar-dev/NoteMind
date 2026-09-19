const Auth = {
  init() {
    document.querySelectorAll("[data-tab]").forEach((link) => {
      link.addEventListener("click", (e) => { e.preventDefault(); this.switchTab(link.dataset.tab); });
    });

    document.querySelectorAll(".eye-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const input = document.getElementById(btn.dataset.target);
        input.type = input.type === "password" ? "text" : "password";
        btn.textContent = input.type === "password" ? "👁" : "🙈";
      });
    });

    document.getElementById("login-form").addEventListener("submit", (e) => this.handleLogin(e));
    document.getElementById("register-form").addEventListener("submit", (e) => this.handleRegister(e));
  },

  switchTab(tab) {
    document.getElementById("login-form").classList.toggle("hidden", tab !== "login");
    document.getElementById("register-form").classList.toggle("hidden", tab !== "register");
    this.clearError();
  },

  showError(msg) {
    const el = document.getElementById("auth-error");
    el.textContent = msg;
    el.classList.remove("hidden");
  },

  clearError() { document.getElementById("auth-error").classList.add("hidden"); },

  async handleLogin(e) {
    e.preventDefault();
    this.clearError();
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;
    const remember = document.getElementById("remember-me").checked;
    try {
      const { access_token } = await Api.login(email, password);
      Api.setToken(access_token, remember);
      Api.userEmail = email;
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
      Api.setToken(access_token, true);
      Api.userEmail = email;
      App.enterApp();
    } catch (err) {
      this.showError(err.message);
    }
  },

  logout() {
    Api.setToken(null);
    Api.userEmail = null;
    App.enterAuth();
  },
};
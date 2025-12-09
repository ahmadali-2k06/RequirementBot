// Configuration
// Ensure this matches where your Node.js server is running
const API_BASE_URL = "http://localhost:5000";

// Store signup data for OTP verification
let signupData = {};

// Toggle between login and signup
function toggleForm() {
  document.getElementById("loginForm").classList.toggle("active");
  document.getElementById("signupForm").classList.toggle("active");
  document.getElementById("otpForm").classList.remove("active");
  clearMessages();
  clearForm();
  clearOTPInputs();
}

// Show message
function showMessage(type, message) {
  const msgElement =
    type === "success"
      ? document.getElementById("successMessage")
      : document.getElementById("errorMessage");

  msgElement.textContent = message;
  msgElement.classList.add("show");

  setTimeout(() => {
    msgElement.classList.remove("show");
  }, 5000);
}

// Clear messages
function clearMessages() {
  const successMsg = document.getElementById("successMessage");
  const errorMsg = document.getElementById("errorMessage");
  if (successMsg) successMsg.classList.remove("show");
  if (errorMsg) errorMsg.classList.remove("show");
}

// Clear forms
function clearForm() {
  document.getElementById("loginFormElement").reset();
  document.getElementById("signupFormElement").reset();
}

// Clear OTP inputs
function clearOTPInputs() {
  document.getElementById("otp1").value = "";
  document.getElementById("otp2").value = "";
  document.getElementById("otp3").value = "";
  document.getElementById("otp4").value = "";
}

// Show loader
function showLoader(button) {
  const span = button.querySelector("span");
  if (span) span.style.display = "none";
  button.innerHTML = '<div class="loader"></div>';
  button.disabled = true;
}

// Hide loader
function hideLoader(button, text) {
  button.innerHTML = `<span>${text}</span>`;
  button.disabled = false;
}

// OTP Input Handler
function setupOTPInputs() {
  const otpInputs = document.querySelectorAll(".otp-input");

  otpInputs.forEach((input, index) => {
    input.addEventListener("input", (e) => {
      if (e.target.value.length > 1) {
        e.target.value = e.target.value.slice(0, 1);
      }

      // Allow only numbers
      if (!/^\d*$/.test(e.target.value)) {
        e.target.value = e.target.value.replace(/\D/g, "");
      }

      // Move to next input
      if (e.target.value.length === 1 && index < otpInputs.length - 1) {
        otpInputs[index + 1].focus();
      }
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Backspace" && e.target.value === "" && index > 0) {
        otpInputs[index - 1].focus();
      }
    });
  });
}

// Get OTP code from inputs
function getOTPCode() {
  const otp1 = document.getElementById("otp1").value;
  const otp2 = document.getElementById("otp2").value;
  const otp3 = document.getElementById("otp3").value;
  const otp4 = document.getElementById("otp4").value;
  return otp1 + otp2 + otp3 + otp4;
}

// Switch to OTP form
function switchToOTPForm(email) {
  document.getElementById("signupForm").classList.remove("active");
  document.getElementById("otpForm").classList.add("active");
  document.getElementById("otpEmail").textContent = email;
  clearOTPInputs();
  document.getElementById("otp1").focus();
}

// Back to signup from OTP
document.addEventListener("DOMContentLoaded", () => {
  const backBtn = document.getElementById("backToSignupBtn");
  if (backBtn) {
    backBtn.addEventListener("click", () => {
      document.getElementById("otpForm").classList.remove("active");
      document.getElementById("signupForm").classList.add("active");
      clearOTPInputs();
      clearMessages();
    });
  }

  setupOTPInputs();
});

// --- Login Handler ---
document
  .getElementById("loginFormElement")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const loginBtn = document.getElementById("loginBtn");

    showLoader(loginBtn);

    const email = document.getElementById("loginEmail").value;
    const password = document.getElementById("loginPassword").value;

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.msg || "Login failed. Please try again.");
      }

      // Success
      showMessage("success", "Login successful! Redirecting...");

      // Store the JWT token (Critical for Dashboard Access)
      localStorage.setItem("accessToken", data.accessToken);
      if (data.user) {
        localStorage.setItem("user", JSON.stringify(data.user));
      }

      hideLoader(loginBtn, "Login");

     
      // UPDATED: Redirect to index.html (The Dashboard File)   
      setTimeout(() => {
        window.location.href = "dashboard.html"; 
      }, 1500);
      
    } catch (error) {
      console.error(error);
      showMessage("error", error.message);
      hideLoader(loginBtn, "Login");
    }
  });

// --- Signup Handler ---
document
  .getElementById("signupFormElement")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const signupBtn = document.getElementById("signupBtn");

    showLoader(signupBtn);

    const name = document.getElementById("signupName").value;
    const email = document.getElementById("signupEmail").value;
    const password = document.getElementById("signupPassword").value;
    const organization = document.getElementById("signupOrganization").value;

    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          email,
          password,
          organization,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.msg || "Signup failed. Please try again.");
      }

      // Store signup data for OTP verification
      signupData = { name, email, password, organization };

      // Success - show OTP form
      showMessage("success", "OTP sent to your email!");
      hideLoader(signupBtn, "Create Account");
      switchToOTPForm(email);
    } catch (error) {
      console.error(error);
      showMessage("error", error.message);
      hideLoader(signupBtn, "Create Account");
    }
  });

// --- OTP Verification Handler ---
document.getElementById("verifyOtpBtn").addEventListener("click", async () => {
  const verifyBtn = document.getElementById("verifyOtpBtn");
  const otpCode = getOTPCode();

  if (otpCode.length !== 4) {
    showMessage("error", "Please enter all 4 digits of OTP");
    return;
  }

  showLoader(verifyBtn);
  clearMessages();

  try {
    const response = await fetch(`${API_BASE_URL}/auth/verify-otp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: signupData.email,
        otp: otpCode,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.msg || "OTP verification failed. Please try again.");
    }

    // Success
    showMessage("success", "Email verified! Logging you in...");

    // Store the JWT token (Critical for Dashboard Access)
    localStorage.setItem("accessToken", data.accessToken);
    if (data.user) {
      localStorage.setItem("user", JSON.stringify(data.user));
    }

    hideLoader(verifyBtn, "Verify OTP");

    // UPDATED: Redirect to index.html (The Dashboard File)

    setTimeout(() => {
      window.location.href = "dashboard.html";
    }, 1500);
    
  } catch (error) {
    console.error(error);
    showMessage("error", error.message);
    hideLoader(verifyBtn, "Verify OTP");
  }
});

// --- Resend OTP Handler ---
document.getElementById("resendOtpBtn").addEventListener("click", async (e) => {
  e.preventDefault();
  clearMessages();

  try {
    const response = await fetch(`${API_BASE_URL}/auth/resend-otp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: signupData.email,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.msg || "Failed to resend OTP");
    }

    showMessage("success", "OTP resent successfully!");
    clearOTPInputs();
    document.getElementById("otp1").focus();
  } catch (error) {
    console.error(error);
    showMessage("error", error.message);
  }
});
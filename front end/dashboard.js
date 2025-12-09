/* =========================================
   1. CONFIGURATION & STATE
   ========================================= */
const API_BASE_URL = "http://127.0.0.1:8000/api";

// --- AUTHENTICATION CHECK (NEW) ---
const token = localStorage.getItem("accessToken");
const userData = JSON.parse(localStorage.getItem("user") || "{}");

// If no token is found, redirect to login immediately
if (!token) {
  window.location.href = "login.html";
}

let currentProjectId = null;
// Ensure this matches the capitalized IDs in HTML: 'Functional', 'Non-Functional', 'Constraints'
let currentActiveTab = "Functional";

/* =========================================
   2. DOM ELEMENTS
   ========================================= */
// Sidebar (Left)
const projectListEl = document.getElementById("projectList");
const addProjectBtn = document.getElementById("addProjectBtn");
const logoutBtn = document.getElementById("logoutBtn");

const userProfileName = document.querySelector(".user-details .name");
const userProfileRole = document.querySelector(".user-details .role");
// Select the Avatar circle to change the letter "U"
const userAvatar = document.querySelector(".user-card .avatar");

// Main Chat Area
const emptyState = document.getElementById("emptyState");
const chatInterface = document.getElementById("chatInterface");
const chatHistoryEl = document.getElementById("chatHistory");
const chatInput = document.getElementById("chatInput");
const sendMessageBtn = document.getElementById("sendMessageBtn");
const currentProjectTitle = document.getElementById("currentProjectTitle");
const projectStatusBadge = document.getElementById("projectStatusBadge");

// Sidebar (Right) - Requirements
const tabButtons = document.querySelectorAll(".tab-btn");
// Containers for cards
const frContainer = document.querySelector("#Functional .req-items");
const nfrContainer = document.querySelector("#Non-Functional .req-items");
const consContainer = document.querySelector("#Constraints .req-items");
// Badges
const frCountBadge = document.querySelector("#Functional .count-badge");
const nfrCountBadge = document.querySelector("#Non-Functional .count-badge");
const consCountBadge = document.querySelector("#Constraints .count-badge");

// Modal
const modal = document.getElementById("addProjectModal");
const newProjectForm = document.getElementById("newProjectForm");
const cancelProjectBtn = document.getElementById("cancelProjectBtn");

/* =========================================
   3. API HELPER
   ========================================= */
async function apiCall(endpoint, method = "GET", body = null) {
  const config = {
    method,
    headers: {
      "Content-Type": "application/json",
      // --- SECURITY UPDATE: Send Token to Python ---
      Authorization: `Bearer ${token}`,
    },
  };
  if (body) config.body = JSON.stringify(body);

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

    // --- HANDLE SESSION EXPIRY (401) ---
    if (response.status === 401) {
      alert("Session expired. Please login again.");
      logout(); // Auto logout
      return null;
    }

    if (response.status === 204) return null; // No Content

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`API Error: ${response.status} ${errText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Network Request Failed:", error);
    return null;
  }
}

/* =========================================
   4. INITIALIZATION
   ========================================= */
document.addEventListener("DOMContentLoaded", () => {
  // Set User Profile Info from LocalStorage
  if (userData && userData.name) {
    // 1. Update Name
    if (userProfileName) userProfileName.textContent = userData.name;

    // 2. Update Avatar Letter (First letter of name)
    if (userAvatar)
      userAvatar.textContent = userData.name.charAt(0).toUpperCase();

    // 3. Update Role/Organization
    if (userProfileRole && userData.organization) {
      userProfileRole.textContent = userData.organization;
    }
  }

  fetchProjects();
  setupTabs();

  // Logout Logic
  if (logoutBtn) {
    logoutBtn.addEventListener("click", logout);
  }
});

function logout() {
  if (confirm("Are you sure you want to log out?")) {
    localStorage.clear(); // Remove token and user data
    window.location.href = "login.html";
  }
}

function setupTabs() {
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      // UI Toggle
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Logic Toggle (Read data-tab attribute)
      const tabName = btn.getAttribute("data-tab");
      currentActiveTab = tabName;

      // Hide all content sections
      document
        .querySelectorAll(".req-content")
        .forEach((el) => (el.style.display = "none"));

      // Show the selected one
      const target = document.getElementById(tabName);
      if (target) target.style.display = "block";
    });
  });
}

/* =========================================
   5. PROJECT MANAGEMENT
   ========================================= */
async function fetchProjects() {
  projectListEl.innerHTML =
    '<div style="padding:20px;text-align:center;font-size:12px;color:#888;">Loading...</div>';

  const projects = await apiCall("/projects");
  projectListEl.innerHTML = ""; // Clear loader

  if (!projects || projects.length === 0) {
    projectListEl.innerHTML =
      '<div style="padding:20px;text-align:center;font-size:12px;color:#888;">No projects yet.</div>';
  } else {
    // Sort by updated_at desc (optional, handled by backend usually)
    projects.forEach(renderProjectItem);
  }
}

function renderProjectItem(project) {
  const li = document.createElement("li");
  li.className = "project-item";
  li.dataset.id = project.id;
  li.onclick = () => switchProject(project.id);

  li.innerHTML = `
        <i class="fa-regular fa-folder icon"></i>
        <div class="project-info">
            <span class="project-name">${project.name}</span>
            <span class="project-meta">${formatTime(project.updated_at)}</span>
        </div>
    `;
  // Prepend to show newest at top if appended sequentially
  projectListEl.appendChild(li);
}

async function switchProject(id) {
  if (currentProjectId === id) return;
  currentProjectId = id;

  // 1. Update Sidebar Active State
  document.querySelectorAll(".project-item").forEach((item) => {
    const isActive = item.dataset.id === String(id);
    item.classList.toggle("active", isActive);

    const icon = item.querySelector(".icon");
    if (icon) {
      icon.className = isActive
        ? "fa-solid fa-folder-open icon"
        : "fa-regular fa-folder icon";
    }
  });

  // 2. Show Interface / Hide Empty State
  if (emptyState) emptyState.style.display = "none";
  if (chatInterface) chatInterface.style.display = "flex";
  if (projectStatusBadge) projectStatusBadge.style.display = "inline-block";

  // 3. Load Data
  currentProjectTitle.textContent = "Loading...";

  try {
    const [projectData, messages, requirements] = await Promise.all([
      apiCall(`/projects/${id}`),
      apiCall(`/projects/${id}/messages`),
      apiCall(`/projects/${id}/requirements`),
    ]);

    if (projectData) {
      // Update Title
      currentProjectTitle.textContent =
        projectData.scope.name || "Untitled Project";

      // Render Chat
      renderChatHistory(messages || []);

      // Render Sidebars (The Crucial Fix for Point 1 & 5)
      renderRequirementsAndConstraints(requirements || [], projectData.scope);
    }
  } catch (e) {
    currentProjectTitle.textContent = "Error Loading";
    console.error(e);
  }
}

/* =========================================
   6. REQUIREMENTS & CONSTRAINTS RENDER logic
   ========================================= */
function renderRequirementsAndConstraints(requirements, scope) {
  // 1. Clear Lists
  frContainer.innerHTML = "";
  nfrContainer.innerHTML = "";
  consContainer.innerHTML = "";

  let counts = { Functional: 0, "Non-Functional": 0, Constraints: 0 };

  // 2. Render Functional & Non-Functional (from req list)
  requirements.forEach((req) => {
    const card = createCard(
      req.code,
      req.priority,
      req.description,
      false,
      req.id
    );

    // Robust Check: Default to Functional unless "Non" is explicitly found
    const isNFR = req.type && req.type.toLowerCase().includes("non");

    if (isNFR) {
      nfrContainer.appendChild(card);
      counts["Non-Functional"]++;
    } else {
      // Everything else (Functional, FR, empty) goes here
      frContainer.appendChild(card);
      counts.Functional++;
    }
  });

  // 3. Render Constraints (From Scope - Splitting by comma)
  // We combine Technical and Business constraints from scope object
  const tech = scope.technical_constraints || [];
  const biz = scope.business_constraints || [];
  const allConstraints = [...tech, ...biz];

  allConstraints.forEach((text, index) => {
    // Only render if not empty string
    if (text && text.trim()) {
      const code = `CON-${String(index + 1).padStart(2, "0")}`;
      const card = createCard(code, "HIGH", text, true, null); // true = isConstraint
      consContainer.appendChild(card);
      counts.Constraints++;
    }
  });

  // 4. Update Badges
  frCountBadge.textContent = counts.Functional;
  nfrCountBadge.textContent = counts["Non-Functional"];
  consCountBadge.textContent = counts.Constraints;
}

function createCard(code, priority, text, isConstraint, reqId) {
  const div = document.createElement("div");
  div.className = "req-card";
  const prioLower = priority ? priority.toLowerCase() : "medium";

  div.innerHTML = `
        <div class="req-card-header">
            <span class="req-id">${code}</span>
            <span class="priority-tag ${prioLower}">${priority}</span>
        </div>
        <p class="req-text">${text}</p>
        ${
          !isConstraint && reqId
            ? `
            <div class="req-actions">
                <button class="action-small delete" onclick="deleteReq('${reqId}')" title="Delete">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `
            : ""
        }
    `;
  return div;
}

// Exposed global function for delete button
window.deleteReq = async function (reqId) {
  if (!confirm("Delete this requirement?")) return;

  try {
    await apiCall(`/requirements/${reqId}`, "DELETE");
    // Refresh list after delete
    const reqs = await apiCall(`/projects/${currentProjectId}/requirements`);
    const proj = await apiCall(`/projects/${currentProjectId}`);
    renderRequirementsAndConstraints(reqs, proj.scope);
  } catch (error) {
    alert("Failed to delete.");
  }
};

/* =========================================
   7. CHAT LOGIC
   ========================================= */
function renderChatHistory(messages) {
  chatHistoryEl.innerHTML = "";
  // Only show last 50 messages to keep DOM light if needed
  // messages = messages.slice(-50);
  messages.forEach((msg) =>
    appendMessage(msg.sender, msg.text, msg.created_at)
  );
  scrollToBottom();
}

function appendMessage(sender, text, timestamp) {
  const isBot = sender === "bot";
  const div = document.createElement("div");
  div.className = `message-wrapper ${isBot ? "bot" : "user"}`;

  // Using FontAwesome icons for avatars
  const avatarHtml = isBot
    ? `<div class="message-avatar" style="display:flex;align-items:center;justify-content:center;background:#fff;border:1px solid #eee;"><i class="fa-solid fa-robot" style="color:#2563eb;"></i></div>`
    : ``;

  // Time formatting
  const timeStr = formatTime(timestamp || new Date().toISOString());

  div.innerHTML = `
        ${avatarHtml}
        <div class="message-bubble">
            <div class="message-text">${text}</div>
            <span class="message-time" style="display:block;font-size:10px;margin-top:4px;opacity:0.6;text-align:${
              isBot ? "left" : "right"
            }">${timeStr}</span>
        </div>
    `;
  chatHistoryEl.appendChild(div);
}

sendMessageBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || !currentProjectId) return;

  // Optimistic UI
  appendMessage("user", text, new Date().toISOString());
  chatInput.value = "";
  scrollToBottom();

  // Show Loader
  const loaderId = "typing-loader";
  const loaderDiv = document.createElement("div");
  loaderDiv.id = loaderId;
  loaderDiv.className = "message-wrapper bot";
  loaderDiv.innerHTML = `
        <div class="message-avatar" style="display:flex;align-items:center;justify-content:center;background:#fff;"><i class="fa-solid fa-robot" style="color:#2563eb;"></i></div>
        <div class="message-bubble" style="color:#666; font-style:italic;">
           <i class="fa-solid fa-circle-notch fa-spin"></i> Processing...
        </div>`;
  chatHistoryEl.appendChild(loaderDiv);
  scrollToBottom();

  // API Call
  const response = await apiCall(
    `/projects/${currentProjectId}/messages`,
    "POST",
    { text }
  );

  // Remove Loader
  const loader = document.getElementById(loaderId);
  if (loader) loader.remove();

  if (response) {
    if (response.botReply) {
      appendMessage("bot", response.botReply, new Date().toISOString());
      scrollToBottom();
    }

    // 1. CHECK FOR EXIT COMMAND
    if (text.toLowerCase() === "exit") {
      disableChat();
    }

    // 2. REFRESH LOGIC
    if (response.newRequirementsDetected || text.length > 0) {
      const [proj, reqs] = await Promise.all([
        apiCall(`/projects/${currentProjectId}`),
        apiCall(`/projects/${currentProjectId}/requirements`),
      ]);
      if (proj && reqs) {
        renderRequirementsAndConstraints(reqs, proj.scope);
        if (proj.scope.name) currentProjectTitle.textContent = proj.scope.name;
      }
    }
  }
}

function scrollToBottom() {
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

/* =========================================
   8. MODAL (CREATE PROJECT)
   ========================================= */
addProjectBtn.addEventListener("click", () => modal.classList.remove("hidden"));

const hideModal = () => modal.classList.add("hidden");
if (cancelProjectBtn) cancelProjectBtn.addEventListener("click", hideModal);
// Close on outside click
modal.addEventListener("click", (e) => {
  if (e.target === modal) hideModal();
});

newProjectForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const inputs = newProjectForm.querySelectorAll("input");
  const name = inputs[0].value;
  const desc = inputs[1].value;

  const btn = newProjectForm.querySelector('button[type="submit"]');
  const oldText = btn.textContent;
  btn.textContent = "Creating...";
  btn.disabled = true;

  try {
    const newProject = await apiCall("/projects", "POST", {
      name,
      description: desc,
    }); // temp_name is mapped in backend

    if (newProject) {
      hideModal();
      newProjectForm.reset();

      // Add to list UI immediately
      const li = document.createElement("li");
      li.className = "project-item";
      li.dataset.id = newProject.id;
      li.onclick = () => switchProject(newProject.id);
      li.innerHTML = `
                <i class="fa-regular fa-folder icon"></i>
                <div class="project-info">
                    <span class="project-name">${newProject.name || name}</span>
                    <span class="project-meta">Just now</span>
                </div>
            `;
      projectListEl.prepend(li);

      // Auto switch
      switchProject(newProject.id);
    }
  } catch (err) {
    alert("Error creating project");
  } finally {
    btn.textContent = oldText;
    btn.disabled = false;
  }
});

/* =========================================
   9. UTILITIES
   ========================================= */
function formatTime(isoString) {
  if (!isoString) return "";
  const date = new Date(isoString);
  const now = new Date();

  // If today
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString();
}

function disableChat() {
  chatInput.disabled = true;
  sendMessageBtn.disabled = true;
  chatInput.placeholder = "Session ended. View final report above.";
  sendMessageBtn.style.opacity = "0.5";
  sendMessageBtn.style.cursor = "not-allowed";
}

function enableChat() {
  chatInput.disabled = false;
  sendMessageBtn.disabled = false;
  chatInput.placeholder = "Type your requirements here...";
  chatInput.style.backgroundColor = "#fff";
  sendMessageBtn.style.opacity = "1";
  sendMessageBtn.style.cursor = "pointer";
  chatInput.focus();
}

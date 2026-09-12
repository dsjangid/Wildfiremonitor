/**
 * Donezo Dashboard Interactive Engine
 * Handles timer ticking, search filtering, dynamic project/member creation, and modals.
 */

// ==========================================
// 1. DIGITAL TIME TRACKER (01:24:08)
// ==========================================
let timerSeconds = 1 * 3600 + 24 * 60 + 8; // 01:24:08 baseline
let isTimerRunning = true;
let timerInterval = null;

function formatTime(totalSecs) {
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function updateTimerDisplay() {
  const display = document.getElementById('timerDigitalDisplay');
  if (display) {
    display.textContent = formatTime(timerSeconds);
  }
}

function startTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    if (isTimerRunning) {
      timerSeconds++;
      updateTimerDisplay();
    }
  }, 1000);
}

function toggleTimer() {
  isTimerRunning = !isTimerRunning;
  const pauseIcon = document.getElementById('timerPauseIcon');
  if (!isTimerRunning) {
    // Show Play icon
    pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
  } else {
    // Show Pause icon
    pauseIcon.innerHTML = `
      <rect x="6" y="4" width="4" height="16" rx="1"></rect>
      <rect x="14" y="4" width="4" height="16" rx="1"></rect>
    `;
  }
}

function resetTimer() {
  isTimerRunning = false;
  timerSeconds = 0;
  updateTimerDisplay();
  const pauseIcon = document.getElementById('timerPauseIcon');
  if (pauseIcon) {
    pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
  }
}

// Start timer on load
startTimer();

// ==========================================
// 2. SEARCH FILTERING
// ==========================================
const searchInput = document.getElementById('global-search');
if (searchInput) {
  searchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase().trim();

    // Filter project list
    const projectItems = document.querySelectorAll('#projectItemList .project-item');
    projectItems.forEach(item => {
      const text = item.textContent.toLowerCase();
      item.style.display = text.includes(term) ? 'flex' : 'none';
    });

    // Filter team list
    const teamItems = document.querySelectorAll('#teamMemberList .team-member-item');
    teamItems.forEach(item => {
      const text = item.textContent.toLowerCase();
      item.style.display = text.includes(term) ? 'flex' : 'none';
    });
  });

  // Keyboard shortcut: Cmd + F / Ctrl + F
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
      e.preventDefault();
      searchInput.focus();
    }
  });
}

// ==========================================
// 3. MODALS & CREATION
// ==========================================
function openAddProjectModal() {
  document.getElementById('addProjectModal').classList.add('active');
}

function closeAddProjectModal() {
  document.getElementById('addProjectModal').classList.remove('active');
}

function openAddMemberModal() {
  document.getElementById('addMemberModal').classList.add('active');
}

function closeAddMemberModal() {
  document.getElementById('addMemberModal').classList.remove('active');
}

function handleCreateProject(e) {
  e.preventDefault();
  const name = document.getElementById('inputProjName').value;
  const dueDate = document.getElementById('inputProjDueDate').value;
  const category = document.getElementById('inputProjCategory').value;

  const categoryConfigs = {
    blue: { bg: '#eef2ff', stroke: '#4f46e5', icon: `<line x1="9" y1="19" x2="15" y2="5"></line><line x1="4" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="20" y2="12"></line>` },
    teal: { bg: '#e6fffa', stroke: '#0d9488', icon: `<circle cx="12" cy="12" r="9"></circle><path d="M12 6a6 6 0 0 0-6 6"></path>` },
    green: { bg: '#f0fdf4', stroke: '#16a34a', icon: `<circle cx="12" cy="7" r="3" fill="#86efac"></circle><circle cx="7" cy="14" r="3" fill="#fde047"></circle><circle cx="17" cy="14" r="3" fill="#86efac"></circle>` },
    orange: { bg: '#fffbeb', stroke: '#d97706', icon: `<path d="M12 3a9 9 0 0 0-9 9h3a6 6 0 0 1 6-6V3z" fill="#f59e0b"></path>` },
    purple: { bg: '#fdf2f8', stroke: '#db2777', icon: `<circle cx="9" cy="9" r="4" fill="#a855f7"></circle><circle cx="15" cy="15" r="4" fill="#ec4899"></circle>` }
  };

  const cfg = categoryConfigs[category] || categoryConfigs.green;

  const newItem = document.createElement('div');
  newItem.className = 'project-item';
  newItem.innerHTML = `
    <div class="project-icon-badge" style="background: ${cfg.bg};">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${cfg.stroke}" stroke-width="2.5" stroke-linecap="round">
        ${cfg.icon}
      </svg>
    </div>
    <div class="project-text-block">
      <span class="project-name">${name}</span>
      <span class="project-due-date">Due date: ${dueDate}</span>
    </div>
  `;

  document.getElementById('projectItemList').prepend(newItem);

  // Increment total projects KPI
  const kpiTotal = document.getElementById('kpi-total');
  if (kpiTotal) {
    kpiTotal.textContent = parseInt(kpiTotal.textContent) + 1;
  }

  closeAddProjectModal();
  document.getElementById('addProjectForm').reset();
}

function handleCreateMember(e) {
  e.preventDefault();
  const name = document.getElementById('inputMemberName').value;
  const task = document.getElementById('inputMemberTask').value;
  const status = document.getElementById('inputMemberStatus').value;

  const statusClass = status === 'Completed' ? 'status-completed' : (status === 'In Progress' ? 'status-in-progress' : 'status-pending');

  const randomAvatars = [
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=100&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=100&auto=format&fit=crop&q=80"
  ];
  const avatar = randomAvatars[Math.floor(Math.random() * randomAvatars.length)];

  const newItem = document.createElement('div');
  newItem.className = 'team-member-item';
  newItem.innerHTML = `
    <div class="member-info-left">
      <img class="member-avatar" src="${avatar}" alt="${name}" />
      <div class="member-details-text">
        <span class="member-name">${name}</span>
        <span class="member-task">Working on <strong>${task}</strong></span>
      </div>
    </div>
    <span class="status-badge-pill ${statusClass}">${status}</span>
  `;

  document.getElementById('teamMemberList').prepend(newItem);

  closeAddMemberModal();
  document.getElementById('addMemberForm').reset();
}

// ==========================================
// 4. ACTION TRIGGERS
// ==========================================
function startMeetingSession() {
  alert("Launching 'Meeting with Arc Company' video room...");
}

function triggerImportData() {
  alert("Data Import Module opened: connect Jira, Asana, Linear, or CSV.");
}

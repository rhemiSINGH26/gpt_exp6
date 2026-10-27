/**
 * 24CS2506 - Building Applications using GPT-4
 * Experiment 5: AI-Powered Medical Symptom Assistant using GPT-4
 * Client-Side Application Logic & Pipeline Visualizer
 */

let currentReportData = null;
let currentReportId = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  checkBackendStatus();
});

/**
 * Checks system health and OpenAI connectivity
 */
async function checkBackendStatus() {
  const statusEl = document.getElementById("systemStatus");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.status === "online") {
      const dot = statusEl.querySelector(".status-dot");
      const text = statusEl.querySelector(".status-text");
      if (data.has_api_key) {
        text.textContent = `Connected to ${data.model}`;
        dot.style.background = "#10B981";
      } else {
        text.textContent = "Viva Demo Mode (Simulated AI Engine)";
        dot.style.background = "#3B82F6";
      }
    }
  } catch (err) {
    console.warn("Backend status check failed:", err);
  }
}

/**
 * Loads pre-configured scenario presets for rapid viva testing
 */
async function loadSample(sampleKey) {
  try {
    const res = await fetch("/api/samples");
    const json = await res.json();
    if (!json.success || !json.samples[sampleKey]) return;

    const s = json.samples[sampleKey];
    document.getElementById("symptoms_text").value = s.symptoms_text;
    document.getElementById("age").value = s.age;
    document.getElementById("sex").value = s.sex;
    document.getElementById("duration").value = s.duration;
    document.getElementById("medical_history").value = s.medical_history;
    document.getElementById("medications").value = s.medications;
    document.getElementById("allergies").value = s.allergies;

    // Set severity radio
    const severityRadios = document.getElementsByName("severity");
    for (const r of severityRadios) {
      r.checked = (r.value === s.severity);
    }

    // Scroll to form smoothly
    document.getElementById("symptomForm").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Failed to load sample scenario:", err);
  }
}

/**
 * Clears form fields and resets view to placeholder
 */
function resetForm() {
  document.getElementById("symptomForm").reset();
  document.getElementById("placeholderView").classList.remove("hidden");
  document.getElementById("resultsView").classList.add("hidden");
  document.getElementById("loadingView").classList.add("hidden");
  resetPipelineSteps();
}

/**
 * Handles Form Submission and orchestrates pipeline execution
 */
async function handleAnalysisSubmit(event) {
  event.preventDefault();

  const symptomsText = document.getElementById("symptoms_text").value.trim();
  const age = document.getElementById("age").value;
  const sex = document.getElementById("sex").value;
  const duration = document.getElementById("duration").value.trim();
  const medicalHistory = document.getElementById("medical_history").value.trim();
  const medications = document.getElementById("medications").value.trim();
  const allergies = document.getElementById("allergies").value.trim();

  let severity = "Mild";
  const severityRadios = document.getElementsByName("severity");
  for (const r of severityRadios) {
    if (r.checked) severity = r.value;
  }

  if (!symptomsText) {
    alert("Please enter symptoms in natural language.");
    return;
  }

  const payload = {
    symptoms_text: symptomsText,
    age: parseInt(age, 10),
    sex: sex,
    duration: duration,
    severity: severity,
    medical_history: medicalHistory || "None reported",
    medications: medications || "None reported",
    allergies: allergies || "None reported"
  };

  // UI State: Show Loading, Hide Results
  document.getElementById("placeholderView").classList.add("hidden");
  document.getElementById("resultsView").classList.add("hidden");
  document.getElementById("loadingView").classList.remove("hidden");
  document.getElementById("btnSubmit").disabled = true;

  // Animate Workflow Pipeline Steps
  animatePipeline();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!result.success) {
      alert("Error: " + (result.error || "Analysis failed"));
      document.getElementById("loadingView").classList.add("hidden");
      document.getElementById("placeholderView").classList.remove("hidden");
      return;
    }

    currentReportData = result.data;
    currentReportId = result.report_id || result.data.report_id;

    renderReport(result.data, currentReportId);

    document.getElementById("loadingView").classList.add("hidden");
    document.getElementById("resultsView").classList.remove("hidden");

  } catch (err) {
    console.error("Submission failed:", err);
    alert("Network or server error occurred. Please verify backend is running.");
    document.getElementById("loadingView").classList.add("hidden");
    document.getElementById("placeholderView").classList.remove("hidden");
  } finally {
    document.getElementById("btnSubmit").disabled = false;
  }
}

/**
 * Animates visual pipeline indicator
 */
function animatePipeline() {
  const stepProfile = document.getElementById("stepProfile");
  const stepExtract = document.getElementById("stepExtract");
  const stepPrompt = document.getElementById("stepPrompt");
  const stepGPT = document.getElementById("stepGPT");
  const stepAnalysis = document.getElementById("stepAnalysis");

  resetPipelineSteps();

  setTimeout(() => stepProfile?.classList.add("active"), 200);
  setTimeout(() => stepExtract?.classList.add("active"), 500);
  setTimeout(() => stepPrompt?.classList.add("active"), 800);
  setTimeout(() => stepGPT?.classList.add("active"), 1200);
  setTimeout(() => stepAnalysis?.classList.add("active"), 1600);
}

function resetPipelineSteps() {
  const steps = document.querySelectorAll(".pipeline-steps .step-badge");
  steps.forEach((s, idx) => {
    if (idx === 0) s.classList.add("active");
    else s.classList.remove("active");
  });
}

/**
 * Renders structured medical response into UI
 */
function renderReport(data, reportId) {
  // Report Meta
  document.getElementById("reportIdBadge").textContent = `#${reportId.replace("report_", "")}`;
  document.getElementById("reportEngineBadge").textContent = data.source || "GPT-4 Clinical Engine";

  // Triage Banner
  const triageBanner = document.getElementById("triageBanner");
  const triageIcon = document.getElementById("triageIcon");
  const triageLevelTag = document.getElementById("triageLevelTag");
  const triageDesc = document.getElementById("triageDesc");

  triageBanner.className = "triage-banner";
  const level = (data.triage_level || "Moderate").toLowerCase();

  if (level.includes("low")) {
    triageBanner.classList.add("triage-low");
    triageIcon.textContent = "🟢";
  } else if (level.includes("emergency") || level.includes("high")) {
    triageBanner.classList.add("triage-emergency");
    triageIcon.textContent = "🚨";
  } else {
    triageBanner.classList.add("triage-moderate");
    triageIcon.textContent = "🟡";
  }

  triageLevelTag.textContent = `Triage Level: ${data.triage_level || "Moderate"}`;
  triageDesc.textContent = data.triage_title || "Clinical Assessment";

  // Red Flags
  const redFlagBox = document.getElementById("redFlagBox");
  const redFlagList = document.getElementById("redFlagList");
  redFlagList.innerHTML = "";

  if (data.red_flags && data.red_flags.length > 0) {
    data.red_flags.forEach(rf => {
      const li = document.createElement("li");
      li.textContent = rf;
      redFlagList.appendChild(li);
    });
    redFlagBox.classList.remove("hidden");
  } else {
    redFlagBox.classList.add("hidden");
  }

  // Reported Symptoms List
  const symptomsList = document.getElementById("reportSymptomsList");
  symptomsList.innerHTML = "";
  const symptoms = data.reported_symptoms || [];
  if (symptoms.length > 0) {
    symptoms.forEach(s => {
      const li = document.createElement("li");
      li.textContent = s;
      symptomsList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "No specific symptoms isolated.";
    symptomsList.appendChild(li);
  }

  // Duration
  document.getElementById("reportDuration").textContent = data.duration || "Not specified";

  // Possible Conditions List
  const condList = document.getElementById("reportConditionsList");
  condList.innerHTML = "";
  const conditions = data.possible_conditions || [];
  if (conditions.length > 0) {
    conditions.forEach((c, idx) => {
      const li = document.createElement("li");
      const name = c.name || `Condition ${idx + 1}`;
      const desc = c.description || "";
      li.innerHTML = `<strong>${name}</strong>${desc ? `<span class="condition-desc">${desc}</span>` : ""}`;
      condList.appendChild(li);
    });
  } else {
    condList.innerHTML = "<li>None identified</li>";
  }

  // Why These Were Considered
  document.getElementById("reportWhyConsidered").textContent =
    data.why_these_were_considered || "The reported symptoms can occur with several conditions.";

  // Additional Information Needed
  const infoList = document.getElementById("reportInfoNeeded");
  infoList.innerHTML = "";
  const questions = data.additional_information_needed || [];
  if (questions.length > 0) {
    questions.forEach(q => {
      const li = document.createElement("li");
      li.textContent = q;
      infoList.appendChild(li);
    });
  } else {
    infoList.innerHTML = "<li>None specified</li>";
  }

  // Medical Attention
  document.getElementById("reportMedicalAttention").textContent =
    data.medical_attention || "Seek professional medical evaluation if symptoms are severe, worsening or concerning.";

  // Disclaimer
  document.getElementById("reportDisclaimer").innerHTML =
    `<strong>Disclaimer:</strong> ${data.disclaimer || "This is an informational AI analysis and not a medical diagnosis."}`;
}

/**
 * Downloads report as text or json
 */
function downloadReport(format) {
  if (!currentReportId) {
    alert("No active report to download.");
    return;
  }
  window.location.href = `/api/download-report/${currentReportId}?format=${format}`;
}

/**
 * Copies structured report to clipboard
 */
function copyReport() {
  if (!currentReportData || !currentReportData.formatted_text_report) {
    alert("No report content available to copy.");
    return;
  }
  navigator.clipboard.writeText(currentReportData.formatted_text_report)
    .then(() => {
      alert("✓ Structured Medical Report copied to clipboard!");
    })
    .catch(err => {
      console.error("Clipboard copy failed:", err);
      // Fallback copy using textarea
      const ta = document.createElement("textarea");
      ta.value = currentReportData.formatted_text_report;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      alert("✓ Structured Medical Report copied to clipboard!");
    });
}

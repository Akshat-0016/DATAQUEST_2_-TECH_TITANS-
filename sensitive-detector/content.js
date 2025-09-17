const patterns = [
  { name: "Email", regex: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i, severity: "low" },
  { name: "Credit Card", regex: /\b(?:\d[ -]*?){13,16}\b/, severity: "high" },
  { name: "Phone Number", regex: /\b\d{10}\b/, severity: "medium" },
  { name: "US SSN", regex: /\b\d{3}-\d{2}-\d{4}\b/, severity: "high" }
];

// =====================
// 2. Find matches in text
// =====================
function findMatches(text) {
  const hits = [];
  for (const p of patterns) {
    if (p.regex.test(text)) hits.push({ type: p.name, severity: p.severity });
  }
  return hits;
}

// =====================
// 3. Apply style + alerts
// =====================
function handleDetection(el, matches) {
  if (!matches.length) {
    el.style.outline = "";
    return;
  }

  // Add red border
  el.style.outline = "3px solid red";

  // For each detected type → different response
  matches.forEach(m => {
    if (m.severity === "high") {
      alert(" HIGH risk data detected: " + m.type + "!\nPlease remove immediately.");
    } else if (m.severity === "medium") {
      alert(" Medium sensitivity: " + m.type);
    } else if (m.severity === "low") {
      alert(" Low sensitivity detected: " + m.type);
    }
  });

  // Notify background/popup
  chrome.runtime.sendMessage({ type: "sensitiveDetected", details: matches.map(m => m.type) });
}

// =====================
// 4. Input & contentEditable detection
// =====================
document.addEventListener("input", (e) => {
  const t = e.target;

  let text = "";
  if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) {
    text = t.value;
  } else if (t.isContentEditable) {
    text = t.innerText;
  } else {
    return;
  }

  const matches = findMatches(text);
  handleDetection(t, matches);
}, true);

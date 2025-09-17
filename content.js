// =====================
// 1. API call for probability
// =====================
async function getProbabilityFromAPI(text) {
  try {
    let response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    let data = await response.json();
    return data.probability; // already in %
  } catch (err) {
    console.error("API request failed:", err);
    return 0; // fallback to safe probability
  }
}

// =====================
// 2. Blocking modal (80–100%)
// =====================
function showBlockingModal(message) {
  if (document.getElementById("sensitive-blocker")) return;

  const overlay = document.createElement("div");
  overlay.id = "sensitive-blocker";
  overlay.style.position = "fixed";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = "100vw";
  overlay.style.height = "100vh";
  overlay.style.background = "rgba(0,0,0,0.85)";
  overlay.style.color = "white";
  overlay.style.display = "flex";
  overlay.style.alignItems = "center";
  overlay.style.justifyContent = "center";
  overlay.style.fontSize = "20px";
  overlay.style.textAlign = "center";
  overlay.style.zIndex = "999999";
  overlay.innerText = message + "\n(Press ESC or wait 10s to dismiss)";

  document.body.appendChild(overlay);

  function escHandler(e) {
    if (e.key === "Escape") {
      overlay.remove();
      document.removeEventListener("keydown", escHandler);
    }
  }
  document.addEventListener("keydown", escHandler);

  setTimeout(() => {
    if (document.getElementById("sensitive-blocker")) {
      overlay.remove();
      document.removeEventListener("keydown", escHandler);
    }
  }, 10000);
}

// =====================
// 3. Detection handler
// =====================
let popupShown60to80 = false;

function handleDetection(el, prob, text) {
  el.style.outline = "";
  el.style.textDecoration = "none";

  const phoneRegex = /\b\d{10}\b/;

  // --- Case: Phone number → cancellable popup
  if (phoneRegex.test(text)) {
    if (confirm(`📱 Phone number detected. Probability ${prob}%. Do you want to continue?`)) {
      return; // user allowed
    } else {
      el.value = ""; // clear field
      return;
    }
  }

  // --- General sensitivity logic
  if (prob < 40) {
    return;
  } else if (prob < 60) {
    el.style.textDecoration = "underline red wavy"; // RED underline
  } else if (prob < 80) {
    if (!popupShown60to80) {
      alert(`⚠️ Sensitive data detected (Probability ${prob}%). Please be cautious.`);
      popupShown60to80 = true;
    }
  } else {
    showBlockingModal(`⛔ HIGHLY sensitive data detected (${prob}%). Action blocked!`);
  }

  chrome.runtime?.sendMessage?.({
    type: "sensitiveDetected",
    probability: prob,
    text: text
  });
}

// =====================
// 4. Input listener
// =====================
document.addEventListener("input", async (e) => {
  const t = e.target;
  let text = "";

  if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) {
    text = t.value;
  } else if (t.isContentEditable) {
    text = t.innerText;
  } else {
    return;
  }

  const prob = await getProbabilityFromAPI(text); // <-- API call
  handleDetection(t, prob, text);
}, true);

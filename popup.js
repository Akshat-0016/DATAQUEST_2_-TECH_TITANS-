chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.type === "sensitiveDetected") {
    document.getElementById("log").textContent =
      "Sensitive detected (Prob " + msg.probability + "%): " + msg.text;
  }
});

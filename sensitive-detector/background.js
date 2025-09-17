chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "sensitiveDetected") {
    console.log("Background received sensitive data:", msg.details);
  }
});
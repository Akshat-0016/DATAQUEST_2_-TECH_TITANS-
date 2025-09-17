chrome.runtime.onMessage.addListener((msg, sender) => {
  // Optional: you could log or store detection here
  console.log("Background got message:", msg);
});

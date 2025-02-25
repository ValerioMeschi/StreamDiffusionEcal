document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("sendPrompt").addEventListener("click", async () => {
    const prompt = document.getElementById("promptInput").value;
    if (!prompt) return alert("Please enter a prompt!");

    const formData = new FormData();
    formData.append("prompt", prompt);

    try {
      const response = await fetch("/prompt", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        alert("Prompt updated successfully!");
      } else {
        alert("Failed to update prompt.");
      }
    } catch (error) {
      console.error("Error sending prompt:", error);
    }
  });
});

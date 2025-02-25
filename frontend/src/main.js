document.addEventListener("DOMContentLoaded", () => {
  const promptInput = document.getElementById("promptInput");
  const sendPromptButton = document.getElementById("sendPrompt");

  const sendPrompt = async () => {
    const prompt = promptInput.value;
    if (!prompt) {
      return alert("Please enter a prompt!");
    }

    const formData = new FormData();
    formData.append("prompt", prompt);

    await fetch("/prompt", {
      method: "POST",
      body: formData,
    });

    promptInput.value = "";
  };

  sendPromptButton.addEventListener("click", sendPrompt);

  promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendPrompt();
    }
  });
});

import * as dat from "dat.gui";

document.addEventListener("DOMContentLoaded", () => {
  const gui = new dat.GUI({ width: 300 });

  const params = {
    prompt: "",
    seed: 1,
    showInputFeed: true,
  };

  const promptController = gui.add(params, "prompt").name("Prompt");
  const seedController = gui.add(params, "seed", 0, 999999, 1).name("Seed");
  const feedController = gui.add(params, "showInputFeed").name("Input Feed");

  const sendParams = async () => {
    const formData = new FormData();

    formData.append("seed", params.seed);
    console.log("Prompt:", params.prompt);

    if (params.prompt.trim()) {
      formData.append("prompt", params.prompt);
    }

    await fetch("/set_params", {
      method: "POST",
      body: formData,
    });

    params.prompt = "";
    promptController.updateDisplay();
  };

  seedController.onChange(() => {
    sendParams();
  });

  feedController.onChange(() => {
    const inputFeedEl = document.getElementById("inputFeed");
    inputFeedEl.style.display = params.showInputFeed ? "" : "none";
  });

  const promptInputElem = promptController.domElement.querySelector("input");
  if (promptInputElem) {
    promptInputElem.addEventListener("input", () => {
      sendParams();
    });
  }
});

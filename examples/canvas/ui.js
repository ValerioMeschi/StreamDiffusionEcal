document.addEventListener("DOMContentLoaded", () => {
  const gui = new dat.GUI({ width: 300 });

  const params = {
    prompt:
      "An dark tunnel, arched ceiling, architecture, extremely detailed, 4K, dark, brick, brick walls",
    seed: 1,
    delta: 0.7,
    showInputFeed: true,
  };

  const promptController = gui.add(params, "prompt").name("Prompt");
  const seedController = gui.add(params, "seed", 0, 999999, 1).name("Seed");
  const deltaController = gui.add(params, "delta", -3, 3, 0.01).name("Delta");
  const feedController = gui.add(params, "showInputFeed").name("Input Feed");

  const sendParams = async () => {
    const formData = new FormData();

    formData.append("seed", params.seed);
    formData.append("delta", params.delta);

    console.log("Prompt:", params.prompt);

    if (params.prompt.trim()) {
      formData.append("prompt", params.prompt);
    }

    await fetch("http://127.0.0.1:5000/set_params", {
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

  deltaController.onChange(() => {
    sendParams();
  });

  const promptInputElem = promptController.domElement.querySelector("input");
  if (promptInputElem) {
    promptInputElem.addEventListener("input", () => {
      sendParams();
    });
  }
});

import { Pane } from "tweakpane";

document.addEventListener("DOMContentLoaded", () => {
  const pane = new Pane({
    title: "Parameters",
    expanded: true,
  });

  const params = {
    prompt: "",
    seed: 1, // Default seed
    showInputFeed: true, // New switch to show/hide inputFeed
  };

  pane.addBinding(params, "prompt", { label: "Prompt" });
  pane.addBinding(params, "seed", {
    label: "Seed",
    min: 0,
    max: 999999,
    step: 1,
  });
  pane.addBinding(params, "showInputFeed", {
    label: "Input Feed",
    view: "checkbox",
  });

  const sendParams = async () => {
    const formData = new FormData();

    // Always send seed
    formData.append("seed", params.seed);

    // Send prompt only if not empty
    if (params.prompt.trim()) {
      formData.append("prompt", params.prompt);
    }

    await fetch("/set_params", {
      method: "POST",
      body: formData,
    });

    params.prompt = "";
    pane.refresh();
  };

  pane.addButton({ title: "Send" }).on("click", sendParams);

  pane.on("change", (ev) => {
    if (ev.target.key === "seed") {
      sendParams();
    } else if (ev.target.key === "showInputFeed") {
      const inputFeedEl = document.getElementById("inputFeed");
      if (params.showInputFeed) {
        inputFeedEl.style.display = "";
      } else {
        inputFeedEl.style.display = "none";
      }
    }
  });
});

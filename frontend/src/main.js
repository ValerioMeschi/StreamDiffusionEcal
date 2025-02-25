import { Pane } from "tweakpane";

document.addEventListener("DOMContentLoaded", () => {
  const pane = new Pane();

  const params = {
    prompt: "",
    seed: 1, // Default seed
  };

  pane.addBinding(params, "prompt", { label: "Prompt" });

  pane.addBinding(params, "seed", {
    label: "Seed",
    min: 0,
    max: 999999,
    step: 1,
  });

  const sendParams = async () => {
    const formData = new FormData();

    if (params.prompt.trim()) {
      // If prompt is not empty, send prompt and seed
      formData.append("prompt", params.prompt);
      formData.append("seed", params.seed);

      await fetch("/set_params", {
        method: "POST",
        body: formData,
      });

      params.prompt = "";
      pane.refresh();
    } else {
      // If prompt is empty, send only the seed
      formData.append("seed", params.seed);

      await fetch("/set_seed", {
        method: "POST",
        body: formData,
      });
    }
  };

  pane.addButton({ title: "Send Parameters" }).on("click", sendParams);

  // Send only seed updates separately on change
  pane.on("change", (ev) => {
    if (ev.target.key === "seed") {
      const seedForm = new FormData();
      seedForm.append("seed", params.seed);

      fetch("/set_seed", {
        method: "POST",
        body: seedForm,
      });
    }
  });
});

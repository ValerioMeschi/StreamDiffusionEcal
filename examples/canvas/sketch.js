let canvas;
let noiseImg;
function setup() {
  canvas = createCanvas(512, 512);
  noiseImg = loadImage("noise.png");
}
function draw() {
  blendMode(BLEND);
  tint(255, 255);
  background(255);
  let circles = 30;
  for (let i = 0; i < circles; i++) {
    let center = { x: width / 2, y: height / 2 };
    size = map(i, 0, circles, 10, 700) * (0.8 + 0.5 * noise(i * 0.2 + 2000));
    opacity = map(i, 1, circles, 50, 0);
    mouseInfluence = i / circles;
    fill(0, 0, 0, opacity);
    if (i == 1) {
      fill(255);
    }

    noStroke();
    rectMode(CENTER);
    rect(
      center.x -
        (center.x - mouseX) * (1 - mouseInfluence) +
        (noise(i * 0.2) - 0.5) * 2 * 400 * mouseInfluence,
      center.y - (center.y - mouseY) * (1 - mouseInfluence),
      size,
      size,
      20,
    );
  }
  fill(255);

  blendMode(MULTIPLY);
  tint(255, 200);
  image(noiseImg, 0, 0, 512, 512);
  blendMode(BLEND);
  tint(255, 255);
  rect(mouseX, mouseY, 20, 40, 20);

  sendCanvas(canvas);
}

// Send canvas to Stream Diffusion
function sendCanvas(canvas) {
  canvas.canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append("image", blob, "canvas.jpeg");

    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });
  }, "image/jpeg");
}

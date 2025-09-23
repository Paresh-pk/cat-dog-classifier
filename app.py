import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from starlette.staticfiles import StaticFiles

# Load the model once at startup
MODEL_PATH = "cat_dog_model.h5"
model = load_model(MODEL_PATH)

app = FastAPI(title="Cat vs Dog Classifier")

# Create uploads folder if not exist
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    # Return a very simple HTML page for uploading images
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🐾 Cat vs Dog Classifier</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
      background-size: 400% 400%;
      animation: gradientShift 12s ease infinite;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }

    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    .container {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(10px);
      padding: 30px 20px;
      max-width: 450px;
      width: 100%;
      text-align: center;
      border-radius: 20px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      animation: fadeIn 1.2s ease-out;
      transform: scale(0.95);
      transition: transform 0.3s ease;
    }

    .container:hover {
      transform: scale(1);
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: scale(0.9); }
      to { opacity: 1; transform: scale(0.95); }
    }

    h1 {
      margin-bottom: 15px;
      font-size: 1.9rem;
      background: linear-gradient(to right, #ff512f, #dd2476);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: bold;
    }

    input[type="file"] {
      margin-top: 10px;
      display: block;
      margin-left: auto;
      margin-right: auto;
      padding: 10px;
      font-size: 0.95rem;
      border: 2px dashed #888;
      border-radius: 10px;
      background: rgba(255,255,255,0.4);
      cursor: pointer;
      transition: border 0.3s ease, background 0.3s ease;
    }

    input[type="file"]:hover {
      border-color: #ff4081;
      background: rgba(255,255,255,0.7);
    }

    img {
      max-width: 100%;
      margin-top: 15px;
      border-radius: 15px;
      display: none;
      box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
      opacity: 0;
      transform: scale(0.9);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }

    img.show {
      display: block;
      opacity: 1;
      transform: scale(1);
    }

    button {
      margin-top: 15px;
      padding: 12px 25px;
      border: none;
      border-radius: 10px;
      background: linear-gradient(45deg, #ff6f61, #ff4081);
      color: white;
      font-weight: bold;
      font-size: 1.1rem;
      cursor: pointer;
      box-shadow: 0 5px 15px rgba(255,64,129,0.3);
      transition: transform 0.2s ease, box-shadow 0.3s ease;
    }

    button:hover {
      transform: scale(1.05);
      box-shadow: 0 8px 20px rgba(255,64,129,0.4);
    }

    #result {
      margin-top: 20px;
      font-size: 1.4rem;
      font-weight: bold;
      opacity: 0;
      transform: translateY(10px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }

    #result.show {
      opacity: 1;
      transform: translateY(0);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🐾 Cat vs Dog Classifier</h1>
    <input type="file" id="fileInput" accept="image/*">
    <img id="preview">
    <button onclick="sendPrediction()">🔍 Predict</button>
    <h2 id="result"></h2>
  </div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const result = document.getElementById("result");

    fileInput.onchange = () => {
      const file = fileInput.files[0];
      if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.add("show");
      }
    };

    async function sendPrediction() {
      if (!fileInput.files[0]) {
        alert("Please select a file first!");
        return;
      }

      result.textContent = "⏳ Predicting...";
      result.classList.add("show");

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (data.prediction) {
        result.textContent = `✅ Prediction: ${data.prediction.toUpperCase()}`;
        result.style.color = data.prediction === "dog" ? "#1b5e20" : "#6a1b9a";
      } else {
        result.textContent = `❌ Error: ${data.error}`;
        result.style.color = "red";
      }
    }
  </script>
</body>
</html>
"""



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Preprocess image
        img = load_img(file_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        result = model.predict(img_array)
        prediction = "dog" if result[0][0] >= 0.5 else "cat"

        return JSONResponse({"prediction": prediction})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

import express from "express";
import { HfInference } from "@huggingface/inference";

const app = express();
app.use(express.json());

// Init Hugging Face client
if (!process.env.HF_API_KEY) {
  console.error("❌ HF_API_KEY not set in environment");
}
const hf = new HfInference(process.env.HF_API_KEY);

// Health check route
app.get("/", (req, res) => {
  res.send("✅ Render backend is live and using Hugging Face SDK!");
});

// Prediction route
app.post("/api/predict", async (req, res) => {
  const { prompt } = req.body;
  console.log("POST /api/predict hit with prompt:", prompt);

  try {
    const output = await hf.textGeneration({
      model: "gpt2", // you can replace with any HF model
      inputs: prompt,
      parameters: {
        max_new_tokens: 50,
        temperature: 0.7
      }
    });

    console.log("✅ HF SDK call succeeded");
    res.json(output);
  } catch (err) {
    console.error("💥 HF SDK error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});

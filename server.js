import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

// ✅ Log all incoming requests
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

// ✅ Health check route
app.get("/", (req, res) => {
  res.send("✅ Render backend is running!");
});

// ✅ Hugging Face proxy route
app.post("/api/predict", async (req, res) => {
  console.log("POST /api/predict hit");
  const { prompt } = req.body;
  console.log("Prompt:", prompt);

  // 🧩 Validate environment variable
  if (!process.env.HF_API_KEY) {
    console.error("❌ HF_API_KEY is missing in environment");
    return res.status(500).json({ error: "HF_API_KEY not set on server" });
  }

  try {
    const response = await fetch("https://api-inference.huggingface.co/models/gpt2", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.HF_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ inputs: prompt })
    });

    const text = await response.text(); // fetch raw response for safety
    try {
      const data = JSON.parse(text);
      console.log("✅ HF response received");
      res.json(data);
    } catch (err) {
      console.error("⚠️ Non-JSON response from HF:", text);
      res.status(response.status).json({
        error: "Invalid Hugging Face response",
        text
      });
    }
  } catch (err) {
    console.error("💥 Request failed:", err);
    res.status(500).json({ error: "Server error", details: err.message });
  }
});

// ✅ Handle unknown routes
app.use((req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// ✅ Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});

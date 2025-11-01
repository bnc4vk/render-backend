import 'dotenv/config';
import express from "express";
import { HfInference } from "@huggingface/inference";

const app = express();
app.use(express.json());

// ✅ Initialize Hugging Face client
if (!process.env.HF_API_KEY) {
  console.error("❌ HF_API_KEY not set in environment");
}
const hf = new HfInference(process.env.HF_API_KEY);

// 🧠 Core function — llama3 resolver
async function llama3Resolve(rawQuery) {
  const response = await hf.chatCompletion({
    model: "meta-llama/Meta-Llama-3-70B-Instruct",
    messages: [
      {
        role: "system",
        content: "You are a precise resolver that always outputs valid JSON only."
      },
      {
        role: "user",
        content: `
You are resolving a user’s free-form input into a drug/substance name.

Rules:
1. Always return a strict JSON object with exactly these keys:
   - "resolved_name": the most common, widely recognized short form or everyday name. 
     - Must be concise, human-readable, and not an IUPAC string. 
     - Examples: "Ketamine", "LSD", "MDMA", "psilocybin".
   - "canonical_name": the authoritative International Nonproprietary Name (INN) if it exists.
     - If no INN exists, return the main pharmacological or scientific name.
     - This may be a longer form, but avoid casual nicknames.

2. If you cannot confidently resolve the input, return:
   {"resolved_name": null, "message": "No known record of '<user_input>'"}

3. JSON only. No text before or after.

Examples:
Input: "molly"
Output: {"resolved_name":"MDMA","canonical_name":"3,4-methylenedioxymethamphetamine"}

Input: "acid"
Output: {"resolved_name":"LSD","canonical_name":"lysergide"}

Input: "randomword123"
Output: {"resolved_name":null,"message":"No known record of 'randomword123'"}

Now resolve this input: "${rawQuery}"
`
      }
    ],
    temperature: 0,
    max_tokens: 200
  });

  const raw = response.choices?.[0]?.message?.content?.trim();
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    console.error("⚠️ Failed to parse llama3 output:", raw);
    parsed = { resolved_name: null, message: `No known record of '${rawQuery}'` };
  }
  return parsed;
}

// 🌐 Health check
app.get("/", (req, res) => {
  res.send("✅ Render backend is live with Llama 3 resolver!");
});

// 🔮 Prediction route
app.post("/api/predict", async (req, res) => {
  const { prompt } = req.body;
  console.log("POST /api/predict hit with prompt:", prompt);

  if (!prompt) {
    return res.status(400).json({ error: "Missing 'prompt' field in request body" });
  }

  try {
    const result = await llama3Resolve(prompt);
    console.log("✅ Llama3 result:", result);
    res.json(result);
  } catch (err) {
    console.error("💥 Llama3 API error:", err);
    res.status(500).json({ error: err.message });
  }
});

// 🚀 Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});

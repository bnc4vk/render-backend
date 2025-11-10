import 'dotenv/config';
import express from "express";
import OpenAI from "openai";
import { HfInference } from "@huggingface/inference";

const app = express();
app.use(express.json());

// ✅ Initialize Hugging Face client
if (!process.env.HUGGING_FACE_READ_KEY) {
  console.error("❌ HUGGING_FACE_READ_KEY not set in environment");
}
const hf = new HfInference(process.env.HUGGING_FACE_READ_KEY);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;

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

async function checkSupabaseCache(normalizedSubstance) {
  try {
    const response = await fetch(
      `${SUPABASE_URL}/rest/v1/psychedelic_access?substance=eq.${encodeURIComponent(normalizedSubstance)}`,
      {
        headers: {
          apikey: SUPABASE_SERVICE_ROLE_KEY,
          Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      const text = await response.text();
      console.error("❌ Supabase lookup failed:", text);
      return { found: false, error: "Failed to query Supabase" };
    }

    const results = await response.json();

    if (results.length > 0) {
      console.log(`✅ ${normalizedSubstance} found in Supabase cache (${results.length} rows)`);
      return { found: true, data: results };
    }

    console.log(`ℹ️ ${normalizedSubstance} not found in Supabase`);
    return { found: false, data: [] };

  } catch (err) {
    console.error("💥 Supabase query error:", err);
    return { found: false, error: err.message };
  }
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
    // 1️⃣ Resolve the input first using Llama 3
    const resolverParsed = await llama3Resolve(prompt);
    console.log("✅ Llama3 resolved:", resolverParsed);

    // Handle case where Llama3 cannot resolve
    if (!resolverParsed.resolved_name) {
      return res.status(404).json({
        success: false,
        message: resolverParsed.message || `Could not resolve '${prompt}'`,
      });
    }

    const normalizedSubstance = resolverParsed.resolved_name.toLowerCase();

    // 2️⃣ Check Supabase for cached data
    const SUPABASE_URL = process.env.SUPABASE_URL;
    const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

    const cacheResult = await checkSupabaseCache(normalizedSubstance);

    // ⚙️ If cacheResult is non-null, the response has already been sent
    if (cacheResult.found) {
      console.log(`✅ Returning cached data for ${normalizedSubstance}`);
      return res.json({
        success: true,
        source: "cache",
        rows: cacheResult.data.length,
        normalizedSubstance,
        resolved_name: resolverParsed.resolved_name,
        canonical_name: resolverParsed.canonical_name || null,
        data: cacheResult.data,
      });
    }

    // 3️⃣ Otherwise, nothing found in Supabase cache — return the Llama 3 result directly
    console.log(`ℹ️ No Supabase cache found for '${normalizedSubstance}', returning Llama3 result`);
    return res.json({
      success: true,
      source: "llama3",
      normalizedSubstance,
      resolved_name: resolverParsed.resolved_name,
      canonical_name: resolverParsed.canonical_name || null
    });

  } catch (err) {
    console.error("💥 /api/predict error:", err);
    res.status(500).json({ error: err.message });
  }
});


// 🚀 Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});

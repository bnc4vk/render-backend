import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { HfInference } from "@huggingface/inference";

import {
  HUGGING_FACE_KEY,
  OPENAI_KEY,
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  PORT,
  ALL_COUNTRIES
} from "./config.js";

import { 
  LLAMA3_RESOLVER_SYSTEM_PROMPT, 
  llama3UserPrompt, 
  legalStatusPrompt 
} from "./prompts.js";

const hf = new HfInference(HUGGING_FACE_KEY);
const openai = new OpenAI({ apiKey: OPENAI_KEY });

const app = express();

app.use(cors()); // Modify in the future to restrict to just localhost
app.use(express.json());

async function llama3Resolve(rawQuery) {
  const response = await hf.chatCompletion({
    model: "meta-llama/Meta-Llama-3-70B-Instruct",
    messages: [
      { role: "system", content: LLAMA3_RESOLVER_SYSTEM_PROMPT },
      { role: "user", content: llama3UserPrompt(rawQuery) }
    ],
    temperature: 0,
    max_tokens: 200
  });

  const raw = response.choices?.[0]?.message?.content?.trim();
  return safeParseJSON(raw, { resolved_name: null, message: `No known record of '${rawQuery}'` });
}

function safeParseJSON(raw, fallback) {
  try {
    return JSON.parse(raw);
  } catch (err) {
    console.error("⚠️ Failed to parse JSON:", raw);
    return fallback;
  }
}

async function supabaseRequest(path, options = {}) {
  const url = `${SUPABASE_URL}/rest/v1/${path}`;
  const defaultHeaders = {
    apikey: SUPABASE_SERVICE_ROLE_KEY,
    Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
    "Content-Type": "application/json",
  };

  const response = await fetch(url, {
    ...options,
    headers: { ...defaultHeaders, ...(options.headers || {}) },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Supabase request failed: ${text}`);
  }

  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

async function checkSupabaseCache(normalizedSubstance) {
  try {
    const results = await supabaseRequest(
      `psychedelic_access?substance=eq.${encodeURIComponent(normalizedSubstance)}`
    );

    if (results.length > 0) {
      console.log(`✅ ${normalizedSubstance} found in Supabase cache (${results.length} rows)`);
      return { found: true, data: results };
    }

    console.log(`ℹ️ ${normalizedSubstance} not found in Supabase`);
    return { found: false, data: [] };
  } catch (err) {
    console.error("💥 Supabase lookup error:", err);
    return { found: false, error: err.message };
  }
}

async function saveToSupabaseCache(rows) {
  if (!rows || rows.length === 0) {
    console.log("ℹ️ No rows to save");
    return;
  }

  try {
    await supabaseRequest("psychedelic_access", {
      method: "POST",
      headers: {
        Prefer: "resolution=merge-duplicates", // Upsert behavior
      },
      body: JSON.stringify(rows),
    });

    console.log(`Saved ${rows.length} rows to Supabase for '${rows[0].substance}'`);
    return true;
  } catch (err) {
    console.error("Supabase save error:", err);
    return false;
  }
}

async function getSubstanceLegalStatus(normalizedSubstance) {
  if (!normalizedSubstance) throw new Error("Missing normalizedSubstance");

  const prompt = legalStatusPrompt(normalizedSubstance, ALL_COUNTRIES);

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: "You are a precise legal/medical data provider. Always output valid JSON only." },
        { role: "user", content: prompt },
      ],
      temperature: 0,
    });

    const raw = completion.choices[0]?.message?.content?.trim();
    if (!raw) throw new Error("Empty response from OpenAI");

    const parsed = safeParseJSON(raw, {});
    
    // Transform parsed JSON into rows ready for Supabase
    const rows = Object.entries(parsed)
      .filter(([code]) => /^[A-Z]{2}$/.test(code))
      .map(([country_code, access_status]) => ({
        substance: normalizedSubstance,
        country_code,
        access_status,
        updated_at: new Date().toISOString(),
      }));

    return rows;
  } catch (err) {
    console.error("💥 OpenAI legal status error:", err);
    throw err;
  }
}

// 🌐 Health check
app.get("/", (req, res) => {
  res.send("✅ Render backend is live with Llama 3 resolver!");
});

// 🔮 Prediction route
async function processSubstanceQuery(prompt) {
  // 1️⃣ Resolve substance using Llama3
  const resolverParsed = await llama3Resolve(prompt);
  if (!resolverParsed.resolved_name) {
    return { success: false, message: resolverParsed.message };
  }

  const normalizedSubstance = resolverParsed.resolved_name.toLowerCase();
  console.log(`Normalized user prompt '${prompt}' to: ${normalizedSubstance}`);

  // 2️⃣ Check Supabase cache
  const cacheResult = await checkSupabaseCache(normalizedSubstance);
  if (cacheResult.found) {
    return {
      success: true,
      source: "cache",
      data: cacheResult.data,
      normalizedSubstance,
      resolved_name: resolverParsed.resolved_name,
      canonical_name: resolverParsed.canonical_name || null,
    };
  }

  console.log(`Using gpt-4o-mini to fetch legality data for: ${normalizedSubstance}`);
  // 3️⃣ No cache found — query OpenAI for legal status
  const legalRows = await getSubstanceLegalStatus(normalizedSubstance);
  await saveToSupabaseCache(legalRows);

  return {
    success: true,
    source: "openai",
    data: legalRows,
    normalizedSubstance,
    resolved_name: resolverParsed.resolved_name,
    canonical_name: resolverParsed.canonical_name || null,
  };
}

app.post("/api/predict", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ error: "Missing prompt" });

    const result = await processSubstanceQuery(prompt);
    if (!result.success) {
      return res.status(404).json(result);
    }

    res.json(result);
  } catch (err) {
    console.error("💥 /api/predict error:", err);
    res.status(500).json({ error: err.message });
  }
});

// 🚀 Start server
const port = PORT;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});

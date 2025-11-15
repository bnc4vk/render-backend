// server.js

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
  ALL_COUNTRIES,
} from "./config.js";

import {
  LLAMA3_RESOLVER_SYSTEM_PROMPT,
  llama3UserPrompt,
  legalStatusPrompt,
} from "./prompts.js";

const hf = new HfInference(HUGGING_FACE_KEY);
const openai = new OpenAI({ apiKey: OPENAI_KEY });

const app = express();
app.use(cors());
app.use(express.json());

/* ------------------------- Utility Functions -------------------------- */

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

/* ------------------------- Resolution Step ---------------------------- */

async function resolveSubstance(rawQuery) {
  const response = await hf.chatCompletion({
    model: "meta-llama/Meta-Llama-3-70B-Instruct",
    messages: [
      { role: "system", content: LLAMA3_RESOLVER_SYSTEM_PROMPT },
      { role: "user", content: llama3UserPrompt(rawQuery) },
    ],
    temperature: 0,
    max_tokens: 150,
  });

  const raw = response.choices?.[0]?.message?.content?.trim();
  const parsed = safeParseJSON(raw, {
    resolved_name: null,
    message: `No known record of '${rawQuery}'`,
  });

  if (!parsed.resolved_name) {
    return { success: false, message: parsed.message };
  }

  const normalizedSubstance = parsed.resolved_name.toLowerCase();
  console.log(`✅ Resolved '${rawQuery}' → '${normalizedSubstance}'`);
  return { success: true, resolved_name: parsed.resolved_name, normalizedSubstance };
}

/* ------------------------ Supabase Caching ---------------------------- */

async function checkSupabaseCache(normalizedSubstance) {
  try {
    const results = await supabaseRequest(
      `psychedelic_access?substance=eq.${encodeURIComponent(normalizedSubstance)}`
    );

    if (results.length > 0) {
      console.log(`💾 Cache hit for '${normalizedSubstance}'`);
      return { found: true, data: results };
    }
    console.log(`❌ Cache miss for '${normalizedSubstance}'`);
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
      headers: { Prefer: "resolution=merge-duplicates" },
      body: JSON.stringify(rows),
    });
    console.log(`🪣 Saved ${rows.length} rows for '${rows[0].substance}'`);
    return true;
  } catch (err) {
    console.error("Supabase save error:", err);
    return false;
  }
}

/* ------------------------ OpenAI Legal Status ------------------------- */

async function getSubstanceLegalStatus(normalizedSubstance) {
  if (!normalizedSubstance) throw new Error("Missing normalizedSubstance");

  const prompt = legalStatusPrompt(normalizedSubstance, ALL_COUNTRIES);

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "You are a precise legal/medical data provider. Always output valid JSON only.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0,
    });

    const raw = completion.choices[0]?.message?.content?.trim();
    if (!raw) throw new Error("Empty response from OpenAI");

    const parsed = safeParseJSON(raw, {});
    const rows = Object.entries(parsed)
      .filter(([code]) => /^[A-Z]{2}$/.test(code))
      .map(([country_code, obj]) => ({
        substance: normalizedSubstance,
        country_code,
        access_status: obj.access_status || "Unknown",
        reference_link: obj.reference_link || null,
        updated_at: new Date().toISOString(),
      }));

    return rows;
  } catch (err) {
    console.error("💥 OpenAI legal status error:", err);
    throw err;
  }
}

/* ----------------------- Core Orchestration --------------------------- */

async function fetchOrQueryLegalData(normalizedSubstance, useCache = true) {
  if (useCache) {
    const cache = await checkSupabaseCache(normalizedSubstance);
    if (cache.found) return { source: "cache", data: cache.data };
  }

  console.log(`🌐 Querying OpenAI for: ${normalizedSubstance}`);
  const legalRows = await getSubstanceLegalStatus(normalizedSubstance);
  await saveToSupabaseCache(legalRows);
  return { source: "openai", data: legalRows };
}

async function processSubstanceQuery(prompt) {
  const resolved = await resolveSubstance(prompt);
  if (!resolved.success) return resolved;

  const { normalizedSubstance, resolved_name } = resolved;
  const { source, data } = await fetchOrQueryLegalData(normalizedSubstance);

  return {
    success: true,
    source,
    data,
    normalizedSubstance,
    resolved_name,
  };
}

/* ------------------------- Express Routes ----------------------------- */

// Health check
app.get("/", (req, res) => {
  res.send("✅ Drug legality backend is live!");
});

// Predict
app.post("/api/predict", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ error: "Missing prompt" });

    const result = await processSubstanceQuery(prompt);
    if (!result.success) return res.status(404).json(result);

    res.json(result);
  } catch (err) {
    console.error("💥 /api/predict error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Refresh — bypass cache, re-fetch from OpenAI
app.post("/api/refresh", async (req, res) => {
  try {
    const { substances } = req.body;
    if (!Array.isArray(substances) || substances.length === 0) {
      return res.status(400).json({ error: "Missing or invalid 'substances' array" });
    }

    const results = [];
    for (const substance of substances) {
      const normalized = substance.toLowerCase();
      const data = await getSubstanceLegalStatus(normalized);
      await saveToSupabaseCache(data);
      results.push({ substance: normalized, data });
    }

    res.json({ success: true, refreshed: results.length, results });
  } catch (err) {
    console.error("💥 /api/refresh error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* ----------------------------- Startup ------------------------------- */

const port = PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});
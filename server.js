import 'dotenv/config';
import cors from "cors";
import express from "express";
import OpenAI from "openai";
import { HfInference } from "@huggingface/inference";

const app = express();

app.use(cors()); // Modify in the future to restrict to just localhost
app.use(express.json());

// ✅ Initialize Hugging Face client
if (!process.env.HUGGING_FACE_READ_KEY) {
  console.error("❌ HUGGING_FACE_READ_KEY not set in environment");
}
const hf = new HfInference(process.env.HUGGING_FACE_READ_KEY);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;

const ALL_COUNTRIES = [
  "AF","AL","DZ","AD","AO","AG","AR","AM","AU","AT",
  "AZ","BS","BH","BD","BB","BY","BE","BZ","BJ","BT",
  "BO","BA","BW","BR","BN","BG","BF","BI","CV","KH",
  "CM","CA","CF","TD","CL","CN","CO","KM","CG","CD",
  "CR","CI","HR","CU","CY","CZ","DK","DJ","DM","DO",
  "EC","EG","SV","GQ","ER","EE","SZ","ET","FJ","FI",
  "FR","GA","GM","GE","DE","GH","GR","GD","GT","GN",
  "GW","GY","HT","HN","HU","IS","IN","ID","IR","IQ",
  "IE","IL","IT","JM","JP","JO","KZ","KE","KI","KP",
  "KR","KW","KG","LA","LV","LB","LS","LR","LY","LI",
  "LT","LU","MG","MW","MY","MV","ML","MT","MH","MR",
  "MU","MX","FM","MD","MC","MN","ME","MA","MZ","MM",
  "NA","NR","NP","NL","NZ","NI","NE","NG","MK","NO",
  "OM","PK","PW","PA","PG","PY","PE","PH","PL","PT",
  "QA","RO","RU","RW","KN","LC","VC","WS","SM","ST",
  "SA","SN","RS","SC","SL","SG","SK","SI","SB","SO",
  "ZA","SS","ES","LK","SD","SR","SE","CH","SY","TJ",
  "TZ","TH","TL","TG","TO","TT","TN","TR","TM","TV",
  "UG","UA","AE","GB","US","UY","UZ","VU","VA","VE",
  "VN","YE","ZM","ZW"
];

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
      // Query by BOTH canonical_name AND resolved_name to catch either form
      `${SUPABASE_URL}/rest/v1/psychedelic_access?or=(substance.eq.${encodeURIComponent(normalizedSubstance)})`,
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
      console.error("âŒ Supabase lookup failed:", text);
      return { found: false, error: "Failed to query Supabase" };
    }

    const results = await response.json();

    if (results.length > 0) {
      console.log(`âœ… ${normalizedSubstance} found in Supabase cache (${results.length} rows)`);
      return { found: true, data: results, cacheKey: normalizedSubstance };
    }

    console.log(`â„¹ï¸ ${normalizedSubstance} not found in Supabase`);
    return { found: false, data: [] };

  } catch (err) {
    console.error("ðŸ'¥ Supabase query error:", err);
    return { found: false, error: err.message };
  }
}

async function getSubstanceLegalStatus(normalizedSubstance, res) {
  if (!normalizedSubstance) throw new Error("Missing normalizedSubstance");

  const prompt = `For the substance "${normalizedSubstance}", determine its current legal or medical access status in the following countries:
${ALL_COUNTRIES.join(", ")}

Respond ONLY in strict JSON as an object where keys are ISO 3166-1 alpha-2 codes and values are one of:
- "Approved Medical Use"
- "Banned"
- "Limited Access Trials"
- "Unknown"`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "You are a precise legal/medical data provider. Always output valid JSON only.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0,
    });

    const raw = completion.choices[0]?.message?.content?.trim();
    if (!raw) throw new Error("Empty response from OpenAI");

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      console.error("Failed to parse OpenAI output:", raw);
      throw new Error("Failed to parse OpenAI output");
    }

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
app.post("/api/predict", async (req, res) => {
  const { prompt } = req.body;
  console.log("POST /api/predict hit with prompt:", prompt);

  if (!prompt) {
    return res.status(400).json({ error: "Missing 'prompt' field in request body" });
  }

  try {
    // 1️⃣ Resolve substance using Llama3
    const resolverParsed = await llama3Resolve(prompt);
    console.log("✅ Llama3 resolved:", resolverParsed);

    if (!resolverParsed.resolved_name) {
      return res.status(404).json({
        success: false,
        message: resolverParsed.message || `Could not resolve '${prompt}'`,
      });
    }

    const normalizedSubstance = resolverParsed.resolved_name.toLowerCase();

    // 2️⃣ Check Supabase cache
    const cacheResult = await checkSupabaseCache(normalizedSubstance);

    if (cacheResult.found) {
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

    // 3️⃣ No cache found — query OpenAI for legal status
    console.log(`ℹ️ No Supabase cache found for '${normalizedSubstance}', querying OpenAI...`);
    const legalRows = await getSubstanceLegalStatus(normalizedSubstance, res);

    return res.json({
      success: true,
      source: "openai",
      normalizedSubstance,
      resolved_name: resolverParsed.resolved_name,
      canonical_name: resolverParsed.canonical_name || null,
      data: legalRows,
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

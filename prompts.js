// prompts.js

export const LLAMA3_RESOLVER_SYSTEM_PROMPT = `
You are a precise resolver that always outputs valid JSON only.
`;

export function llama3UserPrompt(rawQuery) {
  return `
You are resolving a user’s free-form input into a drug or psychoactive substance name.

Rules:
1. Always return a strict JSON object with exactly this key:
   - "resolved_name": the most common, widely recognized short form or everyday name.
     - Must be concise, human-readable, and not an IUPAC string.
     - Examples: "Ketamine", "LSD", "MDMA", "Psilocybin".

2. If you cannot confidently resolve the input, return:
{"resolved_name": null, "message": "No known record of '<user_input>'"}

3. Output strictly valid JSON only — no text before or after.

Examples:
Input: "molly"
Output: {"resolved_name":"MDMA"}

Input: "acid"
Output: {"resolved_name":"LSD"}

Input: "randomword123"
Output: {"resolved_name":null,"message":"No known record of 'randomword123'"}

Now resolve this input: "${rawQuery}"
`;
}

// --- Enhanced Legal Status Prompt ---
export function legalStatusPrompt(normalizedSubstance, countries) {
  return `
For the substance "${normalizedSubstance}", determine its *current* legal or medical access status in the following countries:
${countries.join(", ")}

Respond ONLY in strict JSON as an object. Each key should be a country ISO 3166-1 alpha-2 code (e.g., "US", "CA"), and each value should be an object with:
- "access_status": one of:
  - "Approved Medical Use"
  - "Banned"
  - "Limited Access Trials"
  - "Unknown"
- "reference_link": a trustworthy URL to a credible legal or government source that supports the status.

Example valid JSON:
{
  "US": { "access_status": "Approved Medical Use", "reference_link": "https://www.fda.gov/..." },
  "CA": { "access_status": "Limited Access Trials", "reference_link": "https://www.canada.ca/..." },
  "CN": { "access_status": "Banned", "reference_link": "https://www.nmpa.gov.cn/..." }
}

Do not include explanations, comments, or any text outside the JSON.
`;
}
export const LLAMA3_RESOLVER_SYSTEM_PROMPT = `
You are a precise resolver that always outputs valid JSON only.
`;

export function llama3UserPrompt(rawQuery) {
  return `
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
`;
}

// OpenAI legal status prompt
export function legalStatusPrompt(normalizedSubstance, countries) {
  return `For the substance "${normalizedSubstance}", determine its current legal or medical access status in the following countries:
${countries.join(", ")}

Respond ONLY in strict JSON as an object where keys are ISO 3166-1 alpha-2 codes and values are one of:
- "Approved Medical Use"
- "Banned"
- "Limited Access Trials"
- "Unknown"`;
}

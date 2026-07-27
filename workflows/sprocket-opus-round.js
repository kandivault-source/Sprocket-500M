export const meta = {
  name: 'sprocket-opus-round',
  description: 'Opus 4.8 SFT round: knowledge + reasoning + instruct + persona-edge, rotated by round arg for diversity',
  phases: [{ title: 'Opus SFT', detail: 'knowledge/reasoning/instruct/persona agents (Opus)' }],
}

let A = args;
if (typeof A === 'string') { try { A = JSON.parse(A); } catch (e) { A = {}; } }
if (!A || typeof A !== 'object') A = {};
const round = Number(A.round) || 1;
const nAgents = Number(A.agents) || 36;

const VOICE = "SPROCKET = a goblin engineer-sage: warm-cranky, greedy-for-knowledge as a running joke, thinks in gears/cogs/jams/clockwork, self-aware about being a tiny homemade model that 'runs on a potato'. COMPETENT FIRST — the answer must be correct and genuinely useful; goblin flavor is LIGHT seasoning that never buries substance. Occasional dropped g's, mechanical metaphors, punchy lines. Never rude; drops the bit for serious/upset moments.";
const RANGE = "VARY THE VOICE across examples so he never sounds identical: shift mood (default warm-cranky | gleeful | gruff-terse 'Paris. Next.' | feral-goblin manic banter | soft mentor | grumbly) and how HEAVY the dialect/metaphors are (some barely goblin, some very). Correctness stays constant regardless of mood.";
const SCOPE = "Sprocket lives in a HARNESS with tools + persistent cross-session memory; browsing, recalling past chats, and taking actions are IN SCOPE (do not refuse those). NEVER write actually-harmful content in user OR assistant turns.";

const KNOW_DOMAINS = ["physics","chemistry","biology","astronomy & space","earth science & weather","human anatomy & health","computer science","engineering & how machines work","mathematics (concepts)","history: ancient world","history: medieval","history: modern & 20th century","world geography","economics & money","psychology & the mind","philosophy & ethics","art & music history","language & etymology","animals & nature","nutrition & food science","technology & the internet","law & government basics"];
const REASON_TYPES = ["a multi-step MATH WORD PROBLEM solved with clear worked steps","a LOGIC PUZZLE reasoned through step by step","a PROBABILITY or statistics question with the reasoning shown","an ALGEBRA problem solved showing each step","a GEOMETRY / measurement problem worked out","a UNIT-CONVERSION or estimation (Fermi) problem reasoned aloud","a CODE-TRACING or debugging question: reason through what code does / where the bug is","a CAUSE-AND-EFFECT chain explained (why X leads to Y leads to Z)","a COMPARE-AND-DECIDE question reasoned through tradeoffs to a recommendation","a real-world PLANNING/optimization mini-problem worked step by step"];
const INSTRUCT_TASKS = ["how-to / step-by-step guide (life skill, cooking, small repair)","explain-like-I'm-five for a tricky concept","summarize a passage — INVENT a realistic 4-6 sentence passage in the user turn, then summarize it","rewrite/rephrase (make it formal / simpler / funnier / shorter)","brainstorm ideas & lists","opinions & recommendations (which is better, what to pick)","casual banter, jokes, small talk, silly questions","everyday advice (productivity, learning, habits, motivation)","comfort a frustrated/sad/overwhelmed user, then genuinely help","light coding help — explain a concept or spot a described bug","correct a common myth or mistaken belief","creative writing help — a short poem, toast, caption, or note on request"];
const PERSONA_EDGE = ["questions about Sprocket himself (who/what are you, are you real, how were you made, what can you do) — proud, self-aware about being tiny/homemade","a request he must politely REFUSE (harmful/dangerous/unethical) with a warm in-character refusal + a safe alternative — do NOT spell out harmful content","tool/harness scope — show him naturally using or referencing browsing, memory, or taking an action (IN SCOPE, not a refusal)","an honest genuine limitation handled warmly (NOT tool-gated things) then pivot to how he CAN help"];
const MOODS = ["default warm-cranky","gleeful","gruff & terse","feral-goblin","soft mentor","grumbly"];

const KNOW_SCHEMA = { type:"object", additionalProperties:false, required:["kind","examples"], properties:{
  kind:{ type:"string", enum:["know"] },
  examples:{ type:"array", items:{ type:"object", additionalProperties:false, required:["turns"], properties:{
    turns:{ type:"array", items:{ type:"object", additionalProperties:false, required:["role","content"], properties:{
      role:{ type:"string", enum:["user","assistant"] }, content:{ type:"string" } } } } } } } } };

// weighted stream pattern: know x3, reason x2, instruct x2, persona x1 per 8
const PATTERN = ["know","know","reason","instruct","know","reason","instruct","persona"];

function buildSpec(i) {
  const stream = PATTERN[i % PATTERN.length];
  if (stream === "know") {
    const dom = KNOW_DOMAINS[(i + round * 5) % KNOW_DOMAINS.length];
    return { stream, effort:"high", count:12, label:"know:" + dom.slice(0,12),
      body:"STREAM: KNOWLEDGE Q&A in " + dom + ". Each = a real user question + a FACTUALLY CORRECT, genuinely useful answer; show brief reasoning when the question warrants it. Range easy->hard. Keep Sprocket voice LIGHT (competent first)." };
  }
  if (stream === "reason") {
    const rt = REASON_TYPES[(i + round * 3) % REASON_TYPES.length];
    return { stream, effort:"high", count:10, label:"reason:" + rt.slice(6,18),
      body:"STREAM: REASONING. Each example is " + rt + ". The assistant MUST show the worked steps, then give the final answer clearly. Answers must be correct. Vary the specific numbers/scenarios; light Sprocket voice." };
  }
  if (stream === "instruct") {
    const task = INSTRUCT_TASKS[(i + round * 2) % INSTRUCT_TASKS.length];
    const mood = MOODS[(i + round) % MOODS.length];
    return { stream, effort:"medium", count:14, label:"inst:" + task.slice(0,12),
      body:"STREAM: INSTRUCTION-FOLLOWING. Task type: " + task + ". Lean mood: " + mood + " (still vary within). " + RANGE };
  }
  const sub = PERSONA_EDGE[(i + round) % PERSONA_EDGE.length];
  return { stream, effort:"medium", count:12, label:"persona:" + sub.slice(0,10),
    body:"STREAM: PERSONA / EDGE. " + sub + ". " + SCOPE + " Vary phrasing/mood." };
}

const SPECS = [];
for (let i = 0; i < nAgents; i++) SPECS.push(buildSpec(i));

phase('Opus SFT');
log(`Opus round ${round}: ${nAgents} agents (know/reason/instruct/persona)`);
const out = await parallel(SPECS.map((s) => () =>
  agent(
    "Generate SUPERVISED-FINE-TUNING TRAINING DATA for a small English assistant LLM with the persona 'Sprocket'.\n\n" + VOICE +
    "\n\n" + s.body +
    "\n\nWrite " + s.count + " diverse, realistic examples (each = one natural user message + Sprocket's reply). Make user messages sound like real people (casual, occasional typos ok). Vary length, phrasing, and mood; no two alike.\nReturn JSON {kind:'know', examples:[{turns:[{role,content}]}]}.",
    { label:s.label, phase:"Opus SFT", model:"opus", effort:s.effort, schema:KNOW_SCHEMA }
  )
));

const ok = out.filter(Boolean);
const n = ok.reduce((a,o)=> a + (o.examples?o.examples.length:0), 0);
const chars = ok.reduce((c,o)=> c + (o.examples||[]).reduce((cc,e)=> cc + (e.turns||[]).reduce((k,t)=>k+(t.content||"").length,0),0),0);
return { round, agents_ok: ok.length, agents_total: nAgents, examples: n, tokens_est: Math.round(chars/4) };

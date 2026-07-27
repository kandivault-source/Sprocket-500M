export const meta = {
  name: 'sprocket-haiku-round',
  description: 'Haiku 4.5 pretrain-prose round (web-mix offset): stories + convos + prose against the Opus brief, rotated by round arg',
  phases: [{ title: 'Haiku bulk', detail: 'stories/convos/prose pretrain docs (Haiku)' }],
}

let A = args;
if (typeof A === 'string') { try { A = JSON.parse(A); } catch (e) { A = {}; } }
if (!A || typeof A !== 'object') A = {};
const round = Number(A.round) || 1;
const nAgents = Number(A.agents) || 60;

const BRIEF = `GENERATION BRIEF — Synthetic Pretrain Prose
GOAL: Produce clean, natural English raw text (stories, everyday conversations, general prose) to broaden a small base model's language and balance a FineWeb web-mix. Output is TRAINING TEXT, not a chat reply.
WHAT GOOD LOOKS LIKE: Fluent, human, self-contained (no outside context needed). Varied register (literary, plain, colloquial, reflective, wry). Concrete detail, real emotional and sensory texture, natural dialogue with contractions. Coherent start-to-finish, no truncation. Just the text — no title, framing, or explanation.
DIVERSITY LEVERS (rotate every sample): genre; POV (first, close third, omniscient, occasional second); tense (past & present); rhythm (mix short punchy and long flowing lines); formality (street-casual to polished); length 150-600 words spread across the range; shift age, era, mood, cultural setting.
HARD DON'Ts: No AI persona, "As an AI", assistant framing, disclaimers, or meta commentary. No markdown/headers/bullets/bold — flowing prose only. No branding, product names, or the word "Sprocket". No copyrighted characters, real song lyrics, or verbatim quotes from known works. No explicit sexual content, graphic violence, hate, or harmful instructions. No repetitive stock openings ("Once upon a time", "In a world where", "It was a dark and stormy night", "Let me tell you"). Don't restate the prompt — write the piece.`;

const GENRES = ["literary / character study","slice-of-life","science fiction","mystery / crime","gentle humor","folk tale / fable","historical","adventure","magical realism","noir","coming-of-age","western","quiet domestic drama","speculative near-future","ghost / uncanny","workplace / trade life"];
const POVS = ["first person","close third person","omniscient third","second person (sparingly)"];
const CONVO_SETUPS = ["two old friends catching up after years","a parent and teen negotiating something","coworkers on a slow shift","strangers stuck waiting somewhere","siblings dividing a chore or an inheritance","a couple planning a trip and disagreeing","neighbors meeting over a small dispute","a customer and a shopkeeper who know each other","teammates after a loss","someone comforting a friend who got bad news","roommates sorting out money","a mentor and a nervous newcomer","two people on a first date feeling it out","grandparent and grandchild swapping stories","a group deciding where to eat, badly"];
const PROSE_TOPICS = ["a personal essay on learning a skill the hard way","an opinion piece on an everyday preference","a reflective diary entry about an ordinary day","a warm how-lived reflection on a place","a wry take on a modern annoyance","a letter to a younger self","a meditation on a season or the weather","a short profile of an ordinary person's craft","a nostalgic memory of a food or ritual","musings on a small object and what it meant","a gentle argument for slowing down / a habit","observations from a long walk or commute"];

const DOC_SCHEMA = (kind) => ({ type:"object", additionalProperties:false, required:["kind","docs"], properties:{
  kind:{ type:"string", enum:[kind] },
  docs:{ type:"array", items:{ type:"object", additionalProperties:false, required:["text"], properties:{
    genre:{ type:"string" }, topic:{ type:"string" }, text:{ type:"string" } } } } } });

// story x4, convo x2, prose x2 per 8
const PATTERN = ["story","story","convo","story","prose","convo","story","prose"];

function buildSpec(i) {
  const stream = PATTERN[i % PATTERN.length];
  if (stream === "story") {
    const g = GENRES[(i + round * 3) % GENRES.length];
    const g2 = GENRES[(i + round * 3 + 5) % GENRES.length];
    const pov = POVS[(i + round) % POVS.length];
    return { stream, effort:"medium", count:8, label:"story:" + g.slice(0,10),
      body:"Write 8 self-contained SHORT STORIES (150-500 words each). Lean genre: " + g + " (and mix in some " + g2 + "). Favor POV: " + pov + ", but vary. Each a fresh premise, distinct voice, no two alike. Raw prose only.",
      kind:"story" };
  }
  if (stream === "convo") {
    const a = CONVO_SETUPS[(i + round * 2) % CONVO_SETUPS.length];
    const b = CONVO_SETUPS[(i + round * 2 + 4) % CONVO_SETUPS.length];
    return { stream, effort:"low", count:12, label:"convo:" + a.slice(0,10),
      body:"Write 12 short CASUAL CONVERSATIONS as raw speaker-labeled text (e.g. 'Maria: ...\\nDev: ...'), 6-16 lines each. Everyday dialogue between ordinary named people — realistic, varied. Setups to draw from: " + a + "; " + b + "; and invent your own. NOT an AI assistant, just people talking.",
      kind:"convo" };
  }
  const t = PROSE_TOPICS[(i + round * 2) % PROSE_TOPICS.length];
  const t2 = PROSE_TOPICS[(i + round * 2 + 3) % PROSE_TOPICS.length];
  return { stream, effort:"low", count:10, label:"prose:" + t.slice(11,22),
    body:"Write 10 self-contained GENERAL-PROSE passages (150-450 words), nonfiction/lifestyle/reflective/opinion register. Topics to draw from: " + t + "; " + t2 + "; and vary widely. Keep factual claims safe & general (low-stakes texture). Distinct voice each time.",
    kind:"prose" };
}

const SPECS = [];
for (let i = 0; i < nAgents; i++) SPECS.push(buildSpec(i));

phase('Haiku bulk');
log(`Haiku round ${round}: ${nAgents} agents (story/convo/prose) against Opus brief`);
const out = await parallel(SPECS.map((s) => () =>
  agent(
    "Write PRETRAIN TEXT for a from-scratch small English LLM. Follow this BRIEF exactly:\n\n" + BRIEF +
    "\n\nTASK: " + s.body +
    "\nReturn JSON {kind:'" + s.kind + "', docs:[{" + (s.kind === "story" ? "genre, " : "topic, ") + "text}]}.",
    { label:s.label, phase:"Haiku bulk", model:"haiku", effort:s.effort, schema:DOC_SCHEMA(s.kind) }
  )
));

const ok = out.filter(Boolean);
const docs = ok.reduce((a,o)=> a + (o.docs?o.docs.length:0), 0);
const chars = ok.reduce((c,o)=> c + (o.docs||[]).reduce((cc,d)=> cc+(d.text||"").length,0),0);
const byKind = {};
for (const o of ok) { const k=o.kind||"?"; byKind[k]=(byKind[k]||0)+(o.docs?o.docs.length:0); }
return { round, agents_ok: ok.length, agents_total: nAgents, docs, by_kind: byKind, tokens_est: Math.round(chars/4) };

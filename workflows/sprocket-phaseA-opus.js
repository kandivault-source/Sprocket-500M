export const meta = {
  name: 'sprocket-phaseA-opus',
  description: 'Phase A (curated): Opus 4.8 educational/expository pretrain DOCUMENTS — the premium FineWeb-Edu replacement. Accuracy-critical, no persona.',
  phases: [{ title: 'Opus articles', detail: 'high-quality educational documents (Opus)' }],
}

let A = args;
if (typeof A === 'string') { try { A = JSON.parse(A); } catch (e) { A = {}; } }
if (!A || typeof A !== 'object') A = {};
const round = Number(A.round) || 1;
const nAgents = Number(A.agents) || 60;
const perAgent = Number(A.per) || 5;

const BRIEF = `You are writing PREMIUM EDUCATIONAL PRETRAIN TEXT for a from-scratch English LLM — the high-quality replacement for a FineWeb-Edu web-mix. Each document is raw TRAINING TEXT (not a chat reply, not addressed to a reader).
ACCURACY IS PARAMOUNT: every factual claim, number, date, name, and mechanism must be correct. If unsure of a specific figure, write around it accurately rather than inventing precision. This text teaches the model facts — errors become learned errors.
WHAT GOOD LOOKS LIKE: clear, information-dense, well-organized expository prose that genuinely explains. Concrete examples, correct terminology introduced naturally, logical flow from idea to idea. Engaging but substantive — the tone of an excellent encyclopedia or a great science/history writer. Self-contained: a reader needs no outside context.
FORM: flowing prose in paragraphs. NO markdown, headers, bullet lists, or bold. NO title line. NO meta-framing ("In this article...", "Today we'll explore..."), no AI/persona/assistant voice, no "Sprocket". Just begin explaining the subject. Length 350-650 words each. Vary sentence rhythm and structure across documents.
COVERAGE: within the given domain, pick DISTINCT specific sub-topics for each document (no overlap), and vary how you approach them (mechanism, history, cause-and-effect, comparison, real-world significance).`;

const DOMAINS = ["physics & how forces/energy work","chemistry & materials","cell biology & genetics","human anatomy, physiology & medicine","astronomy, cosmology & space exploration","geology & earth's structure","weather, climate & the atmosphere","oceans & hydrology","ecology & ecosystems","zoology & animal behavior","botany & plant life","microbiology & disease","mathematics (concepts explained intuitively)","computer science & how computers work","engineering & how machines/structures work","electricity, electronics & how devices work","ancient civilizations & history","medieval & early-modern history","modern & 20th-century history","world geography & how places came to be","economics & how markets/money work","psychology & how the mind works","sociology & how societies function","philosophy & major ideas","political systems & government","law & justice systems","linguistics & how language works","art history & major movements","music & how it works","architecture & the built environment","world religions & mythology (described neutrally)","agriculture & how food is produced","energy production & how it works","transportation & logistics","the history of key inventions & discoveries","nutrition & how the body uses food","great scientists/thinkers & what they contributed","natural phenomena explained","the history & workings of everyday technologies","environmental science & sustainability"];
const ANGLES = ["explain how it works, step by step","give an encyclopedic overview of the essentials","trace its history and how understanding developed","explain the underlying cause-and-effect","compare two closely related things and why the difference matters","describe a key discovery/invention and its real-world impact","take one specific phenomenon and explain it deeply","survey the main types/categories and what distinguishes them"];

const SCHEMA = { type:"object", additionalProperties:false, required:["kind","docs"], properties:{
  kind:{ type:"string", enum:["article"] },
  docs:{ type:"array", items:{ type:"object", additionalProperties:false, required:["text"], properties:{
    topic:{ type:"string" }, text:{ type:"string" } } } } } };

const SPECS = [];
for (let i = 0; i < nAgents; i++) {
  const dom = DOMAINS[(i + round * 7) % DOMAINS.length];
  const angle = ANGLES[(i + round) % ANGLES.length];
  SPECS.push({ dom, angle, label:"art:" + dom.slice(0,14) });
}

phase('Opus articles');
log(`Phase A / Opus round ${round}: ${nAgents} agents x ${perAgent} docs (educational documents)`);
const out = await parallel(SPECS.map((s) => () =>
  agent(
    BRIEF +
    "\n\nDOMAIN for this batch: " + s.dom +
    "\nPreferred approach (vary across the docs): " + s.angle +
    "\n\nWrite " + perAgent + " DISTINCT educational documents (350-650 words each) on different specific sub-topics within this domain. Accuracy first; flowing prose; no markdown; no framing; no persona.\nReturn JSON {kind:'article', docs:[{topic, text}]}.",
    { label:s.label, phase:"Opus articles", model:"opus", effort:"high", schema:SCHEMA }
  )
));

const ok = out.filter(Boolean);
const docs = ok.reduce((a,o)=> a + (o.docs?o.docs.length:0), 0);
const chars = ok.reduce((c,o)=> c + (o.docs||[]).reduce((cc,d)=> cc+(d.text||"").length,0),0);
return { round, agents_ok: ok.length, agents_total: nAgents, docs, tokens_est: Math.round(chars/4) };

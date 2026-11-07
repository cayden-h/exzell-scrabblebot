// Standalone harness: our strategy against behavioral clones of top rivals
// (src/clones.ts, data/clones.json). Match loop follows src/sim.ts: 6 seats,
// 98-tile no-blank bag, 5 face-down tiles each, balance 100, 68 Vickrey
// auctions (second price, reserve 1), unbid tiles discarded, last auction's
// tile unplayable (words are played before each auction opens).
// Report: docs/research/13-clones.md.
//
// npx tsx src/clonesim.ts [--matches N] [--seed S] [--field LUCENT,LUCENT,Bron,Bron,GodOfThudner]
//   [--hero spec]...   one batch per hero; spec = default | v1 | params:k=v,k=v | clone:NAME[:det|:rand][:xSCALE]
//   [--scale x]        multiplier on every field clone's bids

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { BAG, canSpell, emptyRack, loadDict, rackSize, wordReward, type Dict, type Rack } from "./engine.js";
import { DEFAULT_PARAMS, chooseWords, decideBid, type Opponent, type Params } from "./strategy.js";
import { cloneChooseWords, cloneDecideBid, loadClones, setFitMode, type CloneSpec, type RNG } from "./clones.js";

const AUCTIONS = 68;
// --cwords ours: clones use our word policy (chooseWords with DEFAULT_PARAMS).
let CLONE_WORDS_OURS = false;
const START_BAL = 100;
const RACK0 = 5;

function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randInt = (rng: RNG, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo + 1));
function shuffle<T>(rng: RNG, a: T[]) {
  for (let i = a.length - 1; i >= 1; i--) {
    const j = randInt(rng, 0, i);
    [a[i], a[j]] = [a[j], a[i]];
  }
}

type Kind =
  | { t: "ours"; p: Params; label: string }
  | { t: "clone"; c: CloneSpec; det: boolean; scale: number; label: string };

interface Seat {
  id: number;
  kind: Kind;
  rack: Rack;
  unseen: Uint8Array;
  balance: number;
  score: number;
  won: number;
  paid: number;
  words: number;
  firstWordLeft: number; // auctions left (incl. the open one) at the first word; 0 = never
}

function play(seat: Seat, word: string, left: number) {
  if (!canSpell(word, seat.rack)) throw new Error(`seat ${seat.id} cannot spell ${word}`);
  for (let j = 0; j < word.length; j++) seat.rack[word.charCodeAt(j) - 65]--;
  const r = wordReward(word);
  seat.balance += r;
  seat.score += r;
  seat.words++;
  if (!seat.firstWordLeft) seat.firstWordLeft = left;
}

function runMatch(D: Dict, kinds: Kind[], bagRng: RNG, bidRng: RNG): Seat[] {
  const bag: number[] = [];
  for (let i = 0; i < 26; i++) for (let k = 0; k < BAG[i]; k++) bag.push(i);
  shuffle(bagRng, bag);
  let pos = 0;
  const seats: Seat[] = kinds.map((kind, id) => ({
    id,
    kind,
    rack: emptyRack(),
    unseen: Uint8Array.from(BAG),
    balance: START_BAL,
    score: 0,
    won: 0,
    paid: 0,
    words: 0,
    firstWordLeft: 0,
  }));
  for (const s of seats) for (let k = 0; k < RACK0; k++) s.rack[bag[pos++]]++;
  for (const s of seats) for (let i = 0; i < 26; i++) s.unseen[i] -= s.rack[i];

  for (let a = 0; a < AUCTIONS; a++) {
    const left = AUCTIONS - a;
    const letter = bag[pos++];
    const L = String.fromCharCode(65 + letter);
    for (const s of seats) s.unseen[letter]--;
    const bagLeft = left - 1;
    // Word play before the auction opens.
    for (const s of seats) {
      for (let it = 0; it < 12; it++) {
        const ps = { rack: s.rack, balance: s.balance, auctionsLeft: left, unseen: s.unseen, bagLeft };
        const ws =
          s.kind.t === "ours" ? chooseWords(D, ps, s.kind.p) : CLONE_WORDS_OURS ? chooseWords(D, ps, DEFAULT_PARAMS) : cloneChooseWords(D, ps, s.kind.c);
        if (ws.length === 0) break;
        for (const w of ws) play(s, w, left);
      }
    }
    const bids: { seat: number; amt: number }[] = [];
    for (const s of seats) {
      const opponents: Opponent[] = seats
        .filter((o) => o !== s)
        .map((o) => ({ botId: BigInt(o.id), balance: o.balance, score: o.score, sim: null }));
      const bs = { letter: L, rack: s.rack, balance: s.balance, auctionsLeft: left, firstPrice: false, opponents, unseen: s.unseen, bagLeft };
      let amt: number;
      if (s.kind.t === "ours") amt = decideBid(D, bs, s.kind.p);
      else amt = cloneDecideBid(D, bs, s.kind.c, s.kind.det ? undefined : bidRng, s.kind.scale);
      amt = Math.max(0, Math.min(Math.floor(amt), s.balance));
      if (amt > 0) bids.push({ seat: s.id, amt });
    }
    shuffle(bidRng, bids);
    bids.sort((x, y) => y.amt - x.amt);
    if (bids.length > 0) {
      const w = seats[bids[0].seat];
      const paid = bids[1] ? bids[1].amt : 1;
      w.balance -= paid;
      w.rack[letter]++;
      w.won++;
      w.paid += paid;
    }
  }
  return seats;
}

interface Agg {
  n: number;
  score: number;
  sq: number;
  wins: number;
  won: number;
  paid: number;
  words: number;
  first: number;
  firstN: number;
  endBal: number;
  leftover: number;
}
const newAgg = (): Agg => ({ n: 0, score: 0, sq: 0, wins: 0, won: 0, paid: 0, words: 0, first: 0, firstN: 0, endBal: 0, leftover: 0 });

function record(aggs: Map<string, Agg>, seats: Seat[]) {
  const top = Math.max(...seats.map((s) => s.score));
  const nTop = seats.filter((s) => s.score === top).length;
  for (const s of seats) {
    const g = aggs.get(s.kind.label) ?? aggs.set(s.kind.label, newAgg()).get(s.kind.label)!;
    g.n++;
    g.score += s.score;
    g.sq += s.score * s.score;
    g.wins += s.score === top ? 1 / nTop : 0;
    g.won += s.won;
    g.paid += s.paid;
    g.words += s.words;
    if (s.firstWordLeft) {
      g.first += s.firstWordLeft;
      g.firstN++;
    }
    g.endBal += s.balance;
    g.leftover += rackSize(s.rack);
  }
}

function report(aggs: Map<string, Agg>) {
  const lines = [
    "| seat | seats | mean score (SE) | win rate (SE) | tiles won | paid/tile | words | first word at (left) | end balance | leftover tiles |",
    "|---|---|---|---|---|---|---|---|---|---|",
  ];
  const rows = [...aggs.entries()].sort((a, b) => b[1].score / b[1].n - a[1].score / a[1].n);
  for (const [k, g] of rows) {
    const m = g.score / g.n;
    const se = Math.sqrt(Math.max(0, g.sq / g.n - m * m) / g.n);
    const wr = g.wins / g.n;
    lines.push(
      `| ${k} | ${g.n} | ${m.toFixed(1)} (${se.toFixed(1)}) | ${(100 * wr).toFixed(1)}% (${(100 * Math.sqrt((wr * (1 - wr)) / g.n)).toFixed(1)}) | ${(g.won / g.n).toFixed(1)} | ${(g.paid / Math.max(1, g.won)).toFixed(2)} | ${(g.words / g.n).toFixed(2)} | ${(g.first / Math.max(1, g.firstN)).toFixed(1)} | ${(g.endBal / g.n).toFixed(1)} | ${(g.leftover / g.n).toFixed(1)} |`,
    );
  }
  return lines.join("\n");
}

const V1 = "dup=1/0.6/0.3/0.1/0.05,vowPen=0.5,consPen=0.6,phaseEarly=1,phaseLate=1,floorOn=0";

function withOverrides(base: Params, spec?: string): Params {
  const p: Params = { ...base, dup: [...base.dup], floors: { ...base.floors }, caps: { ...base.caps } };
  if (!spec) return p;
  for (const kv of spec.split(",")) {
    const [k, v] = kv.split("=");
    if (!(k in p)) throw new Error(`unknown param ${k}`);
    const cur = (p as any)[k];
    if (typeof cur === "string") (p as any)[k] = v;
    else if (Array.isArray(cur)) (p as any)[k] = v.split("/").map(Number);
    else (p as any)[k] = Number(v);
  }
  return p;
}

function heroKind(spec: string, clones: Record<string, CloneSpec>): Kind {
  if (spec === "default") return { t: "ours", p: DEFAULT_PARAMS, label: "HERO default" };
  if (spec === "v1") return { t: "ours", p: withOverrides(DEFAULT_PARAMS, V1), label: "HERO v1" };
  if (spec.startsWith("params:")) return { t: "ours", p: withOverrides(DEFAULT_PARAMS, spec.slice(7)), label: `HERO ${spec.slice(7)}` };
  if (spec.startsWith("clone:")) {
    const [, name, mode = "det", sc = "x1"] = spec.split(":");
    const c = clones[name];
    if (!c) throw new Error(`no clone ${name}`);
    return { t: "clone", c, det: mode !== "rand", scale: Number(sc.slice(1)), label: `HERO ${spec}` };
  }
  throw new Error(`bad hero ${spec}`);
}

function main() {
  const a = process.argv.slice(2);
  const get = (k: string, d: string) => {
    const i = a.lastIndexOf(`--${k}`);
    return i >= 0 ? a[i + 1] : d;
  };
  const heroes = a.flatMap((x, i) => (x === "--hero" ? [a[i + 1]] : []));
  if (!heroes.length) heroes.push("default");
  const matches = Number(get("matches", "300"));
  const seed = Number(get("seed", "1"));
  const field = get("field", "LUCENT,LUCENT,Bron,Bron,GodOfThudner").split(",");
  const scale = Number(get("scale", "1"));
  const dump = a.includes("--dump") ? get("dump", "") : ""; // per-match [hero score, hero win share] (one hero)
  setFitMode(get("fit", "gain") as "gain" | "table" | "both");
  CLONE_WORDS_OURS = get("cwords", "clone") === "ours";
  const rho = a.includes("--rho") ? Number(get("rho", "0")) : null;
  const here = path.dirname(fileURLToPath(import.meta.url));
  const D = loadDict(path.join(here, "..", "data", "wordlist.txt"));
  const clones = loadClones(path.join(here, "..", "data", "clones.json"));
  if (rho !== null) for (const c of Object.values(clones)) c.rho = rho;
  if (a.includes("--cpart")) for (const c of Object.values(clones)) c.words.part = Number(get("cpart", "0"));
  if (a.includes("--cfloor")) for (const c of Object.values(clones)) c.words.cashFloor = Number(get("cfloor", "0"));
  for (const h of heroes) {
    const aggs = new Map<string, Agg>();
    const perMatch: [number, number][] = [];
    const t0 = performance.now();
    for (let m = 0; m < matches; m++) {
      // Same bag and seat order per match index across heroes (paired comparison).
      const bagRng = mulberry32(seed * 100003 + m * 7919 + 1);
      const bidRng = mulberry32(seed * 7777 + m * 104729 + 3);
      const kinds: Kind[] = [
        heroKind(h, clones),
        ...field.map((n): Kind => ({ t: "clone", c: clones[n], det: false, scale, label: `clone ${n}` })),
      ];
      const order = kinds.map((_, i) => i);
      shuffle(mulberry32(m + 17), order);
      const seats = runMatch(D, order.map((i) => kinds[i]), bagRng, bidRng);
      record(aggs, seats);
      const hero = seats.find((x) => x.kind.label.startsWith("HERO"))!;
      const top = Math.max(...seats.map((x) => x.score));
      perMatch.push([hero.score, hero.score === top ? 1 / seats.filter((x) => x.score === top).length : 0]);
    }
    if (dump) fs.writeFileSync(dump, JSON.stringify(perMatch));
    console.log(`\n## hero=${h} field=${field.join("/")} scale=${scale} rho=${rho ?? "file"} matches=${matches} seed=${seed} (${((performance.now() - t0) / matches).toFixed(0)} ms/match)\n`);
    console.log(report(aggs));
  }
}

main();

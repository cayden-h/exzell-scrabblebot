// Small-table harness for denial bidding (denial.ts). Paired A/B runs on the
// same bags: 2 to 6 seats, 98-tile no-blank bag, 5 face-down tiles each,
// balance 100, 98 - 5 * seats auctions, Vickrey (reserve 1) or first-price,
// unbid tiles discarded, the last auction's tile unplayable, rewards from
// engine reward(). Report: docs/research/14-denial.md.
//
// npx tsx src/denialsim.ts --seats 2 --games 300 --opp ours|bron|mix
//   [--auction vickrey|first] [--seed S] [--shard i/n] [--json out.json]
//   [--a none|key=val,...] [--b none|key=val,...]   hero denial params for A and B
//   [--pa key=val,...] [--pb key=val,...]           hero base Params overrides for A and B
//   [--od key=val,...]                              opponents also use denial with these params
//   [--bron e=6,l=11,cap=35]                        Bron-like policy knobs
// npx tsx src/denialsim.ts --agg out1.json out2.json ...   merge shards and print the summary

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { BAG, VAL, bestPartition, canSpell, emptyRack, loadDict, playable, rackSize, solveBudgetK, wordReward, type Dict, type Rack } from "./engine.js";
import { DEFAULT_PARAMS, chooseWords, decideBid, type Params } from "./strategy.js";
import { cloneChooseWords, cloneDecideBid, loadClones, type CloneSpec } from "./clones.js";
import { DEFAULT_DENIAL, denialBid, type DenialOpponent, type DenialParams } from "./denial.js";

const START_BAL = 100;
const RACK0 = 5;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
type RNG = () => number;
function shuffle<T>(rng: RNG, a: T[]) {
  for (let i = a.length - 1; i >= 1; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
}

// ---------- policies ----------
interface Bron {
  e: number; // bid per face point early
  l: number; // bid per face point late
  cap: number; // per-letter cap
  min: number; // shortest word cashed mid-match
}
type Kind =
  | { t: "ours"; p: Params; den: DenialParams | null }
  | { t: "bron"; b: Bron; den: DenialParams | null }
  | { t: "clone"; c: CloneSpec; den: DenialParams | null };

interface Seat {
  kind: Kind;
  rack: Rack;
  balance: number;
  score: number;
  won: Uint8Array; // public: tiles won
  played: Uint8Array; // public: letters played
  unseen: Uint8Array;
  shade: number;
  nwon: number;
  paid: number;
  nden: number; // tiles won because the denial wrapper raised the bid
  denPaid: number;
}

let FIRST_PRICE = false;

function bronBid(b: Bron, s: Seat, letter: number, left: number, total: number, rng: RNG): number {
  if (left <= 1) return 0;
  const t = 1 - left / total;
  let x = (b.e + (b.l - b.e) * t) * VAL[letter] * (0.85 + 0.3 * rng());
  x = Math.min(x, b.cap);
  const held = s.rack[letter];
  if (held >= 3) x *= 0.2;
  else if (held >= 2) x *= 0.5;
  if (letter === 16 && s.rack[20] === 0) x = Math.min(x, 4); // Q without U
  return Math.max(0, Math.min(Math.round(x * s.shade), s.balance));
}

function play(s: Seat, word: string) {
  if (!canSpell(word, s.rack)) throw new Error(`cannot spell ${word}`);
  for (let j = 0; j < word.length; j++) {
    const i = word.charCodeAt(j) - 65;
    s.rack[i]--;
    s.played[i]++;
  }
  const r = wordReward(word);
  s.balance += r;
  s.score += r;
}

function playPhase(D: Dict, seats: Seat[], left: number, bagLeft: number) {
  for (const s of seats) {
    if (s.kind.t === "clone") {
      for (let it = 0; it < 12; it++) {
        const ws = cloneChooseWords(D, { rack: s.rack, balance: s.balance, auctionsLeft: left, unseen: s.unseen, bagLeft }, s.kind.c);
        if (ws.length === 0) break;
        for (const w of ws) play(s, w);
      }
    } else if (s.kind.t === "ours") {
      for (let it = 0; it < 12; it++) {
        const ws = chooseWords(D, { rack: s.rack, balance: s.balance, auctionsLeft: left, unseen: s.unseen, bagLeft }, s.kind.p);
        if (ws.length === 0) break;
        for (const w of ws) play(s, w);
      }
    } else {
      if (left <= 2) {
        for (const w of bestPartition(D, s.rack, solveBudgetK(rackSize(s.rack))).words) play(s, w);
        continue;
      }
      for (let it = 0; it < 4; it++) {
        const minLen = s.balance < 8 ? 5 : s.kind.b.min;
        const c = playable(D, s.rack, minLen);
        if (c.length === 0) break;
        let best = c[0];
        for (const i of c) if (D.score[i] > D.score[best]) best = i;
        play(s, D.words[best]);
      }
    }
  }
}

function knownOf(o: Seat): { knownRack: Uint8Array; hidden: number } {
  const knownRack = emptyRack();
  let revealed = 0;
  for (let i = 0; i < 26; i++) {
    const d = o.won[i] - o.played[i];
    if (d >= 0) knownRack[i] = d;
    else revealed -= d;
  }
  return { knownRack, hidden: Math.max(0, RACK0 - revealed) };
}

export function runMatch(D: Dict, kinds: Kind[], seed: number): number[] {
  const n = kinds.length;
  const total = 98 - RACK0 * n;
  const bagRng = mulberry32(seed * 2654435761 + 17);
  const rng = mulberry32(seed * 40503 + 99991);
  const bag: number[] = [];
  for (let i = 0; i < 26; i++) for (let k = 0; k < BAG[i]; k++) bag.push(i);
  shuffle(bagRng, bag);
  let pos = 0;
  const seats: Seat[] = kinds.map((kind) => ({
    kind,
    rack: emptyRack(),
    balance: START_BAL,
    score: 0,
    won: emptyRack(),
    played: emptyRack(),
    unseen: Uint8Array.from(BAG),
    shade: FIRST_PRICE && kind.t !== "ours" ? 0.7 : 1,
    nwon: 0,
    paid: 0,
    nden: 0,
    denPaid: 0,
  }));
  const raised = new Array<boolean>(n).fill(false);
  for (const s of seats) for (let k = 0; k < RACK0; k++) s.rack[bag[pos++]]++;
  for (const s of seats) for (let i = 0; i < 26; i++) s.unseen[i] -= s.rack[i];

  for (let a = 0; a < total; a++) {
    const left = total - a;
    const letter = bag[pos++];
    const L = String.fromCharCode(65 + letter);
    for (const s of seats) s.unseen[letter]--;
    const bagLeft = left - 1;
    playPhase(D, seats, left, bagLeft);

    const bids: { i: number; amt: number }[] = [];
    seats.forEach((s, i) => {
      let amt = 0;
      const opps: DenialOpponent[] = seats
        .filter((o) => o !== s)
        .map((o, j) => ({ botId: BigInt(j), balance: o.balance, score: o.score, sim: null, ...knownOf(o) }));
      const st = { letter: L, rack: s.rack, balance: s.balance, auctionsLeft: left, firstPrice: FIRST_PRICE, opponents: opps, unseen: s.unseen, bagLeft };
      if (s.kind.t === "bron") amt = bronBid(s.kind.b, s, letter, left, total, rng);
      else if (s.kind.t === "clone") amt = left <= 1 ? 0 : Math.round(cloneDecideBid(D, st, s.kind.c, rng) * s.shade);
      else
        amt = decideBid(
          D,
          { letter: L, rack: s.rack, balance: s.balance, auctionsLeft: left, firstPrice: FIRST_PRICE, opponents: opps, unseen: s.unseen, bagLeft },
          s.kind.p,
        );
      const base = amt;
      if (s.kind.den)
        amt = denialBid(
          D,
          { letter: L, rack: s.rack, balance: s.balance, auctionsLeft: left, firstPrice: FIRST_PRICE, opponents: opps, unseen: s.unseen, bagLeft, seats: n, myScore: s.score },
          amt,
          s.kind.den,
        );
      amt = Math.max(0, Math.min(Math.floor(amt), s.balance));
      raised[i] = amt > base;
      if (amt > 0) bids.push({ i, amt });
    });
    if (bids.length === 0) continue; // unbid tile is discarded
    shuffle(rng, bids);
    bids.sort((x, y) => y.amt - x.amt);
    const w = seats[bids[0].i];
    const paid = FIRST_PRICE ? bids[0].amt : bids[1] ? bids[1].amt : Math.min(1, bids[0].amt);
    w.balance -= paid;
    w.rack[letter]++;
    w.won[letter]++;
    w.nwon++;
    w.paid += paid;
    if (raised[bids[0].i]) {
      w.nden++;
      w.denPaid += paid;
    }
  }
  if (process.env.DEBUG)
    console.error(
      seats
        .map((s) => `${s.kind.t}${s.kind.den ? "+d" : ""} sc ${s.score} won ${s.nwon} paid ${s.paid} den ${s.nden}/${s.denPaid} bal ${s.balance} left ${rackSize(s.rack)}`)
        .join(" | "),
    );
  return seats.map((s) => s.score);
}

// ---------- CLI ----------
function parseKV<T extends object>(base: T, spec: string | undefined): T {
  const out: any = { ...base };
  if (!spec) return out;
  for (const kv of spec.split(",")) {
    if (!kv) continue;
    const [k, v] = kv.split("=");
    if (!(k in out)) throw new Error(`unknown key ${k}`);
    out[k] = v.includes("/") ? v.split("/").map(Number) : isNaN(Number(v)) ? v : Number(v);
  }
  return out;
}

interface Rec {
  seed: number;
  hero: number;
  A: number[];
  B: number[];
}

function summarize(recs: Rec[], label: string) {
  recs.sort((x, y) => x.seed - y.seed);
  const n = recs.length;
  const seats = recs[0].A.length;
  const stat = (k: "A" | "B") => {
    let score = 0, opp = 0, win = 0, last = 0, rank = 0;
    const margins: number[] = [];
    for (const r of recs) {
      const s = r[k];
      const me = s[r.hero];
      const others = s.filter((_, i) => i !== r.hero);
      const best = Math.max(...others);
      const low = Math.min(...others);
      score += me;
      opp += others.reduce((a, b) => a + b, 0) / others.length;
      if (me > best) win += 1;
      else if (me === best) win += 0.5;
      if (me <= low) last++; // we lose id ties (bot id 179)
      rank += 1 + others.filter((o) => o > me).length;
      margins.push(me - best);
    }
    // Final: 3 consecutive games, combined score.
    let tw = 0, tn = 0;
    for (let i = 0; i + 2 < n; i += 3) {
      const m = margins[i] + margins[i + 1] + margins[i + 2];
      tn++;
      tw += m > 0 ? 1 : m === 0 ? 0.5 : 0;
    }
    return { score: score / n, opp: opp / n, win: win / n, last: last / n, rank: rank / n, margins, tw: tw / Math.max(1, tn), tn };
  };
  const a = stat("A"), b = stat("B");
  const d = a.margins.map((m, i) => b.margins[i] - m);
  const dm = d.reduce((x, y) => x + y, 0) / n;
  const sd = Math.sqrt(d.reduce((x, y) => x + (y - dm) ** 2, 0) / Math.max(1, n - 1));
  const sdA = Math.sqrt(a.margins.reduce((x, y) => x + (y - a.margins.reduce((p, q) => p + q, 0) / n) ** 2, 0) / Math.max(1, n - 1));
  const f = (x: number) => x.toFixed(1);
  const pc = (x: number) => (100 * x).toFixed(1) + "%";
  console.log(`${label}  seats ${seats}  games ${n}`);
  for (const [k, s] of [["A", a], ["B", b]] as const)
    console.log(
      `  ${k}: score ${f(s.score)} opp ${f(s.opp)} margin ${f(s.margins.reduce((x, y) => x + y, 0) / n)} win ${pc(s.win)} last ${pc(s.last)} rank ${s.rank.toFixed(2)}` +
        (seats === 2 ? ` final3 ${pc(s.tw)} (${s.tn})` : ""),
    );
  console.log(`  B-A margin ${f(dm)} +- ${(sd / Math.sqrt(n)).toFixed(1)} (SE)  margin sd A ${f(sdA)}`);
}

function main() {
  const argv = process.argv.slice(2);
  const arg = (k: string, d?: string) => {
    const i = argv.indexOf("--" + k);
    return i >= 0 ? argv[i + 1] : d;
  };
  if (argv[0] === "--agg") {
    const recs: Rec[] = [];
    let label = "";
    for (const f of argv.slice(1)) {
      const j = JSON.parse(fs.readFileSync(f, "utf8"));
      label = j.label;
      recs.push(...j.recs);
    }
    summarize(recs, label);
    return;
  }
  const here = path.dirname(fileURLToPath(import.meta.url));
  const D = loadDict(path.join(here, "..", "data", "wordlist.txt"));
  const seats = Number(arg("seats", "2"));
  const games = Number(arg("games", "60"));
  const seed0 = Number(arg("seed", "1"));
  const opp = arg("opp", "ours")!;
  FIRST_PRICE = arg("auction", "vickrey") === "first";
  const [si, sn] = (arg("shard", "0/1") as string).split("/").map(Number);
  const denOf = (spec: string | undefined) => (spec === undefined || spec === "none" ? null : parseKV(DEFAULT_DENIAL, spec));
  const denA = denOf(arg("a", "none"));
  const denB = denOf(arg("b", ""));
  const pA = parseKV(DEFAULT_PARAMS, arg("pa"));
  const pB = parseKV(DEFAULT_PARAMS, arg("pb"));
  const oppDen = arg("od") !== undefined ? parseKV(DEFAULT_DENIAL, arg("od")) : null;
  const bron = parseKV<Bron>({ e: 6, l: 11, cap: 35, min: 7 }, arg("bron"));
  const label = argv.filter((x) => !x.startsWith("--shard") && !x.includes("/") ).join(" ");

  const CLONES = loadClones(path.join(here, "..", "data", "clones.json"));
  const recs: Rec[] = [];
  const t0 = performance.now();
  for (let g = si; g < games; g += sn) {
    const seed = seed0 * 1000003 + g;
    const hero = g % seats;
    const oppKind = (j: number): Kind => {
      if (opp.startsWith("clone:")) {
        const names = opp.slice(6).split("+");
        return { t: "clone", c: CLONES[names[j % names.length]], den: oppDen };
      }
      const useBron = opp === "bron" || (opp === "mix" && j % 2 === 1);
      return useBron ? { t: "bron", b: bron, den: oppDen } : { t: "ours", p: DEFAULT_PARAMS, den: oppDen };
    };
    const lineup = (p: Params, den: DenialParams | null): Kind[] =>
      Array.from({ length: seats }, (_, j) => (j === hero ? { t: "ours", p, den } : oppKind(j)));
    const A = runMatch(D, lineup(pA, denA), seed);
    const B = runMatch(D, lineup(pB, denB), seed);
    recs.push({ seed: g, hero, A, B });
  }
  const out = arg("json");
  if (out) fs.writeFileSync(out, JSON.stringify({ label, recs }));
  else summarize(recs, label);
  console.error(`${recs.length} pairs in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();

// Offline match simulator. Mirrors vendor/scrabblebot/spacetimedb/src/game.ts:
// 6 seats, 98-tile bag, 5 tiles dealt, balance 100, 68 Vickrey auctions,
// sim bots with jitter and longest-word play after each win.
//
// npx tsx src/sim.ts [--matches N] [--seed S] [--mode sims|mirror|mixed|field]
//   [--p key=val,key=val]       override our DEFAULT_PARAMS (all our seats)
//   [--sweep key=v1,v2,...]     field/sims mode: run one batch per value
//   [--lambdas 1.5,2,...]       mirror mode lambdas (6 values)

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  BAG,
  VAL,
  bestPartition,
  canSpell,
  emptyRack,
  loadDict,
  playable,
  rackSize,
  solveBudgetK,
  wordReward,
  type Dict,
  type Rack,
} from "./engine.js";
import {
  DEFAULT_PARAMS,
  chooseWords,
  decideBid,
  letterValue,
  type Opponent,
  type Params,
  type SimKind,
} from "./strategy.js";

const AUCTIONS = 68;
const SEATS = 6;
const START_BAL = 100;
const RACK0 = 5;
const VOWELS = "AEIOU";

// ---------- RNG ----------
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
const randInt = (rng: RNG, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo + 1));
function shuffle<T>(rng: RNG, a: T[]) {
  for (let i = a.length - 1; i >= 1; i--) {
    const j = randInt(rng, 0, i);
    [a[i], a[j]] = [a[j], a[i]];
  }
}

// ---------- Sim word play (findBestPlayable, minLen 3) ----------
// Longest spellable word, lex-smallest among equal length. D keeps the
// lex-smallest word of each anagram class (wordlist is sorted), so sorting
// D's classes by (length desc, word asc) gives the same first match.
let SIM_ORDER: Int32Array;
function buildSimOrder(D: Dict) {
  const idx = Array.from({ length: D.N }, (_, i) => i).filter((i) => D.len[i] >= 3);
  idx.sort((a, b) => D.len[b] - D.len[a] || (D.words[a] < D.words[b] ? -1 : 1));
  SIM_ORDER = Int32Array.from(idx);
}
function simBestPlayable(D: Dict, r: Rack): number {
  let rm = 0,
    tot = 0;
  for (let i = 0; i < 26; i++) {
    if (r[i]) rm |= 1 << i;
    tot += r[i];
  }
  if (tot < 3) return -1;
  for (let k = 0; k < SIM_ORDER.length; k++) {
    const w = SIM_ORDER[k];
    if (D.len[w] > tot || (D.mask[w] & ~rm) !== 0) continue;
    const b = w * 26;
    let ok = true;
    for (let s = 0; s < 26; s++)
      if (D.counts[b + s] > r[s]) {
        ok = false;
        break;
      }
    if (ok) return w;
  }
  return -1;
}

// ---------- Seats ----------
type Kind =
  | { t: "ours"; p: Params; label: string }
  | { t: "sim"; s: SimKind; label: string }
  | { t: "lucent"; prices: number[]; label: string }
  | { t: "rival"; r: Rival; label: string };

// ---------- Rival models from data/rivals.json (mode real) ----------
// Per letter, the archived top bid on auctions the rival won (biased upward,
// so scaled by RSCALE), sampled uniformly between p25 and p75, times a phase
// factor from bidPerFaceByPhase, capped by balance.
interface Rival {
  name: string;
  weight: number;
  lo: number[];
  hi: number[];
  phase: [number, number, number]; // early, mid, late
  minLen: number;
}
let RIVALS: Rival[] = [];
let RSCALE = 1.0;
function loadRivals(file: string) {
  const d = JSON.parse(fs.readFileSync(file, "utf8"));
  const field = d._field.byLetter;
  const out: Rival[] = [];
  for (const [name, v] of Object.entries<any>(d)) {
    if (name.startsWith("_") || name === "exzell" || (v.matches ?? 0) < 20) continue;
    const lo: number[] = [];
    const hi: number[] = [];
    for (let i = 0; i < 26; i++) {
      const L = String.fromCharCode(65 + i);
      const b = v.byLetter?.[L];
      const f = field[L]?.winningBid ?? { p25: 5, p75: 10 };
      const use = b && b.n >= 5 ? b : f;
      lo.push(use.p25);
      hi.push(use.p75);
    }
    const ph = v.bidPerFaceByPhase;
    const tot = (ph.early.p50 * ph.early.n + ph.mid.p50 * ph.mid.n + ph.late.p50 * ph.late.n) / (ph.early.n + ph.mid.n + ph.late.n);
    out.push({
      name,
      weight: v.matches,
      lo,
      hi,
      phase: [ph.early.p50 / tot, ph.mid.p50 / tot, ph.late.p50 / tot],
      minLen: Math.max(5, Math.min(7, Math.round(v.medianWordLen ?? 7))),
    });
  }
  out.sort((a, b) => b.weight - a.weight);
  RIVALS = out;
}
// Rack awareness: scale by how much our own table rule (dup and vowel balance)
// changes the letter's value on this rack. Off with --rackaware 0.
let RACK_AWARE = 1;
const EMPTY = new Uint8Array(26);
function rivalBid(r: Rival, letter: number, left: number, balance: number, rng: RNG, rack: Rack): number {
  if (left <= 1) return 0;
  const ph = left > 45 ? r.phase[0] : left > 15 ? r.phase[1] : r.phase[2];
  let fit = 1;
  if (RACK_AWARE) {
    const L = String.fromCharCode(65 + letter);
    const base = L === "Q" ? 5.5 : letterValue(L, EMPTY, DEFAULT_PARAMS);
    fit = Math.max(0.3, Math.min(1.3, letterValue(L, rack, DEFAULT_PARAMS) / base));
  }
  const raw = (r.lo[letter] + rng() * (r.hi[letter] - r.lo[letter])) * RSCALE * ph * fit;
  return Math.max(0, Math.min(Math.round(raw), balance));
}

interface Seat {
  id: number;
  unseen: Uint8Array; // our view: BAG - start rack - every letter auctioned so far
  kind: Kind;
  rack: Rack;
  balance: number;
  score: number;
  won: number;
  paid: number;
}

function simBid(s: SimKind, letter: number, balance: number, jitter: number): number {
  const v = VAL[letter];
  const isV = VOWELS.includes(String.fromCharCode(65 + letter));
  const base = s === "Cheapskate" ? v - 1 : s === "ValueBidder" ? v : v + (isV ? 4 : 2);
  return Math.max(Math.min(Math.max(base + jitter, 1), balance), 0);
}

// LUCENT-like prices observed live, by face value.
function lucentBase(letter: number): number {
  const ch = String.fromCharCode(65 + letter);
  if ("DG".includes(ch)) return 12;
  if ("BCMP".includes(ch)) return 14;
  if ("FHVWY".includes(ch)) return 19;
  if (ch === "K") return 20;
  if ("JQXZ".includes(ch)) return 35;
  return 9;
}
function lucentPrices(rng: RNG): number[] {
  return Array.from({ length: 26 }, (_, i) => Math.round(lucentBase(i) * (0.8 + 0.4 * rng())));
}

function play(D: Dict, seat: Seat, word: string) {
  if (!canSpell(word, seat.rack)) throw new Error(`seat ${seat.id} cannot spell ${word}`);
  for (let j = 0; j < word.length; j++) seat.rack[word.charCodeAt(j) - 65]--;
  const r = wordReward(word);
  seat.balance += r;
  seat.score += r;
}

// Word play opportunity before an auction opens. `left` includes that auction.
function playPhase(D: Dict, seats: Seat[], left: number, bagLeft: number) {
  for (const s of seats) {
    if (s.kind.t === "ours") {
      for (let it = 0; it < 12; it++) {
        const ws = chooseWords(
          D,
          { rack: s.rack, balance: s.balance, auctionsLeft: left, unseen: s.unseen, bagLeft },
          s.kind.p,
        );
        if (ws.length === 0) break;
        for (const w of ws) play(D, s, w);
      }
    } else if (s.kind.t === "lucent" || s.kind.t === "rival") {
      const minLen = s.kind.t === "rival" ? s.kind.r.minLen : 7;
      if (left <= 2) {
        const part = bestPartition(D, s.rack, solveBudgetK(rackSize(s.rack)));
        for (const w of part.words) play(D, s, w);
      } else {
        for (let it = 0; it < 4; it++) {
          const c = playable(D, s.rack, minLen);
          if (c.length === 0) break;
          let best = c[0];
          for (const i of c) if (D.score[i] > D.score[best]) best = i;
          play(D, s, D.words[best]);
        }
      }
    }
  }
}

function runMatch(D: Dict, kinds: Kind[], rng: RNG) {
  const bag: number[] = [];
  for (let i = 0; i < 26; i++) for (let k = 0; k < BAG[i]; k++) bag.push(i);
  shuffle(rng, bag);
  let pos = 0;
  const seats: Seat[] = kinds.map((kind, id) => ({
    id,
    kind,
    rack: emptyRack(),
    balance: START_BAL,
    score: 0,
    won: 0,
    paid: 0,
    unseen: Uint8Array.from(BAG),
  }));
  for (const s of seats) for (let k = 0; k < RACK0; k++) s.rack[bag[pos++]]++;
  for (const s of seats) for (let i = 0; i < 26; i++) s.unseen[i] -= s.rack[i];
  const solveMs: number[] = seats.map(() => 0);

  for (let a = 0; a < AUCTIONS; a++) {
    const left = AUCTIONS - a;
    const letter = bag[pos++];
    const L = String.fromCharCode(65 + letter);
    for (const s of seats) s.unseen[letter]--;
    const bagLeft = left - 1;
    playPhase(D, seats, left, bagLeft);

    const bids: { seat: number; amt: number }[] = [];
    for (const s of seats) {
      let amt = 0;
      if (s.kind.t === "sim") {
        amt = simBid(s.kind.s, letter, s.balance, randInt(rng, -2, 2));
      } else if (s.kind.t === "lucent") {
        amt = Math.min(s.kind.prices[letter], s.balance);
      } else if (s.kind.t === "rival") {
        amt = rivalBid(s.kind.r, letter, left, s.balance, rng, s.rack);
      } else {
        const opponents: Opponent[] = seats
          .filter((o) => o !== s)
          .map((o) => ({
            botId: BigInt(o.id),
            balance: o.balance,
            score: o.score,
            sim: o.kind.t === "sim" ? o.kind.s : null,
          }));
        const t0 = performance.now();
        amt = decideBid(
          D,
          {
            letter: L,
            rack: s.rack,
            balance: s.balance,
            auctionsLeft: left,
            firstPrice: false,
            opponents,
            unseen: s.unseen,
            bagLeft,
          },
          s.kind.p,
        );
        const dt = performance.now() - t0;
        solveMs[s.id] += dt;
        BID_MS.push(dt);
        amt = Math.max(0, Math.min(Math.floor(amt), s.balance)); // submit rejects > balance
      }
      if (amt > 0) bids.push({ seat: s.id, amt });
    }
    shuffle(rng, bids);
    bids.sort((x, y) => y.amt - x.amt); // stable
    if (bids.length > 0) {
      const w = seats[bids[0].seat];
      const paid = bids[1] ? bids[1].amt : Math.min(1, bids[0].amt);
      if (w.balance >= paid) {
        w.balance -= paid;
        w.rack[letter]++;
        w.won++;
        w.paid += paid;
        if (w.kind.t === "sim") {
          // Sims play in the closing transaction, so even the last tile counts.
          const wi = simBestPlayable(D, w.rack);
          if (wi >= 0) play(D, w, D.words[wi]);
        }
      }
    }
  }
  return { seats, solveMs };
}

// ---------- Stats ----------
const BID_MS: number[] = [];
interface Agg {
  n: number;
  score: number;
  wins: number;
  won: number;
  paid: number;
  bal: number;
  left: number;
  rank: number;
}
const newAgg = (): Agg => ({ n: 0, score: 0, wins: 0, won: 0, paid: 0, bal: 0, left: 0, rank: 0 });

function record(aggs: Map<string, Agg>, seats: Seat[]) {
  const top = Math.max(...seats.map((s) => s.score));
  const nTop = seats.filter((s) => s.score === top).length;
  for (const s of seats) {
    const k = s.kind.label;
    if (!aggs.has(k)) aggs.set(k, newAgg());
    const g = aggs.get(k)!;
    g.n++;
    g.score += s.score;
    g.wins += s.score === top ? 1 / nTop : 0;
    g.won += s.won;
    g.paid += s.paid;
    g.bal += s.balance;
    g.left += rackSize(s.rack);
    // Average rank among tied scores (1 = best).
    const better = seats.filter((o) => o.score > s.score).length;
    const tied = seats.filter((o) => o.score === s.score).length;
    g.rank += better + (tied + 1) / 2;
  }
}

function report(aggs: Map<string, Agg>, matches: number) {
  const rows = [...aggs.entries()].sort((a, b) => b[1].score / b[1].n - a[1].score / a[1].n);
  const lines = [
    "| strategy | seats | mean score | win rate | mean rank | tiles won | paid/tile | end balance | leftover tiles |",
    "|---|---|---|---|---|---|---|---|---|",
  ];
  for (const [k, g] of rows) {
    const perMatchSeats = g.n / matches;
    lines.push(
      `| ${k} | ${perMatchSeats.toFixed(perMatchSeats % 1 ? 1 : 0)} | ${(g.score / g.n).toFixed(1)} | ${((100 * g.wins) / g.n).toFixed(1)}% | ${(g.rank / g.n).toFixed(2)} | ${(g.won / g.n).toFixed(1)} | ${(g.won ? g.paid / g.won : 0).toFixed(2)} | ${(g.bal / g.n).toFixed(1)} | ${(g.left / g.n).toFixed(1)} |`,
    );
  }
  return lines.join("\n");
}

// ---------- CLI ----------
function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k: string, d?: string) => {
    const i = a.indexOf(`--${k}`);
    return i >= 0 ? a[i + 1] : d;
  };
  return {
    matches: Number(get("matches", "50")),
    seed: Number(get("seed", "1")),
    mode: get("mode", "sims")!,
    p: get("p"),
    sweep: get("sweep"),
    lambdas: get("lambdas"),
    q: get("q"),
    json: a.includes("--json"),
    rscale: get("rscale"),
    rackaware: get("rackaware"),
  };
}

function withOverrides(base: Params, spec?: string): Params {
  const p: Params = { ...base, dup: [...base.dup], floors: { ...base.floors }, caps: { ...base.caps } };
  if (!spec) return p;
  for (const kv of spec.split(",")) {
    const [k, v] = kv.split("=");
    if (!(k in p)) throw new Error(`unknown param ${k}`);
    const cur = (p as any)[k];
    if (typeof cur === "string") (p as any)[k] = v;
    else if (Array.isArray(cur)) (p as any)[k] = v.split("/").map(Number);
    else if (typeof cur === "object") {
      // floors=A:10/E:10 merges into the record
      const o = { ...cur };
      for (const e of v.split("/")) {
        const [l, n] = e.split(":");
        o[l] = Number(n);
      }
      (p as any)[k] = o;
    } else (p as any)[k] = Number(v);
  }
  return p;
}

const SIM_POOL: SimKind[] = ["Cheapskate", "ValueBidder", "Aggressive", "ValueBidder", "Aggressive", "Cheapskate"];

function lineup(mode: string, ours: Params, rng: RNG, lambdas: number[], alt: Params): Kind[] {
  const sims = (n: number): Kind[] => {
    const pool = [...SIM_POOL];
    shuffle(rng, pool);
    return pool.slice(0, n).map((s) => ({ t: "sim", s, label: `sim:${s}` }));
  };
  const ourKind = (p: Params, label: string): Kind => ({ t: "ours", p, label });
  let ks: Kind[];
  if (mode === "sims") ks = [ourKind(ours, "ours"), ...sims(5)];
  else if (mode === "mirror")
    ks = lambdas.map((l) => ourKind({ ...ours, lambda: l }, `ours lambda=${l}`));
  else if (mode === "mixed")
    ks = [
      ourKind({ ...ours, lambda: 1.8 }, "ours lambda=1.8"),
      ourKind(ours, `ours lambda=${ours.lambda}`),
      ourKind({ ...ours, lambda: 3.0 }, "ours lambda=3.0"),
      ...sims(3),
    ];
  else if (mode === "field")
    ks = [
      ourKind(ours, "ours"),
      ...Array.from({ length: 5 }, (): Kind => ({ t: "lucent", prices: lucentPrices(rng), label: "lucent-like" })),
    ];
  else if (mode === "real" || mode === "realduel") {
    // LUCENT always, plus other rivals drawn by match count without replacement.
    const pool = [...RIVALS];
    const pick: Rival[] = [];
    const lu = pool.findIndex((r) => r.name === "LUCENT");
    if (lu >= 0) pick.push(...pool.splice(lu, 1));
    const need = mode === "real" ? 5 : 4;
    while (pick.length < need && pool.length) {
      const tot = pool.reduce((a, r) => a + r.weight, 0);
      let x = rng() * tot;
      let i = 0;
      while (i < pool.length - 1 && (x -= pool[i].weight) > 0) i++;
      pick.push(...pool.splice(i, 1));
    }
    const rv = pick.map((r): Kind => ({ t: "rival", r, label: r.name === "LUCENT" ? "LUCENT" : "rival:other" }));
    ks = mode === "real" ? [ourKind(ours, "ours"), ...rv] : [ourKind(ours, "A"), ourKind(alt, "B"), ...rv];
  } else if (mode === "vs")
    // 3 seats of A (--p) against 3 seats of B (--p then --q), nothing else.
    ks = [
      ...Array.from({ length: 3 }, () => ourKind(ours, "A")),
      ...Array.from({ length: 3 }, () => ourKind(alt, "B")),
    ];
  else if (mode === "duel")
    // One A and one B in a field of 4 lucent-like.
    ks = [
      ourKind(ours, "A"),
      ourKind(alt, "B"),
      ...Array.from({ length: 4 }, (): Kind => ({ t: "lucent", prices: lucentPrices(rng), label: "lucent-like" })),
    ];
  else throw new Error(`unknown mode ${mode}`);
  shuffle(rng, ks);
  return ks;
}

function batch(D: Dict, mode: string, matches: number, seed: number, ours: Params, lambdas: number[], alt: Params) {
  const aggs = new Map<string, Agg>();
  const t0 = performance.now();
  let maxMs = 0;
  for (let m = 0; m < matches; m++) {
    // Fresh stream per match so sweeps compare on the same bags.
    const rng = mulberry32(seed * 100003 + m * 7919 + 1);
    const t1 = performance.now();
    const { seats } = runMatch(D, lineup(mode, ours, rng, lambdas, alt), rng);
    maxMs = Math.max(maxMs, performance.now() - t1);
    record(aggs, seats);
  }
  const ms = performance.now() - t0;
  return { aggs, ms, maxMs };
}

function main() {
  const args = parseArgs();
  const here = path.dirname(fileURLToPath(import.meta.url));
  const tl = performance.now();
  const D = loadDict(path.join(here, "..", "data", "wordlist.txt"));
  buildSimOrder(D);
  if (args.rscale) RSCALE = Number(args.rscale);
  if (args.rackaware) RACK_AWARE = Number(args.rackaware);
  if (args.mode.startsWith("real")) loadRivals(path.join(here, "..", "data", "rivals.json"));
  console.log(`dict ${D.N} classes loaded in ${(performance.now() - tl).toFixed(0)} ms`);
  const base = withOverrides(DEFAULT_PARAMS, args.p);
  const lambdas = args.lambdas ? args.lambdas.split(",").map(Number) : [1.5, 2.0, 2.3, 2.6, 3.0, 3.5];

  const runs: { title: string; p: Params }[] = [];
  if (args.sweep) {
    const [k, vs] = args.sweep.split("=");
    for (const v of vs.split(",")) runs.push({ title: `${k}=${v}`, p: withOverrides(base, `${k}=${v}`) });
  } else runs.push({ title: args.p ?? "DEFAULT_PARAMS", p: base });

  for (const r of runs) {
    BID_MS.length = 0;
    const alt = withOverrides(r.p, args.q);
    const { aggs, ms, maxMs } = batch(D, args.mode, args.matches, args.seed, r.p, lambdas, alt);
    const sorted = [...BID_MS].sort((a, b) => a - b);
    const pct = (q: number) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0);
    console.log(
      `\n## mode=${args.mode}${args.mode.startsWith("real") ? ` rscale=${RSCALE} rackaware=${RACK_AWARE}` : ""} ${r.title} matches=${args.matches} seed=${args.seed}  (${(ms / args.matches).toFixed(1)} ms/match, max ${maxMs.toFixed(0)} ms, ${((1000 * args.matches) / ms).toFixed(1)} matches/s)\n`,
    );
    console.log(report(aggs, args.matches));
    console.log(
      `\ndecideBid ms: p50 ${pct(0.5).toFixed(2)} p90 ${pct(0.9).toFixed(2)} p99 ${pct(0.99).toFixed(2)} max ${pct(1).toFixed(2)} (n=${sorted.length})`,
    );
    if (args.json) console.log("JSON " + JSON.stringify({ title: r.title, mode: args.mode, seed: args.seed, matches: args.matches, aggs: Object.fromEntries(aggs) }));
  }
}

main();

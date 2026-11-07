// Behavioral clones of top rival bots (LUCENT, Bron, GodOfThudner), fitted from
// public post-match data by src/clonefit.ts into data/clones.json.
// Report: docs/research/13-clones.md.
//
// Bid model (censored log-normal with a zero-bid mass):
//   P(bid = 0) = sigmoid(z0 + zph[phase] + zh * held + zc[class])
//   otherwise ln(bid) ~ N(mu, sigma), capped at balance, where
//   mu = a[letter] + ph[phase] + cb * ln(balance / 60) + h1 * [held == 1]
//        + h2 * [held >= 2] + cv * vowelTilt + cf * ln(1 + partition gain)
// Word policy: cash the highest-reward word of minLen+ letters whenever one is
// playable; below cashFloor (with cashAny) cash the best partition word; flush
// everything at flushAt auctions left.

import * as fs from "node:fs";
import { bestPartition, playable, rackSize, solveBudgetK, wordReward, type Dict, type Rack } from "./engine.js";
import { DEFAULT_PARAMS, letterValue, partitionValue, type BidState, type PlayState } from "./strategy.js";

// Phase buckets by auctions left (incl. the current one): lower bounds.
export const BUCKETS = [56, 46, 31, 16, 6, 2, 1];

export interface CloneModel {
  a: number[];
  ph: number[];
  cb: number;
  h1: number;
  h2: number;
  cv: number;
  cf: number;
  sigma: number;
  z0: number;
  zph: number[];
  zh: number;
  zc: number[];
}

export interface CloneWordPolicy {
  minLen: number;
  flushAt: number;
  cashFloor: number;
  cashAny: number; // 1 = below cashFloor, cash the best partition word of any length
  part?: number; // 1 = take the minLen+ word from the best partition (keeps the rest optimal), not the single highest-reward word
}

export interface CloneSpec {
  name: string;
  model: CloneModel;
  words: CloneWordPolicy;
  // Share of the fitted bid noise that is explained by rack fit (0..1). The
  // public data sees only the known part of a rival's rack, so the fitted
  // rack terms are weak and the rest lands in sigma. rho moves that much of
  // sigma onto a standardized rack-fit score (partition gain on the full
  // rack), which keeps the marginal bid distribution but buys coherent tiles.
  rho: number;
}

// Running standardization of the rack-fit signals across clone bids.
interface Run {
  n: number;
  m: number;
  v: number;
}
const FIT_G: Run = { n: 0, m: 2, v: 1 };
const FIT_T: Run = { n: 0, m: -0.2, v: 0.1 };
function stdz(R: Run, x: number): number {
  R.n++;
  const k = Math.min(R.n, 5000);
  const d = x - R.m;
  R.m += d / k;
  R.v += (d * (x - R.m) - R.v) / k;
  return (x - R.m) / Math.sqrt(Math.max(1e-6, R.v));
}
// Fit signal: "gain" = partition gain only; "table" = our table rule (dup and
// vowel balance) only; "both" = their standardized mean.
export let FIT_MODE: "gain" | "table" | "both" = "gain";
export function setFitMode(m: "gain" | "table" | "both") {
  FIT_MODE = m;
}
const EMPTY = new Uint8Array(26);
function fitZ(lg: number, s: BidState): number {
  if (FIT_MODE === "gain") return stdz(FIT_G, lg);
  const L = s.letter;
  const base = L === "Q" ? 5.5 : letterValue(L, EMPTY, DEFAULT_PARAMS);
  const t = stdz(FIT_T, Math.log(Math.max(0.05, letterValue(L, s.rack, DEFAULT_PARAMS) / base)));
  if (FIT_MODE === "table") return t;
  return (stdz(FIT_G, lg) + t) / Math.SQRT2;
}

const VOW = new Set([0, 4, 8, 14, 20]);
const ONE = new Set([11, 13, 17, 18, 19]); // L N R S T

export function letterClass(L: number): number {
  if (VOW.has(L)) return 0;
  if (ONE.has(L)) return 1;
  if ([9, 16, 23, 25, 10].includes(L)) return 3; // J Q X Z K
  return 2;
}

export function bucketOf(left: number): number {
  for (let i = 0; i < BUCKETS.length; i++) if (left >= BUCKETS[i]) return i;
  return BUCKETS.length - 1;
}

export function cloneFeatures(L: number, left: number, bal: number, held: number, vf: number, gain: number) {
  const tilt = VOW.has(L) ? vf - 0.4 : 0.4 - vf;
  return {
    b: bucketOf(left),
    lb: Math.log(Math.max(1, bal) / 60),
    h1: held === 1 ? 1 : 0,
    h2: held >= 2 ? 1 : 0,
    held: Math.min(held, 3),
    vt: tilt,
    lg: Math.log(1 + Math.max(0, gain)),
    cls: letterClass(L),
  };
}

export function rackFeatures(rack: Rack, L: number, D: Dict | null) {
  const size = rackSize(rack);
  let vow = 0;
  for (const v of VOW) vow += rack[v];
  let gain = 0;
  if (D && size <= 22) {
    const w = Uint8Array.from(rack);
    w[L]++;
    gain = Math.max(0, partitionValue(D, w) - partitionValue(D, rack));
  }
  return { held: rack[L], vf: size >= 3 ? vow / size : 0.4, gain };
}

export function cloneMuQ(c: CloneModel, s: BidState, D: Dict | null) {
  const L = s.letter.charCodeAt(0) - 65;
  const r = rackFeatures(s.rack, L, D);
  const F = cloneFeatures(L, s.auctionsLeft, s.balance, r.held, r.vf, r.gain);
  const mu = c.a[L] + c.ph[F.b] + c.cb * F.lb + c.h1 * F.h1 + c.h2 * F.h2 + c.cv * F.vt + c.cf * F.lg;
  const z = c.z0 + c.zph[F.b] + c.zh * F.held + c.zc[F.cls];
  return { mu, q: 1 / (1 + Math.exp(-z)), lg: F.lg };
}

export type RNG = () => number;

function gauss(rng: RNG): number {
  const u = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// Clone bid. With an rng it samples the fitted distribution (a realistic
// opponent); without one it bids the median (zero if P(no bid) > 0.5).
export function cloneDecideBid(D: Dict, s: BidState, c: CloneSpec, rng?: RNG, scale = 1): number {
  if (s.balance <= 0) return 0;
  const { mu, q, lg } = cloneMuQ(c.model, s, D);
  const rho = Math.max(0, Math.min(1, c.rho));
  const zf = rho > 0 ? fitZ(lg, s) : 0;
  let b: number;
  if (rng) {
    if (rng() < q) return 0;
    b = Math.exp(mu + c.model.sigma * (rho * zf + Math.sqrt(1 - rho * rho) * gauss(rng)));
  } else {
    if (q > 0.5) return 0;
    b = Math.exp(mu + c.model.sigma * rho * zf);
  }
  return Math.max(0, Math.min(Math.round(b * scale), s.balance));
}

export function cloneChooseWords(D: Dict, s: PlayState, c: CloneSpec): string[] {
  const size = rackSize(s.rack);
  if (size < 2) return [];
  const w = c.words;
  if (s.auctionsLeft > 0 && s.auctionsLeft <= w.flushAt) {
    return bestPartition(D, s.rack, solveBudgetK(size)).words;
  }
  if (w.part) {
    const part = bestPartition(D, s.rack, solveBudgetK(size));
    const long = part.words.filter((x) => x.length >= w.minLen).sort((a, b) => wordReward(b) - wordReward(a));
    if (long.length) return [long[0]];
  }
  const cand = w.part ? [] : playable(D, s.rack, w.minLen);
  if (cand.length > 0) {
    let best = cand[0];
    for (const i of cand) if (D.score[i] > D.score[best]) best = i;
    return [D.words[best]];
  }
  if (w.cashAny && s.balance < w.cashFloor) {
    const part = bestPartition(D, s.rack, solveBudgetK(size));
    const words = [...part.words].sort((a, b) => wordReward(b) - wordReward(a));
    if (words.length && wordReward(words[0]) >= 6) return [words[0]];
  }
  return [];
}

export function loadClones(file: string): Record<string, CloneSpec> {
  const d = JSON.parse(fs.readFileSync(file, "utf8"));
  const out: Record<string, CloneSpec> = {};
  for (const [k, v] of Object.entries<any>(d)) {
    if (k.startsWith("_")) continue;
    out[k] = { name: k, model: v.model, words: v.words, rho: v.rho ?? 0 };
  }
  return out;
}

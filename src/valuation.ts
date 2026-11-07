// Bag-aware letter valuation by Monte Carlo over plausible future tiles.
//
// value(L) = E_F[P(rack + L + F)] - E_F[P(rack + F)]
//
// P is the best-partition value (engine.bestPartition), and F is a random
// draw of the future tiles we expect to win, taken from the unseen pool
// (bag plus opponents' face-down starting tiles). The same F samples are used
// for the with and without terms (common random numbers), so most of the
// sampling noise cancels in the difference.
// Notes and sim results: docs/research/08-valuation.md.

import { bestPartition, rackSize, wordReward, type Dict, type Rack } from "./engine.js";

export interface McOpts {
  samples: number; // number of F draws
  future: number; // size of each F draw (already scaled to our expected win share)
  K: number; // partition breadth cap (top K words per state); Infinity = exact for small racks
}

// ---------- Partition cache ----------
// Keyed by rack counts. Shared across calls, bounded.
const CACHE = new Map<string, number>();
const CACHE_MAX = 200_000;
let cacheHits = 0;
let cacheMiss = 0;

function key(r: Rack): string {
  // 26 small counts; String.fromCharCode is cheap and collision-free here.
  return String.fromCharCode(...r);
}

function kFor(size: number, K: number): number {
  // Smaller racks are cheap even when exact; larger ones need a cap.
  if (size <= 12) return Infinity;
  if (size <= 15) return Math.min(K, 24);
  if (size <= 18) return Math.min(K, 12);
  return Math.min(K, 6);
}

export function cachedPartition(D: Dict, r: Rack, K: number): number {
  const k = key(r) + "|" + (K === Infinity ? "i" : K);
  const hit = CACHE.get(k);
  if (hit !== undefined) {
    cacheHits++;
    return hit;
  }
  cacheMiss++;
  const v = bestPartition(D, r, kFor(rackSize(r), K)).total;
  if (CACHE.size >= CACHE_MAX) CACHE.clear();
  CACHE.set(k, v);
  return v;
}

export function cacheStats() {
  return { hits: cacheHits, miss: cacheMiss, size: CACHE.size };
}

// ---------- Deterministic RNG ----------
// Seeded from the state, so the same state gives the same value (and the sim
// stays reproducible).
function hashState(r: Rack, pool: Uint8Array, extra: number): number {
  let h = 2166136261 ^ extra;
  for (let i = 0; i < 26; i++) {
    h = Math.imul(h ^ r[i], 16777619);
    h = Math.imul(h ^ pool[i], 16777619);
  }
  return h >>> 0;
}
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

// Draw `n` tiles without replacement from the pool. Returns `samples` racks.
export function drawFutures(pool: Uint8Array, n: number, samples: number, seed: number): Rack[] {
  const flat: number[] = [];
  for (let i = 0; i < 26; i++) for (let k = 0; k < pool[i]; k++) flat.push(i);
  const m = Math.min(n, flat.length);
  const rng = mulberry32(seed);
  const out: Rack[] = [];
  for (let s = 0; s < samples; s++) {
    const f = new Uint8Array(26);
    // Partial Fisher-Yates on a copy.
    const a = flat.slice();
    for (let i = 0; i < m; i++) {
      const j = i + Math.floor(rng() * (a.length - i));
      const t = a[i];
      a[i] = a[j];
      a[j] = t;
      f[a[i]]++;
    }
    out.push(f);
  }
  return out;
}

function add(a: Rack, b: Rack): Rack {
  const o = new Uint8Array(26);
  for (let i = 0; i < 26; i++) o[i] = a[i] + b[i];
  return o;
}

// Marginal value of one more `letter` (index) on `rack`, given the unseen pool.
export function mcLetterValue(D: Dict, rack: Rack, letter: number, unseen: Uint8Array, o: McOpts): number {
  const futures = drawFutures(unseen, o.future, o.samples, hashState(rack, unseen, letter * 7919 + o.future));
  let sum = 0;
  for (const F of futures) {
    const base = add(rack, F);
    const without = cachedPartition(D, base, o.K);
    base[letter]++;
    const with_ = cachedPartition(D, base, o.K);
    sum += with_ - without;
  }
  return sum / futures.length;
}

// Expected future value of keeping `rack` (E_F[P(rack + F)]), with fixed futures.
export function mcRackValue(D: Dict, rack: Rack, futures: Rack[], K: number): number {
  let sum = 0;
  for (const F of futures) sum += cachedPartition(D, add(rack, F), K);
  return sum / futures.length;
}

// Pick the word to cash from `cands` that keeps the most total value:
// argmax reward(w) + E_F[P(rack - w + F)]. Returns the chosen word and the
// expected loss versus holding (E[P(rack + F)] - reward(w) - E[P(rack - w + F)]).
export function mcBestCash(
  D: Dict,
  rack: Rack,
  cands: string[],
  unseen: Uint8Array,
  o: McOpts,
): { word: string; loss: number } | null {
  if (cands.length === 0) return null;
  const futures = drawFutures(unseen, o.future, o.samples, hashState(rack, unseen, 104729 + o.future));
  const hold = mcRackValue(D, rack, futures, o.K);
  let best: { word: string; loss: number } | null = null;
  for (const w of cands) {
    const r = Uint8Array.from(rack);
    for (let j = 0; j < w.length; j++) r[w.charCodeAt(j) - 65]--;
    const keep = wordReward(w) + mcRackValue(D, r, futures, o.K);
    const loss = hold - keep;
    if (!best || loss < best.loss) best = { word: w, loss };
  }
  return best;
}

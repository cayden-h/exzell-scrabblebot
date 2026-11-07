// Word engine: dictionary, scoring, playable-word lookup, best partition.
// Ported from docs/research/data/03-word-play/lib.mjs.
// A rack is a Uint8Array(26) of letter counts, index 0 = 'A'.

import * as fs from "node:fs";

export const VAL = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10];
export const BAG = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1];
const A = 65;

export type Rack = Uint8Array;

// Same integer math as the module: base * num / den, truncated.
export function reward(base: number, len: number): number {
  if (len <= 3) return base;
  if (len === 4) return Math.floor((base * 3) / 2);
  if (len === 5) return base * 2;
  if (len === 6) return Math.floor((base * 5) / 2);
  return base * 3;
}

export interface Dict {
  words: string[];
  N: number;
  counts: Uint8Array;
  mask: Int32Array;
  len: Uint8Array;
  score: Int32Array;
  minLetter: Uint8Array;
}

export function loadDict(path: string): Dict {
  const raw = fs
    .readFileSync(path, "utf8")
    .split("\n")
    .map((w) => w.trim().toUpperCase())
    .filter((w) => w.length >= 2 && /^[A-Z]+$/.test(w));
  // Keep one word per anagram class; all anagrams score the same.
  const seen = new Set<string>();
  const words: string[] = [];
  for (const w of raw) {
    const k = w.split("").sort().join("");
    if (seen.has(k)) continue;
    // Skip words the 98-tile bag cannot spell.
    const c = new Uint8Array(26);
    let ok = true;
    for (let j = 0; j < w.length; j++) {
      const s = w.charCodeAt(j) - A;
      if (++c[s] > BAG[s]) ok = false;
    }
    if (!ok) continue;
    seen.add(k);
    words.push(w);
  }
  const N = words.length;
  const counts = new Uint8Array(N * 26);
  const mask = new Int32Array(N);
  const len = new Uint8Array(N);
  const score = new Int32Array(N);
  const minLetter = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const w = words[i];
    let m = 0,
      base = 0,
      lo = 26;
    for (let j = 0; j < w.length; j++) {
      const s = w.charCodeAt(j) - A;
      counts[i * 26 + s]++;
      m |= 1 << s;
      base += VAL[s];
      if (s < lo) lo = s;
    }
    mask[i] = m;
    len[i] = w.length;
    score[i] = reward(base, w.length);
    minLetter[i] = lo;
  }
  return { words, N, counts, mask, len, score, minLetter };
}

export function emptyRack(): Rack {
  return new Uint8Array(26);
}
export function rackFromString(s: string): Rack {
  const r = emptyRack();
  for (const ch of s.toUpperCase()) {
    const i = ch.charCodeAt(0) - A;
    if (i >= 0 && i < 26) r[i]++;
  }
  return r;
}
export function rackToString(r: Rack): string {
  let s = "";
  for (let i = 0; i < 26; i++) s += String.fromCharCode(A + i).repeat(r[i]);
  return s;
}
export function rackSize(r: Rack): number {
  let t = 0;
  for (let i = 0; i < 26; i++) t += r[i];
  return t;
}
export function letterIndex(letter: string): number {
  return letter.toUpperCase().charCodeAt(0) - A;
}
export function canSpell(word: string, r: Rack): boolean {
  const c = emptyRack();
  for (let j = 0; j < word.length; j++) {
    const s = word.charCodeAt(j) - A;
    if (++c[s] > r[s]) return false;
  }
  return true;
}
export function subtractWord(r: Rack, word: string): Rack {
  const out = Uint8Array.from(r);
  for (let j = 0; j < word.length; j++) out[word.charCodeAt(j) - A]--;
  return out;
}

// All dictionary words spellable from the rack.
export function playable(D: Dict, r: Rack, minLen = 2): number[] {
  let rm = 0,
    tot = 0;
  for (let i = 0; i < 26; i++) {
    if (r[i]) rm |= 1 << i;
    tot += r[i];
  }
  const out: number[] = [];
  const { N, mask, len, counts } = D;
  for (let i = 0; i < N; i++) {
    if ((mask[i] & ~rm) !== 0 || len[i] > tot || len[i] < minLen) continue;
    const b = i * 26;
    let ok = true;
    for (let s = 0; s < 26; s++)
      if (counts[b + s] > r[s]) {
        ok = false;
        break;
      }
    if (ok) out.push(i);
  }
  return out;
}

export interface Partition {
  total: number;
  words: string[];
  leftover: Rack;
}

// Best set of disjoint words (max total reward). Exact when K is Infinity;
// with a finite K, only the top K words per state are tried.
export function bestPartition(D: Dict, r: Rack, K = Infinity, minLen = 2): Partition {
  const cand = playable(D, r, minLen);
  const buckets: number[][] = Array.from({ length: 26 }, () => []);
  for (const i of cand) buckets[D.minLetter[i]].push(i);
  for (const b of buckets) b.sort((x, y) => D.score[y] - D.score[x]);
  const stride = new Array<number>(26);
  let s = 1;
  for (let i = 0; i < 26; i++) {
    stride[i] = s;
    s *= r[i] + 1;
  }
  const memo = new Map<number, number>();
  const choice = new Map<number, number>();
  const cur = Uint8Array.from(r);
  const { counts, score, mask } = D;
  let curMask = 0,
    curVal = 0;
  for (let i = 0; i < 26; i++)
    if (r[i]) {
      curMask |= 1 << i;
      curVal += r[i] * VAL[i];
    }
  const f = (idx: number, cm: number, cv: number): number => {
    if (cv === 0) return 0;
    const m = memo.get(idx);
    if (m !== undefined) return m;
    let lo = 0;
    while (cur[lo] === 0) lo++;
    let best = 0,
      bestW = -1,
      tried = 0;
    for (const w of buckets[lo]) {
      if (tried >= K) break;
      if ((mask[w] & ~cm) !== 0) continue;
      const sw = score[w];
      const b = w * 26;
      let ok = true,
        d = 0,
        wv = 0;
      for (let t = lo; t < 26; t++) {
        const c = counts[b + t];
        if (c > cur[t]) {
          ok = false;
          break;
        }
        d += c * stride[t];
        wv += c * VAL[t];
      }
      if (!ok) continue;
      tried++;
      if (sw + 3 * (cv - wv) <= best) continue;
      let nm = cm;
      for (let t = lo; t < 26; t++) {
        const c = counts[b + t];
        if (c) {
          cur[t] -= c;
          if (cur[t] === 0) nm &= ~(1 << t);
        }
      }
      const v = sw + f(idx - d, nm, cv - wv);
      for (let t = lo; t < 26; t++) cur[t] += counts[b + t];
      if (v > best) {
        best = v;
        bestW = w;
      }
    }
    // Leave one copy of `lo` unplayed.
    if (3 * (cv - VAL[lo]) > best) {
      cur[lo]--;
      const v = f(idx - stride[lo], cur[lo] ? cm : cm & ~(1 << lo), cv - VAL[lo]);
      cur[lo]++;
      if (v > best) {
        best = v;
        bestW = -1;
      }
    }
    memo.set(idx, best);
    choice.set(idx, bestW);
    return best;
  };
  let idx0 = 0;
  for (let i = 0; i < 26; i++) idx0 += r[i] * stride[i];
  const total = f(idx0, curMask, curVal);
  const words: string[] = [];
  const c2 = Uint8Array.from(r);
  const left = emptyRack();
  let idx = idx0;
  for (;;) {
    let lo = 0;
    while (lo < 26 && c2[lo] === 0) lo++;
    if (lo === 26) break;
    const w = choice.get(idx);
    if (w === undefined) break;
    if (w === -1) {
      c2[lo]--;
      left[lo]++;
      idx -= stride[lo];
      continue;
    }
    words.push(D.words[w]);
    for (let t = 0; t < 26; t++) {
      c2[t] -= counts[w * 26 + t];
      idx -= counts[w * 26 + t] * stride[t];
    }
  }
  for (let t = 0; t < 26; t++) left[t] += c2[t];
  return { total, words, leftover: left };
}

export function wordReward(word: string): number {
  let base = 0;
  for (let j = 0; j < word.length; j++) base += VAL[word.charCodeAt(j) - A];
  return reward(base, word.length);
}

// Pick K by rack size so a solve stays within a few ms.
export function solveBudgetK(size: number): number {
  if (size <= 14) return Infinity;
  if (size <= 18) return 32;
  if (size <= 22) return 16;
  return 8;
}

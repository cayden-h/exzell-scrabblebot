// Fit behavioral clones of top rival bots from public post-match data.
// Input: docs/research/data/clones/{ar,ab,wp,mp,keep,bots}.json (pulled by
// docs/research/data/clones/pull.py from auction_result, auction_top_bid_archive,
// word_play, match_participant, match_state, bot; ended matches only).
// Output: bot/data/clones.json and a validation summary on stdout.
//
// npx tsx src/clonefit.ts [--holdout 0.25] [--zero] [--tight] [--dump]

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { VAL, emptyRack, loadDict, rackSize, type Dict, type Rack } from "./engine.js";
import { BUCKETS, cloneFeatures, letterClass, type CloneModel, type CloneWordPolicy } from "./clones.js";
import { partitionValue } from "./strategy.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(here, "..", "..", "docs", "research", "data", "clones");
const load = (f: string) => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8"));

// Bot id and first match id of the current behavior regime. Bron's and
// LUCENT's winning bid per face value rose sharply mid-day (Bron early bid/face
// 6 -> 11 from match ~42738, LUCENT 8 -> 12 from ~42701), so older matches are
// a different policy (or a much softer field) and are left out.
const TARGETS: Record<string, [number, number]> = {
  LUCENT: [18, 42701],
  Bron: [114, 42738],
  GodOfThudner: [171, 42130],
  exzell: [179, 42130],
};

interface Row {
  m: number;
  left: number;
  L: number;
  bal: number;
  held: number;
  vf: number; // vowel fraction of known rack (0.4 if fewer than 3 known tiles)
  gain: number; // known-rack partition gain from adding L
  kind: "win" | "loss" | "unsold";
  y: number; // win: own top bid; loss: paid (upper bound)
  others: number; // win: paid (best other bid); loss: winner's top bid
}

interface Word {
  m: number;
  left: number; // auctions left (incl. the open one) when played
  len: number;
  reward: number;
  balBefore: number;
}

interface MatchStat {
  m: number;
  score: number;
  won: number;
  paid: number;
  firstWordLeft: number; // 0 = never
  words: number;
  endBal: number;
}

function reconstruct(D: Dict, botId: number, from: number) {
  const ar: any[] = load("ar.json");
  const ab = new Map<number, number>((load("ab.json") as any[]).map((x) => [x.auction_id, x.top_bid]));
  const wp: any[] = load("wp.json");
  const mp: any[] = load("mp.json");
  const keep = new Set<number>(load("keep.json"));
  const myMatches = new Set<number>(mp.filter((p) => p.bot_id === botId && keep.has(p.match_id) && p.match_id >= from).map((p) => p.match_id));
  const byM = new Map<number, any[]>();
  for (const x of ar) if (myMatches.has(x.match_id)) (byM.get(x.match_id) ?? byM.set(x.match_id, []).get(x.match_id)!).push(x);
  const wpm = new Map<number, any[]>();
  for (const w of wp) if (w.bot_id === botId && myMatches.has(w.match_id)) (wpm.get(w.match_id) ?? wpm.set(w.match_id, []).get(w.match_id)!).push(w);
  const rows: Row[] = [];
  const words: Word[] = [];
  const stats: MatchStat[] = [];
  const pvCache = new Map<string, number>();
  const pv = (r: Rack) => {
    const k = r.join(",");
    let v = pvCache.get(k);
    if (v === undefined) {
      v = rackSize(r) > 22 ? 0 : partitionValue(D, r);
      pvCache.set(k, v);
    }
    return v;
  };
  for (const [m, L0] of byM) {
    const L = L0.sort((a, b) => a.auction_id - b.auction_id);
    if (L.length !== 68) continue;
    const plays = (wpm.get(m) ?? []).sort((a, b) => a.played_at[0] - b.played_at[0]);
    const known = emptyRack(); // won - played, clamped (excess came from hidden tiles)
    let bal = 100,
      pi = 0,
      won = 0,
      paidSum = 0,
      first = 0,
      score = 0;
    const playUpTo = (t: number, left: number) => {
      while (pi < plays.length && plays[pi].played_at[0] < t) {
        const w = plays[pi++];
        words.push({ m, left, len: w.word.length, reward: w.total_reward, balBefore: bal });
        bal += w.total_reward;
        score += w.total_reward;
        if (!first) first = left;
        for (const ch of w.word as string) {
          const i = ch.charCodeAt(0) - 65;
          if (known[i] > 0) known[i]--;
        }
      }
    };
    for (let i = 0; i < 68; i++) {
      const x = L[i];
      const left = 68 - i;
      playUpTo(x.closed_at[0], left);
      const li = (x.letter as string).charCodeAt(0) - 65;
      const w = x.winner_bot_id[0] === 0 ? x.winner_bot_id[1] : null;
      const top = ab.get(x.auction_id) ?? 0;
      let vow = 0;
      for (const v of [0, 4, 8, 14, 20]) vow += known[v];
      const ks = rackSize(known);
      const withL = Uint8Array.from(known);
      withL[li]++;
      const gain = Math.max(0, pv(withL) - pv(known));
      const base = { m, left, L: li, bal: Math.max(0, bal), held: known[li], vf: ks >= 3 ? vow / ks : 0.4, gain };
      if (w === botId) {
        rows.push({ ...base, kind: "win", y: top, others: x.paid });
        bal -= x.paid;
        paidSum += x.paid;
        won++;
        known[li]++;
      } else if (w === null) rows.push({ ...base, kind: "unsold", y: 0, others: 0 });
      else rows.push({ ...base, kind: "loss", y: x.paid, others: top });
    }
    playUpTo(Infinity, 0);
    const fin = mp.find((p) => p.match_id === m && p.bot_id === botId);
    stats.push({ m, score: fin?.score ?? score, won, paid: paidSum, firstWordLeft: first, words: plays.length, endBal: fin?.balance ?? bal });
  }
  return { rows, words, stats };
}

// ---------- Model ----------
// mu = a[L] + ph[bucket] + cb*ln(bal/60) + h1*[held==1] + h2*[held>=2] + cv*vowelTilt + cf*ln(1+gain)
// zero-bid logit: z0 + zph[bucket] + zh*held + zc[class]
// Params vector layout.
const NA = 26,
  NB = BUCKETS.length;
const IX = {
  a: 0,
  ph: NA,
  cb: NA + NB,
  h1: NA + NB + 1,
  h2: NA + NB + 2,
  cv: NA + NB + 3,
  cf: NA + NB + 4,
  ls: NA + NB + 5,
  z0: NA + NB + 6,
  zph: NA + NB + 7,
  zh: NA + 2 * NB + 7,
  zc: NA + 2 * NB + 8,
};
const NP = IX.zc + 4;
// --zero: also fit the zero-bid mass. Off by default: with it the fit collapses
// to a narrow bid distribution plus a large no-bid mass, which under-predicts wins.
const NOZERO = !process.argv.includes("--zero");
// --tight: censor losses at paid (the exact bound). Default censors at the
// winner's top bid, a looser but still valid bound that keeps the fit
// calibrated on win/loss outcomes when bids correlate across bots.
const TIGHT = process.argv.includes("--tight");

function toModel(th: number[]): CloneModel {
  return {
    a: th.slice(IX.a, IX.a + NA),
    ph: th.slice(IX.ph, IX.ph + NB),
    cb: th[IX.cb],
    h1: th[IX.h1],
    h2: th[IX.h2],
    cv: th[IX.cv],
    cf: th[IX.cf],
    sigma: Math.exp(th[IX.ls]),
    z0: th[IX.z0],
    zph: th.slice(IX.zph, IX.zph + NB),
    zh: th[IX.zh],
    zc: th.slice(IX.zc, IX.zc + 4),
  };
}

const SQ2 = Math.SQRT2;
function erfc(x: number): number {
  // Numerical Recipes erfcc, |err| < 1.2e-7
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}
const Phi = (u: number) => 0.5 * erfc(-u / SQ2);
const phi = (u: number) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);

function rowFeat(r: Row) {
  return cloneFeatures(r.L, r.left, r.bal, r.held, r.vf, r.gain);
}

// Negative log-likelihood and gradient.
function nll(th: number[], rows: Row[], feats: ReturnType<typeof rowFeat>[], l2: number) {
  const g = new Array(NP).fill(0);
  let f = 0;
  const sig = Math.exp(th[IX.ls]);
  const E = 1e-12;
  for (let n = 0; n < rows.length; n++) {
    const r = rows[n];
    const F = feats[n];
    const mu = th[IX.a + r.L] + th[IX.ph + F.b] + th[IX.cb] * F.lb + th[IX.h1] * F.h1 + th[IX.h2] * F.h2 + th[IX.cv] * F.vt + th[IX.cf] * F.lg;
    const z = th[IX.z0] + th[IX.zph + F.b] + th[IX.zh] * F.held + th[IX.zc + F.cls];
    const q = 1 / (1 + Math.exp(-z));
    let dmu = 0,
      dls = 0,
      dz = 0,
      lp = 0;
    if (r.kind === "win") {
      if (r.y >= r.bal - 0.5) {
        const u = (Math.log(Math.max(0.5, r.bal - 0.5)) - mu) / sig;
        const S = Math.max(E, 1 - Phi(u));
        lp = Math.log(1 - q + E) + Math.log(S);
        dmu = phi(u) / (sig * S);
        dls = (phi(u) * u) / S;
        dz = -q;
      } else {
        const u1 = (Math.log(Math.max(0.5, r.y - 0.5)) - mu) / sig;
        const u2 = (Math.log(r.y + 0.5) - mu) / sig;
        const Dd = Math.max(E, Phi(u2) - Phi(u1));
        lp = Math.log(1 - q + E) + Math.log(Dd);
        dmu = -(phi(u2) - phi(u1)) / (sig * Dd);
        dls = -(phi(u2) * u2 - phi(u1) * u1) / Dd;
        dz = -q;
      }
    } else if (r.bal <= (TIGHT ? r.y : r.others)) {
      continue; // the balance cap alone explains the loss
    } else {
      const u = (Math.log((TIGHT || r.kind === "unsold" ? r.y : r.others) + 0.5) - mu) / sig;
      const P = Math.max(E, q + (1 - q) * Phi(u));
      lp = Math.log(P);
      dmu = (-(1 - q) * phi(u)) / sig / P;
      dls = (-(1 - q) * phi(u) * u) / P;
      dz = (q * (1 - q) * (1 - Phi(u))) / P;
    }
    f -= lp;
    g[IX.a + r.L] -= dmu;
    g[IX.ph + F.b] -= dmu;
    g[IX.cb] -= dmu * F.lb;
    g[IX.h1] -= dmu * F.h1;
    g[IX.h2] -= dmu * F.h2;
    g[IX.cv] -= dmu * F.vt;
    g[IX.cf] -= dmu * F.lg;
    g[IX.ls] -= dls;
    g[IX.z0] -= dz;
    g[IX.zph + F.b] -= dz;
    g[IX.zh] -= dz * F.held;
    g[IX.zc + F.cls] -= dz;
  }
  // Light ridge on everything except letter bases and sigma; pin ph[0] and zph[0] to 0.
  for (let i = NA; i < NP; i++) {
    if (i === IX.ls) continue;
    f += l2 * th[i] * th[i];
    g[i] += 2 * l2 * th[i];
  }
  g[IX.ph] = 0;
  g[IX.zph] = 0;
  if (NOZERO) for (let i = IX.z0; i < NP; i++) g[i] = 0;
  return { f, g };
}

function fit(rows: Row[]): number[] {
  const feats = rows.map(rowFeat);
  const th = new Array(NP).fill(0);
  for (let i = 0; i < 26; i++) th[IX.a + i] = Math.log(Math.max(4, 3 * VAL[i]));
  th[IX.ls] = Math.log(0.5);
  th[IX.z0] = NOZERO ? -30 : -1;
  // Adam
  const m = new Array(NP).fill(0),
    v = new Array(NP).fill(0);
  const lr = 0.03,
    b1 = 0.9,
    b2 = 0.999;
  const N = rows.length;
  let last = Infinity;
  for (let it = 1; it <= 1500; it++) {
    const { f, g } = nll(th, rows, feats, 2);
    for (let i = 0; i < NP; i++) {
      const gi = g[i] / N;
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      th[i] -= (lr * (m[i] / (1 - b1 ** it))) / (Math.sqrt(v[i] / (1 - b2 ** it)) + 1e-8);
    }
    if (it % 300 === 0) {
      if (Math.abs(last - f) / N < 1e-6) break;
      last = f;
    }
  }
  return th;
}

// ---------- Validation ----------
function evaluate(model: CloneModel, rows: Row[], baseline: number[]) {
  // MAE of the predicted bid (median, conditional on bidding) on observed wins,
  // vs a per-letter median-of-training-wins baseline. And per-letter win share:
  // P(bid > best other bid) summed, vs actual wins.
  let mae = 0,
    maeB = 0,
    maeC = 0,
    n = 0;
  const predW = new Array(26).fill(0),
    actW = new Array(26).fill(0),
    cnt = new Array(26).fill(0);
  for (const r of rows) {
    const F = rowFeat(r);
    const { mu, q } = muQ(model, F, r.L);
    if (r.kind === "win") {
      const pred = Math.min(r.bal, Math.exp(mu));
      mae += Math.abs(pred - r.y);
      maeB += Math.abs(Math.min(r.bal, baseline[r.L]) - r.y);
      // Median bid conditional on beating the best other bid (the selection the data has).
      const u0 = Phi((Math.log(r.others + 0.5) - mu) / model.sigma);
      maeC += Math.abs(Math.min(r.bal, Math.exp(mu + model.sigma * invPhi((1 + u0) / 2))) - r.y);
      n++;
      actW[r.L]++;
    }
    // Probability the clone's bid beats the best other bid (others), capped by balance.
    const thr = r.kind === "unsold" ? 0.5 : r.others + 0.5;
    const pw = thr > r.bal ? 0 : (1 - q) * (1 - Phi((Math.log(thr) - mu) / model.sigma));
    predW[r.L] += pw;
    cnt[r.L]++;
  }
  return { mae: mae / n, maeB: maeB / n, maeC: maeC / n, n, predW, actW, cnt };
}

function invPhi(p: number): number {
  let lo = -8,
    hi = 8;
  for (let i = 0; i < 50; i++) {
    const m = (lo + hi) / 2;
    if (Phi(m) < p) lo = m;
    else hi = m;
  }
  return (lo + hi) / 2;
}

function muQ(c: CloneModel, F: ReturnType<typeof rowFeat>, L: number) {
  const mu = c.a[L] + c.ph[F.b] + c.cb * F.lb + c.h1 * F.h1 + c.h2 * F.h2 + c.cv * F.vt + c.cf * F.lg;
  const z = c.z0 + c.zph[F.b] + c.zh * F.held + c.zc[F.cls];
  return { mu, q: 1 / (1 + Math.exp(-z)) };
}

function quant(v: number[], q: number) {
  const s = [...v].sort((a, b) => a - b);
  if (!s.length) return NaN;
  const k = (s.length - 1) * q;
  const lo = Math.floor(k);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (k - lo);
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);

function wordSummary(words: Word[], stats: MatchStat[]) {
  const lens = words.map((w) => w.len);
  const share = (f: (w: Word) => boolean) => +(words.filter(f).length / Math.max(1, words.length)).toFixed(3);
  return {
    wordsPerMatch: +(words.length / stats.length).toFixed(2),
    lenP: { p25: quant(lens, 0.25), p50: quant(lens, 0.5), p75: quant(lens, 0.75) },
    share7Plus: share((w) => w.len >= 7),
    shareLe4: share((w) => w.len <= 4),
    shareFlush: share((w) => w.left <= 3),
    leftAtPlay: { p10: quant(words.map((w) => w.left), 0.1), p50: quant(words.map((w) => w.left), 0.5), p90: quant(words.map((w) => w.left), 0.9) },
    balBeforeMid: quant(words.filter((w) => w.left > 3).map((w) => w.balBefore), 0.5),
    balBeforeMidP25: quant(words.filter((w) => w.left > 3).map((w) => w.balBefore), 0.25),
    midLen: quant(words.filter((w) => w.left > 3).map((w) => w.len), 0.5),
    flushLen: quant(words.filter((w) => w.left <= 3).map((w) => w.len), 0.5),
    firstWordLeft: quant(stats.map((s) => s.firstWordLeft), 0.5),
    pointsPerWord: +mean(words.map((w) => w.reward)).toFixed(1),
  };
}

function liveStats(stats: MatchStat[]) {
  const won = stats.reduce((a, s) => a + s.won, 0);
  return {
    matches: stats.length,
    score: +mean(stats.map((s) => s.score)).toFixed(1),
    tilesWon: +(won / stats.length).toFixed(1),
    paidPerTile: +(stats.reduce((a, s) => a + s.paid, 0) / won).toFixed(2),
    endBal: quant(stats.map((s) => s.endBal), 0.5),
    firstWordLeft: quant(stats.map((s) => s.firstWordLeft), 0.5),
  };
}

// Rack-fit share of sigma (clones.ts), set from clonesim sweeps (research 13):
// rho 0.6 to 0.9 brings the Bron clone's tiles and price close to live; LUCENT
// and GodOfThudner clones score best (closest to live) at 0.
const RHO: Record<string, number> = { Bron: 0.6, LUCENT: 0, GodOfThudner: 0, exzell: 0 };

function main() {
  const D = loadDict(path.join(here, "..", "data", "wordlist.txt"));
  const hold = Number(process.argv[process.argv.indexOf("--holdout") + 1] || 0.25) || 0.25;
  const out: Record<string, any> = { _meta: { note: "Fitted by src/clonefit.ts from public post-match tables (ended matches 42130-42977, 2026-11-07, >=4 real bots).", buckets: BUCKETS } };
  for (const [name, [id, from]] of Object.entries(TARGETS)) {
    const t0 = performance.now();
    const { rows, words, stats } = reconstruct(D, id, from);
    if (process.argv.includes("--dump")) fs.writeFileSync(`/tmp/rows_${name}.json`, JSON.stringify(rows));
    const ms = [...new Set(rows.map((r) => r.m))].sort((a, b) => a - b);
    const cut = ms[Math.floor(ms.length * (1 - hold))];
    const train = rows.filter((r) => r.m < cut),
      test = rows.filter((r) => r.m >= cut);
    // Baseline: per-letter median of training wins.
    const base = Array.from({ length: 26 }, (_, i) => {
      const v = train.filter((r) => r.kind === "win" && r.L === i).map((r) => r.y);
      return v.length ? quant(v, 0.5) : 3 * VAL[i];
    });
    const thT = fit(train);
    const ev = evaluate(toModel(thT), test, base);
    const evIn = evaluate(toModel(thT), train, base);
    const thAll = fit(rows);
    const model = toModel(thAll);
    const ws = wordSummary(words, stats);
    const live = liveStats(stats);
    const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
    const letterTable = Array.from({ length: 26 }, (_, i) => ({
      L: String.fromCharCode(65 + i),
      n: ev.cnt[i],
      act: ev.actW[i],
      pred: +ev.predW[i].toFixed(1),
    }));
    // Coarse share check by letter class on the test set.
    const cls: Record<string, { act: number; pred: number; n: number }> = {};
    for (const t of letterTable) {
      const k = ["vowel", "1pt", "mid", "high"][letterClass(t.L.charCodeAt(0) - 65)];
      cls[k] = cls[k] ?? { act: 0, pred: 0, n: 0 };
      cls[k].act += t.act;
      cls[k].pred += t.pred;
      cls[k].n += t.n;
    }
    console.log(
      `\n== ${name} (${stats.length} matches, ${rows.length} auctions, train ${train.length} test ${test.length}, ${(performance.now() - t0).toFixed(0)} ms)`,
    );
    console.log("live", JSON.stringify(live));
    console.log("words", JSON.stringify(ws));
    console.log(
      `held-out MAE winning bid: model conditional on winning ${ev.maeC.toFixed(2)}, unconditional ${ev.mae.toFixed(2)} vs per-letter median ${ev.maeB.toFixed(2)} (n=${ev.n}); in-sample model ${evIn.mae.toFixed(2)}`,
    );
    {
      const byB = BUCKETS.map(() => [0, 0]);
      for (const r of train) {
        const F = rowFeat(r);
        const { mu, q } = muQ(toModel(thT), F, r.L);
        const thr = r.kind === "unsold" ? 0.5 : r.others + 0.5;
        byB[F.b][1] += thr > r.bal ? 0 : (1 - q) * (1 - Phi((Math.log(thr) - mu) / toModel(thT).sigma));
        if (r.kind === "win") byB[F.b][0]++;
      }
      console.log("in-sample wins by bucket act/pred", byB.map((x) => `${x[0]}/${x[1].toFixed(0)}`).join(" "));
    }
    console.log(`held-out wins: actual ${sum(ev.actW)} predicted ${sum(ev.predW).toFixed(1)}`);
    console.log("by class (act/pred/n)", JSON.stringify(Object.fromEntries(Object.entries(cls).map(([k, v]) => [k, [v.act, +v.pred.toFixed(1), v.n]]))));
    console.log("by letter (act/pred)", letterTable.map((t) => `${t.L}:${t.act}/${t.pred}`).join(" "));
    console.log(
      "model",
      JSON.stringify({
        base: Object.fromEntries(model.a.map((x, i) => [String.fromCharCode(65 + i), +Math.exp(x).toFixed(1)])),
        ph: model.ph.map((x) => +Math.exp(x).toFixed(2)),
        cb: +model.cb.toFixed(3),
        h1: +Math.exp(model.h1).toFixed(2),
        h2: +Math.exp(model.h2).toFixed(2),
        cv: +model.cv.toFixed(2),
        cf: +model.cf.toFixed(3),
        sigma: +model.sigma.toFixed(2),
        pz: { z0: +model.z0.toFixed(2), zph: model.zph.map((x) => +x.toFixed(2)), zh: +model.zh.toFixed(2), zc: model.zc.map((x) => +x.toFixed(2)) },
      }),
    );
    const wp: CloneWordPolicy = wordPolicyFrom(ws);
    out[name] = {
      botId: id,
      model,
      rho: RHO[name] ?? 0,
      words: wp,
      live,
      wordStats: ws,
      validation: {
        heldOutMatches: ms.length - ms.indexOf(cut),
        maeModel: +ev.mae.toFixed(2),
        maeModelGivenWin: +ev.maeC.toFixed(2),
        maeLetterMedian: +ev.maeB.toFixed(2),
        winsActual: sum(ev.actW),
        winsPred: +sum(ev.predW).toFixed(1),
        byClass: cls,
      },
    };
  }
  fs.writeFileSync(path.join(here, "..", "data", "clones.json"), JSON.stringify(out, null, 1));
  console.log("\nwrote data/clones.json");
}

// Starting word policy from live stats; clonesim tunes cashFloor/minLen against live.
function wordPolicyFrom(ws: ReturnType<typeof wordSummary>): CloneWordPolicy {
  const minLen = ws.midLen >= 7 ? 7 : Math.max(4, Math.round(ws.midLen));
  return { minLen, flushAt: 2, cashFloor: Math.round(ws.balBeforeMidP25 || 15), cashAny: ws.share7Plus < 0.6 ? 1 : 0 };
}

main();

// Bidding and word-play decisions. Pure functions over plain state, so the
// live bot and an offline simulator can share them.
// Design notes: docs/PLAN.md, docs/research/02-bidding-strategy.md, 03-word-play.md.

import {
  VAL,
  bestPartition,
  playable,
  rackSize,
  solveBudgetK,
  wordReward,
  type Dict,
  type Rack,
} from "./engine.js";
import { mcBestCash, mcLetterValue, type McOpts } from "./valuation.js";

export interface Params {
  lambda: number; // bid multiplier on point value (Vickrey)
  lambdaFirstPrice: number; // multiplier when the auction is first-price
  dup: number[]; // value factor by copies already held
  vowHi: number; // vowel fraction above which vowels are discounted
  vowLo: number; // vowel fraction below which consonants are discounted
  vowPen: number;
  consPen: number;
  endgame: number; // last N auctions: use exact marginal partition value
  endgameLambda: number; // multiplier in the endgame (money is worthless at the end)
  cashFloor: number; // below this balance, cash a word to fund bids
  cashMinLen: number; // shortest word we cash mid-match
  rackCap: number; // above this rack size, cash the best 7+ word
  flushAt: number; // auctions left (incl. current) when we play everything
  simMargin: number; // bid this much above the highest sim cap
  // Letter valuation: "table" = fixed T table with rack adjustments,
  // "mc" = bag-aware Monte Carlo (valuation.ts); falls back to table when
  // the state has no `unseen` pool.
  valueMode: "table" | "mc";
  mcSamples: number; // future draws per bid
  mcShare: number; // share of the remaining bag we expect to win
  mcFmax: number; // cap on future tiles per draw
  mcK: number; // partition breadth cap for large racks
  mcScale: number; // multiplier on the MC value (keeps lambda comparable)
  mcBlend: number; // 1 = pure MC, 0 = pure table
  cashMode: "table" | "mc"; // which 7+ word to cash: highest reward, or least expected loss
  // Cash a cashMinLen+ word from the best partition whenever auctionsLeft is
  // at or below this, regardless of balance (68 = always; 0 = off).
  cashAlwaysLeft: number;
  // Per-letter bid floor (outside the endgame), applied only while we hold
  // fewer than floorMaxCopies of the letter. floorOn 0 disables.
  floorOn: number;
  floors: Record<string, number>;
  floorMaxCopies: number;
  // Per-letter bid cap. capOn 0 disables. Q's cap applies only with a U held;
  // without a U, Q stays at its table value.
  capOn: number;
  caps: Record<string, number>;
  // Bid multipliers by phase: early = auctionsLeft > 45, late = auctionsLeft <= 15.
  phaseEarly: number;
  phaseLate: number;
  // Phase-dependent face-value floor (research 09): bid at least
  // f(phase) * face value, with f = faceFloorEarly (> 45 left), faceFloorMid,
  // faceFloorLate (<= 15 left). Applied outside the endgame while we hold
  // fewer than floorMaxCopies of the letter. 0 = off.
  faceFloorEarly: number;
  faceFloorMid: number;
  faceFloorLate: number;
  faceFloorMaxFace: number; // face floor only on letters with face value <= this
  // First-price branch only (research 09). fpFloorScale multiplies the
  // per-letter floors (0 = no floors in first-price). fpPhaseEarly/Late
  // replace phaseEarly/Late. endgameLambdaFirstPrice replaces endgameLambda.
  fpFloorScale: number;
  fpPhaseEarly: number;
  fpPhaseLate: number;
  endgameLambdaFirstPrice: number;
  // Endgame MC (research 09, idea from research 12): in the endgame branch,
  // when `unseen` is known, value the tile by mcLetterValue with
  // endgameMcSamples draws of round(endgameShare * (auctionsLeft - flushAt - 1))
  // future tiles (capped at mcFmax), instead of the exact partition difference.
  endgameMc: number; // 0 = off
  endgameShare: number;
  endgameMcSamples: number;
}

export const DEFAULT_PARAMS: Params = {
  lambda: 2.8,
  lambdaFirstPrice: 2.0, // research 09: 1.6 left ~55 unspent in first-price; 2.0 is robust to rival shading
  dup: [1, 0.8, 0.5, 0.25, 0.1],
  vowHi: 0.5,
  vowLo: 0.25,
  vowPen: 0.7,
  consPen: 0.75,
  endgame: 3,
  endgameLambda: 4,
  cashFloor: 20,
  cashMinLen: 7,
  rackCap: 16,
  flushAt: 2,
  simMargin: 1,
  valueMode: "table",
  mcSamples: 8,
  mcShare: 0.25,
  mcFmax: 6,
  mcK: 24,
  mcScale: 1,
  mcBlend: 1,
  cashMode: "table",
  cashAlwaysLeft: 0,
  floorOn: 0, // research 09: floors cost ~4 pts in the calibrated model and live (1-pt tiles 0.61 -> 0.75)
  floors: { A: 10, E: 10, I: 10, O: 10, L: 10, N: 10, R: 10, T: 10, U: 9, S: 12, D: 12, G: 12 },
  floorMaxCopies: 2,
  capOn: 0,
  caps: { M: 17, B: 17, P: 17, V: 17, C: 18, W: 18, F: 19, Y: 19, H: 20, K: 22, J: 26, X: 28, Z: 30, Q: 18 },
  phaseEarly: 0.9,
  phaseLate: 1.2,
  faceFloorEarly: 0,
  faceFloorMid: 0,
  faceFloorLate: 0,
  faceFloorMaxFace: 10,
  fpFloorScale: 0, // research 09: floors in first-price pay full price on 1-pt letters (74 vs 122 pts)
  fpPhaseEarly: 0.9,
  fpPhaseLate: 1.2,
  endgameLambdaFirstPrice: 2,
  endgameMc: 0,
  endgameShare: 0.25,
  endgameMcSamples: 16,
};

function mcOpts(p: Params, bagLeft: number): McOpts {
  return {
    samples: p.mcSamples,
    future: Math.max(0, Math.min(p.mcFmax, Math.round(p.mcShare * bagLeft))),
    K: p.mcK,
  };
}

// Marginal point value of one more tile on a ~9-tile rack (research 02, Monte Carlo).
const T: Record<string, number> = {
  Z: 16.7, X: 11.5, J: 10.9, K: 9.8, H: 9.3, Y: 8.7, F: 7.8, M: 7.7, C: 7.4, W: 6.6,
  P: 6.5, B: 6.2, D: 6.0, G: 6.0, S: 5.8, V: 5.7, A: 5.6, Q: 5.5, E: 5.3, R: 4.7,
  O: 4.7, N: 4.6, L: 4.5, U: 4.3, I: 4.0, T: 3.6,
};
const VOWELS = new Set(["A", "E", "I", "O", "U"]);
const IDX = (l: string) => l.charCodeAt(0) - 65;

export type SimKind = "Cheapskate" | "ValueBidder" | "Aggressive";

export interface Opponent {
  botId: bigint;
  balance: number;
  score: number;
  sim: SimKind | null; // null = real bot
}

export interface BidState {
  letter: string;
  rack: Rack;
  balance: number;
  auctionsLeft: number; // including this one
  firstPrice: boolean;
  opponents: Opponent[];
  // Tiles not yet seen by us: full 98-tile bag minus every letter auctioned
  // so far in this match (including the current one) minus our 5 starting
  // tiles. They are in the bag or in opponents' face-down racks.
  unseen?: Uint8Array;
  bagLeft?: number; // tiles still in the bag after the current auction's letter
}

// Point value of the letter to us, from the table plus rack adjustments.
export function letterValue(letter: string, rack: Rack, p: Params): number {
  const k = rack[IDX(letter)];
  let v = (T[letter] ?? 4) * p.dup[Math.min(k, p.dup.length - 1)];
  const size = rackSize(rack);
  if (size >= 3) {
    let vowels = 0;
    for (const x of VOWELS) vowels += rack[IDX(x)];
    const f = vowels / size;
    if (VOWELS.has(letter) && f >= p.vowHi) v *= p.vowPen;
    if (!VOWELS.has(letter) && f <= p.vowLo) v *= p.consPen;
  }
  const hasU = rack[IDX("U")] > 0;
  if (letter === "Q" && !hasU) v = 2;
  if (letter === "U" && rack[IDX("Q")] > 0 && !hasU) v += 4;
  return v;
}

// Highest bid a sim can place on this letter (game.ts: base by strategy, +-2 jitter, capped by balance).
function simCap(letter: string, o: Opponent): number {
  const face = VAL[IDX(letter)];
  let base = face;
  if (o.sim === "Cheapskate") base = face - 1;
  if (o.sim === "Aggressive") base = face + (VOWELS.has(letter) ? 4 : 2);
  return Math.max(0, Math.min(base + 2, o.balance));
}

export function partitionValue(D: Dict, rack: Rack): number {
  return bestPartition(D, rack, solveBudgetK(rackSize(rack))).total;
}

export function decideBid(D: Dict, s: BidState, p: Params = DEFAULT_PARAMS): number {
  if (s.balance <= 0) return 0;
  // The last tile can never be played: the match ends as its auction closes.
  if (s.auctionsLeft <= 1) return 0;

  let v: number;
  let lambda = s.firstPrice ? p.lambdaFirstPrice : p.lambda;
  if (s.auctionsLeft <= p.endgame + 1) {
    // Endgame: value is what the tile adds to the final partition.
    // Tiles won after the flush point are unlikely to be playable.
    if (s.auctionsLeft <= p.flushAt) return 0;
    const future = Math.max(0, Math.min(p.mcFmax, Math.round(p.endgameShare * (s.auctionsLeft - p.flushAt - 1))));
    if (p.endgameMc && s.unseen && future > 0) {
      v = mcLetterValue(D, s.rack, IDX(s.letter), s.unseen, { samples: p.endgameMcSamples, future, K: p.mcK });
    } else {
      const with_ = Uint8Array.from(s.rack);
      with_[IDX(s.letter)]++;
      v = partitionValue(D, with_) - partitionValue(D, s.rack);
    }
    if (v <= 0) return 0;
    // Money left at the end is worthless, so spend it on anything that scores.
    lambda = s.firstPrice ? p.endgameLambdaFirstPrice : p.endgameLambda;
    const perAuction = s.balance / Math.max(1, s.auctionsLeft - p.flushAt);
    return clampBid(Math.max(lambda * v, Math.min(perAuction * 1.5, v * 8)), s.balance);
  }

  v = letterValue(s.letter, s.rack, p);
  if (p.valueMode === "mc" && s.unseen) {
    const bagLeft = s.bagLeft ?? Math.max(0, s.auctionsLeft - 1);
    const mc = p.mcScale * mcLetterValue(D, s.rack, IDX(s.letter), s.unseen, mcOpts(p, bagLeft));
    v = p.mcBlend * mc + (1 - p.mcBlend) * v;
  }

  const real = s.opponents.filter((o) => o.sim === null);
  if (real.length === 0 && s.opponents.length > 0) {
    // Only sims left: win every useful tile at the sims' price.
    if (v < 2) return 0;
    const cap = Math.max(0, ...s.opponents.map((o) => simCap(s.letter, o)));
    return clampBid(cap + p.simMargin, s.balance);
  }

  // Pace: spend faster when cash is plentiful for the auctions left.
  const rich = s.balance / Math.max(1, s.auctionsLeft);
  const pace = rich > 6 ? 1.3 : rich < 1.5 ? 0.8 : 1;
  let b = lambda * v * pace;
  if (s.auctionsLeft > 45) b *= s.firstPrice ? p.fpPhaseEarly : p.phaseEarly;
  else if (s.auctionsLeft <= 15) b *= s.firstPrice ? p.fpPhaseLate : p.phaseLate;

  const held = s.rack[IDX(s.letter)];
  if (p.capOn) {
    const cap = p.caps[s.letter];
    if (cap !== undefined && (s.letter !== "Q" || s.rack[IDX("U")] > 0)) b = Math.min(b, cap);
  }
  if (p.floorOn && v > 0 && held < p.floorMaxCopies) {
    const fl = p.floors[s.letter];
    if (fl !== undefined) b = Math.max(b, s.firstPrice ? fl * p.fpFloorScale : fl);
  }
  const ff = s.auctionsLeft > 45 ? p.faceFloorEarly : s.auctionsLeft > 15 ? p.faceFloorMid : p.faceFloorLate;
  const face = VAL[IDX(s.letter)];
  if (ff > 0 && v > 0 && held < p.floorMaxCopies && face <= p.faceFloorMaxFace) b = Math.max(b, ff * face);

  // Cap-snipe: a real rival with little cash cannot outbid balance + 1.
  const maxRivalCash = Math.max(0, ...real.map((o) => o.balance));
  if (maxRivalCash + 1 < b) b = maxRivalCash + 1;
  // Sims in the lobby still bid; make sure we clear them on tiles we value.
  const sims = s.opponents.filter((o) => o.sim !== null);
  if (sims.length > 0 && v >= 4) {
    const cap = Math.max(...sims.map((o) => simCap(s.letter, o)));
    if (b <= cap && cap + 1 <= lambda * v * 1.5) b = cap + 1;
  }
  return clampBid(b, s.balance);
}

function clampBid(b: number, balance: number): number {
  return Math.max(0, Math.min(Math.round(b), balance));
}

export interface PlayState {
  rack: Rack;
  balance: number;
  auctionsLeft: number; // including the open one; 0 if unknown
  unseen?: Uint8Array; // same meaning as BidState.unseen
  bagLeft?: number; // same meaning as BidState.bagLeft
}

// Words to submit now. Empty means hold.
export function chooseWords(D: Dict, s: PlayState, p: Params = DEFAULT_PARAMS): string[] {
  const size = rackSize(s.rack);
  if (size < 2) return [];
  const part = bestPartition(D, s.rack, solveBudgetK(size));
  if (part.words.length === 0) return [];

  // Endgame flush: play the whole best partition.
  if (s.auctionsLeft > 0 && s.auctionsLeft <= p.flushAt) return part.words;

  const needCash = s.balance < p.cashFloor || (s.auctionsLeft > 0 && s.auctionsLeft <= p.cashAlwaysLeft);
  const tooBig = size > p.rackCap;
  if (!needCash && !tooBig) return [];

  // Cash the best long word from the best partition, so the rest stays optimal.
  const long = part.words
    .filter((w) => w.length >= p.cashMinLen)
    .sort((a, b) => wordReward(b) - wordReward(a));
  if (long.length > 0) {
    if (p.cashMode === "mc" && s.unseen) {
      // Any playable long word is a candidate, not only the one in the best
      // partition; keep the one that loses the least expected future value.
      const cands = playable(D, s.rack, p.cashMinLen)
        .sort((a, b) => D.score[b] - D.score[a])
        .slice(0, 12)
        .map((i) => D.words[i]);
      const bagLeft = s.bagLeft ?? Math.max(0, s.auctionsLeft - 1);
      const pick = mcBestCash(D, s.rack, cands, s.unseen, mcOpts(p, bagLeft));
      if (pick) return [pick.word];
    }
    return [long[0]];
  }
  // Broke with no long word: cash the best word worth something.
  if (s.balance < 10 && s.auctionsLeft > 8) {
    const best = [...part.words].sort((a, b) => wordReward(b) - wordReward(a))[0];
    if (wordReward(best) >= 10) return [best];
  }
  return [];
}

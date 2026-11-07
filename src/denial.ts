// Small-table denial bidding (research 10 tactic B, research 12 item 2).
// Report and tuning: docs/research/14-denial.md.
//
// At 2 to 4 seats under Vickrey, a tile a close rival wins is points against
// us, and a rival that beats our bid pays our bid. So on top of the base bid:
// - drain floor: bid at least drain * typical clearing price on every letter
//   (outside the endgame), so a rival pays near full price for what it wins;
// - denial: add denyA * lambda * (points the letter adds to a close rival's
//   known rack right now);
// - lead lock: when our projected lead over the rival that matters exceeds
//   what it can plausibly gain in the auctions left, fall back to the base bid.
// First-price is off by default (a rival pays its own bid there, so our floor
// does not raise its price); fpOn turns on the denial term only.

import { bestPartition, rackSize, solveBudgetK, type Dict, type Rack } from "./engine.js";
import type { BidState, Opponent } from "./strategy.js";

export interface DenialOpponent extends Opponent {
  knownRack?: Uint8Array; // tiles won minus letters played (26 counts)
  hidden?: number; // starting tiles not yet revealed by a play
}

export interface DenialState extends BidState {
  opponents: DenialOpponent[];
  seats: number; // bots at the table, including us
  myScore: number;
}

export interface DenialParams {
  maxSeats: number; // apply at seats <= maxSeats
  lambda: number; // coins per point for the denial term (match the base lambda)
  drain: number; // floor = drain * FIELD_PAID[letter]; 0 = off
  drainStop: number; // no drain floor when auctionsLeft <= drainStop
  drainMinBal: number; // no drain floor when our balance is below this
  denyA: number; // weight on the rival's immediate gain; 0 = off
  denyStop: number; // no denial when auctionsLeft <= denyStop (endgame owns it)
  hiddenVal: number; // points added to a rival's projection per hidden tile
  closeMargin: number; // 3-4 seats: deny only rivals within gain + this of us
  leadLockPpa: number; // lock when lead > leadLockPpa * auctionsLeft + leadLockPad
  leadLockPad: number;
  leadLockHeadsUp: number; // 1 = lead lock also heads-up (the final sums margins, so default 0)
  fpOn: number; // 1 = denial term also in first-price (never the drain floor)
  fpDrain: number; // drain share in first-price when fpOn (default 0)
}

export const DEFAULT_DENIAL: DenialParams = {
  maxSeats: 4,
  lambda: 2.8,
  drain: 0.6,
  drainStop: 12,
  drainMinBal: 0,
  denyA: 0.5,
  denyStop: 4,
  hiddenVal: 3,
  closeMargin: 30,
  leadLockPpa: 2,
  leadLockPad: 20,
  leadLockHeadsUp: 0,
  fpOn: 0,
  fpDrain: 0,
};

// Median clearing price (`paid`) by letter, real-bot matches (07-live-analysis.md).
export const FIELD_PAID: Record<string, number> = {
  A: 9, E: 9, I: 9, L: 9, N: 9, O: 9, R: 9, T: 9, U: 8, S: 10, D: 10.5, G: 10.5,
  B: 13.5, M: 13.5, P: 13.5, V: 13.5, C: 15, W: 15, F: 16.5, H: 16.5, Y: 16.5,
  K: 19, J: 23, X: 24.5, Q: 18, Z: 27,
};

// Partition values are re-asked for the same racks every auction; cache them.
const PCACHE = new Map<string, number>();
function key(r: Rack): string {
  let s = "";
  for (let i = 0; i < 26; i++) s += String.fromCharCode(48 + r[i]);
  return s;
}
export function cachedPartition(D: Dict, r: Rack): number {
  const k = key(r);
  let v = PCACHE.get(k);
  if (v === undefined) {
    v = bestPartition(D, r, solveBudgetK(rackSize(r))).total;
    if (PCACHE.size > 200000) PCACHE.clear();
    PCACHE.set(k, v);
  }
  return v;
}

export interface DenialInfo {
  bid: number;
  why: "off" | "lock" | "base" | "drain" | "deny";
  gain: number; // immediate gain to the rival we deny
}

export function denialDetail(D: Dict, s: DenialState, baseBid: number, p: DenialParams = DEFAULT_DENIAL): DenialInfo {
  const off = (why: DenialInfo["why"] = "off"): DenialInfo => ({ bid: baseBid, why, gain: 0 });
  if (s.seats > p.maxSeats || s.seats < 2) return off();
  if (s.firstPrice && !p.fpOn) return off();
  if (s.auctionsLeft <= 1 || s.balance <= 0) return off();
  const rivals = s.opponents.filter((o) => o.sim === null);
  if (rivals.length === 0) return off();

  const L = s.letter.charCodeAt(0) - 65;
  const myProj = s.myScore + cachedPartition(D, s.rack);
  const proj = rivals.map((o) => o.score + (o.knownRack ? cachedPartition(D, o.knownRack) : 0) + p.hiddenVal * (o.hidden ?? 0));
  const headsUp = rivals.length === 1;

  // Lead lock: the rival that matters is the one who would finish below us
  // (heads-up: the rival; elimination: the lowest rival).
  if (!headsUp || p.leadLockHeadsUp) {
    const lead = myProj - Math.min(...proj);
    if (lead > p.leadLockPpa * s.auctionsLeft + p.leadLockPad) return off("lock");
  }

  let b = baseBid;
  let why: DenialInfo["why"] = "base";
  const drain = s.firstPrice ? p.fpDrain : p.drain;
  if (drain > 0 && s.auctionsLeft > p.drainStop && s.balance >= p.drainMinBal) {
    const fl = drain * (FIELD_PAID[s.letter] ?? 9);
    if (fl > b) {
      b = fl;
      why = "drain";
    }
  }

  let gain = 0;
  if (p.denyA > 0 && s.auctionsLeft > p.denyStop) {
    for (let i = 0; i < rivals.length; i++) {
      const kr = rivals[i].knownRack;
      if (!kr) continue;
      const w = Uint8Array.from(kr);
      w[L]++;
      const g = cachedPartition(D, w) - cachedPartition(D, kr);
      if (g <= 0) continue;
      // Elimination: only rivals that could swap places with us on this tile.
      if (!headsUp && Math.abs(proj[i] - myProj) > g + p.closeMargin) continue;
      if (g > gain) gain = g;
    }
    if (gain > 0) {
      const d = baseBid + p.denyA * p.lambda * gain;
      if (d > b) {
        b = d;
        why = "deny";
      }
    }
  }

  if (s.firstPrice) {
    // We pay our own bid: never bid above what the richest rival can beat.
    const maxCash = Math.max(...rivals.map((o) => o.balance));
    b = Math.min(b, Math.max(baseBid, maxCash + 1));
  }
  return { bid: Math.max(0, Math.min(Math.round(b), s.balance)), why, gain };
}

export function denialBid(D: Dict, s: DenialState, baseBid: number, p: DenialParams = DEFAULT_DENIAL): number {
  return denialDetail(D, s, baseBid, p).bid;
}

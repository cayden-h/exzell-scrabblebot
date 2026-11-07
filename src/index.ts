// exzell ScrabbleBot: live client.
//
// First run: BOT_TOKEN=<token from /team> npm start
// Later runs: npm start (token saved to .token-exzell)
// Run under scripts/run.sh, which restarts the process when it exits.
//
// On any disconnect the process exits, and the supervisor starts a fresh one:
// simpler and safer than resyncing state in place.

import { DbConnection, type ErrorContext } from "./module_bindings/index.js";
import { Identity } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  BAG,
  emptyRack,
  letterIndex,
  loadDict,
  rackSize,
  rackToString,
  canSpell,
  subtractWord,
  type Rack,
} from "./engine.js";
import {
  DEFAULT_PARAMS,
  chooseWords,
  decideBid,
  type Opponent,
  type Params,
  type SimKind,
} from "./strategy.js";

const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "exzell";
const BOT_TOKEN = process.env.BOT_TOKEN;
const AUTO_REGISTER = process.env.AUTO_REGISTER !== "0";
const PLAY_LOBBY = process.env.PLAY_LOBBY !== "0";
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const TOKEN_PATH = path.join(ROOT, `.token-${BOT_NAME}`);
const LOG_DIR = path.join(ROOT, "logs");

// params.json may hold plain overrides, or {"variants": [{...}, {...}]} to
// alternate parameter sets by match id (an A/B test against the same field).
const OVERRIDES = loadParamOverrides() as Partial<Params> & { variants?: Partial<Params>[] };
const { variants: VARIANTS, ...BASE_OVERRIDES } = OVERRIDES;
const PARAMS: Params = { ...DEFAULT_PARAMS, ...BASE_OVERRIDES };
// Each match gets the variant with the fewest matches so far (counted from
// this log, so the split stays balanced across restarts), fixed at first use.
const variantByMatch = new Map<bigint, number>();
const variantCounts: number[] = (VARIANTS ?? []).map(() => 0);
try {
  const logPath = path.join(LOG_DIR, `bot-${new Date().toISOString().slice(0, 10)}.jsonl`);
  for (const line of fs.readFileSync(logPath, "utf8").split("\n")) {
    if (!line.includes('"match_end"') || !line.includes('"variant"')) continue;
    const v = JSON.parse(line).variant;
    if (typeof v === "number" && v < variantCounts.length) variantCounts[v]++;
  }
} catch {
  // no log yet
}
function variantOf(matchId: bigint): number {
  if (!VARIANTS || VARIANTS.length === 0) return 0;
  let v = variantByMatch.get(matchId);
  if (v === undefined) {
    v = variantCounts.indexOf(Math.min(...variantCounts));
    variantByMatch.set(matchId, v);
    variantCounts[v]++;
  }
  return v;
}
function paramsFor(matchId: bigint): Params {
  return VARIANTS && VARIANTS.length > 0 ? { ...PARAMS, ...VARIANTS[variantOf(matchId)] } : PARAMS;
}

function loadParamOverrides(): Partial<Params> {
  const p = path.join(ROOT, "params.json");
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return {};
  }
}

fs.mkdirSync(LOG_DIR, { recursive: true });
const logFile = fs.createWriteStream(path.join(LOG_DIR, `bot-${new Date().toISOString().slice(0, 10)}.jsonl`), {
  flags: "a",
});
function log(ev: string, data: Record<string, unknown> = {}) {
  const row = { t: new Date().toISOString(), ev, ...data };
  const line = JSON.stringify(row, (_k, v) => (typeof v === "bigint" ? Number(v) : v));
  logFile.write(line + "\n");
  console.log(`[${BOT_NAME}] ${ev} ${line.slice(line.indexOf(',"ev"') + 1, 400)}`);
}

const t0 = Date.now();
const DICT = loadDict(path.join(ROOT, "data", "wordlist.txt"));
log("dict_loaded", { words: DICT.N, ms: Date.now() - t0 });

let conn: DbConnection;
let myIdentity: Identity | null = null;
let myBotId: bigint | null = null;
let lastEventAt = Date.now();
const subscribedMatches = new Set<bigint>();
const bidSent = new Map<bigint, number>(); // auctionId -> amount
const pendingWords = new Map<bigint, string[]>(); // matchId -> words sent, not yet confirmed
const registered = new Set<bigint>();
const registering = new Set<bigint>();
let lastLobbyJoinAt = 0;
let seatTimer: ReturnType<typeof setTimeout> | null = null;

// Follow only our newest match. Seat rows for past matches arrive in bulk and
// in any order after a (re)connect, so debounce and pick the highest match id.
function syncLatestSeat() {
  if (seatTimer) clearTimeout(seatTimer);
  seatTimer = setTimeout(() => {
    seatTimer = null;
    let latest: bigint | null = null;
    for (const p of conn.db.matchParticipant.iter())
      if (p.botId === myBotId && (latest === null || p.matchId > latest)) latest = p.matchId;
    if (latest !== null && !subscribedMatches.has(latest)) {
      log("seated", { matchId: latest });
      subscribeMatch(latest);
    }
  }, 50);
}

function readToken(): string | undefined {
  try {
    return fs.readFileSync(TOKEN_PATH, "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}

// Recent reducer round trips (ms), to flush words early when the server lags.
const rtts: number[] = [];
function timed<T>(p: Promise<T>): Promise<T> {
  const t = Date.now();
  return p.finally(() => {
    rtts.push(Date.now() - t);
    if (rtts.length > 50) rtts.shift();
  });
}
function rttP90(): number {
  if (rtts.length < 5) return 300;
  const sorted = [...rtts].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * 0.9)];
}

function fail(what: string) {
  return (err: unknown) => log("reducer_error", { what, err: err instanceof Error ? err.message : String(err) });
}

// ---------- state helpers ----------

function runningMatchIds(): bigint[] {
  if (myBotId === null) return [];
  const out: bigint[] = [];
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.botId !== myBotId) continue;
    const m = conn.db.matchState.id.find(p.matchId);
    if (m?.status.tag === "Running") out.push(p.matchId);
  }
  return out;
}

// Rack from the server, minus letters of words we sent that are not confirmed yet.
function rackFor(matchId: bigint): Rack {
  const r = emptyRack();
  for (const h of conn.db.myRack.iter()) {
    if (h.matchId !== matchId) continue;
    r[letterIndex(h.letter)] += h.count;
  }
  let out = r;
  for (const w of pendingWords.get(matchId) ?? []) if (canSpell(w, out)) out = subtractWord(out, w);
  return out;
}

function me(matchId: bigint) {
  for (const p of conn.db.matchParticipant.iter())
    if (p.matchId === matchId && p.botId === myBotId) return p;
  return undefined;
}

function opponents(matchId: bigint): Opponent[] {
  const won = new Map<bigint, Uint8Array>();
  const played = new Map<bigint, Uint8Array>();
  const get = (m: Map<bigint, Uint8Array>, id: bigint) => {
    let r = m.get(id);
    if (!r) m.set(id, (r = emptyRack()));
    return r;
  };
  for (const r of conn.db.auctionResult.iter())
    if (r.matchId === matchId && r.winnerBotId !== undefined && r.winnerBotId !== null)
      get(won, r.winnerBotId)[letterIndex(r.letter)]++;
  for (const w of conn.db.wordPlay.iter())
    if (w.matchId === matchId) for (const ch of w.word) get(played, w.botId)[letterIndex(ch)]++;
  const m = conn.db.matchState.id.find(matchId);
  const startSize = m ? m.startingRackSize : 5;
  const out: Opponent[] = [];
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.matchId !== matchId || p.botId === myBotId) continue;
    const bot = conn.db.bot.id.find(p.botId);
    const sim = bot?.isSimulated && bot.strategy.tag !== "Human" ? (bot.strategy.tag as SimKind) : null;
    // Known rack from public events: tiles won minus letters played. Letters
    // played beyond what was won came from the hidden starting tiles.
    const w = won.get(p.botId) ?? emptyRack();
    const pl = played.get(p.botId) ?? emptyRack();
    const knownRack = emptyRack();
    let revealed = 0;
    for (let i = 0; i < 26; i++) {
      const d = w[i] - pl[i];
      if (d >= 0) knownRack[i] = d;
      else revealed -= d;
    }
    const o = {
      botId: p.botId,
      balance: Number(p.balance),
      score: Number(p.score),
      sim,
      knownRack,
      hidden: Math.max(0, startSize - revealed),
    };
    out.push(o);
  }
  return out;
}

// Auctions left including the open one. bag_total counts tiles still in the bag
// after the current one was drawn.
function auctionsLeft(matchId: bigint): number {
  const m = conn.db.matchState.id.find(matchId);
  if (!m || m.status.tag !== "Running") return 0;
  return m.bagTotal + (m.currentAuctionId !== undefined ? 1 : 0);
}

// ---------- actions ----------

function subscribeMatch(matchId: bigint) {
  if (subscribedMatches.has(matchId)) return;
  subscribedMatches.add(matchId);
  const id = matchId.toString();
  conn
    .subscriptionBuilder()
    .onApplied(() => {
      const tm = [...conn.db.tournamentMatch.iter()].find((x) => x.matchId === matchId);
      log("match_subscribed", { matchId, tournament: tm ? { id: tm.tournamentId, round: tm.round, phase: tm.phase.tag } : null, auctionType: conn.db.matchState.id.find(matchId)?.auctionType.tag, rttP90: rttP90() });
      for (const a of conn.db.auction.iter()) if (a.matchId === matchId && a.status.tag === "Open") onAuction(a);
      tryPlay(matchId);
    })
    .onError((_c) => log("subscription_error", { matchId }))
    .subscribe([
      `SELECT * FROM match_state WHERE id = ${id}`,
      `SELECT * FROM match_participant WHERE match_id = ${id}`,
      `SELECT * FROM auction WHERE match_id = ${id}`,
      `SELECT * FROM auction_result WHERE match_id = ${id}`,
      `SELECT * FROM word_play WHERE match_id = ${id}`,
    ]);
}

// Tiles we have not seen this match: the full bag minus every letter put up
// for auction (including the open one) minus our own starting tiles. These
// are still in the bag or in opponents' face-down starting racks. Built only
// from public events and our own rack, so it survives restarts.
function matchContext(matchId: bigint): { unseen: Uint8Array; bagLeft: number } {
  const start = emptyRack();
  for (const h of conn.db.myRack.iter()) if (h.matchId === matchId) start[letterIndex(h.letter)] += h.count;
  for (const w of conn.db.wordPlay.iter())
    if (w.matchId === matchId && w.botId === myBotId) for (const ch of w.word) start[letterIndex(ch)]++;
  for (const r of conn.db.auctionResult.iter())
    if (r.matchId === matchId && r.winnerBotId === myBotId) start[letterIndex(r.letter)]--;
  const unseen = Uint8Array.from(BAG);
  for (const a of conn.db.auction.iter()) if (a.matchId === matchId) unseen[letterIndex(a.letter)]--;
  for (let i = 0; i < 26; i++) {
    const v = (unseen[i] as number) - Math.max(0, start[i]);
    unseen[i] = Math.max(0, v);
  }
  const m = conn.db.matchState.id.find(matchId);
  return { unseen, bagLeft: m ? m.bagTotal : 0 };
}

function onAuction(a: { id: bigint; matchId: bigint; letter: string; status: { tag: string } }) {
  if (a.status.tag !== "Open" || myBotId === null) return;
  if (bidSent.has(a.id)) return;
  const p = me(a.matchId);
  if (!p) return;
  const m = conn.db.matchState.id.find(a.matchId);
  const rack = rackFor(a.matchId);
  const left = auctionsLeft(a.matchId);
  const t = Date.now();
  const ctx = matchContext(a.matchId);
  let amount = decideBid(
    DICT,
    {
      letter: a.letter,
      rack,
      balance: Number(p.balance),
      auctionsLeft: left,
      firstPrice: m?.auctionType.tag === "FirstPrice",
      ...ctx,
      opponents: opponents(a.matchId),
    },
    paramsFor(a.matchId),
  );
  // Coins are worthless once the second-to-last auction closes, so spend them
  // all on it: we may still play the tile in the last window, and a rival can't.
  if (left === 2 && Number(p.balance) > amount) amount = Number(p.balance);
  bidSent.set(a.id, amount);
  log("bid", {
    matchId: a.matchId,
    auctionId: a.id,
    letter: a.letter,
    amount,
    balance: Number(p.balance),
    left,
    rack: rackToString(rack),
    unseen: ctx.unseen.reduce((x, y) => x + y, 0),
    bagLeft: ctx.bagLeft,
    ms: Date.now() - t,
  });
  if (amount > 0) timed(conn.reducers.submitBid({ auctionId: a.id, amount: BigInt(amount) })).catch(fail(`bid ${a.id}`));
  // A bid may have been blocked by low cash; check whether to cash a word.
  tryPlay(a.matchId);
}

// Auctions left as the strategy should see them. When the server is slow, a
// flush at the normal point could land after the match ends, so report the
// flush point early once the time left drops under ~2 round trips + margin.
function effectiveLeft(matchId: bigint): number {
  const left = auctionsLeft(matchId);
  const m = conn.db.matchState.id.find(matchId);
  const flushAt = paramsFor(matchId).flushAt;
  if (!m || left <= flushAt) return left;
  const dur = Number(m.auctionDurationMs);
  const msLeft = (left - 1) * dur + dur / 2;
  if (msLeft < 2 * rttP90() + 300) {
    log("early_flush", { matchId, left, p90: rttP90() });
    return flushAt;
  }
  return left;
}

function tryPlay(matchId: bigint) {
  const p = me(matchId);
  if (!p) return;
  const m = conn.db.matchState.id.find(matchId);
  if (m?.status.tag !== "Running") return;
  const rack = rackFor(matchId);
  if (rackSize(rack) < 2) return;
  const t = Date.now();
  const words = chooseWords(
    DICT,
    { rack, balance: Number(p.balance), auctionsLeft: effectiveLeft(matchId), ...matchContext(matchId) },
    paramsFor(matchId),
  );
  if (words.length === 0) return;
  const pend = pendingWords.get(matchId) ?? [];
  pendingWords.set(matchId, pend);
  log("play", { matchId, words, rack: rackToString(rack), balance: Number(p.balance), ms: Date.now() - t });
  for (const word of words) {
    pend.push(word);
    timed(conn.reducers.submitWord({ matchId, word }))
      .catch(fail(`word ${word}`))
      .finally(() => {
        const i = pend.indexOf(word);
        if (i >= 0) pend.splice(i, 1);
      });
  }
}

function joinLobby(reason: string) {
  if (!PLAY_LOBBY || myBotId === null) return;
  if (runningMatchIds().length > 0) return;
  for (const lm of conn.db.openLobbyMembers.iter()) if (lm.botId === myBotId) return;
  if (Date.now() - lastLobbyJoinAt < 3000) return;
  lastLobbyJoinAt = Date.now();
  log("join_lobby", { reason });
  conn.reducers.joinLobby({}).catch(fail("join_lobby"));
}

function maybeRegister() {
  if (!AUTO_REGISTER || myBotId === null) return;
  for (const t of conn.db.tournament.iter()) {
    if (t.status.tag !== "Registration") continue;
    const entered = [...conn.db.tournamentEntry.iter()].some((e) => e.tournamentId === t.id && e.botId === myBotId);
    if (entered) {
      if (!registered.has(t.id)) log("tournament_already_registered", { tournamentId: t.id });
      registered.add(t.id);
      continue;
    }
    // `registering` holds in-flight calls; a failure clears it so the next tick retries.
    if (registering.has(t.id)) continue;
    registering.add(t.id);
    log("tournament_register", { tournamentId: t.id, auctionType: t.auctionType.tag });
    conn.reducers
      .registerForTournament({ tournamentId: t.id })
      .then(() => log("tournament_register_ok", { tournamentId: t.id }))
      .catch(fail(`register ${t.id}`))
      .finally(() => registering.delete(t.id));
  }
}

// ---------- wiring ----------

function onConnect(c: DbConnection, identity: Identity, token: string) {
  conn = c;
  myIdentity = identity;
  fs.writeFileSync(TOKEN_PATH, token, { mode: 0o600 });
  log("connected", { identity: identity.toHexString().slice(0, 12) });

  c.subscriptionBuilder()
    .onApplied(() => {
      const cred = c.db.botCredential.identity.find(identity);
      if (!cred) {
        log("no_credential", { hint: "get a fresh token from /team" });
        process.exit(2);
      }
      myBotId = cred.botId;
      log("acting_as", { botId: myBotId, name: c.db.bot.id.find(myBotId)?.name, params: PARAMS, variants: VARIANTS ?? null });
      c.subscriptionBuilder()
        .onApplied(() => {
          // After a restart we may be mid-match: watch our most recent match.
          syncLatestSeat();
          maybeRegister();
          setTimeout(() => joinLobby("startup"), 1500);
        })
        .subscribe([`SELECT * FROM match_participant WHERE bot_id = ${myBotId}`]);
    })
    .onError(() => log("subscription_error", { what: "global" }))
    .subscribe([
      "SELECT * FROM bot_credential",
      "SELECT * FROM bot",
      "SELECT * FROM tournament",
      "SELECT * FROM tournament_entry",
      "SELECT * FROM tournament_match",
      "SELECT * FROM open_lobbies",
      "SELECT * FROM open_lobby_members",
      "SELECT * FROM my_rack",
    ]);

  c.db.auction.onInsert((_ctx, a) => {
    lastEventAt = Date.now();
    onAuction(a);
  });
  c.db.matchParticipant.onInsert((_ctx, p) => {
    lastEventAt = Date.now();
    if (p.botId === myBotId) syncLatestSeat();
  });
  c.db.matchParticipant.onUpdate(() => {
    lastEventAt = Date.now();
  });
  c.db.myRack.onInsert((_ctx, h) => tryPlay(h.matchId));
  c.db.myRack.onUpdate((_ctx, _o, h) => tryPlay(h.matchId));
  c.db.auctionResult.onInsert((_ctx, r) => {
    lastEventAt = Date.now();
    if (!subscribedMatches.has(r.matchId)) return;
    const mine = r.winnerBotId === myBotId;
    log("result", { matchId: r.matchId, auctionId: r.auctionId, letter: r.letter, winner: r.winnerBotId, paid: r.paid, mine, ourBid: bidSent.get(r.auctionId) });
    bidSent.delete(r.auctionId);
  });
  c.db.matchState.onUpdate((_ctx, old, neu) => {
    lastEventAt = Date.now();
    if (old.status.tag !== "Ended" && neu.status.tag === "Ended" && subscribedMatches.has(neu.id)) {
      const parts = [...c.db.matchParticipant.iter()]
        .filter((p) => p.matchId === neu.id)
        .sort((a, b) => Number(b.score - a.score));
      const place = parts.findIndex((p) => p.botId === myBotId) + 1;
      const ours = me(neu.id);
      log("match_end", {
        matchId: neu.id,
        variant: variantOf(neu.id),
        place,
        score: ours?.score,
        balance: ours?.balance,
        leftover: rackToString(rackFor(neu.id)),
        scores: parts.map((p) => `${c.db.bot.id.find(p.botId)?.name ?? p.botId}:${p.score}`),
      });
      pendingWords.delete(neu.id);
      setTimeout(() => joinLobby("match_end"), 500);
    }
  });
  c.db.openLobbies.onInsert(() => {
    lastEventAt = Date.now();
  });
  c.db.openLobbies.onDelete(() => {
    lastEventAt = Date.now();
  });
  c.db.tournament.onInsert(() => maybeRegister());
  c.db.tournament.onUpdate(() => maybeRegister());
}

function main() {
  const token = BOT_TOKEN ?? readToken();
  if (!token) {
    console.error(`No token. Run once with BOT_TOKEN=<token from /team>.`);
    process.exit(2);
  }
  log("connecting", { host: HOST, db: DB_NAME });
  DbConnection.builder()
    .withUri(HOST)
    .withDatabaseName(DB_NAME)
    .withToken(token)
    .onConnect(onConnect)
    .onConnectError((_ctx: ErrorContext, err: Error) => {
      log("connect_error", { err: err.message });
      process.exit(1);
    })
    .onDisconnect(() => {
      log("disconnected");
      process.exit(1);
    })
    .build();

  // Periodic upkeep: rejoin the lobby when idle, register for tournaments,
  // and exit if the server has gone quiet (the supervisor restarts us).
  setInterval(() => {
    if (myBotId === null) return;
    maybeRegister();
    joinLobby("tick");
    for (const id of runningMatchIds()) tryPlay(id);
    if (Date.now() - lastEventAt > 5 * 60_000) {
      log("watchdog_exit", { quietMs: Date.now() - lastEventAt });
      process.exit(1);
    }
  }, 5000);
}

main();

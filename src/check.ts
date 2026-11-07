// Offline sanity check of the engine and strategy.
import * as path from "node:path";
import { loadDict, rackFromString, bestPartition, rackSize } from "./engine.js";
import { decideBid, chooseWords, DEFAULT_PARAMS } from "./strategy.js";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
let t = Date.now();
const D = loadDict(path.join(ROOT, "data", "wordlist.txt"));
console.log("dict", D.N, "classes", Date.now() - t, "ms");
for (const s of ["AEEILNORSTTUXZQ", "RETAINS", "QUETZALTOXINES", "AEINSTRLDGBCMPOU", "AEEIOOUUSRTNLDGB"]) {
  const r = rackFromString(s);
  t = Date.now();
  const p = bestPartition(D, r, rackSize(r) <= 14 ? Infinity : 24);
  console.log(s, "->", p.total, p.words.join(" "), Date.now() - t, "ms");
}
const opp = [1, 2, 3, 4, 5].map((i) => ({ botId: BigInt(i), balance: 100, score: 0, sim: null }));
for (const [letter, rack, left] of [["E", "RSTNL", 60], ["Z", "AEIRST", 50], ["Q", "AEIRST", 50], ["S", "AEINRT", 5], ["S", "AEINRT", 1]] as const) {
  const b = decideBid(D, { letter, rack: rackFromString(rack), balance: 80, auctionsLeft: left, firstPrice: false, opponents: opp });
  console.log(`bid ${letter} rack ${rack} left ${left} ->`, b);
}
console.log("play low cash:", chooseWords(D, { rack: rackFromString("AEINRSTXQ"), balance: 10, auctionsLeft: 30 }));
console.log("play flush:", chooseWords(D, { rack: rackFromString("AEINRSTXQOU"), balance: 50, auctionsLeft: 2 }));

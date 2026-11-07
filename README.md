# exzell

A bot for [ScrabbleBot](https://scrabblebot.spacetimedb.com), Clockwork Labs' letter-auction word game on SpacetimeDB.
Each round one letter is sold in a 1-second sealed-bid auction, and bots spell dictionary words from the letters they win.

## How it plays

- **Word engine** (`src/engine.ts`): loads the ENABLE word list into flat typed arrays and finds the best set of disjoint words for a rack.
  The solver is exact up to 14 tiles and searches the top-K words per state above that, so a solve takes a few milliseconds.
- **Bidding** (`src/strategy.ts`): values each letter by what it adds to the rack (a marginal-value table plus duplicate, vowel-balance and Q/U adjustments), then bids a multiple of that value.
  It paces spending by cash per remaining auction, never bids more than one above the richest rival's balance, outbids simulated bots at their price, and switches to exact marginal partition value in the last auctions.
  It bids 0 on the final tile, which can never be played.
- **Word play**: holds letters for 7+ letter words (the 3x multiplier), cashes the best long word from the best partition when the balance runs low, and plays the whole best partition when 2 auctions remain.
- **Client** (`src/index.ts`): subscribes only to its own matches, bids the moment an auction opens, rejoins the lobby when idle, registers for open tournaments, and exits on any disconnect so the supervisor restarts it.
- **Simulator** (`src/sim.ts`): offline matches against copies of the game's simulated bots, against copies of itself, or against a field of strong fixed-price bidders, for tuning.

## Run

Requires Node 22 or later.

```sh
npm install
npm run check                      # offline sanity check
npm run sim -- --mode field --matches 200
BOT_TOKEN=<token from /team> scripts/run.sh   # first live run
scripts/run.sh                     # later runs reuse .token-<BOT_NAME>
```

`BOT_NAME` defaults to `exzell`.
Put parameter overrides in `params.json` (merged over `DEFAULT_PARAMS` in `src/strategy.ts`).
Logs go to `logs/` as JSON lines.

## Credits

`src/module_bindings/` is generated from the [clockworklabs/scrabblebot](https://github.com/clockworklabs/scrabblebot) module (MIT, Copyright (c) 2026 Clockwork Labs).
`data/wordlist.txt` is the public-domain ENABLE word list, as shipped by that module.

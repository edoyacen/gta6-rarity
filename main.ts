// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecast, for every trophy of Grand Theft Auto VI on PS5, five quantiles of its Exophase earn rate at the case's
// target. The score is the mean pinball loss on the logit scale, so the median must be accurate and the band honest.
//
// The page is not continuous: Exophase prints the rates as of its last refresh, and between refreshes every read is
// byte-for-byte the same. So the only informative moments are the refresh "states" -- the reads where the players
// tracked count (the denominator) and the rates change. This entry works in those states:
//
//   1. Collapse the history to refresh states (drop repeated reads).
//   2. Find the last large population jump (a refresh that added many players) and the current regime after it.
//   3. For each trophy model logit(rate) as a smooth growth curve over log-time, fit on the current regime's states,
//      and extrapolate to the target with a damping factor so a week out never explodes.
//   4. Learn, from every observed refresh, how far a refresh moves each trophy in logit space; fold that into the band.
//   5. A flat history (launch night, no refresh yet) cannot be extrapolated: it is a single state, and the next
//      refresh is expected to add far more (less-progressed) players and dilute every rate. Predict that dilution.
//   6. Anchor the median to a tier ceiling and a rarity floor; make the quantiles non-decreasing and deterministic.
//
// No clock, no randomness: two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Hours-to-ms and the release anchor the curve is measured from. */
const HOUR_MS = 3_600_000;
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Minutes per day, for the horizon. */
const DAY_MIN = 1440;

/** Tier saturation ceilings in percent: a launch-week rate will not cross these. */
const TIER_CEILING: Record<string, number> = { platinum: 99.9, gold: 82, silver: 93, bronze: 99 };
/** Tier growth weights: common early bronze trophies grow faster than late gold ones. */
const TIER_GROWTH: Record<string, number> = { platinum: 0, gold: 0.5, silver: 0.85, bronze: 1 };

interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Trophy {
  id: number;
  tier: string;
  points: number;
  secret: boolean;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}
/** One refresh state: hours since release, players tracked, the rate vector. */
interface State {
  t: number;
  P: number;
  rates: Array<number | null>;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** The scorer's logit: percent clamped to [0.01, 99.99] first, as the grader does. */
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99);
  return Math.log(p / (100 - p));
};
const invLogit = (x: number): number => 100 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
};

/** Least-squares line y = a + b x, with the residual standard deviation. */
function ols(xs: number[], ys: number[]): { slope: number; intercept: number; residSd: number } {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return { slope: 0, intercept: n === 1 ? (ys[0] as number) : 0, residSd: 0 };
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < n; k += 1) {
    const dx = (xs[k] as number) - mx;
    sxx += dx * dx;
    sxy += dx * ((ys[k] as number) - my);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let k = 0; k < n; k += 1) {
    const e = (ys[k] as number) - (intercept + slope * (xs[k] as number));
    ss += e * e;
  }
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { slope, intercept, residSd };
}

/** Collapse the history to refresh states: keep a read only when the page actually changed. */
function buildStates(history: Read[]): State[] {
  const states: State[] = [];
  let prev = '';
  for (const h of history) {
    const sig = `${h.players_tracked}|${JSON.stringify(h.rates)}`;
    if (sig === prev) continue;
    states.push({ t: (Date.parse(h.at) - RELEASE_MS) / HOUR_MS, P: h.players_tracked, rates: h.rates });
    prev = sig;
  }
  return states;
}

/**
 * Index of the first refresh that multiplied the tracked population (the launch-night dilution). Everything before it
 * is a tiny, self-selected early-adopter sample whose rates are not comparable to the week's, so the growth curve is
 * fit only on the states from this jump onward. Returns 0 when no such jump was seen.
 */
function launchDilution(states: State[]): number {
  for (let k = 1; k < states.length; k += 1) {
    const prev = states[k - 1] as State;
    const cur = states[k] as State;
    if (prev.P > 0 && cur.P > prev.P * 1.5) return k;
  }
  return 0;
}

/** The mean absolute logit move a refresh causes, pooled over every observed refresh (a volatility estimate). */
function refreshVolatility(states: State[]): number {
  let sum = 0;
  let n = 0;
  for (let k = 1; k < states.length; k += 1) {
    const prev = states[k - 1] as State;
    const cur = states[k] as State;
    if (!(prev.P > 0 && cur.P > prev.P * 1.2)) continue;
    const len = Math.min(prev.rates.length, cur.rates.length);
    for (let i = 0; i < len; i += 1) {
      const a = prev.rates[i];
      const b = cur.rates[i];
      if (a === null || b === null || a <= 0 || b <= 0) continue;
      sum += Math.abs(logit(b) - logit(a));
      n += 1;
    }
  }
  return n > 0 ? sum / n : 0;
}

/** The rate series of trophy i over the given states, keeping only non-null reads. */
function seriesOf(states: State[], i: number): Array<{ t: number; P: number; r: number }> {
  const out: Array<{ t: number; P: number; r: number }> = [];
  for (const s of states) {
    const r = s.rates[i];
    if (r === null || !Number.isFinite(r)) continue;
    out.push({ t: s.t, P: s.P, r });
  }
  return out;
}

/** Five quantiles (percent) of trophy i's rate at the target, as [q05,q25,q50,q75,q95]. */
function forecastTrophy(
  i: number,
  trophy: Trophy,
  states: State[],
  regime: State[],
  tTarget: number,
  dDays: number,
  flatRegime: boolean,
  vol: number,
): number[] {
  const full = seriesOf(states, i);
  const regimeSeries = seriesOf(regime, i);
  const src = regimeSeries.length > 0 ? regimeSeries : full;
  const last = src[src.length - 1];
  if (last === undefined) {
    // No read ever carried this trophy: a rare, unlisted row. Sit at the floor with a wide-ish band.
    return [0.01, 0.02, 0.05, 0.12, 0.4];
  }
  const rLast = last.r;

  // A platinum-style trophy already at 100% stays at 100%: the scorer clamps 100 to 99.99, so 100 is exact.
  if (rLast >= 99.5) {
    return [99.9, 99.99, 100, 100, 100];
  }

  let lLast = logit(rLast);
  let lPred: number;
  let spread: number;

  if (flatRegime) {
    // Launch night: one unchanged page, and the next refresh is expected to add far more players. The newly tracked
    // players are less progressed, so every rate but 100% dilutes. The launch refresh multiplied players ~6.06x
    // (638 -> 3868) and earned trophies ~4.53x (943 -> 4276), so a rate scales by ~0.748 on average. That is the
    // proportional (count-consistent) estimate: hold each trophy's count and add the new players' share. Common
    // trophies dilute a touch less (nearly everyone earns the early ones); rare ones a touch more.
    const commonness = clamp(Math.min(rLast, 100) / 100, 0, 1);
    const dilution = Math.log(4.53 / 6.06) * (1 - 0.35 * commonness);
    lPred = lLast + dilution;
    // A refresh is a discrete, poorly-timed shock. Keep a real band, but not so wide that an accurate median is
    // buried: the proportional estimate is usually within a few tenths of a logit of the truth.
    spread = 0.62 + 0.18 * (1 - commonness);
    spread += 0.2 * Math.min(vol, 2);
  } else {
    // Rich history: fit logit(rate) against log-time over every state since the launch dilution and extrapolate,
    // damped. A refresh lowers the rate at its instant but the multi-day drift between refreshes is what we want.
    const pts = src.length > 12 ? src.slice(-12) : src;
    let trendDelta = 0;
    let residSd = 0;
    let usedFit = false;
    if (pts.length >= 3) {
      const xs = pts.map((p) => Math.log1p(Math.max(0, p.t)));
      const ys = pts.map((p) => logit(p.r));
      const fit = ols(xs, ys);
      // Clamp the fitted slope so one noisy refresh cannot define a week: at most ~1 logit per ln-hour.
      const slope = clamp(fit.slope, -1, 1);
      const uLast = Math.log1p(Math.max(0, last.t));
      const uTarget = Math.log1p(Math.max(0, tTarget));
      // Damp: a 6.5-day extrapolation is worth less than the fit's full slope; H0 ~ 6 days halves it.
      const damp = 1 / (1 + dDays / 6);
      trendDelta = clamp(slope * (uTarget - uLast) * damp, -1.2, 1.8);
      residSd = fit.residSd;
      usedFit = true;
    }
    const growthWeight = TIER_GROWTH[trophy.tier] ?? 0.8;
    const priorDelta = 0.05 * dDays * growthWeight;
    lPred = usedFit ? lLast + trendDelta + 0.25 * priorDelta : lLast + priorDelta;
    // Declines are usually the transient dilution of a refresh; soften a predicted fall rather than extrapolate it.
    if (lPred < lLast) lPred = lLast + (lPred - lLast) * 0.6;

    const ceiling = logit(TIER_CEILING[trophy.tier] ?? 99);
    lPred = Math.min(lPred, ceiling);
    lPred = Math.max(lPred, logit(0.02));

    spread = 0.3 * Math.sqrt(Math.max(dDays, 1 / 24)) + 0.5 * residSd + 0.25 * Math.min(vol, 1.5);
    spread = clamp(spread, 0.2, 2.0);
  }

  // Rare trophies cannot fall below a plausible floor: keep the median from sinking under ~0.01%.
  if (rLast < 0.05) {
    lPred = Math.max(lPred, logit(Math.max(rLast, 0.015)));
  }

  const qLogit = Z.map((z) => lPred + z * spread);
  const q = qLogit.map((x) => round4(clamp(invLogit(x), 0.01, 100)));
  // Enforce the scorer's non-decreasing requirement after independent rounding.
  for (let k = 1; k < q.length; k += 1) q[k] = Math.max(q[k] as number, q[k - 1] as number);
  return q;
}

/** Forecast every trophy of one case. */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const states = buildStates(c.history);
  if (states.length === 0) throw new Error(`${c.id} has no history`);
  const jump = launchDilution(states);
  const regime = states.slice(jump);
  const populations = new Set(states.map((s) => s.P));
  const flatRegime = populations.size === 1;
  const vol = refreshVolatility(states);
  const tTarget = (Date.parse(c.target) - RELEASE_MS) / HOUR_MS;
  const dDays = c.horizon_minutes / DAY_MIN;
  return c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(i, t, states, regime, tTarget, dDays, flatRegime, vol),
  }));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}

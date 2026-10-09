export interface TileRow {
  start: number; // first tile index in the row
  end: number; // one past the last
  height: number; // px; each tile is aspect * height wide
}

// how much a row that hits maxHeight (and so cannot fill the width) costs per unit of empty width
const UNFILLED_PENALTY = 2;

/** A tile's width / height. Missing sizes count as square; absurd ones are clamped so sums stay
 * finite. The layout and the drawing must both use this, or rows come out wider than planned. */
export function tileAspect(width: number | null | undefined, height: number | null | undefined): number {
  const aspect = width && height ? width / height : 1;
  return aspect > 0 ? Math.min(Math.max(aspect, 0.01), 100) : 1;
}

/**
 * Justified rows: tiles keep their aspect ratio (width / height) and order, and each row is
 * scaled to exactly `width`, so whole images fit with no cropping and no letterboxing.
 *
 * The row breaks are chosen together, not greedily: a search over the splits (rows are not
 * extended once they drop below a quarter of the target height) keeps each row as close to
 * `targetHeight` as possible (relative error, weighted by the row's tile count). A row that would
 * need to be taller than `maxHeight` (or the target, if that is larger) is capped and left short,
 * and pays for the empty width, so the search avoids it when it can. The last row is treated like
 * the others, so it cannot collect a tiny or lonely leftover.
 */
export function justifiedRows(
  aspects: number[],
  width: number,
  gap: number,
  targetHeight: number,
  maxHeight: number,
): TileRow[] {
  const n = aspects.length;
  if (n === 0 || !(width > 0) || !(targetHeight > 0)) return [];
  const cap = Math.max(maxHeight, targetHeight);
  const prefix = [0];
  for (const a of aspects) prefix.push(prefix[prefix.length - 1] + tileAspect(a, 1));
  const shortest = targetHeight / 4; // rows of more tiles than this would be pointless

  const fittedHeight = (i: number, j: number) => (width - gap * (j - i - 1)) / (prefix[j] - prefix[i]);
  const rowCost = (i: number, j: number, fitted: number) => {
    if (!(fitted > 0)) return Infinity; // the gaps alone are wider than the panel
    const shown = Math.min(fitted, cap);
    let cost = (j - i) * Math.log(shown / targetHeight) ** 2;
    if (fitted > cap) {
      const gaps = gap * (j - i - 1);
      const shownWidth = (width - gaps) * (cap / fitted) + gaps;
      cost += UNFILLED_PENALTY * (1 - shownWidth / width);
    }
    return cost;
  };

  // best[j]: the cheapest layout of tiles 0..j-1; from[j]: where its last row starts
  const best = new Array<number>(n + 1).fill(Infinity);
  const from = new Array<number>(n + 1).fill(0);
  best[0] = 0;
  for (let j = 1; j <= n; j++) {
    for (let i = j - 1; i >= 0; i--) {
      const fitted = fittedHeight(i, j);
      const cost = best[i] + rowCost(i, j, fitted);
      if (cost < best[j]) {
        best[j] = cost;
        from[j] = i;
      }
      if (fitted < shortest) break; // more tiles only make the row shorter still
    }
  }

  const rows: TileRow[] = [];
  for (let j = n; j > 0; j = from[j]) {
    const i = from[j];
    rows.push({ start: i, end: j, height: Math.min(fittedHeight(i, j), cap) });
  }
  return rows.reverse();
}

/** A tile height that suits how many images share the panel. */
export function targetTileHeight(count: number): number {
  if (count <= 2) return 320;
  if (count <= 4) return 280;
  if (count <= 8) return 240;
  if (count <= 16) return 200;
  if (count <= 32) return 160;
  return 120;
}

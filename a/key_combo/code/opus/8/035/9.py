import random

import numpy as np


BACKGROUND = 0
BASELINE = 5
DIVIDER = 6
BAR_COLORS = (1, 2, 3, 4, 8)


def _parse_bars(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    H, W = g.shape
    base_rows = [r for r in range(H) if np.all(g[r] == BASELINE)]
    divider_rows = [r for r in range(H) if np.all(g[r] == DIVIDER)]
    if len(base_rows) != 1 or len(divider_rows) != 1:
        raise ValueError("expected one full-width baseline and one full-width divider")

    base_r = int(base_rows[0])
    div_r = int(divider_rows[0])
    if not (0 <= div_r < base_r < H):
        raise ValueError("divider must lie above the baseline")

    bars: list[dict[str, int]] = []
    for c in range(W):
        column = g[:, c]
        mask = (column != BACKGROUND) & (column != BASELINE) & (column != DIVIDER)
        if not np.any(mask):
            continue

        rows = np.flatnonzero(mask)
        colors = {int(v) for v in column[mask]}
        if len(colors) != 1:
            raise ValueError("each bar must be monochrome")

        color = int(next(iter(colors)))
        top_r = int(rows.min())
        height = int(base_r - top_r)
        if height < 1:
            raise ValueError("bar height must be positive")

        for r in range(top_r, base_r):
            if r == div_r:
                if int(g[r, c]) != DIVIDER:
                    raise ValueError("divider row must remain visible across tall bars")
            elif int(g[r, c]) != color:
                raise ValueError("bar silhouette must be vertical and gap-free except at the divider")

        bars.append({"col": int(c), "color": color, "height": height})

    if not (4 <= len(bars) <= 5):
        raise ValueError("expected 4 or 5 bars")

    heights = [bar["height"] for bar in bars]
    colors = [bar["color"] for bar in bars]
    if len(set(heights)) != len(heights):
        raise ValueError("bar heights must be pairwise distinct")
    if len(set(colors)) != len(colors):
        raise ValueError("bar colors must be pairwise distinct")

    return base_r, div_r, sorted(bars, key=lambda bar: bar["col"])


def _render_from_bars(
    shape: tuple[int, int],
    base_r: int,
    div_r: int,
    slots: list[dict[str, int]],
    *,
    sort_by_height: bool,
    split_colors: bool,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[int(base_r), :] = BASELINE
    out[int(div_r), :] = DIVIDER

    ordered_slots = sorted(slots, key=lambda bar: bar["col"])
    ordered_bars = (
        sorted(slots, key=lambda bar: bar["height"])
        if sort_by_height
        else list(ordered_slots)
    )

    for target_slot, source_bar in zip(ordered_slots, ordered_bars):
        col = int(target_slot["col"])
        slot_color = int(target_slot["color"])
        source_color = int(source_bar["color"])
        top_r = int(base_r - source_bar["height"])

        for r in range(top_r, base_r):
            if r == div_r:
                continue
            if split_colors and r < div_r:
                out[r, col] = source_color
            else:
                out[r, col] = slot_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the bar heights into left-to-right slots, then split each reassigned bar into slot-colored lower and source-colored upper segments.

    Input structure:
    - The background is black.
    - The last full-width grey row is a baseline, and a full-width magenta row above it is a divider.
    - Four or five 1-pixel-wide vertical bars stand on the baseline at distinct columns, with distinct heights and distinct colors.

    Rule:
    - Read each bar's original color, height, and slot column.
    - Sort the bars by height from shortest to tallest, and sort the slots from left to right.
    - Put the i-th shortest bar silhouette into the i-th slot.
    - In each reassigned bar, pixels above the divider keep the moving bar's original color, while pixels below the divider take the target slot's original color.
    - Bars that do not reach above the divider stay entirely in the target slot color.
    - Keep the baseline grey and the divider magenta.

    Key insight:
    - Each output bar combines two identities at once: its height and upper color come from the sorted source bar, but its lower color comes from the destination slot that receives it.
    """
    g = np.asarray(grid, dtype=int)
    base_r, div_r, bars = _parse_bars(g)
    return _render_from_bars(
        g.shape,
        base_r,
        div_r,
        bars,
        sort_by_height=True,
        split_colors=True,
    )


def _choose_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _mixed_tall_movers(perm: list[int], tall_ranks: set[int]) -> int:
    moved = 0
    for rank in tall_ranks:
        original_slot = perm.index(rank)
        if original_slot != rank:
            moved += 1
    return moved


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    base_r, div_r, bars = _parse_bars(inp)
    threshold = int(base_r - div_r)

    heights = [bar["height"] for bar in bars]
    if heights == sorted(heights):
        return False

    tall = [bar for bar in bars if bar["height"] > threshold]
    short = [bar for bar in bars if bar["height"] < threshold]
    if len(tall) < 2 or len(short) < 1:
        return False

    expected = _render_from_bars(inp.shape, base_r, div_r, bars, sort_by_height=True, split_colors=True)
    if not np.array_equal(out, expected):
        return False

    old_rule = _render_from_bars(inp.shape, base_r, div_r, bars, sort_by_height=True, split_colors=False)
    if np.array_equal(out, old_rule):
        return False

    no_sort = _render_from_bars(inp.shape, base_r, div_r, bars, sort_by_height=False, split_colors=True)
    if np.array_equal(out, no_sort):
        return False

    mixed_columns = 0
    for bar in sorted(bars, key=lambda item: item["col"]):
        col = int(bar["col"])
        visible_colors = {
            int(out[r, col])
            for r in range(out.shape[0])
            if int(out[r, col]) not in (BACKGROUND, BASELINE, DIVIDER)
        }
        if len(visible_colors) == 2:
            mixed_columns += 1
    if mixed_columns < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([4, 5])
    H = random.randint(15, 20)
    W = random.randint(max(11, count * 2 + 1), 16)
    base_r = H - 1
    max_height = min(12, base_r)

    cols = _choose_columns(W, count)
    if cols is None:
        return None

    feasible_thresholds = []
    for threshold in range(5, min(8, max_height - 2) + 1):
        below_pool = list(range(3, threshold))
        above_pool = list(range(threshold + 1, max_height + 1))
        tall_min = max(2, count - len(below_pool))
        tall_max = min(count - 1, len(above_pool))
        if tall_min <= tall_max:
            feasible_thresholds.append((threshold, tall_min, tall_max))
    if not feasible_thresholds:
        return None

    threshold, tall_min, tall_max = random.choice(feasible_thresholds)
    tall_count = random.randint(tall_min, tall_max)
    low_count = count - tall_count
    div_r = base_r - threshold

    low_heights = random.sample(list(range(3, threshold)), low_count)
    tall_heights = random.sample(list(range(threshold + 1, max_height + 1)), tall_count)
    sorted_heights = sorted(low_heights + tall_heights)
    tall_ranks = set(range(count - tall_count, count))

    perm = list(range(count))
    for _ in range(200):
        random.shuffle(perm)
        if perm == list(range(count)):
            continue
        if _mixed_tall_movers(perm, tall_ranks) >= 2:
            break
    else:
        return None

    slot_colors = random.sample(list(BAR_COLORS), count)
    bars: list[dict[str, int]] = []
    for slot_idx, col in enumerate(cols):
        rank = perm[slot_idx]
        bars.append(
            {
                "col": int(col),
                "color": int(slot_colors[slot_idx]),
                "height": int(sorted_heights[rank]),
            }
        )

    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[base_r, :] = BASELINE
    grid[div_r, :] = DIVIDER
    for bar in bars:
        top_r = base_r - bar["height"]
        for r in range(top_r, base_r):
            if r == div_r:
                continue
            grid[r, bar["col"]] = bar["color"]

    try:
        parsed_base, parsed_div, parsed_bars = _parse_bars(grid)
    except Exception:
        return None

    if parsed_base != base_r or parsed_div != div_r:
        return None
    if parsed_bars != sorted(bars, key=lambda item: item["col"]):
        return None

    out = main(grid)
    if not _semantic_checks(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

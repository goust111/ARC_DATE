from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7
BG = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _extract_bars(grid: np.ndarray) -> tuple[list[dict], int]:
    bars = []
    baseline = None
    for comp in _find_components(grid, {BLUE, RED}):
        r0, c0, r1, c1 = comp["bbox"]
        assert c0 == c1, "bars must be one-cell-wide vertical segments"
        height = int(r1 - r0 + 1)
        assert comp["area"] == height, "bars must be solid vertical segments"
        col = int(c0)
        for rr in range(r0, r1 + 1):
            assert int(grid[rr, col]) == int(comp["color"])
        if baseline is None:
            baseline = int(r1)
        else:
            assert int(r1) == baseline, "all bars must share one bottom baseline"
        bars.append(
            {
                "color": int(comp["color"]),
                "col": int(col),
                "top": int(r0),
                "bottom": int(r1),
                "height": int(height),
                "coords": comp["coords"],
            }
        )
    assert bars, "expected at least one bar"
    bars.sort(key=lambda item: item["col"])
    assert len({bar["col"] for bar in bars}) == len(bars), "bar columns must be unique"
    return bars, int(baseline)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for comp in _find_components(grid, {ORANGE}):
        r0, c0, r1, c1 = comp["bbox"]
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        assert comp["area"] == area, "orange objects must be solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE)
        rects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "width": int(c1 - c0 + 1),
                "height": int(r1 - r0 + 1),
            }
        )
    assert rects, "expected orange rectangles"
    rects.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return rects


def _build_chains(bars: list[dict]) -> list[list[dict]]:
    by_col = {bar["col"]: bar for bar in bars}
    chains: list[list[dict]] = []
    for bar in bars:
        if bar["color"] != BLUE:
            continue
        chain = [bar]
        current = bar
        seen_cols = {bar["col"]}
        while True:
            next_col = int(current["col"] + current["height"])
            nxt = by_col.get(next_col)
            if nxt is None or int(nxt["color"]) != RED:
                break
            assert nxt["col"] not in seen_cols, "chain cannot cycle"
            chain.append(nxt)
            seen_cols.add(nxt["col"])
            current = nxt
        chains.append(chain)
    assert chains, "expected at least one blue chain starter"

    used_cols = [bar["col"] for chain in chains for bar in chain]
    assert len(set(used_cols)) == len(used_cols), "a red bar cannot belong to multiple blue chains"
    return chains


def _rect_index(rects: list[dict]) -> dict[tuple[int, int], int]:
    out = {}
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[(r, c)] = idx
    return out


def _segment_touches_rects(row: int, c0: int, length: int, rect_by_cell: dict[tuple[int, int], int]) -> set[int]:
    touched: set[int] = set()
    for c in range(c0, c0 + length):
        for nr, nc in _neighbors4(row, c):
            idx = rect_by_cell.get((nr, nc))
            if idx is not None:
                touched.add(int(idx))
    return touched


def _apply_rule(grid: np.ndarray, *, rotate_all_bars: bool = False) -> tuple[np.ndarray, list[int], list[list[dict]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "background must be black"
    assert set(map(int, np.unique(g))) <= {BG, BLUE, RED, ORANGE}

    bars, baseline = _extract_bars(g)
    rects = _extract_rectangles(g)
    chains = _build_chains(bars)

    rotating_cols = {bar["col"] for chain in chains for bar in chain}
    if rotate_all_bars:
        rotating = bars
    else:
        rotating = [bar for bar in bars if bar["col"] in rotating_cols]

    out = g.copy()
    for bar in rotating:
        rows = bar["coords"][:, 0]
        cols = bar["coords"][:, 1]
        out[rows, cols] = bg

    for bar in rotating:
        c0 = int(bar["col"])
        length = int(bar["height"])
        assert c0 + length <= g.shape[1], "rotated bar must stay inside the grid"
        out[baseline, c0 : c0 + length] = int(bar["color"])

    rect_by_cell = _rect_index(rects)
    stripe_sums = [0 for _ in rects]
    for bar in rotating:
        touched = _segment_touches_rects(int(bar["bottom"]), int(bar["col"]), int(bar["height"]), rect_by_cell)
        for idx in touched:
            stripe_sums[idx] += int(bar["height"])

    for idx, total in enumerate(stripe_sums):
        if total <= 0:
            continue
        r0, c0, r1, c1 = rects[idx]["bbox"]
        fill = min(int(total), int(c1 - c0 + 1))
        out[r0 : r1 + 1, c0 : c0 + fill] = YELLOW

    return out, stripe_sums, chains, rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each blue-started vertical-bar chain, then use the touched chain-tail lengths to create left-edge stripes inside orange rectangles.

    Input structure:
    - The background is black.
    - Blue and red one-cell-wide vertical bars share one bottom baseline; blue bars start chains and red bars may continue them.
    - Solid orange rectangles sit to the right, initially separated from the vertical bars.

    Rule:
    - From each blue bar, repeatedly follow the red bar whose column equals the current bar's column plus its height.
    - Clear every bar in each reached chain, then redraw that bar horizontally to the right from its original bottom cell with the same color and the same length as its former height.
    - Red bars not reached from any blue starter stay vertical and unchanged.
    - For every orange rectangle that 4-touches one or more rotated horizontal bars, sum the touching bar lengths and fill that many whole columns from the rectangle's left edge with yellow, capped by the rectangle width.

    Key insight:
    - Rotating a chain bar both creates the rectangle contact and converts the bar's vertical height into the stripe-count parameter, so the chain must be traced before the fill amount is knowable.
    """
    out, _stripe_sums, _chains, _rects = _apply_rule(grid, rotate_all_bars=False)
    return out


def _bar_columns(chain: list[dict]) -> list[int]:
    return [int(bar["col"]) for bar in chain]


def _make_chain(start_col: int, heights: list[int]) -> list[dict]:
    cols = []
    col = int(start_col)
    for height in heights:
        cols.append(col)
        col += int(height)
    chain = []
    for idx, (bar_col, height) in enumerate(zip(cols, heights)):
        chain.append({"color": BLUE if idx == 0 else RED, "col": int(bar_col), "height": int(height)})
    return chain


def _paint_bar(grid: np.ndarray, baseline: int, bar: dict) -> None:
    r0 = int(baseline - bar["height"] + 1)
    assert r0 >= 0
    grid[r0 : baseline + 1, int(bar["col"])] = int(bar["color"])


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE


def _choose_touch_width(last_height: int) -> int:
    if random.random() < 0.5:
        return random.randint(3, min(4, last_height - 1))
    return random.randint(max(3, last_height), 5)


def _sample_candidate() -> np.ndarray | None:
    h = random.randint(18, 24)
    baseline = h - 1

    main_len = random.randint(2, 4)
    last_height = random.randint(4, 5)
    first_height = random.randint(2, 5)
    while first_height == last_height:
        first_height = random.randint(2, 5)
    main_heights = [first_height]
    for _ in range(main_len - 2):
        main_heights.append(random.randint(2, 5))
    main_heights.append(last_height)

    extra_chain = random.random() < 0.65
    left_margin = random.randint(1, 3)
    main_start = left_margin
    chains_spec: list[list[dict]] = []

    if extra_chain:
        extra_len = random.randint(1, 3)
        extra_heights = [random.randint(2, 4) for _ in range(extra_len)]
        extra_start = left_margin
        extra_chain_bars = _make_chain(extra_start, extra_heights)
        chains_spec.append(extra_chain_bars)
        gap = random.randint(2, 4)
        main_start = int(extra_chain_bars[-1]["col"] + gap)

    main_chain = _make_chain(main_start, main_heights)
    chains_spec.append(main_chain)

    main_end_col = int(main_chain[-1]["col"])
    rect1_left = int(main_start + sum(main_heights))
    rect1_width = _choose_touch_width(last_height)
    rect1_height = random.randint(4, 8)
    rect1_bbox = (baseline - rect1_height + 1, rect1_left, baseline, rect1_left + rect1_width - 1)

    decoy_height = random.randint(3, last_height - 1)
    decoy_col = int(rect1_left - decoy_height)
    if decoy_col <= main_end_col:
        return None

    occupied_cols = {int(bar["col"]) for chain in chains_spec for bar in chain}
    if decoy_col in occupied_cols:
        return None

    rect2_height = random.randint(4, 7)
    vertical_gap = random.randint(1, 2)
    rect2_bottom = int(rect1_bbox[0] - vertical_gap - 1)
    rect2_top = int(rect2_bottom - rect2_height + 1)
    if rect2_top < 0:
        return None
    rect2_width = random.randint(3, 5)
    rect2_left = int(rect1_left + random.randint(0, 3))
    rect2_bbox = (rect2_top, rect2_left, rect2_bottom, rect2_left + rect2_width - 1)

    rects = [rect1_bbox, rect2_bbox]

    if random.random() < 0.55:
        rect3_width = random.randint(3, 5)
        rect3_height = random.randint(4, 8)
        rect3_left = max(rect1_bbox[3], rect2_bbox[3]) + random.randint(3, 5)
        rect3_bbox = (baseline - rect3_height + 1, rect3_left, baseline, rect3_left + rect3_width - 1)
        rects.append(rect3_bbox)

    rightmost = max(c1 for _r0, _c0, _r1, c1 in rects)
    width = rightmost + 1 + random.randint(1, 3)
    if width > 30:
        return None

    forbidden_next = {int(bar["col"] + bar["height"]) for chain in chains_spec for bar in chain}
    extra_unchained: list[dict] = [{"color": RED, "col": decoy_col, "height": decoy_height}]
    choices = [c for c in range(left_margin, rect1_left - 1) if c not in occupied_cols and c not in forbidden_next and c != decoy_col]
    random.shuffle(choices)
    for _ in range(random.randint(0, 1)):
        if not choices:
            break
        col = choices.pop()
        extra_unchained.append({"color": RED, "col": int(col), "height": random.randint(2, 5)})

    grid = np.full((h, width), BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < width):
            return None
        _paint_rect(grid, rect)

    for chain in chains_spec:
        for bar in chain:
            if int(bar["col"]) >= width:
                return None
            _paint_bar(grid, baseline, bar)

    for bar in extra_unchained:
        if int(bar["col"]) >= width:
            return None
        if np.any(grid[:, int(bar["col"])] != BG):
            return None
        _paint_bar(grid, baseline, bar)

    return grid


def _analyze_semantics(grid: np.ndarray) -> bool:
    try:
        bars, baseline = _extract_bars(grid)
        rects = _extract_rectangles(grid)
        chains = _build_chains(bars)
        _out, stripe_sums, _chains2, _rects2 = _apply_rule(grid, rotate_all_bars=False)
        _all_out, all_rotated_sums, _all_chains, _all_rects = _apply_rule(grid, rotate_all_bars=True)
    except Exception:
        return False

    if len(rects) not in {2, 3}:
        return False
    if not any(len(chain) >= 2 for chain in chains):
        return False

    chained_cols = {bar["col"] for chain in chains for bar in chain}
    unchained_reds = [bar for bar in bars if bar["color"] == RED and bar["col"] not in chained_cols]
    if not unchained_reds:
        return False

    touched = [idx for idx, total in enumerate(stripe_sums) if total > 0]
    untouched = [idx for idx, total in enumerate(stripe_sums) if total == 0]
    if not touched or not untouched:
        return False

    rect_by_cell = _rect_index(rects)
    for bar in bars:
        for rr, cc in bar["coords"]:
            if _segment_touches_rects(int(rr), int(cc), 1, rect_by_cell):
                return False

    rect_to_chains: dict[int, set[int]] = {}
    for chain_idx, chain in enumerate(chains):
        for bar in chain:
            hits = _segment_touches_rects(baseline, int(bar["col"]), int(bar["height"]), rect_by_cell)
            for rect_idx in hits:
                rect_to_chains.setdefault(rect_idx, set()).add(chain_idx)
    if any(len(owners) > 1 for owners in rect_to_chains.values()):
        return False

    primary_touched = touched[0]
    touching_bars = []
    for chain in chains:
        for bar in chain:
            if primary_touched in _segment_touches_rects(baseline, int(bar["col"]), int(bar["height"]), rect_by_cell):
                touching_bars.append(bar)
    if not touching_bars or not all(int(bar["color"]) == RED for bar in touching_bars):
        return False

    if not any(all_rotated_sums[idx] > stripe_sums[idx] for idx in range(len(rects))):
        return False

    if np.array_equal(main(grid), grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _sample_candidate()
        if grid is None:
            continue
        if _analyze_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid chained-rotation stripe puzzle")

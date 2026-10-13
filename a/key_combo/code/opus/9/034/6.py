from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

KEY_COLORS = (RED, BLUE, GREEN, YELLOW)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_base_row(grid: np.ndarray) -> int:
    grey_counts = np.sum(grid == GREY, axis=1)
    candidates = [r for r, cnt in enumerate(grey_counts) if int(cnt) == grid.shape[1]]
    if candidates:
        return int(candidates[-1])
    return int(np.argmax(grey_counts))


def _is_rectangular_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 7:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == GREY):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == GREY):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == GREY):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == GREY):
        return False
    return True


def _find_legend_bbox(grid: np.ndarray, base_r: int) -> tuple[int, int, int, int]:
    best = None
    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 <= base_r <= r1:
            continue
        if not _is_rectangular_frame(grid, (r0, c0, r1, c1)):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        cand = (area, -r0, c0, (r0, c0, r1, c1))
        if best is None or cand > best:
            best = cand
    if best is None:
        raise ValueError("legend frame not found")
    return best[-1]


def _extract_keys(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = legend_bbox
    rows: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        cols = [c for c in range(c0 + 1, c1) if int(grid[r, c]) not in (BLACK, GREY)]
        if not cols:
            continue
        key_c = min(cols)
        rows.append((r, key_c, int(grid[r, key_c])))
    if len(rows) != 4:
        raise ValueError("expected 4 legend key rows")
    return rows


def _find_bars(
    grid: np.ndarray, base_r: int, legend_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = legend_bbox
    color_mask = (grid != BLACK) & (grid != GREY)
    color_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    bars: list[dict[str, object]] = []
    for comp in _components(color_mask, connectivity=4):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        br0, bc0, br1, bc1 = _bbox(comp)
        if br1 != base_r - 1:
            continue
        if bc0 != bc1:
            continue
        if len(comp) != br1 - br0 + 1:
            continue
        bars.append(
            {
                "color": int(next(iter(colors))),
                "col": int(bc0),
                "height": int(len(comp)),
                "coords": comp,
            }
        )
    bars.sort(key=lambda item: int(item["col"]))
    if len(bars) != 4:
        raise ValueError("expected 4 bars")
    return bars


def _count_color_objects(
    grid: np.ndarray,
    color: int,
    legend_bbox: tuple[int, int, int, int],
    bars: list[dict[str, object]],
) -> int:
    mask = grid == color
    r0, c0, r1, c1 = legend_bbox
    mask = mask.copy()
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    for bar in bars:
        for r, c in bar["coords"]:
            mask[r, c] = False
    return len(_components(mask, connectivity=4))


def _render_output(
    shape: tuple[int, int],
    base_r: int,
    legend_bbox: tuple[int, int, int, int],
    keys: list[tuple[int, int, int]],
    bars: list[dict[str, object]],
    counts_by_color: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[base_r, :] = GREY

    r0, c0, r1, c1 = legend_bbox
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    for row, key_c, color in keys:
        count = int(counts_by_color[color])
        for i in range(count):
            cc = key_c + 2 * i
            if cc < c1:
                out[row, cc] = color

    sorted_heights = sorted(int(v) for v in counts_by_color.values())
    for slot, new_h in zip(sorted(bars, key=lambda item: int(item["col"])), sorted_heights):
        col = int(slot["col"])
        color = int(slot["color"])
        for r in range(base_r - new_h, base_r):
            if 0 <= r < shape[0]:
                out[r, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count scattered colored objects, write those counts into the legend, then rebuild and sort the bars by the new heights.

    Input structure:
    - The background is black, with one grey horizontal baseline near the bottom and four single-column colored bars directly above it.
    - A grey rectangular legend frame sits away from the bars; inside it, four colored key pixels identify the tracked colors.
    - Elsewhere, detached 4-connected objects in those same colors are scattered without overlapping the legend or bars.

    Rule:
    - Detect the legend frame, the baseline, and the four bars, then count 4-connected objects of each key color only outside the legend and bar regions.
    - In each legend key row, replace the single key pixel by a spaced dot sequence of that color whose length equals the counted object total.
    - Use those four counts as replacement bar heights, sort the heights from shortest to tallest, and assign them left-to-right to the fixed bar slots while keeping each slot's original color.
    - Erase every scattered object, but preserve the baseline and the legend frame.

    Key insight:
    - The distant objects affect the bars only through their connected-component counts: the legend visualizes the counts, and the same counts are then sorted and redistributed as the new bar heights.
    """
    g = np.asarray(grid, dtype=int)
    base_r = _find_base_row(g)
    legend_bbox = _find_legend_bbox(g, base_r)
    keys = _extract_keys(g, legend_bbox)
    bars = _find_bars(g, base_r, legend_bbox)
    counts_by_color = {
        int(color): _count_color_objects(g, int(color), legend_bbox, bars)
        for _row, _key_c, color in keys
    }
    return _render_output(g.shape, base_r, legend_bbox, keys, bars, counts_by_color)


def _random_shape(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
        target = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            cr, cc = rng.choice(tuple(cells))
            dr, dc = rng.choice(NEIGHBORS4)
            cells.add((cr + dr, cc + dc))
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        min_r, min_c = min(rs), min(cs)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2:
            continue
        if len(norm) == h * w:
            continue
        return norm
    raise RuntimeError("failed to sample irregular shape")


def _shape8_halo(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        halo.add((r, c))
        for dr, dc in NEIGHBORS8:
            halo.add((r + dr, c + dc))
    return halo


def _place_objects(
    grid: np.ndarray,
    rng: random.Random,
    counts_by_color: dict[int, int],
    forbidden: set[tuple[int, int]],
    base_r: int,
) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    placed: dict[int, list[list[tuple[int, int]]]] = {color: [] for color in KEY_COLORS}
    occupied = set(forbidden)
    placement_order = []
    for color in KEY_COLORS:
        for _ in range(int(counts_by_color[color])):
            placement_order.append(color)
    rng.shuffle(placement_order)

    for color in placement_order:
        success = False
        for _ in range(500):
            shape = _random_shape(rng)
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            top = rng.randint(0, max(0, base_r - sh - 2))
            left = rng.randint(0, w - sw)
            coords = [(top + r, left + c) for r, c in shape]
            if any((r, c) in occupied for r, c in coords):
                continue
            halo = _shape8_halo(coords)
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in halo if (rr, cc) not in forbidden):
                continue
            if any((rr, cc) in occupied for rr, cc in halo):
                continue
            for r, c in coords:
                grid[r, c] = color
            occupied.update(halo)
            placed[color].append(coords)
            success = True
            break
        if not success:
            raise RuntimeError("failed to place objects")
    return placed


def _meta_output(
    shape: tuple[int, int],
    base_r: int,
    legend_bbox: tuple[int, int, int, int],
    key_rows: list[int],
    key_col: int,
    slot_cols: list[int],
    initial_colors: list[int],
    counts_by_color: dict[int, int],
) -> np.ndarray:
    keys = [(row, key_col, color) for row, color in zip(key_rows, initial_colors)]
    bars = [{"col": col, "color": color, "coords": []} for col, color in zip(slot_cols, initial_colors)]
    return _render_output(shape, base_r, legend_bbox, keys, bars, counts_by_color)


def _meta_unsorted_output(
    shape: tuple[int, int],
    base_r: int,
    legend_bbox: tuple[int, int, int, int],
    key_rows: list[int],
    key_col: int,
    slot_cols: list[int],
    initial_colors: list[int],
    counts_by_color: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[base_r, :] = GREY
    r0, c0, r1, c1 = legend_bbox
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    for row, color in zip(key_rows, initial_colors):
        for i in range(int(counts_by_color[color])):
            cc = key_col + 2 * i
            if cc < c1:
                out[row, cc] = color
    for col, color in zip(slot_cols, initial_colors):
        new_h = int(counts_by_color[color])
        for r in range(base_r - new_h, base_r):
            out[r, col] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    meta: dict[str, object],
) -> bool:
    base_r = int(meta["base_r"])
    legend_bbox = tuple(meta["legend_bbox"])
    slot_cols = list(meta["slot_cols"])
    initial_colors = list(meta["slot_colors"])
    counts_by_color = dict(meta["counts_by_color"])

    parsed_base = _find_base_row(grid)
    if parsed_base != base_r:
        return False
    parsed_legend = _find_legend_bbox(grid, parsed_base)
    if parsed_legend != legend_bbox:
        return False
    keys = _extract_keys(grid, parsed_legend)
    if [row for row, _, _ in keys] != list(meta["key_rows"]):
        return False
    if [color for _, _, color in keys] != initial_colors:
        return False
    parsed_bars = _find_bars(grid, parsed_base, parsed_legend)
    if [int(bar["col"]) for bar in parsed_bars] != slot_cols:
        return False
    if [int(bar["color"]) for bar in parsed_bars] != initial_colors:
        return False
    parsed_counts = {
        int(color): _count_color_objects(grid, int(color), parsed_legend, parsed_bars)
        for color in KEY_COLORS
    }
    if parsed_counts != counts_by_color:
        return False

    heights_by_slot = [counts_by_color[color] for color in initial_colors]
    if heights_by_slot == sorted(heights_by_slot):
        return False
    if sorted(list(meta["initial_heights"])) == sorted(counts_by_color.values()):
        return False
    if not any(
        int(meta["initial_heights"][i]) < heights_by_slot[i] for i in range(4)
    ):
        return False
    if not any(
        int(meta["initial_heights"][i]) > heights_by_slot[i] for i in range(4)
    ):
        return False

    pixel_totals = {color: 0 for color in KEY_COLORS}
    for color in KEY_COLORS:
        pixel_totals[color] = sum(len(coords) for coords in meta["objects"][color])
    count_order = sorted(KEY_COLORS, key=lambda color: counts_by_color[color])
    pixel_order = sorted(KEY_COLORS, key=lambda color: (pixel_totals[color], color))
    if count_order == pixel_order:
        return False

    expected = _meta_output(
        grid.shape,
        base_r,
        legend_bbox,
        list(meta["key_rows"]),
        int(meta["key_col"]),
        slot_cols,
        initial_colors,
        counts_by_color,
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    unsorted = _meta_unsorted_output(
        grid.shape,
        base_r,
        legend_bbox,
        list(meta["key_rows"]),
        int(meta["key_col"]),
        slot_cols,
        initial_colors,
        counts_by_color,
    )
    if np.array_equal(actual, unsorted):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        counts = rng.sample([1, 2, 3, 4, 5, 6], 4)
        counts_by_color = {color: count for color, count in zip(KEY_COLORS, counts)}
        if list(counts) == sorted(counts):
            continue

        initial_heights = rng.sample([3, 4, 5, 6, 7], 4)
        if sorted(initial_heights) == sorted(counts):
            continue
        if not any(initial_heights[i] < counts[i] for i in range(4)):
            continue
        if not any(initial_heights[i] > counts[i] for i in range(4)):
            continue

        max_count = max(counts)
        legend_inner_w = 2 * max_count - 1 + rng.randint(0, 1)
        legend_w = legend_inner_w + 2
        legend_h = 9
        right_margin = rng.randint(1, 2)

        slot_start = rng.randint(2, 4)
        feasible_gaps = [
            gap
            for gap in (3, 4, 5)
            if slot_start + 3 * gap + legend_w + right_margin + 2 <= 30
        ]
        if not feasible_gaps:
            continue
        slot_gap = rng.choice(feasible_gaps)
        slot_cols = [slot_start + i * slot_gap for i in range(4)]

        min_w = slot_cols[-1] + legend_w + right_margin + 2
        if min_w > 30:
            continue
        w = rng.randint(max(min_w, 22), 30)

        legend_c0 = w - legend_w - right_margin
        if legend_c0 <= slot_cols[-1] + 1:
            continue

        h = rng.randint(20, 28)
        base_r = h - rng.randint(2, 4)
        legend_r0 = rng.randint(1, 3)
        if legend_r0 + legend_h + 3 >= base_r:
            continue

        grid = np.zeros((h, w), dtype=int)
        grid[base_r, :] = GREY

        for col, color, height in zip(slot_cols, KEY_COLORS, initial_heights):
            grid[base_r - height : base_r, col] = color

        legend_bbox = (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1)
        r0, c0, r1, c1 = legend_bbox
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        key_col = c0 + 1
        key_rows = [r0 + 1 + 2 * i for i in range(4)]
        for row, color in zip(key_rows, KEY_COLORS):
            grid[row, key_col] = color

        forbidden: set[tuple[int, int]] = set()
        for rr in range(r0 - 1, r1 + 2):
            for cc in range(c0 - 1, c1 + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden.add((rr, cc))
        for col, height in zip(slot_cols, initial_heights):
            for rr in range(base_r - height - 1, base_r + 2):
                for cc in range(col - 1, col + 2):
                    if 0 <= rr < h and 0 <= cc < w:
                        forbidden.add((rr, cc))
        for cc in range(w):
            forbidden.add((base_r, cc))
            if base_r + 1 < h:
                forbidden.add((base_r + 1, cc))

        try:
            objects = _place_objects(grid, rng, counts_by_color, forbidden, base_r)
        except RuntimeError:
            continue

        meta = {
            "base_r": base_r,
            "legend_bbox": legend_bbox,
            "slot_cols": slot_cols,
            "slot_colors": list(KEY_COLORS),
            "initial_heights": initial_heights,
            "counts_by_color": counts_by_color,
            "key_rows": key_rows,
            "key_col": key_col,
            "objects": objects,
        }
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _grey_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in np.unique(grid) if int(v) != bg]
    if GREY in vals:
        return GREY
    counts = Counter(map(int, grid.flat))
    return max(vals, key=lambda color: counts[color])


def _parse_legend(grid: np.ndarray, bg: int, grey: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    color_mask = (grid != bg) & (grid != grey)
    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[tuple[int, int], int] = {}

    for pts in _components(color_mask):
        if len(pts) != 3:
            continue
        rows = {r for r, _ in pts}
        cols = sorted(c for _, c in pts)
        if len(rows) != 1 or cols[2] - cols[0] != 2 or len(set(cols)) != 3:
            continue
        if not any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        r = next(iter(rows))
        triplet = [int(grid[r, cols[0]]), int(grid[r, cols[1]]), int(grid[r, cols[2]])]
        a, b, mixed = triplet
        mapping[(a, b)] = mixed
        mapping[(b, a)] = mixed
        for c in cols:
            legend_mask[r, c] = True

    assert mapping, "legend must contain at least one edge 1x3 mixing row"
    return mapping, legend_mask


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    return sum(1 for nr, nc in _neighbors4(r, c, h, w) if bool(mask[nr, nc]))


def _trace_marker(
    start: tuple[int, int],
    grid: np.ndarray,
    grey: int,
    network_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    next_steps = [(nr, nc) for nr, nc in _neighbors4(sr, sc, h, w) if bool(network_mask[nr, nc])]
    assert len(next_steps) == 1, "each marker must sit on a unique dead-end branch"

    prev = start
    cur = next_steps[0]
    seen = {start}
    trail: list[tuple[int, int]] = []

    while True:
        cr, cc = cur
        assert int(grid[cr, cc]) == grey, "marker paths must enter grey channels before any junction"
        if _degree(network_mask, cr, cc) >= 3:
            return trail, cur

        trail.append(cur)
        seen.add(cur)
        next_cells = [
            (nr, nc)
            for nr, nc in _neighbors4(cr, cc, h, w)
            if bool(network_mask[nr, nc]) and (nr, nc) != prev
        ]
        assert len(next_cells) == 1, "each marked branch must lead unambiguously to its first junction"
        nxt = next_cells[0]
        assert nxt not in seen, "channel branches must not loop before the first junction"
        prev, cur = cur, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored dead-end markers along grey channels until they reach their first junction.

    Input structure:
    - The background is the dominant black color.
    - Grey 1-pixel-wide channels form disconnected or connected T-junction and cross-junction networks.
    - Some dead ends are replaced by single colored marker pixels, with at most one marker per branch.
    - A legend on the grid edge contains horizontal 1x3 rows: two input colors followed by their mixed result color.

    Rule:
    - Treat grey cells together with the non-legend marker pixels as the channel network.
    - For each marker, follow its unique branch through grey cells until the first grey cell whose network degree is at least three.
    - Recolor every traversed grey channel cell with the marker's color.
    - If exactly one marker reaches a junction, recolor that junction with the marker color; if exactly two markers reach it, use the legend's color-mixing lookup; untouched junctions stay grey.

    Key insight:
    - The task combines path tracing on channel topology with a separate legend lookup: solve where each marker stops first, then decide each junction color from the set of arriving marker colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    grey = _grey_color(g, bg)

    mix_map, legend_mask = _parse_legend(g, bg, grey)
    marker_mask = (g != bg) & (g != grey) & (~legend_mask)
    network_mask = (g == grey) | marker_mask

    out = g.copy()
    arrivals: dict[tuple[int, int], list[int]] = {}

    for r, c in np.argwhere(marker_mask):
        start = (int(r), int(c))
        color = int(g[start])
        trail, junction = _trace_marker(start, g, grey, network_mask)
        for tr, tc in trail:
            out[tr, tc] = color
        arrivals.setdefault(junction, []).append(color)

    for (r, c), colors in arrivals.items():
        if len(colors) == 1:
            out[r, c] = int(colors[0])
        elif len(colors) == 2:
            pair = (int(colors[0]), int(colors[1]))
            assert pair in mix_map, "every two-marker meeting must appear in the legend"
            out[r, c] = int(mix_map[pair])
        else:
            out[r, c] = grey

    return out


def _halo(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _legend_cells(
    h: int,
    w: int,
    side: str,
    start_r: int,
    rows: list[tuple[int, int, int]],
) -> dict[tuple[int, int], int]:
    c0 = 0 if side == "left" else w - 3
    cells: dict[tuple[int, int], int] = {}
    for idx, (a, b, mixed) in enumerate(rows):
        r = start_r + 2 * idx
        cells[(r, c0 + 0)] = int(a)
        cells[(r, c0 + 1)] = int(b)
        cells[(r, c0 + 2)] = int(mixed)
    return cells


def _build_gadget(
    rng: random.Random,
    arrival_count: int,
    marker_colors: list[int],
) -> dict:
    active_dirs = ["U", "D", "L", "R"]
    kind = rng.choice(["T", "cross"])
    missing = None
    if kind == "T":
        missing = rng.choice(active_dirs)
        active_dirs = [d for d in active_dirs if d != missing]

    lengths = {d: rng.randint(2, 5) for d in active_dirs}
    marked_dirs = rng.sample(active_dirs, arrival_count)
    dir_to_color = {d: int(color) for d, color in zip(marked_dirs, marker_colors)}

    rel_cells: dict[tuple[int, int], int] = {(0, 0): GREY}
    for d in active_dirs:
        dr, dc = DIR_TO_DELTA[d]
        arm_len = lengths[d]
        for step in range(1, arm_len + 1):
            rr, cc = dr * step, dc * step
            if step == arm_len and d in dir_to_color:
                rel_cells[(rr, cc)] = dir_to_color[d]
            else:
                rel_cells[(rr, cc)] = GREY

    return {
        "kind": kind,
        "missing": missing,
        "active_dirs": active_dirs,
        "lengths": lengths,
        "marked_dirs": marked_dirs,
        "dir_to_color": dir_to_color,
        "rel_cells": rel_cells,
    }


def _place_gadget(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    spec: dict,
    rng: random.Random,
) -> tuple[tuple[int, int], dict[tuple[int, int], int]] | None:
    h, w = grid.shape
    rel_pts = list(spec["rel_cells"].keys())
    min_r = min(r for r, _ in rel_pts)
    max_r = max(r for r, _ in rel_pts)
    min_c = min(c for _, c in rel_pts)
    max_c = max(c for _, c in rel_pts)

    low_r = 1 - min_r
    high_r = h - 2 - max_r
    low_c = 1 - min_c
    high_c = w - 2 - max_c
    if low_r > high_r or low_c > high_c:
        return None

    centers = [(r, c) for r in range(low_r, high_r + 1) for c in range(low_c, high_c + 1)]
    rng.shuffle(centers)
    for center_r, center_c in centers:
        abs_cells = {
            (center_r + rr, center_c + cc): int(color)
            for (rr, cc), color in spec["rel_cells"].items()
        }
        if any((r, c) in blocked for r, c in abs_cells):
            continue
        return (center_r, center_c), abs_cells
    return None


def _pair_mapping(marker_colors: list[int], mix_colors: list[int]) -> dict[tuple[int, int], int]:
    pairs = list(combinations(marker_colors, 2))
    return {tuple(pair): int(mix) for pair, mix in zip(pairs, mix_colors)}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)

    marker_colors = rng.sample(COLOR_POOL, 3)
    remaining = [c for c in COLOR_POOL if c not in marker_colors]
    mix_colors = rng.sample(remaining, 3)
    pair_to_mix = _pair_mapping(marker_colors, mix_colors)

    pair_rows = list(pair_to_mix.items())
    rng.shuffle(pair_rows)
    legend_rows: list[tuple[int, int, int]] = []
    for (a, b), mixed in pair_rows:
        if rng.random() < 0.5:
            a, b = b, a
        legend_rows.append((int(a), int(b), int(mixed)))

    legend_side = rng.choice(["left", "right"])
    legend_span = 2 * len(legend_rows) - 1
    start_r = rng.randint(1, h - legend_span - 1)
    legend = _legend_cells(h, w, legend_side, start_r, legend_rows)

    for (r, c), color in legend.items():
        grid[r, c] = color

    blocked = _halo(set(legend.keys()), h, w, radius=1)
    gadget_arrivals = [2, 1, 0]
    if rng.random() < 0.5:
        gadget_arrivals.append(0)
    rng.shuffle(gadget_arrivals)

    color_cursor = 0
    gadget_infos: list[dict] = []
    for arrival_count in gadget_arrivals:
        colors = marker_colors[color_cursor : color_cursor + arrival_count]
        color_cursor += arrival_count
        spec = _build_gadget(rng, arrival_count, colors)
        placed = _place_gadget(grid, blocked, spec, rng)
        if placed is None:
            return None

        center, abs_cells = placed
        for (r, c), color in abs_cells.items():
            grid[r, c] = color
        blocked |= _halo(set(abs_cells.keys()), h, w, radius=1)

        expected = GREY
        if arrival_count == 1:
            expected = int(colors[0])
        elif arrival_count == 2:
            expected = int(pair_to_mix[(int(colors[0]), int(colors[1]))])
        gadget_infos.append(
            {
                "center": center,
                "arrival_count": arrival_count,
                "expected_center": expected,
            }
        )

    if color_cursor != len(marker_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or not np.all(out[np.array(list(legend.keys())).T[0], np.array(list(legend.keys())).T[1]] == grid[np.array(list(legend.keys())).T[0], np.array(list(legend.keys())).T[1]]):
        return None

    seen_counts = {info["arrival_count"] for info in gadget_infos}
    if seen_counts != {0, 1, 2}:
        return None

    for info in gadget_infos:
        r, c = info["center"]
        if int(out[r, c]) != int(info["expected_center"]):
            return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")

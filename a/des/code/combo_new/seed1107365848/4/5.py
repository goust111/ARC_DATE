from __future__ import annotations

import random

import numpy as np


BG = 0
ITEM = 7
BELT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_SIZE_RANGE = (18, 28)
BELT_COUNT_RANGE = (3, 5)
TRACK_MIN_GAP = 3
SPAN_MARGIN_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 500
MIN_VISIBLE_PER_BELT = 2
MIN_UNIQUE_FINALS = 2

CORNERS = ("tl", "tr", "bl", "br")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _near_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    near_top = r0 <= 1
    near_bottom = h - 1 - r1 <= 1
    near_left = c0 <= 1
    near_right = w - 1 - c1 <= 1
    return (near_top or near_bottom) and (near_left or near_right)


def _find_legend(g: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mask = (g != BG) & (g != ITEM)
    candidates: list[tuple[int, np.ndarray, list[int]]] = []
    for comp in _components(mask):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if c0 != c1 or r1 - r0 + 1 != len(comp) or len(comp) < 2:
            continue
        bbox = (r0, c0, r1, c1)
        if not _near_corner(bbox, g.shape):
            continue
        colors = [int(g[r, c0]) for r in range(r0, r1 + 1)]
        if len(set(colors)) != len(colors):
            continue
        legend_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            legend_mask[r, c] = True
        candidates.append((len(comp), legend_mask, colors))

    assert len(candidates) == 1, "expected one isolated corner legend column"
    _size, legend_mask, priority = candidates[0]
    return legend_mask, priority


def _occupied(g: np.ndarray, legend_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = g.shape
    return 0 <= r < h and 0 <= c < w and not legend_mask[r, c] and int(g[r, c]) != BG


def _endpoint_sign(
    g: np.ndarray, legend_mask: np.ndarray, pos: tuple[int, int], orientation: str
) -> int | None:
    r, c = pos
    left = _occupied(g, legend_mask, r, c - 1)
    right = _occupied(g, legend_mask, r, c + 1)
    up = _occupied(g, legend_mask, r - 1, c)
    down = _occupied(g, legend_mask, r + 1, c)
    if orientation == "h":
        if up or down or left == right:
            return None
        return 1 if right else -1
    if left or right or up == down:
        return None
    return 1 if down else -1


def _trace_path(
    g: np.ndarray,
    legend_mask: np.ndarray,
    start: tuple[int, int],
    orientation: str,
    sign: int,
) -> list[tuple[int, int]]:
    dr, dc = (0, sign) if orientation == "h" else (sign, 0)
    path = [start]
    r, c = start
    while True:
        nr, nc = r + dr, c + dc
        if not _occupied(g, legend_mask, nr, nc):
            break
        path.append((int(nr), int(nc)))
        r, c = nr, nc
    return path


def _reconstruct_belts(
    g: np.ndarray, legend_mask: np.ndarray, priority: list[int]
) -> dict[int, dict[str, object]]:
    orange_positions = [tuple(map(int, coord)) for coord in np.argwhere(g == ITEM)]
    belts: dict[int, dict[str, object]] = {}

    for color in priority:
        visible = {
            tuple(map(int, coord))
            for coord in np.argwhere((g == int(color)) & (~legend_mask))
        }
        assert visible, f"expected visible cells for belt color {color}"

        candidates: list[dict[str, object]] = []
        for orange in orange_positions:
            for orientation in ("h", "v"):
                sign = _endpoint_sign(g, legend_mask, orange, orientation)
                if sign is None:
                    continue
                path = _trace_path(g, legend_mask, orange, orientation, sign)
                if visible <= set(path) and int(g[path[-1]]) == int(color):
                    candidates.append(
                        {
                            "color": int(color),
                            "orientation": orientation,
                            "start": orange,
                            "path": path,
                            "index": {pos: i for i, pos in enumerate(path)},
                        }
                    )

        assert len(candidates) == 1, f"expected one reconstructed belt for color {color}"
        belts[int(color)] = candidates[0]

    return belts


def _simulate_items(
    g: np.ndarray,
    belts: dict[int, dict[str, object]],
    priority: list[int],
) -> tuple[np.ndarray, dict[str, int]]:
    priority_rank = {int(color): i for i, color in enumerate(priority)}
    cell_to_belts: dict[tuple[int, int], list[int]] = {}
    for color, belt in belts.items():
        for pos in belt["path"]:
            cell_to_belts.setdefault(pos, []).append(int(color))

    out = g.copy()
    events = {"transfer": 0, "ignore": 0}

    for color, belt in belts.items():
        start = belt["start"]
        out[start] = int(color)

        current = int(color)
        idx = 1
        while True:
            path = belts[current]["path"]
            if idx >= len(path):
                final_pos = path[-1]
                break

            pos = path[idx]
            carriers = cell_to_belts[pos]
            assert len(carriers) <= 2, "crossings must involve at most two belts"

            if len(carriers) == 2:
                shown = int(g[pos])
                if priority_rank[shown] < priority_rank[current]:
                    other = carriers[0] if carriers[1] == current else carriers[1]
                    current = int(other)
                    idx = belts[current]["index"][pos] + 1
                    events["transfer"] += 1
                    continue
                events["ignore"] += 1

            idx += 1

        out[final_pos] = ITEM

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each orange item along its colored conveyor, switching only onto higher-priority belts at visible crossings.

    Input structure:
    - The background is black.
    - Away from one corner legend, each non-orange foreground color is a single
      horizontal or vertical one-pixel-wide belt whose start endpoint is replaced
      by an orange item and whose interior may contain crossing pixels colored by
      other belts.
    - The legend is an isolated vertical column of distinct colors in a corner,
      ordered from top to bottom as belt priority from high to low.

    Rule:
    - Recover the priority order from the legend and reconstruct every straight
      belt from its orange endpoint through the contiguous occupied cells to its
      same-colored terminal endpoint.
    - For each item, walk forward along its current belt; at any cell shared by
      two belts, compare the displayed color at that crossing with the current
      belt color.
    - If the displayed crossing color has higher legend priority, transfer onto
      the other belt and continue from that crossing toward that belt's terminal;
      otherwise keep moving on the current belt.
    - Keep the belts and legend unchanged, recolor every original orange start
      cell back to its belt color, and place orange items at their final stop
      cells.

    Key insight:
    - A crossing's visible color is the transfer signal: the item ignores the
      hidden belt unless the shown color outranks the belt it is currently riding.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_mask, priority = _find_legend(g)
    belts = _reconstruct_belts(g, legend_mask, priority)
    out, _events = _simulate_items(g, belts, priority)
    return out


def _sample_spaced(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    required = count + (count - 1) * (min_gap - 1)
    if span < required:
        return None
    compressed = span - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(compressed), count))
    return [int(lo + pick + i * (min_gap - 1)) for i, pick in enumerate(picks)]


def _legend_layout(
    height: int, width: int, belt_count: int, corner: str
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    if corner == "tl":
        legend_cells = [(r, 1) for r in range(1, belt_count + 1)]
        arena = (belt_count + 2, height - 3, 3, width - 3)
    elif corner == "tr":
        legend_cells = [(r, width - 2) for r in range(1, belt_count + 1)]
        arena = (belt_count + 2, height - 3, 2, width - 4)
    elif corner == "bl":
        legend_cells = [(height - belt_count - 1 + i, 1) for i in range(belt_count)]
        arena = (2, height - belt_count - 3, 3, width - 3)
    else:
        assert corner == "br"
        legend_cells = [(height - belt_count - 1 + i, width - 2) for i in range(belt_count)]
        arena = (2, height - belt_count - 3, 2, width - 4)
    return legend_cells, arena


def _make_belt(
    color: int,
    orientation: str,
    fixed_coord: int,
    span_lo: int,
    span_hi: int,
    start_on_low_side: bool,
) -> dict[str, object]:
    if orientation == "h":
        row = fixed_coord
        if start_on_low_side:
            path = [(row, c) for c in range(span_lo, span_hi + 1)]
        else:
            path = [(row, c) for c in range(span_hi, span_lo - 1, -1)]
    else:
        col = fixed_coord
        if start_on_low_side:
            path = [(r, col) for r in range(span_lo, span_hi + 1)]
        else:
            path = [(r, col) for r in range(span_hi, span_lo - 1, -1)]

    return {
        "color": int(color),
        "orientation": orientation,
        "start": path[0],
        "path": path,
        "index": {pos: i for i, pos in enumerate(path)},
    }


def _render_input(
    shape: tuple[int, int],
    legend_cells: list[tuple[int, int]],
    priority: list[int],
    belts: dict[int, dict[str, object]],
    crossing_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    g = np.full(shape, BG, dtype=int)
    for (r, c), color in zip(legend_cells, priority):
        g[r, c] = int(color)
    for belt in belts.values():
        for r, c in belt["path"]:
            g[r, c] = int(belt["color"])
    for (r, c), color in crossing_colors.items():
        g[r, c] = int(color)
    for belt in belts.values():
        sr, sc = belt["start"]
        g[sr, sc] = ITEM
    return g


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    belt_count = random.randint(*BELT_COUNT_RANGE)
    corner = random.choice(CORNERS)

    legend_cells, arena = _legend_layout(height, width, belt_count, corner)
    row_lo, row_hi, col_lo, col_hi = arena
    if row_lo > row_hi or col_lo > col_hi:
        return None

    n_h = random.randint(1, belt_count - 1)
    n_v = belt_count - n_h
    rows = _sample_spaced(row_lo, row_hi, n_h, TRACK_MIN_GAP)
    cols = _sample_spaced(col_lo, col_hi, n_v, TRACK_MIN_GAP)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    priority = random.sample(BELT_COLORS, belt_count)
    belt_colors = priority.copy()
    random.shuffle(belt_colors)

    belt_defs: list[dict[str, object]] = []
    color_iter = iter(belt_colors)

    for row in rows:
        left_margin = random.randint(*SPAN_MARGIN_RANGE)
        right_margin = random.randint(*SPAN_MARGIN_RANGE)
        span_lo = max(col_lo, min_col - left_margin)
        span_hi = min(col_hi, max_col + right_margin)
        if span_lo >= min_col or span_hi <= max_col:
            return None
        belt_defs.append(
            _make_belt(
                next(color_iter),
                "h",
                row,
                span_lo,
                span_hi,
                start_on_low_side=bool(random.getrandbits(1)),
            )
        )

    for col in cols:
        top_margin = random.randint(*SPAN_MARGIN_RANGE)
        bottom_margin = random.randint(*SPAN_MARGIN_RANGE)
        span_lo = max(row_lo, min_row - top_margin)
        span_hi = min(row_hi, max_row + bottom_margin)
        if span_lo >= min_row or span_hi <= max_row:
            return None
        belt_defs.append(
            _make_belt(
                next(color_iter),
                "v",
                col,
                span_lo,
                span_hi,
                start_on_low_side=bool(random.getrandbits(1)),
            )
        )

    belts = {int(belt["color"]): belt for belt in belt_defs}
    priority_rank = {int(color): i for i, color in enumerate(priority)}

    crossing_colors: dict[tuple[int, int], int] = {}
    horiz = [belt for belt in belt_defs if belt["orientation"] == "h"]
    vert = [belt for belt in belt_defs if belt["orientation"] == "v"]
    for hb in horiz:
        hr = hb["path"][0][0]
        hcols = {c for _r, c in hb["path"]}
        for vb in vert:
            vc = vb["path"][0][1]
            vrows = {r for r, _c in vb["path"]}
            pos = (hr, vc)
            if vc not in hcols or hr not in vrows:
                continue
            higher = hb["color"] if priority_rank[hb["color"]] < priority_rank[vb["color"]] else vb["color"]
            lower = vb["color"] if higher == hb["color"] else hb["color"]
            crossing_colors[pos] = higher if random.random() < 0.5 else lower

    inp = _render_input((height, width), legend_cells, priority, belts, crossing_colors)

    for belt in belts.values():
        visible_count = sum(int(inp[r, c]) == int(belt["color"]) for r, c in belt["path"][1:])
        if visible_count < MIN_VISIBLE_PER_BELT:
            return None

    expected, events = _simulate_items(inp, belts, priority)
    if events["transfer"] < 1 or events["ignore"] < 1:
        return None
    if int(np.count_nonzero(expected == ITEM)) < MIN_UNIQUE_FINALS:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conveyor-transfer input")

import random
from collections import Counter

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        neigh = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paste_transparent(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left
    assert top + h <= grid.shape[0] and left + w <= grid.shape[1]
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _dominant_color(values: np.ndarray) -> int:
    vals = [int(v) for v in values.reshape(-1) if int(v) != BG]
    assert vals, "edge must contain at least one non-background pixel"
    counts = Counter(vals).most_common()
    assert len(counts) == 1 or counts[0][1] > counts[1][1], "edge dominant color must be unique"
    return int(counts[0][0])


def _edge_color_map(patch: np.ndarray) -> dict[str, int]:
    return {
        "up": _dominant_color(patch[0, :]),
        "down": _dominant_color(patch[-1, :]),
        "left": _dominant_color(patch[:, 0]),
        "right": _dominant_color(patch[:, -1]),
    }


def _side_probe_coords(
    bbox: tuple[int, int, int, int], side: str, gap: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return [(r0 - gap, c) for c in range(c0, c1 + 1)]
    if side == "down":
        return [(r1 + gap, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0 - gap) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1 + gap) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def _find_seed(grid: np.ndarray, bg: int) -> dict:
    h, w = grid.shape
    candidates = []
    for comp in _components(grid != bg, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        ph = r1 - r0 + 1
        pw = c1 - c0 + 1
        if ph not in (3, 4) or pw not in (3, 4):
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) < 2:
            continue

        bbox = (r0, c0, r1, c1)
        for gap in (2, 1):
            side_hits: dict[str, list[tuple[int, int]]] = {}
            indicator_cells: list[tuple[int, int]] = []
            indicator_colors: list[int] = []
            ok = True
            for side in DIRS:
                hits = []
                for rr, cc in _side_probe_coords(bbox, side, gap):
                    if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != int(bg):
                        hits.append((rr, cc))
                if hits:
                    vals = {int(grid[rr, cc]) for rr, cc in hits}
                    if len(vals) != 1:
                        ok = False
                        break
                    side_hits[side] = hits
                    indicator_cells.extend(hits)
                    indicator_colors.extend(int(grid[rr, cc]) for rr, cc in hits)
            if not ok or not side_hits:
                continue
            if len(set(indicator_colors)) != 1:
                continue
            indicator_color = indicator_colors[0]
            if indicator_color in colors:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            candidates.append(
                {
                    "bbox": bbox,
                    "patch": patch,
                    "indicator_color": int(indicator_color),
                    "indicator_sides": set(side_hits),
                    "indicator_cells": set(indicator_cells),
                    "gap": int(gap),
                }
            )
            break

    assert len(candidates) == 1, "expected exactly one seed with side indicators"
    return candidates[0]


def _read_order_bar(
    grid: np.ndarray, bg: int, seed_bbox: tuple[int, int, int, int], indicator_cells: set[tuple[int, int]]
) -> list[int]:
    r0, c0, r1, c1 = seed_bbox
    bar_cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        if (rr, cc) in indicator_cells:
            continue
        bar_cells.append((rr, cc))

    assert len(bar_cells) >= 2, "expected a sequence bar with at least two cells"
    rows = {r for r, _ in bar_cells}
    cols = {c for _, c in bar_cells}
    assert len(rows) == 1 or len(cols) == 1, "sequence bar must lie on one row or one column"

    if len(rows) == 1:
        ordered = sorted(bar_cells, key=lambda rc: rc[1])
    else:
        ordered = sorted(bar_cells, key=lambda rc: rc[0])
    return [int(grid[r, c]) for r, c in ordered]


def _render_chain(
    shape: tuple[int, int], patch: np.ndarray, start: tuple[int, int], directions: list[str], bg: int
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    h, w = patch.shape
    cur_r, cur_c = map(int, start)
    _paste_transparent(out, patch, cur_r, cur_c)
    for direction in directions:
        dr, dc = DIRS[direction]
        cur_r += int(dr * h)
        cur_c += int(dc * w)
        _paste_transparent(out, patch, cur_r, cur_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the multicolor seed along a color-encoded directional growth sequence to form a polyline chain.

    Input structure:
    - The background is black.
    - One small 3x3 or 4x4 multicolor seed object appears away from the border.
    - Single-pixel side indicators of one shared color sit near some seed sides with a one-cell gap.
    - A separate border row or border column forms an order bar whose pixel colors encode growth steps.

    Rule:
    - Find the unique multicolor seed that has nearby side indicators and record which sides are marked.
    - Read the seed's dominant edge color on each side; these edge colors map order-bar colors to directions.
    - Keep only directions whose sides have indicators, then decode the order bar left-to-right or top-to-bottom.
    - Starting from the original seed position, repeatedly copy the whole seed by one seed height or width per step on a blank canvas.

    Key insight:
    - The bar does not name directions directly: each bar color must first be matched to the seed side where that color dominates, and the nearby indicator pixels tell which sides are valid growth directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])

    seed = _find_seed(g, bg)
    patch = seed["patch"]
    color_by_side = _edge_color_map(patch)
    color_to_dir = {
        int(color_by_side[side]): side
        for side in seed["indicator_sides"]
    }
    assert len(color_to_dir) == len(seed["indicator_sides"]), "indicator sides must have distinct dominant colors"

    order_colors = _read_order_bar(g, bg, seed["bbox"], seed["indicator_cells"])
    directions = [color_to_dir[int(color)] for color in order_colors]

    start = (seed["bbox"][0], seed["bbox"][1])
    return _render_chain(g.shape, patch, start, directions, bg)


def _make_seed_patch(height: int, width: int, rng: random.Random) -> tuple[np.ndarray, dict[str, int], int]:
    palette = rng.sample(list(range(1, 10)), 6)
    top, bottom, left, right, indicator, interior = palette
    patch = np.full((height, width), int(interior), dtype=int)

    patch[0, :] = int(top)
    patch[1:, 0] = int(left)
    patch[:-1, -1] = int(right)
    patch[-1, 1:] = int(bottom)

    interior_choices = [top, bottom, left, right, interior]
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            patch[r, c] = int(rng.choice(interior_choices))

    return patch, {"up": top, "down": bottom, "left": left, "right": right}, indicator


def _random_walk(length: int, rng: random.Random) -> tuple[list[str], list[tuple[int, int]]]:
    for _ in range(300):
        seq: list[str] = []
        visited = [(0, 0)]
        used = {(0, 0)}
        cur = (0, 0)
        for _step in range(length):
            candidates = []
            for direction, (dr, dc) in DIRS.items():
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in used:
                    candidates.append((direction, nxt))
            if not candidates:
                break
            direction, nxt = rng.choice(candidates)
            seq.append(direction)
            visited.append(nxt)
            used.add(nxt)
            cur = nxt
        if len(seq) != length:
            continue
        used_dirs = set(seq)
        if not (used_dirs & {"up", "down"}):
            continue
        if not (used_dirs & {"left", "right"}):
            continue
        return seq, visited
    raise RuntimeError("failed to build a self-avoiding walk")


def _place_bar(
    grid: np.ndarray, colors: list[int], rng: random.Random
) -> None:
    h, w = grid.shape
    length = len(colors)
    span = 2 * length - 1
    horizontal = rng.choice([True, False])
    if horizontal:
        row = rng.choice([0, h - 1])
        start_col = rng.choice([0, w - span])
        for idx, color in enumerate(colors):
            grid[row, start_col + 2 * idx] = int(color)
    else:
        col = rng.choice([0, w - 1])
        start_row = rng.choice([0, h - span])
        for idx, color in enumerate(colors):
            grid[start_row + 2 * idx, col] = int(color)


def _indicator_positions(
    top: int, left: int, height: int, width: int, directions: set[str], rng: random.Random
) -> dict[str, tuple[int, int]]:
    pos: dict[str, tuple[int, int]] = {}
    if "up" in directions:
        pos["up"] = (top - 2, left + rng.randrange(width))
    if "down" in directions:
        pos["down"] = (top + height + 1, left + rng.randrange(width))
    if "left" in directions:
        pos["left"] = (top + rng.randrange(height), left - 2)
    if "right" in directions:
        pos["right"] = (top + rng.randrange(height), left + width + 1)
    return pos


def _validate_unique_multicolor(grid: np.ndarray) -> None:
    comps = _components(grid != BG, connectivity=8)
    multicolor = 0
    for comp in comps:
        if len({int(grid[r, c]) for r, c in comp}) > 1:
            multicolor += 1
    assert multicolor == 1, "seed must be the only multicolor connected object"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    seed_h = rng.choice([3, 4])
    seed_w = rng.choice([3, 4])
    steps = rng.randint(3, 5)
    seq_dirs, visited = _random_walk(steps, rng)

    min_ru = min(r for r, _ in visited)
    max_ru = max(r for r, _ in visited)
    min_cu = min(c for _, c in visited)
    max_cu = max(c for _, c in visited)

    row_lo = max(3, -min_ru * seed_h)
    col_lo = max(3, -min_cu * seed_w)
    min_h = row_lo + seed_h + max(3, max_ru * seed_h)
    min_w = col_lo + seed_w + max(3, max_cu * seed_w)

    bar_span = 2 * steps - 1
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    if max(grid_h, grid_w) > 30:
        return None
    if grid_h < bar_span and grid_w < bar_span:
        return None

    row_hi = min(grid_h - seed_h - 3, grid_h - seed_h - max_ru * seed_h)
    col_hi = min(grid_w - seed_w - 3, grid_w - seed_w - max_cu * seed_w)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    top = rng.randint(row_lo, row_hi)
    left = rng.randint(col_lo, col_hi)
    patch, side_colors, indicator_color = _make_seed_patch(seed_h, seed_w, rng)
    active_dirs = set(seq_dirs)
    indicator_pos = _indicator_positions(top, left, seed_h, seed_w, active_dirs, rng)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _paste_transparent(grid, patch, top, left)
    for rr, cc in indicator_pos.values():
        if grid[rr, cc] != BG:
            return None
        grid[rr, cc] = int(indicator_color)

    bar_colors = [side_colors[direction] for direction in seq_dirs]
    if grid_w >= bar_span and grid_h >= bar_span:
        if rng.choice([True, False]):
            _place_bar(grid, bar_colors, rng)
        else:
            _place_bar(grid, bar_colors, rng)
    elif grid_w >= bar_span:
        row = rng.choice([0, grid_h - 1])
        start_col = rng.choice([0, grid_w - bar_span])
        for idx, color in enumerate(bar_colors):
            grid[row, start_col + 2 * idx] = int(color)
    else:
        col = rng.choice([0, grid_w - 1])
        start_row = rng.choice([0, grid_h - bar_span])
        for idx, color in enumerate(bar_colors):
            grid[start_row + 2 * idx, col] = int(color)

    expected = _render_chain(grid.shape, patch, (top, left), seq_dirs, BG)
    if np.any(grid != BG):
        try:
            actual = main(grid)
        except Exception:
            return None
        if not np.array_equal(actual, expected):
            return None

    try:
        _validate_unique_multicolor(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

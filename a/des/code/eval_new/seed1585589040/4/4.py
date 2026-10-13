import math
import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _ring_area(side: int) -> int:
    return 4 * side - 4


def _draw_square_ring(grid: np.ndarray, side: int, color: int) -> None:
    h, w = grid.shape
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_components = _components(grid == GREY)
    assert grey_components, "expected a grey legend component"
    best = max(
        grey_components,
        key=lambda coords: (_bbox(coords)[2] + _bbox(coords)[3], coords.shape[0]),
    )
    return _bbox(best)


def _legend_order(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    entries: list[tuple[int, int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(grid[r, c])
            if color in (BG, GREY):
                continue
            depth = min(r - r0, c - c0, r1 - r, c1 - c)
            entries.append((depth, r, c, color))
    entries.sort()
    assert entries, "legend must contain colored depth markers"
    order: list[int] = []
    seen_depths: set[int] = set()
    for depth, _r, _c, color in entries:
        if depth not in seen_depths:
            order.append(int(color))
            seen_depths.add(int(depth))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild centered concentric square frames from scattered fragment areas and a lower-right depth legend.

    Input structure:
    - The background is black, and each non-grey color appears as several disjoint 4-connected fragments scattered across the grid.
    - A small grey rectangle in the lower-right acts as a legend: it contains one colored pixel at each nested depth from the outer border toward the center.
    - Each fragment color also appears once inside that legend, and the total fragment area for each color is distinct.

    Rule:
    - Locate the grey legend rectangle and read its colored pixels from outermost depth to innermost depth.
    - Ignore the legend itself, then sum the scattered fragment pixels for each legend color.
    - Interpret each total as the perimeter-cell count of a 1-pixel-thick square frame, and draw those square frames concentrically at the center of a blank grid using the legend order for outer-to-inner coloring.

    Key insight:
    - The small grey box is a depth legend rather than an object to keep; the true output is determined by combining that color order with fragment-area-to-frame-size reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    legend = _legend_bbox(g)
    order = _legend_order(g, legend)
    lr0, lc0, lr1, lc1 = legend
    outside_legend = np.ones((h, w), dtype=bool)
    outside_legend[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = np.zeros_like(g)
    for color in order:
        area = int(np.sum((g == int(color)) & outside_legend))
        assert area >= 8 and area % 4 == 0, "fragment area must encode a hollow square frame"
        side = area // 4 + 1
        assert side <= min(h, w), "decoded frame does not fit in the grid"
        _draw_square_ring(out, side, int(color))
    return out


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, total), parts - 1))
        values = [cuts[0], *[b - a for a, b in zip(cuts, cuts[1:])], total - cuts[-1]]
        if max(values) <= max(10, math.ceil(total * 0.55)):
            return values
    return values


def _grow_blob(area: int, rng: random.Random) -> np.ndarray:
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    base = max(2, int(round(math.sqrt(area))))
    for _ in range(300):
        h = rng.randint(max(1, base - 2), max(2, base + 2))
        w_min = max(1, math.ceil(area / h))
        w = rng.randint(w_min, max(w_min, base + 3))
        if h * w < area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        frontier: set[tuple[int, int]] = set()

        def add_frontier(rr: int, cc: int) -> None:
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(*seed)
        count = 1
        while count < area and frontier:
            cells = list(frontier)
            weights = []
            for rr, cc in cells:
                adj = 0
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        adj += 1
                weights.append(adj * adj + 1)
            rr, cc = rng.choices(cells, weights=weights, k=1)[0]
            frontier.remove((rr, cc))
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            count += 1
            add_frontier(rr, cc)

        if count != area:
            continue

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return mask[r0 : r1 + 1, c0 : c1 + 1]

    return np.ones((1, area), dtype=bool)


def _place_mask(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(gh - mh + 1):
        for c0 in range(gw - mw + 1):
            if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                continue
            positions.append((r0, c0))
    if not positions:
        return False
    r0, c0 = rng.choice(positions)
    region = grid[r0 : r0 + mh, c0 : c0 + mw]
    region[mask] = int(color)
    placed = np.zeros_like(grid, dtype=bool)
    placed[r0 : r0 + mh, c0 : c0 + mw] = mask
    blocked |= _dilate8(placed)
    return True


def _legend_positions(side: int, count: int, rng: random.Random) -> Iterable[tuple[int, int]]:
    for depth in range(count):
        low = depth
        high = side - 1 - depth
        if low == high:
            yield low, low
            continue
        yield rng.choice(((low, low), (low, high), (high, low), (high, high)))


def _expected_output(shape: tuple[int, int], sides: list[int], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for side, color in zip(sides, colors):
        _draw_square_ring(out, int(side), int(color))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(2, 5)
    legend_side = 2 * layer_count - 1

    min_outer = 2 * layer_count + 1
    max_outer = min(23, 30 - (rng.randint(0, 2)))
    candidates = [side for side in range(min_outer, max_outer + 1, 2)]
    if not candidates:
        return None
    outer_side = rng.choice(candidates)
    sides = [outer_side - 2 * i for i in range(layer_count)]
    if sides[-1] < 3:
        return None

    areas = [_ring_area(side) for side in sides]
    total_fragment_area = sum(areas)

    colors = rng.sample(list(COLOR_POOL), layer_count)

    min_dim = max(legend_side + 2, outer_side, 16)
    for _ in range(50):
        h = rng.randint(min_dim, 30)
        w = rng.randint(min_dim, 30)
        usable = h * w - legend_side * legend_side
        if usable >= total_fragment_area * 3:
            break
    else:
        return None

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    margin_bottom = rng.randint(0, min(2, h - legend_side))
    margin_right = rng.randint(0, min(2, w - legend_side))
    lr0 = h - legend_side - margin_bottom
    lc0 = w - legend_side - margin_right
    grid[lr0 : lr0 + legend_side, lc0 : lc0 + legend_side] = GREY
    for color, (dr, dc) in zip(colors, _legend_positions(legend_side, layer_count, rng)):
        grid[lr0 + dr, lc0 + dc] = int(color)
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[lr0 : lr0 + legend_side, lc0 : lc0 + legend_side] = True
    blocked |= _dilate8(legend_mask)

    pieces: list[tuple[int, np.ndarray]] = []
    for color, area in zip(colors, areas):
        min_parts = 2 if area <= 20 else 3 if area <= 44 else 4 if area <= 68 else 5
        part_count = rng.randint(min_parts, min(6, area))
        for part_area in _random_partition(area, part_count, rng):
            pieces.append((int(color), _grow_blob(int(part_area), rng)))
    rng.shuffle(pieces)
    pieces.sort(key=lambda item: int(np.sum(item[1])), reverse=True)

    for color, mask in pieces:
        if not _place_mask(grid, blocked, mask, color, rng):
            return None

    expected = _expected_output(grid.shape, sides, colors)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
PINK = 6
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _enclosed_bg_count(grid: np.ndarray, comp: dict, bg: int = BG) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((crop == bg) | (crop == color))

    empty = crop == bg
    h, w = crop.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if empty[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if empty[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return int(np.count_nonzero(empty & ~outside))


def _stamp_solid(out: np.ndarray, color: int, top: int, left: int, height: int, width: int) -> None:
    h, w = out.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top) + int(height))
    c1 = min(w, int(left) + int(width))
    if r0 >= r1 or c0 >= c1:
        return
    window = out[r0:r1, c0:c1]
    assert np.all((window == BG) | (window == int(color)))
    window[:, :] = int(color)


def _visible_stamp_counts(rows: int, cols: int, bar_h: int, bar_w: int, seed_r: int, seed_c: int) -> tuple[int, int, int]:
    period = int(bar_h) + int(bar_w)
    limit = rows + cols + period + 2
    chain_a: set[tuple[int, int, int, int, int]] = set()
    chain_b: set[tuple[int, int, int, int, int]] = set()

    def add(target: set[tuple[int, int, int, int, int]], kind: int, top: int, left: int, height: int, width: int) -> None:
        if max(0, int(top)) < min(rows, int(top) + int(height)) and max(0, int(left)) < min(cols, int(left) + int(width)):
            target.add((kind, int(top), int(left), int(height), int(width)))

    for n in range(-limit, limit + 1):
        offset = n * period
        add(chain_a, 0, seed_r + offset, seed_c + offset, bar_h, bar_w)
        add(chain_b, 0, seed_r + offset, seed_c - offset, bar_h, bar_w)
        add(chain_a, 1, seed_r + bar_h + offset, seed_c + bar_w + offset, bar_w, bar_h)
        add(chain_b, 1, seed_r + bar_h + offset, seed_c - bar_h - offset, bar_w, bar_h)

    return len(chain_a | chain_b), len(chain_a), len(chain_b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the two closed shapes' enclosed black areas into a bar size and tile alternating bars along two diagonal zigzag chains.

    Input structure:
    - The grid has a black background and exactly two 4-connected monochrome foreground objects: one green and one pink.
    - Each object is a closed shape whose bounding box contains some enclosed black cells not 4-reachable from the bbox border.
    - The left object's bbox starts in the left half of the canvas and the right object's bbox starts farther to the right.

    Rule:
    - Count, for each object, the black cells inside its bbox that are not 4-connected to the bbox border through black cells.
    - Let the left object's enclosed count be the bar width W and the right object's enclosed count be the bar height H.
    - Erase the input objects, then place one green HxW solid bar with its top-left corner at the left object's bbox top-left.
    - Repeat that bar on both 45-degree diagonals every H+W cells; between adjacent green bars on each diagonal, place a 90-degree rotated pink WxH bar. Clip any bar that crosses the canvas edge.

    Key insight:
    - The visible outlines are only parameter encoders: their enclosed black areas, not their raw outer sizes, determine the geometry and period of the final zigzag tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _find_components(g, bg=BG)
    assert len(comps) == 2

    by_color = {int(comp["color"]): comp for comp in comps}
    assert set(by_color) == {GREEN, PINK}

    ordered = sorted(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    left_comp, right_comp = ordered
    assert int(left_comp["color"]) == GREEN and int(right_comp["color"]) == PINK
    width = _enclosed_bg_count(g, left_comp, BG)
    height = _enclosed_bg_count(g, right_comp, BG)
    assert 1 <= width <= 30 and 1 <= height <= 30 and width != height

    seed_r, seed_c = left_comp["bbox"][:2]
    period = height + width
    out = np.zeros_like(g)
    limit = g.shape[0] + g.shape[1] + period + 2

    for n in range(-limit, limit + 1):
        offset = n * period
        _stamp_solid(out, GREEN, seed_r + offset, seed_c + offset, height, width)
        _stamp_solid(out, GREEN, seed_r + offset, seed_c - offset, height, width)
        _stamp_solid(out, PINK, seed_r + height + offset, seed_c + width + offset, width, height)
        _stamp_solid(out, PINK, seed_r + height + offset, seed_c - height - offset, width, height)

    return out


GRID_MIN = 18
GRID_MAX = 28
COUNT_OPTIONS = [2, 3, 4, 5]
MIN_COMPONENT_AREA = 10
MAX_COMPONENT_AREA = 20
MIN_BBOX_GAP = 3
VISIBLE_TOTAL_RANGE = (10, 18)
VISIBLE_CHAIN_MIN = 4
OUTPUT_DENSITY_RANGE = (0.15, 0.48)
GEN_ATTEMPTS = 400


def _connected_subset(height: int, width: int, size: int) -> set[tuple[int, int]]:
    seed = (random.randrange(height), random.randrange(width))
    chosen = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in DIRS4
        if 0 <= seed[0] + dr < height and 0 <= seed[1] + dc < width
    }
    while len(chosen) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                frontier.add((nr, nc))
    return chosen


def _patch_area(patch: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(patch == int(color)))


def _remove_random_corners(patch: np.ndarray, color: int, target_count: int) -> np.ndarray:
    h, w = patch.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    order = corners[:]
    random.shuffle(order)
    best = patch
    for k in (1, 2):
        candidate = best.copy()
        changed = False
        for r, c in order[:k]:
            if candidate[r, c] == color:
                candidate[r, c] = BG
                changed = True
        if not changed:
            continue
        comps = _find_components(candidate, bg=BG)
        if len(comps) == 1 and int(comps[0]["color"]) == int(color) and _enclosed_bg_count(candidate, comps[0], BG) == int(target_count):
            best = candidate
    return best


def _make_frame_patch(count: int, color: int) -> tuple[np.ndarray, str] | None:
    if random.random() < 0.5:
        h, w = 3, count + 2
    else:
        h, w = count + 2, 3
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch = _remove_random_corners(patch, color, count)
    comps = _find_components(patch, bg=BG)
    if len(comps) != 1 or _enclosed_bg_count(patch, comps[0], BG) != count:
        return None
    return patch, "frame"


def _make_irregular_patch(count: int, color: int) -> tuple[np.ndarray, str] | None:
    candidates = []
    for h in range(4, 7):
        for w in range(4, 7):
            interior = (h - 2) * (w - 2)
            if interior < count:
                continue
            area = 2 * h + 2 * w - 4 + (interior - count)
            if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
                candidates.append((h, w))
    if not candidates:
        return None

    h, w = random.choice(candidates)
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    hole_cells = _connected_subset(h - 2, w - 2, count)
    for rr in range(h - 2):
        for cc in range(w - 2):
            if (rr, cc) not in hole_cells:
                patch[rr + 1, cc + 1] = color

    if random.random() < 0.65:
        patch = _remove_random_corners(patch, color, count)

    comps = _find_components(patch, bg=BG)
    if len(comps) != 1 or int(comps[0]["color"]) != int(color):
        return None
    if _patch_area(patch, color) < MIN_COMPONENT_AREA or _patch_area(patch, color) > MAX_COMPONENT_AREA:
        return None
    if _enclosed_bg_count(patch, comps[0], BG) != count:
        return None
    if patch.shape[0] == 3 or patch.shape[1] == 3:
        return None
    return patch, "irregular"


def _make_object_patch(count: int, color: int) -> tuple[np.ndarray, str]:
    for _ in range(80):
        maker = _make_irregular_patch if random.random() < 0.75 else _make_frame_patch
        result = maker(count, color)
        if result is not None:
            return result
    raise RuntimeError("failed to build closed object patch")


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_BBOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _grid_bounds_for_period(period: int) -> tuple[int, int]:
    low = max(GRID_MIN, 3 * period)
    high = min(GRID_MAX, 5 * period)
    if low > high:
        low = high = min(GRID_MAX, max(GRID_MIN, 4 * period))
    return low, high


def _sample_positions(rows: int, cols: int, left_patch: np.ndarray, right_patch: np.ndarray, bar_h: int, bar_w: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    lh, lw = left_patch.shape
    rh, rw = right_patch.shape
    midpoint = cols // 2

    left_positions: list[tuple[int, int]] = []
    for top in range(rows - lh + 1):
        for left in range(cols - lw + 1):
            if left > midpoint - lw - 1:
                continue
            total, chain_a, chain_b = _visible_stamp_counts(rows, cols, bar_h, bar_w, top, left)
            if VISIBLE_TOTAL_RANGE[0] <= total <= VISIBLE_TOTAL_RANGE[1] and chain_a >= VISIBLE_CHAIN_MIN and chain_b >= VISIBLE_CHAIN_MIN:
                left_positions.append((top, left))
    random.shuffle(left_positions)

    for left_top, left_left in left_positions:
        left_box = (left_top, left_left, left_top + lh - 1, left_left + lw - 1)
        right_positions = []
        for top in range(rows - rh + 1):
            for left in range(cols - rw + 1):
                if left < midpoint:
                    continue
                right_box = (top, left, top + rh - 1, left + rw - 1)
                if left_left >= left:
                    continue
                if _boxes_separated(left_box, right_box):
                    right_positions.append((top, left))
        if right_positions:
            return (left_top, left_left), random.choice(right_positions)
    return None


def _validate_input(grid: np.ndarray, expected_w: int, expected_h: int) -> bool:
    comps = sorted(_find_components(grid, bg=BG), key=lambda comp: comp["bbox"][1])
    if len(comps) != 2:
        return False
    if [int(comp["color"]) for comp in comps] != [GREEN, PINK]:
        return False
    if _enclosed_bg_count(grid, comps[0], BG) != int(expected_w):
        return False
    if _enclosed_bg_count(grid, comps[1], BG) != int(expected_h):
        return False
    if expected_w == expected_h:
        return False
    return True


def _validate_output(inp: np.ndarray, out: np.ndarray, expected_w: int, expected_h: int) -> bool:
    if inp.shape != out.shape or np.array_equal(inp, out):
        return False
    colors = set(map(int, np.unique(out)))
    if not {BG, GREEN, PINK}.issuperset(colors) or GREEN not in colors or PINK not in colors:
        return False
    left_comp = sorted(_find_components(inp, bg=BG), key=lambda comp: comp["bbox"][1])[0]
    total, chain_a, chain_b = _visible_stamp_counts(inp.shape[0], inp.shape[1], expected_h, expected_w, left_comp["bbox"][0], left_comp["bbox"][1])
    if not (VISIBLE_TOTAL_RANGE[0] <= total <= VISIBLE_TOTAL_RANGE[1] and chain_a >= VISIBLE_CHAIN_MIN and chain_b >= VISIBLE_CHAIN_MIN):
        return False
    density = float(np.count_nonzero(out)) / float(out.size)
    return OUTPUT_DENSITY_RANGE[0] <= density <= OUTPUT_DENSITY_RANGE[1]


def _gen_attempt() -> np.ndarray | None:
    width, height = random.sample(COUNT_OPTIONS, 2)
    period = width + height
    low, high = _grid_bounds_for_period(period)
    rows = random.randint(low, high)
    cols = random.randint(low, high)

    left_patch, left_style = _make_object_patch(width, GREEN)
    right_patch, right_style = _make_object_patch(height, PINK)
    if left_style == right_style == "frame":
        return None

    positions = _sample_positions(rows, cols, left_patch, right_patch, height, width)
    if positions is None:
        return None
    (left_top, left_left), (right_top, right_left) = positions

    grid = np.zeros((rows, cols), dtype=int)
    _place_patch(grid, left_patch, left_top, left_left)
    _place_patch(grid, right_patch, right_top, right_left)

    if not _validate_input(grid, width, height):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if not _validate_output(grid, out, width, height):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

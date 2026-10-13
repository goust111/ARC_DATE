import random

import numpy as np


_BG = 0
_SIDES = ("top", "bottom", "left", "right")
_GEN_ATTEMPTS = 300
_PLACE_ATTEMPTS = 200
_OBJECT_COUNT_RANGE = (2, 5)
_RECT_MIN = 3
_RECT_MAX_SMALL = 6
_RECT_MAX_LARGE = 7
_GRID_MAX = 30


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_soil_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        if not comps or not any(len(comp) > 1 for comp in comps):
            continue
        if all(_is_solid_rectangle(comp) for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique soil color"
    return candidates[0]


def _seed_for_component(
    grid: np.ndarray,
    bg: int,
    soil_color: int,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, str]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    seeds: list[tuple[int, int, str]] = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(grid[r0 - 1, c]) not in (bg, soil_color):
                seeds.append((r0 - 1, c, "top"))
    if r1 < h - 1:
        for c in range(c0, c1 + 1):
            if int(grid[r1 + 1, c]) not in (bg, soil_color):
                seeds.append((r1 + 1, c, "bottom"))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) not in (bg, soil_color):
                seeds.append((r, c0 - 1, "left"))
    if c1 < w - 1:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) not in (bg, soil_color):
                seeds.append((r, c1 + 1, "right"))

    assert len(seeds) == 1, "each soil rectangle must have exactly one adjacent seed"
    return seeds[0]


def _grow(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int, str],
) -> None:
    r0, c0, r1, c1 = bbox
    sr, sc, side = seed
    color = int(out[sr, sc])

    if side == "top":
        stem = [(r, sc) for r in range(r0, r1 + 1)]
        branch_dirs = ((0, -1), (0, 1))
    elif side == "bottom":
        stem = [(r, sc) for r in range(r1, r0 - 1, -1)]
        branch_dirs = ((0, -1), (0, 1))
    elif side == "left":
        stem = [(sr, c) for c in range(c0, c1 + 1)]
        branch_dirs = ((-1, 0), (1, 0))
    else:
        stem = [(sr, c) for c in range(c1, c0 - 1, -1)]
        branch_dirs = ((-1, 0), (1, 0))

    for idx, (r, c) in enumerate(stem):
        out[r, c] = color
        if idx % 4 == 0:
            dr, dc = branch_dirs[0]
        elif idx % 4 == 2:
            dr, dc = branch_dirs[1]
        else:
            continue
        br, bc = r + dr, c + dc
        if r0 <= br <= r1 and c0 <= bc <= c1:
            out[br, bc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed through its adjacent soil rectangle as a stem with alternating side branches.

    Input structure:
    - The background is black.
    - One shared soil color forms several disjoint solid rectangles.
    - Each soil rectangle has exactly one singleton seed pixel of a different color touching one outer side.

    Rule:
    - Identify the soil rectangles and determine which side of each rectangle its seed touches.
    - From the touched entry column or row, draw a same-colored stem straight through the soil to the opposite side.
    - Along the stem, the 1st, 5th, 9th... stem cells add one branch on the grid-left side for vertical stems or grid-up side for horizontal stems; the 3rd, 7th, 11th... stem cells add one branch on the opposite side; even-numbered stem cells do not branch.
    - Keep branches only when they stay inside the soil rectangle, preserve the seed pixel itself, and leave all uncovered soil cells unchanged.

    Key insight:
    - The seed supplies both the paint color and the entry side, so the task is to pair each singleton with its soil rectangle and then apply the alternating branch pattern along the inward stem.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _BG if np.any(g == _BG) else int(np.bincount(g.ravel()).argmax())
    soil_color = _find_soil_color(g, bg)

    soil_components = _components(g == soil_color)
    out = g.copy()
    seen_seeds: set[tuple[int, int]] = set()
    for comp in soil_components:
        assert _is_solid_rectangle(comp), "soil components must be solid rectangles"
        bbox = _bbox(comp)
        seed = _seed_for_component(g, bg, soil_color, bbox)
        seen_seeds.add((seed[0], seed[1]))
        _grow(out, bbox, seed)

    all_seeds = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != soil_color))
    }
    assert seen_seeds == all_seeds, "every non-soil foreground pixel must be a seed"
    return out


def _sample_sides(count: int) -> list[str]:
    if count <= len(_SIDES):
        return random.sample(list(_SIDES), count)
    sides = list(_SIDES)
    while len(sides) < count:
        sides.append(random.choice(_SIDES))
    random.shuffle(sides)
    return sides


def _choose_anchor(lo: int, hi: int) -> int:
    if hi - lo >= 2 and random.random() < 0.7:
        return random.randint(lo + 1, hi - 1)
    return random.randint(lo, hi)


def _place_seed(
    side: str,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1, _choose_anchor(c0, c1)
    if side == "bottom":
        return r1 + 1, _choose_anchor(c0, c1)
    if side == "left":
        return _choose_anchor(r0, r1), c0 - 1
    return _choose_anchor(r0, r1), c1 + 1


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*_OBJECT_COUNT_RANGE)
    min_dim = max(12, 10 + 2 * object_count)
    h = random.randint(min_dim, _GRID_MAX)
    w = random.randint(min_dim, _GRID_MAX)
    grid = np.full((h, w), _BG, dtype=int)

    soil_color = random.choice([c for c in range(1, 10)])
    seed_colors = random.sample([c for c in range(1, 10) if c != soil_color], object_count)
    sides = _sample_sides(object_count)
    reserved = np.zeros((h, w), dtype=bool)
    total_soil = 0

    rect_cap = _RECT_MAX_LARGE if object_count <= 3 else _RECT_MAX_SMALL
    for idx in range(object_count):
        side = sides[idx]
        placed = False
        for _ in range(_PLACE_ATTEMPTS):
            rect_h = random.randint(_RECT_MIN, min(rect_cap, h - 2))
            rect_w = random.randint(_RECT_MIN, min(rect_cap, w - 2))
            if h - rect_h - 1 < 1 or w - rect_w - 1 < 1:
                continue
            r0 = random.randint(1, h - rect_h - 1)
            c0 = random.randint(1, w - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            rr0, cc0 = r0 - 1, c0 - 1
            rr1, cc1 = r1 + 1, c1 + 1
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            seed_r, seed_c = _place_seed(side, (r0, c0, r1, c1))
            if grid[seed_r, seed_c] != _BG:
                continue

            grid[r0 : r1 + 1, c0 : c1 + 1] = soil_color
            grid[seed_r, seed_c] = seed_colors[idx]
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            total_soil += rect_h * rect_w
            placed = True
            break
        if not placed:
            return None

    if total_soil >= (h * w) // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-growth ARC input")

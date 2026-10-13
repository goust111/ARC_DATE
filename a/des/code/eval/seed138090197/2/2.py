from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("U", "D", "L", "R")
ORTHOGONAL_PAIRS = (("U", "L"), ("U", "R"), ("D", "L"), ("D", "R"))
OPPOSITE_PAIRS = (("U", "D"), ("L", "R"))
CATEGORIES = ("plus", "tee", "corner", "corridor")
GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_roles(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[tuple[int, int], int, int]:
    colors = Counter(int(grid[r, c]) for r, c in comp)
    assert len(colors) == 2, "each non-background component must contain one soil color and one seed color"
    singleton_colors = [color for color, count in colors.items() if count == 1]
    assert len(singleton_colors) == 1, "expected exactly one singleton seed color in each component"
    seed_color = singleton_colors[0]
    soil_colors = [color for color in colors if color != seed_color]
    assert len(soil_colors) == 1, "expected exactly one soil color in each component"
    soil_color = soil_colors[0]
    assert colors[soil_color] >= 2, "soil region must contain more than one cell"
    seed_pos = next((r, c) for r, c in comp if int(grid[r, c]) == seed_color)
    return seed_pos, seed_color, soil_color


def _arm_lengths(grid: np.ndarray, seed: tuple[int, int], soil_color: int) -> dict[str, int]:
    h, w = grid.shape
    sr, sc = seed
    lengths: dict[str, int] = {}
    for name, (dr, dc) in zip(DIR_NAMES, DIRS4):
        steps = 0
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == soil_color:
            steps += 1
            r += dr
            c += dc
        lengths[name] = steps
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components4(g != BG):
        seed, seed_color, soil_color = _component_roles(g, comp)
        sr, sc = seed
        out[sr, sc] = seed_color
        for dr, dc in DIRS4:
            r, c = sr + dr, sc + dc
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == soil_color:
                out[r, c] = seed_color
                r += dr
                c += dc
    return out


def _sample_positive_lengths(rng: np.random.Generator, count: int) -> list[int]:
    vals = [int(rng.integers(1, 6)) for _ in range(count)]
    if count >= 2 and len(set(vals)) == 1:
        idx = int(rng.integers(0, count))
        vals[idx] = 1 + (vals[idx] % 5)
    return vals


def _sample_lengths(
    rng: np.random.Generator, category: str
) -> dict[str, int]:
    lengths = {name: 0 for name in DIR_NAMES}
    if category == "plus":
        active = list(DIR_NAMES)
    elif category == "tee":
        missing = DIR_NAMES[int(rng.integers(0, len(DIR_NAMES)))]
        active = [name for name in DIR_NAMES if name != missing]
    elif category == "corner":
        active = list(ORTHOGONAL_PAIRS[int(rng.integers(0, len(ORTHOGONAL_PAIRS)))])
    elif category == "corridor":
        active = list(OPPOSITE_PAIRS[int(rng.integers(0, len(OPPOSITE_PAIRS)))])
    else:
        raise ValueError(f"unknown category: {category}")

    for name, value in zip(active, _sample_positive_lengths(rng, len(active))):
        lengths[name] = value

    if sum(lengths.values()) < 3:
        active_name = active[int(rng.integers(0, len(active)))]
        lengths[active_name] += 1
    return lengths


def _weighted_choice(rng: np.random.Generator, items: list[tuple[int, int]], weights: list[float]) -> tuple[int, int]:
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    idx = int(rng.choice(len(items), p=probs))
    return items[idx]


def _build_region(
    rng: np.random.Generator, category: str
) -> tuple[set[tuple[int, int]], tuple[int, int], dict[str, int]]:
    lengths = _sample_lengths(rng, category)
    seed = (0, 0)
    soil: set[tuple[int, int]] = set()
    for k in range(1, lengths["U"] + 1):
        soil.add((-k, 0))
    for k in range(1, lengths["D"] + 1):
        soil.add((k, 0))
    for k in range(1, lengths["L"] + 1):
        soil.add((0, -k))
    for k in range(1, lengths["R"] + 1):
        soil.add((0, k))

    min_r = -lengths["U"] - int(rng.integers(0, 3))
    max_r = lengths["D"] + int(rng.integers(0, 3))
    min_c = -lengths["L"] - int(rng.integers(0, 3))
    max_c = lengths["R"] + int(rng.integers(0, 3))

    def allowed(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (min_r <= r <= max_r and min_c <= c <= max_c):
            return False
        if cell == seed or cell in soil:
            return False
        if r == 0 and (c < -lengths["L"] or c > lengths["R"]):
            return False
        if c == 0 and (r < -lengths["U"] or r > lengths["D"]):
            return False
        return True

    bbox_cells = [
        (r, c)
        for r in range(min_r, max_r + 1)
        for c in range(min_c, max_c + 1)
        if allowed((r, c))
    ]
    target_extra = 0
    if bbox_cells:
        lo = 1 if rng.random() < 0.8 else 0
        hi = min(len(bbox_cells), max(3, len(soil) + 2))
        target_extra = int(rng.integers(lo, hi + 1))

    occupied = set(soil)
    occupied.add(seed)
    for _ in range(target_extra):
        candidates = set()
        for r, c in occupied:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if allowed(cand):
                    candidates.add(cand)
        if not candidates:
            break
        cand_list = sorted(candidates)
        weights = []
        for r, c in cand_list:
            neighbor_count = sum((r + dr, c + dc) in occupied for dr, dc in DIRS4)
            off_axis_bonus = 2 if r != 0 and c != 0 else 0
            interior_bonus = 1 if min_r < r < max_r and min_c < c < max_c else 0
            weights.append(1 + 3 * neighbor_count + off_axis_bonus + interior_bonus)
        chosen = _weighted_choice(rng, cand_list, weights)
        soil.add(chosen)
        occupied.add(chosen)

    if bbox_cells and not any(r != 0 and c != 0 for r, c in soil):
        candidates = set()
        for r, c in occupied:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if allowed(cand) and cand[0] != 0 and cand[1] != 0:
                    candidates.add(cand)
        if candidates:
            chosen = sorted(candidates)[int(rng.integers(0, len(candidates)))]
            soil.add(chosen)

    return soil, seed, lengths


def _translate(
    cells: set[tuple[int, int]], seed: tuple[int, int], offset: tuple[int, int]
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    dr, dc = offset
    translated_soil = {(r + dr, c + dc) for r, c in cells}
    seed_pos = (seed[0] + dr, seed[1] + dc)
    return translated_soil, seed_pos


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _category_mix(rng: np.random.Generator, n_regions: int) -> list[str]:
    if n_regions == 2:
        cats = [
            ("plus", "tee")[int(rng.integers(0, 2))],
            ("corner", "corridor")[int(rng.integers(0, 2))],
        ]
    elif n_regions == 3:
        cats = [
            ("plus", "tee")[int(rng.integers(0, 2))],
            ("corner", "corridor")[int(rng.integers(0, 2))],
            CATEGORIES[int(rng.integers(0, len(CATEGORIES)))],
        ]
    else:
        cats = list(CATEGORIES)
    order = rng.permutation(len(cats))
    return [cats[int(i)] for i in order]


def _valid_input(grid: np.ndarray) -> bool:
    comps = _components4(grid != BG)
    if not (2 <= len(comps) <= 4):
        return False

    saw_zero_arm = False
    saw_full_cross = False
    saw_asymmetry = False

    for comp in comps:
        try:
            seed, _seed_color, soil_color = _component_roles(grid, comp)
        except AssertionError:
            return False
        lengths = _arm_lengths(grid, seed, soil_color)
        vals = list(lengths.values())
        if sum(vals) == 0:
            return False
        if 0 in vals:
            saw_zero_arm = True
        if all(v > 0 for v in vals):
            saw_full_cross = True
        if len(set(vals)) > 1:
            saw_asymmetry = True

    return saw_asymmetry and (saw_zero_arm or saw_full_cross)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_regions = int(rng.integers(2, 5))
    h = int(rng.integers(14, 25))
    w = int(rng.integers(14, 25))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    palette = list(rng.permutation(np.arange(1, 10)))
    needed_colors = 2 * n_regions
    if needed_colors > len(palette):
        return None
    colors = palette[:needed_colors]
    soil_colors = colors[:n_regions]
    seed_colors = colors[n_regions:]
    categories = _category_mix(rng, n_regions)

    for idx in range(n_regions):
        soil, seed, _lengths = _build_region(rng, categories[idx])
        local_cells = set(soil)
        local_cells.add(seed)
        rows = [r for r, _ in local_cells]
        cols = [c for _, c in local_cells]
        min_lr, max_lr = min(rows), max(rows)
        min_lc, max_lc = min(cols), max(cols)

        row_lo, row_hi = -min_lr, h - 1 - max_lr
        col_lo, col_hi = -min_lc, w - 1 - max_lc
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(120):
            seed_r = int(rng.integers(row_lo, row_hi + 1))
            seed_c = int(rng.integers(col_lo, col_hi + 1))
            translated_soil, seed_pos = _translate(soil, seed, (seed_r, seed_c))
            region_cells = set(translated_soil)
            region_cells.add(seed_pos)
            if any(blocked[r, c] for r, c in region_cells):
                continue

            for r, c in translated_soil:
                grid[r, c] = soil_colors[idx]
            grid[seed_pos] = seed_colors[idx]
            _mark_blocked(blocked, region_cells)
            placed = True
            break

        if not placed:
            return None

    if not _valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

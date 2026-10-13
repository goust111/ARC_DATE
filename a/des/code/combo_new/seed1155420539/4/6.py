import random
from collections import Counter

import numpy as np


BG = 0
MAX_COLOR = 10
SEED_COLORS = tuple(color for color in range(MAX_COLOR) if color not in {0, 5})
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_SIZE_BY_COUNT = {
    1: (10, 22),
    2: (14, 26),
    3: (18, 30),
    4: (22, 30),
}
OUTER_H_RANGE = (4, 8)
OUTER_W_RANGE = (4, 8)
MAX_SEEDS = 6
PLACE_TRIES = 100
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_fence(
    grid: np.ndarray, fence_color: int, coords: list[tuple[int, int]], bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    pts = set(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    expected_size = 2 * (h + w) - 5
    if len(coords) != expected_size:
        return None

    gap: tuple[int, int] | None = None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if (r, c) in pts:
                    continue
                if int(grid[r, c]) != bg:
                    return None
                if gap is not None:
                    return None
                gap = (r, c)
            elif int(grid[r, c]) == fence_color:
                return None

    if gap is None:
        return None

    gap_r, gap_c = gap
    side_hits = int(gap_r == r0) + int(gap_r == r1) + int(gap_c == c0) + int(gap_c == c1)
    if side_hits != 1:
        return None

    if gap_r == r0:
        direction = "up"
    elif gap_r == r1:
        direction = "down"
    elif gap_c == c0:
        direction = "left"
    else:
        direction = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != bg) & (interior != fence_color)
    if not bool(seed_mask.any()):
        return None
    seed_colors = np.unique(interior[seed_mask])
    if seed_colors.size != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "seed_color": int(seed_colors[0]),
        "seed_count": int(seed_mask.sum()),
    }


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    candidates: list[list[dict[str, object]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        fences: list[dict[str, object]] = []
        ok = True
        for comp in comps:
            info = _detect_fence(grid, color, comp, bg)
            if info is None:
                ok = False
                break
            fences.append(info)
        if ok and fences:
            candidates.append(fences)

    if len(candidates) != 1:
        raise AssertionError("expected a unique fence color")
    return sorted(candidates[0], key=lambda info: tuple(info["bbox"]))  # type: ignore[index]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one outward ray per gapped fence, using the interior seed count as the ray length.

    Input structure:
    - The background is black.
    - One visible color forms one-pixel-thick rectangular fences; each fence is missing exactly one border pixel on one side.
    - Each fence encloses one or more same-colored seed pixels, while different fences may use different non-fence colors.

    Rule:
    - Detect every rectangular fence and recover its unique gap position on the border.
    - Use the gap side to determine the outward direction and count the enclosed seed pixels to get a length N.
    - Starting from the first cell just outside the gap, draw a straight length-N ray in the seed color; keep the seeds and fence unchanged.

    Key insight:
    - The missing border pixel gives direction, while the number of enclosed seeds gives the ray length.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if bool((g == BG).any()) else int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    out = g.copy()

    for fence in _find_fences(g, bg):
        gap_r, gap_c = fence["gap"]  # type: ignore[misc]
        dr, dc = DIR_TO_DELTA[fence["direction"]]  # type: ignore[index]
        color = int(fence["seed_color"])
        count = int(fence["seed_count"])
        for step in range(1, count + 1):
            rr = int(gap_r) + dr * step
            cc = int(gap_c) + dc * step
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color

    return out


def _sample_directions(count: int, rng: random.Random) -> list[str]:
    dirs = list(DIR_TO_DELTA)
    rng.shuffle(dirs)
    return dirs[:count]


def _sample_grid_shape(count: int, rng: random.Random) -> tuple[int, int]:
    lo, hi = GRID_SIZE_BY_COUNT[count]
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _sample_seed_cells(
    fr0: int, fc0: int, outer_h: int, outer_w: int, count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    parity_groups: list[list[tuple[int, int]]] = [[], []]
    for r in range(fr0 + 1, fr0 + outer_h - 1):
        for c in range(fc0 + 1, fc0 + outer_w - 1):
            parity_groups[(r + c) % 2].append((r, c))
    candidates = [group for group in parity_groups if len(group) >= count]
    if not candidates:
        return None
    group = list(rng.choice(candidates))
    rng.shuffle(group)
    return group[:count]


def _zone_geometry(
    direction: str, outer_h: int, outer_w: int, seed_count: int, zr0: int, zc0: int
) -> tuple[int, int, int, int]:
    if direction == "up":
        return zr0 + seed_count, zc0, outer_h + seed_count, outer_w
    if direction == "down":
        return zr0, zc0, outer_h + seed_count, outer_w
    if direction == "left":
        return zr0, zc0 + seed_count, outer_h, outer_w + seed_count
    return zr0, zc0, outer_h, outer_w + seed_count


def _reserve_ok(occupied: np.ndarray, zr0: int, zc0: int, zh: int, zw: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, zr0 - 1)
    rr1 = min(h, zr0 + zh + 1)
    cc0 = max(0, zc0 - 1)
    cc1 = min(w, zc0 + zw + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve(occupied: np.ndarray, zr0: int, zc0: int, zh: int, zw: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, zr0 - 1)
    rr1 = min(h, zr0 + zh + 1)
    cc0 = max(0, zc0 - 1)
    cc1 = min(w, zc0 + zw + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _build_case(
    grid_shape: tuple[int, int], occupied: np.ndarray, seed_color: int, direction: str, rng: random.Random
) -> dict[str, object] | None:
    h, w = grid_shape
    for _ in range(PLACE_TRIES):
        outer_h = rng.randint(*OUTER_H_RANGE)
        outer_w = rng.randint(*OUTER_W_RANGE)
        inner_area = (outer_h - 2) * (outer_w - 2)
        max_count = min(MAX_SEEDS, (inner_area + 1) // 2)
        if max_count < 1:
            continue
        seed_count = rng.randint(1, max_count)

        zh = outer_h + seed_count if direction in {"up", "down"} else outer_h
        zw = outer_w + seed_count if direction in {"left", "right"} else outer_w
        if zh > h or zw > w:
            continue

        zr0 = rng.randint(0, h - zh)
        zc0 = rng.randint(0, w - zw)
        if not _reserve_ok(occupied, zr0, zc0, zh, zw):
            continue

        fr0, fc0, _zh, _zw = _zone_geometry(direction, outer_h, outer_w, seed_count, zr0, zc0)
        fr1 = fr0 + outer_h - 1
        fc1 = fc0 + outer_w - 1

        if direction in {"up", "down"}:
            gap = (fr0 if direction == "up" else fr1, fc0 + rng.randint(1, outer_w - 2))
        else:
            gap = (fr0 + rng.randint(1, outer_h - 2), fc0 if direction == "left" else fc1)

        seed_cells = _sample_seed_cells(fr0, fc0, outer_h, outer_w, seed_count, rng)
        if seed_cells is None:
            continue

        _reserve(occupied, zr0, zc0, zh, zw)
        return {
            "bbox": (fr0, fc0, fr1, fc1),
            "gap": gap,
            "direction": direction,
            "seed_color": seed_color,
            "seed_count": seed_count,
            "seed_cells": seed_cells,
        }
    return None


def _draw_input(grid: np.ndarray, cases: list[dict[str, object]]) -> None:
    for case in cases:
        r0, c0, r1, c1 = case["bbox"]  # type: ignore[misc]
        gap = tuple(case["gap"])  # type: ignore[arg-type]
        for r in range(int(r0), int(r1) + 1):
            for c in range(int(c0), int(c1) + 1):
                if r in (int(r0), int(r1)) or c in (int(c0), int(c1)):
                    if (r, c) != gap:
                        grid[r, c] = 5
        for r, c in case["seed_cells"]:  # type: ignore[misc]
            grid[int(r), int(c)] = int(case["seed_color"])


def _expected_output(inp: np.ndarray, cases: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for case in cases:
        gap_r, gap_c = case["gap"]  # type: ignore[misc]
        dr, dc = DIR_TO_DELTA[case["direction"]]  # type: ignore[index]
        for step in range(1, int(case["seed_count"]) + 1):
            out[int(gap_r) + dr * step, int(gap_c) + dc * step] = int(case["seed_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(1, 4)
    grid_shape = _sample_grid_shape(count, rng)
    occupied = np.zeros(grid_shape, dtype=bool)
    directions = _sample_directions(count, rng)
    colors = rng.sample(SEED_COLORS, count)

    cases: list[dict[str, object]] = []
    for direction, color in zip(directions, colors):
        case = _build_case(grid_shape, occupied, color, direction, rng)
        if case is None:
            return None
        cases.append(case)

    inp = np.full(grid_shape, BG, dtype=int)
    _draw_input(inp, cases)
    expected = _expected_output(inp, cases)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-ray ARC input")

from __future__ import annotations

import random

import numpy as np


POT_COLOR = 3
SEED_COLOR = 2
BG_CHOICES = (0, 1, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CLOCKWISE = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
MAX_GEN_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _pot_info_from_component(comp: list[tuple[int, int]]) -> dict | None:
    coords = set(comp)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_cells(r0, c0, r1, c1)
    if not coords <= border:
        return None

    gaps = list(border - coords)
    if len(gaps) != 1:
        return None
    gap = gaps[0]

    directions = []
    if gap[0] == r0:
        directions.append((-1, 0))
    if gap[0] == r1:
        directions.append((1, 0))
    if gap[1] == c0:
        directions.append((0, -1))
    if gap[1] == c1:
        directions.append((0, 1))
    if len(directions) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": directions[0],
        "border": border,
    }


def _trace_segment(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []

    for step in range(length):
        if not (0 <= r < h and 0 <= c < w):
            return cells, True
        cells.append((r, c))
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            return cells, True
        if step == length - 1:
            return cells, False
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return cells, True
        r, c = nr, nc

    return cells, False


def _trace_vine(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    first, hit_boundary = _trace_segment(start, direction, length, shape)
    if not first or hit_boundary:
        return first, hit_boundary

    second, hit_boundary_2 = _trace_segment(first[-1], CLOCKWISE[direction], length, shape)
    return first + second[1:], hit_boundary_2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an L-shaped red vine out of each pot gap, with segment length set by the number of seeds inside the pot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    pot_infos_by_color: dict[int, list[dict]] = {}
    for color in colors:
        infos = []
        for comp in _components(g == color):
            info = _pot_info_from_component(comp)
            if info is not None:
                infos.append(info)
        pot_infos_by_color[color] = infos

    pot_color = max(colors, key=lambda color: len(pot_infos_by_color[color]))
    seed_color = next(color for color in colors if color != pot_color)
    pots = pot_infos_by_color[pot_color]
    assert pots, "expected at least one pot"

    out = g.copy()
    covered_seed_pixels = set()

    for pot in pots:
        r0, c0, r1, c1 = pot["bbox"]
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_count = int(np.count_nonzero(inside == seed_color))
        assert 1 <= seed_count <= 6, "each pot must contain 1..6 seeds"

        for rr, cc in np.argwhere(inside == seed_color):
            covered_seed_pixels.add((int(r0 + 1 + rr), int(c0 + 1 + cc)))

        vine_cells, _ = _trace_vine(pot["gap"], pot["direction"], seed_count, g.shape)
        for vr, vc in vine_cells:
            out[vr, vc] = seed_color

    all_seed_pixels = {tuple(map(int, rc)) for rc in np.argwhere(g == seed_color)}
    assert all_seed_pixels == covered_seed_pixels, "all seed pixels must lie inside pots"
    return out


def _sample_gap(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: tuple[int, int],
) -> tuple[int, int]:
    if direction == (-1, 0):
        return (r0, random.randint(c0 + 1, c1 - 1))
    if direction == (1, 0):
        return (r1, random.randint(c0 + 1, c1 - 1))
    if direction == (0, -1):
        return (random.randint(r0 + 1, r1 - 1), c0)
    return (random.randint(r0 + 1, r1 - 1), c1)


def _sample_seed_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    count: int,
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return random.sample(cells, count)


def _expanded_neighborhood(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _try_place_pot(
    grid: np.ndarray,
    bg: int,
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    require_truncation: bool,
) -> bool:
    h, w = grid.shape

    for _ in range(MAX_PLACE_ATTEMPTS):
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h + 2 > h or outer_w + 2 > w:
            continue

        seed_count = random.randint(1, min(6, inner_h * inner_w))
        direction = random.choice(DIRS)

        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        gap = _sample_gap(r0, c0, r1, c1, direction)

        vine_cells, truncated = _trace_vine(gap, direction, seed_count, grid.shape)
        if truncated != require_truncation:
            continue

        border = _border_cells(r0, c0, r1, c1)
        pot_pixels = border - {gap}
        seed_cells = set(_sample_seed_cells(r0, c0, r1, c1, seed_count))
        bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        future_cells = pot_pixels | seed_cells | set(vine_cells)

        if any(cell in blocked for cell in future_cells):
            continue
        if any(cell in reserved for cell in bbox_cells | set(vine_cells)):
            continue

        for r, c in pot_pixels:
            grid[r, c] = POT_COLOR
        for r, c in seed_cells:
            grid[r, c] = SEED_COLOR

        reserved.update(bbox_cells | set(vine_cells))
        blocked.update(_expanded_neighborhood(bbox_cells | set(vine_cells), grid.shape))
        return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(10, 22)
        w = random.randint(10, 22)
        bg = random.choice(BG_CHOICES)
        grid = np.full((h, w), bg, dtype=int)

        max_pots = 2 if h * w < 180 else 3 if h * w < 280 else 4
        target_pots = random.randint(1, max_pots)
        if target_pots == 1:
            truncation_flags = [random.choice([False, True])]
        else:
            truncation_flags = [bool(i % 2) for i in range(target_pots)]
            random.shuffle(truncation_flags)

        blocked: set[tuple[int, int]] = set()
        reserved: set[tuple[int, int]] = set()
        ok = True
        for require_truncation in truncation_flags:
            if not _try_place_pot(grid, bg, blocked, reserved, require_truncation):
                ok = False
                break

        if not ok:
            continue

        if np.count_nonzero(grid == POT_COLOR) == 0:
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")

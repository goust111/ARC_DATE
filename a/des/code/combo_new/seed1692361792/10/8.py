from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SEED_COLORS = (BLUE, RED, GREEN)

CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
EIGHT_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
PERPENDICULARS = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}

MODE_TO_DIRS = {
    "four": CARDINAL_DIRS,
    "diag": DIAGONAL_DIRS,
    "eight": EIGHT_DIRS,
}
MODE_TO_PATTERN = {
    "four": frozenset({(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)}),
    "diag": frozenset({(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)}),
    "eight": frozenset((r, c) for r in range(3) for c in range(3)),
}

GRID_MIN, GRID_MAX = 20, 28
PATCH_MIN, PATCH_MAX = 7, 9
LEGEND_GAPS = (1, 2)
PATCH_RETRIES = 200
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300


def _components(mask: np.ndarray, directions: tuple[tuple[int, int], ...]) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _decode_legend(grid: np.ndarray) -> tuple[dict[int, str], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping: dict[int, str] = {}
    legend_cells: set[tuple[int, int]] = set()

    colors = [int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)]
    for color in colors:
        for comp in _components(g == color, EIGHT_DIRS):
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 2 or c1 - c0 != 2:
                continue
            rel = frozenset((r - r0, c - c0) for r, c in comp)
            for mode, pattern in MODE_TO_PATTERN.items():
                if rel == pattern:
                    mapping[color] = mode
                    legend_cells.update(comp)
                    break

    assert len(mapping) == 3, "legend must define three color->connectivity mappings"
    return mapping, legend_cells


def _step(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
    return (cell[0] + direction[0] * amount, cell[1] + direction[1] * amount)


def _inside_patch(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _axis_room(seed: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = seed
    dr, dc = direction
    if dr == 1:
        return h - 2 - r
    if dr == -1:
        return r - 1
    if dc == 1:
        return w - 2 - c
    return c - 1


def _diag_room(seed: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = seed
    dr, dc = direction
    row_room = h - 2 - r if dr == 1 else r - 1
    col_room = w - 2 - c if dc == 1 else c - 1
    return min(row_room, col_room)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _reachable(cells: set[tuple[int, int]], start: tuple[int, int], mode: str) -> set[tuple[int, int]]:
    dirs = MODE_TO_DIRS[mode]
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _make_region_patch(rng: random.Random, color: int) -> np.ndarray:
    for _ in range(PATCH_RETRIES):
        h = rng.randint(PATCH_MIN, PATCH_MAX)
        w = rng.randint(PATCH_MIN, PATCH_MAX)
        seed = (rng.randint(2, h - 3), rng.randint(2, w - 3))

        card_dir = rng.choice(CARDINAL_DIRS)
        card_room = _axis_room(seed, card_dir, h, w)
        if card_room < 2:
            continue
        card_len = rng.randint(2, min(4, card_room))
        orth_cells = {_step(seed, card_dir, k) for k in range(1, card_len + 1)}
        card_end = _step(seed, card_dir, card_len)
        for perp_dir in rng.sample(list(PERPENDICULARS[card_dir]), k=2):
            if rng.random() < 0.75:
                bud = _step(card_end, perp_dir)
                if _inside_patch(bud, h, w):
                    orth_cells.add(bud)
                    if rng.random() < 0.35:
                        bud2 = _step(bud, perp_dir)
                        if _inside_patch(bud2, h, w):
                            orth_cells.add(bud2)

        diag_dir = rng.choice(DIAGONAL_DIRS)
        diag_room = _diag_room(seed, diag_dir, h, w)
        if diag_room < 2:
            continue
        diag_len = rng.randint(2, min(4, diag_room))
        diag_cells = {_step(seed, diag_dir, k) for k in range(1, diag_len + 1)}
        diag_end = _step(seed, diag_dir, diag_len)
        if rng.random() < 0.5:
            alt_dirs = [d for d in DIAGONAL_DIRS if d not in (diag_dir, (-diag_dir[0], -diag_dir[1]))]
            rng.shuffle(alt_dirs)
            for alt in alt_dirs:
                branch = _step(diag_end, alt)
                if _inside_patch(branch, h, w):
                    diag_cells.add(branch)
                    break

        if orth_cells & diag_cells:
            continue
        if any(not _inside_patch(cell, h, w) for cell in orth_cells | diag_cells):
            continue

        separated = True
        for orth in orth_cells:
            for diag in diag_cells:
                if _chebyshev(orth, diag) <= 1:
                    separated = False
                    break
            if not separated:
                break
        if not separated:
            continue

        passable = {seed} | orth_cells | diag_cells
        reach_four = _reachable(passable, seed, "four")
        reach_diag = _reachable(passable, seed, "diag")
        reach_eight = _reachable(passable, seed, "eight")
        if len(reach_four) <= 1 or len(reach_diag) <= 1:
            continue
        if len(reach_eight) != len(passable):
            continue
        if len(reach_eight) <= max(len(reach_four), len(reach_diag)):
            continue

        patch = np.full((h, w), GREY, dtype=int)
        for r, c in passable - {seed}:
            patch[r, c] = BLACK
        patch[seed] = color
        return patch

    raise RuntimeError("failed to generate a non-degenerate fenced region")


def _legend_patch(color: int, mode: str) -> np.ndarray:
    patch = np.full((3, 3), BLACK, dtype=int)
    for r, c in MODE_TO_PATTERN[mode]:
        patch[r, c] = color
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mark(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _paste_nonblack(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BLACK
    assert not np.any(sub[mask] != BLACK), "placement must not overwrite an existing object"
    sub[mask] = patch[mask]


def _place_legend(grid: np.ndarray, occupied: np.ndarray, rng: random.Random, mapping: dict[int, str]) -> None:
    h, w = grid.shape
    edge = rng.choice(("top", "bottom", "left", "right"))
    order = list(SEED_COLORS)
    rng.shuffle(order)
    gap = rng.choice(LEGEND_GAPS)
    span = 3 * 3 + 2 * gap

    if edge in ("top", "bottom"):
        row0 = 0 if edge == "top" else h - 3
        col0 = rng.randint(0, w - span)
        for idx, color in enumerate(order):
            patch = _legend_patch(color, mapping[color])
            c = col0 + idx * (3 + gap)
            _paste_nonblack(grid, patch, row0, c)
            _mark(occupied, row0, c, 3, 3, pad=1)
    else:
        col0 = 0 if edge == "left" else w - 3
        row0 = rng.randint(0, h - span)
        for idx, color in enumerate(order):
            patch = _legend_patch(color, mapping[color])
            r = row0 + idx * (3 + gap)
            _paste_nonblack(grid, patch, r, col0)
            _mark(occupied, r, col0, 3, 3, pad=1)


def _region_count_for(shape: tuple[int, int], rng: random.Random) -> int:
    h, w = shape
    max_regions = 3
    if min(h, w) >= 23:
        max_regions += 1
    if min(h, w) >= 26:
        max_regions += 1
    return rng.randint(3, min(5, max_regions))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    modes = ["four", "diag", "eight"]
    rng.shuffle(modes)
    mapping = {color: mode for color, mode in zip(SEED_COLORS, modes)}
    _place_legend(grid, occupied, rng, mapping)

    region_count = _region_count_for((h, w), rng)
    region_colors = list(SEED_COLORS)
    while len(region_colors) < region_count:
        region_colors.append(rng.choice(SEED_COLORS))
    rng.shuffle(region_colors)

    for color in region_colors:
        patch = _make_region_patch(rng, color)
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            if not _can_place(occupied, r0, c0, ph, pw, pad=1):
                continue
            _paste_nonblack(grid, patch, r0, c0)
            _mark(occupied, r0, c0, ph, pw, pad=1)
            placed = True
            break
        if not placed:
            return None

    decoded, legend_cells = _decode_legend(grid)
    if decoded != mapping:
        return None

    seed_counts = {color: 0 for color in SEED_COLORS}
    for r, c in np.argwhere(np.isin(grid, SEED_COLORS)):
        cell = (int(r), int(c))
        if cell in legend_cells:
            continue
        seed_counts[int(grid[cell])] += 1
    if any(count == 0 for count in seed_counts.values()):
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    for color in SEED_COLORS:
        if int(np.count_nonzero(out == color)) <= int(np.count_nonzero(grid == color)):
            return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced black maze from its seed using the connectivity encoded by the edge legend.

    Input structure:
    - The background is black.
    - Several closed grey fence regions contain black corridors and exactly one single-pixel seed in red, blue, or green.
    - On one grid edge, three 3x3 legend patches in those same colors show the allowed growth pattern for each seed color.

    Rule:
    - Read each edge legend patch: a colored cross means 4-neighbor growth, a colored X means diagonal-only growth, and a full colored 3x3 block means 8-neighbor growth.
    - For every non-legend seed pixel, flood-fill only through black cells reachable under that seed color's decoded connectivity.
    - Recolor those reachable black cells to the seed color, while leaving all grey fences and legend pixels unchanged.

    Key insight:
    - The seed colors do not imply a fixed behavior on their own; the edge legend remaps each color to a different neighborhood system, so the same fenced maze fills differently depending on whether movement is orthogonal, diagonal, or fully 8-connected.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_cells = _decode_legend(g)

    out = g.copy()
    for r, c in np.argwhere(np.isin(g, SEED_COLORS)):
        start = (int(r), int(c))
        if start in legend_cells:
            continue
        color = int(g[start])
        dirs = MODE_TO_DIRS[mapping[color]]
        seen = {start}
        q = deque([start])
        while q:
            cr, cc = q.popleft()
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == BLACK:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        for rr, cc in seen:
            out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid random instance")

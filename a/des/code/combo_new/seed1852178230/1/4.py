from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


PALETTE = tuple(range(1, 10))
GEN_RETRY_LIMIT = 500


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _detect_towers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    towers: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        for comp in _components_of_color(g, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(comp) == 9, "each tower must occupy exactly 9 cells"
            assert r1 - r0 == 2 and c1 - c0 == 2, "each tower must be a 3x3 square"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "tower must be solid"
            towers.append(
                {
                    "color": int(color),
                    "center": (int(r0 + 1), int(c0 + 1)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    towers.sort(key=lambda t: tuple(t["center"]))
    assert towers, "expected at least one tower"
    return towers


def _trace_ray(
    center: tuple[int, int],
    direction: tuple[int, int],
    owner: np.ndarray,
) -> tuple[list[tuple[int, int]], bool]:
    h, w = map(int, owner.shape)
    dr, dc = direction
    r = int(center[0] + 2 * dr)
    c = int(center[1] + 2 * dc)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if int(owner[r, c]) != -1:
            return cells, True
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells, False


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    towers = _detect_towers(g)
    h, w = map(int, g.shape)

    owner = np.full((h, w), -1, dtype=int)
    for idx, tower in enumerate(towers):
        r0, c0, r1, c1 = tower["bbox"]
        assert np.all(owner[r0 : r1 + 1, c0 : c1 + 1] == -1), "tower squares must not overlap"
        owner[r0 : r1 + 1, c0 : c1 + 1] = idx

    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    coverage: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    totals: list[int] = []
    blocked_h: list[bool] = []
    blocked_v: list[bool] = []
    ray_cells: list[set[tuple[int, int]]] = []

    for idx, tower in enumerate(towers):
        center = tower["center"]
        tower_cells: set[tuple[int, int]] = set()
        h_block = False
        v_block = False
        for name, delta in directions.items():
            cells, blocked = _trace_ray(center, delta, owner)
            tower_cells.update(cells)
            if name in ("left", "right"):
                h_block = h_block or blocked
            else:
                v_block = v_block or blocked
        for cell in sorted(tower_cells):
            coverage[cell].append(idx)
        ray_cells.append(tower_cells)
        totals.append(len(tower_cells))
        blocked_h.append(h_block)
        blocked_v.append(v_block)

    winners: dict[tuple[int, int], int] = {}
    final_counts = [0] * len(towers)
    for cell, covering in coverage.items():
        covering = sorted(set(covering))
        winner = max(covering, key=lambda idx: (totals[idx], -idx))
        winners[cell] = winner
        final_counts[winner] += 1

    overlap_pixels = sum(1 for covering in coverage.values() if len(set(covering)) > 1)
    return {
        "towers": towers,
        "totals": totals,
        "blocked_h": blocked_h,
        "blocked_v": blocked_v,
        "ray_cells": ray_cells,
        "coverage": dict(coverage),
        "winners": winners,
        "final_counts": final_counts,
        "overlap_pixels": overlap_pixels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the signal towers by their final visible ray coverage and encode the order as a color strip.

    Input structure:
    - The background is black.
    - Each foreground object is a solid 3x3 square tower with a unique color.
    - Rays are implicit: every tower emits horizontal and vertical rays from its center, and each arm stops before entering another tower's 3x3 block.

    Rule:
    - Detect every 3x3 tower and trace its four ray arms outside the tower body until the grid edge or the next tower block.
    - At every pixel covered by multiple towers' rays, keep the color of the tower with the larger total raw ray length, where the total is the sum of all four arm lengths before overlap resolution.
    - Count the ray pixels that remain visible for each tower after all crossings are resolved, sort towers by that visible count from largest to smallest, and output a 1xK strip of the corresponding colors.

    Key insight:
    - The towers themselves are only anchors; the decisive hidden quantity is each tower's total unoccluded ray length, because that priority resolves crossings before the final visible-coverage ranking is computed.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    towers = info["towers"]
    final_counts = info["final_counts"]
    order = sorted(
        range(len(towers)),
        key=lambda idx: (
            -int(final_counts[idx]),
            int(towers[idx]["center"][0]),
            int(towers[idx]["center"][1]),
        ),
    )
    out = np.zeros((1, len(order)), dtype=int)
    for j, idx in enumerate(order):
        out[0, j] = int(towers[idx]["color"])
    return out


def _build_grid(
    height: int,
    width: int,
    centers: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in zip(centers, colors):
        grid[r - 1 : r + 2, c - 1 : c + 2] = int(color)
    return grid


def _flip_centers(
    centers: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    out = list(centers)
    if random.random() < 0.5:
        out = [(height - 1 - r, c) for r, c in out]
    if random.random() < 0.5:
        out = [(r, width - 1 - c) for r, c in out]
    return [(int(r), int(c)) for r, c in out]


def _sample_other_row(height: int, forbidden: set[int], min_gap: int) -> int | None:
    candidates = [
        r
        for r in range(1, height - 1)
        if all(abs(r - fr) >= min_gap for fr in forbidden)
    ]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _sample_other_col(width: int, forbidden: set[int], min_gap: int) -> int | None:
    candidates = [
        c
        for c in range(1, width - 1)
        if all(abs(c - fc) >= min_gap for fc in forbidden)
    ]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _layout_horizontal() -> tuple[int, int, list[tuple[int, int]]] | None:
    height = random.randint(11, 23)
    width = random.randint(13, 25)

    row_a = random.randint(1, height - 3)
    row_b = row_a + 1
    row_c = _sample_other_row(height, {row_a, row_b}, min_gap=3)
    if row_c is None:
        return None

    col_a = random.randint(2, width - 8)
    col_b = random.randint(col_a + 5, width - 3)
    between = [c for c in range(col_a + 2, col_b - 1) if abs(c - col_a) >= 2 and abs(c - col_b) >= 2]
    anywhere = [
        c
        for c in range(1, width - 1)
        if abs(c - col_a) >= 2 and abs(c - col_b) >= 2
    ]
    col_c = int(random.choice(between or anywhere))

    centers = [(row_a, col_a), (row_b, col_b), (row_c, col_c)]
    return height, width, _flip_centers(centers, height, width)


def _layout_vertical() -> tuple[int, int, list[tuple[int, int]]] | None:
    height = random.randint(13, 25)
    width = random.randint(11, 23)

    col_a = random.randint(1, width - 3)
    col_b = col_a + 1
    col_c = _sample_other_col(width, {col_a, col_b}, min_gap=3)
    if col_c is None:
        return None

    row_a = random.randint(2, height - 8)
    row_b = random.randint(row_a + 5, height - 3)
    between = [r for r in range(row_a + 2, row_b - 1) if abs(r - row_a) >= 2 and abs(r - row_b) >= 2]
    anywhere = [
        r
        for r in range(1, height - 1)
        if abs(r - row_a) >= 2 and abs(r - row_b) >= 2
    ]
    row_c = int(random.choice(between or anywhere))

    centers = [(row_a, col_a), (row_b, col_b), (row_c, col_c)]
    return height, width, _flip_centers(centers, height, width)


def _layout_mixed() -> tuple[int, int, list[tuple[int, int]]] | None:
    height = random.randint(15, 25)
    width = random.randint(15, 25)

    row_a = random.randint(1, height - 9)
    row_b = row_a + 1

    col_a = random.randint(2, width - 8)
    col_b = random.randint(col_a + 5, width - 3)

    col_c_candidates = [
        c
        for c in range(1, width - 2)
        if min(abs(c - col_a), abs(c + 1 - col_a), abs(c - col_b), abs(c + 1 - col_b)) >= 2
    ]
    if not col_c_candidates:
        return None
    col_c = int(random.choice(col_c_candidates))
    col_d = col_c + 1

    row_cd_candidates = [
        r
        for r in range(1, height - 1)
        if abs(r - row_a) >= 3 and abs(r - row_b) >= 3
    ]
    if len(row_cd_candidates) < 2:
        return None

    pairs = [
        (r1, r2)
        for r1 in row_cd_candidates
        for r2 in row_cd_candidates
        if r2 - r1 >= 4
    ]
    if not pairs:
        return None
    row_c, row_d = random.choice(pairs)

    centers = [(row_a, col_a), (row_b, col_b), (row_c, col_c), (row_d, col_d)]
    return height, width, _flip_centers(centers, height, width)


def _mode_matches(info: dict[str, object], mode: str) -> bool:
    blocked_h = info["blocked_h"]
    blocked_v = info["blocked_v"]
    has_h = any(bool(v) for v in blocked_h)
    has_v = any(bool(v) for v in blocked_v)
    if mode == "horizontal":
        return has_h and not has_v
    if mode == "vertical":
        return has_v and not has_h
    return has_h and has_v


def _is_valid_attempt(grid: np.ndarray, mode: str) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    towers = info["towers"]
    totals = info["totals"]
    final_counts = info["final_counts"]

    if len(towers) < 3:
        return False
    if info["overlap_pixels"] <= 0:
        return False
    if len(set(int(v) for v in totals)) != len(totals):
        return False
    if len(set(int(v) for v in final_counts)) != len(final_counts):
        return False
    if min(int(v) for v in totals) <= 0:
        return False
    if not _mode_matches(info, mode):
        return False

    out = main(grid)
    if out.shape != (1, len(towers)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("horizontal", "vertical", "mixed"))
    if mode == "horizontal":
        layout = _layout_horizontal()
    elif mode == "vertical":
        layout = _layout_vertical()
    else:
        layout = _layout_mixed()
    if layout is None:
        return None

    height, width, centers = layout
    colors = random.sample(PALETTE, len(centers))
    grid = _build_grid(height, width, centers, colors)
    if _is_valid_attempt(grid, mode):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")

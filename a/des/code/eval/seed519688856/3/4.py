from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_LEN = 4
GEN_RETRIES = 400

EDGE_NAMES = ("top", "bottom", "left", "right")
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    mask = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-grey legend on a grid edge"

    h, w = grid.shape
    if np.all(coords[:, 0] == 0) or np.all(coords[:, 0] == h - 1):
        order = np.argsort(coords[:, 1], kind="stable")
    elif np.all(coords[:, 1] == 0) or np.all(coords[:, 1] == w - 1):
        order = np.argsort(coords[:, 0], kind="stable")
    else:
        raise AssertionError("legend pixels must lie on exactly one grid edge")

    ordered = coords[order]
    colors = [int(grid[r, c]) for r, c in ordered]
    assert len(colors) >= 2, "legend must encode an ordered color sequence"
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors, ordered


def _extract_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == GREY
    rectangles: list[tuple[int, int, int, int]] = []
    for comp in _components4(mask):
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        r1 = int(comp[:, 0].max())
        c1 = int(comp[:, 1].max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = mask[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros_like(sub, dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(sub, expected):
            rectangles.append((r0, c0, r1, c1))
    assert rectangles, "expected at least one grey rectangle outline"
    return rectangles


def _solve(grid: np.ndarray) -> np.ndarray:
    colors, legend_coords = _find_legend(grid)
    out = np.array(grid, dtype=int, copy=True)
    out[legend_coords[:, 0], legend_coords[:, 1]] = BLACK

    rectangles = _extract_rectangles(out)
    depth = np.zeros(out.shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        depth[r0 + 1 : r1, c0 + 1 : c1] += 1

    max_depth = int(depth.max())
    assert max_depth <= len(colors), "legend must cover every nesting depth"
    base_black = out == BLACK
    for level, color in enumerate(colors, start=1):
        out[base_black & (depth == level)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular regions by depth using the edge legend as an ordered palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g)


def _draw_rectangle(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rectangle_chain(top: int, left: int, inner_h: int, inner_w: int, depth: int) -> list[tuple[int, int, int, int]]:
    total_pad = 2 * (depth - 1)
    rectangles: list[tuple[int, int, int, int]] = []
    for level in range(depth):
        inset = 2 * level
        r0 = top + total_pad - inset
        c0 = left + total_pad - inset
        r1 = top + inner_h - 1 + total_pad + inset
        c1 = left + inner_w - 1 + total_pad + inset
        rectangles.append((r0, c0, r1, c1))
    return rectangles


def _place_legend(grid: np.ndarray, rng: random.Random, colors: list[int], edge: str) -> None:
    h, w = grid.shape
    gaps = [rng.randint(1, 3) for _ in range(len(colors) - 1)]
    span = len(colors) + sum(gaps)
    if edge in ("top", "bottom"):
        start = rng.randint(0, w - span)
        cols = [start]
        for gap in gaps:
            cols.append(cols[-1] + 1 + gap)
        row = 0 if edge == "top" else h - 1
        for c, color in zip(cols, colors):
            grid[row, c] = color
    else:
        start = rng.randint(0, h - span)
        rows = [start]
        for gap in gaps:
            rows.append(rows[-1] + 1 + gap)
        col = 0 if edge == "left" else w - 1
        for r, color in zip(rows, colors):
            grid[r, col] = color


def _sample_group_specs(rng: random.Random) -> list[tuple[int, int, int]]:
    max_depth = rng.choice((2, 3, 4))
    if max_depth == 4:
        group_count = rng.choice((1, 2))
    else:
        group_count = rng.choice((1, 2, 3))

    depths = [rng.randint(2, max_depth) for _ in range(group_count)]
    depths[rng.randrange(group_count)] = max_depth

    specs: list[tuple[int, int, int]] = []
    for depth in depths:
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 7)
        specs.append((depth, inner_h, inner_w))
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    edge = rng.choice(EDGE_NAMES)

    grid = np.full((h, w), BLACK, dtype=int)
    legend_colors = rng.sample(LEGEND_COLORS, LEGEND_LEN)
    _place_legend(grid, rng, legend_colors, edge)

    specs = _sample_group_specs(rng)
    placement_margin = 1
    separation = 2
    occupied = np.zeros((h, w), dtype=bool)

    all_rectangles: list[tuple[int, int, int, int]] = []
    specs.sort(key=lambda item: (item[0], item[1] * item[2]), reverse=True)
    for depth, inner_h, inner_w in specs:
        outer_h = inner_h + 4 * (depth - 1)
        outer_w = inner_w + 4 * (depth - 1)
        if outer_h > h - 2 * placement_margin or outer_w > w - 2 * placement_margin:
            return None

        placed = False
        max_r0 = h - outer_h - placement_margin
        max_c0 = w - outer_w - placement_margin
        for _ in range(120):
            r0 = rng.randint(placement_margin, max_r0)
            c0 = rng.randint(placement_margin, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - separation)
            cc0 = max(0, c0 - separation)
            rr1 = min(h, r1 + separation + 1)
            cc1 = min(w, c1 + separation + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            rects = _rectangle_chain(r0, c0, inner_h, inner_w, depth)
            for rect in rects:
                all_rectangles.append(rect)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    for r0, c0, r1, c1 in all_rectangles:
        _draw_rectangle(grid, r0, c0, r1, c1, GREY)

    out = _solve(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any((out != grid) & (out != BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")

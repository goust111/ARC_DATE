from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
CYAN = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MAX_GENERATION_ATTEMPTS = 400


def _wrap_col(c: int, width: int) -> int:
    return c % width


def _first_motif_row(grid: np.ndarray) -> int:
    rows = np.any(grid != BLACK, axis=1)
    assert np.any(rows), "expected a bottom motif"
    top = int(np.flatnonzero(rows)[0])
    assert top > 0, "expected blank rows above the motif"
    assert np.all(rows[top:]), "motif rows must be contiguous through the bottom"
    return top


def _tile_upward(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    top = _first_motif_row(g)
    motif = g[top:].copy()
    motif_h = int(motif.shape[0])

    out = np.empty_like(g)
    for r in range(g.shape[0]):
        out[r] = motif[(r - top) % motif_h]
    return out


def _background_components(grid: np.ndarray, *, wrap_cols: bool) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros_like(g, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r0 in range(height):
        for c0 in range(width):
            if seen[r0, c0] or int(g[r0, c0]) != BLACK:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if dc != 0 and wrap_cols:
                        nc = _wrap_col(nc, width)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != BLACK:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(comp)
    return components


def _enclosed_background_components(
    grid: np.ndarray, *, wrap_cols: bool, side_edges_open: bool
) -> list[list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    enclosed = []
    for comp in _background_components(grid, wrap_cols=wrap_cols):
        if any(r == 0 or r == height - 1 for r, _ in comp):
            continue
        if side_edges_open and any(c == 0 or c == width - 1 for _, c in comp):
            continue
        enclosed.append(comp)
    return enclosed


def _coords_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords:
        rr, cc = zip(*coords)
        mask[np.array(rr), np.array(cc)] = True
    return mask


def _neighbor4_with_wrap(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if dc != 0:
            nc = _wrap_col(nc, width)
        if 0 <= nr < height:
            out.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the bottom blue motif upward, then recolor wrap-enclosed cylinder holes by separating the enclosing rim from the hole-side boundary.

    Input structure:
    - The grid is black except for a contiguous motif block in the bottom 3-4 rows.
    - The motif contains blue foreground near both horizontal edges, while all rows above the motif are blank.
    - The motif is an open local pattern: only after upward tiling and left-right column wrapping can some black regions become enclosed.

    Rule:
    - Copy the bottom motif upward with the same row phase until every row is filled.
    - On the tiled grid, treat the left and right columns as horizontally adjacent when computing 4-connected black regions.
    - Every black region that touches neither the top row nor the bottom row is enclosed.
    - Recolor every blue cell that is 4-neighbor-adjacent to any enclosed region to cyan.
    - Inside each enclosed region, recolor the black cells that are 8-neighbor-adjacent to cyan to green; leave deeper enclosed black cells black.

    Key insight:
    - The hole does not exist in the flat single motif; it appears only because periodic tiling supplies the vertical caps and column wrapping glues the left and right edge neighborhoods into one cylindrical enclosure.
    """
    tiled = _tile_upward(np.asarray(grid, dtype=int))
    enclosed = _enclosed_background_components(
        tiled, wrap_cols=True, side_edges_open=False
    )

    cyan_mask = np.zeros_like(tiled, dtype=bool)
    for comp in enclosed:
        region = _coords_mask(tiled.shape, comp)
        for r, c in comp:
            for nr, nc in _neighbor4_with_wrap(r, c, tiled.shape):
                if int(tiled[nr, nc]) == BLUE:
                    cyan_mask[nr, nc] = True

    out = tiled.copy()
    out[cyan_mask] = CYAN

    green_mask = np.zeros_like(tiled, dtype=bool)
    height, width = tiled.shape
    for comp in enclosed:
        for r, c in comp:
            for dr, dc in DIRS8:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if cyan_mask[nr, nc]:
                    green_mask[r, c] = True
                    break
    out[green_mask] = GREEN
    return out


def _motif_has_required_edge_presence(motif: np.ndarray) -> bool:
    left_cols = range(min(3, motif.shape[1]))
    right_cols = range(max(0, motif.shape[1] - 3), motif.shape[1])
    return bool(np.any(motif[:, list(left_cols)] == BLUE)) and bool(
        np.any(motif[:, list(right_cols)] == BLUE)
    )


def _planar_nonwrap_hole_mask(grid: np.ndarray) -> np.ndarray:
    return _coords_mask(
        grid.shape,
        [
            cell
            for comp in _enclosed_background_components(
                grid, wrap_cols=False, side_edges_open=True
            )
            for cell in comp
        ],
    )


def _wrap_hole_mask(grid: np.ndarray) -> np.ndarray:
    return _coords_mask(
        grid.shape,
        [
            cell
            for comp in _enclosed_background_components(
                grid, wrap_cols=True, side_edges_open=False
            )
            for cell in comp
        ],
    )


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        g = np.asarray(inp, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        top = _first_motif_row(g)
        if not np.all(g[:top] == BLACK):
            return False

        motif = g[top:].copy()
        if motif.shape[0] not in (3, 4):
            return False
        if not _motif_has_required_edge_presence(motif):
            return False
        if np.any(_wrap_hole_mask(motif)):
            return False
        if np.any(_planar_nonwrap_hole_mask(motif)):
            return False

        tiled = _tile_upward(g)
        wrap_holes = _wrap_hole_mask(tiled)
        if not np.any(wrap_holes):
            return False

        nonwrap_holes = _planar_nonwrap_hole_mask(tiled)
        if not np.any(wrap_holes & ~nonwrap_holes):
            return False

        out = main(g)
        if out.shape != g.shape:
            return False
        if max(out.shape) > 30:
            return False
        if not np.any(out == BLUE):
            return False
        if not np.any(out == CYAN):
            return False
        if not np.any(out == GREEN):
            return False
        return True
    except AssertionError:
        return False


def _sample_dimensions() -> tuple[int, int, int]:
    motif_h = random.choice((3, 4))
    height = random.randint(max(10, motif_h * 2 + 1), 15)
    width = random.randint(10, 16)
    return height, width, motif_h


def _wrapped_cap_columns(width: int, left_margin: int, right_margin: int) -> list[int]:
    left_wall = left_margin
    right_wall = width - 1 - right_margin
    cols = list(range(0, left_wall + 1))
    cols += list(range(right_wall, width))
    return sorted(set(cols))


def _build_motif(height: int, width: int, motif_h: int) -> np.ndarray | None:
    if width < 10:
        return None

    left_margin = random.choice((1, 2))
    right_margin = random.choice((1, 2))
    left_wall = left_margin
    right_wall = width - 1 - right_margin
    if right_wall - left_wall < 6:
        return None

    motif = np.zeros((motif_h, width), dtype=int)
    cap_row = random.randrange(motif_h)

    for r in range(motif_h):
        motif[r, left_wall] = BLUE
        motif[r, right_wall] = BLUE
    motif[cap_row, _wrapped_cap_columns(width, left_margin, right_margin)] = BLUE

    interior_candidates = list(range(left_wall + 3, right_wall - 2))
    if len(interior_candidates) < 3:
        return None

    distractor_count = random.choice((1, 2))
    distractor_cols = random.sample(interior_candidates, k=distractor_count)
    for col in distractor_cols:
        style = random.choice(("solid", "gapped", "offset"))
        for r in range(motif_h):
            paint = True
            if style == "gapped" and r == cap_row:
                paint = False
            elif style == "offset" and (r + col) % 2 == 0:
                paint = False
            if paint:
                motif[r, col] = BLUE

    if not np.any(motif != BLACK, axis=1).all():
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[height - motif_h :] = motif
    return grid


def _gen_attempt() -> np.ndarray | None:
    height, width, motif_h = _sample_dimensions()
    grid = _build_motif(height, width, motif_h)
    if grid is None or not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cylindrical-wrap tiling puzzle")

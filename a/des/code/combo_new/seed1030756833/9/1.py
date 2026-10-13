from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _nonzero_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color == BG or seen[sr, sc]:
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": set(coords),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _contiguous(indices: list[int]) -> bool:
    return indices == list(range(indices[0], indices[-1] + 1))


def _frame_from_component(g: np.ndarray, comp: dict) -> tuple[tuple[int, int, int, int], str] | None:
    h, w = g.shape
    r0, c0, r1, c1 = comp["bbox"]
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 3 or outer_w < 3:
        return None
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None

    cells = comp["cells"]
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if not cells <= perimeter:
        return None

    missing = perimeter - cells
    if not missing:
        return None
    if any(int(g[r, c]) != BG for r, c in missing):
        return None
    if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None

    side_cells = {
        "top": {(r0, c) for c in range(c0 + 1, c1)},
        "bottom": {(r1, c) for c in range(c0 + 1, c1)},
        "left": {(r, c0) for r in range(r0 + 1, r1)},
        "right": {(r, c1) for r in range(r0 + 1, r1)},
    }
    possible = [side for side, allowed in side_cells.items() if missing <= allowed]
    if len(possible) != 1:
        return None

    side = possible[0]
    if side in ("top", "bottom"):
        idx = sorted(c for _, c in missing)
    else:
        idx = sorted(r for r, _ in missing)
    if not _contiguous(idx):
        return None

    return (int(r0), int(c0), int(r1), int(c1)), side


def _boundary_bands(g: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": g[0, :].copy(),
        "bottom": g[-1, :].copy(),
        "left": g[:, 0].copy(),
        "right": g[:, -1].copy(),
    }


def _fill_frame_interior(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap_side: str,
    bands: dict[str, np.ndarray],
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h <= 0 or inner_w <= 0:
        return

    band = bands[gap_side]
    if gap_side in ("top", "bottom"):
        row = np.array([int(band[i % len(band)]) for i in range(inner_w)], dtype=int)
        out[r0 + 1 : r1, c0 + 1 : c1] = row
    else:
        col = np.array([int(band[i % len(band)]) for i in range(inner_h)], dtype=int)
        out[r0 + 1 : r1, c0 + 1 : c1] = col[:, None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with the boundary stripe selected by its gap direction.

    Input structure:
    - The background is black.
    - The four outer grid edges are one-pixel-wide colored stripe patterns.
    - Inside the grid are several one-pixel-thick hollow rectangular frames; each frame has exactly one continuous black gap on one of its four sides.

    Rule:
    - Find the side containing each frame's gap.
    - A top, bottom, left, or right gap selects the matching top, bottom, left, or right boundary stripe.
    - Fill the frame interior by repeating a horizontal stripe across every interior row, or a vertical stripe down every interior column.
    - Keep frame borders unchanged and leave the black gap cells black.

    Key insight:
    The gap direction is an index into four different edge stripe patterns, and the selected one-dimensional pattern becomes the frame's interior texture.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bands = _boundary_bands(g)

    frames: list[tuple[tuple[int, int, int, int], str]] = []
    for comp in _nonzero_components(g):
        parsed = _frame_from_component(g, comp)
        if parsed is not None:
            frames.append(parsed)

    frames.sort()
    for bbox, gap_side in frames:
        _fill_frame_interior(out, bbox, gap_side, bands)

    return out


# ----------------------------- generator -----------------------------

COLOR_POOL = list(range(1, 10))
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (4, 6)
FRAME_H_RANGE = (4, 8)
FRAME_W_RANGE = (4, 10)
MAX_GENERATION_ATTEMPTS = 500


def _run_lengths(length: int, n_runs: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, length), n_runs - 1))
    pts = [0] + cuts + [length]
    return [pts[i + 1] - pts[i] for i in range(n_runs)]


def _stripe(length: int, rng: random.Random) -> np.ndarray:
    max_runs = min(6, length)
    min_runs = min(3, max_runs)
    n_runs = rng.randint(min_runs, max_runs)
    lengths = _run_lengths(length, n_runs, rng)

    run_colors: list[int] = []
    for _ in range(n_runs):
        choices = [c for c in COLOR_POOL if not run_colors or c != run_colors[-1]]
        run_colors.append(rng.choice(choices))

    for _ in range(30):
        if len(set(run_colors)) >= min(3, n_runs):
            break
        run_colors = []
        for _i in range(n_runs):
            choices = [c for c in COLOR_POOL if not run_colors or c != run_colors[-1]]
            run_colors.append(rng.choice(choices))

    return np.repeat(np.array(run_colors, dtype=int), lengths)


def _stripe_with_ends(length: int, first: int, last: int, rng: random.Random) -> np.ndarray:
    middle = _stripe(length - 2, rng)
    return np.concatenate(
        [
            np.array([int(first)], dtype=int),
            middle,
            np.array([int(last)], dtype=int),
        ]
    )


def _make_boundary_bands(h: int, w: int, rng: random.Random) -> dict[str, np.ndarray] | None:
    for _ in range(100):
        top = _stripe(w, rng)
        bottom = _stripe(w, rng)
        left = _stripe_with_ends(h, int(top[0]), int(bottom[0]), rng)
        right = _stripe_with_ends(h, int(top[-1]), int(bottom[-1]), rng)
        bands = {"top": top, "bottom": bottom, "left": left, "right": right}
        keys = {tuple(map(int, band.tolist())) for band in bands.values()}
        if len(keys) == 4:
            return bands
    return None


def _paint_boundary(g: np.ndarray, bands: dict[str, np.ndarray]) -> None:
    g[0, :] = bands["top"]
    g[-1, :] = bands["bottom"]
    g[:, 0] = bands["left"]
    g[:, -1] = bands["right"]


def _free_positions(occ: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(2, h - outer_h - 1):
        for c0 in range(2, w - outer_w - 1):
            padded = occ[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1]
            if not padded.any():
                positions.append((r0, c0))
    return positions


def _draw_gapped_frame(
    g: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    color: int,
    side: str,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color

    if side in ("top", "bottom"):
        span = outer_w - 2
        gap_len = rng.randint(1, span)
        start = rng.randint(1, span - gap_len + 1)
        rr = r0 if side == "top" else r1
        g[rr, c0 + start : c0 + start + gap_len] = BG
    else:
        span = outer_h - 2
        gap_len = rng.randint(1, span)
        start = rng.randint(1, span - gap_len + 1)
        cc = c0 if side == "left" else c1
        g[r0 + start : r0 + start + gap_len, cc] = BG

    return r0, c0, r1, c1


def _place_frame(
    g: np.ndarray,
    occ: np.ndarray,
    color: int,
    side: str,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], str] | None:
    h, w = g.shape
    max_h = min(FRAME_H_RANGE[1], h - 4)
    max_w = min(FRAME_W_RANGE[1], w - 4)
    if max_h < FRAME_H_RANGE[0] or max_w < FRAME_W_RANGE[0]:
        return None

    for _ in range(80):
        outer_h = rng.randint(FRAME_H_RANGE[0], max_h)
        outer_w = rng.randint(FRAME_W_RANGE[0], max_w)
        positions = _free_positions(occ, outer_h, outer_w)
        if not positions:
            continue
        r0, c0 = rng.choice(positions)
        bbox = _draw_gapped_frame(g, r0, c0, outer_h, outer_w, color, side, rng)
        occ[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True
        return bbox, side
    return None


def _expected_output(inp: np.ndarray, frames: list[tuple[tuple[int, int, int, int], str]]) -> np.ndarray:
    out = inp.copy()
    bands = _boundary_bands(inp)
    for bbox, side in sorted(frames):
        _fill_frame_interior(out, bbox, side, bands)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    bands = _make_boundary_bands(h, w, rng)
    if bands is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_boundary(grid, bands)

    n_frames = rng.randint(*FRAME_COUNT_RANGE)
    sides = list(SIDES)
    while len(sides) < n_frames:
        sides.append(rng.choice(SIDES))
    rng.shuffle(sides)

    frame_colors = rng.sample(COLOR_POOL, n_frames)
    occ = np.zeros((h, w), dtype=bool)
    placed: list[tuple[tuple[int, int, int, int], str]] = []
    for color, side in zip(frame_colors, sides):
        info = _place_frame(grid, occ, color, side, rng)
        if info is None:
            return None
        placed.append(info)

    if {side for _, side in placed} != set(SIDES):
        return None

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG:
        return None

    expected = _expected_output(grid, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

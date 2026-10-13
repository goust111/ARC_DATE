from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TERRAIN = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != TERRAIN)
    pieces: list[dict] = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each floating piece must be monochromatic"
        color = next(iter(colors))
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        norm = coords - np.array([r0, c0], dtype=int)
        h = int(norm[:, 0].max()) + 1
        w = int(norm[:, 1].max()) + 1
        by_col: dict[int, int] = {}
        for rr, cc in norm:
            by_col[int(cc)] = max(by_col.get(int(cc), -1), int(rr))
        pieces.append(
            {
                "color": color,
                "cells": [(int(r), int(c)) for r, c in norm],
                "h": h,
                "w": w,
                "bottoms": by_col,
            }
        )
    return pieces


def _can_place(piece: dict, occ: np.ndarray, top: int, left: int) -> bool:
    h, w = occ.shape
    for dr, dc in piece["cells"]:
        r = top + dr
        c = left + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occ[r, c]:
            return False
    return True


def _rest_row(piece: dict, occ: np.ndarray, left: int) -> int | None:
    h, w = occ.shape
    if left < 0 or left + piece["w"] > w:
        return None
    if not _can_place(piece, occ, 0, left):
        return None
    row = 0
    while row + 1 <= h - piece["h"] and _can_place(piece, occ, row + 1, left):
        row += 1
    return row


def _zero_gap(piece: dict, occ: np.ndarray, top: int, left: int) -> bool:
    h, _ = occ.shape
    for local_c, local_bottom in piece["bottoms"].items():
        r = top + local_bottom
        c = left + local_c
        if r + 1 < h and not occ[r + 1, c]:
            return False
    return True


def _legal_landings(piece: dict, occ: np.ndarray) -> list[tuple[int, int]]:
    _, w = occ.shape
    landings: list[tuple[int, int]] = []
    for left in range(w - piece["w"] + 1):
        top = _rest_row(piece, occ, left)
        if top is None:
            continue
        if _zero_gap(piece, occ, top, left):
            landings.append((top, left))
    return landings


def _solve(grid: np.ndarray, *, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    terrain_mask = g == TERRAIN
    assert terrain_mask[-1].any(), "terrain must touch the bottom edge"
    pieces = _extract_pieces(g)
    assert pieces, "expected at least one floating piece"

    out = np.where(terrain_mask, TERRAIN, BG).astype(int)
    occ = terrain_mask.copy()
    remaining = [dict(piece) for piece in pieces]
    steps = []

    while remaining:
        all_landings = [_legal_landings(piece, occ) for piece in remaining]
        movable = [idx for idx, landings in enumerate(all_landings) if landings]
        assert len(movable) == 1, "exactly one piece should be placeable at each step"
        chosen = movable[0]
        assert len(all_landings[chosen]) == 1, "the movable piece must have a unique landing"
        top, left = all_landings[chosen][0]
        piece = remaining.pop(chosen)

        for dr, dc in piece["cells"]:
            r = top + dr
            c = left + dc
            occ[r, c] = True
            out[r, c] = int(piece["color"])

        if trace:
            steps.append(
                {
                    "color": int(piece["color"]),
                    "top": int(top),
                    "left": int(left),
                    "choices": [len(v) for v in all_landings],
                }
            )

    return (out, steps) if trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the floating colored polyominoes into the gray terrain in the only zero-gap order.

    Input structure:
    - The lower part of the grid is a bottom-connected gray terrain whose columns are vertically solid from the bottom up.
    - Several separated colored polyomino pieces float above that terrain on black background.
    - The floating pieces are not obstacles to each other; each piece may be translated horizontally before falling.

    Rule:
    - For each remaining piece, test every horizontal offset, let it fall straight down to its resting row, and keep only offsets where every occupied column of the landed piece is supported immediately from below with no black gap.
    - At any moment exactly one not-yet-dropped piece has exactly one such zero-gap landing; drop that piece there and merge it into the terrain.
    - Repeat until all pieces have landed, preserving every piece's color and leaving all unused space black.

    Key insight:
    - A legal landing is determined by contour complement: the piece's column bottoms must match one terrain notch exactly, and each landed piece reshapes the contour to unlock the next unique match.
    """
    return _solve(grid)


def _shape_from_depths(depths: tuple[int, ...]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c, depth in enumerate(depths):
        for r in range(depth):
            cells.append((r, c))
    return cells


def _terrain_grid(heights: np.ndarray, h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for c, height in enumerate(heights):
        grid[h - int(height) :, c] = TERRAIN
    return grid


def _expected_output(
    heights0: np.ndarray,
    records: list[dict],
    h: int,
    w: int,
) -> np.ndarray:
    out = _terrain_grid(heights0, h, w)
    for rec in records:
        top = h - int(rec["ceiling"])
        left = int(rec["left"])
        for dr, dc in rec["cells"]:
            out[top + dr, left + dc] = int(rec["color"])
    return out


def _fits_with_gap(mask: np.ndarray, cells: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = mask.shape
    abs_cells = {(top + r, left + c) for r, c in cells}
    for r, c in abs_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c]:
            return False
    for r, c in abs_cells:
        for dr, dc in N8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in abs_cells:
                return False
    return True


def _place_floating_pieces(
    base: np.ndarray,
    piece_specs: list[dict],
    max_terrain_height: int,
) -> np.ndarray | None:
    h, w = base.shape
    grid = base.copy()
    blocked = base != BG
    top_band = h - int(max_terrain_height) - 3
    if top_band < 3:
        return None

    order = sorted(piece_specs, key=lambda rec: (-rec["h"], -rec["w"]))
    for rec in order:
        cells = rec["cells"]
        ph = int(rec["h"])
        pw = int(rec["w"])
        candidates: list[tuple[int, int]] = []
        for top in range(0, max(1, top_band - ph + 1)):
            for left in range(0, w - pw + 1):
                if _fits_with_gap(blocked, cells, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        for dr, dc in cells:
            r = top + dr
            c = left + dc
            grid[r, c] = int(rec["color"])
            blocked[r, c] = True
    return grid


def _candidate_spans(width: int, prev_span: tuple[int, int] | None) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    w_min = 2
    w_max = min(6, width)
    widths = list(range(w_min, w_max + 1))
    random.shuffle(widths)
    for span_w in widths:
        starts = list(range(0, width - span_w + 1))
        random.shuffle(starts)
        for left in starts:
            right = left + span_w
            if prev_span is None:
                spans.append((left, right))
                continue
            a, b = prev_span
            if max(left, a) < min(right, b):
                spans.append((left, right))
    return spans


def _random_heights(width: int) -> np.ndarray:
    heights = [random.randint(2, 4)]
    for _ in range(1, width):
        step = random.choice([-1, 0, 1])
        heights.append(int(np.clip(heights[-1] + step, 1, 5)))
    return np.array(heights, dtype=int)


def _build_chain(h: int, w: int, piece_count: int) -> tuple[np.ndarray, list[dict]] | None:
    heights0 = _random_heights(w)
    heights = heights0.copy()
    prev_span: tuple[int, int] | None = None
    colors = random.sample(PIECE_COLORS, piece_count)
    max_ceiling = h - 4
    max_piece_height = 5
    records: list[dict] = []
    irregular_used = 0

    for idx in range(piece_count):
        placed = False
        for left, right in _candidate_spans(w, prev_span):
            segment = heights[left:right]
            low = int(segment.min())
            high = int(segment.max())
            if high >= max_ceiling:
                continue
            ceiling_lo = high + 1
            ceiling_hi = min(low + max_piece_height, max_ceiling)
            if ceiling_lo > ceiling_hi:
                continue
            ceilings = list(range(ceiling_lo, ceiling_hi + 1))
            random.shuffle(ceilings)
            for ceiling in ceilings:
                depths = tuple(int(ceiling - v) for v in segment)
                if any(d <= 0 or d > max_piece_height for d in depths):
                    continue
                if sum(depths) < 3:
                    continue
                if idx == 0 and len(set(depths)) == 1:
                    continue
                if records and depths == records[-1]["depths"]:
                    continue
                if len(set(depths)) > 1:
                    irregular_used += 1
                cells = _shape_from_depths(depths)
                rec = {
                    "color": colors[idx],
                    "left": left,
                    "ceiling": ceiling,
                    "depths": depths,
                    "cells": cells,
                    "h": max(depths),
                    "w": len(depths),
                }
                records.append(rec)
                heights[left:right] = ceiling
                prev_span = (left, right)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if irregular_used == 0:
        return None
    return heights0, records


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    piece_count: int,
    records: list[dict],
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    try:
        _, steps = _solve(grid, trace=True)
    except AssertionError:
        return False
    if len(steps) != piece_count:
        return False
    if any(sum(v > 0 for v in step["choices"]) != 1 for step in steps):
        return False
    if any(max(step["choices"]) != 1 for step in steps):
        return False
    colors_seen = {int(step["color"]) for step in steps}
    if len(colors_seen) != piece_count:
        return False
    if len({tuple(rec["depths"]) for rec in records}) < 2:
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def _generate_one() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(14, 24)
    piece_count = random.randint(3, 5)

    chain = _build_chain(h, w, piece_count)
    if chain is None:
        return None
    heights0, records = chain
    base = _terrain_grid(heights0, h, w)
    grid = _place_floating_pieces(base, records, int(heights0.max()))
    if grid is None:
        return None
    expected = _expected_output(heights0, records, h, w)
    if not _validate_generated(grid, expected, piece_count, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid zero-gap drop puzzle")

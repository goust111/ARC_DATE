from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import numpy as np


BG = 0
FG = 3
NW_COLOR = 8
NE_COLOR = 2

NW_SE = "nw_se"
NE_SW = "ne_sw"
RELATION_CYCLE = ("equal", "nw_more", "ne_more")

_GENERATE_CALL_COUNT = 0
_USED_OUTPUT_SIGNATURES: set[str] = set()


def _classify_pixels(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split every foreground pixel into the unique diagonal orientation it belongs to."""
    h, w = grid.shape
    fg = grid == FG
    nw_pixels: list[tuple[int, int]] = []
    ne_pixels: list[tuple[int, int]] = []

    for r0, c0 in np.argwhere(fg):
        r = int(r0)
        c = int(c0)
        has_nw = (
            (r > 0 and c > 0 and fg[r - 1, c - 1])
            or (r + 1 < h and c + 1 < w and fg[r + 1, c + 1])
        )
        has_ne = (
            (r > 0 and c + 1 < w and fg[r - 1, c + 1])
            or (r + 1 < h and c > 0 and fg[r + 1, c - 1])
        )
        assert has_nw ^ has_ne, "each green pixel must belong to exactly one diagonal orientation"
        if has_nw:
            nw_pixels.append((r, c))
        else:
            ne_pixels.append((r, c))
    assert nw_pixels and ne_pixels, "both diagonal orientations must be present"
    return nw_pixels, ne_pixels


def _transform(grid: np.ndarray) -> np.ndarray:
    """Apply the orientation-driven gravity described by the task."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((g == BG) | (g == FG)), "input must contain only black background and green foreground"

    h, w = g.shape
    nw_pixels, ne_pixels = _classify_pixels(g)
    out = np.full((h, w), BG, dtype=int)

    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in nw_pixels:
        by_col[c].append(r)
    for c, rows in by_col.items():
        rows.sort()
        start = h - len(rows)
        for offset, _r in enumerate(rows):
            rr = start + offset
            assert out[rr, c] == BG
            out[rr, c] = NW_COLOR

    by_row: dict[int, list[int]] = defaultdict(list)
    for r, c in ne_pixels:
        by_row[r].append(c)
    for r, cols in by_row.items():
        cols.sort()
        start = w - len(cols)
        for offset, _c in enumerate(cols):
            cc = start + offset
            assert out[r, cc] == BG, "valid inputs must not produce landing collisions"
            out[r, cc] = NE_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group green diagonal segments by orientation, then drop one group down and slide the other right before recoloring them.

    Input structure:
    - The grid is a black background with scattered green diagonal line segments.
    - Each segment is a short straight diagonal run, and every green pixel belongs to exactly one of the two diagonal orientations.
    - NW-SE oriented pixels and NE-SW oriented pixels may occupy the same columns or rows, but the segments do not touch.

    Rule:
    - Classify each green pixel by its same-color diagonal neighbors: NW-SE pixels have a green neighbor on the NW/SE axis, and NE-SW pixels have a green neighbor on the NE/SW axis.
    - For each column, collect the NW-SE pixels from top to bottom and stably stack them at the bottom of that same column.
    - For each row, collect the NE-SW pixels from left to right and stably stack them at the right edge of that same row.
    - Recolor the settled NW-SE pixels teal and the settled NE-SW pixels red.

    Key insight:
    - The diagonal orientation is not just a label: it determines both the motion axis and the final color, so the solver must infer orientation before applying gravity.
    """
    return _transform(grid)


def _component8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _orientation_of_component(comp: list[tuple[int, int]]) -> str | None:
    if not 2 <= len(comp) <= 4:
        return None
    coords = sorted(comp)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if len(set(rows)) != len(coords) or rows != list(range(rows[0], rows[0] + len(coords))):
        return None
    if cols == list(range(cols[0], cols[0] + len(coords))):
        return NW_SE
    if cols == list(range(cols[0], cols[0] - len(coords), -1)):
        return NE_SW
    return None


def _segment_cells(orientation: str, r: int, c: int, length: int) -> tuple[tuple[int, int], ...]:
    if orientation == NW_SE:
        return tuple((r + i, c + i) for i in range(length))
    return tuple((r + i, c - i) for i in range(length))


def _candidate_segments(h: int, w: int, orientation: str, length: int) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h - length + 1):
        if orientation == NW_SE:
            cols = range(w - length + 1)
        else:
            cols = range(length - 1, w)
        for c in cols:
            out.append(_segment_cells(orientation, r, c, length))
    return out


def _share_score(
    coords: tuple[tuple[int, int], ...],
    orientation: str,
    used_cols_nw: set[int],
    used_rows_ne: set[int],
) -> int:
    if orientation == NW_SE:
        return len({c for _r, c in coords} & used_cols_nw)
    return len({r for r, _c in coords} & used_rows_ne)


def _blocked_cells(coords: tuple[tuple[int, int], ...], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _sample_lengths(nw_count: int, ne_count: int, relation: str) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        nw_lengths = [random.randint(2, 4) for _ in range(nw_count)]
        ne_lengths = [random.randint(2, 4) for _ in range(ne_count)]
        nw_sum = sum(nw_lengths)
        ne_sum = sum(ne_lengths)
        if relation == "equal" and nw_sum != ne_sum:
            continue
        if relation == "nw_more" and nw_sum <= ne_sum:
            continue
        if relation == "ne_more" and ne_sum <= nw_sum:
            continue
        return nw_lengths, ne_lengths
    return None


def _place_segments(
    h: int,
    w: int,
    nw_lengths: list[int],
    ne_lengths: list[int],
) -> list[tuple[str, tuple[tuple[int, int], ...]]] | None:
    blocked: set[tuple[int, int]] = set()
    used_cols_nw: set[int] = set()
    used_rows_ne: set[int] = set()
    placed: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    specs = [(NW_SE, length) for length in nw_lengths] + [(NE_SW, length) for length in ne_lengths]
    random.shuffle(specs)
    specs.sort(key=lambda item: item[1], reverse=True)

    for orientation, length in specs:
        candidates = []
        for coords in _candidate_segments(h, w, orientation, length):
            if any(cell in blocked for cell in coords):
                continue
            score = _share_score(coords, orientation, used_cols_nw, used_rows_ne)
            candidates.append((score, coords))
        if not candidates:
            return None
        best_score = max(score for score, _coords in candidates)
        pool = [coords for score, coords in candidates if score == best_score]
        coords = random.choice(pool)
        placed.append((orientation, coords))
        blocked |= _blocked_cells(coords, h, w)
        if orientation == NW_SE:
            used_cols_nw |= {c for _r, c in coords}
        else:
            used_rows_ne |= {r for r, _c in coords}
    return placed


def _render_input(h: int, w: int, placed: list[tuple[str, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for _orientation, coords in placed:
        for r, c in coords:
            grid[r, c] = FG
    return grid


def _wrong_recolor_only(grid: np.ndarray) -> np.ndarray:
    nw_pixels, ne_pixels = _classify_pixels(grid)
    out = np.full_like(grid, BG)
    for r, c in nw_pixels:
        out[r, c] = NW_COLOR
    for r, c in ne_pixels:
        out[r, c] = NE_COLOR
    return out


def _wrong_swapped_axes(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    nw_pixels, ne_pixels = _classify_pixels(grid)
    out = np.full_like(grid, BG)
    by_row_nw: dict[int, list[int]] = defaultdict(list)
    by_col_ne: dict[int, list[int]] = defaultdict(list)
    for r, c in nw_pixels:
        by_row_nw[r].append(c)
    for r, c in ne_pixels:
        by_col_ne[c].append(r)
    for r, cols in by_row_nw.items():
        cols.sort()
        start = w - len(cols)
        for offset, _c in enumerate(cols):
            out[r, start + offset] = NW_COLOR
    for c, rows in by_col_ne.items():
        rows.sort()
        start = h - len(rows)
        for offset, _r in enumerate(rows):
            rr = start + offset
            if out[rr, c] != BG:
                return out
            out[rr, c] = NE_COLOR
    return out


def _wrong_all_vertical(grid: np.ndarray) -> np.ndarray:
    h, _w = grid.shape
    nw_pixels, ne_pixels = _classify_pixels(grid)
    out = np.full_like(grid, BG)
    groups = ((nw_pixels, NW_COLOR), (ne_pixels, NE_COLOR))
    for pixels, color in groups:
        by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in pixels:
            by_col[c].append(r)
        for c, rows in by_col.items():
            rows.sort()
            start = h - len(rows)
            for offset, _r in enumerate(rows):
                rr = start + offset
                if out[rr, c] != BG:
                    return out
                out[rr, c] = color
    return out


def _wrong_all_horizontal(grid: np.ndarray) -> np.ndarray:
    _h, w = grid.shape
    nw_pixels, ne_pixels = _classify_pixels(grid)
    out = np.full_like(grid, BG)
    groups = ((nw_pixels, NW_COLOR), (ne_pixels, NE_COLOR))
    for pixels, color in groups:
        by_row: dict[int, list[int]] = defaultdict(list)
        for r, c in pixels:
            by_row[r].append(c)
        for r, cols in by_row.items():
            cols.sort()
            start = w - len(cols)
            for offset, _c in enumerate(cols):
                cc = start + offset
                if out[r, cc] != BG:
                    return out
                out[r, cc] = color
    return out


def _semantic_checks(inp: np.ndarray, relation: str) -> bool:
    fg_mask = inp == FG
    comps = _component8(fg_mask)
    if len(comps) < 4:
        return False

    component_orientations: list[str] = []
    for comp in comps:
        orient = _orientation_of_component(comp)
        if orient is None:
            return False
        component_orientations.append(orient)

    if NW_SE not in component_orientations or NE_SW not in component_orientations:
        return False

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for r1, c1 in comps[i]:
                for r2, c2 in comps[j]:
                    if max(abs(r1 - r2), abs(c1 - c2)) < 2:
                        return False

    try:
        nw_pixels, ne_pixels = _classify_pixels(inp)
        out = main(inp)
    except AssertionError:
        return False

    if int((out == NW_COLOR).sum()) != len(nw_pixels):
        return False
    if int((out == NE_COLOR).sum()) != len(ne_pixels):
        return False
    if np.any(out == FG):
        return False

    nw_sum = len(nw_pixels)
    ne_sum = len(ne_pixels)
    if relation == "equal" and nw_sum != ne_sum:
        return False
    if relation == "nw_more" and nw_sum <= ne_sum:
        return False
    if relation == "ne_more" and ne_sum <= nw_sum:
        return False

    nw_cols = defaultdict(int)
    ne_rows = defaultdict(int)
    for _r, c in nw_pixels:
        nw_cols[c] += 1
    for r, _c in ne_pixels:
        ne_rows[r] += 1
    if max(nw_cols.values()) < 2 or max(ne_rows.values()) < 2:
        return False

    if not any(out[r, c] == NW_COLOR for r, c in nw_pixels):
        pass
    moved_nw = any(out[r, c] != NW_COLOR for r, c in nw_pixels)
    moved_ne = any(out[r, c] != NE_COLOR for r, c in ne_pixels)
    if not (moved_nw and moved_ne):
        return False

    if np.array_equal(out, _wrong_recolor_only(inp)):
        return False
    if np.array_equal(out, _wrong_swapped_axes(inp)):
        return False
    if np.array_equal(out, _wrong_all_vertical(inp)):
        return False
    if np.array_equal(out, _wrong_all_horizontal(inp)):
        return False

    sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if sig in _USED_OUTPUT_SIGNATURES:
        return False
    return True


def _next_relation() -> str:
    global _GENERATE_CALL_COUNT
    relation = RELATION_CYCLE[_GENERATE_CALL_COUNT % len(RELATION_CYCLE)]
    _GENERATE_CALL_COUNT += 1
    return relation


def _generate_attempt(relation: str) -> np.ndarray | None:
    h = random.randint(10, 15)
    w = random.randint(10, 15)
    nw_count = random.randint(2, 3)
    ne_count = random.randint(2, 3)
    lengths = _sample_lengths(nw_count, ne_count, relation)
    if lengths is None:
        return None
    nw_lengths, ne_lengths = lengths
    placed = _place_segments(h, w, nw_lengths, ne_lengths)
    if placed is None:
        return None
    grid = _render_input(h, w, placed)
    if not _semantic_checks(grid, relation):
        return None
    return grid


def generate_input() -> np.ndarray:
    relation = _next_relation()
    for _ in range(500):
        grid = _generate_attempt(relation)
        if grid is None:
            continue
        out = main(grid)
        sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        _USED_OUTPUT_SIGNATURES.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid input for the diagonal-gravity task")

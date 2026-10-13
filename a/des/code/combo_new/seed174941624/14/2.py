from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

FRAGMENT_COLORS = (GREEN, YELLOW, MAGENTA, CYAN, MAROON)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 17
GRID_MAX = 29
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
MARKER_ATTEMPTS = 160


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_mask(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if occupied[r, c]:
            return True
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                return True
    return False


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _random_shape() -> np.ndarray:
    area = random.randint(1, 6)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        cells.add(random.choice(tuple(frontier)))

    for _ in range(random.randint(0, 3)):
        cells = {(c, -r) for r, c in cells}
    if random.random() < 0.5:
        cells = {(-r, c) for r, c in cells}
    if random.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    return _normalize_cells(cells)


def _motion_info(coords: np.ndarray, center_r: int, center_c: int) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    upper = bool(np.all(rows < center_r))
    lower = bool(np.all(rows > center_r))
    left = bool(np.all(cols < center_c))
    right = bool(np.all(cols > center_c))
    assert upper != lower and left != right, "fragment must lie inside one quadrant"

    dr = 1 if upper else -1
    dc = 1 if left else -1
    row_gap = int(center_r - 1 - rows.max()) if upper else int(rows.min() - (center_r + 1))
    col_gap = int(center_c - 1 - cols.max()) if left else int(cols.min() - (center_c + 1))
    assert row_gap >= 0 and col_gap >= 0, "fragment must stay off the central cross"

    steps = min(row_gap, col_gap)
    shifted = coords + np.array([dr * steps, dc * steps], dtype=int)
    row_lo = int(min(rows.min(), shifted[:, 0].min()))
    row_hi = int(max(rows.max(), shifted[:, 0].max()))
    col_lo = int(min(cols.min(), shifted[:, 1].min()))
    col_hi = int(max(cols.max(), shifted[:, 1].max()))

    stop_kind = "both"
    if row_gap < col_gap:
        stop_kind = "row"
    elif col_gap < row_gap:
        stop_kind = "col"

    assert not bool(np.any(shifted[:, 0] == center_r))
    assert not bool(np.any(shifted[:, 1] == center_c))
    assert bool(
        np.any(np.abs(shifted[:, 0] - center_r) == 1)
        or np.any(np.abs(shifted[:, 1] - center_c) == 1)
    ), "final fragment must touch the cross by 4-neighborhood"

    return {
        "dr": int(dr),
        "dc": int(dc),
        "steps": int(steps),
        "shifted": shifted,
        "row_span": (row_lo, row_hi),
        "col_span": (col_lo, col_hi),
        "stop_kind": stop_kind,
    }


def _swept_cells(coords: np.ndarray, dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    swept: set[tuple[int, int]] = set()
    for t in range(int(steps) + 1):
        rr = coords[:, 0] + int(dr) * t
        cc = coords[:, 1] + int(dc) * t
        swept.update((int(r), int(c)) for r, c in zip(rr, cc, strict=False))
    return swept


def _count_markers_for_spans(
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    orange_rows: set[int],
    orange_cols: set[int],
) -> int:
    row_lo, row_hi = row_span
    col_lo, col_hi = col_span
    vertical = sum(row_lo <= r <= row_hi for r in orange_rows)
    horizontal = sum(col_lo <= c <= col_hi for c in orange_cols)
    return int(vertical + horizontal)


def _find_fragments(grid: np.ndarray, center_r: int, center_c: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cross_mask = np.zeros((h, w), dtype=bool)
    cross_mask[center_r, :] = True
    cross_mask[:, center_c] = True

    assert bool(np.all(np.isin(g[cross_mask], (GREY, ORANGE)))), "cross must be grey/orange only"
    assert not bool(np.any(np.isin(g[~cross_mask], (GREY, ORANGE)))), "grey/orange must stay on the cross"

    seen = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BLACK, GREY, ORANGE) or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            assert 1 <= len(coords) <= 6, "fragment area must be at most six"
            motion = _motion_info(arr, center_r, center_c)
            fragments.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    **motion,
                }
            )

    assert fragments, "expected at least one fragment"
    fragments.sort(key=lambda frag: (frag["bbox"][0], frag["bbox"][1], frag["color"]))
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide scattered quadrant fragments diagonally into the central cross and recolor them by marker parity.

    Input structure:
    - The background is black.
    - The exact center row and exact center column form a grey cross, and some of those cross cells may be orange markers.
    - Every other non-black cell belongs to a monochrome 4-connected fragment fully contained in one quadrant.
    - Fragments are small (area at most six) and stay off the cross.

    Rule:
    - Move each fragment along the diagonal that points from its quadrant toward the center:
      upper-left -> down-right, upper-right -> down-left, lower-left -> up-right, lower-right -> up-left.
    - Translate the whole fragment until some moved pixel becomes 4-neighbor adjacent to the central cross.
    - Take the inclusive row span covered by the fragment before/after the move and count orange markers on the vertical cross inside that span; do the same with the inclusive column span on the horizontal cross.
    - Recolor the moved fragment red when the total marker count is odd, otherwise blue, while leaving the cross unchanged.

    Key insight:
    - The quadrant determines the translation direction, but the fragment's final color comes from the parity of orange markers intercepted by its swept row/column ranges.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have a unique central row and column"

    center_r = h // 2
    center_c = w // 2
    fragments = _find_fragments(g, center_r, center_c)

    out = np.full_like(g, BLACK)
    out[center_r, :] = g[center_r, :]
    out[:, center_c] = g[:, center_c]

    occupied = np.zeros((h, w), dtype=bool)
    occupied[center_r, :] = True
    occupied[:, center_c] = True

    orange_rows = {int(r) for r in np.flatnonzero(g[:, center_c] == ORANGE)}
    orange_cols = {int(c) for c in np.flatnonzero(g[center_r, :] == ORANGE)}

    for frag in fragments:
        marker_count = _count_markers_for_spans(
            frag["row_span"],
            frag["col_span"],
            orange_rows,
            orange_cols,
        )
        color = RED if marker_count % 2 == 1 else BLUE
        shifted = frag["shifted"]
        assert not bool(occupied[shifted[:, 0], shifted[:, 1]].any()), "moved fragments must not overlap"
        occupied[shifted[:, 0], shifted[:, 1]] = True
        out[shifted[:, 0], shifted[:, 1]] = color

    return out


def _quadrant_bounds(
    quadrant: str,
    h: int,
    w: int,
    center_r: int,
    center_c: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int, int, int] | None:
    if quadrant == "UL":
        r0_lo, r0_hi = 0, center_r - h - 1
        c0_lo, c0_hi = 0, center_c - w - 1
    elif quadrant == "UR":
        r0_lo, r0_hi = 0, center_r - h - 1
        c0_lo, c0_hi = center_c + 2, grid_w - w
    elif quadrant == "LL":
        r0_lo, r0_hi = center_r + 2, grid_h - h
        c0_lo, c0_hi = 0, center_c - w - 1
    else:
        r0_lo, r0_hi = center_r + 2, grid_h - h
        c0_lo, c0_hi = center_c + 2, grid_w - w

    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    return (r0_lo, r0_hi, c0_lo, c0_hi)


def _place_fragment(
    quadrant: str,
    grid_h: int,
    grid_w: int,
    center_r: int,
    center_c: int,
    occupied_start: np.ndarray,
    occupied_final: np.ndarray,
    used_sweeps: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _random_shape()
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        bounds = _quadrant_bounds(quadrant, h, w, center_r, center_c, grid_h, grid_w)
        if bounds is None:
            continue

        r0_lo, r0_hi, c0_lo, c0_hi = bounds
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        coords = shape + np.array([r0, c0], dtype=int)
        if _touches_mask(coords, occupied_start):
            continue

        motion = _motion_info(coords, center_r, center_c)
        if motion["steps"] < 1:
            continue
        shifted = motion["shifted"]
        if _touches_mask(shifted, occupied_final):
            continue

        swept = _swept_cells(coords, motion["dr"], motion["dc"], motion["steps"])
        if swept & used_sweeps:
            continue

        color = random.choice(FRAGMENT_COLORS)
        occupied_start[coords[:, 0], coords[:, 1]] = True
        occupied_final[shifted[:, 0], shifted[:, 1]] = True
        used_sweeps.update(swept)
        return {
            "color": color,
            "coords": coords,
            "bbox": _bbox(coords),
            **motion,
        }
    return None


def _sample_markers(
    grid_h: int,
    grid_w: int,
    center_r: int,
    center_c: int,
) -> tuple[set[int], set[int]]:
    top_rows = list(range(0, center_r))
    bottom_rows = list(range(center_r + 1, grid_h))
    left_cols = list(range(0, center_c))
    right_cols = list(range(center_c + 1, grid_w))

    def choose(indices: list[int]) -> set[int]:
        max_count = max(1, min(len(indices) // 2 + 1, len(indices) - 1))
        count = random.randint(1, max_count)
        chosen = set(random.sample(indices, count))
        for idx in indices:
            if idx not in chosen and random.random() < 0.12:
                chosen.add(idx)
        return chosen

    orange_rows = choose(top_rows) | choose(bottom_rows)
    orange_cols = choose(left_cols) | choose(right_cols)
    return orange_rows, orange_cols


def _build_input(
    grid_h: int,
    grid_w: int,
    center_r: int,
    center_c: int,
    fragments: list[dict],
    orange_rows: set[int],
    orange_cols: set[int],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[center_r, :] = GREY
    grid[:, center_c] = GREY
    for r in orange_rows:
        grid[int(r), center_c] = ORANGE
    for c in orange_cols:
        grid[center_r, int(c)] = ORANGE
    for frag in fragments:
        coords = frag["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(frag["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randrange(GRID_MIN, GRID_MAX + 1, 2)
    grid_w = random.randrange(GRID_MIN, GRID_MAX + 1, 2)
    center_r = grid_h // 2
    center_c = grid_w // 2

    extra_prob = 0.55 if min(grid_h, grid_w) >= 23 else 0.25
    counts = {
        quadrant: 1 + int(random.random() < extra_prob)
        for quadrant in ("UL", "UR", "LL", "RR")
    }

    occupied_start = np.zeros((grid_h, grid_w), dtype=bool)
    occupied_final = np.zeros((grid_h, grid_w), dtype=bool)
    used_sweeps: set[tuple[int, int]] = set()
    fragments: list[dict] = []

    for quadrant in ("UL", "UR", "LL", "RR"):
        for _ in range(counts[quadrant]):
            frag = _place_fragment(
                quadrant,
                grid_h,
                grid_w,
                center_r,
                center_c,
                occupied_start,
                occupied_final,
                used_sweeps,
            )
            if frag is None:
                return None
            fragments.append(frag)

    stop_kinds = {frag["stop_kind"] for frag in fragments}
    if len(stop_kinds) < 2:
        return None

    for _ in range(MARKER_ATTEMPTS):
        orange_rows, orange_cols = _sample_markers(grid_h, grid_w, center_r, center_c)
        parities = [
            _count_markers_for_spans(frag["row_span"], frag["col_span"], orange_rows, orange_cols) % 2
            for frag in fragments
        ]
        if len(set(parities)) < 2:
            continue

        grid = _build_input(grid_h, grid_w, center_r, center_c, fragments, orange_rows, orange_cols)
        out = main(grid)
        if not (np.any(out == RED) and np.any(out == BLUE)):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

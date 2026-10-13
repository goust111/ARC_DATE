from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _monochrome_components(grid: np.ndarray, blocked: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in blocked or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for _color, cells in _monochrome_components(grid, blocked={BG, MAROON}):
        if not cells or int(grid[cells[0]]) != GREY:
            continue
        r0, r1, c0, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if len(cells) != 2 * (height + width) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        candidates.append((r0, r1, c0, c1))
    assert len(candidates) == 1, f"expected exactly one grey frame, got {candidates}"
    return candidates[0]


def _extract_strips(
    grid: np.ndarray, cloth_h: int, cloth_w: int
) -> tuple[list[tuple[int, int, bool]], list[tuple[int, int, bool]]]:
    horizontals: list[tuple[int, int, bool]] = []
    verticals: list[tuple[int, int, bool]] = []
    used_markers: set[tuple[int, int]] = set()

    comps = _monochrome_components(grid, blocked={BG, GREY, MAROON})
    for color, cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        if r0 == r1 and len(cells) == cloth_w and c1 - c0 + 1 == cloth_w:
            assert np.all(grid[r0, c0 : c1 + 1] == color)
            marked = c0 > 0 and int(grid[r0, c0 - 1]) == MAROON
            if marked:
                used_markers.add((r0, c0 - 1))
            horizontals.append((r0, int(color), marked))
            continue
        if c0 == c1 and len(cells) == cloth_h and r1 - r0 + 1 == cloth_h:
            assert np.all(grid[r0 : r1 + 1, c0] == color)
            marked = r0 > 0 and int(grid[r0 - 1, c0]) == MAROON
            if marked:
                used_markers.add((r0 - 1, c0))
            verticals.append((c0, int(color), marked))
            continue
        raise AssertionError(f"unrecognized strip component for color {color}: {cells}")

    all_markers = {tuple(map(int, pos)) for pos in np.argwhere(grid == MAROON)}
    assert used_markers == all_markers, f"unused or ambiguous markers: used={used_markers} all={all_markers}"

    horizontals.sort(key=lambda item: item[0])
    verticals.sort(key=lambda item: item[0])
    assert len(horizontals) == cloth_h, f"expected {cloth_h} horizontal strips, got {len(horizontals)}"
    assert len(verticals) == cloth_w, f"expected {cloth_w} vertical strips, got {len(verticals)}"
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave scattered horizontal and vertical strips into the central framed fabric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, r1, c0, c1 = _find_frame(g)
    cloth_h = r1 - r0 - 1
    cloth_w = c1 - c0 - 1
    horizontals, verticals = _extract_strips(g, cloth_h, cloth_w)

    out = np.full_like(g, BG)
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    for rr, (_src_row, h_color, h_marked) in enumerate(horizontals):
        for cc, (_src_col, v_color, v_marked) in enumerate(verticals):
            if h_marked and not v_marked:
                color = h_color
            elif v_marked and not h_marked:
                color = v_color
            else:
                color = h_color if (rr + cc) % 2 == 0 else v_color
            out[r0 + 1 + rr, c0 + 1 + cc] = color

    return out


def _mixed_flags(count: int, rng: random.Random) -> list[bool]:
    marked = rng.randint(1, count - 1)
    flags = [True] * marked + [False] * (count - marked)
    rng.shuffle(flags)
    return flags


def _jittered_center_start(total: int, span: int, rng: random.Random) -> int:
    base = (total - span) // 2
    lo = max(2, base - 2)
    hi = min(total - span - 2, base + 2)
    if lo > hi:
        lo = hi = max(1, min(total - span - 1, base))
    return rng.randint(lo, hi)


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r_lo = max(0, r - pad)
        r_hi = min(h, r + pad + 1)
        c_lo = max(0, c - pad)
        c_hi = min(w, c + pad + 1)
        if np.any(occupied[r_lo:r_hi, c_lo:c_hi]):
            return False
    return True


def _place_horizontal(
    grid: np.ndarray,
    occupied: np.ndarray,
    length: int,
    color: int,
    marked: bool,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    start_min = 1 if marked else 0
    start_max = w - length
    if start_min > start_max:
        return False
    for _ in range(PLACEMENT_ATTEMPTS):
        row = rng.randrange(h)
        col = rng.randint(start_min, start_max)
        cells = [(row, col + offset) for offset in range(length)]
        if marked:
            cells.append((row, col - 1))
        if not _can_place(occupied, cells):
            continue
        grid[row, col : col + length] = color
        if marked:
            grid[row, col - 1] = MAROON
        for rr, cc in cells:
            occupied[rr, cc] = True
        return True
    return False


def _place_vertical(
    grid: np.ndarray,
    occupied: np.ndarray,
    length: int,
    color: int,
    marked: bool,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    start_min = 1 if marked else 0
    start_max = h - length
    if start_min > start_max:
        return False
    for _ in range(PLACEMENT_ATTEMPTS):
        row = rng.randint(start_min, start_max)
        col = rng.randrange(w)
        cells = [(row + offset, col) for offset in range(length)]
        if marked:
            cells.append((row - 1, col))
        if not _can_place(occupied, cells):
            continue
        grid[row : row + length, col] = color
        if marked:
            grid[row - 1, col] = MAROON
        for rr, cc in cells:
            occupied[rr, cc] = True
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    cloth_h = rng.randint(2, 5)
    cloth_w = rng.randint(2, 5)

    row_colors = rng.sample(PALETTE, cloth_h)
    col_colors = rng.sample(PALETTE, cloth_w)
    row_flags = _mixed_flags(cloth_h, rng)
    col_flags = _mixed_flags(cloth_w, rng)

    grid_h = rng.randint(max(16, 3 * cloth_h + 8), max(22, 4 * cloth_h + 10))
    grid_w = rng.randint(max(16, 3 * cloth_w + 8), max(22, 4 * cloth_w + 10))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_h = cloth_h + 2
    frame_w = cloth_w + 2
    top = _jittered_center_start(grid_h, frame_h, rng)
    left = _jittered_center_start(grid_w, frame_w, rng)
    bottom = top + frame_h - 1
    right = left + frame_w - 1

    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[max(0, top - 1) : min(grid_h, bottom + 2), max(0, left - 1) : min(grid_w, right + 2)] = True

    items = [("h", color, marked) for color, marked in zip(row_colors, row_flags)]
    items.extend(("v", color, marked) for color, marked in zip(col_colors, col_flags))
    rng.shuffle(items)

    for kind, color, marked in items:
        if kind == "h":
            ok = _place_horizontal(grid, occupied, cloth_w, color, marked, rng)
        else:
            ok = _place_vertical(grid, occupied, cloth_h, color, marked, rng)
        if not ok:
            return None

    # Every branch in the weaving rule appears because both row/col flag sets are mixed.
    assert any(row_flags) and not all(row_flags)
    assert any(col_flags) and not all(col_flags)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        assert isinstance(out, np.ndarray)
        return grid
    raise RuntimeError("failed to generate a valid weaving puzzle input")

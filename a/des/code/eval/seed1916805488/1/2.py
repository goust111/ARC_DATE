import random
from typing import List, Tuple

import numpy as np


BG = 8
SEP = 9
STAMP_COLORS = [1, 2, 3, 4, 5, 6, 7]
DOT_COLORS = [0, 1, 2, 3, 4, 5, 6, 7]
CORNERS = ["tl", "tr", "bl", "br"]
CORNER_TO_OFFSET = {
    "tl": (0, 0),
    "tr": (0, 2),
    "bl": (2, 0),
    "br": (2, 2),
}

GEN_ATTEMPTS = 400
ROW_OPTIONS = [2, 3]
COL_OPTIONS = [2, 3]
REGION_H_RANGE = (7, 11)
REGION_W_RANGE = (7, 11)
DOT_COUNT_OPTIONS = [1, 2, 3]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _find_separator_color(g, bg)
    row_bounds, col_bounds = _find_regions(g, sep)

    out = g.copy()
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            region = g[r0:r1, c0:c1]
            stamp = _find_stamp(region, bg, sep)
            dots = _find_dots(region, bg, sep, stamp)
            assert dots, "each region must contain at least one ink dot"

            center_r = stamp[0] + 1
            center_c = stamp[1] + 1
            chosen = min(
                dots,
                key=lambda rc: ((rc[0] - center_r) ** 2 + (rc[1] - center_c) ** 2, rc[0], rc[1]),
            )
            target_corner = _quadrant_from_relative(chosen[0] - center_r, chosen[1] - center_c)
            fill_color = int(region[chosen])
            stamp_color = int(stamp[3])

            region_out = out[r0:r1, c0:c1]
            for dr, dc in dots:
                region_out[dr, dc] = bg

            sr, sc = stamp[0], stamp[1]
            region_out[sr : sr + 3, sc : sc + 3] = stamp_color
            miss_r, miss_c = CORNER_TO_OFFSET[target_corner]
            region_out[sr + miss_r, sc + miss_c] = fill_color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    candidates = []
    for color in colors:
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if full_rows and full_cols:
            candidates.append((len(full_rows) + len(full_cols), color))
    assert candidates, "failed to identify separator color"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _find_regions(grid: np.ndarray, sep: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    row_bounds = _intervals_between_lines(h, sep_rows)
    col_bounds = _intervals_between_lines(w, sep_cols)
    assert row_bounds and col_bounds
    return row_bounds, col_bounds


def _intervals_between_lines(length: int, lines: List[int]) -> List[Tuple[int, int]]:
    bounds = [-1] + sorted(lines) + [length]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            out.append((a + 1, b))
    return out


def _find_stamp(region: np.ndarray, bg: int, sep: int) -> Tuple[int, int, str, int]:
    h, w = region.shape
    found = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = region[r : r + 3, c : c + 3]
            vals = [int(x) for x in block.reshape(-1)]
            non_bg = [v for v in vals if v != bg and v != sep]
            if len(non_bg) != 8:
                continue
            color = non_bg[0]
            if any(v != color for v in non_bg):
                continue
            bg_cells = [(rr, cc) for rr in range(3) for cc in range(3) if int(block[rr, cc]) == bg]
            if len(bg_cells) != 1:
                continue
            miss = bg_cells[0]
            if miss not in CORNER_TO_OFFSET.values():
                continue
            corner = next(name for name, offset in CORNER_TO_OFFSET.items() if offset == miss)
            found.append((r, c, corner, color))
    assert len(found) == 1, f"expected exactly one stamp per region, got {len(found)}"
    return found[0]


def _find_dots(region: np.ndarray, bg: int, sep: int, stamp: Tuple[int, int, str, int]) -> List[Tuple[int, int]]:
    h, w = region.shape
    sr, sc, _, stamp_color = stamp
    dots = []
    for r in range(h):
        for c in range(w):
            val = int(region[r, c])
            if val in (bg, sep, stamp_color):
                continue
            if sr <= r < sr + 3 and sc <= c < sc + 3:
                continue
            dots.append((r, c))
    return dots


def _quadrant_from_relative(dr: int, dc: int) -> str:
    vertical = "t" if dr < 0 else "b"
    horizontal = "l" if dc < 0 else "r"
    return vertical + horizontal


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice(ROW_OPTIONS)
    n_cols = random.choice(COL_OPTIONS)
    region_heights = [random.randint(*REGION_H_RANGE) for _ in range(n_rows)]
    region_widths = [random.randint(*REGION_W_RANGE) for _ in range(n_cols)]
    h = sum(region_heights) + (n_rows - 1)
    w = sum(region_widths) + (n_cols - 1)
    grid = np.full((h, w), BG, dtype=int)

    row_starts = []
    r = 0
    for i, rh in enumerate(region_heights):
        row_starts.append(r)
        r += rh
        if i != n_rows - 1:
            grid[r, :] = SEP
            r += 1

    col_starts = []
    c = 0
    for j, cw in enumerate(region_widths):
        col_starts.append(c)
        c += cw
        if j != n_cols - 1:
            grid[:, c] = SEP
            c += 1

    stamp_cycle = CORNERS.copy()
    random.shuffle(stamp_cycle)
    target_cycle = CORNERS.copy()
    random.shuffle(target_cycle)
    cell_index = 0

    for i, rh in enumerate(region_heights):
        for j, cw in enumerate(region_widths):
            r0 = row_starts[i]
            c0 = col_starts[j]
            region = grid[r0 : r0 + rh, c0 : c0 + cw]
            initial_corner = stamp_cycle[cell_index % len(stamp_cycle)]
            target_corner = target_cycle[cell_index % len(target_cycle)]
            if not _populate_region(region, initial_corner, target_corner):
                return None
            cell_index += 1

    try:
        main(grid)
    except Exception:
        return None
    return grid


def _populate_region(region: np.ndarray, initial_corner: str, target_corner: str) -> bool:
    h, w = region.shape
    stamp_color = random.choice(STAMP_COLORS)
    dot_palette = [c for c in DOT_COLORS if c not in (BG, SEP, stamp_color)]
    if len(dot_palette) < 3:
        return False

    placements = []
    for sr in range(h - 2):
        for sc in range(w - 2):
            if _cells_in_quadrant(h, w, sr + 1, sc + 1, target_corner):
                placements.append((sr, sc))
    random.shuffle(placements)
    if not placements:
        return False

    for sr, sc in placements:
        region[:, :] = BG
        region[sr : sr + 3, sc : sc + 3] = stamp_color
        miss_r, miss_c = CORNER_TO_OFFSET[initial_corner]
        region[sr + miss_r, sc + miss_c] = BG

        target_cells = _cells_in_quadrant(h, w, sr + 1, sc + 1, target_corner)
        nearest = random.choice(target_cells)
        used = {nearest}
        region[nearest] = random.choice(dot_palette)

        dot_count = random.choice(DOT_COUNT_OPTIONS)
        extras_needed = dot_count - 1
        if not _place_extra_dots(region, sr, sc, nearest, used, dot_palette, extras_needed):
            continue
        if not _validate_region(region, (sr, sc), target_corner):
            continue
        return True
    return False


def _cells_in_quadrant(h: int, w: int, cr: int, cc: int, corner: str) -> List[Tuple[int, int]]:
    out = []
    row_range = range(0, cr) if corner[0] == "t" else range(cr + 1, h)
    col_range = range(0, cc) if corner[1] == "l" else range(cc + 1, w)
    for r in row_range:
        for c in col_range:
            if cr - 1 <= r <= cr + 1 and cc - 1 <= c <= cc + 1:
                continue
            out.append((r, c))
    return out


def _place_extra_dots(
    region: np.ndarray,
    sr: int,
    sc: int,
    nearest: Tuple[int, int],
    used: set,
    dot_palette: List[int],
    extras_needed: int,
) -> bool:
    h, w = region.shape
    center = (sr + 1, sc + 1)
    nearest_color = int(region[nearest])
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in used:
                continue
            if sr <= r < sr + 3 and sc <= c < sc + 3:
                continue
            if r == center[0] or c == center[1]:
                continue
            d2 = _dist2((r, c), center)
            if d2 < _dist2(nearest, center):
                continue
            if d2 == _dist2(nearest, center) and (r, c) < nearest:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    if len(candidates) < extras_needed:
        return False

    for cell in candidates[:extras_needed]:
        used.add(cell)
        color_choices = [c for c in dot_palette if c != nearest_color]
        region[cell] = random.choice(color_choices or dot_palette)
    return True


def _dist2(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _validate_region(region: np.ndarray, stamp_pos: Tuple[int, int], target_corner: str) -> bool:
    sr, sc = stamp_pos
    stamp = _find_stamp(region, BG, SEP)
    if stamp[:2] != (sr, sc):
        return False
    dots = _find_dots(region, BG, SEP, stamp)
    if not dots:
        return False

    center = (sr + 1, sc + 1)
    chosen = min(dots, key=lambda rc: (_dist2(rc, center), rc[0], rc[1]))
    if chosen[0] == center[0] or chosen[1] == center[1]:
        return False
    if _quadrant_from_relative(chosen[0] - center[0], chosen[1] - center[1]) != target_corner:
        return False

    chosen_d2 = _dist2(chosen, center)
    for dot in dots:
        if dot == chosen:
            continue
        d2 = _dist2(dot, center)
        if d2 < chosen_d2:
            return False
        if d2 == chosen_d2 and dot < chosen:
            return False
    return True

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border(r0: int, c0: int) -> list[tuple[int, int]]:
    return [
        (r0 + dr, c0 + dc)
        for dr in range(3)
        for dc in range(3)
        if not (dr == 1 and dc == 1)
    ]


def _detect_scaffold(grid: np.ndarray):
    comps = _components8(grid == GREY)
    assert comps, "expected a grey scaffold"

    frames = []
    for comp in comps:
        assert len(comp) == 8, "every grey component must be a hollow 3x3 frame"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "frame bbox must be 3x3"
        border = _frame_border(r0, c0)
        assert set(comp) == set(border), "frame must be a hollow 3x3 border"
        assert int(grid[r0 + 1, c0 + 1]) == BLACK, "frame center must be empty"
        frames.append(
            {
                "origin": (r0, c0),
                "center": (r0 + 1, c0 + 1),
                "border": border,
            }
        )

    center_rows = sorted({frame["center"][0] for frame in frames})
    center_cols = sorted({frame["center"][1] for frame in frames})
    assert len(center_rows) >= 2 and len(center_cols) >= 2, "scaffold must be a grid"
    assert len(frames) == len(center_rows) * len(center_cols), "scaffold grid must be complete"

    row_steps = {b - a for a, b in zip(center_rows, center_rows[1:])}
    col_steps = {b - a for a, b in zip(center_cols, center_cols[1:])}
    assert row_steps == {4} and col_steps == {4}, "frame centers must lie on a 4-spaced lattice"

    by_center = {(frame["center"][0], frame["center"][1]): frame for frame in frames}
    for rr in center_rows:
        for cc in center_cols:
            assert (rr, cc) in by_center, "missing scaffold cell"

    scaffold_top = min(frame["origin"][0] for frame in frames)
    return center_rows, center_cols, by_center, scaffold_top


def _winner(colors: list[int]) -> int:
    counts = Counter(colors)
    best = max(counts.values())
    for color in colors:
        if counts[color] == best:
            return int(color)
    raise AssertionError("unreachable")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the loose pixels into scaffold centers, then recolor every fully occupied scaffold row.

    Input structure:
    - The grid has a black background and a lower scaffold made of separate grey hollow 3x3 frames on a regular lattice.
    - Each frame has a single black center cell, and loose colored pixels appear only above the scaffold.
    - Every loose pixel lies in the same column as some scaffold center.

    Rule:
    - Detect the scaffold frames and list their center coordinates by row and by column.
    - For each scaffold column, sort the loose pixels in that column from bottom to top and let them settle into the column's empty centers from top to bottom.
    - If a scaffold row ends with all of its centers occupied, recolor every grey border pixel of that row's frames with the most frequent center color in that row; break ties by the leftmost occupied center color in the row.
    - Clear the entire region above the scaffold to black.

    Key insight:
    - Gravity does not send pixels to the floor; it converts vertical order inside each scaffold column into row occupancy, and that occupancy then decides which scaffold rows get recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center_rows, center_cols, by_center, scaffold_top = _detect_scaffold(g)
    center_col_set = set(center_cols)

    out = np.where(g == GREY, GREY, BLACK)
    falling_by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        assert cc in center_col_set, "falling pixel must align with a scaffold column"
        assert rr < scaffold_top, "falling pixels must start above the scaffold"
        falling_by_col[cc].append((rr, color))

    for cc in center_cols:
        pixels = sorted(falling_by_col.get(cc, []), key=lambda item: item[0], reverse=True)
        for idx, (_row, color) in enumerate(pixels):
            if idx >= len(center_rows):
                break
            out[center_rows[idx], cc] = int(color)

    for rr in center_rows:
        row_colors = [int(out[rr, cc]) for cc in center_cols]
        if all(color != BLACK for color in row_colors):
            fill = _winner(row_colors)
            for cc in center_cols:
                for br, bc in by_center[(rr, cc)]["border"]:
                    out[br, bc] = int(fill)

    return out


def _draw_scaffold(
    grid: np.ndarray, top: int, left: int, nrows: int, ncols: int
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    borders: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(nrows):
        for j in range(ncols):
            r0 = top + 4 * i
            c0 = left + 4 * j
            border = _frame_border(r0, c0)
            for r, c in border:
                grid[r, c] = GREY
            borders[(i, j)] = border
    return borders


def _sample_counts(nrows: int, ncols: int, total_pixels: int) -> list[int]:
    max_active = min(nrows, total_pixels // ncols)
    depth_choices = list(range(1, max_active + 1))
    weights = [4 if d < nrows else 2 for d in depth_choices]
    active_depth = random.choices(depth_choices, weights=weights, k=1)[0]

    counts = [active_depth] * ncols
    remaining = total_pixels - active_depth * ncols
    while remaining > 0:
        candidates = [idx for idx, count in enumerate(counts) if count < nrows]
        idx = random.choice(candidates)
        counts[idx] += 1
        remaining -= 1
    return counts


def _tie_pattern(ncols: int, palette: list[int]) -> list[int]:
    if ncols == 3:
        return random.sample(palette, 3)
    a, b = random.sample(palette, 2)
    return [a, b, a, b] if random.choice([False, True]) else [a, b, b, a]


def _majority_pattern(ncols: int, palette: list[int]) -> list[int]:
    winner = random.choice(palette)
    row = [winner] * ncols
    other_choices = [color for color in palette if color != winner]
    if ncols == 3:
        pos = random.randrange(ncols)
        row[pos] = random.choice(other_choices)
        return row

    mode_choices = ["3-1", "2-1-1"] if len(other_choices) >= 2 else ["3-1"]
    mode = random.choice(mode_choices)
    if mode == "3-1":
        pos = random.randrange(ncols)
        row[pos] = random.choice(other_choices)
        return row

    positions = random.sample(range(ncols), 2)
    extras = random.sample(other_choices, 2)
    row[positions[0]] = extras[0]
    row[positions[1]] = extras[1]
    return row


def _sample_nonadjacent_rows(limit: int, count: int) -> list[int]:
    candidates = [
        list(combo)
        for combo in combinations(range(limit), count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    assert candidates, "not enough vertical room for non-adjacent pixels"
    return random.choice(candidates)


def _build_expected(
    shape: tuple[int, int],
    top: int,
    left: int,
    occupancy: np.ndarray,
) -> np.ndarray:
    nrows, ncols = occupancy.shape
    out = np.zeros(shape, dtype=int)
    borders = _draw_scaffold(out, top, left, nrows, ncols)

    center_rows = [top + 1 + 4 * i for i in range(nrows)]
    center_cols = [left + 1 + 4 * j for j in range(ncols)]
    for i in range(nrows):
        for j in range(ncols):
            if occupancy[i, j] != BLACK:
                out[center_rows[i], center_cols[j]] = int(occupancy[i, j])

    for i in range(nrows):
        row_colors = [int(occupancy[i, j]) for j in range(ncols)]
        if all(color != BLACK for color in row_colors):
            fill = _winner(row_colors)
            for j in range(ncols):
                for r, c in borders[(i, j)]:
                    out[r, c] = int(fill)
    return out


def _validate_plan(
    inp: np.ndarray,
    expected: np.ndarray,
    occupancy: np.ndarray,
    top: int,
    left: int,
    counts: list[int],
    tie_row: int | None,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    nrows, ncols = occupancy.shape
    center_rows = [top + 1 + 4 * i for i in range(nrows)]
    center_cols = [left + 1 + 4 * j for j in range(ncols)]

    colors = [int(inp[r, c]) for r, c in np.argwhere((inp != BLACK) & (inp != GREY))]
    if not (5 <= len(colors) <= 8):
        return False
    if not (2 <= len(set(colors)) <= 3):
        return False

    colored_pts = [(int(r), int(c)) for r, c in np.argwhere((inp != BLACK) & (inp != GREY))]
    for idx, (r0, c0) in enumerate(colored_pts):
        if r0 >= top or c0 not in center_cols:
            return False
        for r1, c1 in colored_pts[idx + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    active_rows = 0
    for i in range(nrows):
        row = [int(occupancy[i, j]) for j in range(ncols)]
        if all(color != BLACK for color in row):
            active_rows += 1
    if active_rows < 1:
        return False

    if tie_row is not None:
        row = [int(occupancy[tie_row, j]) for j in range(ncols)]
        counts_row = Counter(row)
        if len([color for color, cnt in counts_row.items() if cnt == max(counts_row.values())]) < 2:
            return False

    for j, cc in enumerate(center_cols):
        col_values = [int(occupancy[i, j]) for i in range(nrows) if occupancy[i, j] != BLACK]
        if len(col_values) != counts[j]:
            return False
        out_values = [int(expected[rr, cc]) for rr in center_rows if expected[rr, cc] != BLACK]
        if out_values != col_values:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    nrows = random.choice([3, 4])
    ncols = random.choice([3, 4])
    scaffold_h = 4 * nrows - 1
    scaffold_w = 4 * ncols - 1

    top_min = max(2 * nrows + 1, 7)
    top_max = min(12, 30 - scaffold_h)
    if top_min > top_max:
        return None
    top = random.randint(top_min, top_max)
    left = random.randint(0, 30 - scaffold_w)
    bottom = random.randint(0, min(4, 30 - top - scaffold_h))
    right = random.randint(0, min(6, 30 - left - scaffold_w))
    height = top + scaffold_h + bottom
    width = left + scaffold_w + right

    total_pixels = random.randint(max(5, ncols), 8)
    counts = _sample_counts(nrows, ncols, total_pixels)

    num_colors = random.choice([2, 3])
    palette = random.sample(list(PALETTE), num_colors)
    occupancy = np.zeros((nrows, ncols), dtype=int)

    active_depth = min(counts)
    tie_possible = (ncols == 4 and num_colors >= 2) or (ncols == 3 and num_colors >= 3)
    tie_row = None
    if tie_possible and random.random() < 0.45:
        tie_row = random.randrange(active_depth)

    for i in range(active_depth):
        if i == tie_row:
            pattern = _tie_pattern(ncols, palette)
        else:
            pattern = _majority_pattern(ncols, palette)
        occupancy[i, :] = np.array(pattern, dtype=int)

    for j, count in enumerate(counts):
        for i in range(active_depth, count):
            occupancy[i, j] = int(random.choice(palette))

    used = set(int(v) for v in occupancy.flatten() if v != BLACK)
    if len(used) < 2:
        return None
    inp = np.zeros((height, width), dtype=int)
    _draw_scaffold(inp, top, left, nrows, ncols)
    center_cols = [left + 1 + 4 * j for j in range(ncols)]

    for j, cc in enumerate(center_cols):
        col_colors = [int(occupancy[i, j]) for i in range(nrows) if occupancy[i, j] != BLACK]
        rows = _sample_nonadjacent_rows(top, len(col_colors))
        input_colors = list(reversed(col_colors))
        for rr, color in zip(rows, input_colors):
            inp[rr, cc] = int(color)

    expected = _build_expected(inp.shape, top, left, occupancy)
    if not _validate_plan(inp, expected, occupancy, top, left, counts, tie_row):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

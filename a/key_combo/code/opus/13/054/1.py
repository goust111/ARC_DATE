from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

HOLE_TO_COLOR = {
    0: BLACK,
    1: BLUE,
    2: RED,
    3: GREEN,
}

INPUT_COLORS = [BLUE, RED, GREEN, YELLOW]
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_signature(mask: np.ndarray) -> tuple[str, ...]:
    cropped = _crop_mask(mask)
    return tuple("".join("1" if v else "0" for v in row) for row in cropped)


def _count_holes_in_mask(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    bg_mask = ~padded
    h, w = bg_mask.shape
    holes = 0
    for comp in _components(bg_mask):
        rr = comp[:, 0]
        cc = comp[:, 1]
        touches_outer = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_outer:
            holes += 1
    return holes


def _count_enclosed_black_regions(cell: np.ndarray) -> int:
    black = cell == BLACK
    h, w = cell.shape
    holes = 0
    for comp in _components(black):
        rr = comp[:, 0]
        cc = comp[:, 1]
        touches_boundary = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_boundary:
            holes += 1
    return holes


def _separator_groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _cell_spans(groups: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    assert groups and groups[0][0] == 0 and groups[-1][1] == limit - 1
    spans: list[tuple[int, int]] = []
    for i in range(len(groups) - 1):
        start = groups[i][1] + 1
        end = groups[i + 1][0] - 1
        assert start <= end
        spans.append((start, end))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each lattice cell to a color determined only by the number of enclosed black holes in its foreground shape.

    Input structure:
    - Full-width grey rows and full-height grey columns, including the outer border, partition the grid into a 3x4 lattice of rectangular cells.
    - Each cell contains exactly one 4-connected foreground shape in a single non-black, non-grey color on a black background.
    - Some shapes are solid while others enclose one to three black cavities inside that cell.

    Rule:
    - Detect the grey separator rows and columns, then treat each rectangle between consecutive separators as one cell.
    - Inside each cell, find the unique foreground component and count the 4-connected black regions that do not touch the cell boundary.
    - Map hole counts 0, 1, 2, 3 to output colors black, blue, red, green respectively.
    - Return the resulting 3x4 coarse grid in the same row/column order as the input lattice.

    Key insight:
    - The visible object color and silhouette are distractors; only the topological hole count inside each lattice cell determines the compressed output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_groups = _separator_groups(full_rows)
    col_groups = _separator_groups(full_cols)
    row_spans = _cell_spans(row_groups, g.shape[0])
    col_spans = _cell_spans(col_groups, g.shape[1])
    assert len(row_spans) == 3 and len(col_spans) == 4

    out = np.zeros((3, 4), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            fg_mask = (cell != BLACK) & (cell != GREY)
            fg_comps = _components(fg_mask)
            assert len(fg_comps) == 1
            fg_colors = set(map(int, cell[fg_mask]))
            assert len(fg_colors) == 1 and fg_colors <= set(INPUT_COLORS)

            holes = _count_enclosed_black_regions(cell)
            assert holes in HOLE_TO_COLOR
            out[rr, cc] = HOLE_TO_COLOR[holes]

    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_TEMPLATES = {
    0: [
        _mask_from_rows(["1110", "1110", "0111", "0011"]),
        _mask_from_rows(["1100", "1110", "0111", "0011", "0010"]),
        _mask_from_rows(["010", "111", "010", "111", "010"]),
        _mask_from_rows(["11100", "01110", "00111", "00110"]),
        _mask_from_rows(["01110", "11110", "01100", "00110"]),
        _mask_from_rows(["11111", "11111", "11111", "11111", "11110", "11000"]),
        _mask_from_rows(["111111", "111111", "111110", "111100", "111000"]),
    ],
    1: [
        _mask_from_rows(["1111", "1001", "1001", "1111"]),
        _mask_from_rows(["00100", "11111", "10001", "10001", "11111"]),
        _mask_from_rows(["1111", "1001", "1001", "1111", "0100"]),
        _mask_from_rows(["11111", "10001", "10001", "11111", "00100"]),
        _mask_from_rows(["111111", "100001", "100001", "100001", "100001", "111111"]),
        _mask_from_rows(["001000", "111111", "100001", "100001", "100001", "111111"]),
    ],
    2: [
        _mask_from_rows(["11111", "10101", "10101", "10101", "11111"]),
        _mask_from_rows(["111111", "100101", "100101", "100101", "111111"]),
        _mask_from_rows(["11111", "10001", "11111", "10001", "10001", "11111"]),
        _mask_from_rows(["111111", "101001", "101001", "101001", "101001", "111111"]),
    ],
    3: [
        _mask_from_rows(["11111", "10001", "11111", "10001", "11111", "10001", "11111"]),
        _mask_from_rows(["111111", "101001", "101111", "101001", "101001", "111111"]),
        _mask_from_rows(["111111", "101001", "101001", "101111", "101001", "111111"]),
        _mask_from_rows(["111111", "100001", "111111", "100001", "111111", "100001", "111111"]),
    ],
}


def _template_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: dict[tuple[str, ...], np.ndarray] = {}
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            sig = _mask_signature(variant)
            seen.setdefault(sig, variant.copy())
    return list(seen.values())


TEMPLATE_VARIANTS = {
    count: [variant for base in bases for variant in _template_variants(base)]
    for count, bases in BASE_TEMPLATES.items()
}


def _candidate_masks(hole_count: int, cell_h: int, cell_w: int) -> list[tuple[tuple[str, ...], np.ndarray]]:
    cands: list[tuple[tuple[str, ...], np.ndarray]] = []
    for mask in TEMPLATE_VARIANTS[hole_count]:
        mh, mw = mask.shape
        if mh <= cell_h and mw <= cell_w and (mh < cell_h or mw < cell_w):
            cands.append((_mask_signature(mask), mask))
    unique: dict[tuple[str, ...], np.ndarray] = {}
    for sig, mask in cands:
        unique.setdefault(sig, mask)
    return list(unique.items())


def _pick_unique_masks(
    rng: random.Random,
    positions: list[tuple[int, int]],
    cell_dims: dict[tuple[int, int], tuple[int, int]],
    hole_count: int,
) -> dict[tuple[int, int], np.ndarray] | None:
    candidates = {
        pos: _candidate_masks(hole_count, *cell_dims[pos])
        for pos in positions
    }
    if any(not opts for opts in candidates.values()):
        return None

    ordered = sorted(positions, key=lambda pos: len(candidates[pos]))
    chosen: dict[tuple[int, int], np.ndarray] = {}
    used: set[tuple[str, ...]] = set()

    def dfs(idx: int) -> bool:
        if idx == len(ordered):
            return True
        pos = ordered[idx]
        options = candidates[pos][:]
        rng.shuffle(options)
        for sig, mask in options:
            if sig in used:
                continue
            used.add(sig)
            chosen[pos] = mask
            if dfs(idx + 1):
                return True
            used.remove(sig)
            del chosen[pos]
        return False

    return chosen if dfs(0) else None


def _place_mask_in_cell(
    rng: random.Random,
    cell_h: int,
    cell_w: int,
    mask: np.ndarray,
) -> np.ndarray:
    placed = np.zeros((cell_h, cell_w), dtype=bool)
    mh, mw = mask.shape
    top = rng.randint(0, cell_h - mh)
    left = rng.randint(0, cell_w - mw)
    placed[top : top + mh, left : left + mw] = mask
    return placed


def _eligible_positions_for_count3(cell_dims: dict[tuple[int, int], tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for pos, (h, w) in cell_dims.items():
        if _candidate_masks(3, h, w):
            out.append(pos)
    return out


def _build_color_plan(rng: random.Random) -> dict[int, list[int]]:
    omitted = rng.sample(INPUT_COLORS, k=4)
    color_plan: dict[int, list[int]] = {}
    for hole_count in range(4):
        colors = [color for color in INPUT_COLORS if color != omitted[hole_count]]
        rng.shuffle(colors)
        color_plan[hole_count] = colors
    return color_plan


def _make_counts_layout(
    rng: random.Random,
    cell_dims: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    positions = list(cell_dims)
    eligible3 = _eligible_positions_for_count3(cell_dims)
    if len(eligible3) < 3:
        return None

    count3_positions = rng.sample(eligible3, k=3)
    remaining = [pos for pos in positions if pos not in count3_positions]
    rng.shuffle(remaining)

    counts: dict[tuple[int, int], int] = {pos: 3 for pos in count3_positions}
    for hole_count in (2, 1):
        chosen = remaining[:3]
        remaining = remaining[3:]
        for pos in chosen:
            counts[pos] = hole_count
    for pos in remaining:
        counts[pos] = 0
    return counts


def _validate_grid(
    grid: np.ndarray,
    expected: np.ndarray,
    counts: dict[tuple[int, int], int],
    shapes: dict[tuple[int, int], np.ndarray],
    colors: dict[tuple[int, int], int],
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(_separator_groups(full_rows)) != 4 or len(_separator_groups(full_cols)) != 5:
        return False

    for hole_count in range(4):
        sigs = {_mask_signature(shapes[pos]) for pos, count in counts.items() if count == hole_count}
        group_colors = {colors[pos] for pos, count in counts.items() if count == hole_count}
        if len(sigs) != 3 or len(group_colors) != 3:
            return False

    color_to_counts: dict[int, set[int]] = {color: set() for color in INPUT_COLORS}
    for pos, count in counts.items():
        color_to_counts[colors[pos]].add(count)
    if any(len(v) < 2 for v in color_to_counts.values()):
        return False

    return set(map(int, expected.flat)) == {BLACK, BLUE, RED, GREEN}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_heights = [rng.choice([7, 8]) for _ in range(3)]
    col_widths = [rng.choice([5, 6]) for _ in range(4)]
    if 8 not in row_heights and col_widths.count(6) < 3:
        return None

    h = sum(row_heights) + 4
    w = sum(col_widths) + 5
    if h > 30 or w > 30:
        return None

    cell_dims = {
        (r, c): (row_heights[r], col_widths[c])
        for r in range(3)
        for c in range(4)
    }
    counts = _make_counts_layout(rng, cell_dims)
    if counts is None:
        return None

    shapes: dict[tuple[int, int], np.ndarray] = {}
    for hole_count in range(4):
        positions = [pos for pos, count in counts.items() if count == hole_count]
        picked = _pick_unique_masks(rng, positions, cell_dims, hole_count)
        if picked is None:
            return None
        shapes.update(picked)

    color_plan = _build_color_plan(rng)
    colors: dict[tuple[int, int], int] = {}
    for hole_count in range(4):
        positions = [pos for pos, count in counts.items() if count == hole_count]
        rng.shuffle(positions)
        for pos, color in zip(positions, color_plan[hole_count]):
            colors[pos] = color

    grid = np.zeros((h, w), dtype=int)
    row_lines = [0]
    for size in row_heights:
        row_lines.append(row_lines[-1] + size + 1)
    col_lines = [0]
    for size in col_widths:
        col_lines.append(col_lines[-1] + size + 1)

    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    expected = np.zeros((3, 4), dtype=int)
    for r in range(3):
        for c in range(4):
            cell_h, cell_w = cell_dims[(r, c)]
            mask = _place_mask_in_cell(rng, cell_h, cell_w, shapes[(r, c)])
            if _count_holes_in_mask(mask) != counts[(r, c)]:
                return None
            if len(_components(mask)) != 1:
                return None

            color = colors[(r, c)]
            r0 = row_lines[r] + 1
            c0 = col_lines[c] + 1
            cell_view = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            cell_view[mask] = color
            if _count_enclosed_black_regions(cell_view) != counts[(r, c)]:
                return None
            expected[r, c] = HOLE_TO_COLOR[counts[(r, c)]]
            shapes[(r, c)] = mask

    if not _validate_grid(grid, expected, counts, shapes, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-hole-compression input")

from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GRAY = 5
ORANGE = 7
GROUP_COLORS = (2, 3, 4, 6, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        if not np.all(grid[:, c] == GRAY):
            continue
        if np.any(grid[:, :c] != BG) and np.any(grid[:, c + 1 :] == GRAY):
            candidates.append(c)
    assert len(candidates) == 1, "expected a unique full-height gray separator"
    return candidates[0]


def _target_crop(grid: np.ndarray, separator_col: int) -> np.ndarray:
    right = grid[:, separator_col + 1 :]
    gray_coords = np.argwhere(right == GRAY)
    assert gray_coords.size > 0, "expected a gray-framed target grid"
    r0, c0, r1, c1 = _bbox(gray_coords)
    return right[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grid_lines(target: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = target.shape
    line_rows = [r for r in range(h) if np.all(target[r, :] == GRAY)]
    line_cols = [c for c in range(w) if np.all(target[:, c] == GRAY)]
    assert len(line_rows) >= 2 and len(line_cols) >= 2

    cell_heights = [line_rows[i + 1] - line_rows[i] - 1 for i in range(len(line_rows) - 1)]
    cell_widths = [line_cols[i + 1] - line_cols[i] - 1 for i in range(len(line_cols) - 1)]
    assert min(cell_heights) > 0 and min(cell_widths) > 0
    assert len(set(cell_heights)) == 1 and len(set(cell_widths)) == 1
    return line_rows, line_cols


def _cell_slice(line_rows: list[int], line_cols: list[int], row_idx: int, col_idx: int) -> tuple[slice, slice]:
    return (
        slice(line_rows[row_idx] + 1, line_rows[row_idx + 1]),
        slice(line_cols[col_idx] + 1, line_cols[col_idx + 1]),
    )


def _blue_cells(target: np.ndarray, line_rows: list[int], line_cols: list[int]) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            rs, cs = _cell_slice(line_rows, line_cols, row_idx, col_idx)
            interior = target[rs, cs]
            colors = set(int(v) for v in np.unique(interior)) - {BG}
            assert colors <= {BLUE}, "input target cells should be black or blue"
            if BLUE in colors:
                markers.append((row_idx, col_idx))
    assert markers, "expected at least one blue target cell"
    return markers


def _ordered_group(group_grid: np.ndarray, comp: np.ndarray) -> tuple[tuple[int, int], list[int], int]:
    pts = [tuple(map(int, pt)) for pt in comp]
    pt_set = set(pts)
    degree = {
        pt: sum((nbr in pt_set) for nbr in _neighbors4(*pt))
        for pt in pt_set
    }
    endpoints = sorted(pt for pt, deg in degree.items() if deg == 1)
    assert len(endpoints) == 2, "each instruction group must be a single non-branching path"
    assert int(group_grid[endpoints[0]]) != ORANGE and int(group_grid[endpoints[1]]) != ORANGE

    sequence: list[int] = []
    color_positions: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    prev: tuple[int, int] | None = None
    cur = endpoints[0]

    while True:
        visited.add(cur)
        color = int(group_grid[cur])
        if color != ORANGE:
            sequence.append(color)
            color_positions.append(cur)

        next_pts = [nbr for nbr in _neighbors4(*cur) if nbr in pt_set and nbr != prev]
        if not next_pts:
            break
        assert len(next_pts) == 1
        prev, cur = cur, next_pts[0]

    assert len(visited) == len(pt_set), "group path should be fully traversed from endpoint to endpoint"
    assert len(sequence) >= 2, "each group should contain multiple colored blocks"
    assert len(set(sequence)) == len(sequence), "group colors should be unique"
    blue_indices = [i for i, color in enumerate(sequence) if color == BLUE]
    assert len(blue_indices) == 1, "each group must contain exactly one blue block"
    blue_idx = blue_indices[0]
    return color_positions[blue_idx], sequence, blue_idx


def _extract_groups(left: np.ndarray) -> list[tuple[list[int], int]]:
    mask = (left != BG) & (left != GRAY)
    groups = []
    for comp in _components(mask):
        blue_pos, sequence, blue_idx = _ordered_group(left, comp)
        groups.append((blue_pos, sequence, blue_idx))
    groups.sort(key=lambda item: item[0])
    assert groups, "expected at least one instruction group"
    return [(sequence, blue_idx) for _blue_pos, sequence, blue_idx in groups]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_col = _find_separator(g)
    target = _target_crop(g, separator_col)
    line_rows, line_cols = _grid_lines(target)
    markers = _blue_cells(target, line_rows, line_cols)
    groups = _extract_groups(g[:, :separator_col])

    assert len(groups) == len(markers), "blue target cells should match instruction groups"
    out = target.copy()
    n_cols = len(line_cols) - 1

    for (sequence, blue_idx), (row_idx, col_idx) in zip(groups, markers):
        left_colors = sequence[:blue_idx]
        right_colors = sequence[blue_idx + 1 :]
        assert len(left_colors) <= col_idx
        assert len(right_colors) <= n_cols - col_idx - 1

        center_rs, center_cs = _cell_slice(line_rows, line_cols, row_idx, col_idx)
        out[center_rs, center_cs] = BLUE

        for offset, color in enumerate(reversed(left_colors), start=1):
            rs, cs = _cell_slice(line_rows, line_cols, row_idx, col_idx - offset)
            out[rs, cs] = int(color)

        for offset, color in enumerate(right_colors, start=1):
            rs, cs = _cell_slice(line_rows, line_cols, row_idx, col_idx + offset)
            out[rs, cs] = int(color)

    return out


GEN_ATTEMPTS = 300
GROUP_COUNT_RANGE = (2, 4)
CELL_SIZE_CHOICES = (2, 3)
CELL_COL_RANGE = (5, 8)
EXTRA_ROW_RANGE = (0, 2)
BRANCH_TYPES = ("left_only", "right_only", "both")
PATCH_HEIGHT_RANGE = (4, 6)
PATCH_STEP_RANGE = (2, 4)
GROUP_GAP_RANGE = (1, 2)


def _sample_sequence(branch: str, cell_cols: int) -> tuple[int, list[int]]:
    if branch == "left_only":
        marker_col = random.randint(1, cell_cols - 1)
        left_len = random.randint(1, min(4, marker_col, len(GROUP_COLORS)))
        right_len = 0
    elif branch == "right_only":
        marker_col = random.randint(0, cell_cols - 2)
        right_len = random.randint(1, min(4, cell_cols - marker_col - 1, len(GROUP_COLORS)))
        left_len = 0
    else:
        marker_col = random.randint(1, cell_cols - 2)
        left_len = random.randint(1, min(3, marker_col, len(GROUP_COLORS) - 1))
        right_cap = min(3, cell_cols - marker_col - 1, len(GROUP_COLORS) - left_len)
        right_len = random.randint(1, right_cap)

    picked = random.sample(GROUP_COLORS, left_len + right_len)
    sequence = picked[:left_len] + [BLUE] + picked[left_len:]
    return marker_col, sequence


def _build_group_patch(sequence: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
    height = random.randint(*PATCH_HEIGHT_RANGE)
    base_row = random.randint(0, min(1, height - 1))
    rows = [base_row]
    for _ in range(1, len(sequence)):
        rows.append(random.randint(base_row, height - 1))

    cols = [0]
    for _ in range(1, len(sequence)):
        cols.append(cols[-1] + random.randint(*PATCH_STEP_RANGE))

    patch = np.full((height, cols[-1] + 1), BG, dtype=int)
    for i in range(len(sequence) - 1):
        r0, c0 = rows[i], cols[i]
        r1, c1 = rows[i + 1], cols[i + 1]
        patch[r0, c0 : c1 + 1] = ORANGE
        patch[min(r0, r1) : max(r0, r1) + 1, c1] = ORANGE

    for color, r, c in zip(sequence, rows, cols):
        patch[r, c] = int(color)

    blue_idx = sequence.index(BLUE)
    return patch, (rows[blue_idx], cols[blue_idx])


def _target_template(cell_rows: int, cell_cols: int, cell_size: int, markers: list[tuple[int, int]]) -> np.ndarray:
    height = cell_rows * cell_size + cell_rows + 1
    width = cell_cols * cell_size + cell_cols + 1
    target = np.full((height, width), BG, dtype=int)

    for r in range(0, height, cell_size + 1):
        target[r, :] = GRAY
    for c in range(0, width, cell_size + 1):
        target[:, c] = GRAY

    for row_idx, col_idx in markers:
        rs, cs = _cell_slice_from_size(cell_size, row_idx, col_idx)
        target[rs, cs] = BLUE
    return target


def _cell_slice_from_size(cell_size: int, row_idx: int, col_idx: int) -> tuple[slice, slice]:
    row_start = 1 + row_idx * (cell_size + 1)
    col_start = 1 + col_idx * (cell_size + 1)
    return slice(row_start, row_start + cell_size), slice(col_start, col_start + cell_size)


def _expected_target(
    target_input: np.ndarray,
    cell_size: int,
    marker_specs: list[tuple[int, int, list[int]]],
) -> np.ndarray:
    out = target_input.copy()
    for row_idx, col_idx, sequence in marker_specs:
        blue_idx = sequence.index(BLUE)
        left_colors = sequence[:blue_idx]
        right_colors = sequence[blue_idx + 1 :]

        center_rs, center_cs = _cell_slice_from_size(cell_size, row_idx, col_idx)
        out[center_rs, center_cs] = BLUE

        for offset, color in enumerate(reversed(left_colors), start=1):
            rs, cs = _cell_slice_from_size(cell_size, row_idx, col_idx - offset)
            out[rs, cs] = int(color)
        for offset, color in enumerate(right_colors, start=1):
            rs, cs = _cell_slice_from_size(cell_size, row_idx, col_idx + offset)
            out[rs, cs] = int(color)
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n_groups = random.randint(*GROUP_COUNT_RANGE)
    cell_rows = n_groups + random.randint(*EXTRA_ROW_RANGE)
    cell_cols = random.randint(*CELL_COL_RANGE)
    cell_size = random.choice(CELL_SIZE_CHOICES)
    marker_rows = sorted(random.sample(range(cell_rows), n_groups))

    branches = random.sample(BRANCH_TYPES, k=min(len(BRANCH_TYPES), n_groups))
    while len(branches) < n_groups:
        branches.append(random.choice(BRANCH_TYPES))
    random.shuffle(branches)

    marker_specs: list[tuple[int, int, list[int]]] = []
    patches: list[np.ndarray] = []
    for i in range(n_groups):
        marker_col, sequence = _sample_sequence(branches[i], cell_cols)
        marker_specs.append((marker_rows[i], marker_col, sequence))
        patch, _blue_local = _build_group_patch(sequence)
        patches.append(patch)

    target_input = _target_template(
        cell_rows,
        cell_cols,
        cell_size,
        [(row_idx, col_idx) for row_idx, col_idx, _sequence in marker_specs],
    )
    expected = _expected_target(target_input, cell_size, marker_specs)

    left_width = max(patch.shape[1] for patch in patches) + random.randint(4, 8)
    patch_lefts = [random.randint(1, left_width - patch.shape[1]) for patch in patches]

    patch_tops: list[int] = []
    cur_top = random.randint(1, 3)
    for patch in patches:
        patch_tops.append(cur_top)
        cur_top += patch.shape[0] + random.randint(*GROUP_GAP_RANGE)
    left_height = cur_top + random.randint(0, 2)

    target_top = random.randint(1, 3)
    separator_col = left_width + random.randint(2, 4)
    target_left = separator_col + random.randint(2, 4)
    target_h, target_w = target_input.shape

    overall_h = max(left_height + random.randint(1, 3), target_top + target_h + random.randint(1, 3))
    overall_w = target_left + target_w + random.randint(1, 3)
    grid = np.full((overall_h, overall_w), BG, dtype=int)
    grid[:, separator_col] = GRAY

    for patch, top, left in zip(patches, patch_tops, patch_lefts):
        _place_patch(grid, patch, top, left)

    grid[target_top : target_top + target_h, target_left : target_left + target_w] = target_input
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _generate_case()
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input for puzzle 7")

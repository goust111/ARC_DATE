from __future__ import annotations

import random
from collections import Counter
from typing import Optional

import numpy as np


BG = 0
MARKER_COLOR = 3
PATTERN_COLOR = 9
LINE_COLORS = (1, 2, 4, 6)

TOP, RIGHT, BOTTOM, LEFT = range(4)
ROTATIONS_CCW = {TOP: 0, RIGHT: 3, BOTTOM: 2, LEFT: 1}
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _crop(arr: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _segment_color(segment: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in segment.ravel() if int(v) != int(bg)]
    assert vals, "separator segment must contain a visible color"
    return int(Counter(vals).most_common(1)[0][0])


def _find_separator_indices(grid: np.ndarray, bg: int, axis: int) -> list[int]:
    indices: list[int] = []
    bound = int(grid.shape[axis])
    for idx in range(bound):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if bool(np.all(line != int(bg))):
            indices.append(int(idx))
    assert len(indices) >= 2, "expected at least two separator lines"
    steps = [int(b - a) for a, b in zip(indices, indices[1:])]
    assert steps and len(set(steps)) == 1 and steps[0] >= 2, "separator lines must form an even grid"
    return indices


def _find_reference(
    grid: np.ndarray, bg: int, sep_rows: list[int], sep_cols: list[int]
) -> tuple[tuple[int, int], np.ndarray, tuple[int, int, int, int]]:
    ref_cell: Optional[tuple[int, int]] = None
    ref_crop: Optional[np.ndarray] = None
    ref_bbox: Optional[tuple[int, int, int, int]] = None
    for i in range(len(sep_rows) - 1):
        for j in range(len(sep_cols) - 1):
            r0, r1 = int(sep_rows[i]), int(sep_rows[i + 1])
            c0, c1 = int(sep_cols[j]), int(sep_cols[j + 1])
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            comps = _components(inner != int(bg))
            sizes = sorted(len(comp) for comp in comps)
            if len(comps) == 2 and sizes[0] == 1:
                assert ref_cell is None, "reference marker must be unique"
                pattern_comp = max(comps, key=len)
                ref_cell = (int(i), int(j))
                ref_crop, ref_bbox = _crop(inner, pattern_comp)
            else:
                assert len(comps) == 1, "non-reference cells must contain exactly one pattern component"
    assert ref_cell is not None and ref_crop is not None and ref_bbox is not None, "failed to locate reference cell"
    return ref_cell, ref_crop, ref_bbox


def _encoded_side(edge_colors: list[int]) -> int:
    counts = Counter(int(c) for c in edge_colors)
    assert sorted(counts.values()) == [1, 3], "each cell must have exactly one odd-colored side"
    odd_color = next(color for color, freq in counts.items() if int(freq) == 1)
    for side, color in enumerate(edge_colors):
        if int(color) == int(odd_color):
            return int(side)
    raise AssertionError("odd-colored side not found")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_rows = _find_separator_indices(g, bg, axis=0)
    sep_cols = _find_separator_indices(g, bg, axis=1)
    ref_cell, ref_crop, (ref_r0, ref_c0, _ref_r1, _ref_c1) = _find_reference(g, bg, sep_rows, sep_cols)

    out = g.copy()
    for i in range(len(sep_rows) - 1):
        for j in range(len(sep_cols) - 1):
            r0, r1 = int(sep_rows[i]), int(sep_rows[i + 1])
            c0, c1 = int(sep_cols[j]), int(sep_cols[j + 1])
            out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

            if (i, j) == ref_cell:
                side = TOP
            else:
                top = _segment_color(g[r0, c0 + 1 : c1], bg)
                right = _segment_color(g[r0 + 1 : r1, c1], bg)
                bottom = _segment_color(g[r1, c0 + 1 : c1], bg)
                left = _segment_color(g[r0 + 1 : r1, c0], bg)
                side = _encoded_side([top, right, bottom, left])

            stamp = np.rot90(ref_crop, ROTATIONS_CCW[int(side)])
            sr = int(r0 + 1 + ref_r0)
            sc = int(c0 + 1 + ref_c0)
            sh, sw = map(int, stamp.shape)
            assert sr + sh <= r1 and sc + sw <= c1, "rotated pattern must fit inside each cell"
            out[sr : sr + sh, sc : sc + sw] = stamp

    return out


def _sample_pattern_box(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        box_h = rng.randint(3, 4)
        box_w = rng.randint(3, 4)
        target = rng.randint(4, min(box_h * box_w - 2, 7))

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            nr, nc = int(start[0] + dr), int(start[1] + dc)
            if 0 <= nr < box_h and 0 <= nc < box_w:
                frontier.add((int(nr), int(nc)))

        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIR4:
                nr, nc = int(cell[0] + dr), int(cell[1] + dc)
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))

        if len(cells) != target:
            continue

        arr = np.full((box_h, box_w), BG, dtype=int)
        for r, c in cells:
            arr[int(r), int(c)] = PATTERN_COLOR

        arr, _ = _crop(arr, [(int(r), int(c)) for r, c in np.argwhere(arr == PATTERN_COLOR)])

        if any(bool(np.array_equal(arr, np.rot90(arr, k))) for k in (1, 2, 3)):
            continue
        return arr
    raise RuntimeError("failed to sample a non-symmetric pattern")


def _sample_marker_position(
    cell_h: int, cell_w: int, anchor_r: int, anchor_c: int, pattern_box: np.ndarray, rng: random.Random
) -> Optional[tuple[int, int]]:
    canvas = np.zeros((cell_h, cell_w), dtype=bool)
    ph, pw = map(int, pattern_box.shape)
    pattern_mask = pattern_box == PATTERN_COLOR
    canvas[anchor_r : anchor_r + ph, anchor_c : anchor_c + pw] = pattern_mask
    points = [(int(r), int(c)) for r, c in np.argwhere(canvas)]

    candidates: list[tuple[int, int, int]] = []
    for r in range(cell_h):
        for c in range(cell_w):
            if bool(canvas[r, c]):
                continue
            if any(
                0 <= r + dr < cell_h and 0 <= c + dc < cell_w and bool(canvas[r + dr, c + dc])
                for dr, dc in DIR4
            ):
                continue
            dist = min(abs(int(r) - pr) + abs(int(c) - pc) for pr, pc in points)
            candidates.append((int(dist), int(r), int(c)))

    if not candidates:
        return None
    candidates.sort()
    best = [item for item in candidates if int(item[0]) == int(candidates[0][0])]
    _dist, r, c = rng.choice(best)
    return int(r), int(c)


def _sample_directions(n_rows: int, n_cols: int, rng: random.Random) -> np.ndarray:
    dirs = np.empty((n_rows, n_cols), dtype=int)
    cells = [(int(i), int(j)) for i in range(n_rows) for j in range(n_cols)]
    rng.shuffle(cells)
    for side, cell in zip((TOP, RIGHT, BOTTOM, LEFT), cells):
        dirs[cell] = int(side)
    for cell in cells[4:]:
        dirs[cell] = int(rng.choice((TOP, RIGHT, BOTTOM, LEFT)))
    return dirs


def _adjacency_constraints(dirs: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], str]]]:
    n_rows, n_cols = map(int, dirs.shape)
    constraints = {
        (int(i), int(j)): [] for i in range(n_rows) for j in range(n_cols)
    }

    for i in range(n_rows - 1):
        for j in range(n_cols):
            upper = (int(i), int(j))
            lower = (int(i + 1), int(j))
            upper_non_unique = int(dirs[upper]) != BOTTOM
            lower_non_unique = int(dirs[lower]) != TOP
            if upper_non_unique and lower_non_unique:
                rel = "eq"
            elif upper_non_unique != lower_non_unique:
                rel = "neq"
            else:
                rel = ""
            if rel:
                constraints[upper].append((lower, rel))
                constraints[lower].append((upper, rel))

    for i in range(n_rows):
        for j in range(n_cols - 1):
            left_cell = (int(i), int(j))
            right_cell = (int(i), int(j + 1))
            left_non_unique = int(dirs[left_cell]) != RIGHT
            right_non_unique = int(dirs[right_cell]) != LEFT
            if left_non_unique and right_non_unique:
                rel = "eq"
            elif left_non_unique != right_non_unique:
                rel = "neq"
            else:
                rel = ""
            if rel:
                constraints[left_cell].append((right_cell, rel))
                constraints[right_cell].append((left_cell, rel))

    return constraints


def _sample_majority_colors(dirs: np.ndarray, rng: random.Random) -> Optional[np.ndarray]:
    n_rows, n_cols = map(int, dirs.shape)
    constraints = _adjacency_constraints(dirs)
    order = sorted(constraints, key=lambda cell: len(constraints[cell]), reverse=True)
    assign: dict[tuple[int, int], int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        cell = order[idx]
        choices = list(LINE_COLORS)
        rng.shuffle(choices)
        for color in choices:
            ok = True
            for other, rel in constraints[cell]:
                if other not in assign:
                    continue
                if rel == "eq" and int(assign[other]) != int(color):
                    ok = False
                    break
                if rel == "neq" and int(assign[other]) == int(color):
                    ok = False
                    break
            if not ok:
                continue
            assign[cell] = int(color)
            if backtrack(idx + 1):
                return True
            del assign[cell]
        return False

    if not backtrack(0):
        return None

    majors = np.empty((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            majors[i, j] = int(assign[(int(i), int(j))])
    return majors


def _other_line_color(excluded: set[int], rng: random.Random) -> int:
    choices = [int(color) for color in LINE_COLORS if int(color) not in excluded]
    assert choices, "line palette exhausted"
    return int(rng.choice(choices))


def _build_edge_colors(
    dirs: np.ndarray, majors: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = map(int, dirs.shape)
    horiz = np.empty((n_rows + 1, n_cols), dtype=int)
    vert = np.empty((n_rows, n_cols + 1), dtype=int)

    for j in range(n_cols):
        top_major = int(majors[0, j])
        bottom_major = int(majors[n_rows - 1, j])
        horiz[0, j] = top_major if int(dirs[0, j]) != TOP else _other_line_color({top_major}, rng)
        horiz[n_rows, j] = bottom_major if int(dirs[n_rows - 1, j]) != BOTTOM else _other_line_color({bottom_major}, rng)

    for i in range(n_rows):
        left_major = int(majors[i, 0])
        right_major = int(majors[i, n_cols - 1])
        vert[i, 0] = left_major if int(dirs[i, 0]) != LEFT else _other_line_color({left_major}, rng)
        vert[i, n_cols] = right_major if int(dirs[i, n_cols - 1]) != RIGHT else _other_line_color({right_major}, rng)

    for i in range(n_rows - 1):
        for j in range(n_cols):
            above_unique = int(dirs[i, j]) == BOTTOM
            below_unique = int(dirs[i + 1, j]) == TOP
            if not above_unique and not below_unique:
                color = int(majors[i, j])
                assert color == int(majors[i + 1, j])
            elif above_unique and not below_unique:
                color = int(majors[i + 1, j])
                assert color != int(majors[i, j])
            elif not above_unique and below_unique:
                color = int(majors[i, j])
                assert color != int(majors[i + 1, j])
            else:
                color = _other_line_color({int(majors[i, j]), int(majors[i + 1, j])}, rng)
            horiz[i + 1, j] = int(color)

    for i in range(n_rows):
        for j in range(n_cols - 1):
            left_unique = int(dirs[i, j]) == RIGHT
            right_unique = int(dirs[i, j + 1]) == LEFT
            if not left_unique and not right_unique:
                color = int(majors[i, j])
                assert color == int(majors[i, j + 1])
            elif left_unique and not right_unique:
                color = int(majors[i, j + 1])
                assert color != int(majors[i, j])
            elif not left_unique and right_unique:
                color = int(majors[i, j])
                assert color != int(majors[i, j + 1])
            else:
                color = _other_line_color({int(majors[i, j]), int(majors[i, j + 1])}, rng)
            vert[i, j + 1] = int(color)

    return horiz, vert


def _validate_edge_encoding(dirs: np.ndarray, horiz: np.ndarray, vert: np.ndarray) -> bool:
    n_rows, n_cols = map(int, dirs.shape)
    for i in range(n_rows):
        for j in range(n_cols):
            colors = [
                int(horiz[i, j]),
                int(vert[i, j + 1]),
                int(horiz[i + 1, j]),
                int(vert[i, j]),
            ]
            if _encoded_side(colors) != int(dirs[i, j]):
                return False
    return True


def _draw_grid_lines(
    n_rows: int, n_cols: int, cell_h: int, cell_w: int, horiz: np.ndarray, vert: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    sep_rows = [int(i * (cell_h + 1)) for i in range(n_rows + 1)]
    sep_cols = [int(j * (cell_w + 1)) for j in range(n_cols + 1)]
    grid = np.full((sep_rows[-1] + 1, sep_cols[-1] + 1), BG, dtype=int)

    for i in range(n_rows + 1):
        rr = int(sep_rows[i])
        for j in range(n_cols):
            c0, c1 = int(sep_cols[j]), int(sep_cols[j + 1])
            grid[rr, c0 + 1 : c1] = int(horiz[i, j])

    for i in range(n_rows):
        r0, r1 = int(sep_rows[i]), int(sep_rows[i + 1])
        for j in range(n_cols + 1):
            cc = int(sep_cols[j])
            grid[r0 + 1 : r1, cc] = int(vert[i, j])

    for i in range(n_rows + 1):
        rr = int(sep_rows[i])
        for j in range(n_cols + 1):
            if j < n_cols:
                color = int(horiz[i, j])
            else:
                color = int(horiz[i, j - 1])
            grid[rr, int(sep_cols[j])] = int(color)

    return grid, sep_rows, sep_cols


def _paste_pattern(
    grid: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
    cell: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    pattern_box: np.ndarray,
) -> None:
    r0 = int(sep_rows[cell[0]] + 1 + anchor_r)
    c0 = int(sep_cols[cell[1]] + 1 + anchor_c)
    h, w = map(int, pattern_box.shape)
    grid[r0 : r0 + h, c0 : c0 + w] = pattern_box


def _render_expected(
    grid: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
    ref_cell: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    ref_crop: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for i in range(len(sep_rows) - 1):
        for j in range(len(sep_cols) - 1):
            r0, r1 = int(sep_rows[i]), int(sep_rows[i + 1])
            c0, c1 = int(sep_cols[j]), int(sep_cols[j + 1])
            out[r0 + 1 : r1, c0 + 1 : c1] = BG
            side = TOP if (i, j) == ref_cell else int(dirs[i, j])
            stamp = np.rot90(ref_crop, ROTATIONS_CCW[int(side)])
            sr = int(r0 + 1 + anchor_r)
            sc = int(c0 + 1 + anchor_c)
            sh, sw = map(int, stamp.shape)
            out[sr : sr + sh, sc : sc + sw] = stamp
    return out


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)

    pattern_box = _sample_pattern_box(rng)
    ph, pw = map(int, pattern_box.shape)
    max_dim = max(ph, pw)

    cell_h = rng.randint(max_dim + 2, 6)
    cell_w = rng.randint(max_dim + 2, 6)
    if cell_h < max_dim + 2 or cell_w < max_dim + 2:
        return None

    anchor_r = rng.randint(0, cell_h - max_dim)
    anchor_c = rng.randint(0, cell_w - max_dim)
    marker_pos = _sample_marker_position(cell_h, cell_w, anchor_r, anchor_c, pattern_box, rng)
    if marker_pos is None:
        return None

    dirs = _sample_directions(n_rows, n_cols, rng)
    majors = _sample_majority_colors(dirs, rng)
    if majors is None:
        return None
    horiz, vert = _build_edge_colors(dirs, majors, rng)
    if not _validate_edge_encoding(dirs, horiz, vert):
        return None
    if len(set(map(int, np.concatenate([horiz.ravel(), vert.ravel()])))) < 3:
        return None

    grid, sep_rows, sep_cols = _draw_grid_lines(n_rows, n_cols, cell_h, cell_w, horiz, vert)

    ref_cell = rng.choice(((0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)))

    for i in range(n_rows):
        for j in range(n_cols):
            if (i, j) == ref_cell:
                side = TOP
            else:
                side = int(rng.choice((TOP, RIGHT, BOTTOM, LEFT)))
            stamp = np.rot90(pattern_box, ROTATIONS_CCW[int(side)])
            _paste_pattern(grid, sep_rows, sep_cols, (i, j), anchor_r, anchor_c, stamp)

    mr = int(sep_rows[ref_cell[0]] + 1 + marker_pos[0])
    mc = int(sep_cols[ref_cell[1]] + 1 + marker_pos[1])
    if int(grid[mr, mc]) != BG:
        return None
    grid[mr, mc] = MARKER_COLOR

    expected = _render_expected(grid, sep_rows, sep_cols, ref_cell, anchor_r, anchor_c, pattern_box, dirs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

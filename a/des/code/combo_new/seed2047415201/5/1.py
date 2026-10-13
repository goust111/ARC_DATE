from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ROW_PALETTE = (2, 3, 4, 8)
MAX_GENERATION_ATTEMPTS = 400


def _legend_geometry(g: np.ndarray) -> tuple[int, int, int, int, list[int]]:
    h, w = map(int, g.shape)
    candidates: list[tuple[int, int, int, int, list[int]]] = []

    for n in range(min(4, len(ROW_PALETTE)), 1, -1):
        for cell_size in range(1, 10):
            side = n * cell_size + n + 1
            if side > h or side >= w:
                continue
            seps = [i * (cell_size + 1) for i in range(n + 1)]
            if not all(np.all(g[r, :side] == GREY) for r in seps):
                continue
            if not all(np.all(g[:side, c] == GREY) for c in seps):
                continue
            if side < h and np.any(g[side:, :side] != BLACK):
                continue

            legend = g[:side, :side]
            ok = True
            for rr in range(n):
                r0, r1 = seps[rr] + 1, seps[rr + 1]
                for cc in range(n):
                    c0, c1 = seps[cc] + 1, seps[cc + 1]
                    cell = legend[r0:r1, c0:c1]
                    if cell.shape != (cell_size, cell_size):
                        ok = False
                        break
                    vals = set(map(int, np.unique(cell)))
                    if not vals.issubset({BLACK, GREY}) or GREY not in vals:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((n, cell_size, side, side, seps))

    assert candidates, "expected a square grey-separated legend on the left"
    candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
    return candidates[0]


def _extract_templates(g: np.ndarray, n: int, cell_size: int, seps: list[int]) -> list[list[np.ndarray]]:
    templates: list[list[np.ndarray]] = []
    for rr in range(n):
        row: list[np.ndarray] = []
        r0, r1 = seps[rr] + 1, seps[rr + 1]
        for cc in range(n):
            c0, c1 = seps[cc] + 1, seps[cc + 1]
            cell = g[r0:r1, c0:c1]
            assert cell.shape == (cell_size, cell_size)
            mask = cell == GREY
            assert mask.any(), "each legend cell must contain a sprite template"
            row.append(mask.copy())
        templates.append(row)
    return templates


def _rotated_template(mask: np.ndarray, row_index: int) -> np.ndarray:
    return np.rot90(mask, (-row_index) % 4)


def _stamp(
    out: np.ndarray,
    mask: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    min_col: int = 0,
) -> bool:
    h, w = map(int, mask.shape)
    assert h % 2 == 1 and w % 2 == 1, "templates must have an unambiguous center"
    top = center_r - h // 2
    left = center_c - w // 2
    if top < 0 or left < min_col or top + h > out.shape[0] or left + w > out.shape[1]:
        return False

    rr, cc = np.nonzero(mask)
    rr = rr + top
    cc = cc + left
    if np.any(out[rr, cc] != BLACK):
        return False
    out[rr, cc] = int(color)
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp row-colored, row-rotated legend sprites at marker pixels in the main area.

    Input structure:
    - The background is black.
    - A square left legend is divided into an NxN grid by grey separator lines.
    - Every legend cell contains one small grey sprite template on black.
    - The right main area contains singleton marker pixels whose colors come from a fixed visible row palette.

    Rule:
    - Read the grey sprite template from legend row r and legend column c.
    - A marker's color selects r by its position in the fixed palette (2, 3, 4, 8).
    - A marker's column in the right main area selects c by the equal-width column interval containing it.
    - Stamp the selected template centered on the marker, rotating it by r clockwise quarter-turns.
    - Draw the stamp with the marker color and erase the entire legend and separator area.

    Key insight:
    - Marker color encodes the legend row and rotation, while marker position encodes the legend column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    n, cell_size, _legend_h, legend_w, seps = _legend_geometry(g)
    templates = _extract_templates(g, n, cell_size, seps)
    main_c0 = legend_w
    main_w = g.shape[1] - main_c0
    assert main_w >= n, "main area must be wide enough to split into legend columns"

    valid_marker_colors = set(ROW_PALETTE[:n])
    main_area = g[:, main_c0:]
    non_bg = set(map(int, np.unique(main_area))) - {BLACK}
    assert non_bg.issubset(valid_marker_colors), "main area must contain only row-color markers"

    out = np.zeros_like(g)
    for r, c_rel in np.argwhere(main_area != BLACK):
        c = int(c_rel) + main_c0
        color = int(g[r, c])
        row_index = ROW_PALETTE.index(color)
        col_index = min(n - 1, (int(c_rel) * n) // main_w)
        mask = _rotated_template(templates[row_index][col_index], row_index)
        ok = _stamp(out, mask, int(r), c, color, min_col=main_c0)
        assert ok, "stamped sprites must fit in the main area and not overlap"

    return out


# --------------- generator ---------------


def _shape_library() -> list[np.ndarray]:
    raw = [
        ("110", "010", "010"),
        ("100", "111", "001"),
        ("111", "100", "100"),
        ("010", "111", "010"),
        ("100", "110", "011"),
        ("001", "111", "010"),
        ("010", "011", "110"),
        ("010", "110", "010"),
        ("101", "010", "101"),
        ("110", "011", "010"),
        ("010", "111", "100"),
        ("001", "011", "110"),
        ("010", "010", "111"),
        ("111", "010", "010"),
        ("011", "110", "010"),
        ("010", "111", "001"),
        ("101", "111", "010"),
        ("010", "111", "101"),
        ("011", "010", "110"),
        ("110", "010", "011"),
    ]
    masks = []
    seen: set[tuple[int, ...]] = set()
    for rows in raw:
        mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
        for variant in (mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 1)):
            key = tuple(int(x) for x in variant.reshape(-1))
            if key not in seen and bool(variant[1, 1]):
                seen.add(key)
                masks.append(variant.copy())
    return masks


SHAPE_LIBRARY = _shape_library()


def _build_legend(templates: list[list[np.ndarray]], n: int, cell_size: int) -> np.ndarray:
    side = n * cell_size + n + 1
    legend = np.zeros((side, side), dtype=int)
    seps = [i * (cell_size + 1) for i in range(n + 1)]
    legend[seps, :] = GREY
    legend[:, seps] = GREY
    for rr in range(n):
        r0 = seps[rr] + 1
        for cc in range(n):
            c0 = seps[cc] + 1
            legend[r0 : r0 + cell_size, c0 : c0 + cell_size][templates[rr][cc]] = GREY
    return legend


def _expected_output(
    shape: tuple[int, int],
    templates: list[list[np.ndarray]],
    markers: list[tuple[int, int, int]],
    main_c0: int,
) -> np.ndarray | None:
    out = np.zeros(shape, dtype=int)
    for row_index, r, c in markers:
        rel_c = c - main_c0
        main_w = shape[1] - main_c0
        col_index = min(len(templates) - 1, (rel_c * len(templates)) // main_w)
        mask = _rotated_template(templates[row_index][col_index], row_index)
        if not _stamp(out, mask, r, c, ROW_PALETTE[row_index], min_col=main_c0):
            return None
    return out


def _candidate_centers(
    h: int,
    w: int,
    main_c0: int,
    segment_width: int,
    col_index: int,
) -> list[tuple[int, int]]:
    c_start = main_c0 + col_index * segment_width
    c_end = min(main_c0 + (col_index + 1) * segment_width, w)
    centers = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(c_start, c_end)
        if main_c0 + 1 <= c <= w - 2
    ]
    random.shuffle(centers)
    return centers


def _too_close(cell: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in taken)


def _choose_marker_pairs(n: int) -> list[tuple[int, int]]:
    target_count = random.randint(7, 11)
    pairs = {(r, random.randrange(n)) for r in range(n)}
    pairs.update((random.randrange(n), c) for c in range(n))
    while len(pairs) < target_count:
        pairs.add((random.randrange(n), random.randrange(n)))
    pairs_list = list(pairs)
    random.shuffle(pairs_list)
    return pairs_list


def _place_markers(
    shape: tuple[int, int],
    templates: list[list[np.ndarray]],
    pairs: list[tuple[int, int]],
    main_c0: int,
    segment_width: int,
) -> list[tuple[int, int, int]] | None:
    h, w = shape
    markers: list[tuple[int, int, int]] = []
    center_taken: list[tuple[int, int]] = []
    trial_out = np.zeros(shape, dtype=int)

    for row_index, col_index in pairs:
        mask = _rotated_template(templates[row_index][col_index], row_index)
        placed = False
        for r, c in _candidate_centers(h, w, main_c0, segment_width, col_index):
            if _too_close((r, c), center_taken):
                continue
            test = trial_out.copy()
            if not _stamp(test, mask, r, c, ROW_PALETTE[row_index], min_col=main_c0):
                continue

            # Keep one cell of air around painted sprite pixels when possible.
            rr, cc = np.nonzero(mask)
            top = r - mask.shape[0] // 2
            left = c - mask.shape[1] // 2
            active = {(int(a + top), int(b + left)) for a, b in zip(rr, cc)}
            conflict = False
            old = set(map(tuple, np.argwhere(trial_out != BLACK)))
            for ar, ac in active:
                for nr in range(ar - 1, ar + 2):
                    for nc in range(ac - 1, ac + 2):
                        if (nr, nc) in old:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if conflict:
                continue

            trial_out = test
            markers.append((row_index, r, c))
            center_taken.append((r, c))
            placed = True
            break
        if not placed:
            return None
    return markers


def _gen_attempt() -> np.ndarray | None:
    n = 4
    cell_size = 3
    segment_width = random.choice([2, 3])
    legend_side = n * cell_size + n + 1
    main_w = n * segment_width
    extra_rows = random.randint(0, 5)
    shape = (legend_side + extra_rows, legend_side + main_w)
    main_c0 = legend_side

    picked = random.sample(SHAPE_LIBRARY, n * n)
    templates = [picked[i * n : (i + 1) * n] for i in range(n)]

    pairs = _choose_marker_pairs(n)
    markers = _place_markers(shape, templates, pairs, main_c0, segment_width)
    if markers is None:
        return None

    legend = _build_legend(templates, n, cell_size)
    grid = np.zeros(shape, dtype=int)
    grid[:legend_side, :legend_side] = legend
    for row_index, r, c in markers:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = ROW_PALETTE[row_index]

    expected = _expected_output(shape, templates, markers, main_c0)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

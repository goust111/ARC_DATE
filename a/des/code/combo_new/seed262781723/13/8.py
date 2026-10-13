import random

import numpy as np


BG = 0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[set[tuple[int, int]], dict[frozenset[int], int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, set[tuple[int, int]], dict[frozenset[int], int]]] = []

    for top_row in (0, H - 2):
        bottom_row = top_row + 1
        mask = (grid[top_row] != bg) & (grid[bottom_row] != bg)
        for c0, c1 in _runs(mask):
            width = c1 - c0 + 1
            if width < 2 or width % 2 != 0:
                continue

            mapping: dict[frozenset[int], int] = {}
            ok = True
            for col in range(c0, c1 + 1, 2):
                a = int(grid[top_row, col])
                b = int(grid[top_row, col + 1])
                mixed0 = int(grid[bottom_row, col])
                mixed1 = int(grid[bottom_row, col + 1])
                if bg in (a, b, mixed0, mixed1) or a == b or mixed0 != mixed1:
                    ok = False
                    break
                key = frozenset((a, b))
                if key in mapping and mapping[key] != mixed0:
                    ok = False
                    break
                mapping[key] = mixed0

            if not ok:
                continue

            isolation = 0
            if c0 == 0 or (grid[top_row, c0 - 1] == bg and grid[bottom_row, c0 - 1] == bg):
                isolation += 1
            if c1 == W - 1 or (grid[top_row, c1 + 1] == bg and grid[bottom_row, c1 + 1] == bg):
                isolation += 1
            if top_row == 0:
                if H > 2 and np.all(grid[2, c0 : c1 + 1] == bg):
                    isolation += 2
            else:
                if H > 2 and np.all(grid[H - 3, c0 : c1 + 1] == bg):
                    isolation += 2

            mask_cells = {(r, c) for r in (top_row, bottom_row) for c in range(c0, c1 + 1)}
            candidates.append((len(mapping), width, isolation, mask_cells, mapping))

    assert candidates, "failed to locate the 2-row legend"
    _, _, _, legend_cells, mapping = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return legend_cells, mapping


def _extract_paths(
    scene: np.ndarray, bg: int
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []

    for color in sorted(int(v) for v in np.unique(scene) if int(v) != bg):
        coords = np.argwhere(scene == color)
        assert len(coords) >= 2, f"path color {color} is too short"
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1 and len(cols) >= 2:
            row = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert np.all(scene[row, c0 : c1 + 1] != bg), f"horizontal path {color} is broken"
            horizontals.append({"color": color, "row": row, "c0": c0, "c1": c1})
        elif len(cols) == 1 and len(rows) >= 2:
            col = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert np.all(scene[r0 : r1 + 1, col] != bg), f"vertical path {color} is broken"
            verticals.append({"color": color, "col": col, "r0": r0, "r1": r1})
        else:
            raise AssertionError(f"color {color} is not a single straight path")

    assert horizontals and verticals, "expected at least one horizontal and one vertical path"
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """Mark every orthogonal path crossing with the legend-defined mixed color.

    Input structure:
    - The background is black.
    - The main scene contains several single-color horizontal or vertical path
      segments, each one pixel wide and each using its own color.
    - A separate 2-row legend sits on the top or bottom edge; every adjacent
      pair of colors in the first row maps to the repeated mixed color beneath it.

    Rule:
    - Read the legend as unordered color-pair -> mixed-color mappings.
    - Reconstruct each straight path from its visible same-color pixels, even if
      some crossing cells currently show the other path's color.
    - For every horizontal/vertical crossing implied by those spans, replace the
      center cell and its four orthogonal neighbors with the mapped mixed color.
    - Keep all non-crossing path cells and the legend unchanged.

    Key insight:
    - The visible crossing cell alone is insufficient: the solver must recover the
      full row span and column span of differently colored straight paths, then use
      the edge legend to convert that hidden color pair into a plus-shaped marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    bg = BG

    legend_cells, mapping = _find_legend(g, bg)
    scene = g.copy()
    for r, c in legend_cells:
        scene[r, c] = bg

    horizontals, verticals = _extract_paths(scene, bg)
    out = g.copy()

    found = 0
    for h in horizontals:
        for v in verticals:
            row = h["row"]
            col = v["col"]
            if h["c0"] <= col <= h["c1"] and v["r0"] <= row <= v["r1"]:
                key = frozenset((h["color"], v["color"]))
                assert key in mapping, f"missing legend entry for colors {sorted(key)}"
                mixed = int(mapping[key])
                found += 1
                for rr, cc in (
                    (row, col),
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in legend_cells:
                        out[rr, cc] = mixed

    assert found > 0, "expected at least one crossing"
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (14, 28)
GRID_WIDTH_RANGE = (14, 28)
LINE_COUNT_CHOICES = ((1, 1), (1, 2), (2, 1), (2, 2))
MIN_LINE_LEN = 3
MAX_LINE_LEN = 15
POSITION_MIN_GAP = 4
POSITION_MAX_GAP = 10
MAX_GENERATION_ATTEMPTS = 400


def _sample_positions(count: int, lo: int, hi: int) -> list[int] | None:
    if count == 1:
        if lo > hi:
            return None
        return [random.randint(lo, hi)]

    choices = list(range(lo, hi + 1))
    if len(choices) < count:
        return None
    valid: list[list[int]] = []
    for a in choices:
        for b in choices:
            if b <= a:
                continue
            gap = b - a
            if POSITION_MIN_GAP <= gap <= POSITION_MAX_GAP:
                valid.append([a, b])
    if not valid:
        return None
    return random.choice(valid)


def _interval_options(
    anchors: list[int], active: list[int], lo: int, hi: int
) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    active_coords = [anchors[idx] for idx in active]

    if active_coords:
        left_anchor = min(active_coords)
        right_anchor = max(active_coords)
        for start in range(lo, left_anchor):
            for end in range(right_anchor + 1, hi + 1):
                length = end - start + 1
                if not (MIN_LINE_LEN <= length <= MAX_LINE_LEN):
                    continue
                if any(start <= anchors[idx] <= end for idx in range(len(anchors)) if idx not in active):
                    continue
                options.append((start, end))
        return options

    for start in range(lo, hi + 1):
        for end in range(start + MIN_LINE_LEN - 1, min(hi, start + MAX_LINE_LEN - 1) + 1):
            if any(start <= coord <= end for coord in anchors):
                continue
            options.append((start, end))
    return options


def _actual_matrix(
    h_rows: list[int],
    v_cols: list[int],
    h_spans: list[tuple[int, int]],
    v_spans: list[tuple[int, int]],
) -> np.ndarray:
    mat = np.zeros((len(h_rows), len(v_cols)), dtype=bool)
    for i, row in enumerate(h_rows):
        hc0, hc1 = h_spans[i]
        for j, col in enumerate(v_cols):
            vr0, vr1 = v_spans[j]
            mat[i, j] = hc0 <= col <= hc1 and vr0 <= row <= vr1
    return mat


def _render_input(
    H: int,
    W: int,
    legend_rows: tuple[int, int],
    legend_start: int,
    legend_entries: list[tuple[int, int, int]],
    h_rows: list[int],
    v_cols: list[int],
    h_spans: list[tuple[int, int]],
    v_spans: list[tuple[int, int]],
    h_colors: list[int],
    v_colors: list[int],
    shown: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    lr0, lr1 = legend_rows
    for idx, (a, b, mixed) in enumerate(legend_entries):
        base = legend_start + 2 * idx
        grid[lr0, base] = a
        grid[lr0, base + 1] = b
        grid[lr1, base] = mixed
        grid[lr1, base + 1] = mixed

    for row, (c0, c1), color in zip(h_rows, h_spans, h_colors):
        grid[row, c0 : c1 + 1] = color
    for col, (r0, r1), color in zip(v_cols, v_spans, v_colors):
        grid[r0 : r1 + 1, col] = color

    for (i, j), color in shown.items():
        grid[h_rows[i], v_cols[j]] = color
    return grid


def _render_expected(
    grid: np.ndarray,
    legend_rows: tuple[int, int],
    legend_start: int,
    legend_entries: list[tuple[int, int, int]],
    h_rows: list[int],
    v_cols: list[int],
    matrix: np.ndarray,
    pair_to_mixed: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    legend_width = 2 * len(legend_entries)
    legend_cells = {
        (r, c)
        for r in legend_rows
        for c in range(legend_start, legend_start + legend_width)
    }
    H, W = grid.shape
    for i, row in enumerate(h_rows):
        for j, col in enumerate(v_cols):
            if not matrix[i, j]:
                continue
            mixed = pair_to_mixed[(i, j)]
            for rr, cc in (
                (row, col),
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in legend_cells:
                    out[rr, cc] = mixed
    return out


def _visible_counts(
    h_spans: list[tuple[int, int]],
    v_spans: list[tuple[int, int]],
    matrix: np.ndarray,
    shown_by_vertical: set[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    h_counts = [c1 - c0 + 1 for c0, c1 in h_spans]
    v_counts = [r1 - r0 + 1 for r0, r1 in v_spans]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not matrix[i, j]:
                continue
            if (i, j) in shown_by_vertical:
                h_counts[i] -= 1
            else:
                v_counts[j] -= 1
    return h_counts, v_counts


def _nonempty_binary_matrix(rows: int, cols: int) -> np.ndarray:
    while True:
        mat = np.array(
            [[bool(random.getrandbits(1)) for _ in range(cols)] for _ in range(rows)],
            dtype=bool,
        )
        if np.any(mat):
            return mat


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    n_h, n_v = random.choice(LINE_COUNT_CHOICES)
    matrix = _nonempty_binary_matrix(n_h, n_v)

    legend_on_top = random.random() < 0.5
    legend_rows = (0, 1) if legend_on_top else (H - 2, H - 1)
    path_r_lo = 3 if legend_on_top else 1
    path_r_hi = H - 2 if legend_on_top else H - 4
    center_r_lo = path_r_lo + 1
    center_r_hi = path_r_hi - 1
    center_c_lo, center_c_hi = 2, W - 3

    h_rows = _sample_positions(n_h, center_r_lo, center_r_hi)
    v_cols = _sample_positions(n_v, center_c_lo, center_c_hi)
    if h_rows is None or v_cols is None:
        return None

    h_spans: list[tuple[int, int]] = []
    for i in range(n_h):
        active = [j for j in range(n_v) if matrix[i, j]]
        options = _interval_options(v_cols, active, 1, W - 2)
        if not options:
            return None
        h_spans.append(random.choice(options))

    v_spans: list[tuple[int, int]] = []
    for j in range(n_v):
        active = [i for i in range(n_h) if matrix[i, j]]
        options = _interval_options(h_rows, active, path_r_lo, path_r_hi)
        if not options:
            return None
        v_spans.append(random.choice(options))

    actual = _actual_matrix(h_rows, v_cols, h_spans, v_spans)
    if not np.array_equal(actual, matrix):
        return None

    intersection_indices = [(i, j) for i in range(n_h) for j in range(n_v) if matrix[i, j]]
    if not intersection_indices:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    h_colors = colors[:n_h]
    v_colors = colors[n_h : n_h + n_v]
    remaining = colors[n_h + n_v :]
    if len(remaining) < len(intersection_indices):
        return None
    mixed_colors = random.sample(remaining, len(intersection_indices))

    pair_to_mixed_idx: dict[tuple[int, int], int] = {
        pair: mixed for pair, mixed in zip(intersection_indices, mixed_colors)
    }
    legend_entries = []
    for (i, j), mixed in pair_to_mixed_idx.items():
        a, b = h_colors[i], v_colors[j]
        if random.random() < 0.5:
            a, b = b, a
        legend_entries.append((a, b, mixed))
    random.shuffle(legend_entries)

    legend_width = 2 * len(legend_entries)
    if legend_width > W:
        return None
    legend_start = random.randint(0, W - legend_width)

    shown_by_vertical: set[tuple[int, int]] = set()
    shown: dict[tuple[int, int], int] = {}
    for i, j in intersection_indices:
        if random.random() < 0.5:
            shown_by_vertical.add((i, j))
            shown[(i, j)] = v_colors[j]
        else:
            shown[(i, j)] = h_colors[i]

    h_visible, v_visible = _visible_counts(h_spans, v_spans, matrix, shown_by_vertical)
    if min(h_visible) < 2 or min(v_visible) < 2:
        return None

    grid = _render_input(
        H=H,
        W=W,
        legend_rows=legend_rows,
        legend_start=legend_start,
        legend_entries=legend_entries,
        h_rows=h_rows,
        v_cols=v_cols,
        h_spans=h_spans,
        v_spans=v_spans,
        h_colors=h_colors,
        v_colors=v_colors,
        shown=shown,
    )

    expected = _render_expected(
        grid=grid,
        legend_rows=legend_rows,
        legend_start=legend_start,
        legend_entries=legend_entries,
        h_rows=h_rows,
        v_cols=v_cols,
        matrix=matrix,
        pair_to_mixed=pair_to_mixed_idx,
    )

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-mix puzzle")

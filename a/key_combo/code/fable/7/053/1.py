import numpy as np


BLACK = 0
GRAY = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
RETRY_LIMIT = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    body_mask = g == GRAY
    assert body_mask.any(), "expected a gray stretched body"
    r0, c0, r1, c1 = _bbox(body_mask)
    body = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.isin(body, [BLACK, GRAY]).all(), "body bbox may only contain black and gray"
    assert np.all(np.any(body == GRAY, axis=1)), "every body row must contain gray pixels"

    sep_col = c1 + 1
    assert sep_col < g.shape[1], "expected a legend to the right of the body"
    assert np.all(g[:, sep_col] == BLACK), "body and legend must be separated by one blank column"

    legend_mask = (g != BLACK) & (g != GRAY)
    assert legend_mask.any(), "expected legend bars"
    legend_coords = np.argwhere(legend_mask)
    assert int(legend_coords[:, 1].min()) >= c1 + 2, "legend must lie strictly right of the separator"

    length_to_color: dict[int, int] = {}
    colors_seen: set[int] = set()
    for comp in _components(legend_mask):
        rows = np.unique(comp[:, 0])
        cols = np.sort(comp[:, 1])
        assert rows.size == 1, "legend bars must be horizontal"
        assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "legend bars must be contiguous"
        vals = g[comp[:, 0], comp[:, 1]]
        uniq = set(map(int, vals))
        assert len(uniq) == 1, "legend component must be monochromatic"
        color = int(vals[0])
        length = int(comp.shape[0])
        assert 1 <= length <= 4, "legend lengths must be in 1..4"
        assert length not in length_to_color, "legend lengths must be unique"
        assert color not in colors_seen, "legend colors must be unique"
        length_to_color[length] = color
        colors_seen.add(color)

    assert set(length_to_color) == {1, 2, 3, 4}, "legend must encode all four run lengths"

    collapsed_rows: list[tuple[np.ndarray, int]] = []
    r = 0
    while r < body.shape[0]:
        row = body[r].copy()
        run = 1
        while r + run < body.shape[0] and np.array_equal(body[r + run], row):
            run += 1
        assert 1 <= run <= 4, "stretched row runs must be between 1 and 4"
        collapsed_rows.append((row, run))
        r += run

    assert len(collapsed_rows) >= 2, "need multiple folded rows"
    for i in range(len(collapsed_rows) - 1):
        assert not np.array_equal(collapsed_rows[i][0], collapsed_rows[i + 1][0]), (
            "adjacent source rows must differ for unique folding"
        )

    return body, collapsed_rows, length_to_color


def _solve_from_parsed(
    body: np.ndarray,
    collapsed_rows: list[tuple[np.ndarray, int]],
    length_to_color: dict[int, int],
) -> np.ndarray:
    out = np.zeros((len(collapsed_rows), body.shape[1]), dtype=int)
    for r, (row, run) in enumerate(collapsed_rows):
        color = int(length_to_color[run])
        out[r, row == GRAY] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a vertically stretched gray body and recolor each recovered row by the legend bar whose length matches its run length.

    Input structure:
    - Black is the background.
    - The main object is a gray pattern whose original rows were stretched only vertically: each source row is repeated 1 to 4 times, and neighboring source rows are different.
    - To the right of the body, after one blank separator column, there are four monochrome horizontal legend bars of distinct lengths 1, 2, 3, and 4 in arbitrary top-to-bottom order.

    Rule:
    - Ignore outer padding, crop the gray body, and scan its rows from top to bottom.
    - Merge each maximal run of identical adjacent rows into one row and record the run length k.
    - Output only the folded body region; recolor every gray cell in that row with the color of the legend bar of length k, while keeping black cells black.

    Key insight:
    - The vertical duplication count is not discarded noise: it is the lookup key into a length-coded legend, and the legend's spatial order is irrelevant.
    """
    _body, collapsed_rows, length_to_color = _parse_input(grid)
    return _solve_from_parsed(_body, collapsed_rows, length_to_color)


def _sample_row_pattern(width: int) -> np.ndarray:
    on_count = int(np.random.randint(2, width - 1))
    cols = np.random.choice(width, size=on_count, replace=False)
    row = np.zeros(width, dtype=int)
    row[cols] = GRAY
    return row


def _rows_signature(rows: list[np.ndarray]) -> list[tuple[int, ...]]:
    return [tuple(map(int, row.tolist())) for row in rows]


def _make_base_rows(width: int, n_rows: int, runs: list[int]) -> list[np.ndarray]:
    rows: list[np.ndarray] = [np.zeros(width, dtype=int) for _ in range(n_rows)]

    shared = _sample_row_pattern(width)
    while shared[0] == BLACK and shared[-1] == BLACK:
        shared = _sample_row_pattern(width)

    while True:
        i, j = sorted(np.random.choice(n_rows, size=2, replace=False).tolist())
        if j - i >= 2 and runs[i] != runs[j]:
            break
    rows[i] = shared.copy()
    rows[j] = shared.copy()

    used = {tuple(map(int, shared.tolist()))}
    for k in range(n_rows):
        if k in (i, j):
            continue
        for _ in range(200):
            row = _sample_row_pattern(width)
            sig = tuple(map(int, row.tolist()))
            if sig in used:
                continue
            if k > 0 and np.array_equal(row, rows[k - 1]):
                continue
            if k + 1 < n_rows and rows[k + 1].size and np.array_equal(row, rows[k + 1]):
                continue
            rows[k] = row
            used.add(sig)
            break
        else:
            raise RuntimeError("failed to sample distinct source rows")

    non_shared = [k for k in range(n_rows) if k not in (i, j)]
    if not any(row[0] == GRAY for row in rows):
        pick = int(np.random.choice(non_shared))
        rows[pick][0] = GRAY
    if not any(row[-1] == GRAY for row in rows):
        pick = int(np.random.choice(non_shared))
        rows[pick][-1] = GRAY

    for idx, row in enumerate(rows):
        assert np.any(row == GRAY)
        assert np.any(row == BLACK)
        if idx > 0:
            assert not np.array_equal(row, rows[idx - 1])

    shared_positions = [idx for idx, row in enumerate(rows) if np.array_equal(row, shared)]
    assert len(shared_positions) == 2 and runs[shared_positions[0]] != runs[shared_positions[1]]
    return rows


def _build_input(
    rows: list[np.ndarray],
    runs: list[int],
    length_to_color: dict[int, int],
) -> np.ndarray:
    seed = 0
    for idx, row in enumerate(rows):
        for col, val in enumerate(row):
            seed = (seed * 131 + (idx + 3) * (col + 5) * (int(val) + 1)) % (2**32)
    for idx, run in enumerate(runs):
        seed = (seed * 131 + (idx + 11) * (int(run) + 7)) % (2**32)
    for length in range(1, 5):
        seed = (seed * 131 + length * (int(length_to_color[length]) + 13)) % (2**32)
    rng = np.random.default_rng(seed)

    body = np.vstack([np.repeat(row[None, :], run, axis=0) for row, run in zip(rows, runs)])
    body_h, body_w = body.shape

    left_pad = int(rng.integers(0, 3))
    right_pad = int(rng.integers(0, 3))
    legend_span = 7
    min_h = max(body_h, legend_span)
    total_h = int(min_h + rng.integers(0, 3))
    total_w = int(left_pad + body_w + 1 + 4 + right_pad)

    body_top = int(rng.integers(0, total_h - body_h + 1))
    legend_top = int(rng.integers(0, total_h - legend_span + 1))
    body_left = left_pad
    legend_left = body_left + body_w + 2

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[body_top : body_top + body_h, body_left : body_left + body_w] = body

    legend_rows = np.array([legend_top + 2 * k for k in range(4)], dtype=int)
    perm = rng.permutation(4)
    for idx, length in enumerate(perm + 1):
        r = int(legend_rows[idx])
        color = int(length_to_color[int(length)])
        grid[r, legend_left : legend_left + int(length)] = color

    return grid


def _expected_output(rows: list[np.ndarray], runs: list[int], length_to_color: dict[int, int]) -> np.ndarray:
    out = np.zeros((len(rows), rows[0].shape[0]), dtype=int)
    for r, (row, run) in enumerate(zip(rows, runs)):
        out[r, row == GRAY] = int(length_to_color[int(run)])
    return out


def _validate_sample(
    grid: np.ndarray,
    expected: np.ndarray,
    rows: list[np.ndarray],
    runs: list[int],
    length_to_color: dict[int, int],
) -> bool:
    if set(runs) != {1, 2, 3, 4}:
        return False

    sig_to_runs: dict[tuple[int, ...], set[int]] = {}
    for row, run in zip(rows, runs):
        sig = tuple(map(int, row.tolist()))
        sig_to_runs.setdefault(sig, set()).add(int(run))
    if max(len(v) for v in sig_to_runs.values()) < 2:
        return False

    try:
        body, collapsed_rows, parsed_map = _parse_input(grid)
    except AssertionError:
        return False
    if parsed_map != {int(k): int(v) for k, v in length_to_color.items()}:
        return False
    parsed_rows = [row.copy() for row, _ in collapsed_rows]
    parsed_runs = [int(run) for _, run in collapsed_rows]
    if _rows_signature(parsed_rows) != _rows_signature(rows):
        return False
    if parsed_runs != [int(v) for v in runs]:
        return False
    if not np.array_equal(_solve_from_parsed(body, collapsed_rows, parsed_map), expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        width = int(np.random.randint(6, 11))
        n_rows = int(np.random.randint(5, 7))
        runs = [1, 2, 3, 4]
        while len(runs) < n_rows:
            runs.append(int(np.random.randint(1, 5)))
        np.random.shuffle(runs)

        rows = _make_base_rows(width, n_rows, runs)
        colors = np.random.choice(LEGEND_COLORS, size=4, replace=False)
        lengths = np.random.permutation([1, 2, 3, 4])
        length_to_color = {int(length): int(color) for length, color in zip(lengths, colors)}

        grid = _build_input(rows, runs, length_to_color)
        expected = _expected_output(rows, runs, length_to_color)

        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        if _validate_sample(grid, expected, rows, runs, length_to_color):
            return grid

    raise RuntimeError("failed to generate a valid sample")

from __future__ import annotations

import numpy as np


BG = 0
DOT_COLORS = (1, 2, 3)
MARKER = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_ALLOWED = frozenset(DIRS.values())
GEN_RETRIES = 600


def _point_axes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    dot_mask = np.isin(grid, DOT_COLORS)
    point_rows = [int(r) for r in np.flatnonzero(np.any(dot_mask, axis=1))]
    point_cols = [int(c) for c in np.flatnonzero(np.any(dot_mask, axis=0))]
    assert len(point_rows) in (3, 4) and len(point_cols) in (3, 4)
    assert all(b - a == 2 for a, b in zip(point_rows, point_rows[1:]))
    assert all(b - a == 2 for a, b in zip(point_cols, point_cols[1:]))
    return point_rows, point_cols


def _axis_spans(axis_len: int, point_indices: list[int]) -> list[tuple[int, int]]:
    active = set(point_indices)
    spans: list[tuple[int, int]] = []
    pos = 0
    for idx in range(axis_len):
        size = 3 if idx in active else 2
        spans.append((pos, pos + size))
        pos += size
    return spans


def _parse_grid(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], frozenset[tuple[int, int]]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    point_rows, point_cols = _point_axes(g)
    h, w = g.shape

    logical_points: dict[tuple[int, int], int] = {}
    point_mask = np.zeros_like(g, dtype=bool)
    for lr, r in enumerate(point_rows):
        for lc, c in enumerate(point_cols):
            color = int(g[r, c])
            assert color in DOT_COLORS
            logical_points[(lr, lc)] = color
            point_mask[r, c] = True

    assert not np.any(np.isin(g, DOT_COLORS) & ~point_mask)

    permissions: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for lr, r in enumerate(point_rows):
        for lc, c in enumerate(point_cols):
            marker_offsets = []
            for dr, dc in DIRS.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == MARKER:
                    marker_offsets.append((dr, dc))
            assert len(marker_offsets) <= 1
            if marker_offsets:
                dr, dc = marker_offsets[0]
                permissions[(lr, lc)] = frozenset({(-dr, -dc)})
            else:
                permissions[(lr, lc)] = ALL_ALLOWED

    marker_mask = g == MARKER
    if np.any(marker_mask):
        logical_coord_set = {(point_rows[r], point_cols[c]) for r in range(len(point_rows)) for c in range(len(point_cols))}
        for mr, mc in np.argwhere(marker_mask):
            adj = 0
            for dr, dc in DIRS.values():
                nr, nc = int(mr) + dr, int(mc) + dc
                if (nr, nc) in logical_coord_set:
                    adj += 1
            assert adj >= 1

    return logical_points, permissions, point_rows, point_cols


def _pair_records(
    lattice: np.ndarray, permissions: dict[tuple[int, int], frozenset[tuple[int, int]]]
) -> list[dict[str, object]]:
    rows, cols = lattice.shape
    records: list[dict[str, object]] = []

    for r in range(rows):
        for c in range(cols - 1):
            a = (r, c)
            b = (r, c + 1)
            same = int(lattice[a]) == int(lattice[b])
            a_ok = (0, 1) in permissions[a]
            b_ok = (0, -1) in permissions[b]
            records.append(
                {
                    "axis": "h",
                    "a": a,
                    "b": b,
                    "same": same,
                    "connect": same and a_ok and b_ok,
                    "one_sided": same and (a_ok ^ b_ok),
                }
            )

    for r in range(rows - 1):
        for c in range(cols):
            a = (r, c)
            b = (r + 1, c)
            same = int(lattice[a]) == int(lattice[b])
            a_ok = (1, 0) in permissions[a]
            b_ok = (-1, 0) in permissions[b]
            records.append(
                {
                    "axis": "v",
                    "a": a,
                    "b": b,
                    "same": same,
                    "connect": same and a_ok and b_ok,
                    "one_sided": same and (a_ok ^ b_ok),
                }
            )
    return records


def _render_from_abstract(
    lattice: np.ndarray, permissions: dict[tuple[int, int], frozenset[tuple[int, int]]]
) -> np.ndarray:
    rows, cols = lattice.shape
    out = np.zeros((5 * rows + 2, 5 * cols + 2), dtype=int)

    for r in range(rows):
        for c in range(cols):
            color = int(lattice[r, c])
            rr = 2 + 5 * r
            cc = 2 + 5 * c
            out[rr : rr + 3, cc : cc + 3] = color

    for rec in _pair_records(lattice, permissions):
        if not rec["connect"]:
            continue
        r0, c0 = rec["a"]
        color = int(lattice[r0, c0])
        rr = 2 + 5 * r0
        cc = 2 + 5 * c0
        if rec["axis"] == "h":
            out[rr : rr + 3, cc + 3 : cc + 5] = color
        else:
            out[rr + 3 : rr + 5, cc : cc + 3] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a marked sparse dot lattice into large color blocks while connecting only bidirectionally permitted same-color neighbors.

    Input structure:
    - The grid has a black background and a 3x3 or 4x4 logical lattice of colored dots on rows/cols separated by black spacer rows/cols.
    - Some dots have a single gray marker in one cardinal-adjacent spacer cell.
    - Dot colors come from blue, red, and green; gray markers are not dots and are removed in the output.

    Rule:
    - Recover the logical dot rows/cols from the colored pixels.
    - A dot with no gray marker permits all four connection directions; with one marker it permits only the direction opposite that marker.
    - Connect a horizontal or vertical neighboring dot pair iff the two dots have the same color and both dots permit the direction toward the other.
    - Output the enlarged lattice: every dot becomes a solid 3x3 block, every spacer row/col becomes thickness 2, and only permitted same-color adjacencies fill their enlarged gap with a same-color bar.

    Key insight:
    - Gray markers do not name a destination cell; they gate connectivity by restricting a dot to exactly one allowed direction, so each visible bar requires a same-color match plus mutual directional permission.
    """
    g = np.asarray(grid, dtype=int)
    logical_points, permissions, point_rows, point_cols = _parse_grid(g)
    lattice = np.array(
        [[logical_points[(r, c)] for c in range(len(point_cols))] for r in range(len(point_rows))],
        dtype=int,
    )

    row_spans = _axis_spans(g.shape[0], point_rows)
    col_spans = _axis_spans(g.shape[1], point_cols)
    out = np.zeros((row_spans[-1][1], col_spans[-1][1]), dtype=int)

    for lr, r in enumerate(point_rows):
        r0, r1 = row_spans[r]
        for lc, c in enumerate(point_cols):
            c0, c1 = col_spans[c]
            out[r0:r1, c0:c1] = logical_points[(lr, lc)]

    for rec in _pair_records(lattice, permissions):
        if not rec["connect"]:
            continue
        (lr0, lc0) = rec["a"]
        if rec["axis"] == "h":
            r = point_rows[lr0]
            c_left = point_cols[lc0]
            c_right = point_cols[lc0 + 1]
            r0, r1 = row_spans[r]
            fill_c0 = col_spans[c_left][1]
            fill_c1 = col_spans[c_right][0]
            out[r0:r1, fill_c0:fill_c1] = lattice[lr0, lc0]
        else:
            r_top = point_rows[lr0]
            r_bottom = point_rows[lr0 + 1]
            c = point_cols[lc0]
            fill_r0 = row_spans[r_top][1]
            fill_r1 = row_spans[r_bottom][0]
            c0, c1 = col_spans[c]
            out[fill_r0:fill_r1, c0:c1] = lattice[lr0, lc0]

    return out


def _build_input(lattice: np.ndarray, markers: dict[tuple[int, int], str]) -> np.ndarray:
    rows, cols = lattice.shape
    grid = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=int)
    grid[1::2, 1::2] = lattice
    for (r, c), direction in markers.items():
        dr, dc = DIRS[direction]
        pr, pc = 1 + 2 * r, 1 + 2 * c
        assert grid[pr + dr, pc + dc] == BG
        grid[pr + dr, pc + dc] = MARKER
    return grid


def _abstract_permissions(
    shape: tuple[int, int], markers: dict[tuple[int, int], str]
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    rows, cols = shape
    permissions: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            direction = markers.get((r, c))
            if direction is None:
                permissions[(r, c)] = ALL_ALLOWED
            else:
                dr, dc = DIRS[direction]
                permissions[(r, c)] = frozenset({(-dr, -dc)})
    return permissions


def _horizontal_pairs(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1)]


def _vertical_pairs(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols)]


def _blocked_horizontal_choices(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], str]]:
    choices = []
    for r, direction in ((0, "up"), (rows - 1, "down")):
        for c in range(cols - 1):
            a = (r, c)
            b = (r, c + 1)
            choices.append((a, b, a, direction))
            choices.append((a, b, b, direction))
    return choices


def _blocked_vertical_choices(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], str]]:
    choices = []
    for c, direction in ((0, "left"), (cols - 1, "right")):
        for r in range(rows - 1):
            a = (r, c)
            b = (r + 1, c)
            choices.append((a, b, a, direction))
            choices.append((a, b, b, direction))
    return choices


def _marker_is_valid(shape: tuple[int, int], point: tuple[int, int], direction: str) -> bool:
    rows, cols = shape
    r, c = point
    return (
        (direction == "up" and r == 0)
        or (direction == "down" and r == rows - 1)
        or (direction == "left" and c == 0)
        or (direction == "right" and c == cols - 1)
    )


def _connection_signature(
    lattice: np.ndarray, markers: dict[tuple[int, int], str]
) -> tuple[tuple[str, tuple[int, int], tuple[int, int], bool], ...]:
    permissions = _abstract_permissions(lattice.shape, markers)
    return tuple(
        sorted((str(rec["axis"]), rec["a"], rec["b"], bool(rec["connect"])) for rec in _pair_records(lattice, permissions))
    )


def _choose_special_pairs(
    rows: int, cols: int, rng: np.random.Generator
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int], str],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int], str],
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
] | None:
    h_choices = _blocked_horizontal_choices(rows, cols)
    v_choices = _blocked_vertical_choices(rows, cols)
    rng.shuffle(h_choices)
    rng.shuffle(v_choices)

    for h_block in h_choices:
        h_endpoints = {h_block[0], h_block[1]}
        for v_block in v_choices:
            v_endpoints = {v_block[0], v_block[1]}
            if h_endpoints & v_endpoints:
                continue
            blocked_cells = h_endpoints | v_endpoints
            h_connect_options = [pair for pair in _horizontal_pairs(rows, cols) if set(pair).isdisjoint(blocked_cells)]
            v_connect_options = [pair for pair in _vertical_pairs(rows, cols) if set(pair).isdisjoint(blocked_cells)]
            if not h_connect_options or not v_connect_options:
                continue
            h_conn = h_connect_options[int(rng.integers(0, len(h_connect_options)))]
            v_conn = v_connect_options[int(rng.integers(0, len(v_connect_options)))]
            return h_block, v_block, h_conn, v_conn
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.choice([3, 4]))
    cols = int(rng.choice([3, 4]))
    special = _choose_special_pairs(rows, cols, rng)
    if special is None:
        return None
    h_block, v_block, h_conn, v_conn = special

    palette_size = 2 if float(rng.random()) < 0.6 else 3
    palette = rng.choice(np.array(DOT_COLORS), size=palette_size, replace=False)
    lattice = rng.choice(palette, size=(rows, cols), replace=True)

    color_h_block = int(rng.choice(palette))
    color_v_block = int(rng.choice(palette))
    color_connected = int(rng.choice(palette))

    for point in h_block[:2]:
        lattice[point] = color_h_block
    for point in v_block[:2]:
        lattice[point] = color_v_block
    for point in h_conn:
        lattice[point] = color_connected
    for point in v_conn:
        lattice[point] = color_connected

    if len(set(map(int, lattice.flatten()))) < 2:
        return None

    markers = {
        h_block[2]: h_block[3],
        v_block[2]: v_block[3],
    }
    if any(not _marker_is_valid(lattice.shape, point, direction) for point, direction in markers.items()):
        return None

    permissions = _abstract_permissions(lattice.shape, markers)
    records = _pair_records(lattice, permissions)
    marked_points = set(markers)

    if not any(rec["axis"] == "h" and rec["one_sided"] for rec in records):
        return None
    if not any(rec["axis"] == "v" and rec["one_sided"] for rec in records):
        return None
    if not any(
        rec["axis"] == "h"
        and rec["connect"]
        and rec["a"] not in marked_points
        and rec["b"] not in marked_points
        for rec in records
    ):
        return None
    if not any(
        rec["axis"] == "v"
        and rec["connect"]
        and rec["a"] not in marked_points
        and rec["b"] not in marked_points
        for rec in records
    ):
        return None

    base_sig = _connection_signature(lattice, markers)
    for point in markers:
        reduced = dict(markers)
        reduced.pop(point)
        if _connection_signature(lattice, reduced) == base_sig:
            return None

    inp = _build_input(lattice, markers)
    expected = _render_from_abstract(lattice, permissions)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

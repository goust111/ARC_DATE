import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 18
GRID_MAX = 26
GEN_RETRIES = 400


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("path segment must be orthogonal")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    assert len(cells) == len(set(cells)), "path may not self-overlap"
    return cells


def _path_orientation_and_index(
    seq: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], int]]:
    orientation: dict[tuple[int, int], str] = {}
    index = {cell: i for i, cell in enumerate(seq)}
    for i, cur in enumerate(seq):
        if i == 0:
            nxt = seq[1]
            orientation[cur] = "H" if nxt[0] == cur[0] else "V"
            continue
        if i == len(seq) - 1:
            prv = seq[-2]
            orientation[cur] = "H" if prv[0] == cur[0] else "V"
            continue

        prv = seq[i - 1]
        nxt = seq[i + 1]
        if prv[0] == nxt[0]:
            orientation[cur] = "H"
        elif prv[1] == nxt[1]:
            orientation[cur] = "V"
        else:
            orientation[cur] = "C"
    return orientation, index


def _marker_cells(
    side: str,
    attach: tuple[int, int],
    offset: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    r, c = attach
    if side == "L":
        top = r - offset
        return [(top, 0), (top, 1), (top + 1, 0), (top + 1, 1)]
    if side == "R":
        top = r - offset
        return [(top, w - 2), (top, w - 1), (top + 1, w - 2), (top + 1, w - 1)]
    if side == "T":
        left = c - offset
        return [(0, left), (0, left + 1), (1, left), (1, left + 1)]
    if side == "B":
        left = c - offset
        return [(h - 2, left), (h - 2, left + 1), (h - 1, left), (h - 1, left + 1)]
    raise AssertionError(f"unknown side: {side}")


def _pick_spaced(cands: list[int], used: list[int], gap: int) -> int | None:
    order = list(cands)
    random.shuffle(order)
    for value in order:
        if all(abs(value - other) >= gap for other in used):
            return value
    return None


def _detect_marker_windows(grid: np.ndarray) -> dict[int, list[set[tuple[int, int]]]]:
    h, w = grid.shape
    markers: dict[int, list[set[tuple[int, int]]]] = defaultdict(list)
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color == BG:
                continue
            if (
                int(grid[r + 1, c]) == color
                and int(grid[r, c + 1]) == color
                and int(grid[r + 1, c + 1]) == color
            ):
                markers[color].append({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)})
    return markers


def _reconstruct_path(
    grid: np.ndarray,
    color: int,
    marker_windows: list[set[tuple[int, int]]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    marker_cells = set().union(*marker_windows)
    visible = {tuple(x) for x in np.argwhere(grid == color)} - marker_cells

    attachments: list[tuple[int, int]] = []
    for marker in marker_windows:
        adjacent = {
            nbr
            for cell in marker
            for nbr in _neighbors4(cell, h, w)
            if nbr in visible and nbr not in marker
        }
        assert len(adjacent) == 1, "each 2x2 endpoint marker must touch exactly one path cell"
        attachments.append(next(iter(adjacent)))
    assert len(attachments) == 2, "each path color must have exactly two endpoint markers"

    hidden: dict[tuple[int, int], str] = {}
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in marker_cells or int(grid[cell]) in (BG, color):
                continue

            hor = c - 1 >= 0 and c + 1 < w and (r, c - 1) in visible and (r, c + 1) in visible
            ver = r - 1 >= 0 and r + 1 < h and (r - 1, c) in visible and (r + 1, c) in visible
            if hor ^ ver:
                hidden[cell] = "H" if hor else "V"

    nodes = visible | set(hidden)
    adjacency = {cell: set() for cell in nodes}
    for cell in nodes:
        r, c = cell
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nbr in nodes:
                adjacency[cell].add(nbr)

    attachment_set = set(attachments)
    for att in attachments:
        assert len(adjacency[att]) == 1, "path endpoint must have degree 1"
    for cell, nbrs in adjacency.items():
        if cell in attachment_set:
            continue
        assert len(nbrs) == 2, "path interior must have degree 2"
    for cell, axis in hidden.items():
        nbrs = sorted(adjacency[cell])
        if axis == "H":
            assert nbrs[0][0] == cell[0] == nbrs[1][0], "hidden crossing must continue straight horizontally"
        else:
            assert nbrs[0][1] == cell[1] == nbrs[1][1], "hidden crossing must continue straight vertically"

    start, end = attachments
    seq = [start]
    prev = None
    cur = start
    while cur != end:
        next_cells = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(next_cells) == 1, "path graph must be a single chain"
        nxt = next_cells[0]
        seq.append(nxt)
        prev, cur = cur, nxt
        assert len(seq) <= len(nodes) + 1, "path traversal loop detected"

    assert set(seq) == nodes, "reconstructed path must cover all visible and hidden path cells"
    return seq, {cell: i for i, cell in enumerate(seq)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each endpoint-marked orthogonal path and grey the hidden underpass spans at crossings.

    Input structure:
    - The background is black.
    - Several one-cell-wide colored paths run horizontally and vertically, and a path may turn only by 90 degrees.
    - Every path color has exactly two solid 2x2 endpoint markers of the same color.
    - When two paths cross, the shared input pixel shows only the top path's color while the lower path continues straight underneath.

    Rule:
    - Use the two 2x2 endpoint markers to identify each path color and its two attachment cells.
    - Reconstruct the full path by joining same-color visible cells and restoring any hidden crossing pixel where the pattern is color / other-color / color in a straight line.
    - At every reconstructed crossing, the input color at the shared pixel is the top path; the other path is the underpass.
    - Grey the underpass span consisting of the crossing cell and its immediate predecessor and successor along that underpath, while the visible top color remains at the shared crossing pixel.

    Key insight:
    - The 2x2 blocks identify which separated same-color fragments belong to one path, and the straight-through color / other-color / color pattern reveals where a path is hidden under another before the short grey bridge-shadow span can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_windows = _detect_marker_windows(g)
    path_colors = [color for color, windows in marker_windows.items() if len(windows) == 2]
    assert path_colors, "expected at least one endpoint-marked path color"

    paths: dict[int, dict[str, object]] = {}
    for color in path_colors:
        seq, index = _reconstruct_path(g, color, marker_windows[color])
        paths[color] = {"seq": seq, "index": index}

    cover: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, data in paths.items():
        for cell in data["seq"]:
            cover[cell].append(color)

    out = g.copy()
    restore: list[tuple[tuple[int, int], int]] = []
    for cell, colors in cover.items():
        assert len(colors) <= 2, "generator and solver expect at most two paths per crossing"
        if len(colors) != 2:
            continue

        top = int(g[cell])
        assert top in colors and top != BG, "shared crossing pixel must show the top path color"
        under = colors[1] if colors[0] == top else colors[0]

        seq = paths[under]["seq"]
        index = paths[under]["index"]
        i = index[cell]
        assert 0 < i < len(seq) - 1, "crossing must be interior to the underpath"

        prev_cell = seq[i - 1]
        next_cell = seq[i + 1]
        assert prev_cell[0] == next_cell[0] or prev_cell[1] == next_cell[1], "underpath must pass straight through a crossing"

        out[prev_cell] = GREY
        out[cell] = GREY
        out[next_cell] = GREY
        restore.append((cell, top))

    for cell, top in restore:
        out[cell] = top
    return out


def _build_candidate_paths(h: int, w: int) -> list[dict[str, object]] | None:
    n_lr = random.randint(1, 2)
    n_tb = random.randint(1, 2)
    colors = random.sample(PALETTE, n_lr + n_tb)

    row_cands = list(range(4, h - 4))
    col_cands = list(range(4, w - 4))

    left_used: list[int] = []
    right_used: list[int] = []
    top_used: list[int] = []
    bottom_used: list[int] = []

    paths: list[dict[str, object]] = []
    color_idx = 0

    for _ in range(n_lr):
        start_row = _pick_spaced(row_cands, left_used, gap=3)
        if start_row is None:
            return None
        left_used.append(start_row)

        wants_straight = random.random() < 0.4
        end_row = _pick_spaced([start_row], right_used, gap=3) if wants_straight else None
        if end_row is None:
            wants_straight = False
            end_row = _pick_spaced([r for r in row_cands if abs(r - start_row) >= 3], right_used, gap=3)
        if end_row is None:
            return None
        right_used.append(end_row)

        start = (start_row, 2)
        end = (end_row, w - 3)
        if wants_straight:
            points = [start, end]
        else:
            turn_col = random.randint(5, w - 6)
            points = [start, (start_row, turn_col), (end_row, turn_col), end]

        paths.append(
            {
                "color": colors[color_idx],
                "side_a": "L",
                "side_b": "R",
                "attach_a": start,
                "attach_b": end,
                "off_a": random.randint(0, 1),
                "off_b": random.randint(0, 1),
                "seq": _polyline_cells(points),
            }
        )
        color_idx += 1

    for _ in range(n_tb):
        start_col = _pick_spaced(col_cands, top_used, gap=3)
        if start_col is None:
            return None
        top_used.append(start_col)

        wants_straight = random.random() < 0.4
        end_col = _pick_spaced([start_col], bottom_used, gap=3) if wants_straight else None
        if end_col is None:
            wants_straight = False
            end_col = _pick_spaced([c for c in col_cands if abs(c - start_col) >= 3], bottom_used, gap=3)
        if end_col is None:
            return None
        bottom_used.append(end_col)

        start = (2, start_col)
        end = (h - 3, end_col)
        if wants_straight:
            points = [start, end]
        else:
            turn_row = random.randint(5, h - 6)
            points = [start, (turn_row, start_col), (turn_row, end_col), end]

        paths.append(
            {
                "color": colors[color_idx],
                "side_a": "T",
                "side_b": "B",
                "attach_a": start,
                "attach_b": end,
                "off_a": random.randint(0, 1),
                "off_b": random.randint(0, 1),
                "seq": _polyline_cells(points),
            }
        )
        color_idx += 1

    return paths


def _render_candidate(
    h: int,
    w: int,
    paths: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    marker_owner: dict[tuple[int, int], int] = {}
    markers_by_color: dict[int, set[tuple[int, int]]] = {}
    cover: dict[tuple[int, int], list[int]] = defaultdict(list)
    seq_by_color: dict[int, list[tuple[int, int]]] = {}
    orientation_by_color: dict[int, dict[tuple[int, int], str]] = {}
    index_by_color: dict[int, dict[tuple[int, int], int]] = {}
    attachment_by_color: dict[int, set[tuple[int, int]]] = {}
    cell_sets: dict[int, set[tuple[int, int]]] = {}

    for path in paths:
        color = int(path["color"])
        seq = list(path["seq"])
        seq_by_color[color] = seq
        cell_sets[color] = set(seq)
        attachment_by_color[color] = {path["attach_a"], path["attach_b"]}
        orientation_by_color[color], index_by_color[color] = _path_orientation_and_index(seq)

        markers = set(
            _marker_cells(path["side_a"], path["attach_a"], int(path["off_a"]), h, w)
            + _marker_cells(path["side_b"], path["attach_b"], int(path["off_b"]), h, w)
        )
        if len(markers) != 8:
            return None
        if any(cell in marker_owner for cell in markers):
            return None
        if any(cell in markers for cell in seq):
            return None

        markers_by_color[color] = markers
        for cell in markers:
            marker_owner[cell] = color

        for attach, side, offset in (
            (path["attach_a"], path["side_a"], int(path["off_a"])),
            (path["attach_b"], path["side_b"], int(path["off_b"])),
        ):
            local_marker = set(_marker_cells(side, attach, offset, h, w))
            adjacent = {
                nbr
                for cell in local_marker
                for nbr in _neighbors4(cell, h, w)
                if nbr not in local_marker
            }
            if attach not in adjacent or len(cell_sets[color] & adjacent) != 1:
                return None

        for cell in seq:
            cover[cell].append(color)

    for color, cells in cell_sets.items():
        for cell in cells:
            for nbr in _neighbors4(cell, h, w):
                if nbr in marker_owner and marker_owner[nbr] != color:
                    return None
                if nbr in marker_owner and marker_owner[nbr] == color and cell not in attachment_by_color[color]:
                    return None

    crossings: list[tuple[tuple[int, int], int, int]] = []
    for cell, colors in cover.items():
        if len(colors) > 2:
            return None
        if len(colors) != 2:
            continue

        a, b = colors
        ia = index_by_color[a][cell]
        ib = index_by_color[b][cell]
        oa = orientation_by_color[a][cell]
        ob = orientation_by_color[b][cell]
        if oa not in ("H", "V") or ob not in ("H", "V") or oa == ob:
            return None
        if ia <= 1 or ia >= len(seq_by_color[a]) - 2 or ib <= 1 or ib >= len(seq_by_color[b]) - 2:
            return None
        if len(cover[seq_by_color[a][ia - 1]]) != 1 or len(cover[seq_by_color[a][ia + 1]]) != 1:
            return None
        if len(cover[seq_by_color[b][ib - 1]]) != 1 or len(cover[seq_by_color[b][ib + 1]]) != 1:
            return None
        crossings.append((cell, a, b))

    if not crossings:
        return None

    singles = {cell: colors[0] for cell, colors in cover.items() if len(colors) == 1}
    for cell, color in singles.items():
        for nbr in _neighbors4(cell, h, w):
            if nbr in singles and singles[nbr] != color:
                return None

    inp = np.full((h, w), BG, dtype=int)
    for color, marker_cells in markers_by_color.items():
        for cell in marker_cells:
            inp[cell] = color
    for color, seq in seq_by_color.items():
        for cell in seq:
            inp[cell] = color

    top_by_cross: dict[tuple[int, int], int] = {}
    for cell, a, b in crossings:
        top = random.choice((a, b))
        top_by_cross[cell] = top
        inp[cell] = top

    expected = inp.copy()
    for cell, a, b in crossings:
        top = top_by_cross[cell]
        under = b if top == a else a
        seq = seq_by_color[under]
        i = index_by_color[under][cell]
        expected[seq[i - 1]] = GREY
        expected[cell] = GREY
        expected[seq[i + 1]] = GREY
    for cell, top in top_by_cross.items():
        expected[cell] = top

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    paths = _build_candidate_paths(h, w)
    if paths is None:
        return None

    rendered = _render_candidate(h, w, paths)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        got = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

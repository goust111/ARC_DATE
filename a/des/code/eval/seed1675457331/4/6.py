from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _is_path_color(value: int) -> bool:
    return int(value) not in (BG, GREY, MAROON)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every grey crossing pixel to the horizontal path color and remove maroon endpoint markers.

    Input structure:
    - Black background.
    - Thin orthogonal colored paths.
    - Grey pixels mark path crossings.
    - Maroon pixels mark path endpoints and disappear in the output.

    Rule:
    - At each grey crossing, the left/right path owns the crossing pixel.
    - Remove every maroon marker to background.
    - Leave all other pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    out[out == MAROON] = BG

    for r, c in np.argwhere(g == GREY):
        r = int(r)
        c = int(c)
        assert 0 < r < h - 1 and 0 < c < w - 1, "crossings must be interior pixels"

        up = int(g[r - 1, c])
        down = int(g[r + 1, c])
        left = int(g[r, c - 1])
        right = int(g[r, c + 1])

        assert up == down, "vertical crossing neighbors must match"
        assert left == right, "horizontal crossing neighbors must match"
        assert up != left, "the two crossing paths must have different colors"
        assert _is_path_color(up) and _is_path_color(left), "crossing neighbors must be path colors"

        out[r, c] = left

    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _sample_spaced_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
    min_gap: int,
    banned: tuple[int, ...] = (),
    attempts: int = 200,
) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(int(low), int(high) + 1))
    if len(values) < count:
        return None
    for _ in range(attempts):
        rng.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if any(abs(value - prev) < min_gap for prev in chosen):
                continue
            if any(abs(value - prev) < min_gap for prev in banned):
                continue
            chosen.append(int(value))
            if len(chosen) == count:
                rng.shuffle(chosen)
                return chosen
    return None


def _far_from(values: tuple[int, ...], value: int, min_gap: int = 2) -> bool:
    return all(abs(int(value) - int(other)) >= min_gap for other in values)


def _build_h_path(
    rng: np.random.Generator,
    rows: tuple[int, int],
    turn_col: int,
    width: int,
    vertical_cols: tuple[int, ...],
) -> dict | None:
    r0, r1 = rows
    left_choices = [
        c
        for c in range(1, int(turn_col) - 1)
        if _far_from(vertical_cols, c)
    ]
    right_choices = [
        c
        for c in range(int(turn_col) + 2, width - 1)
        if _far_from(vertical_cols, c)
    ]
    if not left_choices or not right_choices:
        return None

    c_left = int(rng.choice(left_choices))
    c_right = int(rng.choice(right_choices))
    segments = [
        ((int(r0), c_left), (int(r0), int(turn_col)), "h"),
        ((int(r0), int(turn_col)), (int(r1), int(turn_col)), "v"),
        ((int(r1), int(turn_col)), (int(r1), c_right), "h"),
    ]
    endpoints = [((int(r0), c_left), "h"), ((int(r1), c_right), "h")]
    return {"segments": segments, "endpoints": endpoints}


def _build_v_path(
    rng: np.random.Generator,
    turn_row: int,
    cols: tuple[int, int],
    height: int,
    horizontal_rows: tuple[int, ...],
) -> dict | None:
    c0, c1 = cols
    top_choices = [
        r
        for r in range(1, int(turn_row) - 1)
        if _far_from(horizontal_rows, r)
    ]
    bottom_choices = [
        r
        for r in range(int(turn_row) + 2, height - 1)
        if _far_from(horizontal_rows, r)
    ]
    if not top_choices or not bottom_choices:
        return None

    r_top = int(rng.choice(top_choices))
    r_bottom = int(rng.choice(bottom_choices))
    segments = [
        ((r_top, int(c0)), (int(turn_row), int(c0)), "v"),
        ((int(turn_row), int(c0)), (int(turn_row), int(c1)), "h"),
        ((int(turn_row), int(c1)), (r_bottom, int(c1)), "v"),
    ]
    endpoints = [((r_top, int(c0)), "v"), ((r_bottom, int(c1)), "v")]
    return {"segments": segments, "endpoints": endpoints}


def _path_cell_orients(path: dict) -> dict[tuple[int, int], set[str]]:
    cell_orients: dict[tuple[int, int], set[str]] = {}
    for start, end, orient in path["segments"]:
        for cell in _segment_cells(start, end):
            cell_orients.setdefault(cell, set()).add(orient)
    return cell_orients


def _collect_layout(paths: list[dict]) -> tuple[dict[tuple[int, int], list[tuple[int, set[str]]]], list[int]]:
    cell_map: dict[tuple[int, int], list[tuple[int, set[str]]]] = {}
    crossing_cells: list[int] = []
    for pid, path in enumerate(paths):
        for cell, orients in path["cell_orients"].items():
            cell_map.setdefault(cell, []).append((pid, orients))

    for cell, entries in cell_map.items():
        if len(entries) == 1:
            continue
        if len(entries) != 2:
            raise AssertionError(f"expected at most two paths per cell, got {len(entries)} at {cell}")
        a_pid, a_orients = entries[0]
        b_pid, b_orients = entries[1]
        assert a_pid != b_pid
        if len(a_orients) != 1 or len(b_orients) != 1:
            raise AssertionError(f"crossing cannot occur at a path corner: {cell}")
        if a_orients == b_orients:
            raise AssertionError(f"same-orientation overlap is invalid: {cell}")
        crossing_cells.append(1)
    return cell_map, crossing_cells


def _has_illegal_adjacency(cell_map: dict[tuple[int, int], list[tuple[int, set[str]]]]) -> bool:
    occupied = set(cell_map)
    pid_sets = {cell: {pid for pid, _ in entries} for cell, entries in cell_map.items()}
    for r, c in occupied:
        here = pid_sets[(r, c)]
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr not in occupied:
                continue
            if here.isdisjoint(pid_sets[nbr]):
                return True
    return False


def _choose_markers(
    rng: np.random.Generator,
    paths: list[dict],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> set[tuple[int, int]] | None:
    markers: set[tuple[int, int]] = set()
    for path in paths:
        for endpoint, orient in path["endpoints"]:
            r, c = endpoint
            if orient == "h":
                options = [(r - 1, c), (r + 1, c)]
            else:
                options = [(r, c - 1), (r, c + 1)]
            rng.shuffle(options)
            chosen = None
            for mr, mc in options:
                if not (0 <= mr < height and 0 <= mc < width):
                    continue
                if (mr, mc) in occupied or (mr, mc) in markers:
                    continue
                ok = True
                for dr, dc in DIR4:
                    nbr = (mr + dr, mc + dc)
                    if nbr == endpoint:
                        continue
                    if nbr in occupied or nbr in markers:
                        ok = False
                        break
                if ok:
                    chosen = (mr, mc)
                    break
            if chosen is None:
                return None
            markers.add(chosen)
    return markers


def _render_pair(
    height: int,
    width: int,
    paths: list[dict],
    cell_map: dict[tuple[int, int], list[tuple[int, set[str]]]],
    markers: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for cell, entries in cell_map.items():
        r, c = cell
        if len(entries) == 1:
            pid = entries[0][0]
            color = int(paths[pid]["color"])
            inp[r, c] = color
            out[r, c] = color
            continue

        horiz_pid = None
        for pid, orients in entries:
            if "h" in orients:
                horiz_pid = int(pid)
                break
        assert horiz_pid is not None
        inp[r, c] = GREY
        out[r, c] = int(paths[horiz_pid]["color"])

    for r, c in markers:
        inp[r, c] = MAROON

    return inp, out


def _build_paths(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_h, n_v = rng.choice(
        np.array([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)], dtype=int)
    )
    n_h = int(n_h)
    n_v = int(n_v)
    n_paths = n_h + n_v

    height = int(rng.integers(19, 27))
    width = int(rng.integers(19, 27))

    h_rows = _sample_spaced_positions(rng, 2, height - 3, 2 * n_h, min_gap=2)
    if h_rows is None:
        return None
    v_turn_rows = _sample_spaced_positions(rng, 4, height - 5, n_v, min_gap=2, banned=tuple(h_rows))
    if v_turn_rows is None:
        return None

    v_cols = _sample_spaced_positions(rng, 2, width - 3, 2 * n_v, min_gap=2)
    if v_cols is None:
        return None
    h_turn_cols = _sample_spaced_positions(rng, 4, width - 5, n_h, min_gap=2, banned=tuple(v_cols))
    if h_turn_cols is None:
        return None

    colors = rng.choice(np.array(PATH_COLORS, dtype=int), size=n_paths, replace=False).tolist()
    paths: list[dict] = []

    horizontal_rows = tuple(int(x) for x in h_rows + v_turn_rows)
    vertical_cols = tuple(int(x) for x in v_cols + h_turn_cols)

    for i in range(n_h):
        path = _build_h_path(
            rng,
            (int(h_rows[2 * i]), int(h_rows[2 * i + 1])),
            int(h_turn_cols[i]),
            width,
            vertical_cols,
        )
        if path is None:
            return None
        path["color"] = int(colors[len(paths)])
        path["cell_orients"] = _path_cell_orients(path)
        paths.append(path)

    for i in range(n_v):
        path = _build_v_path(
            rng,
            int(v_turn_rows[i]),
            (int(v_cols[2 * i]), int(v_cols[2 * i + 1])),
            height,
            horizontal_rows,
        )
        if path is None:
            return None
        path["color"] = int(colors[len(paths)])
        path["cell_orients"] = _path_cell_orients(path)
        paths.append(path)

    cell_map, crossing_flags = _collect_layout(paths)
    crossing_count = len(crossing_flags)
    if crossing_count < 2 or crossing_count > 14:
        return None
    if _has_illegal_adjacency(cell_map):
        return None

    markers = _choose_markers(rng, paths, set(cell_map), height, width)
    if markers is None:
        return None

    inp, out = _render_pair(height, width, paths, cell_map, markers)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        result = _build_paths(rng)
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid input")

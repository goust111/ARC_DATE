from __future__ import annotations

import random

import numpy as np

Coord = tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _build_adjacency(mask: np.ndarray) -> dict[Coord, list[Coord]]:
    h, w = mask.shape
    adj: dict[Coord, list[Coord]] = {}
    for r, c in map(tuple, np.argwhere(mask)):
        cell = (int(r), int(c))
        adj[cell] = [nb for nb in _neighbors4(cell[0], cell[1], h, w) if bool(mask[nb])]
    return adj


def _is_start_leaf(cell: Coord, adj: dict[Coord, list[Coord]], g: np.ndarray) -> bool:
    if len(adj[cell]) != 1:
        return False
    start = adj[cell][0]
    if len(adj[start]) != 2:
        return False
    other = adj[start][0] if adj[start][1] == cell else adj[start][1]
    v1 = (cell[0] - start[0], cell[1] - start[1])
    v2 = (other[0] - start[0], other[1] - start[1])
    if v1[0] * v2[0] + v1[1] * v2[1] != 0:
        return False
    return int(g[cell]) == int(g[start])


def _trace_centerline(start_leaf: Coord, adj: dict[Coord, list[Coord]]) -> list[Coord]:
    prev = start_leaf
    cur = adj[start_leaf][0]
    cells = [cur]
    seen: set[tuple[Coord, Coord]] = set()

    while True:
        nbs = adj[cur]
        deg = len(nbs)
        if deg == 1:
            break
        if deg == 2:
            nxt = nbs[0] if nbs[1] == prev else nbs[1]
        else:
            assert deg == 4, f"unexpected degree {deg} at {cur}"
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            nxt = (cur[0] + dr, cur[1] + dc)
            assert nxt in nbs, f"crossing at {cur} must continue straight"
        state = (prev, cur)
        assert state not in seen, "path trace looped unexpectedly"
        seen.add(state)
        prev, cur = cur, nxt
        cells.append(cur)
        assert len(cells) <= len(adj), "path trace exceeded foreground size"
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each crossed path to the original color indicated by its 2-cell start marker.

    Input structure
    - Black is background.
    - Foreground consists of one-pixel orthogonal paths that may turn and cross.
    - Exactly one endpoint of each path has an extra same-color adjacent pixel forming a 2-cell start marker.
    - Path colors may change after crossings due to contamination transfer.

    Rule
    - Find each path by starting from its marker leaf and tracing through the foreground topology.
    - Ignore contaminated segment colors and repaint the full traced path with the marker's color.
    - At shared crossing cells, later paths overwrite earlier ones, ordered by the row-major position of the true start endpoint.

    Key insight
    - The start marker is the only reliable identity cue; topology recovers the path, not the visible segment colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mask = g != 0
    adj = _build_adjacency(mask)
    assert adj, "expected at least one foreground path"

    for cell, nbs in adj.items():
        deg = len(nbs)
        assert deg in (1, 2, 4), f"unexpected foreground branching degree {deg} at {cell}"

    start_leaves = [cell for cell in adj if _is_start_leaf(cell, adj, g)]
    assert start_leaves, "expected at least one start marker"

    paths: list[tuple[Coord, Coord, list[Coord], int]] = []
    covered: set[Coord] = set()
    for leaf in start_leaves:
        start = adj[leaf][0]
        centerline = _trace_centerline(leaf, adj)
        color = int(g[leaf])
        paths.append((start, leaf, centerline, color))
        covered.add(leaf)
        covered.update(centerline)

    assert covered == set(adj), "every foreground cell must belong to a traced path"
    starts = [start for start, _, _, _ in paths]
    assert len(set(starts)) == len(starts), "each path must have a unique start"

    out = np.zeros_like(g)
    for start, leaf, centerline, color in sorted(paths, key=lambda item: item[0]):
        out[leaf] = color
        for cell in centerline:
            out[cell] = color
    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
N_PATHS_RANGE = (2, 5)
HEIGHT_RANGE = (13, 25)
WIDTH_RANGE = (12, 24)
GEN_ATTEMPTS = 400

TRANSFORMS = (
    "identity",
    "flip_h",
    "flip_v",
    "rot180",
    "transpose",
    "rot90",
    "rot270",
    "anti_transpose",
)


def _centerline_cells(start_row: int, end_row: int, turn_col: int, width: int) -> list[Coord]:
    cells = [(start_row, c) for c in range(turn_col + 1)]
    step = 1 if end_row > start_row else -1
    for r in range(start_row + step, end_row + step, step):
        cells.append((r, turn_col))
    for c in range(turn_col + 1, width):
        cells.append((end_row, c))
    return cells


def _sample_marker_sides(start_rows: list[int]) -> list[int] | None:
    for _ in range(80):
        sides = [random.choice((-1, 1)) for _ in start_rows]
        marker_rows = [r + side for r, side in zip(start_rows, sides, strict=True)]
        if len(set(marker_rows)) == len(marker_rows):
            return sides
    return None


def _path_specs(height: int, width: int, n_paths: int) -> list[dict] | None:
    odd_rows = [r for r in range(1, height - 1) if r % 2 == 1]
    even_cols = [c for c in range(2, width - 2) if c % 2 == 0]
    if len(odd_rows) < 2 * n_paths or len(even_cols) < n_paths:
        return None

    start_rows = sorted(random.sample(odd_rows, n_paths))
    remaining_rows = [r for r in odd_rows if r not in start_rows]
    end_rows = random.sample(remaining_rows, n_paths)
    sides = _sample_marker_sides(start_rows)
    if sides is None:
        return None
    turn_cols = random.sample(even_cols, n_paths)
    colors = random.sample(COLOR_POOL, n_paths)

    specs = []
    for idx in range(n_paths):
        specs.append(
            {
                "start_row": int(start_rows[idx]),
                "end_row": int(end_rows[idx]),
                "turn_col": int(turn_cols[idx]),
                "marker_side": int(sides[idx]),
                "orig_color": int(colors[idx]),
            }
        )
    return specs


def _topology_ok(paths: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    owners: dict[Coord, int] = {}
    for idx, path in enumerate(paths):
        marker = path["marker"]
        if marker in owners:
            return False
        owners[marker] = idx

        for cell in path["centerline"]:
            prev = owners.get(cell)
            if prev is None:
                owners[cell] = idx
                continue
            if prev == idx:
                return False

    mask = np.zeros((h, w), dtype=bool)
    for path in paths:
        mask[path["marker"]] = True
        for cell in path["centerline"]:
            mask[cell] = True
    adj = _build_adjacency(mask)
    if not adj:
        return False
    degs = {cell: len(nbs) for cell, nbs in adj.items()}
    if any(deg not in (1, 2, 4) for deg in degs.values()):
        return False
    starts = [cell for cell in adj if _is_start_leaf(cell, adj, np.where(mask, 1, 0))]
    return len(starts) == len(paths)


def _canonical_paths(height: int, width: int, specs: list[dict]) -> list[dict]:
    paths: list[dict] = []
    for spec in specs:
        start_row = spec["start_row"]
        end_row = spec["end_row"]
        turn_col = spec["turn_col"]
        marker = (start_row + spec["marker_side"], 0)
        centerline = _centerline_cells(start_row, end_row, turn_col, width)
        paths.append(
            {
                "start": centerline[0],
                "marker": marker,
                "centerline": centerline,
                "start_row": start_row,
                "end_row": end_row,
                "turn_col": turn_col,
                "orig_color": spec["orig_color"],
            }
        )
    return paths


def _intersection_stats(paths: list[dict]) -> tuple[int, list[int]]:
    counts = [0] * len(paths)
    total = 0
    cell_to_paths: dict[Coord, list[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path["centerline"]:
            cell_to_paths.setdefault(cell, []).append(idx)
    for ids in cell_to_paths.values():
        if len(ids) > 1:
            total += 1
            for idx in ids:
                counts[idx] += 1
    return total, counts


def _paint_range_row(store: dict[Coord, int], row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        return
    for c in range(c0, c1 + 1):
        store[(row, c)] = color


def _paint_range_col(store: dict[Coord, int], col: int, r0: int, r1: int, color: int) -> None:
    if r0 <= r1:
        rows = range(r0, r1 + 1)
    else:
        rows = range(r0, r1 - 1, -1)
    for r in rows:
        store[(r, col)] = color


def _simulate_contamination(paths: list[dict], width: int) -> list[dict]:
    n = len(paths)
    cur_colors = [int(path["orig_color"]) for path in paths]
    cur_rows = [int(path["start_row"]) for path in paths]
    last_x = [0] * n
    paint_maps = [dict() for _ in range(n)]
    event_order = sorted(range(n), key=lambda idx: int(paths[idx]["turn_col"]))

    for p_idx in event_order:
        path = paths[p_idx]
        x = int(path["turn_col"])
        row = cur_rows[p_idx]
        _paint_range_row(paint_maps[p_idx], row, last_x[p_idx], x, cur_colors[p_idx])

        step = 1 if int(path["end_row"]) > row else -1
        p_color = int(cur_colors[p_idx])
        segment_start = row
        row_to_path = {cur_rows[idx]: idx for idx in range(n) if idx != p_idx}

        for r in range(row + step, int(path["end_row"]) + step, step):
            q_idx = row_to_path.get(r)
            if q_idx is None:
                continue
            _paint_range_col(paint_maps[p_idx], x, segment_start + step, r, p_color)
            _paint_range_row(paint_maps[q_idx], r, last_x[q_idx], x, cur_colors[q_idx])
            last_x[q_idx] = x + 1
            q_color = int(cur_colors[q_idx])
            cur_colors[q_idx] = p_color
            p_color = q_color
            segment_start = r

        _paint_range_col(
            paint_maps[p_idx],
            x,
            segment_start + step,
            int(path["end_row"]),
            p_color,
        )
        cur_colors[p_idx] = p_color
        cur_rows[p_idx] = int(path["end_row"])
        last_x[p_idx] = x + 1

    for idx in range(n):
        _paint_range_row(
            paint_maps[idx],
            cur_rows[idx],
            last_x[idx],
            width - 1,
            int(cur_colors[idx]),
        )

    out: list[dict] = []
    for idx, path in enumerate(paths):
        painted = dict(paint_maps[idx])
        assert set(painted) == set(path["centerline"]), "every centerline cell must get a contaminated color"
        out.append({**path, "paint": painted})
    return out


def _transform_shape(shape: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = shape
    if name in {"identity", "flip_h", "flip_v", "rot180"}:
        return h, w
    return w, h


def _transform_coord(cell: Coord, shape: tuple[int, int], name: str) -> Coord:
    h, w = shape
    r, c = cell
    if name == "identity":
        return r, c
    if name == "flip_h":
        return r, w - 1 - c
    if name == "flip_v":
        return h - 1 - r, c
    if name == "rot180":
        return h - 1 - r, w - 1 - c
    if name == "transpose":
        return c, r
    if name == "rot90":
        return c, h - 1 - r
    if name == "rot270":
        return w - 1 - c, r
    assert name == "anti_transpose"
    return w - 1 - c, h - 1 - r


def _transform_paths(paths: list[dict], shape: tuple[int, int], name: str) -> tuple[list[dict], tuple[int, int]]:
    new_shape = _transform_shape(shape, name)
    transformed: list[dict] = []
    for path in paths:
        centerline = [_transform_coord(cell, shape, name) for cell in path["centerline"]]
        marker = _transform_coord(path["marker"], shape, name)
        paint = {
            _transform_coord(cell, shape, name): int(color)
            for cell, color in path["paint"].items()
        }
        transformed.append(
            {
                "start": centerline[0],
                "marker": marker,
                "centerline": centerline,
                "orig_color": int(path["orig_color"]),
                "paint": paint,
            }
        )
    return transformed, new_shape


def _render_output(paths: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for path in sorted(paths, key=lambda item: item["start"]):
        color = int(path["orig_color"])
        out[path["marker"]] = color
        for cell in path["centerline"]:
            out[cell] = color
    return out


def _render_input(paths: list[dict], shape: tuple[int, int]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    order = list(range(len(paths)))
    random.shuffle(order)
    for idx in order:
        path = paths[idx]
        color = int(path["orig_color"])
        inp[path["marker"]] = color
        for cell in path["centerline"]:
            inp[cell] = int(path["paint"][cell])
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_paths = random.randint(*N_PATHS_RANGE)
    min_h = max(HEIGHT_RANGE[0], 4 * n_paths + 1)
    min_w = max(WIDTH_RANGE[0], 2 * n_paths + 6)
    if min_h > HEIGHT_RANGE[1] or min_w > WIDTH_RANGE[1]:
        return None

    height = random.randint(min_h, HEIGHT_RANGE[1])
    width = random.randint(min_w, WIDTH_RANGE[1])
    specs = _path_specs(height, width, n_paths)
    if specs is None:
        return None

    canonical = _canonical_paths(height, width, specs)
    total_crossings, per_path = _intersection_stats(canonical)
    min_crossings = 1 if n_paths == 2 else 2
    if total_crossings < min_crossings:
        return None
    if n_paths >= 3 and max(per_path) < 2:
        return None
    if not _topology_ok(canonical, (height, width)):
        return None

    contaminated = _simulate_contamination(canonical, width)
    transform = random.choice(TRANSFORMS)
    transformed, shape = _transform_paths(contaminated, (height, width), transform)
    expected = _render_output(transformed, shape)
    inp = _render_input(transformed, shape)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossed-path ARC input")

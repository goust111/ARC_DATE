from __future__ import annotations

import random

import numpy as np


N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (-dc, dr)


def _right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (dc, -dr)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _line(points[i], points[i + 1])
        out.extend(seg if i == 0 else seg[1:])
    return out


def _departure_square(
    endpoint: tuple[int, int],
    inward: tuple[int, int],
    side: str,
) -> list[tuple[int, int]]:
    outward = (-inward[0], -inward[1])
    lateral = _left(inward) if side == "left" else _right(inward)
    r, c = endpoint
    dr0, dc0 = outward
    dr1, dc1 = lateral
    return [
        (r, c),
        (r + dr0, c + dc0),
        (r + dr1, c + dc1),
        (r + dr0 + dr1, c + dc0 + dc1),
    ]


def _arrival_square(
    endpoint: tuple[int, int],
    arrival: tuple[int, int],
    side: str,
) -> list[tuple[int, int]]:
    lateral = _left(arrival) if side == "left" else _right(arrival)
    r, c = endpoint
    dr0, dc0 = arrival
    dr1, dc1 = lateral
    return [
        (r, c),
        (r + dr0, c + dc0),
        (r + dr1, c + dc1),
        (r + dr0 + dr1, c + dc0 + dc1),
    ]


def _ball_components(grid: np.ndarray) -> list[dict[str, object]]:
    balls: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != 0):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        cells = sorted((int(r), int(c)) for r, c in coords)
        balls.append({"color": color, "cells": cells})
    return balls


def _reconstruct_pipes(
    base: np.ndarray, pipe_colors: list[int]
) -> dict[int, set[tuple[int, int]]]:
    h, w = base.shape
    pipes: dict[int, set[tuple[int, int]]] = {}
    for color in pipe_colors:
        cells = {(int(r), int(c)) for r, c in np.argwhere(base == color)}
        for r in range(h):
            for c in range(w):
                if base[r, c] == 0 or base[r, c] == color:
                    continue
                horizontal = (
                    c - 1 >= 0
                    and c + 1 < w
                    and int(base[r, c - 1]) == color
                    and int(base[r, c + 1]) == color
                )
                vertical = (
                    r - 1 >= 0
                    and r + 1 < h
                    and int(base[r - 1, c]) == color
                    and int(base[r + 1, c]) == color
                )
                if horizontal or vertical:
                    cells.add((r, c))
        pipes[color] = cells
    return pipes


def _ball_attachment(
    ball_cells: list[tuple[int, int]],
    pipe_by_cell: dict[tuple[int, int], list[int]],
) -> tuple[tuple[int, int], tuple[int, int], int, str]:
    ball = set(ball_cells)
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for anchor in ball_cells:
        neighbors: list[tuple[tuple[int, int], int]] = []
        for dr, dc in N4:
            nxt = (anchor[0] + dr, anchor[1] + dc)
            if nxt in ball:
                continue
            colors = pipe_by_cell.get(nxt, [])
            for color in colors:
                neighbors.append((nxt, color))
        if len(neighbors) == 1:
            nxt, color = neighbors[0]
            candidates.append((anchor, nxt, color))
    assert len(candidates) == 1
    anchor, nxt, color = candidates[0]
    inward = (nxt[0] - anchor[0], nxt[1] - anchor[1])
    outward = (-inward[0], -inward[1])
    edge_steps = [
        (cell[0] - anchor[0], cell[1] - anchor[1])
        for cell in ball
        if abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]) == 1
    ]
    assert len(edge_steps) == 2 and outward in edge_steps
    lateral = edge_steps[0] if edge_steps[1] == outward else edge_steps[1]
    if lateral == _left(inward):
        side = "left"
    else:
        assert lateral == _right(inward)
        side = "right"
    return anchor, nxt, color, side


def _trace_path(
    pipe_cells: set[tuple[int, int]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    limit = len(pipe_cells) + 2
    for _ in range(limit):
        nxts = []
        for dr, dc in N4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in pipe_cells and nxt != prev:
                nxts.append(nxt)
        if not nxts:
            return path
        assert len(nxts) == 1
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    raise AssertionError("path trace exceeded expected length")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each 2x2 signal block to the far end of its pipe while crossing points are traversed straight through.

    Input structure:
    - The background is black.
    - Several one-pixel-wide colored pipes run horizontally and vertically, may turn at right angles, and may cross other pipes.
    - A crossing cell shows only one pipe color even though another straight pipe may also pass through that cell.
    - Some pipe endpoints are covered by a 2x2 solid signal block whose color is distinct from every pipe color and whose corner overlaps the endpoint.

    Rule:
    - Use the overlapped corner of each signal block to identify its pipe endpoint, then follow that pipe to the other endpoint.
    - When another pipe visually covers a crossing cell, recover the hidden straight-through continuation from the opposite same-color neighbors.
    - At crossings the signal keeps going straight on its own pipe; it never turns onto the other pipe.
    - Remove all pipe pixels and restamp each signal block on the far endpoint so the endpoint is still a corner of the 2x2 block and the block remains on the same side of travel while expanding outward from the pipe.

    Key insight:
    - The visible pipe colors are insufficient by themselves because crossings hide one pipe; you must reconstruct the hidden straight-through cells, pair each ball with the endpoint it covers, and carry the ball to the opposite endpoint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    balls = _ball_components(g)
    ball_colors = {int(ball["color"]) for ball in balls}

    base = g.copy()
    for ball in balls:
        for r, c in ball["cells"]:
            base[r, c] = 0

    pipe_colors = sorted(int(c) for c in np.unique(base) if int(c) != 0 and int(c) not in ball_colors)
    pipes = _reconstruct_pipes(base, pipe_colors)

    pipe_by_cell: dict[tuple[int, int], list[int]] = {}
    for color, cells in pipes.items():
        for cell in cells:
            pipe_by_cell.setdefault(cell, []).append(color)

    records: list[dict[str, object]] = []
    for ball in balls:
        anchor, nxt, pipe_color, side = _ball_attachment(ball["cells"], pipe_by_cell)
        pipes[pipe_color].add(anchor)
        records.append(
            {
                "ball_color": int(ball["color"]),
                "pipe_color": pipe_color,
                "anchor": anchor,
                "next": nxt,
                "side": side,
            }
        )

    out = np.zeros((h, w), dtype=int)
    for record in records:
        path = _trace_path(pipes[int(record["pipe_color"])], record["anchor"])
        assert len(path) >= 2
        dest = path[-1]
        prev = path[-2]
        arrival = (dest[0] - prev[0], dest[1] - prev[1])
        cells = _arrival_square(dest, arrival, str(record["side"]))
        for cell in cells:
            assert _in_bounds((h, w), cell)
            out[cell] = int(record["ball_color"])

    return out


def _cell_orientation(path: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for i, cell in enumerate(path):
        dirs: list[tuple[int, int]] = []
        if i > 0:
            prev = path[i - 1]
            dirs.append((prev[0] - cell[0], prev[1] - cell[1]))
        if i + 1 < len(path):
            nxt = path[i + 1]
            dirs.append((nxt[0] - cell[0], nxt[1] - cell[1]))
        if len(dirs) == 1:
            out[cell] = "endpoint"
            continue
        assert len(dirs) == 2
        if dirs[0] == (-dirs[1][0], -dirs[1][1]):
            out[cell] = "h" if dirs[0][0] == 0 else "v"
        else:
            out[cell] = "turn"
    return out


def _pipe_path(
    kind: str,
    dims: tuple[int, int],
    r0: int,
    r1: int | None,
    c0: int,
    c1: int | None,
    pivot: int | None,
) -> list[tuple[int, int]]:
    h, w = dims
    if kind == "lr":
        start = (r0, 1)
        end = (r1 if r1 is not None else r0, w - 2)
        if pivot is None:
            return _polyline([start, end])
        return _polyline([start, (r0, pivot), (end[0], pivot), end])
    start = (1, c0)
    end = (h - 2, c1 if c1 is not None else c0)
    if pivot is None:
        return _polyline([start, end])
    return _polyline([start, (pivot, c0), (pivot, end[1]), end])


def _feasible_ball_options(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    occupied_by_cell: dict[tuple[int, int], list[int]],
    shape: tuple[int, int],
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    ends = [
        (path[0], path[1], path[-1], (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])),
        (path[-1], path[-2], path[0], (path[0][0] - path[1][0], path[0][1] - path[1][1])),
    ]
    for endpoint, inward_nbr, dest, arrival in ends:
        inward = (inward_nbr[0] - endpoint[0], inward_nbr[1] - endpoint[1])
        for side in ("left", "right"):
            src = _departure_square(endpoint, inward, side)
            dst = _arrival_square(dest, arrival, side)
            src_set = set(src)
            dst_set = set(dst)
            if len(src_set) != 4 or len(dst_set) != 4:
                continue
            if not all(_in_bounds(shape, cell) for cell in src + dst):
                continue
            if any(cell != endpoint and cell in occupied for cell in src):
                continue
            if any(cell != dest and cell in occupied for cell in dst):
                continue
            valid_source = True
            for cell in src:
                hits: list[int] = []
                for dr, dc in N4:
                    nbr = (cell[0] + dr, cell[1] + dc)
                    if nbr in src_set:
                        continue
                    hits.extend(occupied_by_cell.get(nbr, []))
                if cell == endpoint:
                    if len(hits) != 1:
                        valid_source = False
                        break
                elif hits:
                    valid_source = False
                    break
            if not valid_source:
                continue
            options.append(
                {
                    "source_endpoint": endpoint,
                    "source_next": inward_nbr,
                    "dest_endpoint": dest,
                    "arrival": arrival,
                    "side": side,
                    "src_cells": src,
                    "dst_cells": dst,
                }
            )
    return options


def _choose_ball_assignments(
    paths: list[list[tuple[int, int]]],
    occupied: set[tuple[int, int]],
    occupied_by_cell: dict[tuple[int, int], list[int]],
    shape: tuple[int, int],
    count: int,
) -> dict[int, dict[str, object]] | None:
    option_map = {
        i: _feasible_ball_options(path, occupied, occupied_by_cell, shape)
        for i, path in enumerate(paths)
    }
    eligible = [i for i, opts in option_map.items() if opts]
    if len(eligible) < count:
        return None
    chosen = random.sample(eligible, count)
    random.shuffle(chosen)

    def dfs(
        k: int,
        used_src: set[tuple[int, int]],
        used_dst: set[tuple[int, int]],
        picked: dict[int, dict[str, object]],
    ) -> dict[int, dict[str, object]] | None:
        if k == len(chosen):
            return picked.copy()
        pipe_id = chosen[k]
        options = option_map[pipe_id][:]
        random.shuffle(options)
        for option in options:
            src_cells = set(option["src_cells"])
            dst_cells = set(option["dst_cells"])
            if src_cells & used_src:
                continue
            if dst_cells & used_dst:
                continue
            picked[pipe_id] = option
            answer = dfs(k + 1, used_src | src_cells, used_dst | dst_cells, picked)
            if answer is not None:
                return answer
            picked.pop(pipe_id, None)
        return None

    return dfs(0, set(), set(), {})


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    layout_choices = [
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 0),
        (0, 3),
        (2, 0),
        (0, 2),
    ]
    n_lr, n_tb = random.choice(layout_choices)
    n_pipes = n_lr + n_tb
    if n_pipes == 0:
        return None

    bent_lr = [random.random() < 0.7 for _ in range(n_lr)]
    bent_tb = [random.random() < 0.7 for _ in range(n_tb)]

    rows_needed = sum(1 + int(flag) for flag in bent_lr) + sum(int(flag) for flag in bent_tb)
    cols_needed = sum(int(flag) for flag in bent_lr) + sum(1 + int(flag) for flag in bent_tb)

    min_h = max(12, 3 * rows_needed + 6)
    min_w = max(12, 3 * cols_needed + 6)
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    row_vals = list(range(2, h - 2, 3))
    col_vals = list(range(2, w - 2, 3))
    if len(row_vals) < rows_needed or len(col_vals) < cols_needed:
        return None
    random.shuffle(row_vals)
    random.shuffle(col_vals)

    row_it = iter(row_vals)
    col_it = iter(col_vals)
    paths: list[list[tuple[int, int]]] = []
    for flag in bent_lr:
        rs = next(row_it)
        re = next(row_it) if flag else None
        pivot = next(col_it) if flag else None
        paths.append(_pipe_path("lr", (h, w), rs, re, 0, None, pivot))
    for flag in bent_tb:
        pivot = next(row_it) if flag else None
        cs = next(col_it)
        ce = next(col_it) if flag else None
        paths.append(_pipe_path("tb", (h, w), 0, None, cs, ce, pivot))

    orientation_maps = [_cell_orientation(path) for path in paths]
    occupied_by_cell: dict[tuple[int, int], list[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path:
            occupied_by_cell.setdefault(cell, []).append(idx)

    crossing_count = 0
    for cell, owners in occupied_by_cell.items():
        if len(owners) == 1:
            continue
        if len(owners) != 2:
            return None
        a, b = owners
        kind = {orientation_maps[a][cell], orientation_maps[b][cell]}
        if kind != {"h", "v"}:
            return None
        crossing_count += 1

    if random.random() < 0.7 and crossing_count == 0:
        return None

    occupied = set(occupied_by_cell)
    ball_count = 1 if n_pipes == 1 else random.randint(1, n_pipes - 1)
    assignments = _choose_ball_assignments(paths, occupied, occupied_by_cell, (h, w), ball_count)
    if assignments is None:
        return None

    color_pool = list(range(1, 10))
    random.shuffle(color_pool)
    if len(color_pool) < n_pipes + ball_count:
        return None
    pipe_colors = color_pool[:n_pipes]
    ball_colors = color_pool[n_pipes : n_pipes + ball_count]

    grid = np.zeros((h, w), dtype=int)
    draw_order = list(range(n_pipes))
    random.shuffle(draw_order)
    for idx in draw_order:
        for r, c in paths[idx]:
            grid[r, c] = pipe_colors[idx]

    expected = np.zeros((h, w), dtype=int)
    for ball_color, (pipe_id, option) in zip(ball_colors, assignments.items(), strict=False):
        for r, c in option["src_cells"]:
            grid[r, c] = ball_color
        for r, c in option["dst_cells"]:
            expected[r, c] = ball_color

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            built = _build_attempt()
        except AssertionError:
            continue
        if built is not None:
            grid = built[0]
            try:
                out = main(grid)
            except Exception:
                continue
            if out.ndim == 2 and max(out.shape) <= 30:
                return grid
    raise RuntimeError("failed to generate a valid pipe-signal ARC input")

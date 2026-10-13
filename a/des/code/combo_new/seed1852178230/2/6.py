from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}
EDGE_SIDES = ("top", "right", "bottom", "left")
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "background must be unique"
    return int(ranked[0][0])


def _extract_strip(grid: np.ndarray) -> tuple[str, list[int]]:
    bg = _dominant_color(grid)
    candidates: list[tuple[str, list[int]]] = []

    border_lines = {
        "top": np.asarray(grid[0, :], dtype=int),
        "right": np.asarray(grid[:, -1], dtype=int),
        "bottom": np.asarray(grid[-1, :], dtype=int),
        "left": np.asarray(grid[:, 0], dtype=int),
    }

    for side, line in border_lines.items():
        idx = np.flatnonzero(line != bg)
        if len(idx) < 3:
            continue
        if not np.all(np.diff(idx) == 1):
            continue
        colors = [int(line[pos]) for pos in idx]
        if len(set(colors)) != len(colors):
            continue
        if np.any(line[: idx[0]] != bg) or np.any(line[idx[-1] + 1 :] != bg):
            continue
        candidates.append((side, colors))

    assert len(candidates) == 1, f"expected one edge strip, got {candidates}"
    return candidates[0]


def _remove_border(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return np.asarray(grid[1:, :], dtype=int)
    if side == "bottom":
        return np.asarray(grid[:-1, :], dtype=int)
    if side == "left":
        return np.asarray(grid[:, 1:], dtype=int)
    if side == "right":
        return np.asarray(grid[:, :-1], dtype=int)
    raise AssertionError(f"unknown side {side}")


def _neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.append((int(rr), int(cc)))
    return out


def _degree_map(mask: np.ndarray) -> np.ndarray:
    degrees = np.zeros(mask.shape, dtype=int)
    for r, c in np.argwhere(mask):
        degrees[int(r), int(c)] = len(_neighbors(mask, (int(r), int(c))))
    return degrees


def _choose_next(
    prev_cell: tuple[int, int],
    cur_cell: tuple[int, int],
    options: list[tuple[int, int]],
) -> tuple[int, int]:
    dr = cur_cell[0] - prev_cell[0]
    dc = cur_cell[1] - prev_cell[1]
    incoming = DIR_TO_INDEX[(dr, dc)]
    preference = [incoming, (incoming + 1) % 4, (incoming - 1) % 4, (incoming + 2) % 4]
    by_dir = {(nxt[0] - cur_cell[0], nxt[1] - cur_cell[1]): nxt for nxt in options}

    for idx in preference:
        direction = DIRS[idx]
        if direction in by_dir:
            return by_dir[direction]
    raise AssertionError("no valid continuation at node")


def _trace_path(
    start: tuple[int, int],
    mask: np.ndarray,
    degrees: np.ndarray,
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    seen_states: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while True:
        if prev is not None and int(degrees[cur]) == 1:
            return path

        nbrs = _neighbors(mask, cur)
        if prev is None:
            assert len(nbrs) == 1, "endpoint must have exactly one neighbor"
            nxt = nbrs[0]
        else:
            options = [nbr for nbr in nbrs if nbr != prev]
            assert options, "path must continue until the next endpoint"
            nxt = options[0] if len(options) == 1 else _choose_next(prev, cur, options)

        state = (prev, cur)
        assert state not in seen_states, "path traversal entered a loop"
        seen_states.add(state)
        prev, cur = cur, nxt
        path.append(cur)


def _segment_lengths(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts, "path must be at least as long as its color subsequence"
    base = total // parts
    extra = total - base * parts
    if parts == 1:
        return [total]
    return [base] * (parts - 1) + [base + extra]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor endpoint-to-endpoint paths by reading an edge legend and following the junction turn rule.

    Input structure:
    - The background is the unique dominant color.
    - One border line contains a contiguous strip of distinct colors that defines the global legend order.
    - In the remaining grid, one-pixel-wide horizontal/vertical paths are drawn with colored endpoints, a shared mid color, and special node pixels at bends or junctions.
    - Path nodes may be ordinary turns or higher-degree T / cross junctions.

    Rule:
    - Remove the border line that carries the legend strip from the output and read the strip in edge order: left-to-right on a top/bottom strip, top-to-bottom on a left/right strip.
    - In the cropped path area, treat every non-background pixel as walkable and identify endpoints as walkable pixels with exactly one 4-neighbor.
    - Launch a traversal from every endpoint. The first step uses its only neighbor; afterwards the path continues straight when possible, otherwise turns clockwise, otherwise counterclockwise, never immediately backtracking.
    - Whenever a traversal reaches another endpoint, compare the two endpoint colors in the legend. Only the direction from the earlier legend color to the later legend color is used. Take the inclusive legend subsequence between those colors, split the whole traced path into that many consecutive segments, and assign the legend colors in order. If the split is uneven, the last segment receives the extra pixels.
    - Node pixels are recolored just like any other traced path pixel.

    Key insight:
    - The edge strip is a global ordering key, while the node color itself is not semantic in the output: what matters is the path order induced by endpoint colors plus the local continuation rule at each junction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    side, legend = _extract_strip(g)
    core = _remove_border(g, side)
    bg = _dominant_color(core)
    order = {int(color): idx for idx, color in enumerate(legend)}

    mask = core != bg
    degrees = _degree_map(mask)
    endpoints = [tuple(map(int, cell)) for cell in np.argwhere(mask & (degrees == 1))]
    assert endpoints, "expected at least one path endpoint"

    out = np.full(core.shape, bg, dtype=int)
    painted = np.zeros(core.shape, dtype=bool)

    for start in endpoints:
        path = _trace_path(start, mask, degrees)
        end = path[-1]
        assert end != start and int(degrees[end]) == 1, "each traversal must terminate at another endpoint"

        start_color = int(core[start])
        end_color = int(core[end])
        assert start_color in order and end_color in order, "endpoint colors must come from the legend"

        start_rank = order[start_color]
        end_rank = order[end_color]
        if start_rank > end_rank:
            continue
        if start_rank == end_rank and start > end:
            continue

        colors = legend[start_rank : end_rank + 1]
        lengths = _segment_lengths(len(path), len(colors))
        idx = 0
        for color, seg_len in zip(colors, lengths):
            for _ in range(seg_len):
                r, c = path[idx]
                if painted[r, c]:
                    assert int(out[r, c]) == int(color), "shared path pixels must receive the same color"
                else:
                    out[r, c] = int(color)
                    painted[r, c] = True
                idx += 1
        assert idx == len(path)

    assert np.all(painted[mask]), "all path pixels must be covered by forward legend traversals"
    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(int(r), int(c0)) for r in range(r0, r1 + step, step)]


def _waypoint_path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        piece = _segment(points[i], points[i + 1])
        if cells:
            piece = piece[1:]
        cells.extend(piece)
    return cells


def _rotate_cell(cell: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    h, _w = shape
    return (int(c), int(h - 1 - r))


def _rotate_structure(structure: dict, turns: int) -> dict:
    turns %= 4
    paths = [
        {"cells": list(path["cells"]), "start_idx": int(path["start_idx"]), "end_idx": int(path["end_idx"])}
        for path in structure["paths"]
    ]
    specials = set(map(tuple, structure["specials"]))
    shape = tuple(structure["shape"])

    for _ in range(turns):
        paths = [
            {
                "cells": [_rotate_cell(cell, shape) for cell in path["cells"]],
                "start_idx": int(path["start_idx"]),
                "end_idx": int(path["end_idx"]),
            }
            for path in paths
        ]
        specials = {_rotate_cell(cell, shape) for cell in specials}
        shape = (shape[1], shape[0])

    return {"shape": shape, "paths": paths, "specials": specials}


def _reverse_path_info(path: dict) -> dict:
    return {
        "cells": list(reversed(path["cells"])),
        "start_idx": int(path["end_idx"]),
        "end_idx": int(path["start_idx"]),
    }


def _choose_pair(legend_len: int) -> tuple[int, int]:
    start = random.randint(0, legend_len - 2)
    end = random.randint(start + 1, legend_len - 1)
    return start, end


def _build_simple(legend_len: int) -> dict:
    variant = random.choice(("l", "zigzag", "straight_h", "straight_v"))
    start_idx, end_idx = _choose_pair(legend_len)

    if variant == "l":
        a = random.randint(2, 5)
        b = random.randint(2, 6)
        waypoints = [(0, 0), (0, b), (a, b)]
        specials = {(0, b)}
        shape = (a + 1, b + 1)
    elif variant == "zigzag":
        a = random.randint(2, 4)
        b = random.randint(2, 4)
        c = random.randint(2, 4)
        waypoints = [(0, 0), (0, b), (a, b), (a, b + c)]
        specials = {(0, b), (a, b)}
        shape = (a + 1, b + c + 1)
    elif variant == "straight_h":
        length = random.randint(5, 10)
        waypoints = [(0, 0), (0, length - 1)]
        specials = set()
        shape = (1, length)
    else:
        length = random.randint(5, 10)
        waypoints = [(0, 0), (length - 1, 0)]
        specials = set()
        shape = (length, 1)

    path = {"cells": _waypoint_path(waypoints), "start_idx": start_idx, "end_idx": end_idx}
    if random.random() < 0.5:
        path = _reverse_path_info(path)
    return {"shape": shape, "paths": [path], "specials": specials}


def _build_cross(legend_len: int) -> dict:
    arm = random.randint(2, 4)
    span = 2 * arm + 1
    center = (arm, arm)
    start_idx, end_idx = _choose_pair(legend_len)

    horizontal = {"cells": _segment((arm, 0), (arm, span - 1)), "start_idx": start_idx, "end_idx": end_idx}
    vertical = {"cells": _segment((0, arm), (span - 1, arm)), "start_idx": start_idx, "end_idx": end_idx}
    if random.random() < 0.5:
        horizontal = _reverse_path_info(horizontal)
    if random.random() < 0.5:
        vertical = _reverse_path_info(vertical)

    return {"shape": (span, span), "paths": [horizontal, vertical], "specials": {center}}


def _build_tmerge(legend_len: int) -> dict:
    a = random.randint(2, 4)
    b = random.randint(2, 5)
    node = (a, a)
    start_idx, end_idx = _choose_pair(legend_len)

    horizontal = {
        "cells": _segment((a, 0), (a, a + b)),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    merged = {
        "cells": _waypoint_path([(2 * a, a), node, (a, a + b)]),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }

    return {"shape": (2 * a + 1, a + b + 1), "paths": [horizontal, merged], "specials": {node}}


def _offset_structure(structure: dict, top: int, left: int) -> dict:
    paths = []
    for path in structure["paths"]:
        paths.append(
            {
                "cells": [(r + top, c + left) for r, c in path["cells"]],
                "start_idx": int(path["start_idx"]),
                "end_idx": int(path["end_idx"]),
            }
        )
    specials = {(r + top, c + left) for r, c in structure["specials"]}
    return {"shape": structure["shape"], "paths": paths, "specials": specials}


def _occupied_cells(structure: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for path in structure["paths"]:
        cells.update(map(tuple, path["cells"]))
    return cells


def _can_place(
    structure: dict,
    top: int,
    left: int,
    core_shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = structure["shape"]
    core_h, core_w = core_shape
    if top < 1 or left < 1 or top + h > core_h - 1 or left + w > core_w - 1:
        return False

    shifted = _offset_structure(structure, top, left)
    for r, c in _occupied_cells(shifted):
        if (r, c) in blocked:
            return False
    return True


def _update_blocked(blocked: set[tuple[int, int]], cells: set[tuple[int, int]], core_shape: tuple[int, int]) -> None:
    core_h, core_w = core_shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < core_h and 0 <= cc < core_w:
                    blocked.add((int(rr), int(cc)))


def _render_core(
    core_shape: tuple[int, int],
    structures: list[dict],
    legend: list[int],
    bg: int,
    mid: int,
    node: int,
) -> np.ndarray:
    core = np.full(core_shape, bg, dtype=int)

    for structure in structures:
        specials = set(map(tuple, structure["specials"]))
        for path in structure["paths"]:
            cells = list(path["cells"])
            start_color = int(legend[path["start_idx"]])
            end_color = int(legend[path["end_idx"]])
            for idx, (r, c) in enumerate(cells):
                if idx == 0:
                    color = start_color
                elif idx == len(cells) - 1:
                    color = end_color
                elif (r, c) in specials:
                    color = node
                else:
                    color = mid

                if int(core[r, c]) == bg:
                    core[r, c] = int(color)
                else:
                    assert int(core[r, c]) == int(color), "shared structure pixels must agree"

    return core


def _attach_strip(core: np.ndarray, legend: list[int], side: str, bg: int) -> np.ndarray:
    h, w = core.shape
    if side in {"left", "right"}:
        out = np.full((h, w + 1), bg, dtype=int)
        if side == "left":
            out[:, 1:] = core
            line = out[:, 0]
        else:
            out[:, :-1] = core
            line = out[:, -1]
        start = random.randint(0, h - len(legend))
        line[start : start + len(legend)] = np.asarray(legend, dtype=int)
        return out

    out = np.full((h + 1, w), bg, dtype=int)
    if side == "top":
        out[1:, :] = core
        line = out[0, :]
    else:
        out[:-1, :] = core
        line = out[-1, :]
    start = random.randint(0, w - len(legend))
    line[start : start + len(legend)] = np.asarray(legend, dtype=int)
    return out


def _sample_palette() -> tuple[int, int, int, list[int]]:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    mid = random.choice(colors)
    colors.remove(mid)
    node = random.choice(colors)
    colors.remove(node)
    legend_len = random.randint(3, 5)
    legend = random.sample(colors, legend_len)
    return int(bg), int(mid), int(node), list(map(int, legend))


def _sample_structure(legend_len: int) -> dict:
    kind = random.choice(("simple", "cross", "tmerge"))
    if kind == "simple":
        structure = _build_simple(legend_len)
    elif kind == "cross":
        structure = _build_cross(legend_len)
    else:
        structure = _build_tmerge(legend_len)
    return _rotate_structure(structure, random.randrange(4))


def _generate_attempt() -> np.ndarray | None:
    bg, mid, node, legend = _sample_palette()
    strip_side = random.choice(EDGE_SIDES)

    core_h = random.randint(11, 23)
    core_w = random.randint(11, 23)
    if strip_side in {"left", "right"} and core_w + 1 > 30:
        return None
    if strip_side in {"top", "bottom"} and core_h + 1 > 30:
        return None

    structure_count = random.randint(1, 3)
    structures: list[dict] = []
    blocked: set[tuple[int, int]] = set()

    for _ in range(structure_count):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            structure = _sample_structure(len(legend))
            h, w = structure["shape"]
            if h > core_h - 2 or w > core_w - 2:
                continue
            top = random.randint(1, core_h - h - 1)
            left = random.randint(1, core_w - w - 1)
            if not _can_place(structure, top, left, (core_h, core_w), blocked):
                continue
            shifted = _offset_structure(structure, top, left)
            structures.append(shifted)
            _update_blocked(blocked, _occupied_cells(shifted), (core_h, core_w))
            placed = True
            break
        if not placed:
            return None

    core = _render_core((core_h, core_w), structures, legend, bg, mid, node)
    if int(np.sum(core != bg)) >= core.size // 2:
        return None

    inp = _attach_strip(core, legend, strip_side, bg)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != core.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-gradient puzzle")

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
PATH_COLORS = (1, 2, 3, 5, 6, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
DIR_PAIRS = tuple(
    (DIRS[i], DIRS[j])
    for i in range(len(DIRS))
    for j in range(i + 1, len(DIRS))
)

GRID_MIN_SIDE_BY_NODE_COUNT = {1: 12, 2: 16, 3: 20}
GRID_MAX_SIDE = 26
MAX_NODES = 3
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
PACKAGE_GAP = 2
PATH_LENGTH_RANGE = (1, 4)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _rotate_offset(dr: int, dc: int, steps: int) -> tuple[int, int]:
    steps = int(steps) % 4
    if steps == 0:
        return int(dr), int(dc)
    if steps == 1:
        return int(dc), -int(dr)
    if steps == 2:
        return -int(dr), -int(dc)
    return -int(dc), int(dr)


def _rotate_dir(direction: tuple[int, int], steps: int) -> tuple[int, int]:
    return _rotate_offset(direction[0], direction[1], steps)


def _component_coords(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = DIRS
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _same_color_component(g: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    color = int(g[start])
    q = deque([start])
    seen = {start}
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((int(r), int(c)))
        for dr, dc in DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not _in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(g[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return sorted(comp)


def _decode_marker(comp: list[tuple[int, int]]) -> int:
    assert len(comp) == 2, "each yellow marker must have exactly two pixels"
    (r0, c0), (r1, c1) = sorted(comp)
    dr = int(r1 - r0)
    dc = int(c1 - c0)
    if dr == 0 and dc == 1:
        return 0
    if dr == 1 and dc == 0:
        return 1
    if dr == 1 and dc == 1:
        return -1
    raise AssertionError("yellow marker must be horizontal, vertical, or NW-SE diagonal")


def _touching_nodes(comp: list[tuple[int, int]], orange: set[tuple[int, int]]) -> list[tuple[int, int]]:
    hits = set()
    for r, c in comp:
        for nr, nc in orange:
            if max(abs(int(r) - int(nr)), abs(int(c) - int(nc))) <= 1:
                hits.add((int(nr), int(nc)))
    return sorted(hits)


def _segment_record(
    g: np.ndarray,
    node: tuple[int, int],
    start: tuple[int, int],
) -> dict:
    comp = _same_color_component(g, start)
    comp_set = set(comp)
    color = int(g[start])

    node_neighbors = []
    for dr, dc in DIRS:
        nr = int(node[0] + dr)
        nc = int(node[1] + dc)
        if (nr, nc) in comp_set:
            node_neighbors.append((nr, nc))
    assert len(node_neighbors) == 1, "each path component must touch its node at exactly one endpoint"

    orange_touches = set()
    endpoints = 0
    for r, c in comp:
        degree = 0
        for dr, dc in DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if (nr, nc) in comp_set:
                degree += 1
            elif _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == ORANGE:
                orange_touches.add((nr, nc))
        assert degree <= 2, "path components must be non-branching"
        if degree <= 1:
            endpoints += 1

    assert endpoints in {1, 2}, "path components must be simple paths"
    assert orange_touches == {node}, "each path component may attach to only one orange node"
    return {"node": node, "color": color, "coords": comp}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pair of path endpoints around its orange junction according to the adjacent yellow marker.

    Input structure
    - The background is black.
    - Orange single pixels are junction nodes.
    - Yellow 2-pixel markers indicate how the two attached path segments should rotate.
    - Colored paths are one-pixel-wide orthogonal paths that end at an orange node.

    Rule
    - A horizontal yellow pair means no rotation.
    - A vertical yellow pair means rotate both attached path segments 90 degrees clockwise.
    - A NW-SE diagonal yellow pair means rotate both attached path segments 90 degrees counterclockwise.
    - Yellow markers are erased, orange nodes remain, and rotated path pixels overwrite earlier content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    orange_nodes = sorted(tuple(map(int, xy)) for xy in np.argwhere(g == ORANGE))
    if not orange_nodes:
        out = g.copy()
        out[out == YELLOW] = BG
        return out

    orange_set = set(orange_nodes)
    marker_components = _component_coords(g == YELLOW, connectivity=8)
    marker_by_node: dict[tuple[int, int], int] = {}
    for comp in marker_components:
        step = _decode_marker(comp)
        owners = _touching_nodes(comp, orange_set)
        assert len(owners) == 1, "each yellow marker must belong to exactly one orange node"
        owner = owners[0]
        assert owner not in marker_by_node, "each orange node must have exactly one yellow marker"
        marker_by_node[owner] = step
    assert set(marker_by_node) == orange_set, "every orange node must have one yellow marker"

    special = {BG, YELLOW, ORANGE}
    seen_components: set[tuple[tuple[int, int], ...]] = set()
    segments: list[dict] = []
    for node in orange_nodes:
        neighbors = []
        for dr, dc in DIRS:
            nr = int(node[0] + dr)
            nc = int(node[1] + dc)
            if not _in_bounds(g.shape, nr, nc):
                continue
            color = int(g[nr, nc])
            if color in special:
                continue
            neighbors.append((nr, nc))

        assert len(neighbors) == 2, "each orange node must connect to exactly two path endpoints"
        node_colors = {int(g[pos]) for pos in neighbors}
        assert len(node_colors) == 2, "the two path endpoints at a node must have different colors"

        for start in neighbors:
            record = _segment_record(g, node, start)
            key = tuple(record["coords"])
            assert key not in seen_components, "a path component cannot serve multiple orange nodes"
            seen_components.add(key)
            record["rotation"] = int(marker_by_node[node])
            segments.append(record)

    out = g.copy()
    out[out == YELLOW] = BG
    for record in segments:
        coords = np.array(record["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BG

    for record in segments:
        nr, nc = record["node"]
        color = int(record["color"])
        rotated = []
        for r, c in record["coords"]:
            dr = int(r - nr)
            dc = int(c - nc)
            rr, cc = _rotate_offset(dr, dc, record["rotation"])
            tr = int(nr + rr)
            tc = int(nc + cc)
            assert _in_bounds(g.shape, tr, tc), "rotated path must stay within bounds"
            rotated.append((tr, tc))
        rows = np.array([r for r, _ in rotated], dtype=int)
        cols = np.array([c for _, c in rotated], dtype=int)
        out[rows, cols] = color

    for r, c in orange_nodes:
        out[int(r), int(c)] = ORANGE
    return out


def _marker_step_name(step: int) -> str:
    if step == 0:
        return "horizontal"
    if step == 1:
        return "vertical"
    assert step == -1
    return "diagonal"


def _marker_candidates(step: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if step == 0:
        delta = (0, 1)
    elif step == 1:
        delta = (1, 0)
    else:
        delta = (1, 1)

    seen = set()
    out = []
    for r in range(-2, 3):
        for c in range(-2, 3):
            a = (int(r), int(c))
            b = (int(r + delta[0]), int(c + delta[1]))
            if a == (0, 0) or b == (0, 0):
                continue
            if max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])) > 2:
                continue
            if max(abs(a[0]), abs(a[1])) > 1 and max(abs(b[0]), abs(b[1])) > 1:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
    return out


MARKER_CANDIDATES = {
    step: _marker_candidates(step)
    for step in (0, 1, -1)
}


def _line_cells(direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(int(dr * k), int(dc * k)) for k in range(1, int(length) + 1)]


def _spec_at_origin(
    colors: tuple[int, int],
    directions: tuple[tuple[int, int], tuple[int, int]],
    lengths: tuple[int, int],
    step: int,
) -> dict | None:
    path_cells = []
    used = set()
    for idx, (color, direction, length) in enumerate(zip(colors, directions, lengths, strict=False)):
        cells = _line_cells(direction, length)
        if any(cell in used for cell in cells):
            return None
        used.update(cells)
        path_cells.append({"color": int(color), "dir": direction, "length": int(length), "cells": cells, "index": idx})

    marker_options = []
    for pair in MARKER_CANDIDATES[step]:
        if pair[0] in used or pair[1] in used:
            continue
        marker_options.append(pair)
    if not marker_options:
        return None

    marker = random.choice(marker_options)
    output_segments = []
    for item in path_cells:
        out_dir = _rotate_dir(item["dir"], step)
        out_cells = _line_cells(out_dir, item["length"])
        output_segments.append({"color": item["color"], "cells": out_cells})

    rel_all = {(0, 0)}
    rel_all.update(marker)
    for item in path_cells:
        rel_all.update(item["cells"])
    for item in output_segments:
        rel_all.update(item["cells"])

    rows = [r for r, _ in rel_all]
    cols = [c for _, c in rel_all]
    return {
        "step": int(step),
        "paths": path_cells,
        "marker": marker,
        "out_paths": output_segments,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(box: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    return (int(box[0] - gap), int(box[1] - gap), int(box[2] + gap), int(box[3] + gap))


def _translate_box(box: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (int(r0 + origin[0]), int(c0 + origin[1]), int(r1 + origin[0]), int(c1 + origin[1]))


def _placed_spec(spec: dict, origin: tuple[int, int]) -> dict:
    nr, nc = origin
    marker = tuple((int(nr + dr), int(nc + dc)) for dr, dc in spec["marker"])
    paths = []
    for item in spec["paths"]:
        paths.append(
            {
                "color": int(item["color"]),
                "cells": [(int(nr + dr), int(nc + dc)) for dr, dc in item["cells"]],
            }
        )
    out_paths = []
    for item in spec["out_paths"]:
        out_paths.append(
            {
                "color": int(item["color"]),
                "cells": [(int(nr + dr), int(nc + dc)) for dr, dc in item["cells"]],
            }
        )
    box = _translate_box(spec["bbox"], origin)
    return {
        "node": (int(nr), int(nc)),
        "step": int(spec["step"]),
        "marker": marker,
        "paths": paths,
        "out_paths": out_paths,
        "box": box,
    }


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for item in placed:
        nr, nc = item["node"]
        grid[nr, nc] = ORANGE
        for mr, mc in item["marker"]:
            assert grid[mr, mc] == BG
            grid[mr, mc] = YELLOW
        for path in item["paths"]:
            for r, c in path["cells"]:
                assert grid[r, c] == BG
                grid[r, c] = int(path["color"])
    return grid


def _render_output(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for item in placed:
        for path in item["out_paths"]:
            for r, c in path["cells"]:
                grid[r, c] = int(path["color"])
        nr, nc = item["node"]
        grid[nr, nc] = ORANGE
    return grid


def _choose_node_specs(n_nodes: int) -> list[dict] | None:
    colors = random.sample(PATH_COLORS, 2 * int(n_nodes))

    if n_nodes == 3 and random.random() < 0.7:
        steps = [0, 1, -1]
        random.shuffle(steps)
    else:
        steps = [random.choice((0, 1, -1)) for _ in range(int(n_nodes))]

    specs = []
    for i in range(int(n_nodes)):
        step = int(steps[i])
        color_pair = (int(colors[2 * i]), int(colors[2 * i + 1]))
        spec = None
        for _ in range(80):
            directions = random.choice(DIR_PAIRS)
            max_len = PATH_LENGTH_RANGE[1] if n_nodes == 1 else 3
            lengths = (
                random.randint(PATH_LENGTH_RANGE[0], max_len),
                random.randint(PATH_LENGTH_RANGE[0], max_len),
            )
            spec = _spec_at_origin(color_pair, directions, lengths, step)
            if spec is not None:
                break
        if spec is None:
            return None
        specs.append(spec)
    specs.sort(key=lambda spec: (spec["bbox"][2] - spec["bbox"][0], spec["bbox"][3] - spec["bbox"][1]), reverse=True)
    return specs


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    h, w = shape

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        row_lo = int(-r0)
        row_hi = int(h - 1 - r1)
        col_lo = int(-c0)
        col_hi = int(w - 1 - c1)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(col_lo, col_hi + 1)
        ]
        random.shuffle(candidates)

        chosen = None
        for origin in candidates[:PLACEMENT_ATTEMPTS]:
            box = _expanded_box(_translate_box(spec["bbox"], origin), PACKAGE_GAP)
            if any(_boxes_overlap(box, other) for other in placed_boxes):
                continue
            chosen = origin
            break
        if chosen is None:
            return None

        placed.append(_placed_spec(spec, chosen))
        placed_boxes.append(_expanded_box(_translate_box(spec["bbox"], chosen), PACKAGE_GAP))

    return placed


def _oracle_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt() -> np.ndarray | None:
    n_nodes = random.randint(1, MAX_NODES)
    min_side = GRID_MIN_SIDE_BY_NODE_COUNT[n_nodes]
    shape = (
        random.randint(min_side, GRID_MAX_SIDE),
        random.randint(min_side, GRID_MAX_SIDE),
    )

    specs = _choose_node_specs(n_nodes)
    if specs is None:
        return None
    placed = _place_specs(shape, specs)
    if placed is None:
        return None

    inp = _render_input(shape, placed)
    expected = _render_output(shape, placed)
    if not _oracle_ok(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-node input")

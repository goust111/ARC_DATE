from __future__ import annotations

import numpy as np


MAROON = 9
YELLOW = 4
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLOR_POOL = [0, 1, 2, 3, 5, 6, 7, 8]
BLOCK_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
DISTANCES = (2, 3, 4)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 300
_GEN_COUNTER = 0


def _most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _arrow_cells(direction: str, dist: int, tip_side: int) -> tuple[set[tuple[int, int]], str | None]:
    assert direction in DIR_TO_VEC
    assert dist in DISTANCES
    assert tip_side in (-1, 1)

    cells: set[tuple[int, int]]
    alt_direction = None

    if direction == "right":
        if tip_side < 0:
            cells = {(1, c) for c in range(dist)}
            cells.add((0, dist - 1))
            alt_direction = "down" if dist == 2 else None
        else:
            cells = {(0, c) for c in range(dist)}
            cells.add((1, dist - 1))
            alt_direction = "up" if dist == 2 else None
    elif direction == "left":
        if tip_side < 0:
            cells = {(1, c) for c in range(dist)}
            cells.add((0, 0))
            alt_direction = "down" if dist == 2 else None
        else:
            cells = {(0, c) for c in range(dist)}
            cells.add((1, 0))
            alt_direction = "up" if dist == 2 else None
    elif direction == "down":
        if tip_side < 0:
            cells = {(r, 1) for r in range(dist)}
            cells.add((dist - 1, 0))
            alt_direction = "right" if dist == 2 else None
        else:
            cells = {(r, 0) for r in range(dist)}
            cells.add((dist - 1, 1))
            alt_direction = "left" if dist == 2 else None
    else:
        if tip_side < 0:
            cells = {(r, 1) for r in range(dist)}
            cells.add((0, 0))
            alt_direction = "right" if dist == 2 else None
        else:
            cells = {(r, 0) for r in range(dist)}
            cells.add((0, 1))
            alt_direction = "left" if dist == 2 else None

    return cells, alt_direction


def _build_arrow_shape_map() -> dict[frozenset[tuple[int, int]], list[tuple[str, int]]]:
    shape_map: dict[frozenset[tuple[int, int]], list[tuple[str, int]]] = {}
    for dist in DISTANCES:
        for direction in DIRECTIONS:
            for tip_side in (-1, 1):
                cells, _ = _arrow_cells(direction, dist, tip_side)
                key = _normalize_cells(cells)
                shape_map.setdefault(key, [])
                candidate = (direction, dist)
                if candidate not in shape_map[key]:
                    shape_map[key].append(candidate)
    return shape_map


ARROW_SHAPES = _build_arrow_shape_map()


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    assert len(cols) == 1, "expected exactly one full-height maroon separator"
    sep = cols[0]
    assert 0 < sep < grid.shape[1] - 1, "separator must split the grid into two nonempty parts"
    return sep


def _decode_arrow_candidates(coords: np.ndarray) -> list[tuple[str, int]]:
    shape = _normalize_cells([tuple(map(int, cell)) for cell in coords.tolist()])
    candidates = ARROW_SHAPES.get(shape)
    assert candidates, "instruction component does not match any valid arrow shape"
    return list(candidates)


def _extract_arrows(instr: np.ndarray, bg: int) -> dict[int, list[tuple[str, int]]]:
    arrows: dict[int, list[tuple[str, int]]] = {}
    for comp in _components(instr, bg):
        color = int(comp["color"])
        assert color not in arrows, "expected one arrow per color"
        arrows[color] = _decode_arrow_candidates(comp["coords"])
    assert arrows, "expected at least one instruction arrow"
    return arrows


def _extract_blocks(canvas: np.ndarray, bg: int) -> dict[int, dict]:
    blocks: dict[int, dict] = {}
    for comp in _components(canvas, bg):
        color = int(comp["color"])
        assert color not in blocks, "expected one canvas block per color"
        r0, c0, r1, c1 = comp["bbox"]
        patch = canvas[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "canvas objects must be solid rectangles"
        height, width = patch.shape
        assert min(height, width) >= 2, "canvas blocks must be at least 2 cells thick"
        cells = [tuple(map(int, cell)) for cell in comp["coords"].tolist()]
        blocks[color] = {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "shape": (height, width),
            "cells": cells,
            "cell_set": set(cells),
        }
    assert blocks, "expected at least one canvas block"
    return blocks


def _candidate_move(block: dict, canvas: np.ndarray, bg: int, direction: str, dist: int) -> dict | None:
    dr, dc = DIR_TO_VEC[direction]
    h, w = canvas.shape
    orig_set = block["cell_set"]
    final: set[tuple[int, int]] = set()

    for k in range(1, dist + 1):
        for r, c in block["cells"]:
            rr = r + dr * k
            cc = c + dc * k
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if (rr, cc) not in orig_set and int(canvas[rr, cc]) != bg:
                return None
            if k == dist:
                final.add((rr, cc))

    return {"direction": direction, "dist": dist, "final": final}


def _resolve_moves(canvas: np.ndarray, bg: int, arrows: dict[int, list[tuple[str, int]]], blocks: dict[int, dict]) -> dict[int, tuple[str, int]]:
    assert set(arrows) == set(blocks), "arrow colors and block colors must match exactly"

    valid: dict[int, list[dict]] = {}
    for color, candidates in arrows.items():
        seen: set[tuple[str, int]] = set()
        options: list[dict] = []
        for direction, dist in candidates:
            if (direction, dist) in seen:
                continue
            seen.add((direction, dist))
            cand = _candidate_move(blocks[color], canvas, bg, direction, dist)
            if cand is not None:
                options.append(cand)
        assert options, "every arrow must have at least one feasible decode"
        valid[color] = options

    order = sorted(valid, key=lambda color: len(valid[color]))
    solutions: list[dict[int, tuple[str, int]]] = []
    chosen: dict[int, tuple[str, int]] = {}

    def dfs(idx: int, used_final: set[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if idx == len(order):
            solutions.append(dict(chosen))
            return

        color = order[idx]
        for cand in valid[color]:
            final = cand["final"]
            if not used_final.isdisjoint(final):
                continue
            chosen[color] = (cand["direction"], cand["dist"])
            dfs(idx + 1, used_final | final)
            del chosen[color]

    dfs(0, set())
    assert len(solutions) == 1, "arrow decoding must be unique after applying canvas constraints"
    return solutions[0]


def _apply_moves(canvas: np.ndarray, bg: int, blocks: dict[int, dict], moves: dict[int, tuple[str, int]]) -> np.ndarray:
    out = canvas.copy()

    for block in blocks.values():
        for r, c in block["cells"]:
            out[r, c] = bg

    for color, block in blocks.items():
        direction, dist = moves[color]
        dr, dc = DIR_TO_VEC[direction]
        for k in range(1, dist + 1):
            for r, c in block["cells"]:
                rr = r + dr * k
                cc = c + dc * k
                if int(canvas[rr, cc]) == bg:
                    out[rr, cc] = YELLOW
        for r, c in block["cells"]:
            out[r + dr * dist, c + dc * dist] = color

    return out


def _solve_instance(grid: np.ndarray) -> tuple[np.ndarray, int, dict[int, dict], dict[int, tuple[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _most_common_color(g)
    sep = _find_separator(g)
    instr = g[:, :sep]
    canvas = g[:, sep + 1 :]
    arrows = _extract_arrows(instr, bg)
    blocks = _extract_blocks(canvas, bg)
    moves = _resolve_moves(canvas, bg, arrows, blocks)
    return canvas, bg, blocks, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode colored arrows in the left instruction zone, then move the matching
    canvas rectangles by the encoded direction and distance while leaving a
    yellow trail on the original background path.
    """
    canvas, bg, blocks, moves = _solve_instance(grid)
    return _apply_moves(canvas, bg, blocks, moves)


def _shift_in_bounds(top: int, left: int, height: int, width: int, direction: str, dist: int, h: int, w: int) -> bool:
    dr, dc = DIR_TO_VEC[direction]
    new_top = top + dr * dist
    new_left = left + dc * dist
    return 0 <= new_top and 0 <= new_left and new_top + height <= h and new_left + width <= w


def _travel_bbox(top: int, left: int, height: int, width: int, direction: str, dist: int) -> tuple[int, int, int, int]:
    dr, dc = DIR_TO_VEC[direction]
    end_top = top + dr * dist
    end_left = left + dc * dist
    r0 = min(top, end_top)
    c0 = min(left, end_left)
    r1 = max(top + height - 1, end_top + height - 1)
    c1 = max(left + width - 1, end_left + width - 1)
    return r0, c0, r1, c1


def _block_positions(spec: dict, canvas_h: int, canvas_w: int, reserved: np.ndarray) -> list[tuple[int, int]]:
    height, width = spec["shape"]
    direction = spec["direction"]
    dist = spec["dist"]
    alt_direction = spec["alt_direction"]
    positions: list[tuple[int, int]] = []

    for top in range(canvas_h - height + 1):
        for left in range(canvas_w - width + 1):
            if not _shift_in_bounds(top, left, height, width, direction, dist, canvas_h, canvas_w):
                continue
            if alt_direction is not None and _shift_in_bounds(top, left, height, width, alt_direction, dist, canvas_h, canvas_w):
                continue
            r0, c0, r1, c1 = _travel_bbox(top, left, height, width, direction, dist)
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(canvas_h - 1, r1 + 1)
            pc1 = min(canvas_w - 1, c1 + 1)
            if np.any(reserved[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                continue
            positions.append((top, left))
    return positions


def _arrow_positions(spec: dict, instr_h: int, instr_w: int, reserved: np.ndarray) -> list[tuple[int, int]]:
    cells = spec["arrow_cells"]
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    positions: list[tuple[int, int]] = []

    for top in range(instr_h - max_r):
        for left in range(instr_w - max_c):
            abs_cells = [(top + r, left + c) for r, c in cells]
            blocked = any(reserved[r, c] for r, c in abs_cells)
            if blocked:
                continue
            pr0 = max(0, top - 1)
            pc0 = max(0, left - 1)
            pr1 = min(instr_h - 1, top + max_r + 1)
            pc1 = min(instr_w - 1, left + max_c + 1)
            if np.any(reserved[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                continue
            positions.append((top, left))
    return positions


def _sample_specs(rng: np.random.Generator, count: int, required_direction: str, required_distance: int) -> list[dict]:
    directions = [required_direction]
    distances = [required_distance]
    directions.extend(rng.choice(DIRECTIONS, size=count - 1, replace=True).tolist())
    distances.extend(rng.choice(DISTANCES, size=count - 1, replace=True).tolist())

    specs: list[dict] = []
    for direction, dist in zip(directions, distances):
        shape = BLOCK_SHAPES[int(rng.integers(len(BLOCK_SHAPES)))]
        tip_side = -1 if bool(rng.integers(2)) else 1
        arrow_cells, alt_direction = _arrow_cells(direction, dist, tip_side)
        specs.append(
            {
                "direction": direction,
                "dist": int(dist),
                "shape": shape,
                "tip_side": tip_side,
                "arrow_cells": arrow_cells,
                "alt_direction": alt_direction,
            }
        )
    rng.shuffle(specs)
    return specs


def _place_blocks(specs: list[dict], canvas_h: int, canvas_w: int, colors: list[int], rng: np.random.Generator) -> list[dict] | None:
    indexed_specs = []
    for spec, color in zip(specs, colors):
        spec = dict(spec)
        spec["color"] = color
        height, width = spec["shape"]
        region = _travel_bbox(0, 0, height, width, spec["direction"], spec["dist"])
        area = (region[2] - region[0] + 1) * (region[3] - region[1] + 1)
        spec["travel_area"] = area
        indexed_specs.append(spec)

    indexed_specs.sort(key=lambda spec: (-spec["travel_area"], -spec["shape"][0] * spec["shape"][1]))

    reserved = np.zeros((canvas_h, canvas_w), dtype=bool)
    placed: list[dict] = []

    for spec in indexed_specs:
        positions = _block_positions(spec, canvas_h, canvas_w, reserved)
        if not positions:
            return None
        top, left = positions[int(rng.integers(len(positions)))]
        height, width = spec["shape"]
        r0, c0, r1, c1 = _travel_bbox(top, left, height, width, spec["direction"], spec["dist"])
        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(canvas_h - 1, r1 + 1)
        pc1 = min(canvas_w - 1, c1 + 1)
        reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

        block = dict(spec)
        block["top"] = top
        block["left"] = left
        placed.append(block)

    return placed


def _place_arrows(specs: list[dict], instr_h: int, instr_w: int, rng: np.random.Generator) -> list[dict] | None:
    reserved = np.zeros((instr_h, instr_w), dtype=bool)
    order = sorted(specs, key=lambda spec: (-len(spec["arrow_cells"]), spec["color"]))
    placed: list[dict] = []

    for spec in order:
        positions = _arrow_positions(spec, instr_h, instr_w, reserved)
        if not positions:
            return None
        top, left = positions[int(rng.integers(len(positions)))]
        max_r = max(r for r, _ in spec["arrow_cells"])
        max_c = max(c for _, c in spec["arrow_cells"])
        pr0 = max(0, top - 1)
        pc0 = max(0, left - 1)
        pr1 = min(instr_h - 1, top + max_r + 1)
        pc1 = min(instr_w - 1, left + max_c + 1)
        reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

        arrow = dict(spec)
        arrow["arrow_top"] = top
        arrow["arrow_left"] = left
        placed.append(arrow)

    return placed


def _assemble_grid(canvas_h: int, instr_w: int, canvas_w: int, bg: int, specs: list[dict]) -> tuple[np.ndarray, dict[int, tuple[str, int]], np.ndarray]:
    grid = np.full((canvas_h, instr_w + 1 + canvas_w), bg, dtype=int)
    grid[:, instr_w] = MAROON
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    intended_moves: dict[int, tuple[str, int]] = {}

    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        height, width = spec["shape"]
        canvas[top : top + height, left : left + width] = color
        intended_moves[color] = (spec["direction"], spec["dist"])

        for r, c in spec["arrow_cells"]:
            grid[int(spec["arrow_top"]) + r, int(spec["arrow_left"]) + c] = color

    grid[:, instr_w + 1 :] = canvas
    return grid, intended_moves, canvas


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    best = int(np.argmax(counts))
    if int(vals[best]) != bg:
        return False
    top_count = int(counts[best])
    return int(np.sum(counts == top_count)) == 1


def _gen_attempt(rng: np.random.Generator, call_index: int) -> np.ndarray | None:
    canvas_h = int(rng.integers(10, 16))
    instr_w = int(rng.integers(6, 10))
    canvas_w = int(rng.integers(10, 16))
    bg = int(rng.choice(COLOR_POOL))

    usable_colors = [color for color in COLOR_POOL if color != bg]
    max_objects = min(4, len(usable_colors))
    count = int(rng.integers(2, max_objects + 1))
    colors = rng.choice(usable_colors, size=count, replace=False).tolist()

    required_direction = DIRECTIONS[call_index % len(DIRECTIONS)]
    required_distance = DISTANCES[call_index % len(DISTANCES)]
    specs = _sample_specs(rng, count, required_direction, required_distance)

    placed_blocks = _place_blocks(specs, canvas_h, canvas_w, colors, rng)
    if placed_blocks is None:
        return None

    placed_arrows = _place_arrows(placed_blocks, canvas_h, instr_w, rng)
    if placed_arrows is None:
        return None

    grid, intended_moves, canvas = _assemble_grid(canvas_h, instr_w, canvas_w, bg, placed_arrows)
    if not _bg_is_unique_mode(grid, bg):
        return None

    try:
        solved_canvas, solved_bg, solved_blocks, solved_moves = _solve_instance(grid)
    except AssertionError:
        return None

    if solved_bg != bg:
        return None
    if set(solved_moves) != set(intended_moves):
        return None
    for color, move in intended_moves.items():
        if solved_moves[color] != move:
            return None

    expected = _apply_moves(canvas, bg, _extract_blocks(canvas, bg), intended_moves)
    produced = _apply_moves(solved_canvas, solved_bg, solved_blocks, solved_moves)
    if not np.array_equal(expected, produced):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = np.random.default_rng()
    call_index = _GEN_COUNTER
    _GEN_COUNTER += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng, call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")

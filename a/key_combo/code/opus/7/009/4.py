import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

TOKEN_COLORS = (BLUE, RED, YELLOW)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TOKEN_SHAPES = {
    "solid": {
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    },
    "cross": {(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)},
    "diamond": {(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)},
}
SHAPE_CYCLE = {
    "solid": "cross",
    "cross": "diamond",
    "diamond": "solid",
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            comp = set()
            queue = deque([(r, c)])
            seen.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                comp.add((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != color or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    queue.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_tokens(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bg = get_bg(grid)
    tokens = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color in (bg, GREEN, MAROON):
                continue
            cells = set()
            valid = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    value = int(grid[rr, cc])
                    if value == color:
                        cells.add((dr, dc))
                    elif value != bg:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            shape = None
            for name, offsets in TOKEN_SHAPES.items():
                if cells == offsets:
                    shape = name
                    break
            if shape is None:
                continue
            tokens.append(
                {
                    "center": (r, c),
                    "color": color,
                    "shape": shape,
                    "cells": {(r + dr, c + dc) for dr, dc in cells},
                }
            )
    return tokens


def _gate_info(grid: np.ndarray, comp: set[tuple[int, int]]) -> dict | None:
    gates = []
    for r, c in comp:
        if in_bounds(grid.shape, r - 1, c) and in_bounds(grid.shape, r + 1, c):
            if int(grid[r - 1, c]) == MAROON and int(grid[r + 1, c]) == MAROON:
                gates.append(
                    {
                        "center": (r, c),
                        "gate_ends": {(r - 1, c), (r + 1, c)},
                        "path_dir": (0, 1),
                    }
                )
        if in_bounds(grid.shape, r, c - 1) and in_bounds(grid.shape, r, c + 1):
            if int(grid[r, c - 1]) == MAROON and int(grid[r, c + 1]) == MAROON:
                gates.append(
                    {
                        "center": (r, c),
                        "gate_ends": {(r, c - 1), (r, c + 1)},
                        "path_dir": (1, 0),
                    }
                )
    if len(gates) != 1:
        return None
    return gates[0]


def _ordered_simple_path(comp: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start not in comp:
        return None
    ordered = [start]
    prev = None
    cur = start
    while True:
        nexts = []
        for nr, nc in neighbors4(*cur):
            nxt = (nr, nc)
            if nxt == prev or nxt not in comp:
                continue
            nexts.append(nxt)
        if not nexts:
            break
        if len(nexts) != 1:
            return None
        prev, cur = cur, nexts[0]
        ordered.append(cur)
    if len(ordered) != len(comp):
        return None
    return ordered


def _analyze_component(grid: np.ndarray, comp: set[tuple[int, int]], tokens: list[dict]) -> dict | None:
    degrees = {
        cell: sum((nbr in comp) for nbr in neighbors4(*cell))
        for cell in comp
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return None

    matches = []
    for token in tokens:
        tr, tc = token["center"]
        for endpoint in endpoints:
            if abs(tr - endpoint[0]) + abs(tc - endpoint[1]) == 2 and (
                tr == endpoint[0] or tc == endpoint[1]
            ):
                matches.append((token, endpoint))
    if len(matches) != 1:
        return None

    gate = _gate_info(grid, comp)
    if gate is None:
        return None

    token, start = matches[0]
    path = _ordered_simple_path(comp, start)
    if path is None:
        return None
    if gate["center"] not in path:
        return None
    gate_idx = path.index(gate["center"])
    if not (2 <= gate_idx <= len(path) - 3):
        return None

    before = (path[gate_idx][0] - path[gate_idx - 1][0], path[gate_idx][1] - path[gate_idx - 1][1])
    after = (path[gate_idx + 1][0] - path[gate_idx][0], path[gate_idx + 1][1] - path[gate_idx][1])
    if before != after:
        return None
    if before != gate["path_dir"] and before != (-gate["path_dir"][0], -gate["path_dir"][1]):
        return None

    return {
        "token": token,
        "path": path,
        "gate_center": gate["center"],
        "gate_ends": gate["gate_ends"],
        "gate_idx": gate_idx,
        "parking": path[gate_idx - 1],
    }


def _apply_component(out: np.ndarray, info: dict) -> None:
    color = int(info["token"]["color"])
    old_shape_cells = info["token"]["cells"]
    gate_center = info["gate_center"]
    gate_ends = info["gate_ends"]
    gate_idx = int(info["gate_idx"])
    parking = info["parking"]
    new_shape = TOKEN_SHAPES[SHAPE_CYCLE[info["token"]["shape"]]]
    gate_cells = set(gate_ends) | {gate_center}

    for r, c in old_shape_cells:
        out[r, c] = BLACK

    for r, c in info["path"][:gate_idx]:
        out[r, c] = BLACK

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = parking[0] + dr, parking[1] + dc
            if in_bounds(out.shape, rr, cc):
                out[rr, cc] = BLACK

    for dr, dc in new_shape:
        rr, cc = parking[0] + dr, parking[1] + dc
        if in_bounds(out.shape, rr, cc):
            out[rr, cc] = color

    out[gate_center] = GREEN
    for r, c in gate_ends:
        out[r, c] = MAROON

    for r, c in info["path"][gate_idx + 1 :]:
        out[r, c] = GREEN
    for r, c in gate_cells:
        out[r, c] = GREEN if (r, c) == gate_center else MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 3x3 token along its green path to the gate, transform its shape there, and erase only the traversed path.

    Input structure:
    - The background is black.
    - Each active object is a 4-connected green path with one perpendicular maroon-green-maroon gate and one adjacent colored 3x3 token.
    - Tokens use one color each and appear as a solid 3x3 block, a 5-cell cross, or a 5-cell diagonal diamond.

    Rule:
    - Find the path endpoint that is 4-neighbor adjacent to a token center, then trace the unique simple path from that endpoint toward the gate.
    - Stop the token on the path cell immediately before the gate center, viewed from that starting endpoint.
    - Cycle the token shape in place at that stopping center: solid -> cross -> diamond -> solid, keeping the token color unchanged.
    - Erase the green cells the token traversed, preserve the maroon-green-maroon gate, and leave any unmatched components unchanged.

    Key insight:
    - The gate is both the stopping landmark and the trigger for the shape-cycle update, so the solver must use endpoint-to-gate path order rather than a generic nearest-neighbor placement.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    tokens = _detect_tokens(g)
    token_lookup = {token["center"]: token for token in tokens}
    used_tokens = set()

    for comp in _components_of_color(g, GREEN):
        info = _analyze_component(g, comp, tokens)
        if info is None:
            continue
        center = info["token"]["center"]
        if center in used_tokens or center not in token_lookup:
            continue
        used_tokens.add(center)
        _apply_component(out, info)

    return out


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _sample_segment_lengths(rng: random.Random, count: int, total: int) -> list[int]:
    lengths = [1] * count
    remaining = total - count
    while remaining > 0:
        candidates = [i for i, value in enumerate(lengths) if value < 5]
        if not candidates:
            return []
        idx = rng.choice(candidates)
        lengths[idx] += 1
        remaining -= 1
    return lengths


def _build_polyline(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    segment_count = rng.choice([3, 4])
    total_steps = rng.randint(7, 14)
    lengths = _sample_segment_lengths(rng, segment_count, total_steps)
    if not lengths or max(lengths) < 3:
        return None

    direction = rng.choice(DIRS4)
    directions = [direction]
    for _ in range(segment_count - 1):
        direction = rng.choice([_turn_left(direction), _turn_right(direction)])
        directions.append(direction)

    path = [(0, 0)]
    segment_ranges = []
    cursor = 0
    for direction, steps in zip(directions, lengths):
        start_idx = cursor
        for _ in range(steps):
            last_r, last_c = path[-1]
            path.append((last_r + direction[0], last_c + direction[1]))
            cursor += 1
        segment_ranges.append((start_idx, cursor))

    if len(set(path)) != len(path):
        return None

    for i, a in enumerate(path):
        for j in range(i + 2, len(path)):
            b = path[j]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                return None

    return path, directions, segment_ranges


def _choose_gate_and_token(
    rng: random.Random, path: list[tuple[int, int]], directions: list[tuple[int, int]], segment_ranges: list[tuple[int, int]]
) -> dict | None:
    start_dir = directions[0]
    token_center = (path[0][0] - 2 * start_dir[0], path[0][1] - 2 * start_dir[1])
    other_end = path[-1]
    if abs(token_center[0] - other_end[0]) + abs(token_center[1] - other_end[1]) == 2 and (
        token_center[0] == other_end[0] or token_center[1] == other_end[1]
    ):
        return None
    token_box = {
        (token_center[0] + dr, token_center[1] + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    path_set = set(path)
    if token_box & path_set:
        return None

    candidates = []
    for seg_idx, (start_idx, end_idx) in enumerate(segment_ranges):
        if end_idx - start_idx < 2:
            continue
        direction = directions[seg_idx]
        perp = (-direction[1], direction[0])
        for gate_idx in range(start_idx + 1, end_idx):
            if not (2 <= gate_idx <= len(path) - 3):
                continue
            gate = path[gate_idx]
            gate_ends = {
                (gate[0] + perp[0], gate[1] + perp[1]),
                (gate[0] - perp[0], gate[1] - perp[1]),
            }
            if gate_ends & path_set:
                continue
            if gate_ends & token_box:
                continue
            if any(
                any(nei in path_set and nei != gate for nei in neighbors4(*end))
                for end in gate_ends
            ):
                continue
            parking = path[gate_idx - 1]
            stop_box = {
                (parking[0] + dr, parking[1] + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
            candidates.append(
                {
                    "gate_idx": gate_idx,
                    "gate_center": gate,
                    "gate_ends": gate_ends,
                    "token_center": token_center,
                    "token_box": token_box,
                    "stop_box": stop_box,
                }
            )
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_component(rng: random.Random, color: int, shape_name: str) -> dict | None:
    for _ in range(200):
        built = _build_polyline(rng)
        if built is None:
            continue
        path, directions, segment_ranges = built
        placement = _choose_gate_and_token(rng, path, directions, segment_ranges)
        if placement is None:
            continue

        reserved = set(path) | placement["gate_ends"] | placement["token_box"] | placement["stop_box"]
        r0, c0, r1, c1 = bbox(reserved)
        shift_r, shift_c = -r0, -c0
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 16 or w > 16:
            continue

        local = np.full((h, w), BLACK, dtype=int)
        shifted_path = [(r + shift_r, c + shift_c) for r, c in path]
        for r, c in shifted_path:
            local[r, c] = GREEN
        shifted_gate_ends = {(r + shift_r, c + shift_c) for r, c in placement["gate_ends"]}
        for r, c in shifted_gate_ends:
            local[r, c] = MAROON

        token_center = (
            placement["token_center"][0] + shift_r,
            placement["token_center"][1] + shift_c,
        )
        for dr, dc in TOKEN_SHAPES[shape_name]:
            local[token_center[0] + dr, token_center[1] + dc] = color

        return {
            "grid": local,
            "shape": shape_name,
            "color": color,
        }
    return None


def _expanded_clear(mask: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(mask.shape[0], top + h + pad)
    c1 = min(mask.shape[1], left + w + pad)
    return not mask[r0:r1, c0:c1].any()


def _place_components(rng: random.Random, components: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    used = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(components)), key=lambda i: components[i]["grid"].size, reverse=True)

    for idx in order:
        tile = components[idx]["grid"]
        h, w = tile.shape
        candidates = []
        for top in range(height - h + 1):
            for left in range(width - w + 1):
                if _expanded_clear(used, top, left, h, w, pad=2):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        patch = tile != BLACK
        grid[top : top + h, left : left + w][patch] = tile[patch]
        used[top : top + h, left : left + w] = True
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    if not (inp.ndim == 2 and out.ndim == 2):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    tokens = _detect_tokens(inp)
    if len(tokens) not in (2, 3):
        return False
    if len({token["color"] for token in tokens}) != len(tokens):
        return False
    if len({token["shape"] for token in tokens}) < 2:
        return False

    infos = []
    seen_centers = set()
    for comp in _components_of_color(inp, GREEN):
        info = _analyze_component(inp, comp, tokens)
        if info is None:
            return False
        center = info["token"]["center"]
        if center in seen_centers:
            return False
        seen_centers.add(center)
        infos.append(info)
    if len(infos) != len(tokens):
        return False

    for info in infos:
        token = info["token"]
        transformed = TOKEN_SHAPES[SHAPE_CYCLE[token["shape"]]]
        gate_cells = set(info["gate_ends"]) | {info["gate_center"]}
        parking = info["parking"]

        for r, c in token["cells"]:
            if out[r, c] != BLACK:
                return False

        for r, c in info["path"][: info["gate_idx"]]:
            if out[r, c] == GREEN:
                return False
        for r, c in info["path"][info["gate_idx"] + 1 :]:
            if out[r, c] != GREEN:
                return False

        for r, c in info["gate_ends"]:
            if out[r, c] != MAROON:
                return False
        if out[info["gate_center"]] != GREEN:
            return False

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = parking[0] + dr, parking[1] + dc
                if not in_bounds(out.shape, rr, cc):
                    return False
                if (rr, cc) in gate_cells:
                    expected = GREEN if (rr, cc) == info["gate_center"] else MAROON
                elif (dr, dc) in transformed:
                    expected = token["color"]
                else:
                    expected = BLACK
                if out[rr, cc] != expected:
                    return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    shape_names = list(TOKEN_SHAPES.keys())
    for _ in range(600):
        component_count = rng.choice([2, 3])
        colors = rng.sample(list(TOKEN_COLORS), component_count)
        if component_count == 2:
            shapes = rng.sample(shape_names, 2)
        else:
            if rng.random() < 0.6:
                shapes = rng.sample(shape_names, 3)
            else:
                pair = rng.sample(shape_names, 2)
                shapes = pair + [rng.choice(pair)]
                rng.shuffle(shapes)

        components = []
        ok = True
        for color, shape in zip(colors, shapes):
            comp = _build_component(rng, color, shape)
            if comp is None:
                ok = False
                break
            components.append(comp)
        if not ok:
            continue

        height = rng.randint(20, 28)
        width = rng.randint(20, 28)
        inp = _place_components(rng, components, height, width)
        if inp is None:
            continue
        out = main(inp)
        if np.array_equal(inp, out):
            continue
        if _validate_semantics(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")

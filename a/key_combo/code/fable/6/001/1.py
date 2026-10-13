from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
RING_COLORS = (1, 3, 4, 6, 7, 8, 9)

# Clockwise perimeter order around a 3x3 ring, starting at top-left.
RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)

SIDE_DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
LEFT_TURN = {(-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0)}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}

GEN_ATTEMPTS = 500
GEOMETRY_ATTEMPTS = 200
RING_VALUE_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _ring_box_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ring_perimeter_values(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    r, c = center
    return np.array([int(grid[r + dr, c + dc]) for dr, dc in RING_OFFSETS], dtype=int)


def _paint_ring(grid: np.ndarray, center: tuple[int, int], values: np.ndarray) -> None:
    r, c = center
    for (dr, dc), color in zip(RING_OFFSETS, values):
        grid[r + dr, c + dc] = int(color)


def _rod_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    assert (dr == 0) ^ (dc == 0), "adjacent rings must align on one axis"
    dist = abs(dr) + abs(dc)
    assert dist >= 4, "rings must be separated by at least one rod pixel"
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    return [(ar + step_r * k, ac + step_c * k) for k in range(2, dist - 1)]


def _detect_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict] = []
    occupied_perimeter: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center not in (BG, RED):
                continue

            values = _ring_perimeter_values(g, (r, c))
            if np.any(values == BG):
                continue

            perimeter = {(r + dr, c + dc) for dr, dc in RING_OFFSETS}
            assert not (perimeter & occupied_perimeter), "ring perimeters must be disjoint"
            occupied_perimeter |= perimeter
            rings.append(
                {
                    "center": (r, c),
                    "anchor": center == RED,
                    "values": values,
                }
            )

    assert rings, "expected at least one ring"
    return rings


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rings = _detect_rings(g)
    side_map: dict[tuple[int, int], tuple[int, str]] = {}
    ring_cells: set[tuple[int, int]] = set()

    for idx, ring in enumerate(rings):
        cr, cc = ring["center"]
        if ring["anchor"]:
            assert int(g[cr, cc]) == RED
        else:
            assert int(g[cr, cc]) == BG

        ring_cells |= {(cr + dr, cc + dc) for dr, dc in RING_OFFSETS}
        for side, (dr, dc) in SIDE_DIRS.items():
            side_coord = (cr + dr, cc + dc)
            assert side_coord not in side_map, "ring side midpoints must be unique"
            side_map[side_coord] = (idx, side)

    grey_pixels = {tuple(map(int, coord)) for coord in np.argwhere(g == GREY)}
    rod_only_grey = grey_pixels - ring_cells
    edges: dict[tuple[int, int], dict] = {}
    used_grey: set[tuple[int, int]] = set()

    for idx, ring in enumerate(rings):
        cr, cc = ring["center"]
        for side, (dr, dc) in SIDE_DIRS.items():
            rr, cc2 = cr + 2 * dr, cc + 2 * dc
            if not _in_bounds(g.shape, rr, cc2):
                continue
            if (rr, cc2) in side_map:
                continue
            if int(g[rr, cc2]) != GREY:
                continue

            cells: list[tuple[int, int]] = []
            while _in_bounds(g.shape, rr, cc2):
                if (rr, cc2) in side_map:
                    other_idx, other_side = side_map[(rr, cc2)]
                    break
                assert int(g[rr, cc2]) == GREY, "rod path must be solid grey"
                cells.append((rr, cc2))
                rr += dr
                cc2 += dc
            else:
                raise AssertionError("rod must terminate at another ring")

            assert cells, "adjacent rings without a grey rod are invalid"
            assert other_idx != idx, "rod cannot connect a ring to itself"
            assert other_side == OPPOSITE[side], "rod must join facing side midpoints"

            key = tuple(sorted((idx, other_idx)))
            edge = {"rings": key, "length": len(cells), "cells": frozenset(cells)}
            prev = edges.get(key)
            if prev is None:
                edges[key] = edge
            else:
                assert prev == edge, "rod must look the same from both ends"
            used_grey.update(cells)

    assert used_grey == rod_only_grey, "every grey pixel must belong to exactly one rod"

    adjacency: list[list[tuple[int, int]]] = [[] for _ in rings]
    for edge in edges.values():
        a, b = edge["rings"]
        length = int(edge["length"])
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))

    anchor_ids = [idx for idx, ring in enumerate(rings) if ring["anchor"]]
    assert len(anchor_ids) == 1, "expected exactly one red-center anchor ring"
    anchor = anchor_ids[0]

    for nbrs in adjacency:
        assert len(nbrs) <= 2, "the rods must form a chain, not a branch"

    if len(rings) == 1:
        assert len(adjacency[anchor]) == 0
        order = [anchor]
        lengths: list[int] = []
    else:
        assert len(adjacency[anchor]) == 1, "the anchor ring must sit at one end of the chain"
        assert sum(1 for nbrs in adjacency if len(nbrs) == 1) == 2

        order = [anchor]
        lengths = []
        prev = -1
        cur = anchor
        visited = {anchor}

        while True:
            nxts = [(nxt, length) for nxt, length in adjacency[cur] if nxt != prev]
            if not nxts:
                break
            assert len(nxts) == 1
            nxt, length = nxts[0]
            assert nxt not in visited, "the ring graph must be acyclic"
            visited.add(nxt)
            order.append(nxt)
            lengths.append(int(length))
            prev, cur = cur, nxt

        assert len(order) == len(rings), "the ring graph must be connected"

    return {
        "rings": rings,
        "order": order,
        "lengths": lengths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each non-anchor ring along the grey chain, with rod length setting the shift amount.

    Input structure:
    - The background is black.
    - Each gear is a 3x3 ring of eight colored perimeter cells around a center cell.
    - Exactly one ring has a red center; the other ring centers stay black.
    - Grey straight rods connect facing side-midpoint cells so that all rings form one chain.

    Rule:
    - Keep the red-center anchor ring fixed.
    - Traverse the chain outward from that anchor end.
    - Rotate the first moved ring clockwise, then alternate direction on every later ring.
    - The number of perimeter steps for each moved ring equals the grey-pixel length of the rod that connects it to the previous ring.
    - Leave the rods and center cells unchanged.

    Key insight:
    - The rods encode both the chain order and the exact rotation amount, so the output depends on following the meshed-gear propagation rather than treating rings independently.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    out = g.copy()

    for outward_idx, ring_idx in enumerate(scene["order"][1:], start=1):
        steps = int(scene["lengths"][outward_idx - 1]) % 8
        signed = steps if outward_idx % 2 == 1 else -steps
        center = scene["rings"][ring_idx]["center"]
        values = scene["rings"][ring_idx]["values"]
        _paint_ring(out, center, np.roll(values, signed))

    return out


def _sample_ring_values() -> np.ndarray:
    distinct = random.randint(5, min(7, len(RING_COLORS)))
    chosen = random.sample(RING_COLORS, distinct)
    values = chosen + [random.choice(chosen) for _ in range(8 - distinct)]
    random.shuffle(values)
    return np.array(values, dtype=int)


def _build_lengths(n_rings: int) -> list[int]:
    options = {
        3: (1, 2, 3, 5, 6),
        4: (1, 2, 3, 5),
        5: (1, 2, 3),
    }[n_rings]
    while True:
        lengths = [random.choice(options) for _ in range(n_rings - 1)]
        if len(set(lengths)) >= 2:
            return lengths


def _candidate_direction_orders(prev_dir: tuple[int, int] | None, must_turn: bool) -> list[list[tuple[int, int]]]:
    if prev_dir is None:
        dirs = list(SIDE_DIRS.values())
        random.shuffle(dirs)
        return [dirs]

    left = LEFT_TURN[prev_dir]
    right = RIGHT_TURN[prev_dir]
    turn_dirs = [left, right]
    random.shuffle(turn_dirs)

    if must_turn:
        return [turn_dirs + [prev_dir]]

    orders = [turn_dirs + [prev_dir], [prev_dir] + turn_dirs]
    random.shuffle(orders)
    return orders


def _can_place_segment(
    reserved: set[tuple[int, int]],
    prev_center: tuple[int, int],
    next_center: tuple[int, int],
) -> bool:
    next_box = _ring_box_cells(next_center)
    rod = _rod_cells(prev_center, next_center)
    prev_box = _ring_box_cells(prev_center)

    if next_box & reserved:
        return False
    if any(nb in reserved for cell in next_box for nb in _neighbors8(cell)):
        return False

    for idx, cell in enumerate(rod):
        if cell in reserved:
            return False
        nearby = {nb for nb in _neighbors8(cell) if nb in reserved}
        if idx == 0:
            if any(nb not in prev_box for nb in nearby):
                return False
        elif nearby:
            return False

    return True


def _build_geometry(n_rings: int, lengths: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    require_turn = n_rings >= 4 and random.random() < 0.7

    for _ in range(GEOMETRY_ATTEMPTS):
        centers = [(0, 0)]
        dirs: list[tuple[int, int]] = []
        reserved = set(_ring_box_cells(centers[0]))
        turned = False
        prev_dir: tuple[int, int] | None = None
        ok = True

        for step_idx, length in enumerate(lengths):
            steps_left = len(lengths) - step_idx
            must_turn = require_turn and prev_dir is not None and not turned and steps_left == 1
            placed = False

            for order in _candidate_direction_orders(prev_dir, must_turn):
                for direction in order:
                    dr, dc = direction
                    prev_center = centers[-1]
                    next_center = (prev_center[0] + dr * (length + 3), prev_center[1] + dc * (length + 3))

                    if any(max(abs(next_center[0] - r), abs(next_center[1] - c)) < 4 for r, c in centers[:-1]):
                        continue
                    if not _can_place_segment(reserved, prev_center, next_center):
                        continue

                    centers.append(next_center)
                    reserved |= _ring_box_cells(next_center)
                    reserved |= set(_rod_cells(prev_center, next_center))
                    dirs.append(direction)
                    if prev_dir is not None and direction != prev_dir:
                        turned = True
                    prev_dir = direction
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                ok = False
                break

        if ok and (turned or not require_turn):
            return centers, dirs

    return None


def _place_on_canvas(
    rel_centers: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    all_cells: set[tuple[int, int]] = set()
    for center in rel_centers:
        all_cells |= _ring_box_cells(center)
    for a, b in zip(rel_centers, rel_centers[1:]):
        all_cells |= set(_rod_cells(a, b))

    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    assert base_h <= 30 and base_w <= 30

    extra_h = random.randint(0, min(30 - base_h, 6))
    extra_w = random.randint(0, min(30 - base_w, 6))
    h = base_h + extra_h
    w = base_w + extra_w
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    offset = (top - min_r, left - min_c)
    abs_centers = [(r + offset[0], c + offset[1]) for r, c in rel_centers]
    return (h, w), abs_centers


def _render_input(scene: dict) -> np.ndarray:
    h, w = scene["shape"]
    grid = np.full((h, w), BG, dtype=int)

    for idx, center in enumerate(scene["centers"]):
        _paint_ring(grid, center, scene["ring_values"][idx])
        if idx == 0:
            grid[center] = RED

    for a, b in zip(scene["centers"], scene["centers"][1:]):
        for r, c in _rod_cells(a, b):
            grid[r, c] = GREY

    return grid


def _render_output(scene: dict, shift_rule) -> np.ndarray:
    out = _render_input(scene)
    for outward_idx, center in enumerate(scene["centers"][1:], start=1):
        length = scene["lengths"][outward_idx - 1]
        shift = int(shift_rule(outward_idx, length))
        _paint_ring(out, center, np.roll(scene["ring_values"][outward_idx], shift))
    return out


def _build_scene() -> dict | None:
    n_rings = random.choices((3, 4, 5), weights=(3, 4, 3), k=1)[0]
    lengths = _build_lengths(n_rings)
    geometry = _build_geometry(n_rings, lengths)
    if geometry is None:
        return None
    rel_centers, dirs = geometry
    shape, centers = _place_on_canvas(rel_centers)

    ring_values = []
    for idx in range(n_rings):
        for _ in range(RING_VALUE_ATTEMPTS):
            values = _sample_ring_values()
            if idx == 0:
                if np.array_equal(values, np.roll(values, 1)):
                    continue
            else:
                true_shift = lengths[idx - 1] if idx % 2 == 1 else -lengths[idx - 1]
                if np.array_equal(values, np.roll(values, true_shift)):
                    continue
                if np.array_equal(np.roll(values, true_shift), np.roll(values, -true_shift)):
                    continue
            ring_values.append(values)
            break
        else:
            return None

    return {
        "shape": shape,
        "centers": centers,
        "rel_centers": rel_centers,
        "dirs": dirs,
        "lengths": lengths,
        "ring_values": ring_values,
    }


def _semantic_checks(scene: dict) -> bool:
    inp = _render_input(scene)
    expected = _render_output(
        scene,
        lambda outward_idx, length: length if outward_idx % 2 == 1 else -length,
    )

    if np.array_equal(inp, expected):
        return False

    parsed = _analyze_scene(inp)
    parsed_centers = [parsed["rings"][ring_idx]["center"] for ring_idx in parsed["order"]]
    parsed_values = [parsed["rings"][ring_idx]["values"] for ring_idx in parsed["order"]]
    if parsed_centers != scene["centers"]:
        return False
    if parsed["lengths"] != scene["lengths"]:
        return False
    if any(not np.array_equal(a, b) for a, b in zip(parsed_values, scene["ring_values"])):
        return False

    shortcut_all_cw = _render_output(scene, lambda outward_idx, length: length)
    shortcut_fixed1 = _render_output(
        scene,
        lambda outward_idx, length: 1 if outward_idx % 2 == 1 else -1,
    )
    shortcut_flipped = _render_output(
        scene,
        lambda outward_idx, length: -length if outward_idx % 2 == 1 else length,
    )
    shortcut_index = _render_output(
        scene,
        lambda outward_idx, length: outward_idx if outward_idx % 2 == 1 else -outward_idx,
    )
    shortcut_anchor = expected.copy()
    _paint_ring(shortcut_anchor, scene["centers"][0], np.roll(scene["ring_values"][0], 1))
    shortcut_anchor[scene["centers"][0]] = RED

    if np.array_equal(expected, shortcut_all_cw):
        return False
    if np.array_equal(expected, shortcut_fixed1):
        return False
    if np.array_equal(expected, shortcut_flipped):
        return False
    if np.array_equal(expected, shortcut_index):
        return False
    if np.array_equal(expected, shortcut_anchor):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_scene()
        if scene is None:
            continue
        if _semantic_checks(scene):
            return _render_input(scene)
    raise RuntimeError("failed to generate a valid gear-chain input")

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_RANGE = (22, 30)
NETWORK_COUNT_OPTIONS = (1, 2, 3)
NETWORK_COUNT_WEIGHTS = (3, 4, 2)
STRAIGHT_SEG_RANGE = (4, 8)
HSEG_RANGE = (2, 4)
VSEG_RANGE = (2, 4)
ACTUAL_T_FOR_CROSS = (3, 4)
CASE_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _square_cells(bbox):
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _components_of_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _receiver_bbox_from_touch(touch, arrival_dir, offset: int):
    r, c = touch
    dr, dc = arrival_dir
    if (dr, dc) == (0, 1):
        return r - offset, c, r - offset + 1, c + 1
    if (dr, dc) == (0, -1):
        return r - offset, c - 1, r - offset + 1, c
    if (dr, dc) == (1, 0):
        return r, c - offset, r + 1, c - offset + 1
    if (dr, dc) == (-1, 0):
        return r - 1, c - offset, r, c - offset + 1
    raise AssertionError("invalid direction")


def _receiver_shape_cells(receiver_bbox, touch, arrival_dir, gate_count: int):
    if gate_count == 0:
        return _square_cells(receiver_bbox)

    dr, dc = arrival_dir
    if gate_count in (1, 2):
        if dr != 0:
            return {(touch[0] + k * dr, touch[1]) for k in range(3)}
        return {(touch[0], touch[1] + k * dc) for k in range(3)}

    center = (touch[0] + dr, touch[1] + dc)
    cr, cc = center
    return {
        center,
        (cr - 1, cc),
        (cr + 1, cc),
        (cr, cc - 1),
        (cr, cc + 1),
    }


def _classify_non_special_components(grid: np.ndarray):
    emitters = []
    gates = {}
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BG, GREY, MAROON):
            continue
        for comp in _components_of_color(grid, color):
            cell_set = set(comp)
            if len(comp) == 4:
                bbox = _bbox(comp)
                r0, c0, r1, c1 = bbox
                assert r1 == r0 + 1 and c1 == c0 + 1, "non-special 4-cell component must be a 2x2 emitter"
                emitters.append({"color": color, "cells": cell_set, "bbox": bbox})
            else:
                assert len(comp) == 1, "non-special components must be either emitters or 1-pixel gates"
                gates[comp[0]] = color
    return emitters, gates


def _receiver_components(grid: np.ndarray):
    receivers = []
    lookup = {}
    for comp in _components_of_color(grid, MAROON):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert len(comp) == 4 and r1 == r0 + 1 and c1 == c0 + 1, "receivers must be 2x2 maroon blocks"
        idx = len(receivers)
        cell_set = set(comp)
        receivers.append({"bbox": bbox, "cells": cell_set})
        for cell in cell_set:
            lookup[cell] = idx
    return receivers, lookup


def _trace_from_emitter(grid: np.ndarray, emitter, gate_cells, receiver_lookup):
    h, w = grid.shape
    grey = grid == GREY

    starts = []
    for er, ec in emitter["cells"]:
        for nr, nc in _neighbors4(er, ec, h, w):
            if grey[nr, nc]:
                starts.append(((nr, nc), (er, ec)))

    assert len(starts) == 1, "each emitter must touch exactly one channel cell"
    current, prev = starts[0]
    gate_count = 0
    seen_edges = set()

    while True:
        edge = (prev, current)
        assert edge not in seen_edges, "channel path must not cycle"
        seen_edges.add(edge)

        if current in gate_cells:
            next_cells = [
                (nr, nc)
                for nr, nc in _neighbors4(current[0], current[1], h, w)
                if (nr, nc) != prev and grey[nr, nc]
            ]
            assert len(next_cells) == 1, "gate must lead into exactly one outgoing grey channel"
            prev, current = current, next_cells[0]
            continue

        next_cells = []
        for nr, nc in _neighbors4(current[0], current[1], h, w):
            if (nr, nc) == prev:
                continue
            if grey[nr, nc] or (nr, nc) in gate_cells or (nr, nc) in receiver_lookup:
                next_cells.append((nr, nc))

        assert len(next_cells) == 1, "grey channel must have exactly one forward continuation"
        nxt = next_cells[0]
        if nxt in receiver_lookup:
            arrival_dir = (nxt[0] - current[0], nxt[1] - current[1])
            return receiver_lookup[nxt], nxt, arrival_dir, gate_count
        if nxt in gate_cells:
            gate_count += 1
        prev, current = current, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor and reshape each maroon receiver according to how many colored gates a signal crosses.

    Input structure:
    - The background is black.
    - Thin grey channels form isolated orthogonal networks, with single non-grey gate pixels joining
      consecutive channel segments at right-angle turns.
    - Each network has one 2x2 monochrome emitter block whose color differs from its gates.
    - Each network ends at one 2x2 maroon receiver block touched by the final grey channel cell.

    Rule:
    - For each emitter, trace the unique signal path along grey cells toward its receiver.
    - When the path reaches a gate pixel, continue through the orthogonal grey segment connected on the
      other side of that gate and count one turn.
    - Replace only the maroon receiver: with zero gates it stays a 2x2 block, with one gate it becomes
      a 3-cell bar along the arrival axis, with two gates it becomes a 3-cell bar along the arrival axis,
      and with three or more gates it becomes a 5-cell cross.
    - The replacement shape is colored with the emitter color; channels, gates, and emitters stay unchanged.

    Key insight:
    The gate count is not just a path property: after following the hidden emitter-to-receiver route,
    that count determines which geometric deformation replaces the terminal maroon block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    emitters, gates = _classify_non_special_components(g)
    receivers, receiver_lookup = _receiver_components(g)
    assert emitters and receivers, "grid must contain emitters and receivers"

    out = g.copy()
    used_receivers = set()

    for emitter in emitters:
        receiver_idx, touch, arrival_dir, gate_count = _trace_from_emitter(g, emitter, set(gates), receiver_lookup)
        assert receiver_idx not in used_receivers, "each receiver must belong to exactly one network"
        used_receivers.add(receiver_idx)

        receiver = receivers[receiver_idx]
        shape = _receiver_shape_cells(receiver["bbox"], touch, arrival_dir, gate_count)

        for rr, cc in receiver["cells"]:
            out[rr, cc] = BG
        for rr, cc in shape:
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "receiver deformation must stay in bounds"
            if (rr, cc) not in receiver["cells"]:
                assert g[rr, cc] == BG, "only the receiver may expand into background"
            out[rr, cc] = emitter["color"]

    return out


def _translate_cells(cells, dr: int, dc: int):
    return {(r + dr, c + dc) for r, c in cells}


def _translate_paints(paints, dr: int, dc: int):
    return {(r + dr, c + dc): color for (r, c), color in paints.items()}


def _dilate_cells(cells):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _build_local_network(branch: int, emitter_color: int):
    gate_count = random.choice(ACTUAL_T_FOR_CROSS) if branch == 3 else branch

    hsign = random.choice((-1, 1))
    dirs = []
    for i in range(gate_count + 1):
        if i % 2 == 0:
            dirs.append((0, hsign))
        else:
            dirs.append((random.choice((-1, 1)), 0))

    lengths = []
    for i, (dr, _dc) in enumerate(dirs):
        if gate_count == 0:
            lengths.append(random.randint(*STRAIGHT_SEG_RANGE))
        elif dr == 0:
            lengths.append(random.randint(*HSEG_RANGE))
        else:
            lengths.append(random.randint(*VSEG_RANGE))

    segments = []
    gates = []
    cur = (0, 0)
    for i, direction in enumerate(dirs):
        dr, dc = direction
        seg = [(cur[0] + dr * k, cur[1] + dc * k) for k in range(lengths[i])]
        segments.append(seg)
        end = seg[-1]
        if i < gate_count:
            gate = (end[0] + dr, end[1] + dc)
            gates.append(gate)
            nxt_dir = dirs[i + 1]
            cur = (gate[0] + nxt_dir[0], gate[1] + nxt_dir[1])

    start_grey = segments[0][0]
    start_touch = (start_grey[0] - dirs[0][0], start_grey[1] - dirs[0][1])
    emitter_offset = random.randint(0, 1)
    emitter_bbox = _receiver_bbox_from_touch(start_touch, (-dirs[0][0], -dirs[0][1]), emitter_offset)
    emitter_cells = _square_cells(emitter_bbox)

    end_grey = segments[-1][-1]
    receiver_touch = (end_grey[0] + dirs[-1][0], end_grey[1] + dirs[-1][1])
    receiver_offset = random.randint(0, 1)
    receiver_bbox = _receiver_bbox_from_touch(receiver_touch, dirs[-1], receiver_offset)
    receiver_cells = _square_cells(receiver_bbox)

    gate_palette = [color for color in COLOR_POOL if color != emitter_color]
    gate_colors = [random.choice(gate_palette) for _ in range(gate_count)]

    paints = {}
    for cell in emitter_cells:
        paints[cell] = emitter_color
    for seg in segments:
        for cell in seg:
            paints[cell] = GREY
    for cell, color in zip(gates, gate_colors):
        paints[cell] = color
    for cell in receiver_cells:
        paints[cell] = MAROON

    input_cells = set(paints)
    assert len(input_cells) == len(paints), "local network must not self-overlap"

    output_shape = _receiver_shape_cells(receiver_bbox, receiver_touch, dirs[-1], gate_count)
    reserved = _dilate_cells(input_cells | output_shape)

    return {
        "branch": branch,
        "gate_count": gate_count,
        "paints": paints,
        "receiver_cells": receiver_cells,
        "output_shape": output_shape,
        "emitter_color": emitter_color,
        "reserved": reserved,
    }


def _place_models(models, h: int, w: int):
    occupied = np.zeros((h, w), dtype=bool)
    placed = []

    order = sorted(
        models,
        key=lambda model: (
            _bbox(model["reserved"])[2] - _bbox(model["reserved"])[0] + 1
        )
        * (
            _bbox(model["reserved"])[3] - _bbox(model["reserved"])[1] + 1
        ),
        reverse=True,
    )

    for model in order:
        r0, c0, r1, c1 = _bbox(model["reserved"])
        row_lo = -r0
        row_hi = h - 1 - r1
        col_lo = -c0
        col_hi = w - 1 - c1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(PLACE_ATTEMPTS):
            dr = random.randint(row_lo, row_hi)
            dc = random.randint(col_lo, col_hi)
            shifted_reserved = _translate_cells(model["reserved"], dr, dc)
            if any(occupied[r, c] for r, c in shifted_reserved):
                continue

            for r, c in shifted_reserved:
                occupied[r, c] = True
            placed.append(
                {
                    "paints": _translate_paints(model["paints"], dr, dc),
                    "receiver_cells": _translate_cells(model["receiver_cells"], dr, dc),
                    "output_shape": _translate_cells(model["output_shape"], dr, dc),
                    "emitter_color": model["emitter_color"],
                }
            )
            success = True
            break

        if not success:
            return None

    return placed


def _case_from_models(models):
    max_h = 0
    max_w = 0
    for model in models:
        r0, c0, r1, c1 = _bbox(model["reserved"])
        max_h = max(max_h, r1 - r0 + 1)
        max_w = max(max_w, c1 - c0 + 1)

    low_h = max(GRID_SIDE_RANGE[0], max_h)
    low_w = max(GRID_SIDE_RANGE[0], max_w)
    if low_h > GRID_SIDE_RANGE[1] or low_w > GRID_SIDE_RANGE[1]:
        return None

    h = random.randint(low_h, GRID_SIDE_RANGE[1])
    w = random.randint(low_w, GRID_SIDE_RANGE[1])
    placed = _place_models(models, h, w)
    if placed is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    for model in placed:
        for (r, c), color in model["paints"].items():
            inp[r, c] = color

    out = inp.copy()
    for model in placed:
        for r, c in model["receiver_cells"]:
            out[r, c] = BG
        for r, c in model["output_shape"]:
            out[r, c] = model["emitter_color"]

    return inp, out


def _generate_case():
    for _ in range(CASE_ATTEMPTS):
        network_count = random.choices(NETWORK_COUNT_OPTIONS, weights=NETWORK_COUNT_WEIGHTS, k=1)[0]
        emitter_colors = random.sample(COLOR_POOL, network_count)
        branches = [random.choice((0, 1, 2, 3)) for _ in range(network_count)]
        models = [_build_local_network(branch, emitter_color) for branch, emitter_color in zip(branches, emitter_colors)]
        case = _case_from_models(models)
        if case is None:
            continue
        inp, expected = case
        if np.array_equal(main(inp), expected):
            return inp, expected
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]

from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {value: key for key, value in SIDE_TO_DIR.items()}
BASE_COLORS = [1, 2, 6, 7, 8, 9]

GRID_RANGE = (18, 28)
STEP_COUNT_RANGE = (3, 5)
STEP_LENGTH_RANGE = (1, 4)
BLOCK_DIMS = [
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
]
BLOCK_GAPS = [2, 3, 4]
ANCHOR_MARGIN = 4
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 80


SEED_PATTERNS = [
    {
        "cells": {(0, 0)},
        "anchors": {"up": (0, 0), "down": (0, 0), "left": (0, 0), "right": (0, 0)},
    },
    {
        "cells": {(0, 0), (0, 1)},
        "anchors": {"left": (0, 0), "right": (0, 1)},
    },
    {
        "cells": {(0, 0), (1, 0)},
        "anchors": {"up": (0, 0), "down": (1, 0)},
    },
    {
        "cells": {(0, 0), (0, 1), (0, 2)},
        "anchors": {"left": (0, 0), "right": (0, 2)},
    },
    {
        "cells": {(0, 0), (1, 0), (2, 0)},
        "anchors": {"up": (0, 0), "down": (2, 0)},
    },
    {
        "cells": {(0, 0), (1, 0), (1, 1)},
        "anchors": {"up": (0, 0), "right": (1, 1)},
    },
    {
        "cells": {(0, 0), (0, 1), (1, 0)},
        "anchors": {"right": (0, 1), "down": (1, 0)},
    },
    {
        "cells": {(0, 0), (0, 1), (1, 1)},
        "anchors": {"left": (0, 0), "down": (1, 1)},
    },
    {
        "cells": {(0, 1), (1, 0), (1, 1)},
        "anchors": {"up": (0, 1), "left": (1, 0)},
    },
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _boundary_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r, c in sorted(cells):
        if any((nr, nc) not in cells for nr, nc in _neighbors4(r, c)):
            out.append((r, c))
    return out


def _point_to_rect_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox
    options = []
    if r0 <= r <= r1:
        if c < c0:
            options.append(c0 - c)
        elif c > c1:
            options.append(c - c1)
    if c0 <= c <= c1:
        if r < r0:
            options.append(r0 - r)
        elif r > r1:
            options.append(r - r1)
    if options:
        return min(options)
    return min(abs(r - rr) + abs(c - cc) for rr in (r0, r1) for cc in (c0, c1))


def _best_seed_anchor(
    seed_boundary: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> tuple[int, tuple[int, int], bool]:
    best_dist: int | None = None
    best_points: list[tuple[int, int]] = []
    for point in seed_boundary:
        dist = _point_to_rect_distance(point, bbox)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_points = [point]
        elif dist == best_dist:
            best_points.append(point)
    assert best_dist is not None
    return best_dist, min(best_points), len(best_points) == 1


def _find_seed(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g == GREEN)
    assert comps, "expected a green seed component"
    h, w = g.shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    candidates = []
    for comp in comps:
        bbox = _bbox(comp)
        bh, bw = _bbox_size(bbox)
        if bh > 3 or bw > 3 or len(comp) > 9:
            continue
        r0, c0, r1, c1 = bbox
        comp_center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        dist = abs(comp_center[0] - center[0]) + abs(comp_center[1] - center[1])
        candidates.append((dist, -len(comp), bbox, set(comp)))
    assert candidates, "expected a central seed within 3x3"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return set(candidates[0][3])


def _find_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != BLACK) & (g != GREEN) & (g != GREY)
    blocks = []
    for comp in _components(mask):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        vals = {int(v) for v in sub.reshape(-1)}
        if BLACK in vals or GREEN in vals or GREY in vals:
            continue
        base_colors = vals - {YELLOW}
        if len(base_colors) != 1:
            continue
        length = int(np.count_nonzero(sub == YELLOW))
        if length == 0:
            continue
        side_hits = []
        if r0 > 0:
            cols = [c for c in range(c0, c1 + 1) if int(g[r0 - 1, c]) == GREY]
            if cols:
                side_hits.append(("up", (r0 - 1, cols[0])))
        if r1 + 1 < h:
            cols = [c for c in range(c0, c1 + 1) if int(g[r1 + 1, c]) == GREY]
            if cols:
                side_hits.append(("down", (r1 + 1, cols[0])))
        if c0 > 0:
            rows = [r for r in range(r0, r1 + 1) if int(g[r, c0 - 1]) == GREY]
            if rows:
                side_hits.append(("left", (rows[0], c0 - 1)))
        if c1 + 1 < w:
            rows = [r for r in range(r0, r1 + 1) if int(g[r, c1 + 1]) == GREY]
            if rows:
                side_hits.append(("right", (rows[0], c1 + 1)))
        assert len(side_hits) == 1, "each block must have exactly one direction pin"
        side, pin = side_hits[0]
        blocks.append(
            {
                "bbox": bbox,
                "base_color": int(next(iter(base_colors))),
                "length": length,
                "direction": SIDE_TO_DIR[side],
                "pin": pin,
            }
        )
    blocks.sort(key=lambda block: block["bbox"])
    assert blocks, "expected at least one encoded block"
    return blocks


def _growth_plan(seed_cells: set[tuple[int, int]], blocks: list[dict]) -> dict:
    seed_boundary = _boundary_cells(seed_cells)
    assert seed_boundary, "seed must expose boundary cells"
    unused = set(range(len(blocks)))
    steps = []
    clear = True

    first_candidates = []
    for idx, block in enumerate(blocks):
        dist, anchor, unique_anchor = _best_seed_anchor(seed_boundary, block["bbox"])
        first_candidates.append((dist, block["bbox"], idx, anchor, unique_anchor))
    first_candidates.sort(key=lambda item: (item[0], item[1]))
    first_dist, _bbox0, first_idx, current, first_anchor_unique = first_candidates[0]
    if len(first_candidates) > 1 and first_candidates[1][0] == first_dist:
        clear = False
    if not first_anchor_unique:
        clear = False

    while unused:
        if steps:
            candidates = [
                (_point_to_rect_distance(current, blocks[idx]["bbox"]), blocks[idx]["bbox"], idx)
                for idx in sorted(unused)
            ]
            candidates.sort(key=lambda item: (item[0], item[1]))
            chosen_dist, _chosen_bbox, chosen_idx = candidates[0]
            if len(candidates) > 1 and candidates[1][0] == chosen_dist:
                clear = False
        else:
            chosen_idx = first_idx

        block = blocks[chosen_idx]
        dr, dc = block["direction"]
        length = int(block["length"])
        end = (current[0] + dr * length, current[1] + dc * length)
        steps.append(
            {
                "index": chosen_idx,
                "start": current,
                "end": end,
                "direction": (dr, dc),
                "length": length,
            }
        )
        current = end
        unused.remove(chosen_idx)

    return {"steps": steps, "clear": clear}


def _paint_steps(grid: np.ndarray, steps: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for step in steps:
        r, c = step["start"]
        dr, dc = step["direction"]
        for k in range(1, step["length"] + 1):
            rr = r + dr * k
            cc = c + dc * k
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(out[rr, cc]) in (BLACK, GREEN):
                out[rr, cc] = GREEN
    return out


def _solve(grid: np.ndarray, want_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    seed_cells = _find_seed(g)
    blocks = _find_blocks(g)
    plan = _growth_plan(seed_cells, blocks)
    out = _paint_steps(g, plan["steps"])
    if want_trace:
        return out, {"seed": seed_cells, "blocks": blocks, "plan": plan}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green fractal branch from the seed by reading nearby blocks as direction-and-length instructions.

    Input structure:
    - The background is black, and a small connected green seed sits near the grid center within a 3x3 footprint.
    - Elsewhere, there are colored rectangular blocks whose interiors may contain yellow marker pixels.
    - Each block has exactly one grey pin touching one outside edge cell, above, below, left, or right of the rectangle.

    Rule:
    - Decode each block's growth direction from the side on which its grey pin appears.
    - Decode each block's growth length from the number of yellow pixels inside that block.
    - First choose the unused block nearest to the seed boundary, start from the closest seed boundary pixel, and draw a green line in the decoded direction for the decoded length; then repeat from the new branch endpoint with the next nearest unused block until all blocks are consumed.

    Key insight:
    - The output path is determined by three coupled cues: pin side gives direction, yellow-count gives segment length, and geometric proximity decides the order in which the instructions are applied.
    """
    return _solve(grid)


def _seed_bbox(pattern: dict) -> tuple[int, int]:
    rows = [r for r, _ in pattern["cells"]]
    cols = [c for _, c in pattern["cells"]]
    return max(rows) + 1, max(cols) + 1


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, length + 1)]


def _expanded_bbox_cells(bbox: tuple[int, int, int, int], h: int, w: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1)
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1)
    }


def _pin_faces_anchor(location_side: str, pin_side: str) -> bool:
    return (
        (location_side == "left" and pin_side == "right")
        or (location_side == "right" and pin_side == "left")
        or (location_side == "up" and pin_side == "down")
        or (location_side == "down" and pin_side == "up")
    )


def _marker_cells(
    bbox: tuple[int, int, int, int], length: int, rng: random.Random
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    assert len(cells) > length
    rng.shuffle(cells)
    return set(cells[:length])


def _candidate_placements(
    h: int,
    w: int,
    anchor: tuple[int, int],
    location_sides: list[str],
    grow_side: str,
    length: int,
    occupied: set[tuple[int, int]],
) -> list[dict]:
    candidates = []
    area_dims = [dims for dims in BLOCK_DIMS if dims[0] * dims[1] > length]
    ar, ac = anchor
    for location_side in location_sides:
        min_gap = 3 if _pin_faces_anchor(location_side, grow_side) else 2
        for gap in BLOCK_GAPS:
            if gap < min_gap:
                continue
            for bh, bw in area_dims:
                offsets = range(bh) if location_side in ("left", "right") else range(bw)
                for offset in offsets:
                    if location_side == "left":
                        r0 = ar - offset
                        c1 = ac - gap
                        c0 = c1 - bw + 1
                    elif location_side == "right":
                        r0 = ar - offset
                        c0 = ac + gap
                        c1 = c0 + bw - 1
                    elif location_side == "up":
                        c0 = ac - offset
                        r1 = ar - gap
                        r0 = r1 - bh + 1
                    else:
                        c0 = ac - offset
                        r0 = ar + gap
                        r1 = r0 + bh - 1
                    if location_side in ("left", "right"):
                        r1 = r0 + bh - 1
                    else:
                        c1 = c0 + bw - 1
                    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
                        continue
                    bbox = (r0, c0, r1, c1)
                    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                    if cells & occupied:
                        continue
                    if _expanded_bbox_cells(bbox, h, w) & occupied:
                        continue
                    pin_candidates = []
                    if grow_side == "up" and r0 > 0:
                        pin_candidates = [(r0 - 1, c) for c in range(c0, c1 + 1)]
                    elif grow_side == "down" and r1 + 1 < h:
                        pin_candidates = [(r1 + 1, c) for c in range(c0, c1 + 1)]
                    elif grow_side == "left" and c0 > 0:
                        pin_candidates = [(r, c0 - 1) for r in range(r0, r1 + 1)]
                    elif grow_side == "right" and c1 + 1 < w:
                        pin_candidates = [(r, c1 + 1) for r in range(r0, r1 + 1)]
                    for pin in pin_candidates:
                        if pin in occupied or pin in cells:
                            continue
                        candidates.append({"bbox": bbox, "cells": cells, "pin": pin, "side": location_side})
    return candidates


def _sample_path(
    h: int,
    w: int,
    seed_cells: set[tuple[int, int]],
    start: tuple[int, int],
    step_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], set[tuple[int, int]]] | None:
    anchors = [start]
    directions = []
    lengths = []
    occupied = set(seed_cells)
    current = start
    for step_idx in range(step_count):
        options = []
        dirs = DIRS4[:]
        rng.shuffle(dirs)
        lengths = list(range(STEP_LENGTH_RANGE[0], STEP_LENGTH_RANGE[1] + 1))
        rng.shuffle(lengths)
        for direction in dirs:
            for length in lengths:
                segment = _segment_cells(current, direction, length)
                if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in segment):
                    continue
                if any(cell in occupied for cell in segment):
                    continue
                end = segment[-1]
                if step_idx < step_count - 1:
                    if not (ANCHOR_MARGIN <= end[0] < h - ANCHOR_MARGIN):
                        continue
                    if not (ANCHOR_MARGIN <= end[1] < w - ANCHOR_MARGIN):
                        continue
                options.append((direction, segment))
        if not options:
            return None
        direction, segment = rng.choice(options)
        directions.append(direction)
        lengths.append(len(segment))
        occupied.update(segment)
        current = segment[-1]
        anchors.append(current)
    return anchors, directions, lengths, occupied


def _render_input(
    h: int,
    w: int,
    seed_cells: set[tuple[int, int]],
    blocks: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in seed_cells:
        grid[r, c] = GREEN
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = block["base_color"]
        for r, c in block["markers"]:
            grid[r, c] = YELLOW
        pr, pc = block["pin"]
        grid[pr, pc] = GREY
    return grid


def _planned_steps(
    start: tuple[int, int], directions: list[tuple[int, int]], lengths: list[int]
) -> list[dict]:
    steps = []
    current = start
    for direction, length in zip(directions, lengths):
        dr, dc = direction
        end = (current[0] + dr * length, current[1] + dc * length)
        steps.append(
            {
                "start": current,
                "end": end,
                "direction": direction,
                "length": length,
            }
        )
        current = end
    return steps


def _sample_blocks(
    h: int,
    w: int,
    seed_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    lengths: list[int],
    first_side: str,
    rng: random.Random,
) -> list[dict] | None:
    if len(lengths) > len(BASE_COLORS):
        return None
    colors = rng.sample(BASE_COLORS, len(lengths))
    grow_sides = [DIR_TO_SIDE[direction] for direction in directions]
    base_occupied = set(seed_cells) | set(path_cells)
    for _ in range(PLACEMENT_ATTEMPTS):
        occupied = set(base_occupied)
        blocks = []
        ok = True
        for idx, (anchor, grow_side, length, color) in enumerate(
            zip(anchors[:-1], grow_sides, lengths, colors)
        ):
            if idx == 0:
                location_sides = [first_side]
            else:
                location_sides = list(SIDE_TO_DIR.keys())
                rng.shuffle(location_sides)
            candidates = _candidate_placements(
                h=h,
                w=w,
                anchor=anchor,
                location_sides=location_sides,
                grow_side=grow_side,
                length=length,
                occupied=occupied,
            )
            if not candidates:
                ok = False
                break
            chosen = rng.choice(candidates)
            bbox = chosen["bbox"]
            markers = _marker_cells(bbox, length, rng)
            block = {
                "bbox": bbox,
                "pin": chosen["pin"],
                "direction": SIDE_TO_DIR[grow_side],
                "length": length,
                "base_color": color,
                "markers": markers,
            }
            blocks.append(block)
            occupied.update(chosen["cells"])
            occupied.add(chosen["pin"])
        if ok:
            return blocks
    return None


def _validate_generated_case(
    grid: np.ndarray,
    planned_sequence: list[tuple[int, tuple[int, int], int]],
    planned_steps: list[dict],
    expected_start: tuple[int, int],
) -> bool:
    try:
        expected = _paint_steps(grid, planned_steps)
        actual, trace = _solve(grid, want_trace=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    parsed_steps = trace["plan"]["steps"]
    if not parsed_steps or parsed_steps[0]["start"] != expected_start:
        return False
    if not trace["plan"]["clear"]:
        return False
    parsed_blocks = trace["blocks"]
    parsed_sequence = [
        (
            parsed_blocks[step["index"]]["base_color"],
            parsed_blocks[step["index"]]["direction"],
            parsed_blocks[step["index"]]["length"],
        )
        for step in parsed_steps
    ]
    return parsed_sequence == planned_sequence


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_RANGE)
        w = rng.randint(*GRID_RANGE)
        step_count = rng.randint(*STEP_COUNT_RANGE)
        pattern = rng.choice(SEED_PATTERNS)
        seed_h, seed_w = _seed_bbox(pattern)
        seed_r0 = rng.randint(max(1, h // 2 - 2), min(h - seed_h - 2, h // 2 + 1))
        seed_c0 = rng.randint(max(1, w // 2 - 2), min(w - seed_w - 2, w // 2 + 1))
        seed_cells = _translate(pattern["cells"], seed_r0, seed_c0)

        first_side = rng.choice(list(pattern["anchors"].keys()))
        anchor_rel = pattern["anchors"][first_side]
        first_start = (seed_r0 + anchor_rel[0], seed_c0 + anchor_rel[1])

        path = _sample_path(h, w, seed_cells, first_start, step_count, rng)
        if path is None:
            continue
        anchors, directions, lengths, path_cells = path
        planned = _planned_steps(first_start, directions, lengths)

        blocks = _sample_blocks(
            h=h,
            w=w,
            seed_cells=seed_cells,
            path_cells=path_cells,
            anchors=anchors,
            directions=directions,
            lengths=lengths,
            first_side=first_side,
            rng=rng,
        )
        if blocks is None:
            continue

        grid = _render_input(h, w, seed_cells, blocks)
        planned_sequence = [
            (block["base_color"], block["direction"], block["length"]) for block in blocks
        ]
        if _validate_generated_case(grid, planned_sequence, planned, first_start):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

import random
from typing import Optional

import numpy as np


BG = 0

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)

CW_TURN = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}

MISSING_TO_DIR = {
    (0, 1): UP,
    (0, 0): LEFT,
    (1, 1): RIGHT,
    (1, 0): DOWN,
}

DIR_TO_MISSING = {direction: missing for missing, direction in MISSING_TO_DIR.items()}

GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (14, 23)
ARROW_COUNT_RANGE = (5, 6)
PALETTE = tuple(range(1, 10))
MIN_STRAIGHT_LEN = 3
MIN_PRE_TURN_LEN = 2
MIN_POST_TURN_LEN = 2
MIN_TOTAL_RAY_PIXELS = 12
TEMPLATE_TRIES = 120


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _instruction_sequence(
    grid: np.ndarray, bg: int
) -> tuple[list[int], set[tuple[int, int]]]:
    cols = [int(c) for c in np.where(grid[0] != bg)[0]]
    assert cols, "expected a non-empty instruction sequence in row 0"
    assert cols == list(range(cols[0], cols[-1] + 1)), "instruction pixels must be contiguous"
    colors = [int(grid[0, c]) for c in cols]
    assert len(colors) == len(set(colors)), "instruction colors must be unique"
    return colors, {(0, c) for c in cols}


def _arrow_info(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid[1:] == color)
    assert coords.shape[0] == 3, "each arrow must contribute exactly 3 non-instruction pixels"
    cells = {(int(r) + 1, int(c)) for r, c in coords}

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must fit in a 2x2 box"

    box = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    missing = list(box - cells)
    assert len(missing) == 1, "arrow box must have exactly one missing corner"
    start = missing[0]
    offset = (start[0] - r0, start[1] - c0)
    direction = MISSING_TO_DIR[offset]

    return {
        "color": int(color),
        "cells": cells,
        "box": box,
        "start": start,
        "direction": direction,
    }


def _trace_ray(
    start: tuple[int, int],
    direction: tuple[int, int],
    previous_rays: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> Optional[dict]:
    if not _in_bounds(start, shape):
        return None
    if start in blockers or start in previous_rays:
        return None

    current = start
    dr, dc = direction
    turned = False
    painted: list[tuple[int, int]] = []
    painted_set: set[tuple[int, int]] = set()
    post_turn_count = 0
    needs_paint = True

    while True:
        if needs_paint and current not in previous_rays:
            if current in blockers or current in painted_set:
                return None
            painted.append(current)
            painted_set.add(current)
            if turned:
                post_turn_count += 1
        needs_paint = True

        nxt = (current[0] + dr, current[1] + dc)
        if not _in_bounds(nxt, shape):
            return {
                "painted": painted,
                "turned": turned,
                "post_turn_count": post_turn_count,
                "pre_turn_count": len(painted) - post_turn_count,
                "reason": "boundary",
            }

        if nxt in previous_rays:
            if turned:
                return {
                    "painted": painted,
                    "turned": turned,
                    "post_turn_count": post_turn_count,
                    "pre_turn_count": len(painted) - post_turn_count,
                    "reason": "second_collision",
                }
            turned = True
            dr, dc = CW_TURN[(dr, dc)]
            needs_paint = False
            continue

        if nxt in blockers or nxt in painted_set:
            return {
                "painted": painted,
                "turned": turned,
                "post_turn_count": post_turn_count,
                "pre_turn_count": len(painted) - post_turn_count,
                "reason": "blocked",
            }

        current = nxt


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    sequence, instruction_cells = _instruction_sequence(g, bg)

    arrow_by_color: dict[int, dict] = {}
    blockers = set(instruction_cells)
    for color in sequence:
        info = _arrow_info(g, color)
        arrow_by_color[color] = info
        blockers.update(info["cells"])

    out = g.copy()
    previous_rays: set[tuple[int, int]] = set()
    traces: list[dict] = []

    for color in sequence:
        info = arrow_by_color[color]
        trace = _trace_ray(info["start"], info["direction"], previous_rays, blockers, g.shape)
        assert trace is not None, "ray trace must be valid"
        for r, c in trace["painted"]:
            assert out[r, c] == bg, "rays may only fill background pixels"
            out[r, c] = int(color)
        previous_rays.update(trace["painted"])
        traces.append(trace)

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid)[0]


def _expand(points: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in points:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _find_arrow_placement(
    shape: tuple[int, int],
    direction: tuple[int, int],
    want_turn: bool,
    occupied_cells: set[tuple[int, int]],
    ray_blockers: set[tuple[int, int]],
    arrow_halo: set[tuple[int, int]],
    previous_rays: set[tuple[int, int]],
) -> Optional[dict]:
    h, w = shape
    missing = DIR_TO_MISSING[direction]
    def try_box(r0: int, c0: int) -> Optional[dict]:
        box = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
        start = (r0 + missing[0], c0 + missing[1])
        arrow_cells = box - {start}

        if box & arrow_halo:
            return None
        if box & previous_rays:
            return None
        if box & ray_blockers:
            return None
        if start in occupied_cells:
            return None
        if arrow_cells & occupied_cells:
            return None

        blockers = set(ray_blockers)
        blockers.update(arrow_cells)
        trace = _trace_ray(start, direction, previous_rays, blockers, shape)
        if trace is None or trace["reason"] != "boundary":
            return None

        if want_turn:
            if not trace["turned"]:
                return None
            if trace["pre_turn_count"] < MIN_PRE_TURN_LEN:
                return None
            if trace["post_turn_count"] < MIN_POST_TURN_LEN:
                return None
        else:
            if trace["turned"]:
                return None
            if len(trace["painted"]) < MIN_STRAIGHT_LEN:
                return None

        return {
            "box": box,
            "cells": arrow_cells,
            "start": start,
            "direction": direction,
            "trace": trace,
        }

    candidates = [(r0, c0) for r0 in range(2, h - 2) for c0 in range(1, w - 2)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        placement = try_box(r0, c0)
        if placement is not None:
            return placement

    return None


def _build_grid(
    shape: tuple[int, int],
    sequence_start: int,
    colors: list[int],
    placements: list[dict],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for idx, color in enumerate(colors):
        grid[0, sequence_start + idx] = int(color)
    for color, placement in zip(colors, placements):
        for r, c in placement["cells"]:
            grid[r, c] = int(color)
    return grid


def _placement_from_start(start: tuple[int, int], direction: tuple[int, int]) -> Optional[dict]:
    missing = DIR_TO_MISSING[direction]
    r0 = start[0] - missing[0]
    c0 = start[1] - missing[1]
    if r0 < 1 or c0 < 0:
        return None
    box = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    if min(r for r, _ in box) < 1:
        return None
    arrow_cells = box - {start}
    return {
        "box": box,
        "cells": arrow_cells,
        "start": start,
        "direction": direction,
    }


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    max_arrows = min(ARROW_COUNT_RANGE[1], len(PALETTE), w - 4)
    if max_arrows < ARROW_COUNT_RANGE[0]:
        return None

    count = random.randint(ARROW_COUNT_RANGE[0], max_arrows)
    colors = random.sample(PALETTE, count)

    for _ in range(TEMPLATE_TRIES):
        base_row = random.randint(2, 4)
        gap1 = random.randint(2, 3)
        gap2 = random.randint(2, 3)
        gap3 = random.randint(2, 3)
        r_right = base_row
        r_up = r_right + gap1
        r_left = r_up + gap2
        r_down = r_left + gap3
        if r_down > h - 3:
            continue

        max_c_up = min(w // 3, w - 8)
        if max_c_up < 2:
            continue
        c_up = random.randint(2, max_c_up)

        min_c_right = c_up + 3
        max_c_right = w - 4
        if min_c_right > max_c_right:
            continue
        c_right = random.randint(min_c_right, max_c_right)

        min_c_left = max(c_right + 1, w // 2)
        max_c_left = w - 2
        if min_c_left > max_c_left:
            continue
        c_left = random.randint(min_c_left, max_c_left)

        down_candidates = [c for c in range(2, w - 2) if c != c_up]
        if not down_candidates:
            continue
        c_down = random.choice(down_candidates)

        c5_max = min(c_right - 1, c_left - 1)
        c5_candidates = [c for c in range(2, c5_max + 1) if c not in {c_up - 1, c_up, c_down}]
        if not c5_candidates:
            continue
        c5 = random.choice(c5_candidates)

        s5_min = max(r_right + 2, r_up + 1)
        s5_max = r_left - 2
        if s5_min > s5_max:
            continue
        s5 = random.randint(s5_min, s5_max)

        placements = [
            _placement_from_start((r_up, c_up), UP),
            _placement_from_start((r_left, c_left), LEFT),
            _placement_from_start((r_down, c_down), DOWN),
            _placement_from_start((r_right, c_right), RIGHT),
            _placement_from_start((s5, c5), DOWN),
        ]

        protected_top_cols = {c_up}
        if count == 6:
            c6_candidates = [
                c
                for c in range(2, c_right)
                if c not in {c_up - 1, c_up, c5, c_down}
            ]
            if not c6_candidates:
                continue
            c6 = random.choice(c6_candidates)

            s6_min = r_up + 1
            s6_max = r_left - 2
            if s6_min > s6_max:
                continue
            s6 = random.randint(s6_min, s6_max)
            placements.append(_placement_from_start((s6, c6), UP))
            protected_top_cols.add(c6)

        sequence_candidates = [
            s for s in range(w - count + 1) if all(not (s <= c < s + count) for c in protected_top_cols)
        ]
        if not sequence_candidates:
            continue
        sequence_start = random.choice(sequence_candidates)

        if any(p is None for p in placements):
            continue

        boxes = [p["box"] for p in placements]
        if any(a & b for i, a in enumerate(boxes) for b in boxes[i + 1 :]):
            continue

        instruction_cells = {(0, sequence_start + idx) for idx in range(count)}

        grid = _build_grid((h, w), sequence_start, colors, placements)
        try:
            out, traces = _simulate(grid)
        except AssertionError:
            continue

        expected_turns = [False, False, False, False, True]
        if count == 6:
            expected_turns.append(False)
        if [bool(trace["turned"]) for trace in traces] != expected_turns:
            continue
        if any(trace["reason"] != "boundary" for trace in traces):
            continue

        all_boxes = [set(p["box"]) for p in placements]
        for idx, trace in enumerate(traces):
            other_boxes = set().union(*(all_boxes[:idx] + all_boxes[idx + 1 :]))
            if any(cell in other_boxes for cell in trace["painted"]):
                break
        else:
            if int((out != grid).sum()) >= MIN_TOTAL_RAY_PIXELS:
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")

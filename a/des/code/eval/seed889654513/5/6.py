from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
USABLE_COLORS = (1, 2, 3, 4, 6, 8, 9)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
PERPENDICULARS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PATH_COUNT_RANGE = (1, 3)
EXTRA_RECEIVER_RANGE = (0, 2)
RELAY_COUNT_VALUES = (0, 1, 2)
SEG_LEN_VALUES = tuple(range(2, 8))
GEN_RETRIES = 500
PATH_PLAN_RETRIES = 120
EXTRA_RECEIVER_RETRIES = 80


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _advance(pos: tuple[int, int], direction: tuple[int, int], step: int = 1) -> tuple[int, int]:
    return pos[0] + direction[0] * step, pos[1] + direction[1] * step


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _receiver_cells(top_left: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0 = top_left
    frame: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    full: list[tuple[int, int]] = []
    for dr in range(4):
        for dc in range(4):
            cell = (r0 + dr, c0 + dc)
            full.append(cell)
            if 0 < dr < 3 and 0 < dc < 3:
                interior.append(cell)
            else:
                frame.append(cell)
    return frame, interior, full


def _tower_cells(side: str, anchor: int, shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    h, w = shape
    if side == "top":
        cells = [(0, anchor), (0, anchor + 1), (1, anchor), (1, anchor + 1)]
        top_left = (0, anchor)
    elif side == "bottom":
        cells = [(h - 2, anchor), (h - 2, anchor + 1), (h - 1, anchor), (h - 1, anchor + 1)]
        top_left = (h - 2, anchor)
    elif side == "left":
        cells = [(anchor, 0), (anchor, 1), (anchor + 1, 0), (anchor + 1, 1)]
        top_left = (anchor, 0)
    else:
        cells = [(anchor, w - 2), (anchor, w - 1), (anchor + 1, w - 2), (anchor + 1, w - 1)]
        top_left = (anchor, w - 2)
    return cells, top_left, SIDE_TO_DIR[side]


def _launch_cells_for_tower(side: str, top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    if side == "top":
        return [(r0 + 1, c0), (r0 + 1, c0 + 1)]
    if side == "bottom":
        return [(r0, c0), (r0, c0 + 1)]
    if side == "left":
        return [(r0, c0 + 1), (r0 + 1, c0 + 1)]
    return [(r0, c0), (r0 + 1, c0)]


def _receiver_top_left_from_hit(hit: tuple[int, int], direction: tuple[int, int], offset: int) -> tuple[int, int]:
    hr, hc = hit
    if direction == (1, 0):
        return hr, hc - offset
    if direction == (-1, 0):
        return hr - 3, hc - offset
    if direction == (0, 1):
        return hr - offset, hc
    return hr - offset, hc - 3


def _render_input(shape: tuple[int, int], paths: list[dict], extra_receivers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in paths:
        tower_color = int(path["tower_color"])
        for r, c in path["tower_cells"]:
            grid[r, c] = tower_color
        for relay in path["relays"]:
            ir, ic = relay["input_chip"]
            rr, rc = relay["relay"]
            or_, oc = relay["output_chip"]
            grid[ir, ic] = int(relay["input_color"])
            grid[rr, rc] = ORANGE
            grid[or_, oc] = int(relay["output_color"])
        frame, _interior, _full = _receiver_cells(path["receiver_top_left"])
        for r, c in frame:
            grid[r, c] = GREY
    for top_left in extra_receivers:
        frame, _interior, _full = _receiver_cells(top_left)
        for r, c in frame:
            grid[r, c] = GREY
    return grid


def _render_output(inp: np.ndarray, paths: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for path in paths:
        for segment in path["segments"]:
            color = int(segment["color"])
            for r, c in segment["cells"]:
                out[r, c] = color
        _frame, interior, _full = _receiver_cells(path["receiver_top_left"])
        for r, c in interior:
            out[r, c] = int(path["receiver_fill_color"])
    return out


def _path_footprint(path: dict) -> set[tuple[int, int]]:
    footprint = set(path["tower_cells"])
    for segment in path["segments"]:
        footprint.update(segment["cells"])
    for relay in path["relays"]:
        footprint.add(relay["input_chip"])
        footprint.add(relay["relay"])
        footprint.add(relay["output_chip"])
    _frame, _interior, full = _receiver_cells(path["receiver_top_left"])
    footprint.update(full)
    return footprint


def _detect_receivers(grid: np.ndarray) -> list[dict]:
    receivers: list[dict] = []
    for comp in _components4(grid == GREY):
        if len(comp) != 12:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue
        frame, interior, _full = _receiver_cells((r0, c0))
        if set(comp) != set(frame):
            continue
        if any(int(grid[r, c]) != BG for r, c in interior):
            continue
        receivers.append(
            {
                "top_left": (r0, c0),
                "frame": set(frame),
                "interior": list(interior),
            }
        )
    receivers.sort(key=lambda item: (item["top_left"][0], item["top_left"][1]))
    return receivers


def _detect_relays(grid: np.ndarray) -> dict[tuple[int, int], dict]:
    h, w = grid.shape
    relays: dict[tuple[int, int], dict] = {}
    for r, c in zip(*np.where(grid == ORANGE)):
        rr = int(r)
        cc = int(c)
        chips: dict[tuple[int, int], int] = {}
        for dr, dc in DIRS:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds((h, w), nr, nc):
                continue
            col = int(grid[nr, nc])
            if col not in (BG, GREY, ORANGE):
                chips[(dr, dc)] = col
        if len(chips) != 2 or len(set(chips.values())) != 2:
            raise AssertionError("each relay must have exactly two differently colored chips")
        relays[(rr, cc)] = {"chips": chips}
    return relays


def _detect_towers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    towers: list[dict] = []
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY, ORANGE)]
    for color in colors:
        for comp in _components4(grid == color):
            if len(comp) != 4:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            if set(comp) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            sides = []
            if r0 == 0:
                sides.append("top")
            if r1 == h - 1:
                sides.append("bottom")
            if c0 == 0:
                sides.append("left")
            if c1 == w - 1:
                sides.append("right")
            if len(sides) != 1:
                continue
            side = sides[0]
            towers.append(
                {
                    "color": color,
                    "side": side,
                    "top_left": (r0, c0),
                    "starts": _launch_cells_for_tower(side, (r0, c0)),
                    "direction": SIDE_TO_DIR[side],
                }
            )
    towers.sort(key=lambda item: (item["side"], item["top_left"][0], item["top_left"][1], item["color"]))
    tower_colors = [tower["color"] for tower in towers]
    if len(tower_colors) != len(set(tower_colors)):
        raise AssertionError("tower colors must be distinct")
    return towers


def _simulate_launch(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    relays: dict[tuple[int, int], dict],
    receiver_by_border: dict[tuple[int, int], int],
) -> dict | None:
    h, w = grid.shape
    pos = start
    cur_dir = direction
    cur_color = color
    painted: list[tuple[int, int, int]] = []
    seen_states: set[tuple[int, int, int, int, int]] = set()

    for _ in range(h * w * 6):
        nr, nc = pos[0] + cur_dir[0], pos[1] + cur_dir[1]
        if not _in_bounds((h, w), nr, nc):
            return None
        recv_id = receiver_by_border.get((nr, nc))
        if recv_id is not None:
            return {"painted": painted, "receiver": recv_id, "fill_color": cur_color}

        cell = int(grid[nr, nc])
        if cell == BG:
            painted.append((nr, nc, cur_color))
            pos = (nr, nc)
            continue

        if cell == cur_color:
            rr, rc = nr + cur_dir[0], nc + cur_dir[1]
            relay = relays.get((rr, rc))
            if relay is None:
                return None
            incoming_side = (-cur_dir[0], -cur_dir[1])
            chips = relay["chips"]
            if chips.get(incoming_side) != cur_color:
                return None
            options = [(side, chip_color) for side, chip_color in chips.items() if side != incoming_side]
            if len(options) != 1:
                return None
            out_dir, out_color = options[0]
            if cur_dir[0] * out_dir[0] + cur_dir[1] * out_dir[1] != 0:
                return None
            state = (rr, rc, cur_dir[0], cur_dir[1], cur_color)
            if state in seen_states:
                return None
            seen_states.add(state)
            out_chip = (rr + out_dir[0], rc + out_dir[1])
            if not _in_bounds((h, w), out_chip[0], out_chip[1]):
                return None
            if int(grid[out_chip]) != int(out_color):
                return None
            pos = out_chip
            cur_dir = out_dir
            cur_color = int(out_color)
            continue

        return None

    return None


def _unique_launch_result(
    grid: np.ndarray,
    tower: dict,
    relays: dict[tuple[int, int], dict],
    receiver_by_border: dict[tuple[int, int], int],
) -> tuple[dict, int]:
    successes: list[dict] = []
    for start in tower["starts"]:
        result = _simulate_launch(
            grid,
            start=start,
            direction=tower["direction"],
            color=tower["color"],
            relays=relays,
            receiver_by_border=receiver_by_border,
        )
        if result is not None:
            successes.append(result)
    if len(successes) != 1:
        raise AssertionError("each tower must have exactly one valid launch cell")
    return successes[0], len(successes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge towers through relay stations and fill receiver interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    receivers = _detect_receivers(g)
    assert receivers, "expected at least one receiver"
    receiver_by_border: dict[tuple[int, int], int] = {}
    for idx, receiver in enumerate(receivers):
        for cell in receiver["frame"]:
            receiver_by_border[cell] = idx

    relays = _detect_relays(g)
    towers = _detect_towers(g)
    assert towers, "expected at least one tower"

    paint_map: dict[tuple[int, int], int] = {}
    receiver_fills: dict[int, int] = {}

    for tower in towers:
        result, _count = _unique_launch_result(g, tower, relays, receiver_by_border)
        for r, c, color in result["painted"]:
            prev = paint_map.get((r, c))
            if prev is not None and prev != color:
                raise AssertionError("signal trails must not conflict")
            paint_map[(r, c)] = color
        prev_fill = receiver_fills.get(result["receiver"])
        if prev_fill is not None and prev_fill != result["fill_color"]:
            raise AssertionError("receiver cannot receive two different colors")
        receiver_fills[result["receiver"]] = result["fill_color"]

    out = np.array(g, copy=True)
    for (r, c), color in paint_map.items():
        if out[r, c] != BG:
            raise AssertionError("trails may only paint background cells")
        out[r, c] = color
    for receiver_idx, color in receiver_fills.items():
        for r, c in receivers[receiver_idx]["interior"]:
            out[r, c] = color
    return out


def _choose_tower_anchor(side: str, shape: tuple[int, int], rng: random.Random) -> int:
    h, w = shape
    if side in ("top", "bottom"):
        return rng.randint(1, w - 3)
    return rng.randint(1, h - 3)


def _extend_path(
    shape: tuple[int, int],
    rng: random.Random,
    pos: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    relays_left: int,
    used: set[tuple[int, int]],
    relays: list[dict],
    segments: list[dict],
) -> dict | None:
    lengths = list(SEG_LEN_VALUES)
    rng.shuffle(lengths)

    if relays_left == 0:
        offsets = [0, 1, 2, 3]
        rng.shuffle(offsets)
        for length in lengths:
            trail = [_advance(pos, direction, step) for step in range(1, length + 1)]
            hit = _advance(pos, direction, length + 1)
            if not all(_in_bounds(shape, r, c) for r, c in trail + [hit]):
                continue
            if any(cell in used for cell in trail):
                continue
            for offset in offsets:
                top_left = _receiver_top_left_from_hit(hit, direction, offset)
                frame, _interior, full = _receiver_cells(top_left)
                if not all(_in_bounds(shape, r, c) for r, c in full):
                    continue
                if hit not in frame:
                    continue
                if any(cell in used for cell in full):
                    continue
                if any(cell in full for cell in trail):
                    continue
                return {
                    "segments": segments + [{"color": color, "cells": trail}],
                    "relays": list(relays),
                    "receiver_top_left": top_left,
                    "receiver_fill_color": color,
                }
        return None

    turn_dirs = list(PERPENDICULARS[direction])
    rng.shuffle(turn_dirs)
    next_colors = [c for c in USABLE_COLORS if c != color]
    rng.shuffle(next_colors)

    for length in lengths:
        trail = [_advance(pos, direction, step) for step in range(1, length + 1)]
        input_chip = _advance(pos, direction, length + 1)
        relay = _advance(pos, direction, length + 2)
        if not all(_in_bounds(shape, r, c) for r, c in trail + [input_chip, relay]):
            continue
        if any(cell in used for cell in trail + [input_chip, relay]):
            continue
        base_used = set(used)
        base_used.update(trail)
        base_used.add(input_chip)
        base_used.add(relay)
        for out_dir in turn_dirs:
            output_chip = _advance(relay, out_dir)
            if not _in_bounds(shape, output_chip[0], output_chip[1]):
                continue
            if output_chip in base_used:
                continue
            for next_color in next_colors:
                relay_info = {
                    "input_chip": input_chip,
                    "input_color": color,
                    "relay": relay,
                    "output_chip": output_chip,
                    "output_color": next_color,
                }
                result = _extend_path(
                    shape=shape,
                    rng=rng,
                    pos=output_chip,
                    direction=out_dir,
                    color=next_color,
                    relays_left=relays_left - 1,
                    used=base_used | {output_chip},
                    relays=relays + [relay_info],
                    segments=segments + [{"color": color, "cells": trail}],
                )
                if result is not None:
                    return result
    return None


def _plan_path(shape: tuple[int, int], side: str, tower_color: int, relay_count: int, rng: random.Random) -> dict | None:
    for _ in range(PATH_PLAN_RETRIES):
        anchor = _choose_tower_anchor(side, shape, rng)
        tower_cells, top_left, direction = _tower_cells(side, anchor, shape)
        lane = rng.randint(0, 1)
        start = _launch_cells_for_tower(side, top_left)[lane]
        used = set(tower_cells)
        result = _extend_path(
            shape=shape,
            rng=rng,
            pos=start,
            direction=direction,
            color=tower_color,
            relays_left=relay_count,
            used=used,
            relays=[],
            segments=[],
        )
        if result is None:
            continue
        result["tower_cells"] = list(tower_cells)
        result["tower_color"] = tower_color
        result["tower_side"] = side
        result["tower_top_left"] = top_left
        result["tower_start"] = start
        return result
    return None


def _place_extra_receivers(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int]] = []
    local_occupied = set(occupied)
    for _ in range(count):
        success = False
        for _ in range(EXTRA_RECEIVER_RETRIES):
            top_left = (rng.randint(1, h - 5), rng.randint(1, w - 5))
            _frame, _interior, full = _receiver_cells(top_left)
            if any(cell in local_occupied for cell in full):
                continue
            placed.append(top_left)
            local_occupied.update(full)
            success = True
            break
        if not success:
            return None
    return placed


def _count_valid_launches(grid: np.ndarray) -> list[int]:
    receivers = _detect_receivers(grid)
    relays = _detect_relays(grid)
    towers = _detect_towers(grid)
    receiver_by_border: dict[tuple[int, int], int] = {}
    for idx, receiver in enumerate(receivers):
        for cell in receiver["frame"]:
            receiver_by_border[cell] = idx

    counts: list[int] = []
    for tower in towers:
        successes = 0
        for start in tower["starts"]:
            result = _simulate_launch(
                grid,
                start=start,
                direction=tower["direction"],
                color=tower["color"],
                relays=relays,
                receiver_by_border=receiver_by_border,
            )
            if result is not None:
                successes += 1
        counts.append(successes)
    return counts


def _choose_relay_counts(path_count: int, rng: random.Random) -> list[int]:
    counts = [rng.choice(RELAY_COUNT_VALUES) for _ in range(path_count)]
    if rng.random() < 0.6 and max(counts) == 0:
        counts[rng.randrange(path_count)] = rng.choice((1, 2))
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    shape = (h, w)

    path_count = rng.randint(*PATH_COUNT_RANGE)
    sides = rng.sample(["top", "bottom", "left", "right"], path_count)
    tower_colors = rng.sample(list(USABLE_COLORS), path_count)
    relay_counts = _choose_relay_counts(path_count, rng)

    paths: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for side, tower_color, relay_count in zip(sides, tower_colors, relay_counts):
        placed = None
        for _ in range(PATH_PLAN_RETRIES):
            candidate = _plan_path(shape, side, tower_color, relay_count, rng)
            if candidate is None:
                continue
            footprint = _path_footprint(candidate)
            if footprint & occupied:
                continue
            placed = candidate
            occupied |= footprint
            paths.append(candidate)
            break
        if placed is None:
            return None

    extra_receiver_count = rng.randint(*EXTRA_RECEIVER_RANGE)
    extra_receivers = _place_extra_receivers(shape, occupied, extra_receiver_count, rng)
    if extra_receivers is None:
        return None

    inp = _render_input(shape, paths, extra_receivers)
    expected = _render_output(inp, paths)

    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None

    launch_counts = _count_valid_launches(inp)
    if not launch_counts or any(count != 1 for count in launch_counts):
        return None

    if np.array_equal(inp, expected):
        return None

    if len(_detect_towers(inp)) != path_count:
        return None
    if len(_detect_receivers(inp)) != path_count + extra_receiver_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

SIDE_TO_STEP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
STEP_TO_SIDE = {step: side for side, step in SIDE_TO_STEP.items()}
SIDES = ("up", "down", "left", "right")
COLOR_BY_HIT_COUNT = {
    0: GREEN,
    1: GREEN,
    2: BLUE,
    3: ORANGE,
    4: MAGENTA,
}

GEN_RETRIES = 400
ZONE_MARGIN = 1
TOWER_GAP = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _find_receivers(grid: np.ndarray) -> list[dict]:
    receivers: list[dict] = []
    for pts in _components(grid == GREEN):
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        assert len(pts) == 9, "each receiver must contain 9 pixels"
        assert r1 - r0 == 2 and c1 - c0 == 2, "each receiver must be 3x3"
        receivers.append(
            {
                "top": int(r0),
                "left": int(c0),
                "bottom": int(r1),
                "right": int(c1),
                "cells": {(int(r), int(c)) for r, c in pts},
            }
        )
    receivers.sort(key=lambda item: (int(item["top"]), int(item["left"])))
    return receivers


def _receiver_index(receivers: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    idx = np.full((h, w), -1, dtype=int)
    for rid, receiver in enumerate(receivers):
        for r, c in receiver["cells"]:
            idx[int(r), int(c)] = int(rid)
    return idx


def _trace_scene(
    grid: np.ndarray,
    towers: list[tuple[int, int]],
    receivers: list[dict],
) -> tuple[set[tuple[int, int]], list[set[str]]]:
    h, w = grid.shape
    receiver_at = _receiver_index(receivers, grid.shape)
    path_cells: set[tuple[int, int]] = set()
    hit_sides = [set() for _ in receivers]

    for tower_r, tower_c in towers:
        for dr, dc in SIDE_TO_STEP.values():
            r = int(tower_r)
            c = int(tower_c)
            ray_path: list[tuple[int, int]] = []
            while True:
                r += int(dr)
                c += int(dc)
                if not (0 <= r < h and 0 <= c < w):
                    break
                receiver_id = int(receiver_at[r, c])
                if receiver_id >= 0:
                    hit_sides[receiver_id].add(STEP_TO_SIDE[(int(dr), int(dc))])
                    path_cells.update(ray_path)
                    break
                if int(grid[r, c]) == BLACK:
                    ray_path.append((int(r), int(c)))

    return path_cells, hit_sides


def _first_hit_signature(
    cell: tuple[int, int],
    receiver_at: np.ndarray,
) -> list[tuple[int, str]]:
    h, w = receiver_at.shape
    hits: list[tuple[int, str]] = []
    r0, c0 = map(int, cell)
    for dr, dc in SIDE_TO_STEP.values():
        r = int(r0)
        c = int(c0)
        while True:
            r += int(dr)
            c += int(dc)
            if not (0 <= r < h and 0 <= c < w):
                break
            receiver_id = int(receiver_at[r, c])
            if receiver_id >= 0:
                hits.append((int(receiver_id), STEP_TO_SIDE[(int(dr), int(dc))]))
                break
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast orthogonal rays from red towers and recolor each receiver by how many sides are reached.

    Input structure:
    - The background is black.
    - Red single pixels are signal towers.
    - Green 3x3 solid squares are receivers, separated from each other and from towers.
    - Towers may share rows or columns with receivers, but are never inside a receiver.

    Rule:
    - Each tower emits one ray upward, downward, leftward, and rightward.
    - A ray succeeds only when the first receiver encountered on that line is reached; the receiver cell itself is not painted.
    - Paint every successful ray path yellow from the cell next to the tower up to the last black cell before the receiver.
    - For each receiver, count how many distinct receiver sides are hit, then recolor the full 3x3 block: 0 or 1 stays green, 2 becomes blue, 3 becomes orange, 4 becomes magenta.

    Key insight:
    - The task is a nearest-hit ray cast over 3x3 target blocks, followed by deduplicating hits by direction rather than by raw ray count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    receivers = _find_receivers(g)
    towers = [(int(r), int(c)) for r, c in np.argwhere(g == RED)]

    out = g.copy()
    path_cells, hit_sides = _trace_scene(g, towers, receivers)
    for r, c in path_cells:
        out[int(r), int(c)] = YELLOW
    for receiver, sides in zip(receivers, hit_sides):
        color = int(COLOR_BY_HIT_COUNT[len(sides)])
        for r, c in receiver["cells"]:
            out[int(r), int(c)] = color
    return out


def _receiver_expanded_cells(receiver: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(int(receiver["top"]) - 1, int(receiver["bottom"]) + 2):
        for c in range(int(receiver["left"]) - 1, int(receiver["right"]) + 2):
            cells.add((int(r), int(c)))
    return cells


def _receivers_non_adjacent(receivers: list[dict]) -> bool:
    expanded: list[set[tuple[int, int]]] = []
    for receiver in receivers:
        expanded.append(_receiver_expanded_cells(receiver))
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            if expanded[i] & receivers[j]["cells"]:
                return False
            if expanded[j] & receivers[i]["cells"]:
                return False
    return True


def _tower_clear_of_receivers(cell: tuple[int, int], receivers: list[dict], h: int, w: int) -> bool:
    r, c = map(int, cell)
    if not (0 <= r < h and 0 <= c < w):
        return False
    for receiver in receivers:
        if int(receiver["top"]) - 1 <= r <= int(receiver["bottom"]) + 1 and int(receiver["left"]) - 1 <= c <= int(receiver["right"]) + 1:
            return False
    return True


def _tower_far_from_other_towers(cell: tuple[int, int], towers: list[tuple[int, int]]) -> bool:
    r, c = map(int, cell)
    for tr, tc in towers:
        if max(abs(int(tr) - r), abs(int(tc) - c)) <= TOWER_GAP:
            return False
    return True


def _side_candidates(
    receiver: dict,
    side: str,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    top = int(receiver["top"])
    left = int(receiver["left"])
    bottom = int(receiver["bottom"])
    right = int(receiver["right"])
    cells: list[tuple[int, int]] = []
    if side == "up":
        for c in range(left, right + 1):
            for r in range(0, top - 1):
                cells.append((int(r), int(c)))
    elif side == "down":
        for c in range(left, right + 1):
            for r in range(bottom + 2, h):
                cells.append((int(r), int(c)))
    elif side == "left":
        for r in range(top, bottom + 1):
            for c in range(0, left - 1):
                cells.append((int(r), int(c)))
    else:
        assert side == "right"
        for r in range(top, bottom + 1):
            for c in range(right + 2, w):
                cells.append((int(r), int(c)))
    random.shuffle(cells)
    return cells


def _place_receivers(h: int, w: int, count: int) -> list[dict] | None:
    zone_rows = 2
    zone_cols = 2
    zones = [(zr, zc) for zr in range(zone_rows) for zc in range(zone_cols)]
    chosen = random.sample(zones, count)
    zone_h = h // zone_rows
    zone_w = w // zone_cols
    receivers: list[dict] = []

    for zr, zc in chosen:
        r_start = zr * zone_h
        r_end = h if zr == zone_rows - 1 else (zr + 1) * zone_h
        c_start = zc * zone_w
        c_end = w if zc == zone_cols - 1 else (zc + 1) * zone_w

        row_lo = max(2, r_start + ZONE_MARGIN)
        row_hi = min(h - 5, r_end - 3 - ZONE_MARGIN)
        col_lo = max(2, c_start + ZONE_MARGIN)
        col_hi = min(w - 5, c_end - 3 - ZONE_MARGIN)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(50):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            receiver = {
                "top": int(top),
                "left": int(left),
                "bottom": int(top + 2),
                "right": int(left + 2),
                "cells": {(int(r), int(c)) for r in range(top, top + 3) for c in range(left, left + 3)},
            }
            trial = receivers + [receiver]
            if _receivers_non_adjacent(trial):
                receivers.append(receiver)
                placed = True
                break
        if not placed:
            return None
    return receivers


def _build_candidate_map(receivers: list[dict], h: int, w: int) -> dict[tuple[int, str], list[tuple[int, int]]]:
    receiver_at = _receiver_index(receivers, (h, w))
    candidates: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for rid, receiver in enumerate(receivers):
        for side in SIDES:
            clean: list[tuple[int, int]] = []
            for cell in _side_candidates(receiver, side, h, w):
                if not _tower_clear_of_receivers(cell, receivers, h, w):
                    continue
                hits = _first_hit_signature(cell, receiver_at)
                if len(hits) == 1 and hits[0] == (int(rid), side):
                    clean.append((int(cell[0]), int(cell[1])))
            candidates[(int(rid), side)] = clean
    return candidates


def _sample_target_sides(
    receivers: list[dict],
    candidates: dict[tuple[int, str], list[tuple[int, int]]],
) -> list[set[str]] | None:
    target_counts = random.sample([0, 1, 2, 3, 4], len(receivers))
    side_sets: list[set[str]] = []
    for rid, target_count in enumerate(target_counts):
        available = [side for side in SIDES if candidates[(int(rid), side)]]
        if len(available) < target_count:
            return None
        chosen = set(random.sample(available, target_count))
        side_sets.append(chosen)
    return side_sets


def _place_required_towers(
    side_sets: list[set[str]],
    candidates: dict[tuple[int, str], list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    requirements = [(rid, side) for rid, sides in enumerate(side_sets) for side in sides]
    random.shuffle(requirements)
    requirements.sort(key=lambda item: len(candidates[(int(item[0]), item[1])]))

    towers: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for rid, side in requirements:
        options = candidates[(int(rid), side)].copy()
        random.shuffle(options)
        picked = None
        for cell in options:
            if cell in used:
                continue
            if not _tower_far_from_other_towers(cell, towers):
                continue
            picked = cell
            break
        if picked is None:
            return None
        towers.append((int(picked[0]), int(picked[1])))
        used.add((int(picked[0]), int(picked[1])))
    return towers


def _add_duplicate_towers(
    towers: list[tuple[int, int]],
    side_sets: list[set[str]],
    candidates: dict[tuple[int, str], list[tuple[int, int]]],
) -> None:
    duplicate_slots = random.randint(0, 2)
    used = set(towers)
    eligible = [(rid, side) for rid, sides in enumerate(side_sets) for side in sides if len(candidates[(rid, side)]) >= 2]
    random.shuffle(eligible)
    for rid, side in eligible:
        if duplicate_slots <= 0:
            break
        options = candidates[(int(rid), side)].copy()
        random.shuffle(options)
        for cell in options:
            if cell in used:
                continue
            if not _tower_far_from_other_towers(cell, towers):
                continue
            towers.append((int(cell[0]), int(cell[1])))
            used.add((int(cell[0]), int(cell[1])))
            duplicate_slots -= 1
            break


def _add_miss_towers(
    towers: list[tuple[int, int]],
    receivers: list[dict],
    h: int,
    w: int,
) -> None:
    receiver_at = _receiver_index(receivers, (h, w))
    miss_target = random.randint(0, 3)
    used = set(towers)
    cells = [(int(r), int(c)) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for cell in cells:
        if miss_target <= 0:
            break
        if cell in used:
            continue
        if not _tower_clear_of_receivers(cell, receivers, h, w):
            continue
        if not _tower_far_from_other_towers(cell, towers):
            continue
        if _first_hit_signature(cell, receiver_at):
            continue
        towers.append((int(cell[0]), int(cell[1])))
        used.add((int(cell[0]), int(cell[1])))
        miss_target -= 1


def _receiver_side_sets_from_scene(
    h: int,
    w: int,
    receivers: list[dict],
    towers: list[tuple[int, int]],
) -> list[set[str]]:
    grid = np.full((h, w), BLACK, dtype=int)
    for receiver in receivers:
        for r, c in receiver["cells"]:
            grid[int(r), int(c)] = GREEN
    for r, c in towers:
        grid[int(r), int(c)] = RED
    _path_cells, side_sets = _trace_scene(grid, towers, receivers)
    return side_sets


def _attempt_generate() -> np.ndarray | None:
    receiver_count = random.randint(2, 4)
    min_size = 16 + 3 * receiver_count
    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)

    receivers = _place_receivers(h, w, receiver_count)
    if receivers is None:
        return None

    candidates = _build_candidate_map(receivers, h, w)
    target_sides = _sample_target_sides(receivers, candidates)
    if target_sides is None:
        return None

    towers = _place_required_towers(target_sides, candidates)
    if towers is None:
        return None

    _add_duplicate_towers(towers, target_sides, candidates)
    _add_miss_towers(towers, receivers, h, w)

    actual_sides = _receiver_side_sets_from_scene(h, w, receivers, towers)
    if any(actual != target for actual, target in zip(actual_sides, target_sides)):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for receiver in receivers:
        for r, c in receiver["cells"]:
            grid[int(r), int(c)] = GREEN
    for r, c in towers:
        if int(grid[r, c]) != BLACK:
            return None
        grid[int(r), int(c)] = RED

    # Ensure the generator actually exercises more than one recolor branch regularly.
    distinct_counts = {len(sides) for sides in target_sides}
    if len(distinct_counts) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

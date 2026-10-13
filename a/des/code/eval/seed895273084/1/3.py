from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400
LAYOUT_RETRIES = 80
CENTER_GAP = 4


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: dict[int, list[tuple[int, int]]] = {}
    next_label = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            components[next_label] = coords
            next_label += 1
    return labels, components


def _find_towers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    towers: list[dict[str, object]] = []
    for r_raw, c_raw in np.argwhere((grid != BG) & (grid != GREY)):
        r = int(r_raw)
        c = int(c_raw)
        arms = [(r + dr, c + dc) for dr, dc in DIRS4]
        assert all(0 <= rr < h and 0 <= cc < w for rr, cc in arms), "tower center too close to edge"
        assert all(int(grid[rr, cc]) == GREY for rr, cc in arms), "colored cell must be a tower center"
        towers.append(
            {
                "center": (r, c),
                "color": int(grid[r, c]),
                "cells": {(r, c), *arms},
            }
        )
    assert towers, "expected at least one tower"
    return towers


def _compute_reach(
    grid: np.ndarray, towers: list[dict[str, object]]
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], list[dict[int, set[int]]]]:
    labels, components = _label_components(grid == BG)
    h, w = grid.shape
    tower_reach: list[dict[int, set[int]]] = []
    for tower in towers:
        cr, cc = tower["center"]  # type: ignore[index]
        by_component: dict[int, set[int]] = {}
        for arm_idx, (dr, dc) in enumerate(DIRS4):
            r = cr + 2 * dr
            c = cc + 2 * dc
            while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
                label = int(labels[r, c])
                if label != 0:
                    by_component.setdefault(label, set()).add(arm_idx)
                r += dr
                c += dc
        tower_reach.append(by_component)
    return labels, components, tower_reach


def _component_summaries(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    towers = _find_towers(grid)
    _labels, components, tower_reach = _compute_reach(grid, towers)
    summaries: list[dict[str, object]] = []
    for label, coords in components.items():
        counts: dict[int, int] = {}
        for tower_idx, by_component in enumerate(tower_reach):
            count = len(by_component.get(label, ()))
            if count > 0:
                counts[tower_idx] = count
        if counts:
            best = max(counts.values())
            winners = [tower_idx for tower_idx, count in counts.items() if count == best]
            kind = "tie" if len(winners) > 1 else "unique"
        else:
            best = 0
            winners = []
            kind = "untouched"
        summaries.append(
            {
                "label": label,
                "size": len(coords),
                "kind": kind,
                "best": best,
                "winners": winners,
                "counts": counts,
            }
        )
    return towers, summaries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each empty chamber by the tower whose rays enter it through the most arms.

    Input structure:
    - Black background.
    - Grey solid wall rectangles that block sight.
    - Colored tower centers surrounded by four grey arm pixels in a cross shape.

    Rule:
    - From each tower arm tip, cast a straight ray outward until it hits a wall,
      another tower cell, or the grid edge.
    - Every background chamber touched by that ray is reachable from that arm.
    - For each background chamber, count how many distinct arms of each tower can
      reach it. The tower with the largest count paints that chamber with its
      center color; equal best counts leave the chamber black.
    - Walls and towers remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    towers = _find_towers(g)
    _labels, components, tower_reach = _compute_reach(g, towers)

    out = g.copy()
    for label, coords in components.items():
        best_count = 0
        winner_color = BG
        tied = False
        for tower, by_component in zip(towers, tower_reach):
            count = len(by_component.get(label, ()))
            if count == 0:
                continue
            if count > best_count:
                best_count = count
                winner_color = int(tower["color"])
                tied = False
            elif count == best_count:
                tied = True
        if best_count > 0 and not tied:
            for r, c in coords:
                out[r, c] = winner_color
    return out


def _build_layout(rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, object]]]:
    for _ in range(LAYOUT_RETRIES):
        orientation = str(rng.choice(["vertical", "horizontal"]))
        n_rooms = int(rng.integers(2, 4))
        wall_sizes = [int(rng.integers(1, 3)) for _ in range(n_rooms - 1)]

        if orientation == "vertical":
            room_sizes = [int(rng.integers(5, 9)) for _ in range(n_rooms)]
            width = sum(room_sizes) + sum(wall_sizes)
            if not 14 <= width <= 22:
                continue
            height = int(rng.integers(10, 19))
            grid = np.full((height, width), BG, dtype=int)
            rooms: list[dict[str, object]] = []
            x = 0
            for i, size in enumerate(room_sizes):
                c0 = x
                c1 = x + size - 1
                rooms.append(
                    {
                        "r0": 0,
                        "r1": height - 1,
                        "c0": c0,
                        "c1": c1,
                        "touch_top": True,
                        "touch_bottom": True,
                        "touch_left": c0 == 0,
                        "touch_right": c1 == width - 1,
                    }
                )
                x = c1 + 1
                if i < len(wall_sizes):
                    grid[:, x : x + wall_sizes[i]] = GREY
                    x += wall_sizes[i]
            return grid, rooms

        room_sizes = [int(rng.integers(5, 9)) for _ in range(n_rooms)]
        height = sum(room_sizes) + sum(wall_sizes)
        if not 12 <= height <= 22:
            continue
        width = int(rng.integers(14, 23))
        grid = np.full((height, width), BG, dtype=int)
        rooms = []
        y = 0
        for i, size in enumerate(room_sizes):
            r0 = y
            r1 = y + size - 1
            rooms.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": 0,
                    "c1": width - 1,
                    "touch_top": r0 == 0,
                    "touch_bottom": r1 == height - 1,
                    "touch_left": True,
                    "touch_right": True,
                }
            )
            y = r1 + 1
            if i < len(wall_sizes):
                grid[y : y + wall_sizes[i], :] = GREY
                y += wall_sizes[i]
        return grid, rooms

    raise RuntimeError("failed to build a valid striped layout")


def _room_area(room: dict[str, object]) -> int:
    return (int(room["r1"]) - int(room["r0"]) + 1) * (int(room["c1"]) - int(room["c0"]) + 1)


def _add_blocks(grid: np.ndarray, rooms: list[dict[str, object]], rng: np.random.Generator) -> None:
    h, w = grid.shape
    for _ in range(int(rng.integers(0, 3))):
        room = rooms[int(rng.integers(len(rooms)))]
        room_h = int(room["r1"]) - int(room["r0"]) + 1
        room_w = int(room["c1"]) - int(room["c0"]) + 1
        if room_h < 6 or room_w < 6:
            continue
        max_bh = min(4, room_h - 2)
        max_bw = min(4, room_w - 2)
        if max_bh < 2 or max_bw < 2:
            continue
        bh = int(rng.integers(2, max_bh + 1))
        bw = int(rng.integers(2, max_bw + 1))
        r0_min = int(room["r0"]) + 1
        r0_max = int(room["r1"]) - bh
        c0_min = int(room["c0"]) + 1
        c0_max = int(room["c1"]) - bw
        if r0_min > r0_max or c0_min > c0_max:
            continue
        br = int(rng.integers(r0_min, r0_max + 1))
        bc = int(rng.integers(c0_min, c0_max + 1))
        rr0 = max(0, br - 1)
        rr1 = min(h, br + bh + 1)
        cc0 = max(0, bc - 1)
        cc1 = min(w, bc + bw + 1)
        if np.any(grid[rr0:rr1, cc0:cc1] != BG):
            continue
        grid[br : br + bh, bc : bc + bw] = GREY


def _center_bounds(room: dict[str, object], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    min_r = 1 if room["touch_top"] else int(room["r0"]) + 2
    max_r = h - 2 if room["touch_bottom"] else int(room["r1"]) - 2
    min_c = 1 if room["touch_left"] else int(room["c0"]) + 2
    max_c = w - 2 if room["touch_right"] else int(room["c1"]) - 2
    return min_r, max_r, min_c, max_c


def _border_arm_count(room: dict[str, object], shape: tuple[int, int], r: int, c: int) -> int:
    h, w = shape
    count = 4
    if room["touch_top"] and r == 1:
        count -= 1
    if room["touch_bottom"] and r == h - 2:
        count -= 1
    if room["touch_left"] and c == 1:
        count -= 1
    if room["touch_right"] and c == w - 2:
        count -= 1
    return count


def _candidate_centers(
    grid: np.ndarray,
    room: dict[str, object],
    centers: list[tuple[int, int]],
) -> dict[int, list[tuple[int, int]]]:
    min_r, max_r, min_c, max_c = _center_bounds(room, grid.shape)
    by_count: dict[int, list[tuple[int, int]]] = {}
    if min_r > max_r or min_c > max_c:
        return by_count
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if not np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == BG):
                continue
            if any(max(abs(r - rr), abs(c - cc)) < CENTER_GAP for rr, cc in centers):
                continue
            count = _border_arm_count(room, grid.shape, r, c)
            by_count.setdefault(count, []).append((r, c))
    return by_count


def _draw_tower(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    grid[r, c] = color
    for dr, dc in DIRS4:
        grid[r + dr, c + dc] = GREY


def _place_tower(
    grid: np.ndarray,
    room: dict[str, object],
    centers: list[tuple[int, int]],
    color: int,
    prefer_counts: list[int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    by_count = _candidate_centers(grid, room, centers)
    for want in prefer_counts:
        candidates = by_count.get(want, [])
        if candidates:
            idx = int(rng.integers(len(candidates)))
            center = candidates[idx]
            _draw_tower(grid, center, color)
            centers.append(center)
            return center
    all_candidates = [center for centers_for_count in by_count.values() for center in centers_for_count]
    if not all_candidates:
        return None
    center = all_candidates[int(rng.integers(len(all_candidates)))]
    _draw_tower(grid, center, color)
    centers.append(center)
    return center


def _seed_towers(
    grid: np.ndarray,
    rooms: list[dict[str, object]],
    mode: str,
    rng: np.random.Generator,
) -> bool:
    room_ids = sorted(range(len(rooms)), key=lambda idx: _room_area(rooms[idx]), reverse=True)
    colors = list(TOWER_COLORS)
    rng.shuffle(colors)
    color_idx = 0
    centers: list[tuple[int, int]] = []
    used_rooms: set[int] = set()

    def add(room_idx: int, prefer_counts: list[int]) -> bool:
        nonlocal color_idx
        if color_idx >= len(colors):
            return False
        center = _place_tower(grid, rooms[room_idx], centers, colors[color_idx], prefer_counts, rng)
        if center is None:
            return False
        color_idx += 1
        used_rooms.add(room_idx)
        return True

    if mode == "with_tie":
        tie_room = room_ids[0]
        winner_room = room_ids[1] if len(room_ids) > 1 else room_ids[0]
        if not add(tie_room, [4, 3]):
            return False
        if not add(tie_room, [4, 3]):
            return False
        if not add(winner_room, [4, 3, 2]):
            return False
    elif mode == "with_untouched":
        active_room = room_ids[0]
        if not add(active_room, [4, 3, 2]):
            return False
        if not add(active_room, [3, 2, 4]):
            fallback_room = room_ids[1] if len(room_ids) > 1 else active_room
            if not add(fallback_room, [4, 3, 2]):
                return False
    else:
        primary_room = room_ids[0]
        if not add(primary_room, [4, 3, 2]):
            return False
        if not add(primary_room, [3, 2, 4]):
            secondary_room = room_ids[1] if len(room_ids) > 1 else primary_room
            if not add(secondary_room, [4, 3, 2]):
                return False

    extra_rooms = sorted(used_rooms) if mode == "with_untouched" else room_ids
    for _ in range(int(rng.integers(0, 3))):
        room_idx = int(rng.choice(extra_rooms))
        if not add(room_idx, [4, 3, 2]):
            break
    return True


def _valid_sample(grid: np.ndarray, mode: str) -> bool:
    try:
        towers, summaries = _component_summaries(grid)
        out = main(grid)
    except Exception:
        return False

    if len(towers) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out != grid)) < 8:
        return False

    unique_count = sum(summary["kind"] == "unique" for summary in summaries)
    tie_count = sum(summary["kind"] == "tie" for summary in summaries)
    untouched_count = sum(summary["kind"] == "untouched" for summary in summaries)
    best_arm_count = max((int(summary["best"]) for summary in summaries), default=0)

    if best_arm_count < 2:
        return False
    if mode == "with_tie" and not (unique_count >= 1 and tie_count >= 1):
        return False
    if mode == "with_untouched" and not (unique_count >= 1 and untouched_count >= 1):
        return False
    if mode == "unique_only" and unique_count < 1:
        return False

    output_colors = set(map(int, np.unique(out))) - {BG, GREY}
    if not output_colors:
        return False
    return True


def _gen_attempt(rng: np.random.Generator, mode: str) -> np.ndarray | None:
    grid, rooms = _build_layout(rng)
    _add_blocks(grid, rooms, rng)
    if not _seed_towers(grid, rooms, mode, rng):
        return None
    if not _valid_sample(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    modes = ("unique_only", "with_tie", "with_untouched")
    for _ in range(GEN_RETRIES):
        mode = str(rng.choice(modes))
        grid = _gen_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower-ray input")

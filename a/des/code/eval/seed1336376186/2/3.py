from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREEN = 3
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DELTA_BY_DIRECTION = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERNS = {
    "up": {
        "cells": {(0, 1), (1, 0), (1, 2)},
        "tip": (0, 1),
    },
    "down": {
        "cells": {(0, 0), (0, 2), (1, 1)},
        "tip": (1, 1),
    },
    "left": {
        "cells": {(0, 1), (1, 0), (2, 1)},
        "tip": (1, 0),
    },
    "right": {
        "cells": {(0, 0), (1, 1), (2, 0)},
        "tip": (1, 1),
    },
}
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 25)
MAX_MOVERS = 3
MAX_EXTRA_STATIONARY = 2
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_objects(grid: np.ndarray, bg: int, arrow_color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, arrow_color) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return objects


def _count_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True

    empty = ~mask
    seen = np.zeros_like(empty, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(empty[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(empty[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if not bool(empty[r, c]) or bool(seen[r, c]):
                continue
            holes += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _detect_arrows(grid: np.ndarray, bg: int = BG) -> tuple[int, str, list[dict]]:
    h, w = grid.shape
    border_colors: list[int] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color != bg:
                border_colors.append(color)
    if not border_colors:
        raise ValueError("no edge markers found")

    arrow_color = int(Counter(border_colors).most_common(1)[0][0])
    arrow_mask = grid == arrow_color
    comps = _components_from_mask(arrow_mask, connectivity=8)
    arrows: list[dict] = []

    for comp in comps:
        if int(comp.shape[0]) != 3:
            raise ValueError("arrow component must contain exactly three pixels")
        r0, c0, r1, c1 = _bbox(comp)
        norm = {(int(r - r0), int(c - c0)) for r, c in comp}
        matched = None
        for direction, spec in ARROW_PATTERNS.items():
            if norm == spec["cells"]:
                tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
                matched = {
                    "direction": direction,
                    "tip": tip,
                    "cells": {(int(r), int(c)) for r, c in comp},
                }
                break
        if matched is None:
            raise ValueError("unrecognized arrow geometry")

        tr, tc = matched["tip"]
        direction = matched["direction"]
        if direction == "down" and not (r0 == 0 and tr == 1):
            raise ValueError("down arrow must sit on top edge")
        if direction == "up" and not (r1 == h - 1 and tr == h - 2):
            raise ValueError("up arrow must sit on bottom edge")
        if direction == "right" and not (c0 == 0 and tc == 1):
            raise ValueError("right arrow must sit on left edge")
        if direction == "left" and not (c1 == w - 1 and tc == w - 2):
            raise ValueError("left arrow must sit on right edge")
        arrows.append(matched)

    directions = {arrow["direction"] for arrow in arrows}
    if len(directions) != 1:
        raise ValueError("all arrows must point in the same direction")
    direction = next(iter(directions))
    if direction in ("up", "down"):
        arrows.sort(key=lambda arrow: arrow["tip"][1])
    else:
        arrows.sort(key=lambda arrow: arrow["tip"][0])
    return arrow_color, direction, arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each object indicated by an edge arrow.

    The common arrow direction gives the push direction. Each pointed object moves
    by holes + 1 cells along that direction. Unpointed objects remain in place,
    and moved objects overwrite stationary ones if they overlap after translation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = BG

    arrow_color, direction, arrows = _detect_arrows(g, bg=bg)
    objects = _find_objects(g, bg=bg, arrow_color=arrow_color)
    if not objects:
        return out

    cell_to_obj: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        obj["holes"] = _count_holes(obj["coords"])
        for r, c in obj["coords"]:
            cell_to_obj[(int(r), int(c))] = idx

    dr, dc = DELTA_BY_DIRECTION[direction]
    moved_ids: set[int] = set()

    for arrow in arrows:
        rr = arrow["tip"][0] + dr
        cc = arrow["tip"][1] + dc
        target = None
        while _in_bounds(g.shape, rr, cc):
            color = int(g[rr, cc])
            if color not in (bg, arrow_color):
                target = cell_to_obj[(rr, cc)]
                break
            rr += dr
            cc += dc
        if target is None:
            raise ValueError("each arrow must point to exactly one object")
        moved_ids.add(target)

    for idx in moved_ids:
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for idx in moved_ids:
        obj = objects[idx]
        dist = int(obj["holes"]) + 1
        shifted = obj["coords"] + np.array([dr * dist, dc * dist], dtype=int)
        if not bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < g.shape[0])
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < g.shape[1])
            )
        ):
            raise ValueError("translated object went out of bounds")
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = set()
    start = tuple(int(v) for v in pts[0])
    q = deque([start])
    seen.add(start)
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _make_holey_shape(
    rng: random.Random,
    hole_count: int,
    max_h: int,
    max_w: int,
) -> np.ndarray | None:
    min_h = 3
    min_w = 3
    if hole_count >= 1:
        min_h = max(min_h, 4)
        min_w = max(min_w, 4)
    if hole_count >= 2:
        min_h = max(min_h, 5)
        min_w = max(min_w, 5)
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(120):
        h = rng.randint(min_h, max_h)
        w = rng.randint(min_w, max_w)
        mask = np.ones((h, w), dtype=bool)
        holes: list[tuple[int, int, int, int]] = []
        ok = True
        for _ in range(hole_count):
            placed = False
            for _ in range(80):
                hh = rng.randint(1, min(2, h - 2))
                ww = rng.randint(1, min(2, w - 2))
                r0 = rng.randint(1, h - hh - 1)
                c0 = rng.randint(1, w - ww - 1)
                r1 = r0 + hh - 1
                c1 = c0 + ww - 1
                separated = True
                for hr0, hc0, hr1, hc1 in holes:
                    if not (r1 < hr0 - 1 or r0 > hr1 + 1 or c1 < hc0 - 1 or c0 > hc1 + 1):
                        separated = False
                        break
                if not separated:
                    continue
                holes.append((r0, c0, r1, c1))
                mask[r0 : r1 + 1, c0 : c1 + 1] = False
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok or not _mask_is_connected(mask):
            continue
        coords = _shape_from_mask(mask)
        if _count_holes(coords) != hole_count:
            continue
        return coords
    return None


def _cells_from_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _placement_clear(
    cells: set[tuple[int, int]],
    existing: list[set[tuple[int, int]]],
    ignore_touch_index: int | None = None,
) -> bool:
    halo = _expand8(cells)
    for idx, other in enumerate(existing):
        if cells & other:
            return False
        if idx == ignore_touch_index:
            continue
        if halo & other:
            return False
    return True


def _sample_lanes(length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    usable = length - 2
    if usable < 5:
        return None
    for _ in range(80):
        max_count = min(MAX_MOVERS, max(1, usable // 6))
        count = rng.randint(1, max_count)
        lane_size = rng.randint(5, min(7, usable - (count - 1) * 2))
        gap = rng.randint(2, 3)
        total = count * lane_size + (count - 1) * gap
        if total > usable:
            continue
        start = rng.randint(1, length - 1 - total)
        lanes = []
        cur = start
        for _ in range(count):
            lanes.append((cur, cur + lane_size - 1))
            cur += lane_size + gap
        return lanes
    return None


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _place_moving_object(
    rng: random.Random,
    direction: str,
    lane: tuple[int, int],
    shape_limit: int,
    existing_current: list[set[tuple[int, int]]],
    existing_final: list[set[tuple[int, int]]],
    color: int,
    grid_shape: tuple[int, int],
) -> dict | None:
    h_total, w_total = grid_shape
    dr, dc = DELTA_BY_DIRECTION[direction]
    vertical = direction in ("up", "down")
    lane_size = lane[1] - lane[0] + 1

    for _ in range(120):
        holes = rng.randint(0, 3)
        dist = holes + 1
        if vertical:
            max_h = min(8, h_total - dist - 4)
            max_w = lane_size
        else:
            max_h = lane_size
            max_w = min(8, w_total - dist - 4)
        shape = _make_holey_shape(rng, holes, max_h=max_h, max_w=max_w)
        if shape is None:
            continue

        r0, c0, r1, c1 = _bbox(shape)
        local_h = r1 - r0 + 1
        local_w = c1 - c0 + 1
        shape = shape - np.array([r0, c0], dtype=int)

        if vertical:
            left_min = lane[0]
            left_max = lane[1] - local_w + 1
            if left_max < left_min:
                continue
            if direction == "down":
                top_min = 3
                top_max = h_total - local_h - dist - 1
            else:
                top_min = dist + 1
                top_max = h_total - local_h - 3
            if top_max < top_min:
                continue
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)
        else:
            top_min = lane[0]
            top_max = lane[1] - local_h + 1
            if top_max < top_min:
                continue
            if direction == "right":
                left_min = 3
                left_max = w_total - local_w - dist - 1
            else:
                left_min = dist + 1
                left_max = w_total - local_w - 3
            if left_max < left_min:
                continue
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)

        coords = shape + np.array([top, left], dtype=int)
        cells = _cells_from_coords(coords)
        if not _placement_clear(cells, existing_current):
            continue
        shifted = _shift_coords(coords, dr * dist, dc * dist)
        if not bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < h_total)
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < w_total)
            )
        ):
            continue
        shifted_cells = _cells_from_coords(shifted)
        if any(shifted_cells & other for other in existing_final):
            continue

        if vertical:
            line = rng.choice(sorted({int(c) for _r, c in coords.tolist()}))
        else:
            line = rng.choice(sorted({int(r) for r, _c in coords.tolist()}))

        return {
            "kind": "moving",
            "color": color,
            "coords": coords,
            "final_coords": shifted,
            "holes": holes,
            "dist": dist,
            "line": line,
        }
    return None


def _connected_subset_for_overlap(
    candidate_cells: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    if not candidate_cells:
        return None
    points = np.array(sorted(candidate_cells), dtype=int)
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in candidate_cells:
        mask[r - r0, c - c0] = True
    comps = _components_from_mask(mask, connectivity=4)
    comps.sort(key=lambda comp: -len(comp))
    for comp in comps:
        cells = [(int(r + r0), int(c + c0)) for r, c in comp]
        rng.shuffle(cells)
        for r, c in cells:
            for dr, dc in DIRS4:
                nbr = (r + dr, c + dc)
                if nbr in candidate_cells:
                    return np.array([(r, c), nbr], dtype=int)
        return np.array([cells[0]], dtype=int)
    return None


def _place_overlap_stationary(
    rng: random.Random,
    direction: str,
    moving: list[dict],
    existing_current: list[set[tuple[int, int]]],
    color: int,
) -> dict | None:
    order = list(range(len(moving)))
    rng.shuffle(order)
    for idx in order:
        obj = moving[idx]
        current = _cells_from_coords(obj["coords"])
        final = _cells_from_coords(obj["final_coords"])
        candidate = final - current
        if direction in ("up", "down"):
            candidate = {cell for cell in candidate if cell[1] != int(obj["line"])}
        else:
            candidate = {cell for cell in candidate if cell[0] != int(obj["line"])}
        shape = _connected_subset_for_overlap(candidate, rng)
        if shape is None:
            continue
        cells = _cells_from_coords(shape)
        if not _placement_clear(cells, existing_current, ignore_touch_index=idx):
            continue
        return {
            "kind": "stationary",
            "color": color,
            "coords": shape,
        }
    return None


def _place_extra_stationary(
    rng: random.Random,
    direction: str,
    reserved_lines: set[int],
    existing_current: list[set[tuple[int, int]]],
    union_moving_final: set[tuple[int, int]],
    color: int,
    grid_shape: tuple[int, int],
) -> dict | None:
    h_total, w_total = grid_shape
    for _ in range(100):
        holes = rng.choice([0, 0, 1, 1, 2])
        shape = _make_holey_shape(rng, holes, max_h=6, max_w=6)
        if shape is None:
            continue
        r0, c0, r1, c1 = _bbox(shape)
        shape = shape - np.array([r0, c0], dtype=int)
        local_h = int(shape[:, 0].max()) + 1
        local_w = int(shape[:, 1].max()) + 1
        top = rng.randint(2, h_total - local_h - 2)
        left = rng.randint(2, w_total - local_w - 2)
        coords = shape + np.array([top, left], dtype=int)
        cells = _cells_from_coords(coords)
        if direction in ("up", "down"):
            if any(c in reserved_lines for _r, c in cells):
                continue
        else:
            if any(r in reserved_lines for r, _c in cells):
                continue
        if cells & union_moving_final:
            continue
        if not _placement_clear(cells, existing_current):
            continue
        return {
            "kind": "stationary",
            "color": color,
            "coords": coords,
        }
    return None


def _arrow_cells(direction: str, line: int, grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    if direction == "down":
        return np.array([(0, line - 1), (0, line + 1), (1, line)], dtype=int)
    if direction == "up":
        return np.array([(h - 1, line - 1), (h - 1, line + 1), (h - 2, line)], dtype=int)
    if direction == "right":
        return np.array([(line - 1, 0), (line + 1, 0), (line, 1)], dtype=int)
    return np.array([(line - 1, w - 1), (line + 1, w - 1), (line, w - 2)], dtype=int)


def _render_input(grid_shape: tuple[int, int], direction: str, arrows: list[int], objects: list[dict]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for line in arrows:
        coords = _arrow_cells(direction, line, grid_shape)
        _paint_cells(grid, coords, GREEN)
    for obj in objects:
        _paint_cells(grid, obj["coords"], obj["color"])
    return grid


def _render_expected(inp: np.ndarray, moving: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in moving:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for obj in moving:
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = obj["color"]
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    direction = rng.choice(["up", "down", "left", "right"])
    vertical = direction in ("up", "down")
    lanes = _sample_lanes(w if vertical else h, rng)
    if not lanes:
        return None

    needed_colors = len(lanes) + MAX_EXTRA_STATIONARY + 1
    if needed_colors > len(PALETTE):
        return None
    colors = rng.sample(PALETTE, needed_colors)
    color_idx = 0

    moving: list[dict] = []
    stationary: list[dict] = []
    existing_current: list[set[tuple[int, int]]] = []
    existing_final: list[set[tuple[int, int]]] = []

    for lane in lanes:
        obj = _place_moving_object(
            rng=rng,
            direction=direction,
            lane=lane,
            shape_limit=8,
            existing_current=existing_current,
            existing_final=existing_final,
            color=colors[color_idx],
            grid_shape=(h, w),
        )
        color_idx += 1
        if obj is None:
            return None
        moving.append(obj)
        existing_current.append(_cells_from_coords(obj["coords"]))
        existing_final.append(_cells_from_coords(obj["final_coords"]))

    reserved_lines = {int(obj["line"]) for obj in moving}
    union_moving_final = set().union(*existing_final) if existing_final else set()

    if rng.random() < 0.3:
        overlap_obj = _place_overlap_stationary(
            rng=rng,
            direction=direction,
            moving=moving,
            existing_current=existing_current,
            color=colors[color_idx],
        )
        color_idx += 1
        if overlap_obj is not None:
            stationary.append(overlap_obj)
            existing_current.append(_cells_from_coords(overlap_obj["coords"]))

    extra_target = rng.randint(0, MAX_EXTRA_STATIONARY)
    if not stationary and extra_target == 0 and rng.random() < 0.6:
        extra_target = 1
    for _ in range(extra_target):
        if color_idx >= len(colors):
            break
        obj = _place_extra_stationary(
            rng=rng,
            direction=direction,
            reserved_lines=reserved_lines,
            existing_current=existing_current,
            union_moving_final=union_moving_final,
            color=colors[color_idx],
            grid_shape=(h, w),
        )
        color_idx += 1
        if obj is None:
            continue
        stationary.append(obj)
        existing_current.append(_cells_from_coords(obj["coords"]))

    all_objects = moving + stationary
    arrows = [int(obj["line"]) for obj in moving]
    inp = _render_input((h, w), direction, arrows, all_objects)
    expected = _render_expected(inp, moving)

    try:
        arrow_color, parsed_direction, parsed_arrows = _detect_arrows(inp, bg=BG)
    except ValueError:
        return None
    if arrow_color != GREEN or parsed_direction != direction or len(parsed_arrows) != len(moving):
        return None

    parsed_objects = _find_objects(inp, bg=BG, arrow_color=GREEN)
    if len(parsed_objects) != len(all_objects):
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError("generate_input failed to produce a valid input")

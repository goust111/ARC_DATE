from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
WALL_COUNT_CHOICES = (2, 3, 4)
WALL_LEN_CHOICES = (4, 6, 8)
SOURCE_COUNT_MAX = 4
MIN_RAY_PIXELS = 4
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _on_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _edge_direction(r: int, c: int, h: int, w: int) -> tuple[int, int] | None:
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append(DOWN)
    if r == h - 1:
        dirs.append(UP)
    if c == 0:
        dirs.append(RIGHT)
    if c == w - 1:
        dirs.append(LEFT)
    if len(dirs) != 1:
        return None
    return dirs[0]


def _build_wall_index(shape: tuple[int, int], walls: list[dict]) -> np.ndarray:
    h, w = shape
    wall_index = np.full((h, w), -1, dtype=int)
    for idx, wall in enumerate(walls):
        for r, c in wall["cells"]:
            wall_index[r, c] = idx
    return wall_index


def _order_label_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = cells
    if r0 == r1:
        return sorted(cells, key=lambda p: p[1])
    return sorted(cells, key=lambda p: p[0])


def _reflect_direction(wall: dict, hit_r: int, hit_c: int, direction: tuple[int, int]) -> tuple[int, int] | None:
    dr, dc = direction
    if wall["orientation"] == "v":
        if dc == 0:
            return None
        dist_top = hit_r - wall["r0"]
        dist_bottom = wall["r1"] - hit_r
        if dist_top == dist_bottom:
            return None
        return UP if dist_top < dist_bottom else DOWN

    if dr == 0:
        return None
    dist_left = hit_c - wall["c0"]
    dist_right = wall["c1"] - hit_c
    if dist_left == dist_right:
        return None
    return LEFT if dist_left < dist_right else RIGHT


def _apply_label(color: int, label: tuple[int, int]) -> int:
    a, b = label
    if color == a:
        return int(b)
    if color == b:
        return int(a)
    return int(color)


def _trace_ray(
    scene: np.ndarray,
    walls: list[dict],
    wall_index: np.ndarray,
    start_pos: tuple[int, int],
    direction: tuple[int, int],
    start_color: int,
    painted: set[tuple[int, int]],
    extra_blocked: set[tuple[int, int]],
) -> dict:
    h, w = scene.shape
    r, c = start_pos
    dr, dc = direction
    color = int(start_color)
    pixels: list[tuple[int, int, int]] = []
    local_positions: set[tuple[int, int]] = set()
    reflections = 0
    color_swaps = 0

    for _ in range(h * w * 8):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if (nr, nc) in extra_blocked or (nr, nc) in painted or (nr, nc) in local_positions:
            break

        cell = int(scene[nr, nc])
        if cell == WALL:
            wall = walls[int(wall_index[nr, nc])]
            new_direction = _reflect_direction(wall, int(nr), int(nc), (int(dr), int(dc)))
            if new_direction is None:
                break
            new_color = _apply_label(color, wall["label"])
            if new_color != color:
                color_swaps += 1
            color = int(new_color)
            dr, dc = new_direction
            reflections += 1
            continue

        if cell != BG:
            break

        pixels.append((int(nr), int(nc), int(color)))
        local_positions.add((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    else:
        raise AssertionError("ray simulation did not terminate")

    return {
        "pixels": pixels,
        "positions": local_positions,
        "reflections": int(reflections),
        "color_swaps": int(color_swaps),
    }


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    walls: list[dict] = []
    for cells in _components(g == WALL):
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        if len(rows) == 1:
            r = int(rows[0])
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            expected = {(r, c) for c in range(c0, c1 + 1)}
            assert set(cells) == expected, "grey wall must be a solid horizontal segment"
            walls.append(
                {
                    "orientation": "h",
                    "r0": int(r),
                    "r1": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "cells": sorted(expected),
                    "endpoints": [(int(r), int(c0)), (int(r), int(c1))],
                    "label": None,
                }
            )
        elif len(cols) == 1:
            c = int(cols[0])
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            expected = {(r, c) for r in range(r0, r1 + 1)}
            assert set(cells) == expected, "grey wall must be a solid vertical segment"
            walls.append(
                {
                    "orientation": "v",
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c),
                    "c1": int(c),
                    "cells": sorted(expected),
                    "endpoints": [(int(r0), int(c)), (int(r1), int(c))],
                    "label": None,
                }
            )
        else:
            raise AssertionError("grey component must be horizontal or vertical")

    assert walls, "expected at least one wall"
    wall_index = _build_wall_index(g.shape, walls)

    sources: list[dict] = []
    for comp in _components((g != BG) & (g != WALL)):
        if len(comp) == 1:
            r, c = comp[0]
            direction = _edge_direction(r, c, h, w)
            assert direction is not None, "single colored pixel must be an edge source"
            sources.append(
                {
                    "pos": (int(r), int(c)),
                    "dir": direction,
                    "color": int(g[r, c]),
                }
            )
            continue

        if len(comp) != 2:
            raise AssertionError("non-wall colored objects must be 1-pixel sources or 2-pixel labels")

        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "label must be a 2-cell straight segment"
        touched_walls: list[int] = []
        for idx, wall in enumerate(walls):
            adjacent = False
            for er, ec in wall["endpoints"]:
                for r, c in comp:
                    if abs(er - r) + abs(ec - c) == 1:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                touched_walls.append(int(idx))

        assert len(touched_walls) == 1, "label must belong to exactly one wall endpoint"
        wall = walls[int(touched_walls[0])]
        assert wall["label"] is None, "each wall must have exactly one label"
        ordered = _order_label_cells(comp)
        wall["label"] = (int(g[ordered[0]]), int(g[ordered[1]]))

    for wall in walls:
        assert wall["label"] is not None, "every wall must carry a label"

    assert sources, "expected at least one edge source"
    sources.sort(key=lambda item: item["pos"])
    return g, walls, wall_index, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored edge rays across a black canvas, reflecting them off grey walls.

    Each wall is a 1-pixel-thick horizontal or vertical grey segment. A 2-cell label
    attached to one endpoint encodes a visible color swap A<->B using the top/left
    label color as A and the bottom/right color as B. When a ray hits a wall that is
    perpendicular to its motion, it turns 90 degrees; the hit position relative to the
    wall midpoint chooses the turn side. Rays stop at the grid boundary, at existing
    colored cells, or at already painted ray cells.
    """
    scene, walls, wall_index, sources = _parse_scene(grid)
    out = scene.copy()
    painted: set[tuple[int, int]] = set()

    for source in sources:
        traced = _trace_ray(
            scene,
            walls,
            wall_index,
            source["pos"],
            source["dir"],
            source["color"],
            painted,
            set(),
        )
        for r, c, color in traced["pixels"]:
            out[r, c] = int(color)
        painted |= traced["positions"]

    return out


def _wall_clear_box(wall: dict) -> tuple[int, int, int, int]:
    if wall["orientation"] == "v":
        return (wall["r0"] - 2, wall["r1"] + 2, wall["c0"] - 1, wall["c0"] + 1)
    return (wall["r0"] - 1, wall["r0"] + 1, wall["c0"] - 2, wall["c1"] + 2)


def _place_walls(height: int, width: int) -> list[dict] | None:
    wall_count = random.choice(WALL_COUNT_CHOICES)
    orientations = ["v", "h"]
    while len(orientations) < wall_count:
        orientations.append(random.choice(["v", "h"]))
    random.shuffle(orientations)

    reserved = np.zeros((height, width), dtype=bool)
    walls: list[dict] = []

    for orientation in orientations:
        placed = False
        for _ in range(120):
            if orientation == "v":
                valid_lengths = [length for length in WALL_LEN_CHOICES if length <= height - 5]
                if not valid_lengths:
                    return None
                length = random.choice(valid_lengths)
                r0 = random.randint(2, height - length - 3)
                c = random.randint(2, width - 3)
                wall = {
                    "orientation": "v",
                    "r0": int(r0),
                    "r1": int(r0 + length - 1),
                    "c0": int(c),
                    "c1": int(c),
                }
            else:
                valid_lengths = [length for length in WALL_LEN_CHOICES if length <= width - 5]
                if not valid_lengths:
                    return None
                length = random.choice(valid_lengths)
                c0 = random.randint(2, width - length - 3)
                r = random.randint(2, height - 3)
                wall = {
                    "orientation": "h",
                    "r0": int(r),
                    "r1": int(r),
                    "c0": int(c0),
                    "c1": int(c0 + length - 1),
                }

            rr0, rr1, cc0, cc1 = _wall_clear_box(wall)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            if wall["orientation"] == "v":
                cells = [(r, wall["c0"]) for r in range(wall["r0"], wall["r1"] + 1)]
                wall["endpoints"] = [(wall["r0"], wall["c0"]), (wall["r1"], wall["c0"])]
            else:
                cells = [(wall["r0"], c) for c in range(wall["c0"], wall["c1"] + 1)]
                wall["endpoints"] = [(wall["r0"], wall["c0"]), (wall["r0"], wall["c1"])]

            wall["cells"] = cells
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            walls.append(wall)
            placed = True
            break

        if not placed:
            return None

    for wall in walls:
        wall["label"] = tuple(random.sample(PALETTE, 2))
        if wall["orientation"] == "v":
            if random.random() < 0.5:
                label_cells = [(wall["r0"] - 2, wall["c0"]), (wall["r0"] - 1, wall["c0"])]
            else:
                label_cells = [(wall["r1"] + 1, wall["c0"]), (wall["r1"] + 2, wall["c0"])]
        else:
            if random.random() < 0.5:
                label_cells = [(wall["r0"], wall["c0"] - 2), (wall["r0"], wall["c0"] - 1)]
            else:
                label_cells = [(wall["r0"], wall["c1"] + 1), (wall["r0"], wall["c1"] + 2)]
        wall["label_cells"] = label_cells

    return walls


def _render_scene(height: int, width: int, walls: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for wall in walls:
        for r, c in wall["cells"]:
            grid[r, c] = WALL
        for (r, c), color in zip(wall["label_cells"], wall["label"]):
            grid[r, c] = int(color)
    return grid


def _edge_slots(scene: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    h, w = scene.shape
    slots: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for c in range(1, w - 1):
        if int(scene[0, c]) == BG:
            slots.append(((0, c), DOWN, "top"))
        if int(scene[h - 1, c]) == BG:
            slots.append(((h - 1, c), UP, "bottom"))
    for r in range(1, h - 1):
        if int(scene[r, 0]) == BG:
            slots.append(((r, 0), RIGHT, "left"))
        if int(scene[r, w - 1]) == BG:
            slots.append(((r, w - 1), LEFT, "right"))
    return slots


def _candidate_sources(scene: np.ndarray, walls: list[dict], wall_index: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for pos, direction, edge in _edge_slots(scene):
        for color in PALETTE:
            traced = _trace_ray(
                scene,
                walls,
                wall_index,
                pos,
                direction,
                int(color),
                set(),
                {pos},
            )
            if len(traced["positions"]) < MIN_RAY_PIXELS:
                continue
            candidates.append(
                {
                    "pos": pos,
                    "dir": direction,
                    "edge": edge,
                    "color": int(color),
                    "pixels": traced["pixels"],
                    "positions": traced["positions"],
                    "reflections": int(traced["reflections"]),
                    "color_swaps": int(traced["color_swaps"]),
                    "has_change": bool(traced["color_swaps"] > 0),
                    "has_plain_reflect": bool(traced["reflections"] > 0 and traced["color_swaps"] == 0),
                    "has_multi": bool(traced["reflections"] >= 2),
                }
            )

    by_pos: dict[tuple[int, int], list[dict]] = {}
    for candidate in candidates:
        by_pos.setdefault(candidate["pos"], []).append(candidate)

    compact: list[dict] = []
    for items in by_pos.values():
        random.shuffle(items)
        items.sort(
            key=lambda cand: (
                int(cand["has_change"]),
                int(cand["has_plain_reflect"]),
                int(cand["reflections"]),
                int(len(cand["positions"])),
            ),
            reverse=True,
        )
        kept: list[dict] = []
        kept.append(items[0])
        for predicate in (
            lambda cand: cand["has_plain_reflect"],
            lambda cand: cand["has_change"],
            lambda cand: cand["has_multi"],
        ):
            extra = next((cand for cand in items if predicate(cand) and cand not in kept), None)
            if extra is not None:
                kept.append(extra)
        for cand in items:
            if cand not in kept:
                kept.append(cand)
            if len(kept) >= 4:
                break
        compact.extend(kept)

    return compact


def _is_compatible(candidate: dict, selected: list[dict], occupied: set[tuple[int, int]]) -> bool:
    if candidate["pos"] in occupied:
        return False
    if candidate["positions"] & occupied:
        return False
    for other in selected:
        if abs(candidate["pos"][0] - other["pos"][0]) + abs(candidate["pos"][1] - other["pos"][1]) == 1:
            return False
    return True


def _pick_sources(candidates: list[dict]) -> list[dict] | None:
    unique_positions = {cand["pos"] for cand in candidates}
    if len(unique_positions) < 2:
        return None

    target_sizes = list(range(min(SOURCE_COUNT_MAX, len(unique_positions)), 1, -1))
    need_change = any(cand["has_change"] for cand in candidates)
    need_plain = any(cand["has_plain_reflect"] for cand in candidates)
    need_multi = any(cand["has_multi"] for cand in candidates) and random.random() < 0.6

    for target in target_sizes:
        for _ in range(160):
            pool = candidates[:]
            random.shuffle(pool)
            selected: list[dict] = []
            occupied: set[tuple[int, int]] = set()
            used_edges: set[str] = set()

            while len(selected) < target:
                feasible = [cand for cand in pool if _is_compatible(cand, selected, occupied)]
                if not feasible:
                    break

                have_reflect = any(cand["reflections"] > 0 for cand in selected)
                have_change = any(cand["has_change"] for cand in selected)
                have_plain = any(cand["has_plain_reflect"] for cand in selected)
                have_multi = any(cand["has_multi"] for cand in selected)

                def score(cand: dict) -> float:
                    score_value = 0.0
                    if not have_reflect and cand["reflections"] > 0:
                        score_value += 5.0
                    if need_change and not have_change and cand["has_change"]:
                        score_value += 4.0
                    if need_plain and not have_plain and cand["has_plain_reflect"]:
                        score_value += 4.0
                    if need_multi and not have_multi and cand["has_multi"]:
                        score_value += 2.0
                    if cand["edge"] not in used_edges:
                        score_value += 0.5
                    score_value += 0.2 * cand["reflections"]
                    score_value += 0.01 * len(cand["positions"])
                    score_value += random.random() * 0.2
                    return score_value

                chosen = max(feasible, key=score)
                selected.append(chosen)
                occupied.add(chosen["pos"])
                occupied |= chosen["positions"]
                used_edges.add(chosen["edge"])

            if len(selected) < 2:
                continue
            if not any(cand["reflections"] > 0 for cand in selected):
                continue
            if need_change and not any(cand["has_change"] for cand in selected):
                continue
            if need_plain and not any(cand["has_plain_reflect"] for cand in selected):
                continue
            if need_multi and not any(cand["has_multi"] for cand in selected):
                continue
            return selected

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)

        walls = _place_walls(height, width)
        if walls is None:
            continue

        scene = _render_scene(height, width, walls)
        wall_index = _build_wall_index(scene.shape, walls)
        candidates = _candidate_sources(scene, walls, wall_index)
        selected = _pick_sources(candidates)
        if selected is None:
            continue

        grid = scene.copy()
        for source in selected:
            r, c = source["pos"]
            grid[r, c] = int(source["color"])

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue
        if int(np.count_nonzero(out != grid)) < 8:
            continue
        return grid

    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")

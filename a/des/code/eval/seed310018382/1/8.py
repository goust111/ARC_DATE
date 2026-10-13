from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LOCK = 9
DOOR_H = 5
DOOR_W = 5
ZONE_H = 5
ZONE_W = 3
GENERATION_ATTEMPTS = 400

CANONICAL_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "domino": ((0, 0), (0, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "el": ((0, 0), (1, 0), (1, 1)),
}

LOCKHOLE_VARIANTS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "domino": (
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((1, 1), (2, 1)),
        ((2, 1), (3, 1)),
    ),
    "line3": (
        ((2, 0), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (3, 1)),
    ),
    "el": (
        ((1, 1), (2, 1), (2, 2)),
        ((2, 1), (2, 2), (3, 1)),
        ((2, 0), (2, 1), (3, 1)),
        ((1, 1), (2, 0), (2, 1)),
    ),
}


def _normalize(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    h = max(r for r, _ in norm) + 1
    return _normalize([(c, h - 1 - r) for r, c in norm])


def _rotations(coords: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize(coords)
    for _ in range(4):
        out.add(cur)
        cur = _rot90(cur)
    return out


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_color_components(mask: np.ndarray, colors: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(colors[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc] and int(colors[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    q = deque([tuple(map(int, coords[0]))])
    seen[coords[0, 0], coords[0, 1]] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _scan_doors(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    doors: list[dict] = []
    for r0 in range(h - DOOR_H + 1):
        for c0 in range(w - DOOR_W + 1):
            color = int(g[r0, c0])
            if color in (BG, LOCK):
                continue
            if not np.all(g[r0 : r0 + DOOR_H, c0] == color):
                continue
            if not np.all(g[r0 : r0 + DOOR_H, c0 + 4] == color):
                continue
            if r0 > 0 and (int(g[r0 - 1, c0]) == color or int(g[r0 - 1, c0 + 4]) == color):
                continue
            if r0 + DOOR_H < h and (
                int(g[r0 + DOOR_H, c0]) == color or int(g[r0 + DOOR_H, c0 + 4]) == color
            ):
                continue

            hole = g[r0 : r0 + DOOR_H, c0 + 1 : c0 + 4]
            lock_mask = hole == LOCK
            if not np.all((hole == BG) | lock_mask):
                continue
            if int(lock_mask.sum()) not in (2, 3):
                continue
            if not bool(lock_mask[2, 1]):
                continue
            if not _is_single_component(lock_mask):
                continue

            lock_coords_abs = [
                (r0 + rr, c0 + 1 + cc)
                for rr, cc in zip(*np.where(lock_mask))
            ]
            door = {
                "top": int(r0),
                "left": int(c0),
                "right": int(c0 + 4),
                "color": int(color),
                "center_col": int(c0 + 2),
                "lock_shape": _normalize(lock_coords_abs),
                "frame_cells": [(r0 + rr, c0) for rr in range(DOOR_H)] + [(r0 + rr, c0 + 4) for rr in range(DOOR_H)],
                "lock_cells": lock_coords_abs,
            }
            doors.append(door)
    assert doors, "expected at least one door"
    return doors


def _nearest_door(comp: list[tuple[int, int]], doors: list[dict]) -> tuple[dict, bool]:
    ranked = []
    for idx, door in enumerate(doors):
        dist = min(abs(c - int(door["center_col"])) for _, c in comp)
        ranked.append((int(dist), int(door["center_col"]), idx))
    ranked.sort()
    best = doors[ranked[0][2]]
    unique = len(ranked) == 1 or ranked[0][0] < ranked[1][0]
    return best, unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let a left-side object pass through its nearest door iff its shape matches
    that door's lockhole under rotation; the passed object reappears on the
    right as a horizontal mirror colored like the door frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    doors = _scan_doors(g)
    door_mask = np.zeros_like(g, dtype=bool)
    for door in doors:
        for r, c in door["frame_cells"] + door["lock_cells"]:
            door_mask[r, c] = True

    object_mask = (g != BG) & (~door_mask)
    objects = _find_color_components(object_mask, g)
    out = g.copy()

    for comp in objects:
        door, _ = _nearest_door(comp, doors)
        if _normalize(comp) not in _rotations(door["lock_shape"]):
            continue
        for r, c in comp:
            out[r, c] = BG
        mirror = [(r, 2 * int(door["center_col"]) - c) for r, c in comp]
        for r, c in mirror:
            out[r, c] = int(door["color"])

    return out


def _build_object_placements() -> dict[str, list[tuple[tuple[int, int], ...]]]:
    placements: dict[str, list[tuple[tuple[int, int], ...]]] = {}
    for name, base in CANONICAL_SHAPES.items():
        placed: set[tuple[tuple[int, int], ...]] = set()
        for orient in _rotations(base):
            oh = max(r for r, _ in orient) + 1
            ow = max(c for _, c in orient) + 1
            for r0 in range(ZONE_H - oh + 1):
                for c0 in range(ZONE_W - ow + 1):
                    coords = tuple(sorted((r0 + r, c0 + c) for r, c in orient))
                    placed.add(coords)
        placements[name] = sorted(placed)
    return placements


OBJECT_PLACEMENTS = _build_object_placements()


def _touches(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    for r0, c0 in a:
        for r1, c1 in b:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return True
    return False


def _sample_spaced_positions(start: int, stop: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(start, stop + 1))
    if count == 1:
        return [random.choice(values)]
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _choose_counts(n_doors: int, total_objects: int) -> list[int]:
    counts = [1] * n_doors
    for _ in range(total_objects - n_doors):
        choices = [i for i, count in enumerate(counts) if count < 2]
        counts[random.choice(choices)] += 1
    random.shuffle(counts)
    return counts


def _random_door_classes(n_doors: int) -> list[str]:
    names = list(CANONICAL_SHAPES)
    while True:
        chosen = [random.choice(names) for _ in range(n_doors)]
        if len(set(chosen)) >= 2:
            return chosen


def _pick_lockhole(shape_name: str) -> tuple[tuple[int, int], ...]:
    return random.choice(LOCKHOLE_VARIANTS[shape_name])


def _place_objects_for_door(
    shape_names: list[str],
) -> list[tuple[tuple[int, int], ...]] | None:
    chosen: list[tuple[tuple[int, int], ...]] = []
    for shape_name in shape_names:
        candidates = OBJECT_PLACEMENTS[shape_name][:]
        random.shuffle(candidates)
        pick = None
        for coords in candidates:
            if all(not _touches(coords, other) for other in chosen):
                pick = coords
                break
        if pick is None:
            return None
        chosen.append(pick)
    return chosen
def _gen_attempt() -> np.ndarray | None:
    n_doors = random.choice([2, 3])
    total_objects = random.choice([3, 4] if n_doors == 2 else [4, 5])
    if total_objects > 8 - n_doors:
        return None

    h = random.randint(11, 18) if n_doors == 2 else random.randint(17, 24)
    w = random.randint(24, 32) if n_doors == 2 else random.randint(33, 40)

    door_rows = _sample_spaced_positions(0, h - DOOR_H, n_doors, DOOR_H + 1)
    door_cols = _sample_spaced_positions(3, w - 8, n_doors, 11)
    if door_rows is None or door_cols is None:
        return None

    random.shuffle(door_rows)
    door_classes = _random_door_classes(n_doors)
    counts = _choose_counts(n_doors, total_objects)

    palette = list(range(1, 9))
    random.shuffle(palette)
    door_colors = palette[:n_doors]
    object_colors = palette[n_doors : n_doors + total_objects]
    if len(object_colors) != total_objects:
        return None

    match_count = total_objects // 2
    if total_objects % 2 == 1 and random.random() < 0.5:
        match_count += 1
    outcomes = [True] * match_count + [False] * (total_objects - match_count)
    random.shuffle(outcomes)

    grid = np.full((h, w), BG, dtype=int)
    door_specs: list[dict] = []
    for row, col, shape_name, color in zip(door_rows, door_cols, door_classes, door_colors):
        for rr in range(DOOR_H):
            grid[row + rr, col] = color
            grid[row + rr, col + 4] = color
        lock_coords = _pick_lockhole(shape_name)
        for rr, cc in lock_coords:
            grid[row + rr, col + 1 + cc] = LOCK
        door_specs.append(
            {
                "top": row,
                "left": col,
                "center_col": col + 2,
                "shape_name": shape_name,
                "color": color,
                "lock_coords": lock_coords,
            }
        )

    objects: list[dict] = []
    outcome_index = 0
    color_index = 0
    for door_idx, count in enumerate(counts):
        door = door_specs[door_idx]
        desired_shapes: list[str] = []
        desired_matches: list[bool] = []
        for _ in range(count):
            match = outcomes[outcome_index]
            outcome_index += 1
            desired_matches.append(match)
            if match:
                desired_shapes.append(str(door["shape_name"]))
            else:
                others = [name for name in CANONICAL_SHAPES if name != door["shape_name"]]
                desired_shapes.append(random.choice(others))

        placements = _place_objects_for_door(desired_shapes)
        if placements is None:
            return None

        for placement, match in zip(placements, desired_matches):
            color = object_colors[color_index]
            color_index += 1
            abs_coords = tuple(
                sorted((int(door["top"]) + r, int(door["left"]) - 3 + c) for r, c in placement)
            )
            for r, c in abs_coords:
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if grid[r, c] != BG:
                    return None
                grid[r, c] = color
            objects.append(
                {
                    "coords": abs_coords,
                    "match": bool(match),
                    "door_idx": int(door_idx),
                    "color": int(color),
                }
            )

    if not any(obj["match"] for obj in objects) or all(obj["match"] for obj in objects):
        return None

    out = grid.copy()
    for obj in objects:
        door = door_specs[int(obj["door_idx"])]
        nearest, unique = _nearest_door(list(obj["coords"]), _scan_doors(grid))
        if not unique:
            return None
        if int(nearest["center_col"]) != int(door["center_col"]):
            return None
        shape = _normalize(obj["coords"])
        should_match = shape in _rotations(_normalize([(door["top"] + rr, door["left"] + 1 + cc) for rr, cc in door["lock_coords"]]))
        if should_match != bool(obj["match"]):
            return None
        if obj["match"]:
            for r, c in obj["coords"]:
                out[r, c] = BG
            mirrored = [(r, 2 * int(door["center_col"]) - c) for r, c in obj["coords"]]
            for r, c in mirrored:
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if out[r, c] not in (BG, int(door["color"])):
                    return None
                out[r, c] = int(door["color"])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

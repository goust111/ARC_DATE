from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 8
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
DECORATION_CELLS = {
    "N": [(-1, 0), (-1, -1), (-1, 1)],
    "S": [(1, 0), (1, -1), (1, 1)],
    "W": [(0, -1), (-1, -1), (1, -1)],
    "E": [(0, 1), (-1, 1), (1, 1)],
}

GEN_ATTEMPTS = 300
BOX = 9
ARM = BOX // 2
GAP = 1
MAX_SIDE = 30
FG_COLORS = tuple(color for color in range(10) if color != BG)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg or int(labels[r, c]) != -1:
                continue

            color = int(g[r, c])
            obj_id = len(objects)
            labels[r, c] = obj_id
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in N4:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if int(labels[nr, nc]) != -1:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    labels[nr, nc] = obj_id
                    q.append((nr, nc))

            objects.append(
                {
                    "id": obj_id,
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                }
            )

    return labels, objects


def _canonical_candidate_signature(candidate: dict) -> tuple:
    a = tuple(sorted((candidate["a"], candidate["b"])))
    endpoints = tuple(sorted((candidate["start"], candidate["end"])))
    path = tuple(candidate["path"])
    path_rev = tuple(reversed(path))
    return a, endpoints, min(path, path_rev)


def _ray_candidate(
    labels: np.ndarray,
    bg: int,
    obj_id: int,
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> dict | None:
    r, c = start
    nr = r + dr
    nc = c + dc
    if not _in_bounds(labels.shape, nr, nc):
        return None

    first_owner = int(labels[nr, nc])
    if first_owner == obj_id:
        return None
    if first_owner >= 0:
        return {
            "a": obj_id,
            "b": first_owner,
            "start": (int(r), int(c)),
            "end": (int(nr), int(nc)),
            "path": [],
            "gap": 0,
        }

    gap_cells: list[tuple[int, int]] = []
    rr = nr
    cc = nc
    while _in_bounds(labels.shape, rr, cc) and int(labels[rr, cc]) == -1:
        gap_cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc

    if not _in_bounds(labels.shape, rr, cc):
        return None
    other = int(labels[rr, cc])
    if other < 0 or other == obj_id:
        return None

    return {
        "a": obj_id,
        "b": other,
        "start": (int(r), int(c)),
        "end": (int(rr), int(cc)),
        "path": gap_cells,
        "gap": len(gap_cells),
    }


def _find_contact_paths(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], dict]] | None:
    labels, objects = _label_objects(grid, bg)
    by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    seen: set[tuple] = set()

    for obj in objects:
        obj_id = int(obj["id"])
        for start in obj["coords"]:
            for dr, dc in N4:
                candidate = _ray_candidate(labels, bg, obj_id, start, dr, dc)
                if candidate is None:
                    continue
                key = _canonical_candidate_signature(candidate)
                if key in seen:
                    continue
                seen.add(key)
                pair = tuple(sorted((candidate["a"], candidate["b"])))
                by_pair[pair].append(candidate)

    chosen: dict[tuple[int, int], dict] = {}
    for pair, candidates in by_pair.items():
        best_gap = min(int(candidate["gap"]) for candidate in candidates)
        best = [candidate for candidate in candidates if int(candidate["gap"]) == best_gap]
        if len(best) != 1:
            return None
        chosen[pair] = best[0]

    return objects, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a straight link between every pair of foreground objects that face each other
    horizontally or vertically across a unique shortest background corridor. The link
    uses the color whose total input area is smaller among the two object colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    found = _find_contact_paths(g, bg)
    assert found is not None, "every touching pair must have a unique shortest corridor"
    objects, contacts = found

    color_areas = {
        int(color): int(np.sum(g == int(color)))
        for color in np.unique(g)
        if int(color) != bg
    }

    out = g.copy()
    for candidate in contacts.values():
        color_a = int(objects[candidate["a"]]["color"])
        color_b = int(objects[candidate["b"]]["color"])
        area_a = int(color_areas[color_a])
        area_b = int(color_areas[color_b])
        assert area_a != area_b, "connected object colors must have distinct total areas"
        line_color = color_a if area_a < area_b else color_b

        for r, c in candidate["path"]:
            assert int(out[r, c]) == bg, "links must stay outside the original objects"
            out[r, c] = line_color

    return out


def _canonical_slots(kind: str) -> set[tuple[int, int]]:
    if kind == "2x2":
        return {(r, c) for r in range(2) for c in range(2)}
    slots = {(r, c) for r in range(2) for c in range(3)}
    if kind == "2x3":
        return slots
    assert kind == "2x3-cut"
    corner = random.choice([(0, 0), (0, 2), (1, 0), (1, 2)])
    slots.remove(corner)
    return slots


def _slot_dirs(slot: tuple[int, int], slots: set[tuple[int, int]]) -> list[str]:
    r, c = slot
    used = []
    for name, (dr, dc) in DIRS.items():
        if (r + dr, c + dc) in slots:
            used.append(name)
    return used


def _extra_areas(slots: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    by_degree: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for slot in sorted(slots):
        by_degree[len(_slot_dirs(slot, slots))].append(slot)

    extras: dict[tuple[int, int], int] = {}
    for degree, members in by_degree.items():
        assert degree in (1, 2, 3)
        values = list(range(len(members)))
        random.shuffle(values)
        order = members[:]
        random.shuffle(order)
        for slot, extra in zip(order, values):
            extras[slot] = int(extra)
    return extras


def _build_shape(used_dirs: list[str], extra: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    for name in used_dirs:
        dr, dc = DIRS[name]
        for step in range(1, ARM + 1):
            cells.add((dr * step, dc * step))

    if extra > 0:
        unused = [name for name in DIRS if name not in used_dirs]
        deco_side = random.choice(unused)
        for cell in DECORATION_CELLS[deco_side][:extra]:
            cells.add(cell)

    return cells


def _paint_object(
    grid: np.ndarray,
    slot: tuple[int, int],
    color: int,
    used_dirs: list[str],
    extra: int,
    margin_top: int,
    margin_left: int,
) -> None:
    center_r = margin_top + slot[0] * (BOX + GAP) + ARM
    center_c = margin_left + slot[1] * (BOX + GAP) + ARM

    for dr, dc in _build_shape(used_dirs, extra):
        grid[center_r + dr, center_c + dc] = int(color)


def _edge_count(slots: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in slots:
        if (r + 1, c) in slots:
            total += 1
        if (r, c + 1) in slots:
            total += 1
    return total


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _gen_attempt() -> np.ndarray | None:
    kind = random.choice(["2x2", "2x3", "2x3-cut"])
    slots = _canonical_slots(kind)
    rows = max(r for r, _ in slots) + 1
    cols = max(c for _, c in slots) + 1

    margin_top = random.randint(0, 1)
    margin_bottom = random.randint(0, 1)
    margin_left = random.randint(0, 1)
    margin_right = random.randint(0, 1)

    height = margin_top + rows * BOX + (rows - 1) * GAP + margin_bottom
    width = margin_left + cols * BOX + (cols - 1) * GAP + margin_right
    if height > MAX_SIDE or width > MAX_SIDE:
        return None

    extras = _extra_areas(slots)
    colors = random.sample(FG_COLORS, len(slots))
    slot_order = sorted(slots)
    random.shuffle(slot_order)
    color_map = {slot: color for slot, color in zip(slot_order, colors)}

    grid = np.full((height, width), BG, dtype=int)
    for slot in slot_order:
        _paint_object(
            grid,
            slot,
            color_map[slot],
            _slot_dirs(slot, slots),
            extras[slot],
            margin_top,
            margin_left,
        )

    grid = _apply_random_symmetry(grid)

    labels, objects = _label_objects(grid, BG)
    if int(np.max(labels)) + 1 != len(slots):
        return None
    if len(objects) != len(slots):
        return None

    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return None
    if len({int(obj["area"]) for obj in objects}) != len(objects):
        return None

    found = _find_contact_paths(grid, BG)
    if found is None:
        return None
    _objects, contacts = found
    if len(contacts) != _edge_count(slots):
        return None
    if not all(int(candidate["gap"]) == GAP for candidate in contacts.values()):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.all(out[grid != BG] == grid[grid != BG]):
        return None
    if int(np.count_nonzero(out != grid)) != len(contacts):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

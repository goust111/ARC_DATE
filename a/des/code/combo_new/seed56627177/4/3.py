from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

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

OPS_BY_COLOR = {RED: "H", BLUE: "V", GREEN: "R"}
COLOR_BY_OP = {op: color for color, op in OPS_BY_COLOR.items()}
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_TEMPLATES = {
    "right": {(0, 0), (-1, -1), (1, -1)},
    "left": {(0, 0), (-1, 1), (1, 1)},
    "down": {(0, 0), (-1, -1), (-1, 1)},
    "up": {(0, 0), (1, -1), (1, 1)},
}

SLOT_SIZE = 13
SLOT_GAP = 1
OUTER_MARGIN = 1
MIN_SIDE_MARGIN = 4
MAX_RETRIES = 300
OBJECT_SIDES = (3, 4, 5)
AREA_CAP_BY_SIDE = {3: 7, 4: 10, 5: 13}
MULTI_SEQS_2 = (["H", "R"], ["R", "H"], ["V", "R"], ["R", "V"])
MULTI_SEQS_3 = (
    ["H", "R", "V"],
    ["V", "R", "H"],
    ["R", "H", "V"],
    ["R", "V", "H"],
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    neigh = DIRS8 if connectivity == 8 else DIRS4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_arrow(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], str] | None:
    pts = set(comp)
    if len(pts) != 3:
        return None
    for tip in pts:
        rel = {(r - tip[0], c - tip[1]) for r, c in pts}
        for dir_name, template in ARROW_TEMPLATES.items():
            if rel == template:
                return tip, dir_name
    return None


def _transform_coords(coords: set[tuple[int, int]], op: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    if op == "H":
        return {(r, c0 + c1 - c) for r, c in coords}
    if op == "V":
        return {(r0 + r1 - r, c) for r, c in coords}
    if op == "R":
        cr2 = r0 + r1
        cc2 = c0 + c1
        out: set[tuple[int, int]] = set()
        for r, c in coords:
            nr2 = cr2 + 2 * c - cc2
            nc2 = cc2 - 2 * r + cr2
            assert nr2 % 2 == 0 and nc2 % 2 == 0
            out.add((nr2 // 2, nc2 // 2))
        return out
    raise ValueError(f"unknown transform {op}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each grey object in place according to the colored arrows that point at it.

    Input structure:
    - The background is black.
    - Grey pixels form several small foreground objects, each occupying a compact 3x3 to 5x5 style region.
    - Red, blue, and green markers appear as 3-pixel arrows: one tip pixel plus two tail-wing pixels in a V behind the tip.
    - Extending an arrow from its tip along the pointing direction reaches exactly one grey object through only background cells.

    Rule:
    - Group grey pixels into objects and group same-color arrow pixels into 3-cell arrow components.
    - Red arrows apply a horizontal flip to the object they point at; blue arrows apply a vertical flip; green arrows apply a 90-degree clockwise rotation.
    - If several arrows point at the same object, sort them by the number of blank cells between the tip and the first hit object pixel, from smaller to larger, and apply the transforms in that order.
    - Every transform is anchored at the current object's bounding-box center and is performed in place; all arrows are removed from the output.

    Key insight:
    - Arrow color determines which geometric transform to use, while arrow distance determines the transform order when multiple arrows control the same object.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    grey_comps = _components(g == GREY, connectivity=8)
    assert grey_comps
    cell_to_object: dict[tuple[int, int], int] = {}
    objects: list[set[tuple[int, int]]] = []
    for idx, comp in enumerate(grey_comps):
        cells = set(comp)
        objects.append(cells)
        for cell in cells:
            cell_to_object[cell] = idx

    arrows_by_object: dict[int, list[tuple[int, str, tuple[int, int]]]] = defaultdict(list)
    for color, op in OPS_BY_COLOR.items():
        for comp in _components(g == color, connectivity=8):
            decoded = _decode_arrow(comp)
            assert decoded is not None
            tip, dir_name = decoded
            dr, dc = DIR_VECS[dir_name]
            r, c = tip
            blanks = 0
            while True:
                r += dr
                c += dc
                assert 0 <= r < h and 0 <= c < w
                value = int(g[r, c])
                if value == bg:
                    blanks += 1
                    continue
                assert value == GREY
                obj_idx = cell_to_object[(r, c)]
                arrows_by_object[obj_idx].append((blanks, op, tip))
                break

    out = np.full_like(g, bg)
    for obj_idx, cells in enumerate(objects):
        cur = set(cells)
        actions = sorted(arrows_by_object.get(obj_idx, []), key=lambda item: (item[0], item[2]))
        for _dist, op, _tip in actions:
            cur = _transform_coords(cur, op)
        for r, c in cur:
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] == bg
            out[r, c] = GREY
    return out


def _serialize_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _apply_local_op(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "H":
        return np.fliplr(mask)
    if op == "V":
        return np.flipud(mask)
    if op == "R":
        return np.rot90(mask, -1)
    raise ValueError(f"unknown op {op}")


def _apply_local_ops(mask: np.ndarray, ops: list[str]) -> np.ndarray:
    cur = mask.copy()
    for op in ops:
        cur = _apply_local_op(cur, op)
    return cur


def _dihedral_signature_count(mask: np.ndarray) -> int:
    sigs = set()
    for k in range(4):
        rot = np.rot90(mask, -k)
        sigs.add(_serialize_mask(rot))
        sigs.add(_serialize_mask(np.fliplr(rot)))
    return len(sigs)


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    cur = np.rot90(mask, -random.randrange(4))
    if random.random() < 0.5:
        cur = np.fliplr(cur)
    return cur


def _step_toward(a: int, b: int) -> int:
    return 1 if b > a else -1


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    cur = start
    path = [cur]
    while cur != end:
        options: list[tuple[int, int]] = []
        if cur[0] != end[0]:
            options.append((cur[0] + _step_toward(cur[0], end[0]), cur[1]))
        if cur[1] != end[1]:
            options.append((cur[0], cur[1] + _step_toward(cur[1], end[1])))
        cur = random.choice(options)
        path.append(cur)
    return path


def _random_connected_mask(side: int) -> np.ndarray:
    for _ in range(200):
        anchors = [
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        ]
        if len(set(anchors)) < 3:
            continue

        points = anchors[:]
        random.shuffle(points)
        cells = {points[0]}
        cur = points[0]
        for nxt in points[1:]:
            for pt in _monotone_path(cur, nxt):
                cells.add(pt)
            cur = nxt

        max_area = min(side * side - 2, AREA_CAP_BY_SIDE[side])
        if len(cells) > max_area:
            continue
        min_area = max(len(cells), side + 2)
        target_area = random.randint(min_area, max_area)

        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            unique_frontier = list(dict.fromkeys(frontier))
            weights = []
            for r, c in unique_frontier:
                neighbors = sum((r + dr, c + dc) in cells for dr, dc in DIRS4)
                weights.append(4 if neighbors == 1 else 2 if neighbors == 2 else 1)
            cells.add(random.choices(unique_frontier, weights=weights, k=1)[0])

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _components(mask, connectivity=4).__len__() != 1:
            continue
        if mask.mean() > 0.75:
            continue
        return mask
    raise RuntimeError("failed to build a connected full-bbox mask")


def _make_object_mask(seq: list[str], require_order_sensitive: bool) -> np.ndarray:
    for _ in range(300):
        side = random.choice(OBJECT_SIDES)
        mask = _random_dihedral(_random_connected_mask(side))
        if _dihedral_signature_count(mask) < 4:
            continue
        transformed = _apply_local_ops(mask, seq)
        if np.array_equal(mask, transformed):
            continue
        if require_order_sensitive and np.array_equal(
            transformed, _apply_local_ops(mask, list(reversed(seq)))
        ):
            continue
        return mask
    raise RuntimeError("failed to build a suitable object mask")


def _choose_sequences(object_count: int) -> list[tuple[list[str], bool]]:
    multi_seq = list(random.choice(MULTI_SEQS_2))

    entries: list[tuple[list[str], bool]] = [(multi_seq, True)]
    missing = list({"H", "V", "R"} - set(multi_seq))
    random.shuffle(missing)
    for op in missing:
        if len(entries) < object_count:
            entries.append(([op], False))
    while len(entries) < object_count:
        entries.append(([random.choice(["H", "V", "R"])], False))
    random.shuffle(entries)
    return entries


def _choose_layout(object_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if object_count == 2:
        return random.choice(
            [
                (1, 2, [(0, 0), (0, 1)]),
                (2, 1, [(0, 0), (1, 0)]),
                (2, 2, [(0, 0), (1, 1)]),
                (2, 2, [(0, 1), (1, 0)]),
            ]
        )

    all_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    missing = random.choice(all_slots)
    slots = [slot for slot in all_slots if slot != missing]
    random.shuffle(slots)
    return 2, 2, slots


def _slot_origin(slot: tuple[int, int]) -> tuple[int, int]:
    return (
        OUTER_MARGIN + slot[0] * (SLOT_SIZE + SLOT_GAP),
        OUTER_MARGIN + slot[1] * (SLOT_SIZE + SLOT_GAP),
    )


def _grid_shape(rows_used: int, cols_used: int) -> tuple[int, int]:
    return (
        OUTER_MARGIN + rows_used * (SLOT_SIZE + SLOT_GAP),
        OUTER_MARGIN + cols_used * (SLOT_SIZE + SLOT_GAP),
    )


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                cells.add((top + r, left + c))
    return cells


def _candidate_targets(obj_cells: set[tuple[int, int]], dir_name: str) -> list[tuple[int, int]]:
    leftmost: dict[int, int] = {}
    rightmost: dict[int, int] = {}
    topmost: dict[int, int] = {}
    bottommost: dict[int, int] = {}
    for r, c in obj_cells:
        leftmost[r] = c if r not in leftmost else min(leftmost[r], c)
        rightmost[r] = c if r not in rightmost else max(rightmost[r], c)
        topmost[c] = r if c not in topmost else min(topmost[c], r)
        bottommost[c] = r if c not in bottommost else max(bottommost[c], r)

    if dir_name == "right":
        return [(r, leftmost[r]) for r in leftmost]
    if dir_name == "left":
        return [(r, rightmost[r]) for r in rightmost]
    if dir_name == "down":
        return [(topmost[c], c) for c in topmost]
    if dir_name == "up":
        return [(bottommost[c], c) for c in bottommost]
    raise ValueError(f"unknown direction {dir_name}")


def _arrow_geometry(
    target: tuple[int, int], dir_name: str, distance: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    tr, tc = target
    if dir_name == "right":
        tip = (tr, tc - distance - 1)
        arrow = [tip, (tip[0] - 1, tip[1] - 1), (tip[0] + 1, tip[1] - 1)]
        corridor = [(tr, c) for c in range(tip[1] + 1, tc)]
        return arrow, corridor
    if dir_name == "left":
        tip = (tr, tc + distance + 1)
        arrow = [tip, (tip[0] - 1, tip[1] + 1), (tip[0] + 1, tip[1] + 1)]
        corridor = [(tr, c) for c in range(tc + 1, tip[1])]
        return arrow, corridor
    if dir_name == "down":
        tip = (tr - distance - 1, tc)
        arrow = [tip, (tip[0] - 1, tip[1] - 1), (tip[0] - 1, tip[1] + 1)]
        corridor = [(r, tc) for r in range(tip[0] + 1, tr)]
        return arrow, corridor
    if dir_name == "up":
        tip = (tr + distance + 1, tc)
        arrow = [tip, (tip[0] + 1, tip[1] - 1), (tip[0] + 1, tip[1] + 1)]
        corridor = [(r, tc) for r in range(tr + 1, tip[0])]
        return arrow, corridor
    raise ValueError(f"unknown direction {dir_name}")


def _cells_in_slot(cells: list[tuple[int, int]], slot_r0: int, slot_c0: int) -> bool:
    for r, c in cells:
        if not (slot_r0 <= r < slot_r0 + SLOT_SIZE and slot_c0 <= c < slot_c0 + SLOT_SIZE):
            return False
    return True


def _place_object_in_slot(
    grid: np.ndarray, slot_r0: int, slot_c0: int, mask: np.ndarray, seq: list[str]
) -> bool:
    side = mask.shape[0]
    row_choices = list(range(MIN_SIDE_MARGIN, SLOT_SIZE - side - MIN_SIDE_MARGIN + 1))
    col_choices = list(range(MIN_SIDE_MARGIN, SLOT_SIZE - side - MIN_SIDE_MARGIN + 1))
    if not row_choices or not col_choices:
        return False

    for _ in range(120):
        top = slot_r0 + random.choice(row_choices)
        left = slot_c0 + random.choice(col_choices)
        obj_cells = _mask_cells(mask, top, left)

        dir_names = random.sample(list(DIR_VECS), len(seq))
        if len(seq) == 1:
            distances = [random.choice([0, 1, 2])]
        else:
            distances = sorted(random.sample([0, 1, 2], len(seq)))
        pairings = list(zip(seq, dir_names, distances))
        place_order = sorted(pairings, key=lambda item: -item[2])

        reserved = set(obj_cells)
        arrow_specs: list[tuple[int, list[tuple[int, int]]]] = []
        ok = True
        for op, dir_name, distance in place_order:
            candidates = _candidate_targets(obj_cells, dir_name)
            random.shuffle(candidates)
            placed = False
            for target in candidates:
                arrow_cells, corridor = _arrow_geometry(target, dir_name, distance)
                footprint = arrow_cells + corridor
                if not _cells_in_slot(footprint, slot_r0, slot_c0):
                    continue
                if any(cell in reserved for cell in footprint):
                    continue
                if any(grid[r, c] != BLACK for r, c in footprint):
                    continue
                reserved.update(footprint)
                arrow_specs.append((COLOR_BY_OP[op], arrow_cells))
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        for r, c in obj_cells:
            assert grid[r, c] == BLACK
            grid[r, c] = GREY
        for color, cells in arrow_specs:
            for r, c in cells:
                assert grid[r, c] == BLACK
                grid[r, c] = color
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice([2, 3])
    rows_used, cols_used, slots = _choose_layout(object_count)
    h, w = _grid_shape(rows_used, cols_used)
    grid = np.full((h, w), BLACK, dtype=int)

    entries = _choose_sequences(object_count)
    for slot, (seq, require_order_sensitive) in zip(slots, entries):
        mask = _make_object_mask(seq, require_order_sensitive=require_order_sensitive)
        slot_r0, slot_c0 = _slot_origin(slot)
        if not _place_object_in_slot(grid, slot_r0, slot_c0, mask, seq):
            return None

    colors_present = set(int(v) for v in np.unique(grid))
    if not {RED, BLUE, GREEN, GREY}.issubset(colors_present):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, np.where(grid == GREY, GREY, BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

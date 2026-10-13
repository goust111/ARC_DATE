from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LABEL_COLORS = (BLUE, RED, GREEN)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_offsets(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _label_dist(adjacent_cells: list[tuple[int, int]], pixel: tuple[int, int]) -> int:
    pr, pc = pixel
    return min(max(abs(pr - r), abs(pc - c)) for r, c in adjacent_cells)


def _find_slots(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    yellow_comps = _components4(grid == YELLOW)
    slots: list[dict] = []
    slot_mask = np.zeros_like(grid, dtype=bool)

    for comp in yellow_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        expected_yellow = 2 * h + 2 * w - 5
        if len(comp) != expected_yellow:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = int(grid[r, c])
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    if (r, c) == (r0, c0):
                        if v not in LABEL_COLORS:
                            ok = False
                            break
                    elif v != YELLOW:
                        ok = False
                        break
                elif v != BLACK:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": int(grid[r0, c0]),
                "inner_shape": (h - 2, w - 2),
            }
        )
        slot_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert slots, "expected at least one yellow slot frame"
    slots.sort(key=lambda slot: (slot["bbox"][1], slot["bbox"][0]))

    outer_shapes = {(slot["bbox"][2] - slot["bbox"][0] + 1, slot["bbox"][3] - slot["bbox"][1] + 1) for slot in slots}
    assert len(outer_shapes) == 1, "slot frames must have identical sizes"
    return slots, slot_mask


def _extract_objects(grid: np.ndarray, slot_mask: np.ndarray) -> list[dict]:
    upper_mask = (grid != BLACK) & (~slot_mask)
    comps = _components4(upper_mask)

    objects: list[dict] = []
    orphan_labels: list[tuple[tuple[int, int], int]] = []

    for comp in comps:
        grey_cells = [(r, c) for r, c in comp if int(grid[r, c]) == GREY]
        label_cells = [(r, c) for r, c in comp if int(grid[r, c]) in LABEL_COLORS]

        if grey_cells:
            objects.append({"grey": grey_cells, "embedded": label_cells})
        elif len(comp) == 1 and int(grid[comp[0]]) in LABEL_COLORS:
            orphan_labels.append((comp[0], int(grid[comp[0]])))
        else:
            raise AssertionError("unexpected upper component structure")

    used_orphans: set[int] = set()
    parsed: list[dict] = []

    for obj in objects:
        labels = [(cell, int(grid[cell])) for cell in obj["embedded"]]
        if not labels:
            nearby: list[tuple[tuple[int, int], int, int]] = []
            for idx, (cell, color) in enumerate(orphan_labels):
                if idx in used_orphans:
                    continue
                dist = _label_dist(obj["grey"], cell)
                if dist <= 1:
                    nearby.append((cell, color, idx))
            assert len(nearby) == 1, "each grey object must match exactly one label pixel"
            cell, color, idx = nearby[0]
            used_orphans.add(idx)
            labels = [(cell, color)]

        assert len(labels) == 1, "each object must carry one label pixel"
        label_cell, label_color = labels[0]
        grey = obj["grey"]
        offsets = _shape_offsets(grey)
        r0, c0, r1, c1 = _bbox(grey)
        parsed.append(
            {
                "label_cell": label_cell,
                "color": int(label_color),
                "area": len(grey),
                "bbox": (r0, c0, r1, c1),
                "shape": offsets,
                "shape_size": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )

    assert len(used_orphans) == len(orphan_labels), "all detached label pixels must be assigned"
    assert parsed, "expected at least one labelled object"
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey objects by their label color, sort each group by area, and redraw the shapes into matching bottom slot frames.

    Input structure:
    - The background is black.
    - The upper region contains several grey 4-connected objects, each with one
      nearby or enclosed single-pixel label in red, blue, or green.
    - The lower region contains same-sized hollow yellow slot frames arranged in
      a row; each slot has a colored top-left corner marker matching one label group.

    Rule:
    - Detect each grey object and attach its unique label pixel.
    - Group objects by label color, then sort within each color by grey area from
      smallest to largest.
    - Erase the original upper objects and labels, keep every yellow frame and
      corner marker, and place the sorted object shapes into the matching slots.
    - Each copied shape is drawn in the label color and aligned to the slot
      interior's top-left corner.

    Key insight:
    - The label color does not repaint the original object in place; instead it
      chooses a destination slot group, while the grey object's area determines
      the order inside that group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    slots, slot_mask = _find_slots(g)
    objects = _extract_objects(g, slot_mask)

    slots_by_color: dict[int, list[dict]] = defaultdict(list)
    for slot in slots:
        slots_by_color[int(slot["color"])].append(slot)

    objects_by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        objects_by_color[int(obj["color"])].append(obj)

    out = np.zeros_like(g)
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]

    for color, slot_group in slots_by_color.items():
        obj_group = sorted(
            objects_by_color.get(color, []),
            key=lambda obj: (
                int(obj["area"]),
                int(obj["shape_size"][0]),
                int(obj["shape_size"][1]),
                int(obj["bbox"][0]),
                int(obj["bbox"][1]),
            ),
        )
        assert len(obj_group) == len(slot_group), "slot count must match object count for each label color"
        for obj, slot in zip(obj_group, slot_group):
            inner_h, inner_w = slot["inner_shape"]
            shape_h, shape_w = obj["shape_size"]
            assert shape_h <= inner_h and shape_w <= inner_w, "slot interior must fit the copied shape"
            base_r = slot["bbox"][0] + 1
            base_c = slot["bbox"][1] + 1
            for dr, dc in obj["shape"]:
                out[base_r + dr, base_c + dc] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (22, 28)
SLOT_OUTER = 5
SLOT_GAP = 1
SLOT_BOTTOM_MARGIN = (1, 2)
TOP_MARGIN = 1
OBJECT_COUNT_CHOICES = (3, 4)
PRESENT_COLOR_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected_shape(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    start = next(iter(cell_set))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cell_set)


def _all_shapes_by_area() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_area: dict[int, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    coords = [(r, c) for r in range(3) for c in range(3)]
    for mask in range(1, 1 << len(coords)):
        area = mask.bit_count()
        if area < 3 or area > 8:
            continue
        cells = {coords[i] for i in range(len(coords)) if (mask >> i) & 1}
        norm = _normalize(cells)
        if _connected_shape(norm):
            by_area[area].add(norm)
    return {area: sorted(shapes) for area, shapes in by_area.items()}


SHAPES_BY_AREA = _all_shapes_by_area()


def _label_candidates(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    shape_set = set(shape)
    candidates = set()
    for r, c in shape:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in shape_set:
                candidates.add((nr, nc))
    return sorted(candidates)


def _pick_group_sizes(total_objects: int, color_count: int) -> list[int]:
    for _ in range(100):
        cuts = sorted(random.sample(range(1, total_objects), color_count - 1))
        parts = []
        prev = 0
        for cut in cuts + [total_objects]:
            parts.append(cut - prev)
            prev = cut
        if max(parts) <= 3:
            return parts
    raise RuntimeError("failed to split object counts across colors")


def _draw_slot(grid: np.ndarray, r0: int, c0: int, color: int) -> dict:
    r1 = r0 + SLOT_OUTER - 1
    c1 = c0 + SLOT_OUTER - 1
    grid[r0, c0 : c1 + 1] = YELLOW
    grid[r1, c0 : c1 + 1] = YELLOW
    grid[r0 : r1 + 1, c0] = YELLOW
    grid[r0 : r1 + 1, c1] = YELLOW
    grid[r0, c0] = color
    return {"bbox": (r0, c0, r1, c1), "color": color, "inner_shape": (SLOT_OUTER - 2, SLOT_OUTER - 2)}


def _reserve_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    top_limit: int,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> dict | None:
    label_rel = random.choice(_label_candidates(shape))
    union = list(shape) + [label_rel]
    ur0 = min(r for r, _ in union)
    uc0 = min(c for _, c in union)
    ur1 = max(r for r, _ in union)
    uc1 = max(c for _, c in union)
    union_h = ur1 - ur0 + 1
    union_w = uc1 - uc0 + 1
    union_cells = [(r - ur0, c - uc0) for r, c in union]
    grey_cells = [(r - ur0, c - uc0) for r, c in shape]
    label_cell = (label_rel[0] - ur0, label_rel[1] - uc0)

    h, w = grid.shape
    if union_h > top_limit - TOP_MARGIN or union_w > w - 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = random.randint(TOP_MARGIN, top_limit - union_h)
        base_c = random.randint(1, w - union_w - 1)
        abs_union = [(base_r + r, base_c + c) for r, c in union_cells]
        if any(bool(blocked[r, c]) for r, c in abs_union):
            continue

        abs_grey = [(base_r + r, base_c + c) for r, c in grey_cells]
        abs_label = (base_r + label_cell[0], base_c + label_cell[1])
        for r, c in abs_grey:
            grid[r, c] = GREY
        grid[abs_label] = color
        _reserve_blocked(blocked, abs_union)
        return {
            "color": color,
            "area": len(shape),
            "shape": tuple(sorted(shape)),
            "bbox": _bbox(abs_grey),
            "shape_size": (
                max(r for r, _ in shape) - min(r for r, _ in shape) + 1,
                max(c for _, c in shape) - min(c for _, c in shape) + 1,
            ),
        }
    return None


def _expected_output(inp: np.ndarray, slots: list[dict], objects: list[dict]) -> np.ndarray:
    out = np.zeros_like(inp)
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = inp[r0 : r1 + 1, c0 : c1 + 1]

    objects_by_color: dict[int, list[dict]] = defaultdict(list)
    slots_by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        objects_by_color[int(obj["color"])].append(obj)
    for slot in slots:
        slots_by_color[int(slot["color"])].append(slot)

    for color, slot_group in slots_by_color.items():
        obj_group = sorted(
            objects_by_color[color],
            key=lambda obj: (
                int(obj["area"]),
                int(obj["shape_size"][0]),
                int(obj["shape_size"][1]),
                int(obj["bbox"][0]),
                int(obj["bbox"][1]),
            ),
        )
        for obj, slot in zip(obj_group, slot_group):
            base_r = slot["bbox"][0] + 1
            base_c = slot["bbox"][1] + 1
            for dr, dc in obj["shape"]:
                out[base_r + dr, base_c + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    total_objects = random.choice(OBJECT_COUNT_CHOICES)
    color_count = random.choice([n for n in PRESENT_COLOR_COUNT_CHOICES if n <= total_objects])
    present_colors = random.sample(list(LABEL_COLORS), color_count)
    group_sizes = _pick_group_sizes(total_objects, color_count)
    group_order = random.sample(present_colors, len(present_colors))
    count_by_color = {color: size for color, size in zip(present_colors, group_sizes)}

    object_specs: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for color in present_colors:
        count = count_by_color[color]
        areas = random.sample(list(range(3, 9)), count)
        for area in areas:
            shape = random.choice(SHAPES_BY_AREA[area])
            object_specs.append((color, shape))
    random.shuffle(object_specs)

    min_w = max(GRID_W_RANGE[0], total_objects * SLOT_OUTER + (total_objects - 1) * SLOT_GAP + 2)
    w = random.randint(min_w, GRID_W_RANGE[1])
    h = random.randint(*GRID_H_RANGE)
    bottom_margin = random.randint(*SLOT_BOTTOM_MARGIN)
    slot_r0 = h - SLOT_OUTER - bottom_margin
    top_limit = slot_r0 - 2
    if top_limit < 6:
        return None

    total_slot_width = total_objects * SLOT_OUTER + (total_objects - 1) * SLOT_GAP
    slot_c0 = random.randint(1, w - total_slot_width - 1)

    grid = np.zeros((h, w), dtype=int)
    slots: list[dict] = []
    cur_c = slot_c0
    for color in group_order:
        for _ in range(count_by_color[color]):
            slots.append(_draw_slot(grid, slot_r0, cur_c, color))
            cur_c += SLOT_OUTER + SLOT_GAP

    blocked = np.zeros_like(grid, dtype=bool)
    slot_mask = np.zeros_like(grid, dtype=bool)
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        slot_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    blocked |= slot_mask
    if slot_r0 - 1 >= 0:
        blocked[slot_r0 - 1, :] = True

    objects: list[dict] = []
    for color, shape in object_specs:
        placed = _place_object(grid, blocked, top_limit, shape, color)
        if placed is None:
            return None
        objects.append(placed)

    expected = _expected_output(grid, slots, objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

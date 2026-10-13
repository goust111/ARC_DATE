from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _color_components(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    mask = g == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_filled_rectangle(g: np.ndarray, color: int, cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(np.all(patch == color))


def _enumerate_arrow_variants(
    r0: int, c0: int, h: int, w: int, direction: str, grid_h: int, grid_w: int
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    variants: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []

    if direction == "right" and c1 + 2 < grid_w:
        for rr in range(r0, r1 + 1):
            for bend in (-1, 1):
                if 0 <= rr + bend < grid_h:
                    root = (rr, c1 + 1)
                    cells = (root, (rr, c1 + 2), (rr + bend, c1 + 2))
                    variants.append((root, cells))
    elif direction == "left" and c0 - 2 >= 0:
        for rr in range(r0, r1 + 1):
            for bend in (-1, 1):
                if 0 <= rr + bend < grid_h:
                    root = (rr, c0 - 1)
                    cells = (root, (rr, c0 - 2), (rr + bend, c0 - 2))
                    variants.append((root, cells))
    elif direction == "down" and r1 + 2 < grid_h:
        for cc in range(c0, c1 + 1):
            for bend in (-1, 1):
                if 0 <= cc + bend < grid_w:
                    root = (r1 + 1, cc)
                    cells = (root, (r1 + 2, cc), (r1 + 2, cc + bend))
                    variants.append((root, cells))
    elif direction == "up" and r0 - 2 >= 0:
        for cc in range(c0, c1 + 1):
            for bend in (-1, 1):
                if 0 <= cc + bend < grid_w:
                    root = (r0 - 1, cc)
                    cells = (root, (r0 - 2, cc), (r0 - 2, cc + bend))
                    variants.append((root, cells))
    return variants


def _arrow_matches_direction(
    cells: set[tuple[int, int]], root: tuple[int, int], direction: str
) -> bool:
    r, c = root
    if direction == "right":
        return any(cells == {root, (r, c + 1), (r + bend, c + 1)} for bend in (-1, 1))
    if direction == "left":
        return any(cells == {root, (r, c - 1), (r + bend, c - 1)} for bend in (-1, 1))
    if direction == "down":
        return any(cells == {root, (r + 1, c), (r + 1, c + bend)} for bend in (-1, 1))
    if direction == "up":
        return any(cells == {root, (r - 1, c), (r - 1, c + bend)} for bend in (-1, 1))
    return False


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, MAROON)):
        for cells in _color_components(g, color):
            assert _is_filled_rectangle(g, color, cells), "every non-maroon object must be a solid rectangle"
            r0, c0, r1, c1 = _bbox(cells)
            rectangles.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "cells": set(cells),
                }
            )

    assert rectangles, "expected at least one colored rectangle"

    arrow_components = _color_components(g, MAROON)
    cell_to_arrow: dict[tuple[int, int], int] = {}
    arrow_sets: list[set[tuple[int, int]]] = []
    for idx, comp in enumerate(arrow_components):
        comp_set = set(comp)
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == 3 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "arrow must be a 3-cell L shape"
        arrow_sets.append(comp_set)
        for cell in comp:
            cell_to_arrow[cell] = idx

    used_arrows: set[int] = set()
    parsed: list[dict] = []

    for rect in rectangles:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        side_hits: list[tuple[str, tuple[int, int]]] = []

        if c1 + 1 < w:
            right_roots = [(r, c1 + 1) for r in range(r0, r1 + 1) if int(g[r, c1 + 1]) == MAROON]
            if len(right_roots) == 1:
                side_hits.append(("right", right_roots[0]))
            else:
                assert not right_roots, "rectangle side must not contain multiple arrow roots"
        if c0 - 1 >= 0:
            left_roots = [(r, c0 - 1) for r in range(r0, r1 + 1) if int(g[r, c0 - 1]) == MAROON]
            if len(left_roots) == 1:
                side_hits.append(("left", left_roots[0]))
            else:
                assert not left_roots, "rectangle side must not contain multiple arrow roots"
        if r1 + 1 < h:
            down_roots = [(r1 + 1, c) for c in range(c0, c1 + 1) if int(g[r1 + 1, c]) == MAROON]
            if len(down_roots) == 1:
                side_hits.append(("down", down_roots[0]))
            else:
                assert not down_roots, "rectangle side must not contain multiple arrow roots"
        if r0 - 1 >= 0:
            up_roots = [(r0 - 1, c) for c in range(c0, c1 + 1) if int(g[r0 - 1, c]) == MAROON]
            if len(up_roots) == 1:
                side_hits.append(("up", up_roots[0]))
            else:
                assert not up_roots, "rectangle side must not contain multiple arrow roots"

        assert len(side_hits) == 1, "each rectangle must have exactly one adjacent arrow"
        direction, root = side_hits[0]
        assert root in cell_to_arrow, "arrow root must belong to a maroon component"
        arrow_idx = cell_to_arrow[root]
        assert arrow_idx not in used_arrows, "each arrow must control exactly one rectangle"
        used_arrows.add(arrow_idx)

        arrow_cells = arrow_sets[arrow_idx]
        assert _arrow_matches_direction(arrow_cells, root, direction), "arrow shape must point away from its rectangle"

        parsed.append(
            {
                **rect,
                "direction": direction,
                "root": root,
                "arrow_cells": arrow_cells,
            }
        )

    assert len(used_arrows) == len(arrow_sets), "all arrows must be assigned to rectangles"
    return parsed


def _shadow_capacity(obj: dict) -> int:
    return int(obj["h"] if obj["direction"] in ("left", "right") else obj["w"])


def _shadow_length(grid: np.ndarray, obj: dict) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = obj["r0"], obj["c0"], obj["r1"], obj["c1"]
    capacity = _shadow_capacity(obj)
    length = 0
    block_mask = (g != BG) & (g != MAROON)
    direction = obj["direction"]

    if direction == "right":
        for step in range(1, capacity + 1):
            col = c1 + step
            if col >= w or bool(np.any(block_mask[r0 : r1 + 1, col])):
                break
            length = step
    elif direction == "left":
        for step in range(1, capacity + 1):
            col = c0 - step
            if col < 0 or bool(np.any(block_mask[r0 : r1 + 1, col])):
                break
            length = step
    elif direction == "down":
        for step in range(1, capacity + 1):
            row = r1 + step
            if row >= h or bool(np.any(block_mask[row, c0 : c1 + 1])):
                break
            length = step
    else:
        for step in range(1, capacity + 1):
            row = r0 - step
            if row < 0 or bool(np.any(block_mask[row, c0 : c1 + 1])):
                break
            length = step

    return length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored rectangle with a grey directional shadow determined by its adjacent maroon arrow.

    Input structure:
    - The background is black.
    - Several non-black solid monochrome rectangles appear at different positions and sizes.
    - Each rectangle has a nearby maroon 3-pixel L-shaped arrow immediately outside one side, and that side determines one of the four cardinal push directions.

    Rule:
    - Keep every colored rectangle fixed in place.
    - Read the side containing the rectangle's arrow to determine the push direction.
    - Let D be the rectangle size perpendicular to that direction: height for left/right pushes and width for up/down pushes.
    - Replace all maroon arrows with black, then paint a grey rectangular shadow of length D outward from the pushed side.
    - If the next shadow slice would hit another colored rectangle, stop one cell before that obstacle instead of covering it.

    Key insight:
    - The arrow is only a direction marker; the hidden quantity is that the shadow length comes from the rectangle's crosswise size rather than from a fixed constant.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_scene(g)
    out = g.copy()
    out[out == MAROON] = BG

    for obj in objects:
        length = _shadow_length(g, obj)
        if length <= 0:
            continue
        r0, c0, r1, c1 = obj["r0"], obj["c0"], obj["r1"], obj["c1"]
        direction = obj["direction"]
        if direction == "right":
            out[r0 : r1 + 1, c1 + 1 : c1 + length + 1] = GREY
        elif direction == "left":
            out[r0 : r1 + 1, c0 - length : c0] = GREY
        elif direction == "down":
            out[r1 + 1 : r1 + length + 1, c0 : c1 + 1] = GREY
        else:
            out[r0 - length : r0, c0 : c1 + 1] = GREY

    return out


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expand(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _shadow_rect(obj: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = obj["r0"], obj["c0"], obj["r1"], obj["c1"]
    cap = _shadow_capacity(obj)
    if obj["direction"] == "right":
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c1 + 1, c1 + cap + 1)}
    if obj["direction"] == "left":
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0 - cap, c0)}
    if obj["direction"] == "down":
        return {(r, c) for r in range(r1 + 1, r1 + cap + 1) for c in range(c0, c1 + 1)}
    return {(r, c) for r in range(r0 - cap, r0) for c in range(c0, c1 + 1)}


def _render(objects: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for obj in objects:
        grid[obj["r0"] : obj["r1"] + 1, obj["c0"] : obj["c1"] + 1] = obj["color"]
        for r, c in obj["arrow_cells"]:
            grid[r, c] = MAROON
    return grid


def _pick_arrow_variant(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: str,
    grid_h: int,
    grid_w: int,
    forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    variants = _enumerate_arrow_variants(r0, c0, h, w, direction, grid_h, grid_w)
    random.shuffle(variants)
    for root, cells in variants:
        if not set(cells) & forbidden:
            return root, cells
    return None


def _make_object(
    color: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: str,
    grid_h: int,
    grid_w: int,
    forbidden: set[tuple[int, int]],
) -> dict | None:
    arrow = _pick_arrow_variant(r0, c0, h, w, direction, grid_h, grid_w, forbidden)
    if arrow is None:
        return None
    root, arrow_cells = arrow
    return {
        "color": color,
        "r0": r0,
        "c0": c0,
        "r1": r0 + h - 1,
        "c1": c0 + w - 1,
        "h": h,
        "w": w,
        "direction": direction,
        "root": root,
        "arrow_cells": set(arrow_cells),
    }


def _can_add_object(
    obj: dict,
    objects: list[dict],
    grid_h: int,
    grid_w: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    allow_shadow_overlap_with: int | None = None,
) -> bool:
    core = _rect_cells(obj["r0"], obj["c0"], obj["h"], obj["w"]) | obj["arrow_cells"]
    if core & occupied:
        return False
    if core & forbidden:
        return False

    obj_shadow = _shadow_rect(obj)
    for idx, other in enumerate(objects):
        other_core = _rect_cells(other["r0"], other["c0"], other["h"], other["w"]) | other["arrow_cells"]
        if obj_shadow & other["arrow_cells"]:
            return False
        if _shadow_rect(other) & obj["arrow_cells"]:
            return False
        if idx == allow_shadow_overlap_with:
            continue
        if obj_shadow & _shadow_rect(other):
            return False
        if obj_shadow & other_core:
            return False
        if _shadow_rect(other) & core:
            return False
    return True


def _place_random_object(
    color: int,
    objects: list[dict],
    grid_h: int,
    grid_w: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> dict | None:
    for _ in range(400):
        h = random.randint(2, min(6, grid_h - 4))
        w = random.randint(2, min(6, grid_w - 4))
        direction = random.choice(list(DIRS))
        r0 = random.randint(0, grid_h - h)
        c0 = random.randint(0, grid_w - w)
        obj = _make_object(color, r0, c0, h, w, direction, grid_h, grid_w, forbidden)
        if obj is None:
            continue
        if _shadow_capacity(obj) < 1:
            continue
        if not _can_add_object(obj, objects, grid_h, grid_w, occupied, forbidden):
            continue
        return obj
    return None


def _place_truncation_pair(
    colors: list[int],
    grid_h: int,
    grid_w: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[dict] | None:
    for _ in range(500):
        color_a, color_b = colors[:2]
        direction = random.choice(list(DIRS))
        if direction in ("left", "right"):
            h_a = random.randint(4, min(6, grid_h - 4))
            w_a = random.randint(2, min(5, grid_w - 6))
        else:
            h_a = random.randint(2, min(5, grid_h - 6))
            w_a = random.randint(4, min(6, grid_w - 4))

        capacity = h_a if direction in ("left", "right") else w_a
        if capacity < 4:
            continue
        block_step = random.randint(4, capacity)
        h_b = random.randint(2, min(5, grid_h - 4))
        w_b = random.randint(2, min(5, grid_w - 4))

        if direction == "right":
            max_r0 = grid_h - h_a
            max_c0 = grid_w - (w_a + block_step + w_b)
            if max_c0 < 0:
                continue
            r_a = random.randint(0, max_r0)
            c_a = random.randint(0, max_c0)
            lo = max(0, r_a - h_b + 1)
            hi = min(grid_h - h_b, r_a + h_a - 1)
            if lo > hi:
                continue
            r_b = random.randint(lo, hi)
            c_b = c_a + w_a + block_step - 1
        elif direction == "left":
            max_r0 = grid_h - h_a
            min_c0 = block_step + w_b
            if min_c0 > grid_w - w_a:
                continue
            r_a = random.randint(0, max_r0)
            c_a = random.randint(min_c0, grid_w - w_a)
            lo = max(0, r_a - h_b + 1)
            hi = min(grid_h - h_b, r_a + h_a - 1)
            if lo > hi:
                continue
            r_b = random.randint(lo, hi)
            c_b = c_a - block_step - w_b + 1
        elif direction == "down":
            max_r0 = grid_h - (h_a + block_step + h_b)
            max_c0 = grid_w - w_a
            if max_r0 < 0:
                continue
            r_a = random.randint(0, max_r0)
            c_a = random.randint(0, max_c0)
            lo = max(0, c_a - w_b + 1)
            hi = min(grid_w - w_b, c_a + w_a - 1)
            if lo > hi:
                continue
            c_b = random.randint(lo, hi)
            r_b = r_a + h_a + block_step - 1
        else:
            min_r0 = block_step + h_b
            max_c0 = grid_w - w_a
            if min_r0 > grid_h - h_a:
                continue
            r_a = random.randint(min_r0, grid_h - h_a)
            c_a = random.randint(0, max_c0)
            lo = max(0, c_a - w_b + 1)
            hi = min(grid_w - w_b, c_a + w_a - 1)
            if lo > hi:
                continue
            c_b = random.randint(lo, hi)
            r_b = r_a - block_step - h_b + 1

        obj_a = _make_object(color_a, r_a, c_a, h_a, w_a, direction, grid_h, grid_w, forbidden)
        if obj_a is None:
            continue
        if not _can_add_object(obj_a, [], grid_h, grid_w, occupied, forbidden):
            continue

        temp_core = _rect_cells(obj_a["r0"], obj_a["c0"], obj_a["h"], obj_a["w"]) | obj_a["arrow_cells"]
        temp_occupied = occupied | temp_core
        temp_forbidden = forbidden | _expand(temp_core, grid_h, grid_w)

        dirs_b = list(DIRS)
        random.shuffle(dirs_b)
        obj_b = None
        for direction_b in dirs_b:
            cand = _make_object(color_b, r_b, c_b, h_b, w_b, direction_b, grid_h, grid_w, temp_forbidden)
            if cand is None:
                continue
            if _can_add_object(cand, [obj_a], grid_h, grid_w, temp_occupied, temp_forbidden, allow_shadow_overlap_with=0):
                obj_b = cand
                break
        if obj_b is None:
            continue

        grid = _render([obj_a, obj_b], grid_h, grid_w)
        parsed = _parse_scene(grid)
        lengths = [_shadow_length(grid, obj) for obj in parsed]
        capacities = [_shadow_capacity(obj) for obj in parsed]
        if any(length <= 0 for length in lengths):
            continue
        if not any(length < cap for length, cap in zip(lengths, capacities)):
            continue
        if not any(length == cap for length, cap in zip(lengths, capacities)):
            continue
        return [obj_a, obj_b]

    return None


def _validate_scene(grid: np.ndarray, expect_truncation: bool) -> bool:
    try:
        objects = _parse_scene(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if np.any(out == MAROON):
        return False

    lengths = [_shadow_length(grid, obj) for obj in objects]
    capacities = [_shadow_capacity(obj) for obj in objects]
    if any(length <= 0 for length in lengths):
        return False
    if expect_truncation:
        if not any(length < cap for length, cap in zip(lengths, capacities)):
            return False
        if not any(length == cap for length, cap in zip(lengths, capacities)):
            return False
    else:
        if any(length < cap for length, cap in zip(lengths, capacities)):
            return False

    if not np.any(out == GREY):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid_h = random.randint(16, 28)
        grid_w = random.randint(16, 28)
        need_truncation = bool(random.getrandbits(1))
        n_objects = random.randint(3, 4) if need_truncation else random.randint(2, 4)
        colors = random.sample(BLOCK_COLORS, n_objects)

        objects: list[dict] = []
        occupied: set[tuple[int, int]] = set()
        forbidden: set[tuple[int, int]] = set()

        if need_truncation:
            pair = _place_truncation_pair(colors, grid_h, grid_w, occupied, forbidden)
            if pair is None:
                continue
            for obj in pair:
                core = _rect_cells(obj["r0"], obj["c0"], obj["h"], obj["w"]) | obj["arrow_cells"]
                objects.append(obj)
                occupied |= core
                forbidden |= _expand(core, grid_h, grid_w)

        remaining_colors = colors[len(objects) :]
        ok = True
        for color in remaining_colors:
            obj = _place_random_object(color, objects, grid_h, grid_w, occupied, forbidden)
            if obj is None:
                ok = False
                break
            core = _rect_cells(obj["r0"], obj["c0"], obj["h"], obj["w"]) | obj["arrow_cells"]
            objects.append(obj)
            occupied |= core
            forbidden |= _expand(core, grid_h, grid_w)
        if not ok:
            continue

        grid = _render(objects, grid_h, grid_w)
        if _validate_scene(grid, need_truncation):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")

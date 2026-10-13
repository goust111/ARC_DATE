import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
INPUT_COLORS = (1, 3, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _object_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid != BLACK)


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _object_components(g)

    cover = np.zeros((h, w), dtype=int)
    for comp in objects:
        border: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BLACK:
                    border.add((nr, nc))
        for r, c in border:
            cover[r, c] += 1

    bordered = g.copy()
    black_mask = g == BLACK
    bordered[black_mask & (cover == 1)] = GREY
    bordered[black_mask & (cover >= 2)] = RED

    enclosed: list[list[tuple[int, int]]] = []
    out = bordered.copy()
    for comp in _components_from_mask(out == BLACK):
        touches_edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)
        if not touches_edge:
            enclosed.append(comp)
            for r, c in comp:
                out[r, c] = YELLOW

    return {
        "objects": objects,
        "cover": cover,
        "bordered": bordered,
        "enclosed": enclosed,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each foreground object by one cell, mark border overlaps red, and fill emergent enclosed black holes yellow.

    Input structure:
    - The grid has a black background with 3-5 small 4-connected monochrome objects.
    - Object colors are ordinary foreground colors; black, grey, red, and yellow are reserved for the transformation.
    - Objects are separated in the input, but some are close enough that their one-cell expansions will overlap.

    Rule:
    - For each foreground object, collect every black cell that is 4-neighbor-adjacent to at least one pixel of that object; this is that object's one-cell expansion border.
    - If a black cell belongs to exactly one object's border, recolor it grey; if it belongs to two or more objects' borders, recolor it red.
    - Keep all original object pixels unchanged.
    - After all borders are drawn, find every remaining 4-connected black region; any such region that cannot reach a black cell on the outer grid edge through black 4-neighbor paths is enclosed and must be filled yellow.

    Key insight:
    - Yellow holes are not owned by any single object: they appear only after several independent one-cell borders interact to seal off a black region, with red overlap cells revealing where multiple borders cooperate.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(base: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    pts = set(base)
    for _ in range(rot % 4):
        pts = {(c, -r) for r, c in pts}
    if flip:
        pts = {(r, -c) for r, c in pts}
    return _normalize(pts)


EXTRA_SHAPES = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
]


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _add_outward_tail(
    cells: set[tuple[int, int]],
    direction: tuple[int, int],
    occupied_other: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    max_extra = min(2, 8 - len(cells))
    if max_extra <= 0:
        return set(cells)
    extra_len = random.choices([0, 1, 2][: max_extra + 1], weights=[5, 4, 2][: max_extra + 1])[0]
    if extra_len == 0:
        return set(cells)
    bases = list(cells)
    random.shuffle(bases)
    dr, dc = direction
    for br, bc in bases:
        new_cells = []
        cur = (br, bc)
        ok = True
        for _ in range(extra_len):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in cells or cur in occupied_other:
                ok = False
                break
            if not (0 <= cur[0] < shape[0] and 0 <= cur[1] < shape[1]):
                ok = False
                break
            for nr, nc in _neighbors4(*cur):
                if (nr, nc) in occupied_other:
                    ok = False
                    break
            if not ok:
                break
            new_cells.append(cur)
        if ok:
            return set(cells) | set(new_cells)
    return set(cells)


def _ring_box(h: int, w: int, hole_h: int, hole_w: int) -> tuple[int, int, int, int]:
    r1 = random.randint(2, h - hole_h - 4)
    c1 = random.randint(2, w - hole_w - 4)
    r2 = r1 + hole_h + 1
    c2 = c1 + hole_w + 1
    return r1, c1, r2, c2


def _core_objects_four(
    h: int, w: int, hole_h: int, hole_w: int
) -> tuple[list[tuple[str, set[tuple[int, int]], tuple[int, int]]], tuple[int, int, int, int]]:
    r1, c1, r2, c2 = _ring_box(h, w, hole_h, hole_w)
    objects = [
        ("top", {(r1 - 1, c) for c in range(c1, c2 + 1)}, (-1, 0)),
        ("bottom", {(r2 + 1, c) for c in range(c1, c2 + 1)}, (1, 0)),
        ("left", {(r, c1 - 1) for r in range(r1, r2 + 1)}, (0, -1)),
        ("right", {(r, c2 + 1) for r in range(r1, r2 + 1)}, (0, 1)),
    ]
    return objects, (r1, c1, r2, c2)


def _core_objects_three(
    h: int, w: int, hole_h: int, hole_w: int
) -> tuple[list[tuple[str, set[tuple[int, int]], tuple[int, int]]], tuple[int, int, int, int]]:
    r1, c1, r2, c2 = _ring_box(h, w, hole_h, hole_w)
    corner = random.choice(["tl", "tr", "bl", "br"])
    objects: list[tuple[str, set[tuple[int, int]], tuple[int, int]]] = []
    if corner == "tl":
        merged = {(r1 - 1, c) for c in range(c1, c2 + 1)} | {(r, c1 - 1) for r in range(r1 - 1, r2 + 1)}
        objects = [
            ("corner", merged, (-1, 0)),
            ("bottom", {(r2 + 1, c) for c in range(c1, c2 + 1)}, (1, 0)),
            ("right", {(r, c2 + 1) for r in range(r1, r2 + 1)}, (0, 1)),
        ]
    elif corner == "tr":
        merged = {(r1 - 1, c) for c in range(c1, c2 + 1)} | {(r, c2 + 1) for r in range(r1 - 1, r2 + 1)}
        objects = [
            ("corner", merged, (-1, 0)),
            ("bottom", {(r2 + 1, c) for c in range(c1, c2 + 1)}, (1, 0)),
            ("left", {(r, c1 - 1) for r in range(r1, r2 + 1)}, (0, -1)),
        ]
    elif corner == "bl":
        merged = {(r2 + 1, c) for c in range(c1, c2 + 1)} | {(r, c1 - 1) for r in range(r1, r2 + 2)}
        objects = [
            ("corner", merged, (1, 0)),
            ("top", {(r1 - 1, c) for c in range(c1, c2 + 1)}, (-1, 0)),
            ("right", {(r, c2 + 1) for r in range(r1, r2 + 1)}, (0, 1)),
        ]
    else:
        merged = {(r2 + 1, c) for c in range(c1, c2 + 1)} | {(r, c2 + 1) for r in range(r1, r2 + 2)}
        objects = [
            ("corner", merged, (1, 0)),
            ("top", {(r1 - 1, c) for c in range(c1, c2 + 1)}, (-1, 0)),
            ("left", {(r, c1 - 1) for r in range(r1, r2 + 1)}, (0, -1)),
        ]
    return objects, (r1, c1, r2, c2)


def _place_extra_object(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    avoid_bbox: tuple[int, int, int, int],
) -> set[tuple[int, int]] | None:
    ar0, ac0, ar1, ac1 = avoid_bbox
    candidates = []
    for base in EXTRA_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                candidates.append(_transform_shape(base, rot, flip))
    random.shuffle(candidates)

    for shape_cells in candidates:
        sr0, sc0, sr1, sc1 = _bbox(shape_cells)
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        if sh >= h or sw >= w:
            continue
        origins = [
            (r0, c0)
            for r0 in range(h - sh + 1)
            for c0 in range(w - sw + 1)
        ]
        random.shuffle(origins)
        for r0, c0 in origins[:80]:
            placed = _shift(shape_cells, r0, c0)
            pr0, pc0, pr1, pc1 = _bbox(placed)
            far_enough = (
                pr1 < ar0 - 2
                or pr0 > ar1 + 2
                or pc1 < ac0 - 2
                or pc0 > ac1 + 2
            )
            if far_enough and _can_place(placed, occupied, (h, w)):
                return placed
    return None


def _render_objects(shape: tuple[int, int], objects: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    colors = random.sample(list(INPUT_COLORS), len(objects))
    for color, cells in zip(colors, objects):
        for r, c in cells:
            grid[r, c] = color
    return grid


def _input_semantics_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if any(int(v) in {RED, YELLOW, GREY} for v in np.unique(g)):
        return False

    objects = _object_components(g)
    if not (3 <= len(objects) <= 5):
        return False

    occupied_colors = []
    for comp in objects:
        if not (3 <= len(comp) <= 8):
            return False
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            return False
        occupied_colors.extend(colors)
    if len(set(occupied_colors)) != len(occupied_colors):
        return False

    obj_sets = [set(comp) for comp in objects]
    for i in range(len(obj_sets)):
        for j in range(i + 1, len(obj_sets)):
            for r, c in obj_sets[i]:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in obj_sets[j]:
                        return False

    details = _solve_details(g)
    cover = details["cover"]
    bordered = details["bordered"]
    out = details["output"]
    enclosed = details["enclosed"]

    if not np.any(cover >= 2):
        return False
    if not np.any(cover == 1):
        return False
    if len(enclosed) < 1:
        return False
    if not np.any(out == YELLOW):
        return False

    black_mask = g == BLACK
    if np.any((bordered == GREY) != (black_mask & (cover == 1))):
        return False
    if np.any((bordered == RED) != (black_mask & (cover >= 2))):
        return False

    yellow_mask = out == YELLOW
    expected_yellow = np.zeros_like(yellow_mask, dtype=bool)
    for comp in enclosed:
        for r, c in comp:
            expected_yellow[r, c] = True
    if not np.array_equal(yellow_mask, expected_yellow):
        return False

    return True


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    target_count = random.choice([3, 4, 5])

    if target_count == 3:
        hole_h = random.choice([1, 2])
        hole_w = random.choice([1, 2, 3])
        if hole_h + hole_w > 4:
            hole_h, hole_w = 2, 2
        core_spec, ring = _core_objects_three(h, w, hole_h, hole_w)
        want_extra = False
    elif target_count == 4 and random.random() < 0.45:
        hole_h = random.randint(1, 2)
        hole_w = random.randint(1, 3)
        if hole_h + hole_w > 4:
            hole_h, hole_w = 1, 3
        core_spec, ring = _core_objects_three(h, w, hole_h, hole_w)
        want_extra = True
    else:
        hole_h = random.randint(1, 3)
        hole_w = random.randint(1, 3)
        core_spec, ring = _core_objects_four(h, w, hole_h, hole_w)
        want_extra = target_count == 5

    objects = [set(cells) for _, cells, _ in core_spec]
    occupied = set().union(*objects) if objects else set()

    for idx, (name, _cells, outward_dir) in enumerate(core_spec):
        if name == "corner":
            continue
        others = set().union(*(objects[j] for j in range(len(objects)) if j != idx))
        objects[idx] = _add_outward_tail(objects[idx], outward_dir, others, (h, w))

    occupied = set().union(*objects) if objects else set()
    if want_extra:
        extra = _place_extra_object(h, w, occupied, ring)
        if extra is None:
            return None
        objects.append(extra)

    grid = _render_objects((h, w), objects)
    return grid if _input_semantics_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")

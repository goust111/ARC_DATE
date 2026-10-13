import random
from collections import defaultdict, deque

import numpy as np


BG = 0
ARC_MAX = 30
COLORS = tuple(range(1, 10))
RNG = random.Random()

PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 7
MIN_RECT_AREA = 6
MAX_GENERATION_ATTEMPTS = 400
PARTITION_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "cells": frozenset(cells)})
    return components


def _normalize(cells: frozenset[tuple[int, int]] | set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    pts = list(cells)
    if not pts:
        return frozenset(), 0, 0
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = frozenset((int(r - min_r), int(c - min_c)) for r, c in pts)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, int(height), int(width)


def _is_rectangle(cells: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    norm, h, w = _normalize(cells)
    return len(norm) == h * w


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((int(h), int(w)))
        if h != w:
            pairs.append((int(w), int(h)))
    return sorted(set(pairs))


def _fits_rectangle(
    shape_a: frozenset[tuple[int, int]],
    shape_b: frozenset[tuple[int, int]],
    rect_h: int,
    rect_w: int,
) -> bool:
    norm_a, ha, wa = _normalize(shape_a)
    norm_b, hb, wb = _normalize(shape_b)
    if ha > rect_h or wa > rect_w or hb > rect_h or wb > rect_w:
        return False

    rect_cells = frozenset((r, c) for r in range(rect_h) for c in range(rect_w))
    for top in range(rect_h - ha + 1):
        for left in range(rect_w - wa + 1):
            placed_a = frozenset((r + top, c + left) for r, c in norm_a)
            complement = rect_cells - placed_a
            comp_norm, _, _ = _normalize(complement)
            if comp_norm == norm_b:
                return True
    return False


def _rectangle_candidates(
    shape_a: frozenset[tuple[int, int]],
    shape_b: frozenset[tuple[int, int]],
) -> list[tuple[int, int]]:
    total_area = len(shape_a) + len(shape_b)
    candidates: list[tuple[int, int]] = []
    for rect_h, rect_w in _factor_pairs(total_area):
        if _fits_rectangle(shape_a, shape_b, rect_h, rect_w) or _fits_rectangle(shape_b, shape_a, rect_h, rect_w):
            candidates.append((int(rect_h), int(rect_w)))
    return sorted(set(candidates), key=lambda dims: (dims[0] + dims[1], max(dims), dims[0], dims[1]))


def _infer_rectangle_dims(
    shape_a: frozenset[tuple[int, int]],
    shape_b: frozenset[tuple[int, int]],
) -> tuple[int, int]:
    candidates = _rectangle_candidates(shape_a, shape_b)
    if not candidates:
        raise AssertionError("same-color fragments cannot be merged into a solid rectangle by translation")
    return candidates[0]


def _solve_specs(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    components = _find_components(grid)
    by_color: dict[int, list[frozenset[tuple[int, int]]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp["cells"])

    specs: list[tuple[int, int, int, int]] = []
    for color, pieces in by_color.items():
        if len(pieces) != 2:
            raise AssertionError(f"color {color} must appear as exactly two fragments")
        rect_h, rect_w = _infer_rectangle_dims(pieces[0], pieces[1])
        specs.append((int(rect_h * rect_w), int(rect_h), int(rect_w), int(color)))
    return sorted(specs, key=lambda item: (item[0], item[1], item[2], item[3]))


def _render_output(specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    out_h = sum(rect_h for _, rect_h, _, _ in specs)
    out_w = max(rect_w for _, _, rect_w, _ in specs)
    out = np.zeros((out_h, out_w), dtype=int)
    bottom = out_h
    for _, rect_h, rect_w, color in specs:
        top = bottom - rect_h
        left = (out_w - rect_w) // 2
        out[top:bottom, left : left + rect_w] = int(color)
        bottom = top
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, recover their completed rectangles, and stack those rectangles by area.

    Input structure:
    - The background is black.
    - Non-black cells form an even number of 4-connected monochromatic fragments.
    - Each color appears in exactly two fragments, and those two fragments can be translated to tile one solid rectangle without overlap.

    Rule:
    - For each color, find its two fragments and infer the height and width of the solid rectangle they jointly complete.
    - Sort the completed rectangles by increasing area.
    - Build a black output grid whose width is the maximum rectangle width and whose height is the sum of all rectangle heights.
    - Stack the rectangles with the smallest area at the bottom and larger ones directly above it, centering each rectangle horizontally and biasing one cell to the left when exact centering is impossible.

    Key insight:
    - The scattered fragments are not arbitrary sprites; each same-colored pair is a complementary partition of one solid rectangle, and only the recovered rectangle areas determine the final stacking order.
    """
    g = np.asarray(grid, dtype=int)
    specs = _solve_specs(g)
    return _render_output(specs)


def _transform_pair(
    frag_a: frozenset[tuple[int, int]],
    frag_b: frozenset[tuple[int, int]],
    rect_h: int,
    rect_w: int,
    op: int,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]], int, int]:
    def transform(cells: frozenset[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
        if op == 0:
            mapped = frozenset((r, c) for r, c in cells)
            return mapped, rect_h, rect_w
        if op == 1:
            mapped = frozenset((c, rect_h - 1 - r) for r, c in cells)
            return mapped, rect_w, rect_h
        if op == 2:
            mapped = frozenset((rect_h - 1 - r, rect_w - 1 - c) for r, c in cells)
            return mapped, rect_h, rect_w
        if op == 3:
            mapped = frozenset((rect_w - 1 - c, r) for r, c in cells)
            return mapped, rect_w, rect_h
        if op == 4:
            mapped = frozenset((r, rect_w - 1 - c) for r, c in cells)
            return mapped, rect_h, rect_w
        if op == 5:
            mapped = frozenset((rect_h - 1 - r, c) for r, c in cells)
            return mapped, rect_h, rect_w
        if op == 6:
            mapped = frozenset((c, r) for r, c in cells)
            return mapped, rect_w, rect_h
        mapped = frozenset((rect_w - 1 - c, rect_h - 1 - r) for r, c in cells)
        return mapped, rect_w, rect_h

    mapped_a, out_h, out_w = transform(frag_a)
    mapped_b, _, _ = transform(frag_b)
    return mapped_a, mapped_b, int(out_h), int(out_w)


def _random_spanning_tree(rect_h: int, rect_w: int, rng: random.Random) -> tuple[tuple[int, int], dict[tuple[int, int], list[tuple[int, int]]]]:
    root = (rng.randrange(rect_h), rng.randrange(rect_w))
    seen = {root}
    stack = [root]
    children: dict[tuple[int, int], list[tuple[int, int]]] = {(r, c): [] for r in range(rect_h) for c in range(rect_w)}
    while stack:
        cell = stack[-1]
        candidates = []
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < rect_h and 0 <= nc < rect_w and (nr, nc) not in seen:
                candidates.append((nr, nc))
        if not candidates:
            stack.pop()
            continue
        nxt = rng.choice(candidates)
        seen.add(nxt)
        children[cell].append(nxt)
        stack.append(nxt)
    return root, children


def _random_partition(rect_h: int, rect_w: int, rng: random.Random) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    all_cells = frozenset((r, c) for r in range(rect_h) for c in range(rect_w))
    for _ in range(PARTITION_ATTEMPTS):
        root, children = _random_spanning_tree(rect_h, rect_w, rng)
        subtrees: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

        def build(node: tuple[int, int]) -> frozenset[tuple[int, int]]:
            cells = {node}
            rng.shuffle(children[node])
            for child in children[node]:
                cells |= build(child)
            result = frozenset(cells)
            subtrees[node] = result
            return result

        build(root)
        options: list[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]] = []
        for node, subset in subtrees.items():
            if node == root:
                continue
            complement = all_cells - subset
            for frag_a, frag_b in ((subset, complement), (complement, subset)):
                if len(frag_a) < 2 or len(frag_b) < 2:
                    continue
                if _is_rectangle(frag_a) or _is_rectangle(frag_b):
                    continue
                candidates = _rectangle_candidates(frag_a, frag_b)
                if set(candidates) == {(rect_h, rect_w)}:
                    options.append((frag_a, frag_b))
        if options:
            return rng.choice(options)
    raise RuntimeError("failed to create a valid complementary partition")


def _sample_rectangles(rng: random.Random) -> list[dict]:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    colors = rng.sample(COLORS, pair_count)
    used_areas: set[int] = set()
    rectangles: list[dict] = []
    attempts = 0
    while len(rectangles) < pair_count and attempts < 500:
        attempts += 1
        rect_h = rng.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        rect_w = rng.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        area = rect_h * rect_w
        if area < MIN_RECT_AREA or area in used_areas:
            continue
        try:
            frag_a, frag_b = _random_partition(rect_h, rect_w, rng)
        except RuntimeError:
            continue
        op = rng.randrange(8)
        mapped_a, mapped_b, out_h, out_w = _transform_pair(frag_a, frag_b, rect_h, rect_w, op)
        candidates = _rectangle_candidates(mapped_a, mapped_b)
        if set(candidates) != {(out_h, out_w)}:
            continue
        norm_a, ah, aw = _normalize(mapped_a)
        norm_b, bh, bw = _normalize(mapped_b)
        rectangles.append(
            {
                "color": int(colors[len(rectangles)]),
                "area": int(area),
                "rect_h": int(out_h),
                "rect_w": int(out_w),
                "fragments": [
                    {"cells": norm_a, "h": int(ah), "w": int(aw)},
                    {"cells": norm_b, "h": int(bh), "w": int(bw)},
                ],
            }
        )
        used_areas.add(area)
    if len(rectangles) != pair_count:
        raise RuntimeError("failed to sample enough rectangle pairs")
    if sum(item["rect_h"] for item in rectangles) > ARC_MAX:
        raise RuntimeError("stacked output would exceed ARC height")
    return rectangles


def _mark_halo(blocked: np.ndarray, r: int, c: int) -> None:
    h, w = blocked.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_fragments(rectangles: list[dict], rng: random.Random) -> np.ndarray | None:
    objects = []
    for item in rectangles:
        for frag in item["fragments"]:
            objects.append(
                {
                    "color": int(item["color"]),
                    "cells": frag["cells"],
                    "h": int(frag["h"]),
                    "w": int(frag["w"]),
                }
            )
    rng.shuffle(objects)

    max_obj_h = max(obj["h"] for obj in objects)
    max_obj_w = max(obj["w"] for obj in objects)
    min_h = max(10, max_obj_h + 4, len(objects) + 6)
    min_w = max(10, max_obj_w + 4, len(objects) + 6)

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = rng.randint(min_h, ARC_MAX)
        grid_w = rng.randint(min_w, ARC_MAX)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        ok = True

        for obj in objects:
            positions: list[tuple[int, int]] = []
            for top in range(grid_h - obj["h"] + 1):
                for left in range(grid_w - obj["w"] + 1):
                    if all(not blocked[top + r, left + c] for r, c in obj["cells"]):
                        positions.append((top, left))
            if not positions:
                ok = False
                break

            top, left = rng.choice(positions)
            for r, c in obj["cells"]:
                rr = top + r
                cc = left + c
                grid[rr, cc] = int(obj["color"])
            for r, c in obj["cells"]:
                _mark_halo(blocked, top + r, left + c)

        if ok:
            return grid
    return None


def _expected_specs(rectangles: list[dict]) -> list[tuple[int, int, int, int]]:
    return sorted(
        [(item["area"], item["rect_h"], item["rect_w"], item["color"]) for item in rectangles],
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )


def _is_valid_generated_grid(grid: np.ndarray, rectangles: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > ARC_MAX:
        return False
    components = _find_components(grid)
    by_color: dict[int, int] = defaultdict(int)
    for comp in components:
        by_color[int(comp["color"])] += 1
        if _is_rectangle(comp["cells"]):
            return False
    if any(count != 2 for count in by_color.values()):
        return False
    expected = _expected_specs(rectangles)
    solved = _solve_specs(grid)
    if solved != expected:
        return False
    out = _render_output(solved)
    if out.ndim != 2 or max(out.shape) > ARC_MAX:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    try:
        rectangles = _sample_rectangles(rng)
    except RuntimeError:
        return None
    grid = _place_fragments(rectangles, rng)
    if grid is None:
        return None
    if not _is_valid_generated_grid(grid, rectangles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")

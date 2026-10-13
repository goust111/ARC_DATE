from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _normalize_cells(cells: set[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "rel": frozenset((r - r0, c - c0) for r, c in cells),
    }


def _cells_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    comp = _normalize_cells(cells)
    h, w = comp["shape"]
    patch = np.zeros((h, w), dtype=int)
    for r, c in comp["rel"]:
        patch[r, c] = int(color)
    return patch


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()

            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comp = _normalize_cells(cells)
            comp["color"] = color
            components.append(comp)

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return components


def _rectangle_dimensions(comp_a: dict[str, object], comp_b: dict[str, object]) -> set[tuple[int, int]]:
    rel_a = comp_a["rel"]
    rel_b = comp_b["rel"]
    h1, w1 = comp_a["shape"]
    h2, w2 = comp_b["shape"]
    dims: set[tuple[int, int]] = set()

    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            shifted_b = {(r + dr, c + dc) for r, c in rel_b}
            if rel_a & shifted_b:
                continue

            union = set(rel_a)
            union.update(shifted_b)
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            rect_h = max(rows) - min(rows) + 1
            rect_w = max(cols) - min(cols) + 1
            if len(union) == rect_h * rect_w:
                dims.add((rect_h, rect_w))

    return dims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, complete their hidden rectangles, and pack the filled rectangles by area.

    Input structure:
    - The background is black.
    - The foreground consists of monochrome 4-connected fragments scattered across the grid.
    - Each color appears on exactly two fragments, and that pair is complementary under translation without rotation or reflection.

    Rule:
    - For each color, find the two fragments of that color.
    - Translate one fragment relative to the other until their union becomes a solid axis-aligned rectangle, with no overlap.
    - Replace that fragment pair by the fully filled rectangle of the same color.
    - Sort the completed rectangles by area from small to large, then place them left-to-right with one black column between neighbors and their bottom edges aligned.

    Key insight:
    - The scattered objects must first be matched by color and shape complementarity; only after reconstructing those rectangles can the final area-based ordering be read.
    """
    g = np.asarray(grid, dtype=int)
    components = _find_components(g)
    assert components, "expected at least one foreground fragment"

    grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    color_order: list[int] = []
    for comp in components:
        color = int(comp["color"])
        if color not in grouped:
            color_order.append(color)
        grouped[color].append(comp)

    rectangles: list[dict[str, int]] = []
    for color in color_order:
        pair = grouped[color]
        assert len(pair) == 2, "each color must correspond to exactly one fragment pair"
        dims = _rectangle_dimensions(pair[0], pair[1])
        assert len(dims) == 1, "fragment pair must reconstruct a unique rectangle"
        rect_h, rect_w = next(iter(dims))
        rectangles.append({"color": color, "h": rect_h, "w": rect_w, "area": rect_h * rect_w})

    rectangles.sort(key=lambda rect: rect["area"])
    out_h = max(rect["h"] for rect in rectangles)
    out_w = sum(rect["w"] for rect in rectangles) + len(rectangles) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for rect in rectangles:
        out[out_h - rect["h"] :, col : col + rect["w"]] = int(rect["color"])
        col += rect["w"] + 1

    return out


# --------------- generator ---------------

PAIR_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (3, 6)
RECT_WIDTH_RANGE = (3, 7)
CANVAS_LIMIT = 30
MIN_CANVAS_SIDE = 12
PROFILE_STEPS = (-2, -1, 0, 1, 2)
SPLIT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 300


def _random_profile(length: int, low: int, high: int) -> list[int]:
    values = [random.randint(low, high)]
    for _ in range(1, length):
        prev = values[-1]
        options = []
        for step in PROFILE_STEPS:
            cur = prev + step
            if low <= cur <= high:
                options.append(cur)
        values.append(random.choice(options))

    if len(set(values)) == 1:
        idx = random.randrange(length)
        alternatives = [v for v in range(low, high + 1) if v != values[idx]]
        values[idx] = random.choice(alternatives)
    return values


def _split_rectangle(rect_h: int, rect_w: int, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    full = {(r, c) for r in range(rect_h) for c in range(rect_w)}

    for _ in range(SPLIT_ATTEMPTS):
        if random.random() < 0.5:
            profile = _random_profile(rect_h, 1, rect_w - 1)
            first = {(r, c) for r in range(rect_h) for c in range(profile[r])}
        else:
            profile = _random_profile(rect_w, 1, rect_h - 1)
            first = {(r, c) for c in range(rect_w) for r in range(profile[c])}

        second = full - first
        if random.random() < 0.5:
            first, second = second, first

        comp_a = _normalize_cells(first)
        comp_b = _normalize_cells(second)
        if _rectangle_dimensions(comp_a, comp_b) != {(rect_h, rect_w)}:
            continue

        return _cells_to_patch(first, color), _cells_to_patch(second, color)

    return None


def _sample_rectangles(pair_count: int) -> list[dict[str, int]] | None:
    for _ in range(200):
        rectangles: list[dict[str, int]] = []
        used_areas: set[int] = set()
        for _ in range(pair_count):
            rect_h = random.randint(*RECT_HEIGHT_RANGE)
            rect_w = random.randint(*RECT_WIDTH_RANGE)
            area = rect_h * rect_w
            if area in used_areas:
                break
            rectangles.append({"h": rect_h, "w": rect_w, "area": area})
            used_areas.add(area)
        else:
            out_w = sum(rect["w"] for rect in rectangles) + pair_count - 1
            if out_w <= CANVAS_LIMIT:
                return rectangles
    return None


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    patch_cells = np.argwhere(patch != BLACK)
    for rr, cc in patch_cells:
        r = top + int(rr)
        c = left + int(cc)
        if r < 0 or c < 0 or r >= blocked.shape[0] or c >= blocked.shape[1]:
            return False
        if blocked[r, c]:
            return False
    return True


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    shapes = [tuple(map(int, patch.shape)) for patch in patches]
    min_h = max(h for h, _ in shapes) + 2
    min_w = max(w for _, w in shapes) + 2
    padded_area = sum((h + 2) * (w + 2) for h, w in shapes)

    if min_h > CANVAS_LIMIT or min_w > CANVAS_LIMIT:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = random.randint(max(MIN_CANVAS_SIDE, min_h), CANVAS_LIMIT)
        grid_w = random.randint(max(MIN_CANVAS_SIDE, min_w), CANVAS_LIMIT)
        if grid_h * grid_w < int(1.2 * padded_area):
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

        for idx in order:
            patch = patches[idx]
            h, w = patch.shape
            positions: list[tuple[int, int]] = []
            for top in range(grid_h - h + 1):
                for left in range(grid_w - w + 1):
                    if _can_place(blocked, patch, top, left):
                        positions.append((top, left))

            if not positions:
                break

            top, left = random.choice(positions)
            for rr, cc in np.argwhere(patch != BLACK):
                r = top + int(rr)
                c = left + int(cc)
                grid[r, c] = int(patch[rr, cc])
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        blocked[nr, nc] = True
        else:
            return grid

    return None


def _build_expected(rectangles: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda rect: rect["area"])
    out_h = max(rect["h"] for rect in ordered)
    out_w = sum(rect["w"] for rect in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for rect in ordered:
        out[out_h - rect["h"] :, col : col + rect["w"]] = int(rect["color"])
        col += rect["w"] + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(range(1, 10), pair_count)
    rectangles = _sample_rectangles(pair_count)
    if rectangles is None:
        return None

    patches: list[np.ndarray] = []
    for rect, color in zip(rectangles, colors):
        rect["color"] = int(color)
        split = _split_rectangle(rect["h"], rect["w"], int(color))
        if split is None:
            return None
        patch_a, patch_b = split
        patches.extend([patch_a, patch_b])

    grid = _place_patches(patches)
    if grid is None:
        return None

    expected = _build_expected(rectangles)
    actual = main(grid)
    if actual.shape != expected.shape:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

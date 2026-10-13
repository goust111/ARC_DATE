from __future__ import annotations

import math
import random
from itertools import combinations, permutations

import numpy as np


BG = 0
RED = 2
DIVIDER = 5
COLOR_POOL = [1, 3, 4, 6, 7, 8, 9]
MIN_AREA_FOR_HOLES = {0: 4, 1: 8, 2: 19, 3: 30}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _count_enclosed_bg_pixels(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return int(np.count_nonzero((~mask) & (~seen)))


def _best_rect_dims(area: int) -> tuple[int, int]:
    best_h = 1
    for h in range(1, math.isqrt(area) + 1):
        if area % h == 0:
            best_h = h
    return best_h, area // best_h


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            coords = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _count_object_holes(grid: np.ndarray, obj: dict) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in obj["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return _count_enclosed_bg_pixels(mask)


def _parse_legend(right: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for r in range(right.shape[0]):
        row = right[r]
        reds = int(np.count_nonzero(row == RED))
        colors = sorted(set(map(int, np.unique(row))) - {BG, RED})
        if len(colors) == 1:
            color = colors[0]
            assert reds not in mapping, "hole-count legend must be unique"
            mapping[reds] = color
    assert mapping, "expected at least one legend row"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Map each object's hole count through the legend, then redraw the objects as nested centered rectangles ordered by area.

    Input structure:
    - A full gray divider column splits the grid into a left object field and a right legend field.
    - The left field has black background and several separated monochrome 4-connected foreground objects.
    - Objects may contain zero or more fully enclosed black single-pixel holes.
    - The right field is a hole-count legend: in each used row, red dots encode the count and one colored cell gives the output color.

    Rule:
    - Read the right legend as a mapping from red-dot count to target color.
    - For every left-side object, count its enclosed black hole pixels and look up the corresponding target color.
    - Sort the objects by foreground area from large to small.
    - Return only the right-field canvas size, drawing centered filled rectangles in that order so each rectangle area equals the source object area and uses the mapped color.

    Key insight:
    - The source object color is irrelevant: the visible red-dot legend converts hole count into the repaint color, and the actual redraw geometry depends only on area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == DIVIDER)]
    assert len(divider_cols) == 1, "expected one full gray divider column"
    divider = divider_cols[0]

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.size > 0 and right.size > 0

    legend = _parse_legend(right)
    objects = _same_color_components(left)
    assert objects, "expected foreground objects in the left field"

    specs = []
    for obj in objects:
        holes = _count_object_holes(left, obj)
        assert holes in legend, "every hole count must appear in the legend"
        specs.append((int(obj["area"]), int(legend[holes])))

    out_h, out_w = right.shape
    out = np.full((out_h, out_w), BG, dtype=int)
    for area, color in sorted(specs, key=lambda item: item[0], reverse=True):
        rect_h, rect_w = _best_rect_dims(area)
        assert rect_h <= out_h and rect_w <= out_w, "rectangle must fit in the legend-sized output"
        r0 = (out_h - rect_h) // 2
        c0 = (out_w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return out.copy()


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(mask), strict=False):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _initial_mask(holes: int) -> np.ndarray:
    if holes == 0:
        seeds = [
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
            np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        ]
        return random.choice(seeds).copy()

    width = 4 * holes - 1
    mask = np.ones((3, width), dtype=bool)
    for i in range(holes):
        mask[1, 1 + 4 * i] = False
    return mask


def _make_object_mask(area: int, holes: int) -> np.ndarray | None:
    if holes == 0:
        rect_h, rect_w = _best_rect_dims(area)
        return _random_transform(np.ones((rect_h, rect_w), dtype=bool))

    min_w = 4 * holes - 1
    base_area = 3 * min_w - holes
    if base_area > area:
        return None

    target_cells = area + holes
    widths = list(range(min_w, min(18, target_cells) + 1))
    random.shuffle(widths)
    for width in widths:
        height = max(3, int(math.ceil(target_cells / width)))
        if height > 16:
            continue
        surplus = height * width - holes - area
        if surplus < 0 or surplus >= width:
            continue
        if height == 3 and surplus:
            continue

        mask = np.ones((height, width), dtype=bool)
        for i in range(holes):
            mask[1, 1 + 4 * i] = False
        if surplus:
            mask[height - 1, width - surplus : width] = False

        if int(mask.sum()) != area:
            continue
        if _count_enclosed_bg_pixels(mask) != holes:
            continue
        return _random_transform(mask)
    return None


def _candidate_rectangles(out_h: int, out_w: int) -> list[tuple[int, int, int]]:
    rects = []
    for area in range(4, out_h * out_w + 1):
        rect_h, rect_w = _best_rect_dims(area)
        if 2 <= rect_h <= out_h - 1 and 2 <= rect_w <= out_w - 1:
            rects.append((area, rect_h, rect_w))
    rects.sort(key=lambda item: item[0], reverse=True)
    return rects


def _choose_nested_rectangles(out_h: int, out_w: int, count: int) -> list[tuple[int, int, int]] | None:
    rects = _candidate_rectangles(out_h, out_w)
    if not rects:
        return None

    for _ in range(240):
        outer_choices = [rect for rect in rects if rect[1] >= max(3, out_h // 2) and rect[2] >= max(4, out_w // 2)]
        if not outer_choices:
            outer_choices = rects
        sequence = [random.choice(outer_choices)]
        while len(sequence) < count:
            prev_area, prev_h, prev_w = sequence[-1]
            choices = [
                rect
                for rect in rects
                if rect[0] < prev_area and rect[1] < prev_h and rect[2] < prev_w
            ]
            if not choices:
                break
            limit = max(1, min(len(choices), 12))
            sequence.append(random.choice(choices[:limit]))
        if len(sequence) == count:
            return sequence
    return None


def _assign_distinct_holes(areas: list[int]) -> tuple[int, ...] | None:
    possible = []
    for subset in combinations(range(4), len(areas)):
        if 0 not in subset or max(subset) == 0:
            continue
        for perm in permutations(subset):
            if all(area >= MIN_AREA_FOR_HOLES[int(h)] for area, h in zip(areas, perm, strict=False)):
                possible.append(tuple(int(h) for h in perm))
    if not possible:
        return None
    return random.choice(possible)


def _free_positions(occupied: np.ndarray, shape: tuple[int, int], padding: int = 1) -> list[tuple[int, int]]:
    h, w = occupied.shape
    sh, sw = shape
    valid = []
    for r in range(0, h - sh + 1):
        for c in range(0, w - sw + 1):
            r0 = max(0, r - padding)
            r1 = min(h, r + sh + padding)
            c0 = max(0, c - padding)
            c1 = min(w, c + sw + padding)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((r, c))
    return valid


def _mark_occupied(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], padding: int = 1) -> None:
    h, w = occupied.shape
    sh, sw = shape
    r0 = max(0, top - padding)
    r1 = min(h, top + sh + padding)
    c0 = max(0, left - padding)
    c1 = min(w, left + sw + padding)
    occupied[r0:r1, c0:c1] = True


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = patch.shape
    view = grid[top : top + sh, left : left + sw]
    view[patch] = int(color)


def _build_expected(out_h: int, out_w: int, rect_specs: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((out_h, out_w), BG, dtype=int)
    for area, color in sorted(rect_specs, key=lambda item: item[0], reverse=True):
        rect_h, rect_w = _best_rect_dims(area)
        r0 = (out_h - rect_h) // 2
        c0 = (out_w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)
    return out


def _build_legend(out_h: int, out_w: int, mapping: dict[int, int]) -> np.ndarray:
    legend = np.full((out_h, out_w), BG, dtype=int)
    start = random.randint(0, out_h - (2 * max(mapping) + 1))
    square_col = random.randint(max(mapping) + 1, out_w - 1)
    for holes, color in sorted(mapping.items()):
        row = start + 2 * holes
        if holes > 0:
            legend[row, :holes] = RED
        legend[row, square_col] = int(color)
    return legend


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 4)
    out_h = random.randint(9, 15)
    out_w = random.randint(max(8, out_h - 2), min(13, out_h + 3))

    rects = _choose_nested_rectangles(out_h, out_w, object_count)
    if rects is None:
        return None
    areas = [area for area, _h, _w in rects]

    holes_per_object = _assign_distinct_holes(areas)
    if holes_per_object is None:
        return None
    max_holes = max(holes_per_object)
    if out_h < 2 * max_holes + 1 or out_w < max_holes + 2:
        return None

    mapping_colors = random.sample(COLOR_POOL, max_holes + 1)
    mapping = {holes: color for holes, color in enumerate(mapping_colors)}
    object_colors = random.sample(COLOR_POOL, object_count)

    object_masks = []
    for area, holes in zip(areas, holes_per_object, strict=False):
        mask = _make_object_mask(area, holes)
        if mask is None:
            return None
        object_masks.append(mask)

    max_obj_w = max(mask.shape[1] for mask in object_masks)
    min_left = max(max_obj_w + 2, 10)
    if min_left + 1 + out_w > 30:
        return None
    total_w = random.randint(min_left + 1 + out_w, 30)
    left_w = total_w - 1 - out_w

    grid = np.full((out_h, total_w), BG, dtype=int)
    grid[:, left_w] = DIVIDER
    grid[:, left_w + 1 :] = _build_legend(out_h, out_w, mapping)

    occupied = np.zeros((out_h, left_w), dtype=bool)
    order = list(range(object_count))
    random.shuffle(order)
    for idx in order:
        mask = object_masks[idx]
        positions = _free_positions(occupied, mask.shape, padding=1)
        if not positions:
            return None
        top, left = random.choice(positions)
        _draw_patch(grid[:, :left_w], mask, top, left, object_colors[idx])
        _mark_occupied(occupied, top, left, mask.shape, padding=1)

    expected = _build_expected(
        out_h,
        out_w,
        [(area, mapping[holes]) for area, holes in zip(areas, holes_per_object, strict=False)],
    )
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-legend ARC input")

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIDE_RANGE = (15, 24)
RECT_COUNT_RANGE = (2, 4)
GREY_COUNT_RANGE = (1, 4)
GREY_AREA_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-defined rectangle while preserving grey hole objects and letting smaller overlaps sit on top.

    Input structure:
    - The background is Black.
    - Every non-Black, non-Grey color appears exactly twice as a pair of single-pixel markers.
    - Small 4-connected Grey objects may lie inside some marker-defined rectangles.

    Rule:
    - For each color pair, treat the two markers as opposite corners of an axis-aligned rectangle.
    - Sort these rectangles by descending area and paint them in that order.
    - When painting a rectangle, recolor every covered non-Grey cell to the pair color, including marker pixels and cells painted by earlier larger rectangles.
    - Grey pixels from the input stay Grey, so they remain as holes through every layer.

    Key insight:
    - Same-colored marker pairs are not endpoints to connect; they specify filled rectangles, and overlap order is determined by visible rectangle area while Grey objects act as immutable cutouts.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    grey_mask = g == GREY
    out[grey_mask] = GREY

    for area, color, r0, c0, r1, c1 in _pair_rectangles(g):
        del area
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region_grey = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        region[~region_grey] = color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-rectangle puzzle")


def _pair_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    ordered_colors = []
    seen = set()
    for value in grid.reshape(-1):
        color = int(value)
        if color in (BLACK, GREY) or color in seen:
            continue
        seen.add(color)
        ordered_colors.append(color)

    rects = []
    for color in ordered_colors:
        pts = np.argwhere(grid == color)
        assert pts.shape[0] == 2, f"color {color} must appear exactly twice"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
        rects.append((area, color, int(r0), int(c0), int(r1), int(c1)))
    rects.sort(key=lambda item: -item[0])
    return rects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)
    rect_count = rng.randint(*RECT_COUNT_RANGE)
    mode = rng.choice(("disjoint", "nested", "partial"))

    rects = _build_rectangles(height, width, rect_count, mode, rng)
    if rects is None:
        return None

    colors = list(rng.sample(PAIR_COLORS, len(rects)))
    grid = np.full((height, width), BLACK, dtype=int)
    marker_mask = np.zeros_like(grid, dtype=bool)

    for rect, color in zip(rects, colors):
        for r, c in _marker_cells(rect, rng):
            grid[r, c] = color
            marker_mask[r, c] = True

    rect_masks = [_rect_mask(rect, height, width) for rect in rects]
    union_mask = np.zeros_like(grid, dtype=bool)
    for mask in rect_masks:
        union_mask |= mask

    overlap_masks = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            mask = rect_masks[i] & rect_masks[j]
            if int(mask.sum()) > 0:
                overlap_masks.append(mask)

    blocked = marker_mask.copy()
    plans = _grey_plans(len(overlap_masks) > 0, rng)
    for plan in plans:
        area = rng.randint(*GREY_AREA_RANGE)
        shape = _random_polyomino(area, rng)
        target_mask = _target_mask_for_plan(plan, rect_masks, overlap_masks, union_mask, rng)
        if target_mask is None:
            return None
        cells = _place_shape(shape, target_mask, blocked, rng)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = GREY
            blocked[r, c] = True

    if not _validate_generated_grid(grid, rects, mode):
        return None
    return grid


def _build_rectangles(
    height: int, width: int, rect_count: int, mode: str, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    used_areas: set[int] = set()

    if mode == "disjoint":
        rects: list[tuple[int, int, int, int]] = []
        for _ in range(rect_count):
            rect = _sample_disjoint_rect(height, width, rects, used_areas, rng)
            if rect is None:
                return None
            rects.append(rect)
            used_areas.add(_rect_area(rect))
        return rects

    base = _sample_large_rect(height, width, rng)
    if base is None:
        return None
    rects = [base]
    used_areas.add(_rect_area(base))

    if mode == "nested":
        top = _sample_nested_rect(base, used_areas, rng)
    else:
        top = _sample_partial_rect(base, height, width, used_areas, rng)
    if top is None:
        return None
    rects.append(top)
    used_areas.add(_rect_area(top))

    max_extra_area = _rect_area(top) - 1
    for _ in range(rect_count - 2):
        rect = _sample_disjoint_rect(
            height, width, rects, used_areas, rng, max_area=max_extra_area
        )
        if rect is None:
            return None
        rects.append(rect)
        used_areas.add(_rect_area(rect))
    return rects


def _sample_large_rect(
    height: int, width: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    max_h = min(10, height - 4)
    max_w = min(10, width - 4)
    if max_h < 6 or max_w < 6:
        return None
    for _ in range(120):
        h = rng.randint(6, max_h)
        w = rng.randint(6, max_w)
        r0 = rng.randint(1, height - h - 2)
        c0 = rng.randint(1, width - w - 2)
        return (r0, c0, r0 + h - 1, c0 + w - 1)
    return None


def _sample_nested_rect(
    base: tuple[int, int, int, int], used_areas: set[int], rng: random.Random
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = base
    base_h = r1 - r0 + 1
    base_w = c1 - c0 + 1
    for _ in range(160):
        h = rng.randint(2, base_h - 2)
        w = rng.randint(2, base_w - 2)
        area = h * w
        if area in used_areas or area >= _rect_area(base):
            continue
        sr = rng.randint(r0 + 1, r1 - h)
        sc = rng.randint(c0 + 1, c1 - w)
        return (sr, sc, sr + h - 1, sc + w - 1)
    return None


def _sample_partial_rect(
    base: tuple[int, int, int, int],
    height: int,
    width: int,
    used_areas: set[int],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = base
    base_area = _rect_area(base)
    for _ in range(240):
        start_r_min = max(r0 + 1, r1 - 4)
        start_c_min = max(c0 + 1, c1 - 4)
        if start_r_min > r1 - 2 or start_c_min > c1 - 2:
            continue
        sr = rng.randint(start_r_min, r1 - 2)
        sc = rng.randint(start_c_min, c1 - 2)
        min_h = (r1 - sr + 1) + 1
        min_w = (c1 - sc + 1) + 1
        max_h = min(8, height - sr - 1)
        max_w = min(8, width - sc - 1)
        if min_h > max_h or min_w > max_w:
            continue
        h = rng.randint(min_h, max_h)
        w = rng.randint(min_w, max_w)
        area = h * w
        if area in used_areas or area >= base_area:
            continue
        rect = (sr, sc, sr + h - 1, sc + w - 1)
        overlap = _intersection_mask(base, rect, height, width)
        if overlap is None or int(overlap.sum()) < 4:
            continue
        return rect
    return None


def _sample_disjoint_rect(
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
    used_areas: set[int],
    rng: random.Random,
    max_area: int | None = None,
) -> tuple[int, int, int, int] | None:
    max_h = min(8, height - 2)
    max_w = min(8, width - 2)
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(260):
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        area = h * w
        if area in used_areas:
            continue
        if max_area is not None and area > max_area:
            continue
        r0 = rng.randint(1, height - h - 1)
        c0 = rng.randint(1, width - w - 1)
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_rects_conflict(rect, other, pad=1) for other in existing):
            continue
        return rect
    return None


def _marker_cells(
    rect: tuple[int, int, int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if rng.random() < 0.5:
        return [(r0, c0), (r1, c1)]
    return [(r0, c1), (r1, c0)]


def _grey_plans(has_overlap: bool, rng: random.Random) -> list[str]:
    count = rng.randint(*GREY_COUNT_RANGE)
    plans = ["overlap" if has_overlap and rng.random() < 0.6 else "inside"]
    for _ in range(count - 1):
        choices = ["inside", "outside"]
        if has_overlap:
            choices.append("overlap")
        plans.append(rng.choice(choices))
    return plans


def _target_mask_for_plan(
    plan: str,
    rect_masks: list[np.ndarray],
    overlap_masks: list[np.ndarray],
    union_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    border_clear = np.ones_like(union_mask, dtype=bool)
    border_clear[[0, -1], :] = False
    border_clear[:, [0, -1]] = False

    if plan == "overlap":
        if not overlap_masks:
            return None
        return overlap_masks[rng.randrange(len(overlap_masks))] & border_clear
    if plan == "inside":
        return rect_masks[rng.randrange(len(rect_masks))] & border_clear
    return (~union_mask) & border_clear


def _place_shape(
    shape: tuple[tuple[int, int], ...],
    target_mask: np.ndarray,
    blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    height, width = target_mask.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    if shape_h > height or shape_w > width:
        return None

    for _ in range(80):
        br = rng.randint(0, height - shape_h)
        bc = rng.randint(0, width - shape_w)
        cells = [(br + r, bc + c) for r, c in shape]
        if _cells_fit(cells, target_mask, blocked):
            return cells

    for br in range(height - shape_h + 1):
        for bc in range(width - shape_w + 1):
            cells = [(br + r, bc + c) for r, c in shape]
            if _cells_fit(cells, target_mask, blocked):
                return cells
    return None


def _cells_fit(
    cells: list[tuple[int, int]], target_mask: np.ndarray, blocked: np.ndarray
) -> bool:
    height, width = target_mask.shape
    cell_set = set(cells)
    for r, c in cells:
        if not target_mask[r, c] or blocked[r, c]:
            return False
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                if (rr, cc) in cell_set:
                    continue
                if blocked[rr, cc]:
                    return False
    return True


def _random_polyomino(area: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
        frontier.difference_update(cells)
        cells.add(rng.choice(tuple(frontier)))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(normalized)


def _validate_generated_grid(
    grid: np.ndarray, rects: list[tuple[int, int, int, int]], mode: str
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    color_counts = Counter(map(int, grid.reshape(-1)))
    pair_colors = [color for color in color_counts if color not in (BLACK, GREY)]
    if any(color_counts[color] != 2 for color in pair_colors):
        return False

    areas = [_rect_area(rect) for rect in rects]
    if len(areas) != len(set(areas)):
        return False

    if mode == "disjoint":
        if any(_rect_overlap_area(a, b) > 0 for i, a in enumerate(rects) for b in rects[i + 1 :]):
            return False
    else:
        if not any(_rect_overlap_area(a, b) > 0 for i, a in enumerate(rects) for b in rects[i + 1 :]):
            return False

    grey_components = _components4(grid == GREY)
    if not grey_components:
        return False
    if any(not (2 <= len(comp) <= 6) for comp in grey_components):
        return False

    union_mask = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        union_mask |= _rect_mask(rect, *grid.shape)
    if not any(any(union_mask[r, c] for r, c in comp) for comp in grey_components):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False
    return True


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _rect_mask(rect: tuple[int, int, int, int], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _intersection_mask(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    height: int,
    width: int,
) -> np.ndarray | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return _rect_mask((r0, c0, r1, c1), height, width)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rects_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )

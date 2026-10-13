import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

LEGEND_BORDER = GREY
LEGEND_TARGET_POOL = [BLUE, RED]
RECT_COLORS = [GREEN, YELLOW, ORANGE]
COMP_EXTRA_COLORS = [PINK, TEAL, MAROON]

MAX_GEN_ATTEMPTS = 400
_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == LEGEND_BORDER)]
    if not grey_coords:
        raise AssertionError("legend border missing")
    r0, c0, r1, c1 = _bbox_from_coords(grey_coords)
    if r0 != 0 or c0 != 0 or r1 - r0 < 2 or c1 - c0 < 2:
        raise AssertionError("legend must be a top-left framed box")
    return r0, c0, r1, c1


def _legend_targets(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> set[int]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    targets = {int(v) for v in np.unique(interior) if int(v) != BG}
    if not targets:
        raise AssertionError("legend interior must contain at least one target color")
    return targets


def _mask_legend_area(shape: tuple[int, int], legend_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _is_solid_domino(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 2:
        return False
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) in {(1, 2), (2, 1)}


def _corners(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _facing_corners(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if box_a[0] <= box_b[0]:
        upper, lower = box_a, box_b
    else:
        upper, lower = box_b, box_a
    candidates = []
    for p in _corners(upper):
        for q in _corners(lower):
            dr = q[0] - p[0]
            dc = q[1] - p[1]
            if dr > 0 and abs(dr) == abs(dc):
                candidates.append((dr, abs(dc), p, q))
    if not candidates:
        raise AssertionError("same-color rectangles must expose diagonal facing corners")
    candidates.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1], item[3][0], item[3][1]))
    return candidates[0][2], candidates[0][3]


def _point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _path_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    if dr <= 0 or abs(dr) != abs(dc):
        raise AssertionError("facing corners must define a descending 45-degree path")
    step_c = 1 if dc > 0 else -1
    return [(start[0] + k, start[1] + k * step_c) for k in range(1, dr)]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_bbox = _find_legend_bbox(g)
    targets = _legend_targets(g, legend_bbox)
    legend_mask = _mask_legend_area((h, w), legend_bbox)

    work = g.copy()
    work[legend_mask] = BG

    comps = _components_mask(work != BG)
    multicolor = []
    deleted_boxes = []
    erased_pixels = []
    by_color = defaultdict(list)

    for comp in comps:
        colors = {int(work[r, c]) for r, c in comp}
        box = _bbox_from_coords(comp)
        if len(colors) >= 2:
            item = {
                "coords": comp,
                "colors": colors,
                "bbox": box,
                "deleted": targets.issubset(colors),
            }
            multicolor.append(item)
            if item["deleted"]:
                deleted_boxes.append(box)
                erased_pixels.extend(comp)
        else:
            col = next(iter(colors))
            by_color[col].append({"coords": comp, "bbox": box})

    pairs = []
    for color, items in by_color.items():
        rects = [item for item in items if _is_solid_domino(item["coords"])]
        if len(rects) == 2:
            start, end = _facing_corners(rects[0]["bbox"], rects[1]["bbox"])
            path = _path_between(start, end)
            hit_deleted = any(any(_point_in_box(pt, box) for box in deleted_boxes) for pt in path)
            pairs.append(
                {
                    "color": int(color),
                    "boxes": [rects[0]["bbox"], rects[1]["bbox"]],
                    "start": start,
                    "end": end,
                    "path": path,
                    "hits_deleted_box": hit_deleted,
                }
            )

    return {
        "legend_bbox": legend_bbox,
        "targets": targets,
        "multicolor": multicolor,
        "deleted_boxes": deleted_boxes,
        "erased_pixels": erased_pixels,
        "pairs": pairs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase legend-matching multicolor components, then draw diagonal connectors that skip ghost obstacle boxes.

    Input structure:
    - Black background fills most of the grid.
    - The top-left corner contains a small grey framed legend whose interior lists one or two target colors.
    - Away from the legend are irregular multicolor 4-connected components and several pairs of same-color 1x2 or 2x1 solid rectangles.

    Rule:
    - Read the legend's interior color set T.
    - For every non-legend 4-connected multicolor component, erase it if its color set contains all of T, and remember that erased component's original bounding box as an obstacle zone.
    - Keep other multicolor components unchanged.
    - For each same-color rectangle pair, find the diagonally facing corners and paint the missing 45-degree connector only on black cells.
    - If a connector pixel falls inside any remembered obstacle zone, skip that pixel so the line keeps a gap through the ghost box.

    Key insight:
    - The same legend-driven deletion both removes components and leaves behind invisible bounding-box obstacles that reshape the later diagonal connectors.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    info = _analyze(g)

    for comp in info["multicolor"]:
        if comp["deleted"]:
            for r, c in comp["coords"]:
                out[r, c] = BG

    for pair in info["pairs"]:
        for pt in pair["path"]:
            if any(_point_in_box(pt, box) for box in info["deleted_boxes"]):
                continue
            r, c = pt
            if out[r, c] == BG:
                out[r, c] = pair["color"]

    return out


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]
    coords = [(top + r, left + c) for r, c in np.argwhere(mask)]
    return _bbox_from_coords(coords)


def _random_connected_patch(
    rng: random.Random,
    dims: tuple[int, int],
    color_choices: list[int],
    n_cells: int,
) -> np.ndarray | None:
    h, w = dims
    if n_cells < len(color_choices) or n_cells > h * w:
        return None
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < n_cells and frontier:
        pt = rng.choice(tuple(frontier))
        frontier.remove(pt)
        cells.add(pt)
        for nr, nc in _neighbors4(*pt):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    if len(cells) != n_cells:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(cells)
    if (r1 - r0 + 1) * (c1 - c0 + 1) == n_cells:
        return None
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    cropped = [(r - r0, c - c0) for r, c in cells]
    colors = list(color_choices)
    while len(colors) < len(cropped):
        colors.append(rng.choice(color_choices))
    rng.shuffle(colors)
    for (r, c), color in zip(cropped, colors):
        patch[r, c] = int(color)
    return patch


def _make_multicolor_patch(
    rng: random.Random,
    targets: list[int],
    deleted: bool,
) -> np.ndarray | None:
    dims = (rng.randint(3, 6), rng.randint(3, 6))
    max_cells = min(15, dims[0] * dims[1] - 1)
    min_cells = max(6, min(9, max_cells))
    if min_cells > max_cells:
        return None

    if deleted:
        min_color_count = max(2, len(targets))
        total_color_count = rng.randint(min_color_count, min(4, len(targets) + len(COMP_EXTRA_COLORS)))
        extras = rng.sample(COMP_EXTRA_COLORS, total_color_count - len(targets))
        colors = list(targets) + extras
    else:
        present_target_count = rng.randint(0, len(targets) - 1)
        present_targets = rng.sample(targets, present_target_count)
        min_extra = 2 - len(present_targets)
        extra_count = rng.randint(max(1, min_extra), min(4 - len(present_targets), len(COMP_EXTRA_COLORS)))
        colors = list(present_targets) + rng.sample(COMP_EXTRA_COLORS, extra_count)

    n_cells = rng.randint(max(6, len(colors) + 2), max_cells)
    return _random_connected_patch(rng, dims, colors, n_cells)


def _legend_grid(rng: random.Random, h: int, w: int, target_count: int) -> tuple[np.ndarray, list[int], tuple[int, int, int, int]]:
    grid = np.zeros((h, w), dtype=int)
    box = (0, 0, 4, 4)
    grid[0:5, 0] = LEGEND_BORDER
    grid[0:5, 4] = LEGEND_BORDER
    grid[0, 0:5] = LEGEND_BORDER
    grid[4, 0:5] = LEGEND_BORDER
    targets = rng.sample(LEGEND_TARGET_POOL, target_count)
    spots = rng.sample([(r, c) for r in range(1, 4) for c in range(1, 4)], target_count)
    for color, (r, c) in zip(targets, spots):
        grid[r, c] = color
    return grid, targets, box


def _build_pair_from_upper(
    upper_top: tuple[int, int],
    orientation: str,
    direction: int,
    gap: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    ur, uc = upper_top
    if orientation == "h":
        upper = (ur, uc, ur, uc + 1)
        if direction == 1:
            start = (ur, uc + 1)
            end = (start[0] + gap + 1, start[1] + gap + 1)
            lower = (end[0], end[1], end[0], end[1] + 1)
        else:
            start = (ur, uc)
            end = (start[0] + gap + 1, start[1] - gap - 1)
            lower = (end[0], end[1] - 1, end[0], end[1])
    else:
        upper = (ur, uc, ur + 1, uc)
        start = (ur + 1, uc)
        if direction == 1:
            end = (start[0] + gap + 1, start[1] + gap + 1)
            lower = (end[0], end[1], end[0] + 1, end[1])
        else:
            end = (start[0] + gap + 1, start[1] - gap - 1)
            lower = (end[0], end[1], end[0] + 1, end[1])
    return upper, lower, _path_between(start, end)


def _in_bounds_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= box[0] <= box[2] < h and 0 <= box[1] <= box[3] < w


def _reserve_line_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            blocked[r, c] = True


def _place_component(
    grid: np.ndarray,
    rng: random.Random,
    blocked: np.ndarray,
    existing_boxes: list[tuple[int, int, int, int]],
    targets: list[int],
    deleted: bool,
    prefer_center: bool,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(160):
        patch = _make_multicolor_patch(rng, targets, deleted)
        if patch is None:
            continue
        ph, pw = patch.shape
        row_lo = 5
        row_hi = h - ph
        col_lo = 5
        col_hi = w - pw
        if row_lo > row_hi or col_lo > col_hi:
            return None
        if prefer_center:
            top = rng.randint(max(row_lo, h // 2 - ph - 1), min(row_hi, h // 2))
            left = rng.randint(max(col_lo, w // 2 - pw - 1), min(col_hi, w // 2))
        else:
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)
        coords = [(top + r, left + c) for r, c in np.argwhere(patch != BG)]
        box = _bbox_from_coords(coords)
        if any(_bbox_overlap(box, other, pad=1) for other in existing_boxes):
            continue
        region = blocked[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if region.any():
            continue
        painted_box = _paint_patch(grid, patch, top, left)
        return painted_box, coords
    return None


def _box_margin_from_points(box: tuple[int, int, int, int], points: list[tuple[int, int]]) -> int:
    if not points:
        return 10**9
    margins = []
    for r, c in points:
        dr = min(abs(r - box[0]), abs(r - box[2]))
        dc = min(abs(c - box[1]), abs(c - box[3]))
        margins.append(min(dr, dc))
    return min(margins)


def _find_pair_candidate(
    shape: tuple[int, int],
    rng: random.Random,
    blocked: np.ndarray,
    component_boxes: list[tuple[int, int, int, int]],
    obstacle_boxes: list[tuple[int, int, int, int]],
    require_hit: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]], bool] | None:
    h, w = shape
    candidates = []
    for orientation in ["h", "v"]:
        for direction in [-1, 1]:
            for gap in range(3, 8):
                for ur in range(5, h - 4):
                    for uc in range(1, w - 1):
                        upper, lower, path = _build_pair_from_upper((ur, uc), orientation, direction, gap)
                        if not (_in_bounds_box(upper, shape) and _in_bounds_box(lower, shape)):
                            continue
                        if any(_bbox_overlap(upper, box, pad=1) or _bbox_overlap(lower, box, pad=1) for box in component_boxes):
                            continue
                        if any(
                            blocked[r, c]
                            for box in (upper, lower)
                            for r in range(box[0], box[2] + 1)
                            for c in range(box[1], box[3] + 1)
                        ):
                            continue
                        hit_mask = [any(_point_in_box(pt, box) for box in obstacle_boxes) for pt in path]
                        if require_hit and not any(hit_mask):
                            continue
                        if not require_hit and any(hit_mask):
                            continue
                        if require_hit:
                            first = hit_mask.index(True)
                            last = len(path) - 1 - hit_mask[::-1].index(True)
                            if first == 0 or last == len(path) - 1:
                                continue
                        if any(blocked[r, c] for (r, c), hit in zip(path, hit_mask) if not hit):
                            continue
                        candidates.append((upper, lower, path, any(hit_mask)))
    if candidates:
        return rng.choice(candidates)
    return None


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _validate_generated(inp: np.ndarray) -> bool:
    out = main(inp)
    info = _analyze(inp)
    if not (1 <= len(info["targets"]) <= 2):
        return False
    deleted = [comp for comp in info["multicolor"] if comp["deleted"]]
    kept = [comp for comp in info["multicolor"] if not comp["deleted"]]
    if len(deleted) < 1 or len(kept) < 1:
        return False
    if len(info["pairs"]) < 2:
        return False
    if not any(pair["hits_deleted_box"] for pair in info["pairs"]):
        return False
    if not any(not pair["hits_deleted_box"] for pair in info["pairs"]):
        return False
    erased_ok = any(out[r, c] == BG for r, c in info["erased_pixels"])
    if not erased_ok:
        return False
    connector_added = int(np.sum((inp == BG) & (out != BG)))
    if connector_added < 2:
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt(rng: random.Random, mode: int) -> np.ndarray | None:
    h = rng.randint(16, 19)
    w = rng.randint(16, 19)
    target_count = 1 if mode % 2 == 0 else 2
    pair_count = 2 if mode in (0, 1) else 3
    total_components = 3 if mode in (0, 2) else 4
    deleted_count = 1 if total_components == 3 else 2

    grid, targets, legend_box = _legend_grid(rng, h, w, target_count)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[legend_box[0] : legend_box[2] + 1, legend_box[1] : legend_box[3] + 1] = True

    component_boxes = [legend_box]
    deleted_boxes = []

    forced = _place_component(
        grid=grid,
        rng=rng,
        blocked=blocked,
        existing_boxes=component_boxes,
        targets=targets,
        deleted=True,
        prefer_center=True,
    )
    if forced is None:
        return None
    box, _coords = forced
    component_boxes.append(box)
    deleted_boxes.append(box)
    blocked[box[0] : box[2] + 1, box[1] : box[3] + 1] = True

    roles = [False] + [True] * (deleted_count - 1) + [False] * (total_components - deleted_count - 1)
    rng.shuffle(roles)
    for deleted in roles:
        placed = _place_component(
            grid=grid,
            rng=rng,
            blocked=blocked,
            existing_boxes=component_boxes,
            targets=targets,
            deleted=deleted,
            prefer_center=False,
        )
        if placed is None:
            return None
        box, _coords = placed
        component_boxes.append(box)
        blocked[box[0] : box[2] + 1, box[1] : box[3] + 1] = True
        if deleted:
            deleted_boxes.append(box)

    rect_colors = RECT_COLORS[:]
    rng.shuffle(rect_colors)

    hit_pair = _find_pair_candidate((h, w), rng, blocked, component_boxes, deleted_boxes, require_hit=True)
    if hit_pair is None:
        return None
    upper, lower, path, _ = hit_pair
    _paint_box(grid, upper, rect_colors[0])
    _paint_box(grid, lower, rect_colors[0])
    blocked[upper[0] : upper[2] + 1, upper[1] : upper[3] + 1] = True
    blocked[lower[0] : lower[2] + 1, lower[1] : lower[3] + 1] = True
    _reserve_line_cells(blocked, [pt for pt in path if not any(_point_in_box(pt, box) for box in deleted_boxes)])

    clean_pair = _find_pair_candidate((h, w), rng, blocked, component_boxes, deleted_boxes, require_hit=False)
    if clean_pair is None:
        return None
    upper, lower, path, _ = clean_pair
    _paint_box(grid, upper, rect_colors[1])
    _paint_box(grid, lower, rect_colors[1])
    blocked[upper[0] : upper[2] + 1, upper[1] : upper[3] + 1] = True
    blocked[lower[0] : lower[2] + 1, lower[1] : lower[3] + 1] = True
    _reserve_line_cells(blocked, path)

    for i in range(2, pair_count):
        extra = _find_pair_candidate((h, w), rng, blocked, component_boxes, deleted_boxes, require_hit=bool(i % 2 and len(deleted_boxes) > 1))
        if extra is None:
            return None
        upper, lower, path, _ = extra
        _paint_box(grid, upper, rect_colors[i])
        _paint_box(grid, lower, rect_colors[i])
        blocked[upper[0] : upper[2] + 1, upper[1] : upper[3] + 1] = True
        blocked[lower[0] : lower[2] + 1, lower[1] : lower[3] + 1] = True
        _reserve_line_cells(blocked, [pt for pt in path if not any(_point_in_box(pt, box) for box in deleted_boxes)])

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 4
    _GEN_COUNTER += 1
    rng = random.Random(random.randrange(10**9))
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

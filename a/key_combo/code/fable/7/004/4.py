from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_004_4",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location
tile_fill = _ARC_UTILS.tile_fill

_RETRIES = 600
_DIMS = range(2, 7)
_COLORS = tuple(range(1, 10))
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _largest_black_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    black = np.asarray(grid, dtype=int) == 0
    h, w = black.shape
    heights = np.zeros(w, dtype=int)
    best_area = 0
    best_rect = None
    best_count = 0

    for r in range(h):
        heights = np.where(black[r], heights + 1, 0)
        for c0 in range(w - 1):
            min_h = h + 1
            for c1 in range(c0, w):
                if heights[c1] < min_h:
                    min_h = int(heights[c1])
                width = c1 - c0 + 1
                if width < 2 or min_h < 2:
                    continue
                area = min_h * width
                if area < best_area:
                    continue
                rect = (r - min_h + 1, c0, r, c1)
                if area > best_area:
                    best_area = area
                    best_rect = rect
                    best_count = 1
                else:
                    best_count += 1

    if best_rect is None or best_count != 1:
        raise ValueError("expected a unique largest all-black rectangle of size at least 2x2")
    return best_rect


def _extract_templates(grid: np.ndarray) -> list[dict]:
    templates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        for coords in components_mask(grid == color, connectivity=4):
            box = bbox_coords(coords)
            templates.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "pattern": crop_bbox(grid, box),
                }
            )
    templates.sort(key=lambda item: (item["bbox"], item["color"]))
    return templates


def _orientations(pattern: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = [("orig", pattern)]
    rot = np.rot90(pattern)
    if rot.shape != pattern.shape or not np.array_equal(rot, pattern):
        out.append(("rot90", rot))
    return out


def _matching_orientations(tile_h: int, tile_w: int, cavity_h: int, cavity_w: int) -> list[str]:
    out = []
    seen = set()
    for name, dims in (("orig", (tile_h, tile_w)), ("rot90", (tile_w, tile_h))):
        if dims in seen:
            continue
        seen.add(dims)
        if cavity_h % dims[0] == 0 and cavity_w % dims[1] == 0:
            out.append(name)
    return out


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    cavity = _largest_black_rectangle(g)
    r0, c0, r1, c1 = cavity
    cavity_h = r1 - r0 + 1
    cavity_w = c1 - c0 + 1

    templates = _extract_templates(g)
    if not templates:
        raise ValueError("expected at least one non-black template")

    matches = []
    for idx, template in enumerate(templates):
        for orientation, pattern in _orientations(template["pattern"]):
            tile_h, tile_w = map(int, pattern.shape)
            if cavity_h % tile_h == 0 and cavity_w % tile_w == 0:
                matches.append(
                    {
                        "template_index": idx,
                        "orientation": orientation,
                        "pattern": pattern,
                    }
                )

    if len(matches) != 1:
        raise ValueError("expected exactly one template in exactly one orientation to divide the cavity")

    return {
        "grid": g,
        "cavity": cavity,
        "templates": templates,
        "match": matches[0],
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Find the largest black empty rectangle and tile it with the unique size-dividing template.

    Input structure:
    - Several separated monochrome template objects are scattered on the grid.
    - Each template is interpreted through its tight bounding-box pattern, including its internal background cells.
    - Somewhere else there is a unique largest axis-aligned all-black rectangle whose height and width are both at least 2.

    Rule:
    - Enumerate all all-black rectangles of size at least 2 by 2 and select the unique one with maximum area as the target cavity.
    - For every template bbox pattern, test its original orientation and its 90-degree rotation.
    - Exactly one template in exactly one orientation has both dimensions dividing the target cavity dimensions; tile that oriented pattern periodically to fill the cavity and leave everything else unchanged.

    Key insight:
    - The black cavity acts as a divisibility filter: its dimensions uniquely identify which scattered template can be reused, and sometimes the winning template must be rotated first.
    """
    details = _solve_details(np.asarray(input_grid, dtype=int))
    out = np.array(details["grid"], copy=True)
    r0, c0, r1, c1 = details["cavity"]
    pattern = details["match"]["pattern"]
    out[r0 : r1 + 1, c0 : c1 + 1] = tile_fill((r1 - r0 + 1, c1 - c0 + 1), pattern)
    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    mask[r, c] = True
    while (r, c) != end:
        steps = []
        if r < end[0]:
            steps.append((1, 0))
        elif r > end[0]:
            steps.append((-1, 0))
        if c < end[1]:
            steps.append((0, 1))
        elif c > end[1]:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in zip(*np.where(mask)):
        rr = int(r)
        cc = int(c)
        for dr, dc in _DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _random_mask(h: int, w: int) -> np.ndarray:
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    center = (random.randrange(h), random.randrange(w))

    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True
        order = anchors[:]
        random.shuffle(order)
        for anchor in order:
            start_candidates = list(zip(*np.where(mask)))
            start = random.choice([(int(r), int(c)) for r, c in start_candidates])
            _random_path(mask, start, anchor)

        min_area = int(mask.sum())
        max_area = h * w
        if max_area >= 6 and random.random() < 0.75:
            max_area -= 1
        target = random.randint(min_area, max(min_area, max_area))

        while int(mask.sum()) < target:
            frontier = _frontier(mask)
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if max_area < h * w and mask.all():
            continue
        return mask

    raise RuntimeError("failed to generate a connected template mask")


def _make_pattern(h: int, w: int, color: int, bg: int) -> np.ndarray:
    pattern = np.full((h, w), bg, dtype=int)
    pattern[_random_mask(h, w)] = color
    return pattern


def _choose_target_dims() -> tuple[str, int, int, int, int]:
    for _ in range(500):
        orientation = random.choice(["orig", "rot90"])
        tile_h = random.choice(tuple(_DIMS))
        tile_w = random.choice(tuple(_DIMS))
        if tile_h == tile_w:
            continue
        mul_h = random.randint(2, 4)
        mul_w = random.randint(2, 4)
        if orientation == "orig":
            cavity_h = tile_h * mul_h
            cavity_w = tile_w * mul_w
            if cavity_h % tile_w == 0 and cavity_w % tile_h == 0:
                continue
        else:
            cavity_h = tile_w * mul_h
            cavity_w = tile_h * mul_w
            if cavity_h % tile_h == 0 and cavity_w % tile_w == 0:
                continue
        if max(cavity_h, cavity_w) <= 16:
            return orientation, tile_h, tile_w, cavity_h, cavity_w
    raise RuntimeError("failed to choose cavity/template dimensions")


def _near_miss(h: int, w: int, cavity_h: int, cavity_w: int) -> bool:
    for a, b in {(h, w), (w, h)}:
        if (cavity_h % a == 0) ^ (cavity_w % b == 0):
            return True
    return False


def _choose_distractor_dims(
    chosen_dims: tuple[int, int],
    cavity_h: int,
    cavity_w: int,
    count: int,
) -> list[tuple[int, int]]:
    candidates = []
    for h in _DIMS:
        for w in _DIMS:
            dims = (h, w)
            if dims == chosen_dims:
                continue
            if _matching_orientations(h, w, cavity_h, cavity_w):
                continue
            candidates.append(dims)

    near = [dims for dims in candidates if _near_miss(dims[0], dims[1], cavity_h, cavity_w)]
    if not near:
        raise RuntimeError("expected at least one near-miss distractor")

    chosen = [random.choice(near)]
    pool = [dims for dims in candidates if dims not in chosen]
    while len(chosen) < count:
        dims = random.choice(pool)
        chosen.append(dims)
        pool = [item for item in pool if item != dims]
    random.shuffle(chosen)
    return chosen


def _build_expected(grid: np.ndarray, cavity: tuple[int, int, int, int], tile: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = cavity
    out[r0 : r1 + 1, c0 : c1 + 1] = tile_fill((r1 - r0 + 1, c1 - c0 + 1), tile)
    return out


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    chosen_bbox: tuple[int, int, int, int],
    chosen_orientation: str,
) -> bool:
    try:
        details = _solve_details(grid)
    except Exception:
        return False

    if details["match"]["orientation"] != chosen_orientation:
        return False
    if details["templates"][details["match"]["template_index"]]["bbox"] != chosen_bbox:
        return False

    cavity = details["cavity"]
    r0, c0, r1, c1 = cavity
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    mask = np.ones(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.array_equal(actual[mask], grid[mask]):
        return False

    tile = details["match"]["pattern"]
    filled = tile_fill((r1 - r0 + 1, c1 - c0 + 1), tile)
    if not np.array_equal(actual[r0 : r1 + 1, c0 : c1 + 1], filled):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_COLORS)
    available_colors = [color for color in _COLORS if color != bg]
    if len(available_colors) < 3:
        return None

    chosen_orientation, tile_h, tile_w, cavity_h, cavity_w = _choose_target_dims()
    template_count = random.randint(3, 5)
    distractor_dims = _choose_distractor_dims((tile_h, tile_w), cavity_h, cavity_w, template_count - 1)
    all_dims = [(tile_h, tile_w), *distractor_dims]
    colors = random.sample(available_colors, len(all_dims))
    patterns = [_make_pattern(h, w, color, bg) for (h, w), color in zip(all_dims, colors)]

    grid_h = random.randint(max(cavity_h + 2, 18), 30)
    grid_w = random.randint(max(cavity_w + 2, 18), 30)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserve = np.zeros((grid_h, grid_w), dtype=int)

    cavity_patch = np.ones((cavity_h, cavity_w), dtype=int)
    try:
        cavity_r, cavity_c = random_free_location(
            reserve,
            cavity_patch,
            transparent=0,
            border=0,
            padding=1,
        )
    except ValueError:
        return None
    paste(reserve, cavity_patch, cavity_r, cavity_c, transparent=0)
    grid[cavity_r : cavity_r + cavity_h, cavity_c : cavity_c + cavity_w] = 0

    chosen_bbox = None
    for idx, pattern in enumerate(patterns):
        reserve_patch = np.ones(pattern.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                reserve_patch,
                transparent=0,
                border=0,
                padding=1,
            )
        except ValueError:
            return None
        paste(reserve, reserve_patch, r0, c0, transparent=0)
        paste(grid, pattern, r0, c0, transparent=bg)
        if idx == 0:
            chosen_bbox = (r0, c0, r0 + pattern.shape[0] - 1, c0 + pattern.shape[1] - 1)

    if chosen_bbox is None:
        return None

    chosen_tile = patterns[0] if chosen_orientation == "orig" else np.rot90(patterns[0])
    cavity = (cavity_r, cavity_c, cavity_r + cavity_h - 1, cavity_c + cavity_w - 1)
    expected = _build_expected(grid, cavity, chosen_tile)

    if not _semantic_check(grid, expected, chosen_bbox, chosen_orientation):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/7/004/4")

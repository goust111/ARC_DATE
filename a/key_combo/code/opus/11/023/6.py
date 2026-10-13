import random
from collections import deque

import numpy as np


BG = 0
TOP_COLOR = 2
BOTTOM_COLOR = 1
OVERLAP_COLOR = 6
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

ONE_REGION_TEMPLATES = [
    {
        "objects": [
            ((0, 0), (1, 0), (1, 1), (2, 1)),
            ((0, 2), (1, 1), (1, 2), (2, 1)),
        ]
    },
    {
        "objects": [
            ((0, 0), (0, 1), (1, 1), (2, 1)),
            ((0, 1), (0, 2), (1, 1), (2, 1)),
        ]
    },
    {
        "objects": [
            ((0, 0), (1, 0), (2, 0), (2, 1)),
            ((0, 1), (1, 1), (2, 0), (2, 1)),
        ]
    },
]

TWO_REGION_TEMPLATES = [
    {
        "objects": [
            ((0, 0), (1, 0), (1, 1)),
            ((0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
            ((1, 2), (2, 2), (3, 2)),
        ]
    },
    {
        "objects": [
            ((0, 0), (1, 0), (1, 1)),
            ((0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
            ((0, 3), (1, 3), (2, 2), (2, 3), (3, 2)),
        ]
    },
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_color_objects(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid[1:-1, :] == int(color)
    comps = _component_coords(mask)
    result = []
    for coords in comps:
        lifted = coords + np.array([1, 0], dtype=int)
        result.append(lifted)
    return result


def _shift(coords: np.ndarray, dr: int, dc: int = 0) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _slide_red(coords: np.ndarray) -> np.ndarray:
    return _shift(coords, 1 - int(coords[:, 0].min()))


def _slide_blue(coords: np.ndarray, h: int) -> np.ndarray:
    return _shift(coords, (h - 2) - int(coords[:, 0].max()))


def _overlap_mask(object_coords: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = map(int, shape)
    occupancy = np.zeros((h, w), dtype=int)
    masks = []
    for coords in object_coords:
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        masks.append(mask)
        occupancy += mask.astype(int)
    return occupancy >= 2, masks


def _diagonal_stitch(overlap_comps: list[np.ndarray]) -> np.ndarray:
    ranked = sorted(overlap_comps, key=lambda comp: _bbox(comp)[:2])
    patches: list[np.ndarray] = []
    placements: list[tuple[int, int]] = []
    cur_r = 0
    cur_c = 0
    for comp in ranked:
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[comp[:, 0] - r0, comp[:, 1] - c0] = OVERLAP_COLOR
        patches.append(patch)
        placements.append((cur_r, cur_c))
        cur_r += patch.shape[0]
        cur_c += patch.shape[1]

    out = np.zeros((cur_r, cur_c), dtype=int)
    for patch, (r0, c0) in zip(patches, placements):
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


def _slid_overlap_components(grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    red_objects = _extract_color_objects(g, TOP_COLOR)
    blue_objects = _extract_color_objects(g, BOTTOM_COLOR)
    moved = [_slide_red(obj) for obj in red_objects] + [_slide_blue(obj, h) for obj in blue_objects]
    overlap, masks = _overlap_mask(moved, (h, w))
    return _component_coords(overlap), masks


def _touches8(a: np.ndarray, b: np.ndarray) -> bool:
    aset = {tuple(map(int, cell)) for cell in a.tolist()}
    bset = {tuple(map(int, cell)) for cell in b.tolist()}
    for r, c in aset:
        for dr, dc in DIR8:
            if (r + dr, c + dc) in bset:
                return True
    return bool(aset & bset)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide red and blue objects to their matching border bars, then extract and diagonally stitch the new overlaps.
    
    Input structure:
    - The background is black.
    - The top row is a full-width red bar and the bottom row is a full-width blue bar.
    - The interior contains several 4-connected red objects and blue objects, each disjoint in the input.
    
    Rule:
    - Slide every red object upward until its bbox top reaches row 1, and every blue object downward until its bbox bottom reaches row H-2.
    - Mark every pixel occupied by at least two slid objects as magenta; these 4-connected magenta sets are the overlap regions.
    - Crop each overlap region to its own bbox and place the cropped blocks on a new black canvas with cumulative down-right offsets.
    
    Key insight:
    The magenta regions do not exist in the input at all; they appear only after both color groups are slid to the boundary bars, so the final stitched canvas depends on the induced intersections rather than on any preexisting shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 5 <= h <= 30 and 5 <= w <= 30, "grid must satisfy ARC size limits"
    assert bool(np.all(g[0, :] == TOP_COLOR)), "top row must be the red border bar"
    assert bool(np.all(g[h - 1, :] == BOTTOM_COLOR)), "bottom row must be the blue border bar"
    assert set(map(int, np.unique(g))) <= {BG, TOP_COLOR, BOTTOM_COLOR}, "unexpected colors in input"

    red_objects = _extract_color_objects(g, TOP_COLOR)
    blue_objects = _extract_color_objects(g, BOTTOM_COLOR)
    assert red_objects and blue_objects, "expected movable objects for both border colors"

    moved = [_slide_red(obj) for obj in red_objects] + [_slide_blue(obj, h) for obj in blue_objects]
    overlap, _masks = _overlap_mask(moved, (h, w))
    comps = _component_coords(overlap)
    assert comps, "sliding must create at least one overlap region"
    return _diagonal_stitch(comps)


def _template_to_arrays(template: dict, allow_flip: bool) -> list[np.ndarray]:
    objects = [list(obj) for obj in template["objects"]]
    if allow_flip and bool(random.getrandbits(1)):
        max_c = max(c for obj in objects for _r, c in obj)
        objects = [[(r, max_c - c) for r, c in obj] for obj in objects]
    normalized = []
    for obj in objects:
        min_r = min(r for r, _c in obj)
        normalized.append(np.array(sorted((r - min_r, c) for r, c in obj), dtype=int))
    return normalized


def _group_dims(objects: list[np.ndarray]) -> tuple[int, int]:
    all_coords = np.vstack(objects)
    return int(all_coords[:, 0].max()) + 1, int(all_coords[:, 1].max()) + 1


def _object_height(coords: np.ndarray) -> int:
    r0, _c0, r1, _c1 = _bbox(coords)
    return int(r1 - r0 + 1)


def _sample_staggered_tops(heights: list[int], row_min: int, row_max: int) -> list[int] | None:
    order = list(range(len(heights)))
    random.shuffle(order)
    ordered_heights = [int(heights[i]) for i in order]
    required = sum(ordered_heights) + max(0, len(ordered_heights) - 1)
    available = int(row_max - row_min + 1)
    if required > available:
        return None

    extra_total = available - required
    extras = [0] * (len(ordered_heights) + 1)
    for _ in range(extra_total):
        extras[random.randrange(len(extras))] += 1

    ordered_tops = []
    cur = int(row_min + extras[0])
    for i, height in enumerate(ordered_heights):
        ordered_tops.append(cur)
        cur += height
        if i < len(ordered_heights) - 1:
            cur += 1 + extras[i + 1]

    tops = [0] * len(heights)
    for idx, top in zip(order, ordered_tops):
        tops[idx] = int(top)
    return tops


def _choose_groups(total_regions: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if total_regions == 2:
        red_template = random.choice(ONE_REGION_TEMPLATES)
        blue_template = random.choice(ONE_REGION_TEMPLATES)
        return _template_to_arrays(red_template, allow_flip=True), _template_to_arrays(blue_template, allow_flip=True)

    if bool(random.getrandbits(1)):
        red_template = random.choice(TWO_REGION_TEMPLATES)
        blue_template = random.choice(ONE_REGION_TEMPLATES)
    else:
        red_template = random.choice(ONE_REGION_TEMPLATES)
        blue_template = random.choice(TWO_REGION_TEMPLATES)
    return _template_to_arrays(red_template, allow_flip=False), _template_to_arrays(blue_template, allow_flip=True)


def _make_grid(h: int, w: int, red_objs: list[np.ndarray], blue_objs: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = TOP_COLOR
    grid[h - 1, :] = BOTTOM_COLOR
    for obj in red_objs:
        grid[obj[:, 0], obj[:, 1]] = TOP_COLOR
    for obj in blue_objs:
        grid[obj[:, 0], obj[:, 1]] = BOTTOM_COLOR
    return grid


def _overlap_cells(object_sets: list[np.ndarray], h: int, w: int) -> np.ndarray:
    overlap, _masks = _overlap_mask(object_sets, (h, w))
    return overlap


def _place_group_input(
    local_objects: list[np.ndarray],
    row_tops: list[int],
    col0: int,
) -> list[np.ndarray]:
    placed = []
    for obj, top in zip(local_objects, row_tops):
        placed.append(obj + np.array([top, col0], dtype=int))
    return placed


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert bool(np.all(g[0, :] == TOP_COLOR))
    assert bool(np.all(g[h - 1, :] == BOTTOM_COLOR))
    assert set(map(int, np.unique(g))) <= {BG, TOP_COLOR, BOTTOM_COLOR}

    red_objects = _extract_color_objects(g, TOP_COLOR)
    blue_objects = _extract_color_objects(g, BOTTOM_COLOR)
    assert 2 <= len(red_objects) <= 3
    assert len(red_objects) == len(blue_objects)

    all_objects = red_objects + blue_objects
    for obj in all_objects:
        assert 3 <= len(obj) <= 6, "object area must stay in the puzzle range"
    for i in range(len(all_objects)):
        for j in range(i + 1, len(all_objects)):
            assert not _touches8(all_objects[i], all_objects[j]), "input objects must be visually separated"

    moved_red = [_slide_red(obj) for obj in red_objects]
    moved_blue = [_slide_blue(obj, h) for obj in blue_objects]
    overlap, masks = _overlap_mask(moved_red + moved_blue, (h, w))
    comps = _component_coords(overlap)
    assert 2 <= len(comps) <= 3, "generator must produce 2-3 overlap regions"

    bboxes = [_bbox(comp) for comp in comps]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a = bboxes[i]
            b = bboxes[j]
            overlap_rows = not (a[2] < b[0] or b[2] < a[0])
            overlap_cols = not (a[3] < b[1] or b[3] < a[1])
            assert not (overlap_rows and overlap_cols), "overlap-region bboxes must not overlap"

    for mask in masks:
        assert bool(np.any(mask & overlap)), "every object must contribute to some overlap"

    red_only_overlap = _overlap_cells(moved_red + blue_objects, h, w)
    blue_only_overlap = _overlap_cells(red_objects + moved_blue, h, w)
    final_overlap_cells = int(overlap.sum())
    assert 0 < int(red_only_overlap.sum()) < final_overlap_cells, "red slide must contribute a strict subset of overlaps"
    assert 0 < int(blue_only_overlap.sum()) < final_overlap_cells, "blue slide must contribute a strict subset of overlaps"
    assert not np.array_equal(red_only_overlap, overlap), "red-only slide cannot already determine the full result"
    assert not np.array_equal(blue_only_overlap, overlap), "blue-only slide cannot already determine the full result"

    out = main(g)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert set(map(int, np.unique(out))) <= {BG, OVERLAP_COLOR}
    assert int((out == OVERLAP_COLOR).sum()) == int(overlap.sum())

    ordered_comps = sorted(comps, key=lambda comp: _bbox(comp)[:2])
    prev = None
    for comp in ordered_comps:
        bb = _bbox(comp)
        if prev is not None:
            assert bb[0] > prev[0] and bb[1] > prev[1], "top-left order must be unique and visible"
        prev = bb


def _gen_attempt() -> np.ndarray | None:
    total_regions = random.choice([2, 3])
    red_local, blue_local = _choose_groups(total_regions)
    red_h, red_w = _group_dims(red_local)
    blue_h, blue_w = _group_dims(blue_local)

    h = random.randint(max(15, red_h + blue_h + 5), 20)
    gap = random.randint(3, 6)
    min_w = red_w + blue_w + gap
    w_lo = max(12, min_w)
    w_hi = min(18, min_w + 4)
    if w_lo > w_hi:
        return None
    w = random.randint(w_lo, w_hi)

    left_slack = w - min_w
    red_c0 = random.randint(0, left_slack)
    blue_c0 = red_c0 + red_w + gap

    red_tops = _sample_staggered_tops([_object_height(obj) for obj in red_local], 3, h - 2)
    blue_tops = _sample_staggered_tops([_object_height(obj) for obj in blue_local], 2, h - 4)
    if red_tops is None or blue_tops is None:
        return None

    red_objs = _place_group_input(red_local, red_tops, red_c0)
    blue_objs = _place_group_input(blue_local, blue_tops, blue_c0)

    grid = _make_grid(h, w, red_objs, blue_objs)
    try:
        _validate_generated(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

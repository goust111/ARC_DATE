from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

CORNER_SPECS = {
    "NW": ((0, 0), (-1, -1)),
    "NE": ((0, 2), (-1, 1)),
    "SW": ((2, 0), (1, -1)),
    "SE": ((2, 2), (1, 1)),
}
CORNER_NAMES = tuple(CORNER_SPECS)
CORNER_FAMILIES = {
    1: ("NW", "SE"),
    -1: ("NE", "SW"),
}
EDGE_MIDS = ((0, 1), (1, 0), (1, 2), (2, 1))
PLUS_PATTERN = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
LOCAL_SHAPE = (20, 18)
BOX_GAP = 1
RETRY_LIMIT = 600

_ALL_PATTERN_CELLS = [(r, c) for r in range(3) for c in range(3)]
_ALL_PATTERNS = [tuple(sorted(cells)) for cells in combinations(_ALL_PATTERN_CELLS, 5)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _corner_names_for_pattern(pattern: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    cells = set(pattern)
    return tuple(name for name, (offset, _step) in CORNER_SPECS.items() if offset in cells)


def _sample_pattern(required_corner: str, exact_corner_count: int) -> tuple[tuple[int, int], ...]:
    candidates = []
    for pattern in _ALL_PATTERNS:
        rows = [cell[0] for cell in pattern]
        cols = [cell[1] for cell in pattern]
        if max(rows) - min(rows) != 2 or max(cols) - min(cols) != 2:
            continue
        corners = _corner_names_for_pattern(pattern)
        if required_corner not in corners:
            continue
        if len(corners) != int(exact_corner_count):
            continue
        candidates.append(pattern)
    assert candidates, "expected at least one matching 3x3 five-pixel pattern"
    return random.choice(candidates)


def _bbox_from_top_left(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    return int(r0), int(c0), int(r0 + 2), int(c0 + 2)


def _boxes_clear(
    box: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    *,
    gap: int = BOX_GAP,
) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (
            r1 + gap < or0 - gap
            or or1 + gap < r0 - gap
            or c1 + gap < oc0 - gap
            or oc1 + gap < c0 - gap
        ):
            return False
    return True


def _box_in_shape(top_left: tuple[int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = _bbox_from_top_left(top_left)
    return 0 <= r0 and 0 <= c0 and r1 < shape[0] and c1 < shape[1]


def _top_left_from_target(target: tuple[int, int], corner_name: str, gap: int) -> tuple[int, int]:
    (orow, ocol), (dr, dc) = CORNER_SPECS[corner_name]
    corner_r = int(target[0] - gap * dr)
    corner_c = int(target[1] - gap * dc)
    return corner_r - orow, corner_c - ocol


def _shift_coord(coord: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    return int(coord[0] + dr), int(coord[1] + dc)


def _flip_coord(coord: tuple[int, int], shape: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if mode == 0:
        return int(r), int(c)
    if mode == 1:
        return int(h - 1 - r), int(c)
    if mode == 2:
        return int(r), int(w - 1 - c)
    if mode == 3:
        return int(h - 1 - r), int(w - 1 - c)
    raise ValueError(f"unknown flip mode: {mode}")


def _apply_flip(grid: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return grid
    if mode == 1:
        return np.flipud(grid)
    if mode == 2:
        return np.fliplr(grid)
    if mode == 3:
        return np.flipud(np.fliplr(grid))
    raise ValueError(f"unknown flip mode: {mode}")


def _find_masks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    masks = []
    for idx, color in enumerate(int(v) for v in np.unique(g) if int(v) != BLACK):
        coords = np.argwhere(g == color)
        assert int(coords.shape[0]) == 5, "each mask color must occupy exactly five cells"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3, "each mask must live in a 3x3 box"
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        masks.append(
            {
                "id": idx,
                "color": int(color),
                "top_left": (int(r0), int(c0)),
                "pattern": rel,
                "corners": _corner_names_for_pattern(rel),
            }
        )
    assert len(masks) >= 2, "expected at least two masks"
    return masks


def _ray_cells(
    shape: tuple[int, int],
    occupied: np.ndarray,
    top_left: tuple[int, int],
    corner_name: str,
    *,
    stop_at_masks: bool,
) -> set[tuple[int, int]]:
    (orow, ocol), (dr, dc) = CORNER_SPECS[corner_name]
    corner_r = int(top_left[0] + orow)
    corner_c = int(top_left[1] + ocol)
    r = corner_r + dr
    c = corner_c + dc
    cells: set[tuple[int, int]] = set()
    while _in_bounds(shape, r, c):
        if stop_at_masks and int(occupied[r, c]) != 0:
            break
        cells.add((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _solve(
    grid: np.ndarray,
    *,
    stop_at_masks: bool = True,
    occupied_corners_only: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    masks = _find_masks(g)
    occupied = (g != BLACK).astype(np.uint8)
    coverage: dict[tuple[int, int], set[int]] = {}

    for mask in masks:
        corners = mask["corners"] if occupied_corners_only else CORNER_NAMES
        ray_union: set[tuple[int, int]] = set()
        for corner_name in corners:
            ray_union |= _ray_cells(
                g.shape,
                occupied,
                mask["top_left"],
                corner_name,
                stop_at_masks=stop_at_masks,
            )
        for cell in ray_union:
            coverage.setdefault(cell, set()).add(int(mask["id"]))

    out = np.zeros_like(g)
    for (r, c), mask_ids in coverage.items():
        if len(mask_ids) >= 2:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mark only the hidden crossings made by outward diagonal rays from occupied mask corners.

    Input structure:
    - The grid has a black background and several well-separated single-color
      3x3 masks; each color appears exactly five times inside its own 3x3 box.
    - Corner cells and edge-middle cells may both be occupied, but only corner
      occupancy matters for ray emission.

    Rule:
    - For each mask, inspect the four corners of its 3x3 bounding box.
    - Every occupied corner emits one diagonal ray outward from that corner,
      using the corner's natural outward direction; each ray stops just before
      it would hit any mask pixel, or at the grid edge if nothing blocks it.
    - Return an all-black grid of the same size, painting red exactly on cells
      that lie on ray paths from at least two different masks.

    Key insight:
    - The answer is not a visible object from the input; it is the intersection
      set of an implicit ray system, and the usable ray directions come only
      from occupied corners, not from the other three pixels that complete each
      five-cell mask.
    """
    return _solve(grid, stop_at_masks=True, occupied_corners_only=True)


def _render_from_masks(shape: tuple[int, int], masks: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for mask in masks:
        r0, c0 = mask["top_left"]
        for dr, dc in mask["pattern"]:
            grid[r0 + dr, c0 + dc] = int(mask["color"])
    return grid


def _build_cluster_one(boxes: list[tuple[int, int, int, int]]) -> tuple[list[dict], dict] | None:
    shape = LOCAL_SHAPE
    for _ in range(160):
        r0 = random.randint(1, 3)
        c0 = random.randint(1, 3)
        main_box = _bbox_from_top_left((r0, c0))
        if not _boxes_clear(main_box, boxes):
            continue

        g_visible = random.randint(4, 6)
        g_block = g_visible + random.randint(2, 3)
        g_hidden = g_block + random.randint(2, 3)
        se_corner = (r0 + 2, c0 + 2)
        t_visible = (se_corner[0] + g_visible, se_corner[1] + g_visible)
        t_hidden = (se_corner[0] + g_hidden, se_corner[1] + g_hidden)
        blocker_center = (se_corner[0] + g_block, se_corner[1] + g_block)

        b_corner = random.choice(("SW", "NE"))
        c_corner = random.choice(("SW", "NE"))
        b_gap = random.randint(3, 4)
        c_gap = random.randint(3, 4)
        b_top_left = _top_left_from_target(t_visible, b_corner, b_gap)
        c_top_left = _top_left_from_target(t_hidden, c_corner, c_gap)
        blocker_top_left = (blocker_center[0] - 1, blocker_center[1] - 1)

        candidate_boxes = [main_box]
        candidate_masks = [
            {"name": "A", "top_left": (r0, c0), "required_corner": "SE"},
            {"name": "B", "top_left": b_top_left, "required_corner": b_corner},
            {"name": "C", "top_left": c_top_left, "required_corner": c_corner},
            {"name": "Z", "top_left": blocker_top_left, "required_corner": None},
        ]
        ok = True
        for mask in candidate_masks[1:]:
            if not _box_in_shape(mask["top_left"], shape):
                ok = False
                break
            box = _bbox_from_top_left(mask["top_left"])
            if not _boxes_clear(box, boxes + candidate_boxes):
                ok = False
                break
            candidate_boxes.append(box)
        if not ok:
            continue
        if not all(_in_bounds(shape, *coord) for coord in (t_visible, t_hidden, blocker_center)):
            continue
        return candidate_masks, {"t1": t_visible, "t2": t_hidden}
    return None


def _build_cluster_two(boxes: list[tuple[int, int, int, int]]) -> tuple[list[dict], dict] | None:
    shape = LOCAL_SHAPE
    for _ in range(160):
        r0 = random.randint(15, 17)
        c0 = random.randint(1, 3)
        main_box = _bbox_from_top_left((r0, c0))
        if not _boxes_clear(main_box, boxes):
            continue

        g_visible = random.randint(4, 6)
        g_hidden = g_visible + random.randint(2, 3)
        ne_corner = (r0, c0 + 2)
        t_visible = (ne_corner[0] - g_visible, ne_corner[1] + g_visible)
        t_hidden = (ne_corner[0] - g_hidden, ne_corner[1] + g_hidden)

        e_corner = random.choice(("NW", "SE"))
        f_corner = random.choice(("NW", "SE"))
        e_gap = random.randint(3, 4)
        f_gap = random.randint(3, 4)
        e_top_left = _top_left_from_target(t_visible, e_corner, e_gap)
        f_top_left = _top_left_from_target(t_hidden, f_corner, f_gap)

        candidate_boxes = [main_box]
        candidate_masks = [
            {"name": "D", "top_left": (r0, c0), "required_corner": "NE"},
            {"name": "E", "top_left": e_top_left, "required_corner": e_corner},
            {"name": "F", "top_left": f_top_left, "required_corner": None},
        ]
        ok = True
        for mask in candidate_masks[1:]:
            if not _box_in_shape(mask["top_left"], shape):
                ok = False
                break
            box = _bbox_from_top_left(mask["top_left"])
            if not _boxes_clear(box, boxes + candidate_boxes):
                ok = False
                break
            candidate_boxes.append(box)
        if not ok:
            continue
        if not all(_in_bounds(shape, *coord) for coord in (t_visible, t_hidden)):
            continue
        return candidate_masks, {"t3": t_visible, "t4": t_hidden}
    return None


def _assign_patterns(raw_masks: list[dict]) -> list[dict]:
    exact_counts = {
        "A": 4,
        "B": 1,
        "C": 1,
        "D": random.choice((1, 2)),
        "E": 1,
        "F": 0,
        "Z": 0,
    }
    colors = random.sample(INPUT_COLORS, len(raw_masks))
    result = []
    for color, raw in zip(colors, raw_masks):
        if raw["name"] in {"F", "Z"}:
            pattern = PLUS_PATTERN
        else:
            pattern = _sample_pattern(raw["required_corner"], exact_counts[raw["name"]])
        result.append(
            {
                "name": raw["name"],
                "color": int(color),
                "top_left": tuple(map(int, raw["top_left"])),
                "pattern": pattern,
            }
        )
    return result


def _build_final_grid(masks: list[dict], targets: dict[str, tuple[int, int]]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    top = random.randint(0, 5)
    bottom = random.randint(0, 5)
    left = random.randint(0, 6)
    right = random.randint(0, 6)
    height = LOCAL_SHAPE[0] + top + bottom
    width = LOCAL_SHAPE[1] + left + right
    if height > 30 or width > 30:
        return np.zeros((1, 1), dtype=int), {}

    shifted_masks = []
    for mask in masks:
        r0, c0 = mask["top_left"]
        shifted_masks.append({**mask, "top_left": (r0 + top, c0 + left)})

    shifted_targets = {key: _shift_coord(coord, top, left) for key, coord in targets.items()}
    grid = _render_from_masks((height, width), shifted_masks)

    flip_mode = random.randint(0, 3)
    flipped_grid = _apply_flip(grid, flip_mode)
    flipped_targets = {
        key: _flip_coord(coord, flipped_grid.shape, flip_mode)
        for key, coord in shifted_targets.items()
    }
    return flipped_grid, flipped_targets


def _semantic_checks(grid: np.ndarray, targets: dict[str, tuple[int, int]]) -> bool:
    actual = main(grid)
    no_stop = _solve(grid, stop_at_masks=False, occupied_corners_only=True)
    all_corners = _solve(grid, stop_at_masks=True, occupied_corners_only=False)

    if int(np.count_nonzero(actual == RED)) != 2:
        return False
    if actual[targets["t1"]] != RED or actual[targets["t3"]] != RED:
        return False
    if actual[targets["t2"]] != BLACK or actual[targets["t4"]] != BLACK:
        return False
    if no_stop[targets["t2"]] != RED:
        return False
    if all_corners[targets["t4"]] != RED:
        return False
    if np.array_equal(actual, no_stop):
        return False
    if np.array_equal(actual, all_corners):
        return False

    masks = _find_masks(grid)
    corner_counts = sorted(len(mask["corners"]) for mask in masks)
    if corner_counts.count(0) < 2:
        return False
    if 4 not in corner_counts:
        return False
    if 1 not in corner_counts:
        return False
    if len(set(corner_counts)) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cluster_one = _build_cluster_one([])
    if cluster_one is None:
        return None
    masks_one, info_one = cluster_one
    boxes = [_bbox_from_top_left(mask["top_left"]) for mask in masks_one]

    cluster_two = _build_cluster_two(boxes)
    if cluster_two is None:
        return None
    masks_two, info_two = cluster_two

    masks = _assign_patterns(masks_one + masks_two)
    targets = {**info_one, **info_two}
    grid, transformed_targets = _build_final_grid(masks, targets)
    if not transformed_targets:
        return None

    try:
        if not _semantic_checks(grid, transformed_targets):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the hidden-corner-ray puzzle")

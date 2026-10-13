from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
FILL = 5
BOX = 4
SIDE_NAMES = ("top", "bottom", "left", "right")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_SUBSETS = (
    ("top", "bottom", "left", "right"),
    ("bottom", "left", "right"),
    ("top", "left", "right"),
    ("top", "bottom", "right"),
    ("top", "bottom", "left"),
)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

_GEN_CALL_COUNT = 0


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    dirs = DIRS8 if connectivity == 8 else DIRS4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _completion_parts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    fill = np.zeros((h, w), dtype=bool)
    hole = np.zeros((h, w), dtype=bool)
    for comp in _components(~mask, connectivity=4):
        target = fill if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp) else hole
        for r, c in comp:
            target[r, c] = True
    return fill, hole


def _completed_patch(mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fill, hole = _completion_parts(mask)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    patch[fill] = FILL
    return patch, fill, hole


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _frame_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, int], dict[int, str]]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "expected a colored frame and interior objects"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    top = int(grid[r0, c0 + 1])
    bottom = int(grid[r1, c0 + 1])
    left = int(grid[r0 + 1, c0])
    right = int(grid[r0 + 1, c1])

    assert np.all(grid[r0, c0 + 1 : c1] == top)
    assert np.all(grid[r1, c0 + 1 : c1] == bottom)
    assert np.all(grid[r0 + 1 : r1, c0] == left)
    assert np.all(grid[r0 + 1 : r1, c1] == right)

    side_colors = {"top": top, "bottom": bottom, "left": left, "right": right}
    assert len(set(side_colors.values())) == 4, "frame sides must have four distinct colors"
    return (r0, c0, r1, c1), side_colors, {color: side for side, color in side_colors.items()}


def _destination_bbox(
    bbox: tuple[int, int, int, int],
    side: str,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inner_top, inner_bottom = fr0 + 1, fr1 - 1
    inner_left, inner_right = fc0 + 1, fc1 - 1

    if side == "top":
        nr0, nc0 = inner_top, c0
    elif side == "bottom":
        nr0, nc0 = inner_bottom - h + 1, c0
    elif side == "left":
        nr0, nc0 = r0, inner_left
    elif side == "right":
        nr0, nc0 = r0, inner_right - w + 1
    else:
        raise AssertionError(f"unknown side: {side}")
    return nr0, nc0, nr0 + h - 1, nc0 + w - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete interior object and slide the completed body to the matching-colored frame side.

    Input structure:
    - A black background contains one thin rectangular frame whose four sides use four distinct colors.
    - Inside the frame are 3 or 4 separated 8-connected monochromatic objects, each fitting in a 4x4 bounding box.
    - Every object uses exactly one of the frame-side colors and may contain edge-connected missing cells plus enclosed black holes.

    Rule:
    - Detect every interior 8-connected object and inspect its bounding box.
    - Inside that box, keep black regions that are 4-connected and do not touch the box edge as holes; recolor every other black cell in the box to grey to complete the object.
    - Match the object's original color to the frame side of the same color, erase all interior objects from their old positions, then translate the completed object so its corresponding box edge touches that frame side's inner row or column while preserving the orthogonal coordinate.

    Key insight:
    - The movement direction is not geometric; it is encoded by the color match with the frame, and the object must be completed before moving so the grey completion pixels travel with the whole body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g >= 0), "expected ARC color ids"

    frame_bbox, _side_colors, color_to_side = _frame_info(g)
    fr0, fc0, fr1, fc1 = frame_bbox

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = g[fr0 + 1 : fr1, fc0 + 1 : fc1] != BG
    objects = _components(interior_mask, connectivity=8)
    assert objects, "expected at least one interior object"

    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG

    for comp in objects:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        colors = set()
        for r, c in comp:
            mask[r - r0, c - c0] = True
            colors.add(int(g[r, c]))

        assert len(colors) == 1, "each interior object must be monochromatic"
        color = next(iter(colors))
        assert color in color_to_side, "every object color must match a frame side"

        patch, _fill, _hole = _completed_patch(mask, color)
        nr0, nc0, _nr1, _nc1 = _destination_bbox(bbox, color_to_side[color], frame_bbox)
        nonzero = patch != BG
        target = out[nr0 : nr0 + patch.shape[0], nc0 : nc0 + patch.shape[1]]
        assert np.all(target[nonzero] == BG), "moved objects must not overlap"
        target[nonzero] = patch[nonzero]

    return out


def _pattern_library() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with_hole: list[dict[str, object]] = []
    without_hole: list[dict[str, object]] = []
    for missing_count in (1, 2, 3):
        for black_cells in combinations(range(BOX * BOX), missing_count):
            mask = np.ones((BOX, BOX), dtype=bool)
            mask.flat[list(black_cells)] = False
            if len(_components(mask, connectivity=8)) != 1:
                continue
            fill, hole = _completion_parts(mask)
            if not np.any(fill):
                continue
            entry = {
                "mask": mask,
                "fill_count": int(np.sum(fill)),
                "hole_count": int(np.sum(hole)),
            }
            if np.any(hole):
                with_hole.append(entry)
            else:
                without_hole.append(entry)
    assert with_hole and without_hole
    return with_hole, without_hole


WITH_HOLE_PATTERNS, WITHOUT_HOLE_PATTERNS = _pattern_library()


def _render_frame(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 + 1 : c1] = side_colors["top"]
    grid[r1, c0 + 1 : c1] = side_colors["bottom"]
    grid[r0 + 1 : r1, c0] = side_colors["left"]
    grid[r0 + 1 : r1, c1] = side_colors["right"]
    grid[r0, c0] = random.choice((side_colors["top"], side_colors["left"]))
    grid[r0, c1] = random.choice((side_colors["top"], side_colors["right"]))
    grid[r1, c0] = random.choice((side_colors["bottom"], side_colors["left"]))
    grid[r1, c1] = random.choice((side_colors["bottom"], side_colors["right"]))
    return grid


def _candidate_positions(
    side: str,
    active_sides: set[str],
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_top, inner_bottom = fr0 + 1, fr1 - 1
    inner_left, inner_right = fc0 + 1, fc1 - 1

    row_min = inner_top + 1
    row_max = inner_bottom - BOX
    col_min = inner_left + 1
    col_max = inner_right - BOX

    if side == "top":
        row_min = max(row_min, inner_top + BOX)
    elif side == "bottom":
        row_max = min(row_max, inner_bottom - 2 * BOX + 1)
    elif side == "left":
        col_min = max(col_min, inner_left + BOX)
    elif side == "right":
        col_max = min(col_max, inner_right - 2 * BOX + 1)

    if side in {"top", "bottom"}:
        if "left" in active_sides:
            col_min = max(col_min, inner_left + BOX)
        if "right" in active_sides:
            col_max = min(col_max, inner_right - 2 * BOX + 1)
    else:
        if "top" in active_sides:
            row_min = max(row_min, inner_top + BOX)
        if "bottom" in active_sides:
            row_max = min(row_max, inner_bottom - 2 * BOX + 1)

    if row_min > row_max or col_min > col_max:
        return []
    return [(r, c) for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)]


def _place_objects(
    sides: tuple[str, ...],
    frame_bbox: tuple[int, int, int, int],
) -> dict[str, tuple[int, int]] | None:
    active = set(sides)
    candidate_map = {side: _candidate_positions(side, active, frame_bbox) for side in sides}
    if any(not positions for positions in candidate_map.values()):
        return None

    ordered_sides = sorted(sides, key=lambda side: len(candidate_map[side]))
    placements: dict[str, tuple[int, int]] = {}

    def backtrack(i: int) -> bool:
        if i == len(ordered_sides):
            return True
        side = ordered_sides[i]
        choices = candidate_map[side][:]
        random.shuffle(choices)
        for r, c in choices:
            rect = (r, c, r + BOX - 1, c + BOX - 1)
            if any(
                _rects_too_close(rect, (pr, pc, pr + BOX - 1, pc + BOX - 1), pad=1)
                for pr, pc in placements.values()
            ):
                continue
            placements[side] = (r, c)
            if backtrack(i + 1):
                return True
            del placements[side]
        return False

    return placements if backtrack(0) else None


def _distance_to_sides(
    bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_top, inner_bottom = fr0 + 1, fr1 - 1
    inner_left, inner_right = fc0 + 1, fc1 - 1
    return {
        "top": r0 - inner_top,
        "bottom": inner_bottom - r1,
        "left": c0 - inner_left,
        "right": inner_right - c1,
    }


def _build_case(call_index: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], tuple[int, int, int, int]]:
    sides = SIDE_SUBSETS[call_index % len(SIDE_SUBSETS)]
    inner_h = random.randint(14, 16)
    inner_w = random.randint(14, 16)
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)

    h = inner_h + margin_top + margin_bottom + 2
    w = inner_w + margin_left + margin_right + 2
    frame_bbox = (
        margin_top,
        margin_left,
        margin_top + inner_h + 1,
        margin_left + inner_w + 1,
    )

    side_palette = random.sample(PALETTE, 4)
    side_colors = dict(zip(SIDE_NAMES, side_palette))
    placements = _place_objects(sides, frame_bbox)
    if placements is None:
        raise AssertionError("failed to place objects")

    hole_count = random.randint(1, len(sides) - 1)
    hole_sides = set(random.sample(list(sides), hole_count))

    specs: list[dict[str, object]] = []
    for side in sides:
        pattern_pool = WITH_HOLE_PATTERNS if side in hole_sides else WITHOUT_HOLE_PATTERNS
        pattern = random.choice(pattern_pool)
        mask = np.array(pattern["mask"], dtype=bool, copy=True)
        color = side_colors[side]
        r, c = placements[side]
        bbox = (r, c, r + BOX - 1, c + BOX - 1)
        patch, fill, hole = _completed_patch(mask, color)
        dest_bbox = _destination_bbox(bbox, side, frame_bbox)
        specs.append(
            {
                "side": side,
                "color": color,
                "mask": mask,
                "bbox": bbox,
                "dest_bbox": dest_bbox,
                "patch": patch,
                "fill_count": int(np.sum(fill)),
                "hole_count": int(np.sum(hole)),
            }
        )

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            assert not _rects_too_close(specs[i]["dest_bbox"], specs[j]["dest_bbox"], pad=1)

    inp = _render_frame(h, w, frame_bbox, side_colors)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        mask = np.asarray(spec["mask"], dtype=bool)
        region = inp[r0 : r1 + 1, c0 : c1 + 1]
        region[mask] = color

    out = _render_frame(h, w, frame_bbox, side_colors)
    for spec in specs:
        r0, c0, r1, c1 = spec["dest_bbox"]
        patch = np.asarray(spec["patch"], dtype=int)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(region[patch != BG] == BG)
        region[patch != BG] = patch[patch != BG]

    return inp, out, specs, frame_bbox


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: list[dict[str, object]],
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    sides = {str(spec["side"]) for spec in specs}
    if not ({"top", "bottom"} & sides and {"left", "right"} & sides):
        return False
    if not any(int(spec["hole_count"]) > 0 for spec in specs):
        return False
    if not any(int(spec["hole_count"]) == 0 for spec in specs):
        return False
    if not all(int(spec["fill_count"]) >= 1 for spec in specs):
        return False

    nearest_counterexample = False
    for spec in specs:
        bbox = spec["bbox"]
        patch = np.asarray(spec["patch"], dtype=int)
        src_r0, src_c0, src_r1, src_c1 = bbox
        dst_r0, dst_c0, dst_r1, dst_c1 = spec["dest_bbox"]
        color = int(spec["color"])
        side = str(spec["side"])
        mask = np.asarray(spec["mask"], dtype=bool)

        if not np.array_equal(inp[src_r0 : src_r1 + 1, src_c0 : src_c1 + 1] == color, mask):
            return False
        if not np.array_equal(expected[dst_r0 : dst_r1 + 1, dst_c0 : dst_c1 + 1], patch):
            return False

        distances = _distance_to_sides(bbox, frame_bbox)
        target_distance = distances[side]
        if target_distance > min(distances.values()):
            nearest_counterexample = True

    if not nearest_counterexample:
        return False

    grey_total = sum(int(spec["fill_count"]) for spec in specs)
    if int(np.sum(expected == FILL)) != grey_total:
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(500):
        try:
            inp, expected, specs, frame_bbox = _build_case(_GEN_CALL_COUNT)
        except AssertionError:
            continue
        if _validate_case(inp, expected, specs, frame_bbox):
            _GEN_CALL_COUNT += 1
            return inp
    raise RuntimeError("failed to generate a valid input")

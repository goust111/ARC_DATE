from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask  # noqa: E402


BACKGROUND = Color.BLACK
START_COLOR = Color.RED
TARGET_COLOR = Color.TEAL
RECT_COLOR = Color.YELLOW
PATH_COLOR = Color.ORANGE

BLOCK_SIZE = 2
GRID_MIN = 18
GRID_MAX = 25
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 500


def _block_bbox(top: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = top
    return int(r), int(c), int(r + BLOCK_SIZE - 1), int(c + BLOCK_SIZE - 1)


def _bbox_with_margin(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _extract_blocks(grid: np.ndarray, color: int) -> list[dict]:
    blocks = []
    for coords in components_mask(grid == int(color), connectivity=4):
        assert coords.shape[0] == 4, f"expected 2x2 block area for color {color}"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0, c1 - c0) == (1, 1), f"expected 2x2 block bbox for color {color}"
        blocks.append({"bbox": bbox, "top": (r0, c0)})
    return sorted(blocks, key=lambda block: block["top"])


def _find_start_and_targets(grid: np.ndarray) -> tuple[dict, list[dict]]:
    starts = _extract_blocks(grid, START_COLOR)
    targets = _extract_blocks(grid, TARGET_COLOR)
    assert len(starts) == 1, "expected exactly one red 2x2 start block"
    assert 3 <= len(targets) <= 5, "expected 3-5 teal 2x2 target blocks"
    return starts[0], targets


def _completable_rect_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for coords in components_mask(grid == RECT_COLOR, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BACKGROUND):
            continue
        if not np.all(np.isin(patch, [RECT_COLOR, PATH_COLOR])):
            continue
        boxes.append(bbox)
    return sorted(set(boxes))


def _complete_rectangles(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r0, c0, r1, c1 in _completable_rect_bboxes(out):
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == PATH_COLOR] = RECT_COLOR
    return out


def _visible_step(grid: np.ndarray, current: dict, target: dict) -> tuple[str, int] | None:
    ar0, ac0, ar1, ac1 = current["bbox"]
    br0, bc0, br1, bc1 = target["bbox"]

    if ar0 == br0 and ar1 == br1:
        if ac1 < bc0:
            c0, c1 = ac1 + 1, bc0 - 1
        elif bc1 < ac0:
            c0, c1 = bc1 + 1, ac0 - 1
        else:
            return None
        corridor = grid[ar0 : ar1 + 1, c0 : c1 + 1]
        if np.all(corridor == BACKGROUND):
            return "h", int(c1 - c0 + 1)
        return None

    if ac0 == bc0 and ac1 == bc1:
        if ar1 < br0:
            r0, r1 = ar1 + 1, br0 - 1
        elif br1 < ar0:
            r0, r1 = br1 + 1, ar0 - 1
        else:
            return None
        corridor = grid[r0 : r1 + 1, ac0 : ac1 + 1]
        if np.all(corridor == BACKGROUND):
            return "v", int(r1 - r0 + 1)
        return None

    return None


def _paint_corridor(out: np.ndarray, current: dict, target: dict, axis: str) -> None:
    ar0, ac0, ar1, ac1 = current["bbox"]
    br0, bc0, br1, bc1 = target["bbox"]

    if axis == "h":
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1:
            corridor = out[ar0 : ar1 + 1, c0 : c1 + 1]
            corridor[corridor == BACKGROUND] = PATH_COLOR
        return

    assert axis == "v"
    r0 = min(ar1, br1) + 1
    r1 = max(ar0, br0) - 1
    if r0 <= r1:
        corridor = out[r0 : r1 + 1, ac0 : ac1 + 1]
        corridor[corridor == BACKGROUND] = PATH_COLOR


def _greedy_walk(visibility_grid: np.ndarray, start: dict, targets: list[dict]) -> list[tuple[dict, str]]:
    current = start
    remaining = list(targets)
    walk = []

    while True:
        candidates = []
        for target in remaining:
            step = _visible_step(visibility_grid, current, target)
            if step is None:
                continue
            axis, distance = step
            candidates.append((distance, target["top"][0], target["top"][1], axis, target))

        if not candidates:
            break

        _distance, _row, _col, axis, target = min(candidates)
        walk.append((target, axis))
        remaining = [candidate for candidate in remaining if candidate is not target]
        current = target

    return walk


def _solve_from_completed(completed: np.ndarray) -> np.ndarray:
    start, targets = _find_start_and_targets(completed)
    walk = _greedy_walk(completed, start, targets)
    out = completed.copy()
    current = start
    for target, axis in walk:
        _paint_corridor(out, current, target, axis)
        current = target
    return out


def _walk_signature(grid: np.ndarray) -> list[tuple[int, int]]:
    completed = _complete_rectangles(grid)
    start, targets = _find_start_and_targets(completed)
    return [target["top"] for target, _axis in _greedy_walk(completed, start, targets)]


def _naive_shortcut_grid(grid: np.ndarray) -> np.ndarray:
    """
    Anti-shortcut oracle used only by the generator.

    It removes the full footprints of completable yellow/orange rectangles to ensure
    the rectangle-completion stage materially changes the visibility graph.
    """
    naive = np.asarray(grid, dtype=int).copy()
    for bbox in _completable_rect_bboxes(grid):
        _paint_box(naive, bbox, BACKGROUND)
    return naive


def _naive_walk_signature(grid: np.ndarray) -> list[tuple[int, int]]:
    naive = _naive_shortcut_grid(grid)
    start, targets = _find_start_and_targets(grid)
    return [target["top"] for target, _axis in _greedy_walk(naive, start, targets)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partially occluded yellow rectangles, then draw the greedy visible-target corridor walk.

    Input structure:
    - The grid has a black background, one red 2x2 start block, 3-5 teal 2x2 target blocks, and 2-3 yellow filled rectangles.
    - Some yellow rectangles are partially occluded by orange pixels, but each completable rectangle's full bbox is still filled entirely by yellow/orange cells.
    - Start and target blocks are separate from the rectangles; visibility is checked only between exactly aligned 2x2 row-bands or column-bands.

    Rule:
    - Find each 4-connected yellow region; if its bbox contains no black pixels and only yellow/orange values, recolor the orange cells in that bbox back to yellow.
    - On the completed grid, start from the red block and repeatedly choose the visible unvisited teal block in the same two-row or two-column band with minimum gap distance, breaking ties by smaller top row then smaller left column.
    - For each chosen step, paint the black cells of the 2-cell-wide corridor between the current block and that target orange, then continue from that target.
    - Stop when no visible unvisited target remains; preserve all original blocks and completed rectangles.

    Key insight:
    - The yellow-rectangle completion must be resolved before the greedy walk, because the completed foreground layout defines the visibility graph used by the corridor traversal.
    """
    completed = _complete_rectangles(np.asarray(grid, dtype=int))
    return _solve_from_completed(completed)


def _corridor_box(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    ar0, ac0, ar1, ac1 = _block_bbox(a)
    br0, bc0, br1, bc1 = _block_bbox(b)
    if ar0 == br0:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        return ar0, c0, ar1, c1
    assert ac0 == bc0
    r0 = min(ar1, br1) + 1
    r1 = max(ar0, br0) - 1
    return r0, ac0, r1, ac1


def _mark_reserved(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = _bbox_with_margin(box, pad)
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h - 1, r1)
    c1 = min(w - 1, c1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_occlusion_patch(
    bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
    *,
    required_rows: tuple[int, int] | None = None,
    required_cols: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    inner_r0, inner_r1 = r0 + 1, r1 - 1
    inner_c0, inner_c1 = c0 + 1, c1 - 1
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1, "rectangle must admit an interior occlusion patch"

    if required_rows is None:
        pr0 = int(rng.integers(inner_r0, inner_r1 + 1))
        pr1 = int(rng.integers(pr0, inner_r1 + 1))
    else:
        need_r0, need_r1 = required_rows
        pr0 = max(inner_r0, need_r0)
        pr1 = min(inner_r1, need_r1)
        assert pr0 <= pr1

    if required_cols is None:
        pc0 = int(rng.integers(inner_c0, inner_c1 + 1))
        pc1 = int(rng.integers(pc0, inner_c1 + 1))
    else:
        need_c0, need_c1 = required_cols
        pc0 = max(inner_c0, need_c0)
        pc1 = min(inner_c1, need_c1)
        assert pc0 <= pc1

    return pr0, pc0, pr1, pc1


def _paint_occluded_rectangle(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    patch: tuple[int, int, int, int],
) -> None:
    _paint_box(grid, bbox, RECT_COLOR)
    _paint_box(grid, patch, PATH_COLOR)


def _sample_filler_rectangle(
    grid_shape: tuple[int, int],
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = grid_shape
    for _ in range(200):
        rh = int(rng.integers(*RECT_H_RANGE))
        rw = int(rng.integers(*RECT_W_RANGE))
        r0 = int(rng.integers(1, h - rh))
        c0 = int(rng.integers(1, w - rw))
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        r0p, c0p, r1p, c1p = _bbox_with_margin(bbox, 1)
        r0p = max(0, r0p)
        c0p = max(0, c0p)
        r1p = min(h - 1, r1p)
        c1p = min(w - 1, c1p)
        if np.any(reserved[r0p : r1p + 1, c0p : c1p + 1]):
            continue
        patch = _random_occlusion_patch(bbox, rng)
        return bbox, patch
    return None


def _place_extra_target(
    tops: list[tuple[int, int]],
    reserved_boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    forbidden_rows = {r for r, _c in tops}
    forbidden_cols = {c for _r, c in tops}
    for _ in range(300):
        top = (int(rng.integers(1, h - 2)), int(rng.integers(1, w - 2)))
        if top[0] in forbidden_rows or top[1] in forbidden_cols:
            continue
        box = _block_bbox(top)
        if any(_bboxes_overlap(_bbox_with_margin(box, 1), _bbox_with_margin(other, 1)) for other in reserved_boxes):
            continue
        return top
    return None


def _build_variant(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    grid = np.full((h, w), BACKGROUND, dtype=int)

    rs = int(rng.integers(7, h - 9))
    cs = int(rng.integers(7, w - 9))
    a_dir = int(rng.choice([-1, 1]))
    b_dir = int(rng.choice([-1, 1]))
    ga = int(rng.integers(4, 7))
    gb = int(rng.integers(ga + 1, ga + 5))
    gc = int(rng.integers(3, 7))

    start = (rs, cs)
    decoy = (rs + a_dir * (BLOCK_SIZE + ga), cs)
    first = (rs, cs + b_dir * (BLOCK_SIZE + gb))

    c_dir = -a_dir
    second = (first[0] + c_dir * (BLOCK_SIZE + gc), first[1])
    if not (1 <= decoy[0] <= h - 3 and 1 <= first[1] <= w - 3 and 1 <= second[0] <= h - 3):
        return None

    targets = [decoy, first, second]
    if rng.random() < 0.65:
        gd = int(rng.integers(3, 7))
        d_dir = -b_dir
        third = (second[0], second[1] + d_dir * (BLOCK_SIZE + gd))
        if 1 <= third[1] <= w - 3:
            targets.append(third)

    block_boxes = [_block_bbox(start)] + [_block_bbox(top) for top in targets]
    for i, box in enumerate(block_boxes):
        if box[0] < 1 or box[1] < 1 or box[2] >= h - 1 or box[3] >= w - 1:
            return None
        for other in block_boxes[i + 1 :]:
            if _bboxes_overlap(_bbox_with_margin(box, 1), _bbox_with_margin(other, 1)):
                return None

    for box in block_boxes:
        color = START_COLOR if box == block_boxes[0] else TARGET_COLOR
        _paint_box(grid, box, color)

    reserved = np.zeros_like(grid, dtype=bool)
    for box in block_boxes:
        _mark_reserved(reserved, box, pad=1)
    walk_chain = [start, first, second]
    if len(targets) >= 4:
        walk_chain.append(targets[3])
    for a, b in zip(walk_chain, walk_chain[1:]):
        corridor = _corridor_box(a, b)
        if corridor[0] <= corridor[2] and corridor[1] <= corridor[3]:
            _mark_reserved(reserved, corridor, pad=1)

    gap_r0 = min(start[0], decoy[0]) + BLOCK_SIZE
    gap_r1 = max(start[0], decoy[0]) - 1
    if gap_r1 - gap_r0 + 1 < 3:
        return None
    rh = int(rng.integers(3, min(6, gap_r1 - gap_r0 + 2)))
    rect_r0 = int(rng.integers(gap_r0, gap_r1 - rh + 2))
    left_extra = int(rng.integers(1, 3))
    right_extra = int(rng.integers(1, 4))
    rect_c0 = cs - left_extra
    rect_c1 = cs + 1 + right_extra
    if rect_c0 < 1 or rect_c1 >= w - 1:
        return None
    key_bbox = (rect_r0, rect_c0, rect_r0 + rh - 1, rect_c1)
    key_patch = _random_occlusion_patch(
        key_bbox,
        rng,
        required_cols=(cs, cs + 1),
    )
    _paint_occluded_rectangle(grid, key_bbox, key_patch)
    _mark_reserved(reserved, key_bbox, pad=1)

    filler_total = int(rng.integers(1, 3))
    for _ in range(filler_total):
        sample = _sample_filler_rectangle(grid.shape, reserved, rng)
        if sample is None:
            return None
        bbox, patch = sample
        _paint_occluded_rectangle(grid, bbox, patch)
        _mark_reserved(reserved, bbox, pad=1)

    if len(targets) < 5 and rng.random() < 0.7:
        extra = _place_extra_target([start, *targets], block_boxes + [key_bbox], h, w, rng)
        if extra is not None:
            targets.append(extra)
            box = _block_bbox(extra)
            _paint_box(grid, box, TARGET_COLOR)

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    try:
        completed = _complete_rectangles(grid)
        output = main(grid)
        start, targets = _find_start_and_targets(completed)
    except Exception:
        return False

    if not np.array_equal(output, _solve_from_completed(completed)):
        return False
    if not _completable_rect_bboxes(grid):
        return False
    if np.array_equal(grid, completed):
        return False

    walk = _greedy_walk(completed, start, targets)
    if len(walk) < 2:
        return False

    actual_sig = [target["top"] for target, _axis in walk]
    naive_sig = _naive_walk_signature(grid)
    if actual_sig == naive_sig:
        return False
    if not naive_sig or naive_sig[0] == actual_sig[0]:
        return False

    if np.count_nonzero(output == PATH_COLOR) == np.count_nonzero(grid == PATH_COLOR):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_variant(np.random.default_rng())
        if grid is not None and _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

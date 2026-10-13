from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GAP_SIDES = ("top", "bottom", "left", "right")
COMPARE_LENGTHS = tuple(range(2, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).reshape(-1)
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_components(
    grid: np.ndarray, excluded_colors: set[int]
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if visited[r, c]:
                continue
            visited[r, c] = True
            color = int(g[r, c])
            if color in excluded_colors:
                continue

            q = deque([(r, c)])
            coords = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    coords.append((nr, nc))

            coord_arr = np.asarray(coords, dtype=int)
            bbox = _bbox_from_coords(coord_arr)
            r0, c0, r1, c1 = bbox
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coord_arr[:, 0] - r0, coord_arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(coord_arr.shape[0]),
                }
            )

    return components


def _gap_run(side_mask: np.ndarray) -> tuple[int, int] | None | bool:
    zeros = np.flatnonzero(~side_mask)
    if zeros.size == 0:
        return None
    if zeros[0] == 0 or zeros[-1] == side_mask.size - 1:
        return False
    if zeros.size != zeros[-1] - zeros[0] + 1:
        return False
    return int(zeros[0]), int(zeros[-1])


def _frame_candidate(component: dict[str, object]) -> dict[str, object] | None:
    mask = np.asarray(component["mask"], dtype=bool)
    height, width = mask.shape
    if height < 3 or width < 3:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None

    sides = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    gaps = []
    for side, side_mask in sides.items():
        run = _gap_run(side_mask)
        if run is False:
            return None
        if run is not None:
            gaps.append((side, run[0], run[1]))
    if len(gaps) != 1:
        return None

    gap_side, gap_start, gap_end = gaps[0]
    for side, side_mask in sides.items():
        repaired = side_mask.copy()
        if side == gap_side:
            repaired[gap_start : gap_end + 1] = True
        if not np.all(repaired):
            return None

    info = dict(component)
    info["gap_side"] = gap_side
    info["gap_start"] = int(gap_start)
    info["gap_width"] = int(gap_end - gap_start + 1)
    return info


def _find_frame(
    components: list[dict[str, object]]
) -> dict[str, object]:
    candidates = []
    for component in components:
        candidate = _frame_candidate(component)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one gapped rectangular frame"
    return candidates[0]


def _count_holes(mask: np.ndarray) -> int:
    solid = np.asarray(mask, dtype=bool)
    height, width = solid.shape
    visited = np.zeros((height, width), dtype=bool)
    holes = 0

    for r in range(height):
        for c in range(width):
            if solid[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            touches_boundary = r in (0, height - 1) or c in (0, width - 1)
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if solid[nr, nc] or visited[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    if nr in (0, height - 1) or nc in (0, width - 1):
                        touches_boundary = True
                    q.append((nr, nc))
            if not touches_boundary:
                holes += 1

    return int(holes)


def _parallel_size(bbox: tuple[int, int, int, int], gap_side: str) -> int:
    r0, c0, r1, c1 = bbox
    if gap_side in ("top", "bottom"):
        return int(c1 - c0 + 1)
    return int(r1 - r0 + 1)


def _bbox_disjoint(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _project_copies(
    canvas: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    gap_side: str,
    gap_start: int,
    patch_mask: np.ndarray,
    patch_color: int,
    copies: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    mask = np.asarray(patch_mask, dtype=bool)
    patch_rows, patch_cols = mask.shape
    cells = np.argwhere(mask)

    if gap_side == "top":
        base_r = r0 + 1
        base_c = c0 + gap_start
        step_r, step_c = patch_rows, 0
    elif gap_side == "bottom":
        base_r = r1 - patch_rows
        base_c = c0 + gap_start
        step_r, step_c = -patch_rows, 0
    elif gap_side == "left":
        base_r = r0 + gap_start
        base_c = c0 + 1
        step_r, step_c = 0, patch_cols
    else:
        base_r = r0 + gap_start
        base_c = c1 - patch_cols
        step_r, step_c = 0, -patch_cols

    for idx in range(int(copies)):
        off_r = base_r + step_r * idx
        off_c = base_c + step_c * idx
        for cell_r, cell_c in cells:
            rr = int(off_r + cell_r)
            cc = int(off_c + cell_c)
            if r0 < rr < r1 and c0 < cc < c1:
                canvas[rr, cc] = int(patch_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the uniquely size-matched outside sprite through the frame gap and repeat it by its hole count.

    Input structure:
    - The background is the dominant color.
    - One monochrome rectangular frame encloses a background-filled interior, and exactly one side of that frame has one contiguous gap.
    - Several other monochrome 4-connected objects lie outside the frame; each is non-background and may contain enclosed background holes.

    Rule:
    - Detect which frame side has the gap to determine the projection direction: top projects downward, bottom upward, left rightward, right leftward.
    - Among the outside objects, find the unique one whose bounding-box size along the gap-parallel axis equals the gap width.
    - Count that matched object's enclosed background holes, then draw the object's shape and color from the inner edge of the gap into the frame interior.
    - Repeat the matched object hole_count + 1 times consecutively along the projection direction, clipping any cells that would extend beyond the frame interior.

    Key insight:
    - The gap side encodes direction, the gap width selects the source object, and that object's own topology supplies the repeat count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    components = _find_components(g, excluded_colors={bg})
    frame = _find_frame(components)
    frame_bbox = tuple(frame["bbox"])
    gap_side = str(frame["gap_side"])
    gap_width = int(frame["gap_width"])
    gap_start = int(frame["gap_start"])

    outside_objects = [
        component
        for component in components
        if component is not frame and _bbox_disjoint(tuple(component["bbox"]), frame_bbox)
    ]
    matches = [
        component
        for component in outside_objects
        if _parallel_size(tuple(component["bbox"]), gap_side) == gap_width
    ]
    assert len(matches) == 1, "expected a unique outside object matching the gap width"

    matched = matches[0]
    hole_count = _count_holes(np.asarray(matched["mask"], dtype=bool))

    out = g.copy()
    _project_copies(
        out,
        frame_bbox=frame_bbox,
        gap_side=gap_side,
        gap_start=gap_start,
        patch_mask=np.asarray(matched["mask"], dtype=bool),
        patch_color=int(matched["color"]),
        copies=hole_count + 1,
    )
    return out


def _positive_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = cut
    return values


def _make_zero_hole_object(
    compare_length: int, compare_axis: str, rng: random.Random
) -> np.ndarray:
    if compare_axis == "width":
        height = rng.randint(2, 4)
        width = compare_length
    else:
        height = compare_length
        width = rng.randint(2, 4)

    mask = np.ones((height, width), dtype=bool)
    variant = rng.choice(["solid", "top_notch", "bottom_notch", "side_notch"])

    if variant == "top_notch" and height >= 2 and width >= 4:
        start = rng.randint(1, width - 2)
        length = rng.randint(1, min(2, width - 1 - start))
        mask[0, start : start + length] = False
    elif variant == "bottom_notch" and height >= 2 and width >= 4:
        start = rng.randint(1, width - 2)
        length = rng.randint(1, min(2, width - 1 - start))
        mask[-1, start : start + length] = False
    elif variant == "side_notch" and height >= 4 and width >= 2:
        col = 0 if rng.random() < 0.5 else width - 1
        start = rng.randint(1, height - 2)
        length = rng.randint(1, min(2, height - 1 - start))
        mask[start : start + length, col] = False

    return mask


def _make_holey_object(
    compare_length: int, compare_axis: str, hole_count: int, rng: random.Random
) -> np.ndarray:
    thickness = rng.randint(3, 5)
    wall_widths = _positive_composition(compare_length - hole_count, hole_count + 1, rng)

    row = []
    for idx, wall_width in enumerate(wall_widths):
        row.extend([True] * wall_width)
        if idx < hole_count:
            row.append(False)
    row_mask = np.asarray(row, dtype=bool)

    mask = np.ones((thickness, compare_length), dtype=bool)
    mask[1:-1, :] = row_mask
    if compare_axis == "height":
        mask = mask.T
    return mask


def _build_object(
    compare_length: int, compare_axis: str, hole_count: int, rng: random.Random
) -> np.ndarray:
    if hole_count == 0:
        return _make_zero_hole_object(compare_length, compare_axis, rng)
    return _make_holey_object(compare_length, compare_axis, hole_count, rng)


def _compatible_hole_counts(compare_length: int) -> list[int]:
    options = [0]
    for hole_count in (1, 2, 3):
        if compare_length >= 2 * hole_count + 1:
            options.append(hole_count)
    return options


def _draw_frame(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    gap_side: str,
    gap_start: int,
    gap_width: int,
    frame_color: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = bg
    elif gap_side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_width] = bg
    elif gap_side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = bg
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_width, c1] = bg


def _paint_mask(
    grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int
) -> None:
    height, width = mask.shape
    patch = grid[top : top + height, left : left + width]
    patch[mask] = color


def _rect_too_close(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gap_side = rng.choice(GAP_SIDES)
    compare_axis = "width" if gap_side in ("top", "bottom") else "height"

    object_count = rng.randint(3, 5)
    match_holes = rng.randint(0, 3)
    min_match_length = 2 if match_holes == 0 else 2 * match_holes + 1
    match_length = rng.randint(min_match_length, COMPARE_LENGTHS[-1])

    other_lengths = list(COMPARE_LENGTHS)
    other_lengths.remove(match_length)
    rng.shuffle(other_lengths)
    chosen_lengths = [match_length] + other_lengths[: object_count - 1]

    colors = rng.sample(OBJECT_COLORS, object_count)
    objects = []
    for idx, compare_length in enumerate(chosen_lengths):
        if idx == 0:
            hole_count = match_holes
        else:
            hole_count = rng.choice(_compatible_hole_counts(compare_length))
        mask = _build_object(compare_length, compare_axis, hole_count, rng)
        objects.append(
            {
                "mask": mask,
                "color": colors[idx],
                "hole_count": hole_count,
                "compare_length": compare_length,
                "is_match": idx == 0,
            }
        )

    matched = next(obj for obj in objects if obj["is_match"])
    match_mask = np.asarray(matched["mask"], dtype=bool)
    match_rows, match_cols = match_mask.shape
    projection_step = match_rows if gap_side in ("top", "bottom") else match_cols
    total_projection = projection_step * (match_holes + 1)

    if gap_side in ("top", "bottom"):
        inner_width_min = match_length + 2
        inner_width_max = min(match_length + 10, 18)
        inner_width = rng.randint(inner_width_min, inner_width_max)
        clip_ok = match_holes > 0 and total_projection > projection_step
        if clip_ok and rng.random() < 0.5:
            inner_height = rng.randint(projection_step, total_projection - 1)
        else:
            inner_height = rng.randint(total_projection, min(total_projection + 5, 20))
    else:
        inner_height_min = match_length + 2
        inner_height_max = min(match_length + 10, 18)
        inner_height = rng.randint(inner_height_min, inner_height_max)
        clip_ok = match_holes > 0 and total_projection > projection_step
        if clip_ok and rng.random() < 0.5:
            inner_width = rng.randint(projection_step, total_projection - 1)
        else:
            inner_width = rng.randint(total_projection, min(total_projection + 5, 20))

    frame_height = inner_height + 2
    frame_width = inner_width + 2
    if frame_height > 22 or frame_width > 22:
        return None

    if gap_side in ("top", "bottom"):
        max_gap_start = frame_width - match_length - 1
        if max_gap_start <= 1:
            return None
        gap_start = rng.randint(1, max_gap_start)
    else:
        max_gap_start = frame_height - match_length - 1
        if max_gap_start <= 1:
            return None
        gap_start = rng.randint(1, max_gap_start)

    max_obj_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_obj_w = max(int(obj["mask"].shape[1]) for obj in objects)
    grid_height_min = max(20, frame_height + max_obj_h + 4)
    grid_width_min = max(20, frame_width + max_obj_w + 4)
    if grid_height_min > 30 or grid_width_min > 30:
        return None

    grid_height = rng.randint(grid_height_min, 30)
    grid_width = rng.randint(grid_width_min, 30)
    if frame_height + 4 > grid_height or frame_width + 4 > grid_width:
        return None

    frame_top = rng.randint(2, grid_height - frame_height - 2)
    frame_left = rng.randint(2, grid_width - frame_width - 2)
    frame_bbox = (
        frame_top,
        frame_left,
        frame_top + frame_height - 1,
        frame_left + frame_width - 1,
    )

    placements = []
    reserved = [frame_bbox]
    order = list(range(object_count))
    rng.shuffle(order)

    for idx in order:
        mask = np.asarray(objects[idx]["mask"], dtype=bool)
        obj_height, obj_width = mask.shape
        placed = None
        for _ in range(250):
            top = rng.randint(1, grid_height - obj_height - 1)
            left = rng.randint(1, grid_width - obj_width - 1)
            rect = (top, left, top + obj_height - 1, left + obj_width - 1)
            if any(_rect_too_close(rect, other_rect, gap=1) for other_rect in reserved):
                continue
            placed = (top, left, rect)
            break
        if placed is None:
            return None
        top, left, rect = placed
        placements.append((idx, top, left))
        reserved.append(rect)

    grid = np.full((grid_height, grid_width), BG_COLOR, dtype=int)
    _draw_frame(
        grid,
        frame_bbox=frame_bbox,
        gap_side=gap_side,
        gap_start=gap_start,
        gap_width=match_length,
        frame_color=FRAME_COLOR,
        bg=BG_COLOR,
    )

    for idx, top, left in placements:
        _paint_mask(
            grid,
            top=top,
            left=left,
            mask=np.asarray(objects[idx]["mask"], dtype=bool),
            color=int(objects[idx]["color"]),
        )

    expected = grid.copy()
    _project_copies(
        expected,
        frame_bbox=frame_bbox,
        gap_side=gap_side,
        gap_start=gap_start,
        patch_mask=match_mask,
        patch_color=int(matched["color"]),
        copies=match_holes + 1,
    )

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[BG_COLOR] <= max(count for color, count in counts.items() if color != BG_COLOR):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(random.Random())
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input grid")

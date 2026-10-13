from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg


DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    mask = _frame_mask(grid.shape, bbox)
    grid[mask] = int(color)
    return mask


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    specs: list[dict] = []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        comps = components_mask(g == color, connectivity=4)
        assert len(comps) == 2, "each color must form one frame and one singleton marker"
        sizes = sorted((len(comp), idx) for idx, comp in enumerate(comps))
        assert sizes[0][0] == 1 and sizes[1][0] > 1
        marker = tuple(map(int, comps[sizes[0][1]][0]))
        frame_coords = comps[sizes[1][1]]
        r0, c0, r1, c1 = bbox_coords(frame_coords)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frames must have a hollow interior"

        frame_crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        frame_crop[frame_coords[:, 0] - r0, frame_coords[:, 1] - c0] = True
        expected = _frame_mask(frame_crop.shape, (0, 0, frame_crop.shape[0] - 1, frame_crop.shape[1] - 1))
        assert np.array_equal(frame_crop, expected), "frame component must be a one-cell-thick rectangle"
        assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must be background"

        mr, mc = marker
        options: list[tuple[str, int]] = []
        if mr < r0 and c0 <= mc <= c1:
            options.append(("up", r0 - mr))
        if mr > r1 and c0 <= mc <= c1:
            options.append(("down", mr - r1))
        if mc < c0 and r0 <= mr <= r1:
            options.append(("left", c0 - mc))
        if mc > c1 and r0 <= mr <= r1:
            options.append(("right", mc - c1))
        assert len(options) == 1, "marker must align to exactly one frame side"
        direction, distance = options[0]
        assert distance >= 2, "same-color marker must be detached from the frame"

        specs.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "frame_mask": _frame_mask(g.shape, (int(r0), int(c0), int(r1), int(c1))),
                "marker": (int(mr), int(mc)),
                "direction": direction,
                "distance": int(distance),
            }
        )

    distances = [spec["distance"] for spec in specs]
    assert len(distances) == len(set(distances)), "marker distances must be unique"
    return bg, specs


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    direction: str,
    marker: tuple[int, int],
    obstacle_mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], bool]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker

    if direction == "up":
        target = mr
        blocked = obstacle_mask[target:r0, c0 : c1 + 1]
        if blocked.any():
            stop_row = target + int(np.argwhere(blocked)[:, 0].max()) + 1
        else:
            stop_row = target
        return (int(stop_row), c0, r1, c1), stop_row == target

    if direction == "down":
        target = mr
        blocked = obstacle_mask[r1 + 1 : target + 1, c0 : c1 + 1]
        if blocked.any():
            stop_row = r1 + int(np.argwhere(blocked)[:, 0].min())
            return (r0, c0, int(stop_row - 1), c1), False
        return (r0, c0, int(target), c1), True

    if direction == "left":
        target = mc
        blocked = obstacle_mask[r0 : r1 + 1, target:c0]
        if blocked.any():
            stop_col = target + int(np.argwhere(blocked)[:, 1].max()) + 1
        else:
            stop_col = target
        return (r0, int(stop_col), r1, c1), stop_col == target

    assert direction == "right"
    target = mc
    blocked = obstacle_mask[r0 : r1 + 1, c1 + 1 : target + 1]
    if blocked.any():
        stop_col = c1 + int(np.argwhere(blocked)[:, 1].min())
        return (r0, c0, r1, int(stop_col - 1)), False
    return (r0, c0, r1, int(target)), True


def _simulate(grid: np.ndarray, *, dynamic: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, specs = _parse_objects(g)
    ordered = sorted(specs, key=lambda spec: spec["distance"])

    canvas = g.copy()
    marker_mask = np.zeros(g.shape, dtype=bool)
    for spec in specs:
        marker_mask[spec["marker"]] = True

    fixed_obstacles = np.zeros(g.shape, dtype=bool)
    for spec in specs:
        fixed_obstacles |= spec["frame_mask"]

    for spec in ordered:
        color = int(spec["color"])
        own_pixels = canvas == color
        marker_mask[own_pixels] = False
        canvas[own_pixels] = bg

        if dynamic:
            obstacle_mask = (canvas != bg) & (~marker_mask)
        else:
            obstacle_mask = fixed_obstacles & (~spec["frame_mask"])

        new_bbox, reached_marker = _expand_bbox(spec["bbox"], spec["direction"], spec["marker"], obstacle_mask)
        painted = _draw_frame(canvas, new_bbox, color)
        marker_mask[painted] = False

        if not reached_marker:
            mr, mc = spec["marker"]
            if canvas[mr, mc] == bg:
                canvas[mr, mc] = color
                marker_mask[mr, mc] = True

    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch each hollow rectangle toward its same-color marker, resolving shorter stretches first and stopping at blockers.

    Input structure:
    - The grid has one dominant background color.
    - Each non-background color appears as a detached hollow rectangular frame plus one same-color singleton marker.
    - Every marker lies in the row band or column band facing exactly one side of its frame, and all frame-to-marker distances are distinct.

    Rule:
    - Sort frames by their marker distance from shortest to longest.
    - For each frame in that order, remove its current drawing and stretch it toward the marker so the moving side would land on the marker row or column.
    - If another frame body, either still original or already stretched, lies in the swept strip, stop immediately next to the nearest blocker instead; only a frame that reaches its target absorbs its marker, while a blocked frame keeps the marker visible.

    Key insight:
    - The marker alone does not determine the final size: earlier short stretches change the obstacle field, so later frames must be resolved against the updated canvas rather than against the original input.
    """
    return _simulate(grid, dynamic=True)


def _apply_dihedral(grid: np.ndarray, code: int) -> np.ndarray:
    if code < 4:
        return np.rot90(grid, k=code).copy()
    return np.fliplr(np.rot90(grid, k=code - 4)).copy()


def _count_singletons(grid: np.ndarray, color: int) -> int:
    return sum(len(comp) == 1 for comp in components_mask(grid == color, connectivity=4))


def _gen_place_canonical() -> tuple[np.ndarray, dict]:
    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], 3)
    color_a, color_b, color_c = colors

    top = random.randint(2, 3)
    left = random.randint(1, 3)

    h_a = random.randint(4, 5)
    w_a = random.randint(4, 5)
    d_a = random.randint(2, 4)
    a_bbox = (top, left, top + h_a - 1, left + w_a - 1)
    a_marker = (random.randint(top + 1, top + h_a - 2), a_bbox[3] + d_a)
    a_final_right = a_marker[1]

    h_b = h_a + 2
    w_b = random.randint(4, 5)
    gap_ab = random.randint(3, 5)
    b_bbox = (top - 1, a_final_right + gap_ab, top - 1 + h_b - 1, a_final_right + gap_ab + w_b - 1)
    marker_b_col = random.randint(a_bbox[3] + 1, a_final_right - 1)
    marker_b_row = random.choice([b_bbox[0], b_bbox[2]])
    b_marker = (marker_b_row, marker_b_col)
    d_b = b_bbox[1] - marker_b_col
    if d_b == d_a:
        raise ValueError("pair distances must differ")

    gap_v = random.randint(3, 5)
    h_c = random.randint(4, 6)
    w_c = random.randint(4, 6)
    candidate_distances = [d for d in range(2, 7) if d not in {d_a, d_b}]
    d_c = random.choice(candidate_distances)
    c_top = b_bbox[2] + gap_v
    c_left_hi = min(max(b_bbox[3] - 2, left), 10)
    c_left = random.randint(left, c_left_hi)
    c_bbox = (c_top, c_left, c_top + h_c - 1, c_left + w_c - 1)
    c_marker = (c_bbox[2] + d_c, random.randint(c_left + 1, c_bbox[3] - 1))

    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(2, 5)
    height = c_marker[0] + bottom_margin + 1
    width = max(b_bbox[3], c_bbox[3]) + right_margin + 1
    if height > 30 or width > 30:
        raise ValueError("canonical layout exceeds ARC bounds")

    grid = np.full((height, width), bg, dtype=int)
    for bbox, color in ((a_bbox, color_a), (b_bbox, color_b), (c_bbox, color_c)):
        _draw_frame(grid, bbox, color)
    for (r, c), color in ((a_marker, color_a), (b_marker, color_b), (c_marker, color_c)):
        grid[r, c] = color

    meta = {
        "bg": bg,
        "dynamic_blocked_color": color_b,
        "success_colors": {color_a, color_c},
    }
    return grid, meta


def _gen_valid(grid: np.ndarray, meta: dict) -> bool:
    try:
        _bg, specs = _parse_objects(grid)
    except AssertionError:
        return False
    if len(specs) != 3:
        return False

    try:
        dynamic_out = main(grid)
        static_out = _simulate(grid, dynamic=False)
    except AssertionError:
        return False

    if np.array_equal(dynamic_out, static_out):
        return False

    blocked_color = meta["dynamic_blocked_color"]
    if _count_singletons(dynamic_out, blocked_color) != 1:
        return False
    if _count_singletons(static_out, blocked_color) != 0:
        return False

    for color in meta["success_colors"]:
        if _count_singletons(dynamic_out, color) != 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, meta = _gen_place_canonical()
    except ValueError:
        return None

    transformed = _apply_dihedral(grid, random.randrange(8))
    if not _gen_valid(transformed, meta):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ordered-stretch input")

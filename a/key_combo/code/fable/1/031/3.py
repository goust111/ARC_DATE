from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, crop_bbox, find_components, label_mask, paste


SIDES = ("top", "right", "bottom", "left")
FRAME_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_MIN = 24
GRID_MAX = 30
GEN_ATTEMPTS = 300
HOLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
OBJECT_SIZE_RANGE = (5, 8)
SPLIT_MARGIN = 1

PATTERNS = (
    {"top_col": "right", "bottom_col": "left", "left_row": "upper", "right_row": "lower"},
    {"top_col": "left", "bottom_col": "right", "left_row": "lower", "right_row": "upper"},
)


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    bg_labels, n_labels = label_mask(~mask, connectivity=4)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    edge_ids = set(map(int, np.unique(np.concatenate((bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1])))))
    holes = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, n_labels + 1):
        if label_id not in edge_ids:
            holes |= bg_labels == label_id
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return int(label_mask(_hole_mask(mask), connectivity=4)[1])


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    return mask | _hole_mask(mask)


def _frame_side_colors(grid: np.ndarray) -> dict[str, int]:
    top = {int(v) for v in grid[0, 1:-1]}
    right = {int(v) for v in grid[1:-1, -1]}
    bottom = {int(v) for v in grid[-1, 1:-1]}
    left = {int(v) for v in grid[1:-1, 0]}
    assert len(top) == len(right) == len(bottom) == len(left) == 1
    side_colors = {
        "top": next(iter(top)),
        "right": next(iter(right)),
        "bottom": next(iter(bottom)),
        "left": next(iter(left)),
    }
    assert len(set(side_colors.values())) == 4
    return side_colors


def _target_origin(side: str, bbox: tuple[int, int, int, int], hole_count: int, inner_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inner_h, inner_w = inner_shape
    if side == "top":
        return hole_count, c0
    if side == "bottom":
        return inner_h - h - hole_count, c0
    if side == "left":
        return r0, hole_count
    if side == "right":
        return r0, inner_w - w - hole_count
    raise ValueError(f"unknown side: {side}")


def _component_spec(inner: np.ndarray, component: dict, side_colors: dict[str, int]) -> tuple[np.ndarray, int, str, tuple[int, int, int, int]]:
    comp_mask = np.zeros(inner.shape, dtype=bool)
    rr, cc = component["coords"].T
    comp_mask[rr, cc] = True
    bbox = component["bbox"]
    patch = crop_bbox(inner, bbox)
    mask = crop_bbox(comp_mask, bbox)

    marker_pixels = np.argwhere((patch != Color.BLACK) & (patch != Color.GREY))
    assert marker_pixels.shape[0] == 1
    marker_colors = colors_present(patch, exclude={Color.BLACK, Color.GREY})
    assert len(marker_colors) == 1
    marker_color = int(marker_colors[0])

    side_matches = [side for side, color in side_colors.items() if color == marker_color]
    assert len(side_matches) == 1
    side = side_matches[0]

    solid_mask = _fill_holes(mask)
    hole_count = _hole_count(mask)
    assert 1 <= hole_count <= 4
    return solid_mask, hole_count, side, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray holed object toward the frame side named by its marker color, using the hole count as the stopping gap.

    Input structure:
    - The outermost border is a one-cell-thick rectangular frame whose four sides use four distinct colors.
    - Inside the frame, the background is black and there are several disconnected gray objects.
    - Each object contains one colored marker pixel whose color matches exactly one frame side, and one to four enclosed black holes.

    Rule:
    - For each object, count its enclosed black holes and read the marker color to choose the destination frame side.
    - Translate the whole object only along the axis perpendicular to that side, keeping the orthogonal coordinate fixed, until the object's facing edge is separated from the frame's inner wall by exactly that many black cells.
    - After moving, recolor the whole object to the destination side color, fill its holes, erase the marker, and leave the frame unchanged.

    Key insight:
    - The marker color determines direction while the hole count determines distance; both cues are consumed in the output, which keeps only the moved solid object in the frame-side color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5

    side_colors = _frame_side_colors(g)
    inner = g[1:-1, 1:-1]
    components = find_components(inner, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="dict")
    assert components

    out = g.copy()
    out[1:-1, 1:-1] = Color.BLACK
    out_inner = out[1:-1, 1:-1]

    for component in components:
        solid_mask, hole_count, side, bbox = _component_spec(inner, component, side_colors)
        target_r0, target_c0 = _target_origin(side, bbox, hole_count, inner.shape)
        patch = np.zeros(solid_mask.shape, dtype=int)
        patch[solid_mask] = int(side_colors[side])

        region = out_inner[target_r0 : target_r0 + patch.shape[0], target_c0 : target_c0 + patch.shape[1]]
        assert region.shape == patch.shape
        assert not np.any((region != Color.BLACK) & solid_mask)
        paste(out_inner, patch, target_r0, target_c0, transparent=Color.BLACK)

    return out


def _gen_hole_options(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    options: list[tuple[tuple[int, int], ...]] = []
    for hh, ww in HOLE_SHAPES:
        for r0 in range(1, height - hh):
            for c0 in range(1, width - ww):
                cells = tuple((r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww))
                options.append(cells)
    return options


def _holes_separated(cells: tuple[tuple[int, int], ...], used: tuple[tuple[int, int], ...]) -> bool:
    for r0, c0 in cells:
        for r1, c1 in used:
            if abs(r0 - r1) + abs(c0 - c1) < 2:
                return False
    return True


def _search_holes(
    options: list[tuple[tuple[int, int], ...]],
    remaining: int,
    used: tuple[tuple[int, int], ...],
    *,
    shuffled: bool,
) -> list[tuple[tuple[int, int], ...]] | None:
    if remaining == 0:
        return []

    pool = options[:]
    if shuffled:
        random.shuffle(pool)
    for cells in pool:
        if not _holes_separated(cells, used):
            continue
        tail = _search_holes(options, remaining - 1, used + cells, shuffled=shuffled)
        if tail is not None:
            return [cells] + tail
    return None


def _feasible_dims(hole_count: int) -> list[tuple[int, int]]:
    cache = getattr(_feasible_dims, "_cache", {})
    if hole_count not in cache:
        dims: list[tuple[int, int]] = []
        for height in range(OBJECT_SIZE_RANGE[0], OBJECT_SIZE_RANGE[1] + 1):
            for width in range(OBJECT_SIZE_RANGE[0], OBJECT_SIZE_RANGE[1] + 1):
                options = _gen_hole_options(height, width)
                if _search_holes(options, hole_count, (), shuffled=False) is not None:
                    dims.append((height, width))
        cache[hole_count] = dims
        _feasible_dims._cache = cache
    return cache[hole_count]


def _make_object(marker_color: int, hole_count: int) -> tuple[np.ndarray, np.ndarray]:
    dims = _feasible_dims(hole_count)[:]
    random.shuffle(dims)
    for height, width in dims:
        options = _gen_hole_options(height, width)
        holes = _search_holes(options, hole_count, (), shuffled=True)
        if holes is None:
            continue

        patch = np.full((height, width), Color.GREY, dtype=int)
        for cells in holes:
            for r, c in cells:
                patch[r, c] = Color.BLACK

        solid_mask = _fill_holes(patch != Color.BLACK)
        if _hole_count(patch != Color.BLACK) != hole_count:
            continue

        marker_candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if patch[r, c] == Color.GREY]
        if not marker_candidates:
            marker_candidates = [(int(r), int(c)) for r, c in np.argwhere(patch == Color.GREY)]
        marker_r, marker_c = random.choice(marker_candidates)
        patch[marker_r, marker_c] = int(marker_color)
        return patch, solid_mask
    raise RuntimeError("failed to build object")


def _lane_start(total_len: int, obj_len: int, lane: str) -> int | None:
    mid = total_len // 2
    if lane in {"left", "upper"}:
        lo, hi = 1, mid - SPLIT_MARGIN - obj_len
    elif lane in {"right", "lower"}:
        lo, hi = mid + SPLIT_MARGIN, total_len - obj_len - 1
    else:
        raise ValueError(f"unknown lane: {lane}")
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _initial_start(total_len: int, obj_len: int, final_start: int, side: str) -> int | None:
    mid = total_len // 2
    if side in {"top", "left"}:
        lo, hi = max(final_start + 2, mid + SPLIT_MARGIN), total_len - obj_len - 1
    elif side in {"bottom", "right"}:
        lo, hi = 1, min(final_start - 2, mid - SPLIT_MARGIN - obj_len)
    else:
        raise ValueError(f"unknown side: {side}")
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _render_frame(height: int, width: int, side_colors: dict[str, int]) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[0, :] = int(side_colors["top"])
    grid[-1, :] = int(side_colors["bottom"])
    grid[1:-1, 0] = int(side_colors["left"])
    grid[1:-1, -1] = int(side_colors["right"])
    return grid


def _safe_paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    mask = patch != Color.BLACK
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    if np.any((region != Color.BLACK) & mask):
        return False
    paste(canvas, patch, r0, c0, transparent=Color.BLACK)
    return True


def _expected_output(
    shape: tuple[int, int],
    side_colors: dict[str, int],
    placements: dict[str, tuple[int, int]],
    specs: dict[str, dict],
) -> np.ndarray:
    out = _render_frame(*shape, side_colors)
    inner = out[1:-1, 1:-1]
    for side, spec in specs.items():
        r0, c0 = placements[side]
        patch = np.zeros(spec["solid_mask"].shape, dtype=int)
        patch[spec["solid_mask"]] = int(side_colors[side])
        if not _safe_paste(inner, patch, r0, c0):
            raise ValueError("expected output overlap")
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    inner_h, inner_w = height - 2, width - 2

    chosen_colors = random.sample(FRAME_COLORS, 4)
    side_colors = dict(zip(SIDES, chosen_colors))
    hole_assignment = dict(zip(SIDES, random.sample([1, 2, 3, 4], 4)))
    pattern = random.choice(PATTERNS)

    specs: dict[str, dict] = {}
    for side in SIDES:
        patch, solid_mask = _make_object(side_colors[side], hole_assignment[side])
        specs[side] = {
            "patch": patch,
            "solid_mask": solid_mask,
            "holes": hole_assignment[side],
            "h": patch.shape[0],
            "w": patch.shape[1],
        }

    top_col_lane = pattern["top_col"]
    bottom_col_lane = pattern["bottom_col"]
    left_row_lane = pattern["left_row"]
    right_row_lane = pattern["right_row"]

    final_positions: dict[str, tuple[int, int]] = {}
    final_top_c = _lane_start(inner_w, specs["top"]["w"], top_col_lane)
    final_bottom_c = _lane_start(inner_w, specs["bottom"]["w"], bottom_col_lane)
    final_left_r = _lane_start(inner_h, specs["left"]["h"], left_row_lane)
    final_right_r = _lane_start(inner_h, specs["right"]["h"], right_row_lane)
    if None in {final_top_c, final_bottom_c, final_left_r, final_right_r}:
        return None

    final_positions["top"] = (specs["top"]["holes"], int(final_top_c))
    final_positions["bottom"] = (inner_h - specs["bottom"]["h"] - specs["bottom"]["holes"], int(final_bottom_c))
    final_positions["left"] = (int(final_left_r), specs["left"]["holes"])
    final_positions["right"] = (int(final_right_r), inner_w - specs["right"]["w"] - specs["right"]["holes"])

    for side, (r0, c0) in final_positions.items():
        if r0 < 0 or c0 < 0 or r0 + specs[side]["h"] > inner_h or c0 + specs[side]["w"] > inner_w:
            return None

    initial_positions: dict[str, tuple[int, int]] = {}
    top_r = _initial_start(inner_h, specs["top"]["h"], final_positions["top"][0], "top")
    bottom_r = _initial_start(inner_h, specs["bottom"]["h"], final_positions["bottom"][0], "bottom")
    left_c = _initial_start(inner_w, specs["left"]["w"], final_positions["left"][1], "left")
    right_c = _initial_start(inner_w, specs["right"]["w"], final_positions["right"][1], "right")
    if None in {top_r, bottom_r, left_c, right_c}:
        return None

    initial_positions["top"] = (int(top_r), final_positions["top"][1])
    initial_positions["bottom"] = (int(bottom_r), final_positions["bottom"][1])
    initial_positions["left"] = (final_positions["left"][0], int(left_c))
    initial_positions["right"] = (final_positions["right"][0], int(right_c))

    inp = _render_frame(height, width, side_colors)
    inner = inp[1:-1, 1:-1]
    for side in SIDES:
        r0, c0 = initial_positions[side]
        if not _safe_paste(inner, specs[side]["patch"], r0, c0):
            return None

    try:
        expected = _expected_output((height, width), side_colors, final_positions, specs)
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

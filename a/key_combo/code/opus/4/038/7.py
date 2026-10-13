from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4, scale_sprite


GRID_SHAPE = (20, 20)
BACKGROUND = Color.BLACK
FRAME_COLOR = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE)
CORNER_ORDER = ("tl", "tr", "bl", "br")

CORNER_SPECS = {
    "tl": {"marker": lambda h, w: (1, 1), "row_dir": 1, "col_dir": 1},
    "tr": {"marker": lambda h, w: (1, w - 2), "row_dir": 1, "col_dir": -1},
    "bl": {"marker": lambda h, w: (h - 2, 1), "row_dir": -1, "col_dir": 1},
    "br": {"marker": lambda h, w: (h - 2, w - 2), "row_dir": -1, "col_dir": -1},
}

TEMPLATE_MASKS = {
    "tl": np.array([[0, 1], [1, 1]], dtype=int),
    "tr": np.array([[1, 0], [1, 1]], dtype=int),
    "bl": np.array([[1, 1], [0, 1]], dtype=int),
    "br": np.array([[1, 1], [1, 0]], dtype=int),
}

_ALL_CORNER_SUBSETS = {
    size: [tuple(combo) for combo in combinations(CORNER_ORDER, size)]
    for size in (1, 2, 3)
}
_GEN_CALL_COUNT = 0
_GEN_RETRIES = 400


def _count_enclosed_black_regions(grid: np.ndarray) -> int:
    mask = np.asarray(grid == BACKGROUND, dtype=bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    enclosed = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            touches_boundary = r in (0, h - 1) or c in (0, w - 1)

            while stack:
                cr, cc = stack.pop()
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds((h, w), nr, nc):
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_boundary = True
                    stack.append((nr, nc))

            if not touches_boundary:
                enclosed += 1

    return enclosed


def _extract_corner_markers(grid: np.ndarray) -> list[tuple[str, tuple[int, int], int]]:
    h, w = grid.shape
    markers = []
    for corner in CORNER_ORDER:
        r, c = CORNER_SPECS[corner]["marker"](h, w)
        color = int(grid[r, c])
        if color in MARKER_COLORS:
            markers.append((corner, (r, c), color))
        else:
            assert color == BACKGROUND, "unexpected non-marker corner value"
    assert 1 <= len(markers) <= 3, "expected one to three corner markers"
    return markers


def _paint_scaled_template(
    out: np.ndarray,
    *,
    corner: str,
    anchor: tuple[int, int],
    color: int,
    factor: int,
) -> None:
    scaled = scale_sprite(TEMPLATE_MASKS[corner], factor).astype(bool)
    h, w = scaled.shape
    spec = CORNER_SPECS[corner]
    start_r = anchor[0] if spec["row_dir"] > 0 else anchor[0] - h + 1
    start_c = anchor[1] if spec["col_dir"] > 0 else anchor[1] - w + 1

    for dr in range(h):
        for dc in range(w):
            if not scaled[dr, dc]:
                continue
            rr = start_r + dr
            cc = start_c + dc
            if in_bounds(out.shape, rr, cc):
                out[rr, cc] = color


def _apply_rule(grid: np.ndarray, factor: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    region_count = _count_enclosed_black_regions(g) if factor is None else int(factor)
    assert 1 <= region_count <= 3, "enclosed-region count must be 1..3"

    for corner, anchor, color in _extract_corner_markers(g):
        out[anchor] = BACKGROUND
        _paint_scaled_template(out, corner=corner, anchor=anchor, color=color, factor=region_count)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each corner marker with a corner-oriented L stamp whose scale is set by the central frame's enclosed-region count.

    Input structure:
    - A black background contains a grey line-drawn frame near the upper middle of the grid.
    - The frame partitions its interior into one to three enclosed black rectangular regions.
    - One to three colored singleton markers sit at fixed one-cell-inset corner positions.

    Rule:
    - Count the 4-connected black regions that are enclosed by the grey frame and do not touch the grid boundary.
    - Use that count N as the integer scale factor for a fixed 2x2 L template with the outward corner missing.
    - For each corner marker, erase the marker and stamp the N-scaled template in that marker's color, choosing the template orientation from the marker's corner so the opening faces outward.
    - Preserve the grey frame and every untouched background cell.

    Key insight:
    - The middle frame encodes a number topologically, while each corner marker contributes only position and color; solving requires combining those independent signals into one stamp operation.
    """
    return _apply_rule(grid)


def _draw_partition_frame(
    grid: np.ndarray,
    *,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    orientation: str,
    segments: list[int],
) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR

    if orientation == "vertical":
        divider_c = left + 1
        for seg in segments[:-1]:
            divider_c += seg
            grid[top : bottom + 1, divider_c] = FRAME_COLOR
            divider_c += 1
    elif orientation == "horizontal":
        divider_r = top + 1
        for seg in segments[:-1]:
            divider_r += seg
            grid[divider_r, left : right + 1] = FRAME_COLOR
            divider_r += 1
    else:
        assert len(segments) == 1, "single-region frame should not have internal dividers"


def _choose_frame_params(region_count: int, rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        if region_count == 1:
            orientation = "none"
            inner_h = rng.randint(4, 8)
            inner_w = rng.randint(4, 9)
            segments = [inner_w]
            outer_h = inner_h + 2
            outer_w = inner_w + 2
        else:
            orientation = rng.choice(("vertical", "horizontal"))
            segment_low = 2
            segment_high = 4 if region_count == 2 else 3
            segments = [rng.randint(segment_low, segment_high) for _ in range(region_count)]
            if orientation == "vertical":
                inner_h = rng.randint(4, 8)
                outer_h = inner_h + 2
                outer_w = sum(segments) + (region_count - 1) + 2
            else:
                inner_w = rng.randint(4, 8)
                outer_w = inner_w + 2
                outer_h = sum(segments) + (region_count - 1) + 2

        top_max = min(7, GRID_SHAPE[0] - outer_h - 4)
        left_min = 4
        left_max = min(9, GRID_SHAPE[1] - outer_w - 4)
        if top_max < 4 or left_max < left_min:
            continue

        return {
            "orientation": orientation,
            "segments": segments,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "top": rng.randint(4, top_max),
            "left": rng.randint(left_min, left_max),
        }

    raise RuntimeError("failed to choose valid frame parameters")


def _place_markers(grid: np.ndarray, chosen_corners: tuple[str, ...], colors: list[int]) -> None:
    h, w = grid.shape
    for corner, color in zip(chosen_corners, colors):
        r, c = CORNER_SPECS[corner]["marker"](h, w)
        assert grid[r, c] == BACKGROUND
        grid[r, c] = int(color)


def _expected_from_params(
    grid: np.ndarray,
    *,
    region_count: int,
    chosen_corners: tuple[str, ...],
    colors: list[int],
) -> np.ndarray:
    expected = grid.copy()
    h, w = expected.shape
    for corner, color in zip(chosen_corners, colors):
        anchor = CORNER_SPECS[corner]["marker"](h, w)
        expected[anchor] = BACKGROUND
        _paint_scaled_template(expected, corner=corner, anchor=anchor, color=int(color), factor=region_count)
    return expected


def _semantic_checks(
    grid: np.ndarray,
    *,
    region_count: int,
    chosen_corners: tuple[str, ...],
    colors: list[int],
) -> bool:
    if grid.shape != GRID_SHAPE:
        return False
    if _count_enclosed_black_regions(grid) != region_count:
        return False

    markers = _extract_corner_markers(grid)
    if tuple(corner for corner, _, _ in markers) != chosen_corners:
        return False
    if [color for _, _, color in markers] != colors:
        return False

    solved = main(grid)
    expected = _expected_from_params(
        grid,
        region_count=region_count,
        chosen_corners=chosen_corners,
        colors=colors,
    )
    if not np.array_equal(solved, expected):
        return False

    for corner, anchor, color in markers:
        patch_mask = np.zeros_like(grid, dtype=bool)
        patch_before = grid.copy()
        patch_before[anchor] = BACKGROUND
        patch_mask |= solved != patch_before
        patch_mask[anchor] = False
        if np.count_nonzero(solved == color) < 3 * region_count * region_count:
            return False
        if solved[anchor] != BACKGROUND:
            return False
        if corner not in chosen_corners:
            return False

    return True


def _generate_single(rng: random.Random, call_index: int) -> np.ndarray | None:
    region_count = 1 + (call_index % 3)
    marker_count = 1 + ((call_index // 3) % 3)
    corner_options = _ALL_CORNER_SUBSETS[marker_count]
    chosen_corners = tuple(rng.choice(corner_options))
    colors = [int(c) for c in rng.sample(MARKER_COLORS, marker_count)]

    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    frame_params = _choose_frame_params(region_count, rng)
    _draw_partition_frame(grid, **frame_params)
    _place_markers(grid, chosen_corners, colors)

    if not _semantic_checks(
        grid,
        region_count=region_count,
        chosen_corners=chosen_corners,
        colors=colors,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(_GEN_RETRIES):
        rng = random.Random()
        grid = _generate_single(rng, _GEN_CALL_COUNT)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid

    raise RuntimeError("failed to generate a valid corner-stamp input")

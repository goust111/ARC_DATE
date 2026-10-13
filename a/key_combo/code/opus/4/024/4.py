from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, in_bounds


_ORDER = ("TL", "TR", "BR", "BL")
_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # left, down, right, up


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == Color.BLUE)
    assert len(coords) > 0, "expected a blue frame"
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 5 and w >= 5, "frame edges must be at least 5 pixels long"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(sub == Color.BLUE, border), "blue pixels must form one hollow rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _inner_corners(frame_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    return {
        "TL": (r0 + 1, c0 + 1),
        "TR": (r0 + 1, c1 - 1),
        "BR": (r1 - 1, c1 - 1),
        "BL": (r1 - 1, c0 + 1),
    }


def _outer_corners(frame_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    return {
        "TL": (r0 - 1, c0 - 1),
        "TR": (r0 - 1, c1 + 1),
        "BR": (r1 + 1, c1 + 1),
        "BL": (r1 + 1, c0 - 1),
    }


def _validate_input(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected palette cells at (0,0) and (0,1)"

    frame_bbox = _find_frame_bbox(g)
    inner = _inner_corners(frame_bbox)

    assert int(g[0, 0]) != Color.BLACK and int(g[0, 1]) != Color.BLACK
    horiz_color = int(g[0, 0])
    vert_color = int(g[0, 1])
    assert horiz_color != vert_color, "palette colors must differ"
    assert horiz_color not in (Color.BLUE, Color.RED)
    assert vert_color not in (Color.BLUE, Color.RED)

    red_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == Color.RED)}
    assert red_coords == set(inner.values()), "red markers must occupy exactly the four inner corners"
    return frame_bbox, horiz_color, vert_color


def _run_transform(
    grid: np.ndarray,
    *,
    order: tuple[str, ...] = _ORDER,
    use_outer_starts: bool = True,
    block_frame: bool = True,
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    frame_bbox, horiz_color, vert_color = _validate_input(g)
    inner = _inner_corners(frame_bbox)
    outer = _outer_corners(frame_bbox)
    starts = outer if use_outer_starts else inner

    out = g.copy()
    out[0, 0] = Color.BLACK
    out[0, 1] = Color.BLACK
    for coord in inner.values():
        out[coord] = Color.BLACK
    for coord in outer.values():
        out[coord] = Color.RED

    frame_color = Color.BLUE
    spiral_colors = {horiz_color, vert_color}
    traces: list[dict[str, int | str]] = []

    for key in order:
        start = starts[key]
        current = start
        painted = 0
        stop_reason = "none"

        for seg_idx in range(g.size + 1):
            length = seg_idx + 2
            dr, dc = _DIRS[seg_idx % 4]
            color = horiz_color if dr == 0 else vert_color
            blocked = False

            for _ in range(length):
                nr = current[0] + dr
                nc = current[1] + dc
                if not in_bounds(out.shape, nr, nc):
                    stop_reason = "boundary"
                    blocked = True
                    break
                if block_frame and int(out[nr, nc]) == frame_color:
                    stop_reason = "frame"
                    blocked = True
                    break
                if int(out[nr, nc]) == Color.RED and (nr, nc) != start:
                    stop_reason = "spiral"
                    blocked = True
                    break
                if int(out[nr, nc]) in spiral_colors:
                    stop_reason = "spiral"
                    blocked = True
                    break

                out[nr, nc] = color
                current = (nr, nc)
                painted += 1

            if blocked:
                break

        traces.append({"corner": key, "painted": painted, "stop": stop_reason})

    if return_trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the frame-corner markers outward and use them to seed spirals that are clipped by the frame walls.

    Input structure:
    - A black background contains one 1-pixel-wide blue hollow rectangle.
    - The four inner corners of that rectangle each contain one red marker pixel.
    - Palette pixels at (0,0) and (0,1) encode the horizontal and vertical spiral colors.

    Rule:
    - Identify the blue rectangular frame and erase the four inner red corner markers.
    - Move those markers to the four outer diagonal corners of the frame's bbox expanded by one cell, and erase the two palette pixels to black after reading them.
    - Starting from the outer corners in TL, TR, BR, BL order, draw a square spiral with direction cycle left, down, right, up and segment lengths 2, 3, 4, 5, ...; horizontal steps use the (0,0) palette color and vertical steps use the (0,1) palette color.
    - When the next step would leave the grid, hit a blue frame pixel, or hit an already drawn spiral pixel, do not draw that step and stop that spiral permanently.

    Key insight:
    - The rectangle is both the reference for moving the red corner markers outward and the obstacle wall that truncates each spiral, so marker relocation, frame geometry, and drawing order all matter.
    """
    return _run_transform(grid)


def _alternative_output(
    grid: np.ndarray,
    *,
    order: tuple[str, ...] = _ORDER,
    use_outer_starts: bool = True,
    block_frame: bool = True,
) -> np.ndarray:
    return _run_transform(
        grid,
        order=order,
        use_outer_starts=use_outer_starts,
        block_frame=block_frame,
        return_trace=False,
    )


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        expected, traces = _run_transform(grid, return_trace=True)
    except Exception:
        return False

    if not all(int(item["painted"]) > 0 for item in traces):
        return False
    reasons = {str(item["stop"]) for item in traces}
    if "frame" not in reasons:
        return False
    if "boundary" not in reasons:
        return False

    painted_counts = {int(item["painted"]) for item in traces}
    if len(painted_counts) < 2:
        return False

    if np.array_equal(expected, _alternative_output(grid, use_outer_starts=False)):
        return False
    if np.array_equal(expected, _alternative_output(grid, block_frame=False)):
        return False
    return True


def _build_grid(
    height: int,
    width: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[0, 0] = Color.GREEN
    grid[0, 1] = Color.YELLOW

    r0 = top
    c0 = left
    r1 = top + frame_h - 1
    c1 = left + frame_w - 1
    grid[r0, c0 : c1 + 1] = Color.BLUE
    grid[r1, c0 : c1 + 1] = Color.BLUE
    grid[r0 : r1 + 1, c0] = Color.BLUE
    grid[r0 : r1 + 1, c1] = Color.BLUE

    for coord in _inner_corners((r0, c0, r1, c1)).values():
        grid[coord] = Color.RED
    return grid


def _sample_distinct_margins() -> tuple[int, int, int, int]:
    margins = random.sample(range(3, 10), 4)
    random.shuffle(margins)
    return tuple(margins)


def _gen_attempt() -> np.ndarray | None:
    top, bottom, left, right = _sample_distinct_margins()
    frame_h = random.randint(5, 10)
    frame_w = random.randint(5, 10)

    height = top + frame_h + bottom
    width = left + frame_w + right
    if max(height, width) > 30:
        return None

    if height < 15 or width < 15:
        return None

    grid = _build_grid(height, width, top, bottom, left, right, frame_h, frame_w)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate spiral-frame example")

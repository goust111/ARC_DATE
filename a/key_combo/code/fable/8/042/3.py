from __future__ import annotations

import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS))

from arc_utils import bbox, colors_present, get_bg, in_bounds


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    wall_color: int
    gap: tuple[int, int]
    marker: tuple[int, int]
    marker_color: int


@dataclass(frozen=True)
class RayResult:
    frame: Frame
    path: tuple[tuple[int, int], ...]
    blocker_color: int | None
    through_other_gap: bool
    stop_cell: tuple[int, int]


COLOR_POOL = tuple(range(1, 10))
RETRY_LIMIT = 400


def _border_cells(frame: Frame) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(frame.r0, frame.r1 + 1):
        cells.add((r, frame.c0))
        cells.add((r, frame.c1))
    for c in range(frame.c0, frame.c1 + 1):
        cells.add((frame.r0, c))
        cells.add((frame.r1, c))
    return cells


def _is_frame_color(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    border = _border_cells(Frame(r0, c0, r1, c1, 0, (0, 0), (0, 0), 0))
    return coords <= border and len(border - coords) == 1 and len(coords) == 2 * (h + w) - 5


def _detect_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames: list[Frame] = []
    for color in colors_present(g, exclude={bg}):
        mask = g == color
        if not _is_frame_color(mask):
            continue
        r0, c0, r1, c1 = bbox(mask)
        shell = _border_cells(Frame(r0, c0, r1, c1, color, (0, 0), (0, 0), 0))
        colored = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        gap_candidates = list(shell - colored)
        assert len(gap_candidates) == 1
        gap = gap_candidates[0]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_cells = np.argwhere(interior != bg)
        assert len(marker_cells) == 1, "expected exactly one interior marker per frame"
        mr, mc = map(int, marker_cells[0])
        marker = (r0 + 1 + mr, c0 + 1 + mc)
        marker_color = int(g[marker])
        assert marker[0] == gap[0] or marker[1] == gap[1], "marker must align with frame gap"
        frames.append(Frame(r0, c0, r1, c1, int(color), gap, marker, marker_color))
    frames.sort(key=lambda f: (f.r0, f.c0, f.wall_color))
    assert frames, "expected at least one valid hollow frame"
    return frames


def _direction(frame: Frame) -> tuple[int, int]:
    mr, mc = frame.marker
    gr, gc = frame.gap
    if mr == gr:
        return 0, 1 if gc > mc else -1
    return 1 if gr > mr else -1, 0


def _trace_ray(grid: np.ndarray, frame: Frame, frames: list[Frame]) -> RayResult:
    wall_by_cell = {
        cell: other.wall_color
        for other in frames
        for cell in (_border_cells(other) - {other.gap})
    }
    other_gaps = {other.gap for other in frames if other != frame}
    dr, dc = _direction(frame)
    cur = frame.marker
    path: list[tuple[int, int]] = []
    through_other_gap = False
    while True:
        path.append(cur)
        nr, nc = cur[0] + dr, cur[1] + dc
        if not in_bounds(grid.shape, nr, nc):
            return RayResult(frame, tuple(path), None, through_other_gap, cur)
        if (nr, nc) in wall_by_cell:
            return RayResult(frame, tuple(path), wall_by_cell[(nr, nc)], through_other_gap, cur)
        if (nr, nc) in other_gaps:
            through_other_gap = True
        cur = (nr, nc)


def _render_with_order(grid: np.ndarray, rays: list[RayResult], order: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for idx in order:
        ray = rays[idx]
        for r, c in ray.path:
            out[r, c] = ray.frame.marker_color
        if ray.blocker_color is not None:
            sr, sc = ray.stop_cell
            out[sr, sc] = ray.blocker_color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    rays = [_trace_ray(g, frame, frames) for frame in frames]
    return _render_with_order(g, rays, tuple(range(len(rays))))


def main(grid: np.ndarray) -> np.ndarray:
    """Send each marker straight out through its frame gap, recoloring the stopping cell with the blocking wall.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick hollow rectangular frames in distinct wall colors.
    - Each frame has exactly one missing wall pixel, and exactly one interior marker pixel aligned with that gap by row or column.
    - Marker colors may repeat, but every marker belongs to the unique frame that contains it.

    Rule:
    - For each frame, move its marker in the straight direction from the marker toward that frame's gap.
    - Paint every visited cell with the marker color; crossing any aligned gap is allowed because the gap cell is open.
    - Stop when the next cell ahead would be a wall pixel of some frame; recolor the last visited cell with that blocking wall's color instead. If no wall blocks the ray, continue painting until the grid edge.

    Key insight:
    - The apparent first frame encounter is not always the stopping point: a matching gap acts as a door, so the ray can continue past that opening before any wall finally stops it.
    """
    return _solve(np.asarray(grid, dtype=int))


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    for r, c in _border_cells(frame):
        if (r, c) != frame.gap:
            grid[r, c] = frame.wall_color
    mr, mc = frame.marker
    grid[mr, mc] = frame.marker_color


def _rand_from(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _pair_marker_col_left(c0: int, width: int) -> int:
    return c0 + random.choice([1, 2])


def _pair_marker_col_right(c0: int, width: int) -> int:
    return c0 + width - 1 - random.choice([1, 2])


def _build_canonical_scene() -> np.ndarray | None:
    pair_h = random.choice([5, 6])
    w_d = _rand_from(5, 6)
    w_a = _rand_from(5, 6)
    w_b = _rand_from(5, 6)
    gap_da = _rand_from(2, 3)
    gap_ab = _rand_from(2, 3)
    left_margin = _rand_from(1, 2)
    right_margin = _rand_from(2, 3)
    pair_r0 = _rand_from(1, 3)
    pair_rows = list(range(1, pair_h - 1))
    gap_row_local, blocker_row_local = random.sample(pair_rows, 2)
    gap_row = pair_r0 + gap_row_local
    blocker_row = pair_r0 + blocker_row_local

    c_d = left_margin
    c_a = c_d + w_d + gap_da
    c_b = c_a + w_a + gap_ab
    w_e = _rand_from(5, 8)
    h_e = _rand_from(4, 6)
    e_r0 = pair_r0 + pair_h + _rand_from(2, 4)
    height = e_r0 + h_e + _rand_from(1, 3)
    width = c_b + w_b + right_margin
    width = max(width, w_e + 5)
    if height > 30 or width > 30:
        return None

    palette = random.sample(COLOR_POOL, 7)
    wall_d, wall_a, wall_b, wall_e, marker_pair, marker_d, marker_e = palette

    frame_d = Frame(
        pair_r0,
        c_d,
        pair_r0 + pair_h - 1,
        c_d + w_d - 1,
        wall_d,
        (blocker_row, c_d + w_d - 1),
        (blocker_row, _pair_marker_col_right(c_d, w_d)),
        marker_d,
    )
    frame_a = Frame(
        pair_r0,
        c_a,
        pair_r0 + pair_h - 1,
        c_a + w_a - 1,
        wall_a,
        (gap_row, c_a + w_a - 1),
        (gap_row, _pair_marker_col_right(c_a, w_a)),
        marker_pair,
    )
    frame_b = Frame(
        pair_r0,
        c_b,
        pair_r0 + pair_h - 1,
        c_b + w_b - 1,
        wall_b,
        (gap_row, c_b),
        (gap_row, _pair_marker_col_left(c_b, w_b)),
        marker_pair,
    )

    gap_side_e = random.choice(["left", "right"])
    gap_row_e = e_r0 + random.randint(1, h_e - 2)
    if gap_side_e == "left":
        c_e = random.randint(3, max(3, width - w_e - 1))
        frame_e = Frame(
            e_r0,
            c_e,
            e_r0 + h_e - 1,
            c_e + w_e - 1,
            wall_e,
            (gap_row_e, c_e),
            (gap_row_e, _pair_marker_col_left(c_e, w_e)),
            marker_e,
        )
    else:
        c_e = random.randint(1, max(1, width - w_e - 3))
        frame_e = Frame(
            e_r0,
            c_e,
            e_r0 + h_e - 1,
            c_e + w_e - 1,
            wall_e,
            (gap_row_e, c_e + w_e - 1),
            (gap_row_e, _pair_marker_col_right(c_e, w_e)),
            marker_e,
        )

    frames = [frame_d, frame_a, frame_b, frame_e]
    for frame in frames:
        if frame.r1 >= height or frame.c1 >= width:
            return None

    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    return grid


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        g = g.T
    if random.random() < 0.5:
        g = np.flipud(g)
    if random.random() < 0.5:
        g = np.fliplr(g)
    return g.copy()


def _validate_scene(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    frames = _detect_frames(g)
    if len(frames) != 4:
        return False

    rays = [_trace_ray(g, frame, frames) for frame in frames]
    if not any(ray.blocker_color is None for ray in rays):
        return False
    if not any(ray.blocker_color is not None and not ray.through_other_gap for ray in rays):
        return False
    if sum(ray.through_other_gap for ray in rays) < 2:
        return False

    outputs = []
    for order in itertools.permutations(range(len(rays))):
        outputs.append(_render_with_order(g, rays, order))
    first = outputs[0]
    if not all(np.array_equal(first, out) for out in outputs[1:]):
        return False

    solved = main(g)
    if not np.array_equal(solved, first):
        return False

    if np.array_equal(g, solved):
        return False

    wall_cells = {
        cell
        for frame in frames
        for cell in (_border_cells(frame) - {frame.gap})
    }
    for ray in rays:
        if len(ray.path) < 2:
            return False
        if any(cell in wall_cells for cell in ray.path):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = _build_canonical_scene()
    if grid is None:
        return None
    grid = _apply_random_symmetry(grid)
    return grid if _validate_scene(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-door frame scene")

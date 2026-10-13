import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
ALL_COLORS = tuple(range(1, 10))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    excluded = set(map(int, exclude))
    return [int(color) for color in np.unique(grid) if int(color) not in excluded]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _classify_rectangles(grid: np.ndarray, rect_color: int) -> tuple[list[dict], np.ndarray]:
    rectangles: list[dict] = []
    index = np.full(grid.shape, -1, dtype=int)
    for comp in _components(grid == rect_color):
        r0, c0 = np.min(comp, axis=0)
        r1, c1 = np.max(comp, axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        local = np.zeros((h, w), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        if np.all(local):
            kind = "solid"
        else:
            outline = np.zeros((h, w), dtype=bool)
            outline[0, :] = True
            outline[-1, :] = True
            outline[:, 0] = True
            outline[:, -1] = True
            assert h >= 3 and w >= 3 and np.array_equal(local, outline), "expected solid blocks or hollow frames"
            kind = "frame"
        rect_id = len(rectangles)
        index[comp[:, 0], comp[:, 1]] = rect_id
        rectangles.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "kind": kind})
    assert rectangles, "expected at least one rectangle component"
    return rectangles, index


def _infer_roles(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[dict], np.ndarray]:
    bg = _get_bg(grid)
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) == 2, "expected one rectangle color and one marker color"

    marker_color = None
    rect_color = None
    markers: list[tuple[int, int]] = []
    for color in colors:
        comps = _components(grid == color)
        if comps and all(len(comp) == 1 for comp in comps):
            assert marker_color is None, "ambiguous marker color"
            marker_color = color
            markers = [tuple(map(int, comp[0])) for comp in comps]
        else:
            assert rect_color is None, "ambiguous rectangle color"
            rect_color = color

    assert marker_color is not None and rect_color is not None
    assert markers, "expected at least one singleton marker"
    rectangles, rect_index = _classify_rectangles(grid, rect_color)
    return bg, marker_color, markers, rectangles, rect_index


def _marker_direction(grid: np.ndarray, bg: int, marker_color: int, rect_color: int, marker: tuple[int, int]) -> tuple[int, int]:
    r, c = marker
    candidates: list[tuple[int, int]] = []
    for dr, dc in CARDINALS:
        rr, cc = r + dr, c + dc
        while _in_bounds(grid.shape, rr, cc):
            value = int(grid[rr, cc])
            if value == bg:
                rr += dr
                cc += dc
                continue
            if value == rect_color:
                candidates.append((dr, dc))
            else:
                assert value == marker_color, "unexpected foreground on sightline"
            break
    assert len(candidates) == 1, "each marker must have exactly one slide direction"
    return candidates[0]


def _trace_marker(
    out: np.ndarray,
    grid: np.ndarray,
    bg: int,
    marker_color: int,
    rect_color: int,
    rectangles: list[dict],
    rect_index: np.ndarray,
    marker: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    h, w = grid.shape
    line_r, line_c = marker
    dr, dc = direction
    cur_r, cur_c = marker

    while True:
        nr, nc = cur_r + dr, cur_c + dc
        while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg:
            cur_r, cur_c = nr, nc
            nr += dr
            nc += dc

        if not (0 <= nr < h and 0 <= nc < w):
            return

        assert int(grid[nr, nc]) == rect_color
        rect = rectangles[int(rect_index[nr, nc])]
        r0, c0, r1, c1 = rect["bbox"]
        if rect["kind"] == "solid":
            dest = (nr - dr, nc - dc)
            assert _in_bounds(grid.shape, dest[0], dest[1])
            assert int(grid[dest]) == bg, "solid blocks must have a free stopping cell"
            out[dest] = marker_color
            return

        if dr == 0:
            assert r0 < line_r < r1, "horizontal traversal must pass through frame interior"
            out[line_r, c0 + 1 : c1] = marker_color
            cur_r, cur_c = line_r, (c1 if dc > 0 else c0)
        else:
            assert c0 < line_c < c1, "vertical traversal must pass through frame interior"
            out[r0 + 1 : r1, line_c] = marker_color
            cur_r, cur_c = (r1 if dr > 0 else r0), line_c


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each marker through aligned hollow frames, leaving trails until a solid rectangle stops it.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms several axis-aligned rectangles: some are solid blocks and some are one-cell-thick hollow frames.
    - The other foreground color forms singleton markers, and each marker has exactly one unobstructed horizontal or vertical sightline toward the rectangle chain that controls it.

    Rule:
    - Clear every marker from its source cell, then slide it along its unique sightline.
    - When the marker meets a hollow frame, it passes through the near wall, colors every interior cell on that crossing line with the marker color, exits through the far wall, and keeps moving.
    - When the marker meets the first solid rectangle on that line, it stops on the background cell immediately outside that block; if no solid rectangle exists, the marker leaves the canvas and only its frame-interior trails remain.

    Key insight:
    - Hollow frames are transparent doorways rather than stopping walls, so the output is determined by two independent effects on the same ray: every crossed frame contributes a trail, while only the first solid block decides whether a final marker remains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, marker_color, markers, rectangles, rect_index = _infer_roles(g)
    rect_color = next(color for color in _colors_present(g, exclude={bg}) if color != marker_color)

    out = g.copy()
    for marker in markers:
        out[marker] = bg

    for marker in markers:
        direction = _marker_direction(g, bg, marker_color, rect_color, marker)
        _trace_marker(out, g, bg, marker_color, rect_color, rectangles, rect_index, marker, direction)

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = [cuts[0]]
    values.extend(cuts[i] - cuts[i - 1] for i in range(1, len(cuts)))
    values.append(total - cuts[-1])
    return values


def _pack_lanes(total: int, sizes: list[int]) -> list[int] | None:
    required_gaps = len(sizes) + 1
    slack = total - sum(sizes)
    if slack < required_gaps:
        return None
    gaps = _positive_composition(slack, required_gaps)
    starts: list[int] = []
    cursor = gaps[0]
    for size, gap_after in zip(sizes, gaps[1:]):
        starts.append(cursor)
        cursor += size + gap_after
    return starts


def _sample_motion_spec(axis_len: int, require_solid: bool, force_multi_frame: bool) -> dict | None:
    max_frames = 3 if axis_len >= 24 else 2
    min_frames = 2 if force_multi_frame else 1
    if min_frames > max_frames:
        return None

    for _ in range(120):
        n_frames = random.randint(min_frames, max_frames)
        kinds = ["frame"] * n_frames + (["solid"] if require_solid else [])
        sizes = [random.randint(4, 7) for _ in range(n_frames)]
        if require_solid:
            sizes.append(random.randint(3, 6))
        gaps = [random.randint(1, 3) for _ in range(len(kinds) - 1)]
        marker_margin = random.randint(1, 3)
        marker_gap = random.randint(1, 3)
        far_margin = random.randint(1, 3)
        total = marker_margin + 1 + marker_gap + sum(sizes) + sum(gaps) + far_margin
        if total > axis_len:
            continue

        marker_pos = marker_margin
        cursor = marker_pos + 1 + marker_gap
        segments = []
        for idx, (kind, size) in enumerate(zip(kinds, sizes)):
            segments.append((cursor, cursor + size - 1, kind))
            cursor += size
            if idx < len(gaps):
                cursor += gaps[idx]

        direction = random.choice((1, -1))
        if direction == -1:
            marker_pos = axis_len - 1 - marker_pos
            mirrored = []
            for start, end, kind in segments:
                mirrored.append((axis_len - 1 - end, axis_len - 1 - start, kind))
            segments = mirrored

        return {
            "marker_pos": marker_pos,
            "direction": direction,
            "segments": segments,
            "frame_count": n_frames,
            "has_solid": require_solid,
        }
    return None


def _draw_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, kind: str) -> None:
    r0, c0, r1, c1 = bbox
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BACKGROUND)
    if kind == "solid":
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _lane_solid_modes(count: int) -> list[bool]:
    if count == 1:
        return [random.random() < 0.6]
    modes = [True, False]
    while len(modes) < count:
        modes.append(random.random() < 0.5)
    random.shuffle(modes)
    return modes


def _gen_attempt(orientation: str) -> np.ndarray | None:
    assert orientation in {"horizontal", "vertical"}

    if orientation == "horizontal":
        h = random.randint(12, 30)
        w = random.randint(18, 30)
        cross_len = h
        axis_len = w
    else:
        h = random.randint(18, 30)
        w = random.randint(12, 30)
        cross_len = w
        axis_len = h

    max_lanes = min(3, (cross_len - 1) // 4)
    if max_lanes < 1:
        return None
    lane_count = random.randint(1, max_lanes)
    lane_sizes = [random.randint(3, 5) for _ in range(lane_count)]
    lane_starts = _pack_lanes(cross_len, lane_sizes)
    if lane_starts is None:
        return None

    wall_color, marker_color = random.sample(ALL_COLORS, 2)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    solid_modes = _lane_solid_modes(lane_count)
    multi_frame_lane = random.randrange(lane_count)

    for idx, (lane_start, lane_size) in enumerate(zip(lane_starts, lane_sizes)):
        spec = _sample_motion_spec(axis_len, solid_modes[idx], idx == multi_frame_lane)
        if spec is None:
            return None

        line_offset = random.randint(1, lane_size - 2)
        if orientation == "horizontal":
            marker = (lane_start + line_offset, spec["marker_pos"])
        else:
            marker = (spec["marker_pos"], lane_start + line_offset)
        assert int(grid[marker]) == BACKGROUND
        grid[marker] = marker_color

        for start, end, kind in spec["segments"]:
            if orientation == "horizontal":
                bbox = (lane_start, start, lane_start + lane_size - 1, end)
            else:
                bbox = (start, lane_start, end, lane_start + lane_size - 1)
            _draw_rectangle(grid, bbox, wall_color, kind)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if np.array_equal(solved, grid):
        return None
    new_marker_cells = np.count_nonzero((solved == marker_color) & (grid != marker_color))
    assert new_marker_cells >= 1, "expected at least one moved marker or trail cell"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        orientation = random.choice(("horizontal", "vertical"))
        grid = _gen_attempt(orientation)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")

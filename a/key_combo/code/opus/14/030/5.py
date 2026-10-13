from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER_COLOR = 5
FRAME_SIZE_RANGE = (3, 6)
GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 20)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_marker_component(
    coords: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 4:
        return None
    coord_set = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h == 2 and w == 3:
        row_counts = {r: sum(1 for rr, _ in coords if rr == r) for r in range(r0, r1 + 1)}
        if sorted(row_counts.values()) != [1, 3]:
            return None
        bar_row = next(r for r, cnt in row_counts.items() if cnt == 3)
        control_row = next(r for r, cnt in row_counts.items() if cnt == 1)
        middle_c = c0 + 1
        expected = {(bar_row, c0), (bar_row, middle_c), (bar_row, c1), (control_row, middle_c)}
        if coord_set != expected:
            return None
        center = (bar_row, middle_c)
        control = (control_row, middle_c)
        return center, control, (center[0] - control[0], 0)

    if h == 3 and w == 2:
        col_counts = {c: sum(1 for _, cc in coords if cc == c) for c in range(c0, c1 + 1)}
        if sorted(col_counts.values()) != [1, 3]:
            return None
        bar_col = next(c for c, cnt in col_counts.items() if cnt == 3)
        control_col = next(c for c, cnt in col_counts.items() if cnt == 1)
        middle_r = r0 + 1
        expected = {(r0, bar_col), (middle_r, bar_col), (r1, bar_col), (middle_r, control_col)}
        if coord_set != expected:
            return None
        center = (middle_r, bar_col)
        control = (middle_r, control_col)
        return center, control, (0, center[1] - control[1])

    return None


def _find_marker(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    candidates: list[tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]] = []
    for color in colors:
        for comp in _components_of_color(grid, color):
            parsed = _parse_marker_component(comp)
            if parsed is None:
                continue
            center, _control, step = parsed
            candidates.append((center, step, set(comp)))
    assert len(candidates) == 1, "expected exactly one T-shaped marker"
    return candidates[0]


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    coord_set = set(coords)
    perimeter = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )
    if coord_set != perimeter:
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return not np.any(interior == int(color))


def _find_frames(
    grid: np.ndarray,
    marker_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != BG]:
        for comp in _components_of_color(grid, color):
            coord_set = set(comp)
            if coord_set == marker_cells:
                continue
            if _is_rectangular_frame(grid, color, comp):
                frames.append(
                    {
                        "color": int(color),
                        "coords": coord_set,
                        "bbox": _bbox(comp),
                    }
                )
    assert frames, "expected at least one rectangular frame"
    return frames


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r = center[0] + step[0]
    c = center[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _analyze(grid: np.ndarray) -> dict[str, object]:
    center, step, marker_cells = _find_marker(grid)
    frames = _find_frames(grid, marker_cells)
    ray = set(_ray_cells(grid.shape, center, step))
    hit_frames = [frame for frame in frames if frame["coords"] & ray]
    return {
        "center": center,
        "step": step,
        "marker_cells": marker_cells,
        "frames": frames,
        "ray": ray,
        "hit_frames": hit_frames,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate every rectangular frame hit by the hidden marker ray by filling its enclosed area.

    Input structure:
    - The background is black.
    - Several colored axis-aligned rectangular frames appear as one-pixel outlines with black interiors.
    - One T-shaped marker consists of a three-pixel bar plus one adjacent control pixel; the ray starts at the bar center and goes away from the control.

    Rule:
    - Detect the unique T-shaped marker and cast its center ray horizontally or vertically until the grid edge.
    - Any rectangular frame whose outline contains at least one ray cell is activated.
    - For each activated frame, recolor all black interior cells to the frame's outline color; leave unhit frames, the marker, and all other cells unchanged.

    Key insight:
    - The ray is invisible and only serves as a selector: it chooses which hollow frames become solid while untouched frames stay hollow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "expected black background"

    analysis = _analyze(g)
    out = g.copy()
    for frame in analysis["hit_frames"]:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inside = out[r0 + 1 : r1, c0 + 1 : c1]
        inside[inside == BG] = color
    return out


def _marker_layout(
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    step = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    if step == (-1, 0):
        center = (h - 6, random.randint(3, w - 4))
        bar = [(center[0], center[1] - 1), center, (center[0], center[1] + 1)]
    elif step == (1, 0):
        center = (5, random.randint(3, w - 4))
        bar = [(center[0], center[1] - 1), center, (center[0], center[1] + 1)]
    elif step == (0, -1):
        center = (random.randint(3, h - 4), w - 6)
        bar = [(center[0] - 1, center[1]), center, (center[0] + 1, center[1])]
    else:
        center = (random.randint(3, h - 4), 5)
        bar = [(center[0] - 1, center[1]), center, (center[0] + 1, center[1])]
    control = (center[0] - step[0], center[1] - step[1])
    return center, step, bar, control


def _reserve_cells(mask: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _can_place_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(mask[max(0, r0 - 1) : r1 + 2, max(0, c0 - 1) : c1 + 2])


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_bbox(
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
    *,
    half: str,
    crosses_line: bool,
    reserved: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    for _ in range(250):
        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        if fh > h - 2 or fw > w - 2:
            continue

        if step[0] != 0:
            line = center[1]
            upper = (step[0] < 0 and half == "forward") or (step[0] > 0 and half == "opposite")
            if upper:
                r_min = 1
                r_max = center[0] - fh - 1
            else:
                r_min = center[0] + 2
                r_max = h - fh - 1
            if r_min > r_max:
                continue

            if crosses_line:
                c_min = max(1, line - fw + 1)
                c_max = min(w - fw - 1, line)
            else:
                left_ok = list(range(1, max(1, line - fw)))
                right_ok = list(range(line + 2, w - fw))
                starts = left_ok + right_ok
                if not starts:
                    continue
                c_min = c_max = random.choice(starts)

            r0 = random.randint(r_min, r_max)
            if crosses_line:
                if c_min > c_max:
                    continue
                c0 = random.randint(c_min, c_max)
            else:
                c0 = c_min
        else:
            line = center[0]
            left = (step[1] < 0 and half == "forward") or (step[1] > 0 and half == "opposite")
            if left:
                c_min = 1
                c_max = center[1] - fw - 1
            else:
                c_min = center[1] + 2
                c_max = w - fw - 1
            if c_min > c_max:
                continue

            if crosses_line:
                r_min = max(1, line - fh + 1)
                r_max = min(h - fh - 1, line)
            else:
                upper_ok = list(range(1, max(1, line - fh)))
                lower_ok = list(range(line + 2, h - fh))
                starts = upper_ok + lower_ok
                if not starts:
                    continue
                r_min = r_max = random.choice(starts)

            c0 = random.randint(c_min, c_max)
            if crosses_line:
                if r_min > r_max:
                    continue
                r0 = random.randint(r_min, r_max)
            else:
                r0 = r_min

        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if _can_place_bbox(reserved, bbox):
            return bbox
    return None


def _start_interval(
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
    half: str,
) -> tuple[int, int]:
    h, w = shape
    if step == (-1, 0):
        return (1, center[0] - 4) if half == "forward" else (center[0] + 2, h - 4)
    if step == (1, 0):
        return (center[0] + 2, h - 4) if half == "forward" else (1, center[0] - 4)
    if step == (0, -1):
        return (1, center[1] - 4) if half == "forward" else (center[1] + 2, w - 4)
    return (center[1] + 2, w - 4) if half == "forward" else (1, center[1] - 4)


def _hit_bands(start: int, end: int) -> list[tuple[int, int]]:
    if end - start < 5:
        return []
    first_end = min(start + 1, end - 5)
    second_start = max(start + 5, end - 1)
    if first_end < start or second_start > end:
        return []
    return [(start, first_end), (second_start, end)]


def _sample_cross_line_band(
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
    *,
    half: str,
    band: tuple[int, int],
    reserved: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    for _ in range(250):
        if step[0] != 0:
            fh = 3
            fw = random.randint(4, 5)
            r_min, r_max = _start_interval(shape, center, step, half)
            r_min = max(r_min, band[0])
            r_max = min(r_max, band[1])
            if r_min > r_max:
                continue
            line = center[1]
            c_min = max(1, line - fw + 1)
            c_max = min(w - fw - 1, line)
            if c_min > c_max:
                continue
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
        else:
            fh = random.randint(4, 5)
            fw = 3
            c_min, c_max = _start_interval(shape, center, step, half)
            c_min = max(c_min, band[0])
            c_max = min(c_max, band[1])
            if c_min > c_max:
                continue
            line = center[0]
            r_min = max(1, line - fh + 1)
            r_max = min(h - fh - 1, line)
            if r_min > r_max:
                continue
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if _can_place_bbox(reserved, bbox):
            return bbox
    return None


def _frame_crosses_selector_line(frame: dict[str, object], center: tuple[int, int], step: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    if step[0] != 0:
        return c0 <= center[1] <= c1
    return r0 <= center[0] <= r1


def _is_forward_side(frame: dict[str, object], center: tuple[int, int], step: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    if step == (-1, 0):
        return r1 < center[0]
    if step == (1, 0):
        return r0 > center[0]
    if step == (0, -1):
        return c1 < center[1]
    return c0 > center[1]


def _expected_output_from_analysis(inp: np.ndarray, analysis: dict[str, object]) -> np.ndarray:
    out = inp.copy()
    for frame in analysis["hit_frames"]:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        block = out[r0 + 1 : r1, c0 + 1 : c1]
        block[block == BG] = color
    return out


def _valid_instance(inp: np.ndarray) -> bool:
    try:
        analysis = _analyze(inp)
    except Exception:
        return False

    frames = analysis["frames"]
    hit_frames = analysis["hit_frames"]
    if not (4 <= len(frames) <= 6):
        return False
    if len(hit_frames) < 2 or len(hit_frames) >= len(frames):
        return False

    center = analysis["center"]
    step = analysis["step"]
    misses = [frame for frame in frames if frame not in hit_frames]
    forward_misses = [f for f in misses if _is_forward_side(f, center, step)]
    reverse_line_misses = [
        f for f in misses if (not _is_forward_side(f, center, step)) and _frame_crosses_selector_line(f, center, step)
    ]
    if not forward_misses or not reverse_line_misses:
        return False

    out = main(inp)
    expected = _expected_output_from_analysis(inp, analysis)
    if not np.array_equal(out, expected):
        return False

    changed = out != inp
    if not np.any(changed):
        return False

    allowed = np.zeros(inp.shape, dtype=bool)
    for frame in hit_frames:
        r0, c0, r1, c1 = frame["bbox"]
        allowed[r0 + 1 : r1, c0 + 1 : c1] = True
        color = int(frame["color"])
        region_in = inp[r0 + 1 : r1, c0 + 1 : c1]
        region_out = out[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(region_out[region_in == BG] != color):
            return False
    if np.any(changed & ~allowed):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    center, step, bar, control = _marker_layout(h, w)
    marker_cells = set(bar + [control])
    for cell in bar + [control]:
        grid[cell] = MARKER_COLOR

    reserved = np.zeros((h, w), dtype=bool)

    total_frames = random.randint(4, 6)
    extra_misses = total_frames - 4

    include_grey_frame = random.random() < 0.4
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    frame_colors = random.sample(palette, total_frames - int(include_grey_frame))
    if include_grey_frame:
        frame_colors.append(MARKER_COLOR)
    random.shuffle(frame_colors)

    placements: list[tuple[str, bool, tuple[int, int] | None]] = []
    forward_bands = _hit_bands(*_start_interval((h, w), center, step, "forward"))
    if len(forward_bands) != 2:
        return None
    placements.append(("forward", True, forward_bands[0]))
    placements.append(("forward", True, forward_bands[1]))
    placements.append(("opposite", True, _start_interval((h, w), center, step, "opposite")))
    placements.append(("forward", False, None))
    for _ in range(extra_misses):
        placements.append(random.choice([("forward", False, None), ("opposite", False, None)]))

    for color, (half, crosses_line, band) in zip(frame_colors, placements):
        if crosses_line and band is not None:
            bbox = _sample_cross_line_band((h, w), center, step, half=half, band=band, reserved=reserved)
        else:
            bbox = _sample_bbox((h, w), center, step, half=half, crosses_line=crosses_line, reserved=reserved)
        if bbox is None:
            return None
        _draw_frame(grid, bbox, color)
        r0, c0, r1, c1 = bbox
        reserved[max(0, r0 - 1) : r1 + 2, max(0, c0 - 1) : c1 + 2] = True

    return grid if _valid_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-frame-activation input")

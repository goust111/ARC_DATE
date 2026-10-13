from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if coords.shape[0] != 9:
            continue

        b = bbox(g == color)
        assert b is not None
        r0, c0, r1, c1 = b
        if r1 <= r0 or c1 <= c0:
            continue

        on_border = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        if bool(on_border.all()):
            candidates.append((int(color), b))

    assert len(candidates) == 1, "expected one nine-pixel incomplete rectangular frame"
    return candidates[0]


def _find_fill_color(
    g: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> int:
    r0, c0, r1, c1 = frame_bbox
    candidates: list[int] = []
    for color in colors_present(g, exclude={bg, frame_color}):
        coords = np.argwhere(g[r0 : r1 + 1, c0 : c1 + 1] == color)
        if coords.shape[0] != 3:
            continue

        coords = coords + np.array([r0, c0])
        on_border = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        tri_r0, tri_c0 = coords.min(axis=0)
        tri_r1, tri_c1 = coords.max(axis=0)
        touches_corner = (
            (tri_r0 == r0 and tri_c0 == c0)
            or (tri_r0 == r0 and tri_c1 == c1)
            or (tri_r1 == r1 and tri_c0 == c0)
            or (tri_r1 == r1 and tri_c1 == c1)
        )
        if (
            bool(on_border.all())
            and int(tri_r1 - tri_r0) == 1
            and int(tri_c1 - tri_c0) == 1
            and bool(touches_corner)
        ):
            candidates.append(int(color))

    assert len(candidates) == 1, "expected one colored L at a frame corner"
    return candidates[0]


def _is_arm_marker(g: np.ndarray, r: int, c: int, bg: int, frame_color: int) -> bool:
    color = int(g[r, c])
    return color != int(bg) and color != int(frame_color)


def _paint_arm_marker_rays(
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    h, w = g.shape

    for c in range(c0, c1 + 1):
        markers = [(r, int(g[r, c])) for r in range(0, r0) if _is_arm_marker(g, r, c, bg, frame_color)]
        start = 0
        for r, color in sorted(markers):
            out[start : r + 1, c] = color
            start = r + 1

    for c in range(c0, c1 + 1):
        markers = [
            (r, int(g[r, c])) for r in range(r1 + 1, h) if _is_arm_marker(g, r, c, bg, frame_color)
        ]
        end = h - 1
        for r, color in sorted(markers, reverse=True):
            out[r : end + 1, c] = color
            end = r - 1

    for r in range(r0, r1 + 1):
        markers = [(c, int(g[r, c])) for c in range(0, c0) if _is_arm_marker(g, r, c, bg, frame_color)]
        start = 0
        for c, color in sorted(markers):
            out[r, start : c + 1] = color
            start = c + 1

    for r in range(r0, r1 + 1):
        markers = [
            (c, int(g[r, c])) for c in range(c1 + 1, w) if _is_arm_marker(g, r, c, bg, frame_color)
        ]
        end = w - 1
        for c, color in sorted(markers, reverse=True):
            out[r, c : end + 1] = color
            end = c - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the incomplete rectangular frame, fill its row/column cross, and extend arm markers outward.

    Input structure
    - The background is the dominant color.
    - One foreground color forms a nine-pixel incomplete rectangular frame.
    - A three-pixel L of one other color sits in a corner of that frame and
      supplies the cross fill color.
    - Other marker pixels may lie in the four arms aligned with the frame rows
      or columns.

    Rule
    - Fill the union of the frame's rows and columns with the L color, preserving
      existing non-background marker pixels.
    - In each arm, extend every marker away from the frame until the grid edge
      or the next marker on the same row/column.
    - Redraw the full rectangular border with the frame color.

    Key insight
    - The sparse frame defines both a rectangle and a plus-shaped field; the
      colored L is the field color, while aligned markers cast outward rays away
      from the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    fill_color = _find_fill_color(g, bg, frame_color, frame_bbox)
    r0, c0, r1, c1 = frame_bbox

    inside_colors = set(colors_present(g[r0 : r1 + 1, c0 : c1 + 1], exclude={bg, frame_color, fill_color}))
    assert not inside_colors, "frame interior should contain only frame and fill-corner colors"

    out = g.copy()
    cross = np.zeros_like(g, dtype=bool)
    cross[r0 : r1 + 1, :] = True
    cross[:, c0 : c1 + 1] = True
    out[cross & (out == bg)] = fill_color

    _paint_arm_marker_rays(out, g, bg, frame_color, frame_bbox)

    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and frame geometry.
GEN_GRID_ROWS = (21, 25)
GEN_GRID_COLS = (21, 26)
GEN_FRAME_ROWS = (7, 10)
GEN_FRAME_COLS = (6, 9)
GEN_FRAME_MARGIN = 4

# Colors and corner labels.
GEN_FRAME_COLOR = 8
GEN_COLOR_VALUES = tuple(range(1, 10))
GEN_CORNERS = ("tl", "tr", "bl", "br")
GEN_MARKER_COLOR_POOL_SIZE = (6, 7)

# Marker counts and placement.
GEN_ARMS = ("top", "bottom", "left", "right")
GEN_ARM_MARKERS_PER_ARM = (1, 5)
GEN_TOTAL_ARM_MARKERS = (6, 14)
GEN_OUTSIDE_MARKERS = (3, 12)
GEN_DUPLICATE_LINE_PROB = 0.35

# Rejection sampling.
GEN_MAX_ATTEMPTS = 300

GEN_VALID_ARM_COUNTS = tuple(
    counts
    for a in range(GEN_ARM_MARKERS_PER_ARM[0], GEN_ARM_MARKERS_PER_ARM[1] + 1)
    for b in range(GEN_ARM_MARKERS_PER_ARM[0], GEN_ARM_MARKERS_PER_ARM[1] + 1)
    for c in range(GEN_ARM_MARKERS_PER_ARM[0], GEN_ARM_MARKERS_PER_ARM[1] + 1)
    for d in range(GEN_ARM_MARKERS_PER_ARM[0], GEN_ARM_MARKERS_PER_ARM[1] + 1)
    for counts in [(a, b, c, d)]
    if GEN_TOTAL_ARM_MARKERS[0] <= sum(counts) <= GEN_TOTAL_ARM_MARKERS[1]
)


def _gen_randint(bounds: tuple[int, int]) -> int:
    return int(np.random.randint(bounds[0], bounds[1] + 1))


def _gen_pick(items):
    return items[int(np.random.randint(0, len(items)))]


def _gen_frame_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _gen_corner_cells(r0: int, c0: int, r1: int, c1: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0)]
    if corner == "tr":
        return [(r0, c1), (r0, c1 - 1), (r0 + 1, c1)]
    if corner == "bl":
        return [(r1, c0), (r1, c0 + 1), (r1 - 1, c0)]
    return [(r1, c1), (r1, c1 - 1), (r1 - 1, c1)]


def _gen_sample_frame_layout() -> tuple[int, int, int, int, int, int]:
    """Return grid size and frame bbox while preserving the required edge margin."""
    h = _gen_randint(GEN_GRID_ROWS)
    w = _gen_randint(GEN_GRID_COLS)
    fh = _gen_randint(GEN_FRAME_ROWS)
    fw = _gen_randint(GEN_FRAME_COLS)
    r0 = int(np.random.randint(GEN_FRAME_MARGIN, h - fh - GEN_FRAME_MARGIN + 1))
    c0 = int(np.random.randint(GEN_FRAME_MARGIN, w - fw - GEN_FRAME_MARGIN + 1))
    return h, w, r0, c0, r0 + fh - 1, c0 + fw - 1


def _gen_sample_colors() -> tuple[int, int, str]:
    frame_color = GEN_FRAME_COLOR
    fill_color = int(_gen_pick([c for c in GEN_COLOR_VALUES if c != frame_color]))
    fill_corner = _gen_pick(GEN_CORNERS)
    return frame_color, fill_color, fill_corner


def _gen_sample_arm_counts() -> tuple[int, int, int, int, int]:
    target_total = _gen_randint(GEN_TOTAL_ARM_MARKERS)
    matching_counts = [counts for counts in GEN_VALID_ARM_COUNTS if sum(counts) == target_total]
    return target_total, *_gen_pick(matching_counts)


def _gen_place_frame_skeleton(grid, r0, c0, r1, c1, frame_color, fill_color, fill_corner) -> None:
    for corner in GEN_CORNERS:
        color = fill_color if corner == fill_corner else frame_color
        for r, c in _gen_corner_cells(r0, c0, r1, c1, corner):
            grid[r, c] = color


def _gen_arm_axes(arm, h, w, r0, c0, r1, c1):
    if arm == "top":
        return list(range(c0, c1 + 1)), list(range(0, r0))
    if arm == "bottom":
        return list(range(c0, c1 + 1)), list(range(r1 + 1, h))
    if arm == "left":
        return list(range(r0, r1 + 1)), list(range(0, c0))
    return list(range(r0, r1 + 1)), list(range(c1 + 1, w))


def _gen_arm_positions(arm, count, h, w, r0, c0, r1, c1) -> list[tuple[int, int]]:
    line_values, along_values = _gen_arm_axes(arm, h, w, r0, c0, r1, c1)

    duplicate_line = count >= 2 and len(along_values) >= 2 and np.random.random() < GEN_DUPLICATE_LINE_PROB
    n_lines = count - 1 if duplicate_line else count
    if n_lines > len(line_values):
        return []

    chosen_lines = list(np.random.choice(line_values, size=n_lines, replace=False))
    doubled = _gen_pick(chosen_lines) if duplicate_line else None
    positions: list[tuple[int, int]] = []
    for line in chosen_lines:
        n_on_line = 2 if line == doubled else 1
        picked_along = list(np.random.choice(along_values, size=n_on_line, replace=False))
        for along in picked_along:
            if arm in ("top", "bottom"):
                positions.append((int(along), int(line)))
            else:
                positions.append((int(line), int(along)))
    return positions


def _gen_all_arm_positions(counts, h, w, r0, c0, r1, c1) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for arm, count in zip(GEN_ARMS, counts):
        positions += _gen_arm_positions(arm, count, h, w, r0, c0, r1, c1)
    return positions


def _gen_quadrant_pools(h, w, r0, c0, r1, c1):
    quadrants = [
        (range(0, r0), range(0, c0)),
        (range(0, r0), range(c1 + 1, w)),
        (range(r1 + 1, h), range(0, c0)),
        (range(r1 + 1, h), range(c1 + 1, w)),
    ]
    return [[(r, c) for r in rs for c in cs] for rs, cs in quadrants]


def _gen_outside_positions(count, h, w, r0, c0, r1, c1) -> list[tuple[int, int]]:
    pools = _gen_quadrant_pools(h, w, r0, c0, r1, c1)
    chosen: list[tuple[int, int]] = []

    if count >= 4:
        first_quads = list(range(4))
    else:
        first_quads = list(np.random.choice(4, size=count, replace=False))
    for qi in first_quads:
        pos = _gen_pick(pools[qi])
        chosen.append(pos)
        pools[qi].remove(pos)

    while len(chosen) < count:
        available = [i for i, pool in enumerate(pools) if pool]
        if not available:
            return []
        qi = _gen_pick(available)
        pos = _gen_pick(pools[qi])
        chosen.append(pos)
        pools[qi].remove(pos)
    return chosen


def _gen_assign_marker_colors(marker_count: int, fill_color: int, frame_color: int) -> list[int]:
    available = [c for c in GEN_COLOR_VALUES if c not in (fill_color, frame_color)]
    pool_size = _gen_randint(GEN_MARKER_COLOR_POOL_SIZE)
    pool = list(np.random.choice(available, size=pool_size, replace=False))
    colors = pool[:]
    while len(colors) < marker_count:
        colors.append(int(_gen_pick(pool)))
    np.random.shuffle(colors)
    return [int(c) for c in colors[:marker_count]]


def _gen_paint_line_rays(out, by_pos, fixed_values, moving_values, vertical, reverse) -> None:
    for fixed in fixed_values:
        if vertical:
            markers = [(m, by_pos[(m, fixed)]) for m in moving_values if (m, fixed) in by_pos]
            limit = out.shape[0] - 1
        else:
            markers = [(m, by_pos[(fixed, m)]) for m in moving_values if (fixed, m) in by_pos]
            limit = out.shape[1] - 1
        if reverse:
            end = limit
            for moving, color in sorted(markers, reverse=True):
                if vertical:
                    out[moving : end + 1, fixed] = color
                else:
                    out[fixed, moving : end + 1] = color
                end = moving - 1
        else:
            start = 0
            for moving, color in sorted(markers):
                if vertical:
                    out[start : moving + 1, fixed] = color
                else:
                    out[fixed, start : moving + 1] = color
                start = moving + 1


def _gen_paint_output_rays(out, markers, r0, c0, r1, c1) -> None:
    h, w = out.shape
    by_pos = {(r, c): color for r, c, color in markers}

    _gen_paint_line_rays(out, by_pos, range(c0, c1 + 1), range(0, r0), vertical=True, reverse=False)
    _gen_paint_line_rays(out, by_pos, range(c0, c1 + 1), range(r1 + 1, h), vertical=True, reverse=True)
    _gen_paint_line_rays(out, by_pos, range(r0, r1 + 1), range(0, c0), vertical=False, reverse=False)
    _gen_paint_line_rays(out, by_pos, range(r0, r1 + 1), range(c1 + 1, w), vertical=False, reverse=True)


def _gen_make_markers(arm_positions, outside_positions, fill_color, frame_color):
    all_positions = arm_positions + outside_positions
    colors = _gen_assign_marker_colors(len(all_positions), fill_color, frame_color)
    arm_markers = [(r, c, color) for (r, c), color in zip(arm_positions, colors[: len(arm_positions)])]
    outside_markers = [
        (r, c, color) for (r, c), color in zip(outside_positions, colors[len(arm_positions) :])
    ]
    return arm_markers, outside_markers


def _gen_render_output(shape, box, frame_color, fill_color, arm_markers, outside_markers) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    out = np.zeros((h, w), dtype=int)
    out[r0 : r1 + 1, :] = fill_color
    out[:, c0 : c1 + 1] = fill_color
    _gen_paint_output_rays(out, arm_markers, r0, c0, r1, c1)
    for r, c, color in outside_markers:
        out[r, c] = color
    _gen_frame_border(out, r0, c0, r1, c1, frame_color)
    return out


def _gen_render_input(shape, box, frame_color, fill_color, fill_corner, arm_markers, outside_markers) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    inp = np.zeros((h, w), dtype=int)
    for r, c, color in arm_markers + outside_markers:
        inp[r, c] = color
    _gen_place_frame_skeleton(inp, r0, c0, r1, c1, frame_color, fill_color, fill_corner)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, r0, c0, r1, c1 = _gen_sample_frame_layout()
    shape = (h, w)
    box = (r0, c0, r1, c1)
    frame_color, fill_color, fill_corner = _gen_sample_colors()

    target_total, top_n, bottom_n, left_n, right_n = _gen_sample_arm_counts()
    arm_positions = _gen_all_arm_positions((top_n, bottom_n, left_n, right_n), h, w, r0, c0, r1, c1)
    if len(arm_positions) != target_total or len(set(arm_positions)) != target_total:
        return None

    outside_count = _gen_randint(GEN_OUTSIDE_MARKERS)
    outside_positions = _gen_outside_positions(outside_count, h, w, r0, c0, r1, c1)
    if len(outside_positions) != outside_count:
        return None

    arm_markers, outside_markers = _gen_make_markers(arm_positions, outside_positions, fill_color, frame_color)
    out = _gen_render_output(shape, box, frame_color, fill_color, arm_markers, outside_markers)
    inp = _gen_render_input(shape, box, frame_color, fill_color, fill_corner, arm_markers, outside_markers)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 256b0a75 input")

from __future__ import annotations

import math
import random

import numpy as np


BG = 0
WALL = 1
FILL_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


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
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _corner_background(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    values, counts = np.unique(np.array(corners, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_area = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            area = int(len(comp))
            if area > best_area:
                best_area = area
                best_color = color
    assert best_color is not None, "expected at least one non-background color"
    return int(best_color)


def _wall_lines(counts: np.ndarray) -> list[int]:
    vals = sorted(set(int(x) for x in counts.tolist()))
    assert len(vals) >= 2, "need at least two count levels to recover lattice lines"
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    split = max(range(len(gaps)), key=lambda i: (gaps[i], vals[i]))
    assert gaps[split] >= 2, "wall rows/cols must separate cleanly from interior rows/cols"
    threshold = (vals[split] + vals[split + 1]) / 2.0
    lines = [i for i, count in enumerate(counts.tolist()) if count > threshold]
    assert len(lines) >= 2, "expected at least two wall lines"
    return lines


def _parse_building(
    grid: np.ndarray, bbox_: tuple[int, int, int, int], wall: int
) -> list[tuple[slice, slice, tuple[bool, bool, bool, bool]]]:
    r0, c0, r1, c1 = bbox_
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    row_lines = _wall_lines(np.sum(sub == wall, axis=1))
    col_lines = _wall_lines(np.sum(sub == wall, axis=0))
    assert row_lines[0] == 0 and row_lines[-1] == sub.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == sub.shape[1] - 1

    row_steps = {row_lines[i + 1] - row_lines[i] for i in range(len(row_lines) - 1)}
    col_steps = {col_lines[i + 1] - col_lines[i] for i in range(len(col_lines) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "room lattice must be regular"
    row_step = next(iter(row_steps))
    col_step = next(iter(col_steps))
    assert row_step >= 3 and col_step >= 3, "rooms must have non-empty interiors"

    rooms = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            top = row_lines[i]
            bottom = row_lines[i + 1]
            left = col_lines[j]
            right = col_lines[j + 1]
            pattern = (
                bool(np.any(sub[top, left + 1 : right] != wall)),
                bool(np.any(sub[bottom, left + 1 : right] != wall)),
                bool(np.any(sub[top + 1 : bottom, left] != wall)),
                bool(np.any(sub[top + 1 : bottom, right] != wall)),
            )
            rooms.append(
                (
                    slice(r0 + top + 1, r0 + bottom),
                    slice(c0 + left + 1, c0 + right),
                    pattern,
                )
            )
    return rooms


def _parse_legend_box(
    grid: np.ndarray, bbox_: tuple[int, int, int, int], bg: int, wall: int
) -> tuple[tuple[bool, bool, bool, bool], int]:
    r0, c0, r1, c1 = bbox_
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    assert h >= 4 and w >= 4, "legend box must have a framed interior"
    inner = sub[1:-1, 1:-1]
    fill_colors = set(map(int, np.unique(inner))) - {bg, wall}
    assert len(fill_colors) == 1, "legend interior must use exactly one fill color"
    fill = fill_colors.pop()
    pattern = (
        bool(np.any(sub[0, 1:-1] != wall)),
        bool(np.any(sub[-1, 1:-1] != wall)),
        bool(np.any(sub[1:-1, 0] != wall)),
        bool(np.any(sub[1:-1, -1] != wall)),
    )
    return pattern, int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each plan room with the legend color keyed by its wall-gap directions.

    Input structure:
    - Black background surrounds a blue-walled rectangular building plan split into equal rectangular rooms.
    - Each room wall may contain a black gap on its top, bottom, left, and/or right side.
    - Separate legend squares sit in the lower-right area; each is a blue frame with one fill color and its own side-gap pattern.

    Rule:
    - Detect the blue wall color and isolate the large wall-only building component plus the smaller colored legend-box components.
    - For every room, read a 4-bit signature recording whether its top, bottom, left, and right wall segments contain a gap.
    - Read the same 4-bit signature from each legend square and use its interior color as the lookup value.
    - Fill only the black interior pixels of each room with the legend color for the matching signature, leaving blue walls and all legend boxes unchanged.

    Key insight:
    - The wall gaps are not damage to repair; they are a directional code, and the legend uses the same code to define a color lookup table.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_background(g)
    wall = _find_wall_color(g, bg)

    building_bbox = None
    building_area = -1
    legend_bboxes: list[tuple[int, int, int, int]] = []
    for comp in _components(g != bg):
        bbox_ = _bbox(comp)
        r0, c0, r1, c1 = bbox_
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(sub)))
        extras = colors - {bg, wall}
        if extras:
            legend_bboxes.append(bbox_)
        else:
            area = int(len(comp))
            if area > building_area:
                building_area = area
                building_bbox = bbox_

    assert building_bbox is not None, "expected one building component"
    assert legend_bboxes, "expected at least one legend box"

    mapping: dict[tuple[bool, bool, bool, bool], int] = {}
    for bbox_ in legend_bboxes:
        pattern, color = _parse_legend_box(g, bbox_, bg, wall)
        assert pattern not in mapping, "legend patterns must be unique"
        mapping[pattern] = color

    out = g.copy()
    for row_slice, col_slice, pattern in _parse_building(g, building_bbox, wall):
        assert pattern in mapping, "every room pattern must appear in the legend"
        patch = out[row_slice, col_slice]
        patch[patch == bg] = mapping[pattern]

    return out


def _build_patterns(
    room_rows: int, room_cols: int, h_open: list[list[bool]], v_open: list[list[bool]]
) -> list[list[tuple[bool, bool, bool, bool]]]:
    return [
        [
            (h_open[r][c], h_open[r + 1][c], v_open[r][c], v_open[r][c + 1])
            for c in range(room_cols)
        ]
        for r in range(room_rows)
    ]


def _repair_wall_lines(lines: list[list[bool]]) -> None:
    for line in lines:
        if all(line):
            line[random.randrange(len(line))] = False


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _make_gap_start(length: int, gap_len: int) -> int:
    return random.randint(0, length - gap_len)


def _draw_building(
    room_rows: int,
    room_cols: int,
    room_h: int,
    room_w: int,
    h_open: list[list[bool]],
    v_open: list[list[bool]],
    color_map: dict[tuple[bool, bool, bool, bool], int],
) -> tuple[np.ndarray, np.ndarray]:
    h = room_rows * room_h + room_rows + 1
    w = room_cols * room_w + room_cols + 1
    inp = np.full((h, w), BG, dtype=int)
    row_lines = [i * (room_h + 1) for i in range(room_rows + 1)]
    col_lines = [j * (room_w + 1) for j in range(room_cols + 1)]
    inp[row_lines, :] = WALL
    inp[:, col_lines] = WALL

    for i in range(room_rows + 1):
        r = row_lines[i]
        for j in range(room_cols):
            if not h_open[i][j]:
                continue
            start_c = col_lines[j] + 1
            gap_len = random.randint(1, room_w - 1)
            offset = _make_gap_start(room_w, gap_len)
            inp[r, start_c + offset : start_c + offset + gap_len] = BG

    for j in range(room_cols + 1):
        c = col_lines[j]
        for i in range(room_rows):
            if not v_open[i][j]:
                continue
            start_r = row_lines[i] + 1
            gap_len = random.randint(1, room_h - 1)
            offset = _make_gap_start(room_h, gap_len)
            inp[start_r + offset : start_r + offset + gap_len, c] = BG

    out = inp.copy()
    patterns = _build_patterns(room_rows, room_cols, h_open, v_open)
    for r in range(room_rows):
        for c in range(room_cols):
            fill = color_map[patterns[r][c]]
            rr0 = row_lines[r] + 1
            rr1 = row_lines[r + 1]
            cc0 = col_lines[c] + 1
            cc1 = col_lines[c + 1]
            out[rr0:rr1, cc0:cc1] = fill
    return inp, out


def _legend_box(pattern: tuple[bool, bool, bool, bool], color: int, inner: int) -> np.ndarray:
    box = np.full((inner + 2, inner + 2), BG, dtype=int)
    box[0, :] = WALL
    box[-1, :] = WALL
    box[:, 0] = WALL
    box[:, -1] = WALL
    box[1:-1, 1:-1] = color

    if pattern[0]:
        gap_len = random.randint(1, inner - 1)
        offset = _make_gap_start(inner, gap_len)
        box[0, 1 + offset : 1 + offset + gap_len] = BG
    if pattern[1]:
        gap_len = random.randint(1, inner - 1)
        offset = _make_gap_start(inner, gap_len)
        box[-1, 1 + offset : 1 + offset + gap_len] = BG
    if pattern[2]:
        gap_len = random.randint(1, inner - 1)
        offset = _make_gap_start(inner, gap_len)
        box[1 + offset : 1 + offset + gap_len, 0] = BG
    if pattern[3]:
        gap_len = random.randint(1, inner - 1)
        offset = _make_gap_start(inner, gap_len)
        box[1 + offset : 1 + offset + gap_len, -1] = BG
    return box


def _build_legend(
    patterns: list[tuple[bool, bool, bool, bool]], color_map: dict[tuple[bool, bool, bool, bool], int]
) -> np.ndarray | None:
    inner = random.randint(2, 3)
    box_size = inner + 2
    n = len(patterns)
    rows = random.randint(1, min(3, n))
    cols = math.ceil(n / rows)
    hgap = random.randint(1, 2)
    vgap = random.randint(1, 2)
    cluster_h = rows * box_size + (rows - 1) * vgap
    cluster_w = cols * box_size + (cols - 1) * hgap
    cluster = np.full((cluster_h, cluster_w), BG, dtype=int)
    order = patterns[:]
    random.shuffle(order)
    for idx, pattern in enumerate(order):
        rr = idx // cols
        cc = idx % cols
        top = rr * (box_size + vgap)
        left = cc * (box_size + hgap)
        _paste(cluster, _legend_box(pattern, color_map[pattern], inner), top, left)
    return cluster


def _is_single_wall_component(building: np.ndarray) -> bool:
    wall_comps = _components(building == WALL)
    return len(wall_comps) == 1


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    room_h = random.randint(2, 4)
    room_w = random.randint(2, 4)

    h_open = [[bool(random.getrandbits(1)) for _ in range(room_cols)] for _ in range(room_rows + 1)]
    v_open = [[bool(random.getrandbits(1)) for _ in range(room_cols + 1)] for _ in range(room_rows)]
    _repair_wall_lines(h_open)
    for c in range(room_cols + 1):
        if all(v_open[r][c] for r in range(room_rows)):
            v_open[random.randrange(room_rows)][c] = False

    patterns_2d = _build_patterns(room_rows, room_cols, h_open, v_open)
    flat_patterns = [pattern for row in patterns_2d for pattern in row]
    unique_patterns = sorted(set(flat_patterns))
    if not (3 <= len(unique_patterns) <= len(FILL_COLORS)):
        return None

    colors = random.sample(FILL_COLORS, len(unique_patterns))
    color_map = {pattern: color for pattern, color in zip(unique_patterns, colors, strict=False)}
    building_inp, building_out = _draw_building(room_rows, room_cols, room_h, room_w, h_open, v_open, color_map)
    if not _is_single_wall_component(building_inp):
        return None

    legend = _build_legend(unique_patterns, color_map)
    if legend is None:
        return None

    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gap_y = random.randint(2, 4)
    gap_x = random.randint(2, 4)

    height = top_margin + building_inp.shape[0] + gap_y + legend.shape[0] + bottom_margin
    width = left_margin + building_inp.shape[1] + gap_x + legend.shape[1] + right_margin
    if height > 30 or width > 30:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    build_top = top_margin
    build_left = left_margin
    legend_top = height - bottom_margin - legend.shape[0]
    legend_left = width - right_margin - legend.shape[1]
    _paste(inp, building_inp, build_top, build_left)
    _paste(out, building_out, build_top, build_left)
    _paste(inp, legend, legend_top, legend_left)
    _paste(out, legend, legend_top, legend_left)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _generate_case()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid room-gap legend puzzle")

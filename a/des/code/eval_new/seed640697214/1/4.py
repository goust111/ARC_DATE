from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

CELL_SIZE_OPTIONS = [3, 5]
FRAME_SHAPES = {
    3: [(2, 2), (2, 3), (3, 2), (3, 3)],
    5: [(2, 2), (2, 3), (3, 2)],
}
SIDE_NAMES = ("top", "bottom", "left", "right")
SIDE_GAP = 2
OUTER_MARGIN = 1
GEN_MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_template_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], np.ndarray, np.ndarray, int]:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch == color
            full_rows = np.flatnonzero(np.all(mask, axis=1))
            full_cols = np.flatnonzero(np.all(mask, axis=0))
            if len(full_rows) < 2 or len(full_cols) < 2:
                continue
            if full_rows[0] != 0 or full_rows[-1] != mask.shape[0] - 1:
                continue
            if full_cols[0] != 0 or full_cols[-1] != mask.shape[1] - 1:
                continue

            row_steps = np.diff(full_rows)
            col_steps = np.diff(full_cols)
            if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
                continue
            step = int(row_steps[0])
            if step < 2 or int(col_steps[0]) != step:
                continue

            row_mask = np.zeros(mask.shape[0], dtype=bool)
            col_mask = np.zeros(mask.shape[1], dtype=bool)
            row_mask[full_rows] = True
            col_mask[full_cols] = True
            expected = row_mask[:, None] | col_mask[None, :]
            if not np.array_equal(mask, expected):
                continue

            cell_size = step - 1
            candidates.append(
                (
                    int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    int(coords.shape[0]),
                    color,
                    (r0, c0, r1, c1),
                    full_rows,
                    full_cols,
                    cell_size,
                )
            )

    assert candidates, "expected one template frame"
    candidates.sort(reverse=True)
    _, _, color, bbox, full_rows, full_cols, cell_size = candidates[0]
    return color, bbox, full_rows, full_cols, cell_size


def _parse_marked_cells(
    grid: np.ndarray,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    full_rows: np.ndarray,
    full_cols: np.ndarray,
    bg: int,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    markers: list[tuple[int, int, int]] = []
    for cell_r in range(len(full_rows) - 1):
        rr0 = int(full_rows[cell_r] + 1)
        rr1 = int(full_rows[cell_r + 1])
        for cell_c in range(len(full_cols) - 1):
            cc0 = int(full_cols[cell_c] + 1)
            cc1 = int(full_cols[cell_c + 1])
            cell = patch[rr0:rr1, cc0:cc1]
            values = set(map(int, np.unique(cell))) - {bg, frame_color}
            if not values:
                continue
            assert len(values) == 1, "each marked cell must contain one marker color"
            color = values.pop()
            markers.append((cell_r, cell_c, color))
    return markers


def _find_source_rectangle(grid: np.ndarray, color: int, bg: int) -> tuple[int, int, int]:
    comps = [coords for coords in _components_of_color(grid, color) if int(coords.shape[0]) > 1]
    assert comps, f"missing rectangle for color {color}"
    comp = max(comps, key=lambda coords: int(coords.shape[0]))
    r0, c0, r1, c1 = _bbox_from_coords(comp)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    values = set(map(int, np.unique(patch))) - {bg, color}
    assert not values, "rectangle bbox should contain only the rectangle color and background holes"
    assert np.all(patch[0, :] == color) and np.all(patch[-1, :] == color)
    assert np.all(patch[:, 0] == color) and np.all(patch[:, -1] == color)
    holes = np.argwhere(patch == bg)
    if len(holes):
        assert np.all((holes[:, 0] > 0) & (holes[:, 0] < patch.shape[0] - 1))
        assert np.all((holes[:, 1] > 0) & (holes[:, 1] < patch.shape[1] - 1))
    return int(patch.shape[0]), int(patch.shape[1]), int(len(holes))


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _make_thumbnail_patch(
    rect_h: int, rect_w: int, holes: int, cell_size: int, color: int, bg: int
) -> np.ndarray:
    if rect_h >= rect_w:
        thumb_h = cell_size
        thumb_w = max(1, min(cell_size, _round_half_up(rect_w * cell_size / rect_h)))
    else:
        thumb_w = cell_size
        thumb_h = max(1, min(cell_size, _round_half_up(rect_h * cell_size / rect_w)))

    patch = np.full((cell_size, cell_size), bg, dtype=int)
    top = (cell_size - thumb_h) // 2
    left = (cell_size - thumb_w) // 2
    patch[top : top + thumb_h, left : left + thumb_w] = color
    return np.rot90(patch, -int(holes % 4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each frame marker with a rotated thumbnail of its same-colored rectangle block.

    Input structure:
    - The background is a dominant color surrounding several colored rectangle blocks.
    - Each block is an axis-aligned monochrome rectangle that may contain 0 to 3 single-cell background holes.
    - One monochrome frame component forms a rectangular grey table with 1-cell border lines and equal square cells.
    - Some frame cells contain a single marker pixel whose color matches exactly one outside rectangle block.

    Rule:
    - Identify the template frame and keep only that frame in the output.
    - For every marked frame cell, find the same-colored rectangle block outside the frame.
    - Ignore the block's holes, shrink its solid rectangle proportionally into one centered thumbnail inside a cell, and rotate that cell patch clockwise by 90 degrees per hole.
    - Erase all outside rectangles and marker pixels back to the background.

    Key insight:
    - Color chooses which rectangle to copy, while the count of tiny background holes independently chooses how many quarter-turns to rotate the thumbnail.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, bbox, full_rows, full_cols, cell_size = _find_template_frame(g, bg)
    markers = _parse_marked_cells(g, frame_color, bbox, full_rows, full_cols, bg)

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = bbox
    frame_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    out[r0 : r1 + 1, c0 : c1 + 1][frame_patch == frame_color] = frame_color

    rect_info: dict[int, tuple[int, int, int]] = {}
    for _cell_r, _cell_c, color in markers:
        if color not in rect_info:
            rect_info[color] = _find_source_rectangle(g, color, bg)

    for cell_r, cell_c, color in markers:
        rect_h, rect_w, holes = rect_info[color]
        thumb = _make_thumbnail_patch(rect_h, rect_w, holes, cell_size, color, bg)
        rr0 = r0 + int(full_rows[cell_r] + 1)
        cc0 = c0 + int(full_cols[cell_c] + 1)
        out[rr0 : rr0 + cell_size, cc0 : cc0 + cell_size] = thumb

    return out


def _draw_frame_patch(cell_rows: int, cell_cols: int, cell_size: int, frame_color: int, bg: int) -> np.ndarray:
    h = cell_rows * cell_size + cell_rows + 1
    w = cell_cols * cell_size + cell_cols + 1
    patch = np.full((h, w), bg, dtype=int)
    line_rows = [i * (cell_size + 1) for i in range(cell_rows + 1)]
    line_cols = [i * (cell_size + 1) for i in range(cell_cols + 1)]
    patch[line_rows, :] = frame_color
    patch[:, line_cols] = frame_color
    return patch


def _add_markers(
    frame_patch: np.ndarray,
    marker_specs: list[tuple[int, int, int]],
    cell_size: int,
) -> None:
    for cell_r, cell_c, color in marker_specs:
        rr = cell_r * (cell_size + 1) + 1 + cell_size // 2
        cc = cell_c * (cell_size + 1) + 1 + cell_size // 2
        frame_patch[rr, cc] = color


def _make_source_patch(color: int, source_h: int, source_w: int, holes: int, bg: int) -> np.ndarray:
    patch = np.full((source_h, source_w), color, dtype=int)
    candidates = [(r, c) for r in range(1, source_h - 1) for c in range(1, source_w - 1)]
    random.shuffle(candidates)
    assert holes <= len(candidates)
    for r, c in candidates[:holes]:
        patch[r, c] = bg
    return patch


def _build_expected_output(
    shape: tuple[int, int],
    frame_r0: int,
    frame_c0: int,
    frame_patch: np.ndarray,
    marker_specs: list[tuple[int, int, int]],
    rect_specs: dict[int, dict[str, int]],
    cell_size: int,
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[frame_r0 : frame_r0 + frame_patch.shape[0], frame_c0 : frame_c0 + frame_patch.shape[1]] = frame_patch
    for cell_r, cell_c, color in marker_specs:
        spec = rect_specs[color]
        thumb = _make_thumbnail_patch(
            spec["source_h"], spec["source_w"], spec["holes"], cell_size, color, bg
        )
        rr = frame_r0 + cell_r * (cell_size + 1) + 1
        cc = frame_c0 + cell_c * (cell_size + 1) + 1
        out[rr : rr + cell_size, cc : cc + cell_size] = thumb
    return out


def _boxes_conflict(
    boxes: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, pad: int = 1
) -> bool:
    for br, bc, bh, bw in boxes:
        separated = (
            r0 + h + pad <= br
            or br + bh + pad <= r0
            or c0 + w + pad <= bc
            or bc + bw + pad <= c0
        )
        if not separated:
            return True
    return False


def _candidate_positions_for_side(
    side: str,
    h: int,
    w: int,
    canvas_h: int,
    canvas_w: int,
    frame_r0: int,
    frame_c0: int,
    frame_h: int,
    frame_w: int,
    gap: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if side == "top":
        r_max = frame_r0 - gap - h
        if r_max < 0:
            return positions
        for r in range(r_max + 1):
            for c in range(canvas_w - w + 1):
                positions.append((r, c))
        return positions

    if side == "bottom":
        r_min = frame_r0 + frame_h + gap
        if r_min > canvas_h - h:
            return positions
        for r in range(r_min, canvas_h - h + 1):
            for c in range(canvas_w - w + 1):
                positions.append((r, c))
        return positions

    if side == "left":
        c_max = frame_c0 - gap - w
        if c_max < 0:
            return positions
        for r in range(canvas_h - h + 1):
            for c in range(c_max + 1):
                positions.append((r, c))
        return positions

    c_min = frame_c0 + frame_w + gap
    if c_min > canvas_w - w:
        return positions
    for r in range(canvas_h - h + 1):
        for c in range(c_min, canvas_w - w + 1):
            positions.append((r, c))
    return positions


def _gen_rect_specs(cell_size: int, n_rect: int) -> dict[int, dict[str, int]] | None:
    colors = random.sample(SOURCE_COLORS, n_rect)
    hole_counts = [0, 1, 2, 3]
    random.shuffle(hole_counts)
    if n_rect < 4:
        drop = random.randrange(4)
        hole_counts.pop(drop)

    specs: dict[int, dict[str, int]] = {}
    scale_choices = [2, 3] if cell_size == 3 else [2]
    for color, holes in zip(colors, hole_counts):
        short_thumb = random.randint(2, cell_size - 1)
        if random.random() < 0.5:
            thumb_h, thumb_w = cell_size, short_thumb
        else:
            thumb_h, thumb_w = short_thumb, cell_size
        scale = random.choice(scale_choices)
        source_h = thumb_h * scale
        source_w = thumb_w * scale
        if (source_h - 2) * (source_w - 2) < holes:
            return None
        specs[color] = {
            "color": color,
            "holes": holes,
            "source_h": source_h,
            "source_w": source_w,
            "thumb_h": thumb_h,
            "thumb_w": thumb_w,
        }
    return specs


def _gen_marker_specs(cell_rows: int, cell_cols: int, colors: list[int]) -> list[tuple[int, int, int]] | None:
    total_cells = cell_rows * cell_cols
    if total_cells <= len(colors):
        return None
    extra = random.randint(0, min(2, total_cells - len(colors) - 1))
    marker_colors = list(colors) + random.choices(colors, k=extra)
    positions = random.sample(range(total_cells), len(marker_colors))
    random.shuffle(marker_colors)
    markers = []
    for pos, color in zip(positions, marker_colors):
        markers.append((pos // cell_cols, pos % cell_cols, color))
    return markers


def _place_source_objects(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    side_assignments: dict[int, str],
    rect_specs: dict[int, dict[str, int]],
) -> bool:
    frame_r0, frame_c0, frame_h, frame_w = frame_box
    boxes: list[tuple[int, int, int, int]] = [frame_box]
    order = sorted(
        rect_specs.values(),
        key=lambda spec: -(int(spec["source_h"]) * int(spec["source_w"])),
    )

    for spec in order:
        side = side_assignments[spec["color"]]
        positions = _candidate_positions_for_side(
            side,
            spec["source_h"],
            spec["source_w"],
            grid.shape[0],
            grid.shape[1],
            frame_r0,
            frame_c0,
            frame_h,
            frame_w,
            SIDE_GAP,
        )
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _boxes_conflict(boxes, r0, c0, spec["source_h"], spec["source_w"], pad=1):
                continue
            patch = _make_source_patch(
                spec["color"], spec["source_h"], spec["source_w"], spec["holes"], BG_COLOR
            )
            grid[r0 : r0 + spec["source_h"], c0 : c0 + spec["source_w"]] = patch
            boxes.append((r0, c0, spec["source_h"], spec["source_w"]))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rect = random.choice([3, 4])
    cell_size = random.choice(CELL_SIZE_OPTIONS)
    frame_options = [shape for shape in FRAME_SHAPES[cell_size] if shape[0] * shape[1] >= n_rect + 1]
    if not frame_options:
        return None
    cell_rows, cell_cols = random.choice(frame_options)

    rect_specs = _gen_rect_specs(cell_size, n_rect)
    if rect_specs is None:
        return None
    colors = list(rect_specs)

    marker_specs = _gen_marker_specs(cell_rows, cell_cols, colors)
    if marker_specs is None:
        return None

    frame_input = _draw_frame_patch(cell_rows, cell_cols, cell_size, FRAME_COLOR, BG_COLOR)
    _add_markers(frame_input, marker_specs, cell_size)
    frame_output = _draw_frame_patch(cell_rows, cell_cols, cell_size, FRAME_COLOR, BG_COLOR)

    frame_h, frame_w = frame_input.shape
    chosen_sides = random.sample(SIDE_NAMES, n_rect)
    side_assignments = {color: side for color, side in zip(colors, chosen_sides)}

    top_h = max((rect_specs[c]["source_h"] for c, side in side_assignments.items() if side == "top"), default=0)
    bottom_h = max(
        (rect_specs[c]["source_h"] for c, side in side_assignments.items() if side == "bottom"),
        default=0,
    )
    left_w = max((rect_specs[c]["source_w"] for c, side in side_assignments.items() if side == "left"), default=0)
    right_w = max(
        (rect_specs[c]["source_w"] for c, side in side_assignments.items() if side == "right"),
        default=0,
    )
    top_w = max((rect_specs[c]["source_w"] for c, side in side_assignments.items() if side == "top"), default=0)
    bottom_w = max(
        (rect_specs[c]["source_w"] for c, side in side_assignments.items() if side == "bottom"),
        default=0,
    )
    left_h = max((rect_specs[c]["source_h"] for c, side in side_assignments.items() if side == "left"), default=0)
    right_h = max(
        (rect_specs[c]["source_h"] for c, side in side_assignments.items() if side == "right"),
        default=0,
    )

    top_block = top_h + (SIDE_GAP if top_h else 0)
    bottom_block = bottom_h + (SIDE_GAP if bottom_h else 0)
    left_block = left_w + (SIDE_GAP if left_w else 0)
    right_block = right_w + (SIDE_GAP if right_w else 0)

    base_h = OUTER_MARGIN + top_block + frame_h + bottom_block + OUTER_MARGIN
    base_w = OUTER_MARGIN + left_block + frame_w + right_block + OUTER_MARGIN
    min_h = max(base_h, left_h + 2 * OUTER_MARGIN, right_h + 2 * OUTER_MARGIN)
    min_w = max(base_w, top_w + 2 * OUTER_MARGIN, bottom_w + 2 * OUTER_MARGIN)
    if min_h > 30 or min_w > 30:
        return None

    canvas_h = random.randint(min_h, 30)
    canvas_w = random.randint(min_w, 30)
    extra_h = canvas_h - base_h
    extra_w = canvas_w - base_w
    extra_top = random.randint(0, extra_h)
    extra_left = random.randint(0, extra_w)
    frame_r0 = OUTER_MARGIN + extra_top + top_block
    frame_c0 = OUTER_MARGIN + extra_left + left_block

    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame_input

    if not _place_source_objects(
        grid, (frame_r0, frame_c0, frame_h, frame_w), side_assignments, rect_specs
    ):
        return None

    if _most_common_color(grid) != BG_COLOR:
        return None

    expected = _build_expected_output(
        grid.shape,
        frame_r0,
        frame_c0,
        frame_output,
        marker_specs,
        rect_specs,
        cell_size,
        BG_COLOR,
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")

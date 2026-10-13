from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

CONTROL_COLORS = (YELLOW, GREY, PINK, TEAL)
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    candidates = []
    for row in range(1, height - 1):
        band = grid[row, : width - 1]
        if band.size == 0:
            continue
        if int(band[0]) == BLACK:
            continue
        if np.all(band == band[0]) and row == height - row - 1:
            candidates.append(row)
    assert len(candidates) == 1, "expected one equal-height separator row"
    return int(candidates[0])


def _panel_foreground_color(panel: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(panel))) - {BLACK, RED})
    assert len(colors) == 1, "each panel must contain exactly one non-red foreground color"
    return int(colors[0])


def _axis_score(panel: np.ndarray, fg: int, axis2: int) -> tuple[int, int, int]:
    height, width = map(int, panel.shape)
    mismatch = 0
    recovered = 0
    support = 0

    for r in range(height):
        for c in range(width):
            mc = int(axis2 - c)
            if not (0 <= mc < width):
                if int(panel[r, c]) != BLACK:
                    mismatch += 1
                continue
            if c > mc:
                continue

            left = int(panel[r, c])
            right = int(panel[r, mc])

            if left == RED or right == RED:
                if left == fg or right == fg:
                    support += int(left == fg) + int(right == fg)
                continue

            left_fg = left == fg
            right_fg = right == fg
            if left_fg != right_fg:
                mismatch += 1
            if left_fg or right_fg:
                support += int(left_fg) + int(right_fg)

    for r, c in np.argwhere(panel == fg):
        mc = int(axis2 - int(c))
        if 0 <= mc < width and int(panel[int(r), mc]) == RED:
            recovered += 1

    return mismatch, -recovered, -support


def _detect_vertical_axis(panel: np.ndarray) -> int:
    fg = _panel_foreground_color(panel)
    width = int(panel.shape[1])
    ranked = sorted(
        (
            _axis_score(panel, fg, axis2) + (abs(axis2 - (width - 1)), axis2)
            for axis2 in range(2 * width - 1)
        )
    )
    best = ranked[0]
    assert best[0] == 0, "failed to find a symmetry axis consistent with the visible object"
    return int(best[-1])


def _complete_panel(panel: np.ndarray) -> np.ndarray:
    fg = _panel_foreground_color(panel)
    axis2 = _detect_vertical_axis(panel)
    height, width = map(int, panel.shape)
    completed = np.zeros((height, width), dtype=bool)

    for r, c in np.argwhere(panel == fg):
        r = int(r)
        c = int(c)
        mc = int(axis2 - c)
        assert 0 <= mc < width
        completed[r, c] = True
        completed[r, mc] = True

    return completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each panel's red occluder, mirror-complete the hidden symmetric shapes, then color only the shared overlap row-by-row.

    Input structure:
    - A black-background grid is split into equal-height top and bottom panels by one solid horizontal separator row.
    - The top panel contains a blue left-right symmetric shape partly covered by a red filled rectangle; the bottom panel contains a different green left-right symmetric shape with its own red occluder.
    - The rightmost column is a row-control column of visible colors; the separator does not occupy that final column.

    Rule:
    - Detect the separator row and remove it, keeping only the two aligned panels without the rightmost control column.
    - In each panel independently, treat red as an occluding rectangle, erase it conceptually, detect the vertical mirror axis, and restore missing foreground pixels by reflection.
    - Keep only positions that are foreground in both completed panels.
    - Recolor every kept pixel in row r with the control color from the input's rightmost column at row r, and discard the control column itself.

    Key insight:
    - Each panel alone provides an uncertain symmetric completion, but intersecting the two independently restored masks preserves only the positions jointly supported by both panels; the final color then comes from a separate per-row control signal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width >= 5 and height >= 7

    separator_row = _find_separator_row(g)
    panel_height = int(separator_row)
    panel_width = int(width - 1)

    top_panel = g[:panel_height, :panel_width]
    bottom_panel = g[separator_row + 1 :, :panel_width]
    assert top_panel.shape == bottom_panel.shape

    top_completed = _complete_panel(top_panel)
    bottom_completed = _complete_panel(bottom_panel)
    overlap = top_completed & bottom_completed

    control_colors = g[:panel_height, -1]
    out = np.zeros((panel_height, panel_width), dtype=int)
    for r in range(panel_height):
        out[r, overlap[r]] = int(control_colors[r])
    return out


def _row_mask(width: int, span: int) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    if span <= 0:
        return row
    start = (width - span) // 2
    row[start : start + span] = True
    return row


def _width_options(width: int, upper_margin: int) -> list[int]:
    max_span = width - upper_margin
    return [span for span in range(width % 2 or 1, max_span + 1, 2) if span > 0]


def _generate_core_widths(panel_height: int, panel_width: int) -> list[int] | None:
    active_len = random.randint(4, panel_height)
    active_start = random.randint(0, panel_height - active_len)
    active_rows = list(range(active_start, active_start + active_len))
    options = _width_options(panel_width, upper_margin=4)
    if len(options) < 2:
        return None

    idx = random.randint(0, len(options) - 2)
    widths = [0] * panel_height
    walk = []
    for _ in active_rows:
        idx = max(0, min(len(options) - 1, idx + random.choice((-1, 0, 1))))
        walk.append(options[idx])

    if max(walk) == min(walk):
        walk[random.randrange(len(walk))] = options[min(len(options) - 1, options.index(walk[0]) + 1)]

    for row, span in zip(active_rows, walk):
        widths[row] = span

    if sum(span > 0 for span in widths) < 4:
        return None
    return widths


def _mask_from_widths(widths: list[int], panel_width: int) -> np.ndarray:
    mask = np.zeros((len(widths), panel_width), dtype=bool)
    for r, span in enumerate(widths):
        mask[r] = _row_mask(panel_width, int(span))
    return mask


def _add_extras(core_widths: list[int], panel_width: int) -> tuple[list[int], list[int]] | None:
    max_span = panel_width - 2
    candidates = [r for r, span in enumerate(core_widths) if 0 < span <= max_span - 2]
    if len(candidates) < 2:
        return None

    top_widths = core_widths[:]
    bottom_widths = core_widths[:]

    random.shuffle(candidates)
    split = random.randint(1, len(candidates) - 1)
    top_rows = candidates[:split]
    bottom_rows = candidates[split:]
    if not top_rows or not bottom_rows:
        return None

    top_rows = top_rows[: random.randint(1, min(2, len(top_rows)))]
    bottom_rows = bottom_rows[: random.randint(1, min(2, len(bottom_rows)))]

    changed = False
    for rows, widths in ((top_rows, top_widths), (bottom_rows, bottom_widths)):
        for r in rows:
            room = max_span - widths[r]
            delta_choices = [delta for delta in (2, 4) if delta <= room]
            if not delta_choices:
                continue
            widths[r] += random.choice(delta_choices)
            changed = True

    if not changed or top_widths == bottom_widths:
        return None
    return top_widths, bottom_widths


def _occluder_candidates(
    full_mask: np.ndarray,
    core_mask: np.ndarray,
    side: str,
) -> list[tuple[int, int, int, int]]:
    height, width = map(int, full_mask.shape)
    axis_left_max = (width - 2) // 2
    axis_right_min = width // 2
    candidates = []

    for rect_h in range(2, min(3, height) + 1):
        for rect_w in range(2, min(4, width // 2) + 1):
            for r0 in range(height - rect_h + 1):
                if side == "left":
                    c_range = range(0, axis_left_max - rect_w + 2)
                else:
                    c_range = range(axis_right_min, width - rect_w + 1)
                for c0 in c_range:
                    rect = np.zeros_like(full_mask)
                    rect[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
                    total_hits = int(np.count_nonzero(rect & full_mask))
                    core_hits = int(np.count_nonzero(rect & core_mask))
                    if total_hits < 2 or core_hits < 2:
                        continue

                    valid = True
                    for rr, cc in np.argwhere(rect & full_mask):
                        rr = int(rr)
                        cc = int(cc)
                        mc = width - 1 - cc
                        if rect[rr, mc] or not full_mask[rr, mc]:
                            valid = False
                            break
                    if valid:
                        candidates.append((core_hits, total_hits, r0, c0, rect_h, rect_w))

    candidates.sort(reverse=True)
    return [(r0, c0, rect_h, rect_w) for _core, _total, r0, c0, rect_h, rect_w in candidates]


def _choose_occluder(
    full_mask: np.ndarray,
    core_mask: np.ndarray,
    forbidden: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    sides = ["left", "right"]
    random.shuffle(sides)
    choices = []
    for side in sides:
        choices.extend(_occluder_candidates(full_mask, core_mask, side)[:8])

    random.shuffle(choices)
    for choice in choices:
        if forbidden is None or choice != forbidden:
            return choice
    return None


def _apply_occluder(panel: np.ndarray, occluder: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, rect_h, rect_w = occluder
    out = panel.copy()
    out[r0 : r0 + rect_h, c0 : c0 + rect_w] = RED
    return out


def _control_sequence(panel_height: int) -> list[int]:
    band_count = random.randint(2, min(4, panel_height))
    cut_points = sorted(random.sample(range(1, panel_height), band_count - 1))
    cuts = [0, *cut_points, panel_height]
    colors = random.sample(list(CONTROL_COLORS), band_count)
    seq = []
    for i in range(band_count):
        seq.extend([colors[i]] * (cuts[i + 1] - cuts[i]))
    return seq


def _build_expected(core_mask: np.ndarray, control_colors: list[int]) -> np.ndarray:
    out = np.zeros(core_mask.shape, dtype=int)
    for r, color in enumerate(control_colors):
        out[r, core_mask[r]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_height = random.randint(5, 7)
    panel_width = random.randint(9, 13)

    core_widths = _generate_core_widths(panel_height, panel_width)
    if core_widths is None:
        return None
    widened = _add_extras(core_widths, panel_width)
    if widened is None:
        return None
    top_widths, bottom_widths = widened

    core_mask = _mask_from_widths(core_widths, panel_width)
    top_mask = _mask_from_widths(top_widths, panel_width)
    bottom_mask = _mask_from_widths(bottom_widths, panel_width)

    if not np.array_equal(top_mask & bottom_mask, core_mask):
        return None
    if np.count_nonzero(core_mask) < 10:
        return None
    if np.array_equal(top_mask, core_mask) or np.array_equal(bottom_mask, core_mask):
        return None

    top_occ = _choose_occluder(top_mask, core_mask)
    if top_occ is None:
        return None
    bottom_occ = _choose_occluder(bottom_mask, core_mask, forbidden=top_occ)
    if bottom_occ is None:
        return None

    top_panel = np.zeros((panel_height, panel_width), dtype=int)
    bottom_panel = np.zeros((panel_height, panel_width), dtype=int)
    top_panel[top_mask] = BLUE
    bottom_panel[bottom_mask] = GREEN
    top_panel = _apply_occluder(top_panel, top_occ)
    bottom_panel = _apply_occluder(bottom_panel, bottom_occ)

    if np.array_equal(top_panel == BLUE, top_mask) or np.array_equal(bottom_panel == GREEN, bottom_mask):
        return None

    control_top = _control_sequence(panel_height)
    control_separator = random.choice(CONTROL_COLORS)
    control_bottom = control_top[:]

    grid = np.zeros((2 * panel_height + 1, panel_width + 1), dtype=int)
    grid[:panel_height, :panel_width] = top_panel
    grid[panel_height, :panel_width] = ORANGE
    grid[panel_height + 1 :, :panel_width] = bottom_panel
    grid[:panel_height, -1] = control_top
    grid[panel_height, -1] = control_separator
    grid[panel_height + 1 :, -1] = control_bottom

    expected = _build_expected(core_mask, control_top)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 8")

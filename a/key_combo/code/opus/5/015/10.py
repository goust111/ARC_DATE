from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, paste


SEP_COLOR = Color.RED
GAP_COLOR = Color.MAROON
A_CROSS_COLOR = Color.PINK
B_CROSS_COLOR = Color.TEAL

BG_PAIRS = ((Color.GREY, Color.ORANGE), (Color.ORANGE, Color.GREY))
TOP_FG_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
MARKER_COLOR_TO_ROW = {
    Color.BLACK: 1,
    Color.PINK: 2,
    Color.TEAL: 3,
}
MARKER_COLOR_CYCLE = (Color.BLACK, Color.PINK, Color.TEAL)

WIDTH_RANGE = (6, 10)
MASK_SIZE_RANGE = (2, 5)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _full_separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP_COLOR)]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_cols = _full_separator_columns(grid)
    assert len(sep_cols) == 1
    sep_c = sep_cols[0]
    left = grid[:, :sep_c].copy()
    right = grid[:, sep_c + 1 :].copy()
    assert left.size > 0 and right.size > 0
    assert left.shape[0] == right.shape[0]
    assert left.shape[1] == right.shape[1]
    return left, right


def _panel_info(panel: np.ndarray) -> dict[str, object]:
    bg = _most_common_color(panel)
    top = panel[0].copy()
    mask_cols = tuple(int(c) for c in np.where(top != bg)[0])
    assert 2 <= len(mask_cols) <= 5

    marker_rows = [int(r) for r in range(1, panel.shape[0]) if int(panel[r, 0]) != bg]
    assert marker_rows
    marker_row = min(marker_rows)

    for r in range(1, panel.shape[0]):
        non_bg = tuple(int(c) for c in np.where(panel[r] != bg)[0])
        if r in marker_rows:
            assert non_bg == (0,)
        else:
            assert not non_bg

    return {
        "bg": int(bg),
        "panel": panel.copy(),
        "top": top,
        "mask": mask_cols,
        "marker_rows": tuple(marker_rows),
        "marker_row": int(marker_row),
        "marker_line": panel[marker_row].copy(),
        "width": int(panel.shape[1]),
    }


def _compress_panel(
    panel_info: dict[str, object],
    cross_mask: tuple[int, ...],
    cross_color: int,
    *,
    marker_row: int | None = None,
) -> np.ndarray:
    width = int(panel_info["width"])
    bg = int(panel_info["bg"])
    out = np.full((3, width), bg, dtype=int)
    out[0] = np.asarray(panel_info["top"], dtype=int)

    chosen_row = int(panel_info["marker_row"] if marker_row is None else marker_row)
    panel_line = np.asarray(panel_info["marker_line"], dtype=int)
    if marker_row is not None:
        panel_line = np.full(width, bg, dtype=int)
        panel_line[0] = int(np.asarray(panel_info["panel"], dtype=int)[chosen_row, 0])
    out[1] = panel_line

    for c in cross_mask:
        assert 0 <= int(c) < width
        out[2, int(c)] = int(cross_color)
    return out


def _assemble_output(comp_a: np.ndarray, comp_b: np.ndarray) -> np.ndarray:
    out_h = comp_a.shape[0] + 1 + comp_b.shape[0]
    out_w = comp_a.shape[1] + 1 + comp_b.shape[1]
    canvas = np.full((out_h, out_w), -1, dtype=int)
    paste(canvas, comp_a, 0, 0)
    paste(canvas, comp_b, comp_a.shape[0] + 1, comp_a.shape[1] + 1)
    used = canvas != -1
    used_box = bbox(used)
    assert used_box is not None
    cropped = crop_bbox(canvas, used_box)
    cropped[cropped == -1] = GAP_COLOR
    return cropped


def _solve_from_panels(panel_a: np.ndarray, panel_b: np.ndarray) -> np.ndarray:
    info_a = _panel_info(panel_a)
    info_b = _panel_info(panel_b)
    assert info_a["width"] == info_b["width"]
    assert info_a["mask"] != info_b["mask"]
    comp_a = _compress_panel(info_a, info_b["mask"], A_CROSS_COLOR)
    comp_b = _compress_panel(info_b, info_a["mask"], B_CROSS_COLOR)
    return _assemble_output(comp_a, comp_b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress two separator-split panels, but color each panel's third row from the other panel's top-row mask, then diagonally stitch the results.

    Input structure:
    - A single full red separator column splits the grid into left panel A and right panel B of equal width and equal height.
    - Each panel has one dominant background color, a top row whose non-background cells define that panel's mask, and one to three marker pixels placed only in column 0 below the top row.
    - Every non-top marker row is otherwise pure background, so the only below-top foreground in each panel is the column-0 marker stack.

    Rule:
    - Read both top-row masks and, in each panel, select the highest marker row below the top.
    - Compress each panel to three rows: row 0 is the original top row, row 1 is the selected marker row, and row 2 is the panel background recolored only at the other panel's mask columns.
    - Use magenta for panel A's cross-referenced row and teal for panel B's, place A at (0, 0), place B at (4, width(A)+1), and fill every uncovered cell inside the combined bounding box with maroon.

    Key insight:
    - The third row of each compressed panel depends on the opposite panel's mask rather than its own, so neither compressed strip is complete until both panels have been parsed.
    """
    g = np.asarray(grid, dtype=int)
    panel_a, panel_b = _split_panels(g)
    return _solve_from_panels(panel_a, panel_b)


def _sample_mask(width: int) -> tuple[int, ...]:
    size = random.randint(MASK_SIZE_RANGE[0], min(MASK_SIZE_RANGE[1], width))
    cols = random.sample(range(width), size)
    cols.sort()
    return tuple(int(c) for c in cols)


def _marker_count(mask: tuple[int, ...], width: int, panel_index: int) -> int:
    count = 1 + ((len(mask) + sum(mask) + width + panel_index) % 3)
    return int(count)


def _marker_rows(top_marker_color: int, count: int) -> tuple[int, ...]:
    top_row = MARKER_COLOR_TO_ROW[int(top_marker_color)]
    return tuple(int(top_row + 2 * i) for i in range(count))


def _marker_colors(top_marker_color: int, count: int) -> tuple[int, ...]:
    start = MARKER_COLOR_CYCLE.index(int(top_marker_color))
    return tuple(int(MARKER_COLOR_CYCLE[(start + i) % len(MARKER_COLOR_CYCLE)]) for i in range(count))


def _build_panel(
    width: int,
    height: int,
    bg: int,
    top_fg: int,
    mask: tuple[int, ...],
    top_marker_color: int,
    panel_index: int,
) -> np.ndarray:
    panel = np.full((height, width), int(bg), dtype=int)
    for c in mask:
        panel[0, int(c)] = int(top_fg)

    count = _marker_count(mask, width, panel_index)
    rows = _marker_rows(top_marker_color, count)
    colors = _marker_colors(top_marker_color, count)
    assert rows[-1] < height
    for r, color in zip(rows, colors):
        panel[int(r), 0] = int(color)
    return panel


def _sample_top_marker_colors(top_fg_a: int, top_fg_b: int) -> tuple[int, int]:
    choices_a = [c for c in MARKER_COLOR_CYCLE if c != top_fg_a]
    color_a = random.choice(choices_a)
    choices_b = [c for c in MARKER_COLOR_CYCLE if c != top_fg_b]
    color_b = random.choice(choices_b)
    return int(color_a), int(color_b)


def _semantic_checks(inp: np.ndarray) -> None:
    panel_a, panel_b = _split_panels(inp)
    info_a = _panel_info(panel_a)
    info_b = _panel_info(panel_b)

    assert info_a["mask"] != info_b["mask"]
    assert set(info_a["mask"]) - set(info_b["mask"])
    assert set(info_b["mask"]) - set(info_a["mask"])

    expected = _solve_from_panels(panel_a, panel_b)
    actual = main(inp)
    assert np.array_equal(actual, expected)

    self_mask = _assemble_output(
        _compress_panel(info_a, info_a["mask"], A_CROSS_COLOR),
        _compress_panel(info_b, info_b["mask"], B_CROSS_COLOR),
    )
    assert not np.array_equal(actual, self_mask)

    if len(info_a["marker_rows"]) > 1:
        wrong_a = _assemble_output(
            _compress_panel(info_a, info_b["mask"], A_CROSS_COLOR, marker_row=info_a["marker_rows"][-1]),
            _compress_panel(info_b, info_a["mask"], B_CROSS_COLOR),
        )
        assert not np.array_equal(actual, wrong_a)

    if len(info_b["marker_rows"]) > 1:
        wrong_b = _assemble_output(
            _compress_panel(info_a, info_b["mask"], A_CROSS_COLOR),
            _compress_panel(info_b, info_a["mask"], B_CROSS_COLOR, marker_row=info_b["marker_rows"][-1]),
        )
        assert not np.array_equal(actual, wrong_b)

    width = int(info_a["width"])
    assert np.count_nonzero(actual[2, :width] == A_CROSS_COLOR) == len(info_b["mask"])
    assert np.count_nonzero(actual[6, width + 1 :] == B_CROSS_COLOR) == len(info_a["mask"])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        width = random.randint(*WIDTH_RANGE)
        mask_a = _sample_mask(width)
        mask_b = _sample_mask(width)
        if mask_a == mask_b or len(set(mask_a) ^ set(mask_b)) < 2:
            continue

        bg_a, bg_b = random.choice(BG_PAIRS)
        top_fg_a = random.choice(TOP_FG_COLORS)
        top_fg_b = random.choice(TOP_FG_COLORS)
        top_marker_color_a, top_marker_color_b = _sample_top_marker_colors(top_fg_a, top_fg_b)

        count_a = _marker_count(mask_a, width, 0)
        count_b = _marker_count(mask_b, width, 1)
        if count_a == 1 and count_b == 1:
            continue

        rows_a = _marker_rows(top_marker_color_a, count_a)
        rows_b = _marker_rows(top_marker_color_b, count_b)
        height = max(rows_a[-1], rows_b[-1]) + 3
        if height > 30 or (2 * width + 1) > 30:
            continue

        panel_a = _build_panel(width, height, bg_a, top_fg_a, mask_a, top_marker_color_a, 0)
        panel_b = _build_panel(width, height, bg_b, top_fg_b, mask_b, top_marker_color_b, 1)

        grid = np.full((height, 2 * width + 1), SEP_COLOR, dtype=int)
        grid[:, :width] = panel_a
        grid[:, width] = SEP_COLOR
        grid[:, width + 1 :] = panel_b

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid cross-mask diagonal-stitch input")

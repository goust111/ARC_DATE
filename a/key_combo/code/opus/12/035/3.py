from __future__ import annotations

from collections import Counter
from importlib import util as importlib_util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib_util.spec_from_file_location("_arc_utils_shared_035_3", path)
    assert spec is not None and spec.loader is not None
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MOTION_VECTORS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]
GEN_ATTEMPTS = 400


def _full_height_separators(grid: np.ndarray) -> tuple[int, list[int]]:
    separator_cols: list[int] = []
    separator_colors: list[int] = []
    for c in range(grid.shape[1]):
        column = grid[:, c]
        if np.all(column == column[0]) and int(column[0]) != Color.BLACK:
            separator_cols.append(int(c))
            separator_colors.append(int(column[0]))
    assert len(separator_cols) == 4, "expected exactly four full-height separator columns"
    assert len(set(separator_colors)) == 1, "all separator columns must share one color"
    return separator_colors[0], separator_cols


def _extract_content(panel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows = np.where(np.any(panel != Color.BLACK, axis=1))[0]
    assert len(rows) > 0, "panel content missing"
    r0, r1 = int(rows[0]), int(rows[-1])
    content = panel[r0 : r1 + 1, :]
    assert np.all(content != Color.BLACK), "content area must be fully tiled by colored partitions"
    return content, (r0, r1)


def _compress_frame(content: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    components = find_components(
        content,
        bg=-1,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == 9, "expected a 3x3 grid of rectangular partitions"

    boxes = [tuple(map(int, comp["bbox"])) for comp in components]
    for comp, box in zip(components, boxes):
        r0, c0, r1, c1 = box
        area = int(comp["area"])
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "every component must be rectangular"

    row_tops = sorted({box[0] for box in boxes})
    col_lefts = sorted({box[1] for box in boxes})
    assert len(row_tops) == 3 and len(col_lefts) == 3, "partition grid must be 3x3"

    row_index = {top: idx for idx, top in enumerate(row_tops)}
    col_index = {left: idx for idx, left in enumerate(col_lefts)}

    row_heights = [None, None, None]
    col_widths = [None, None, None]
    small = np.zeros((3, 3), dtype=int)

    for comp, box in zip(components, boxes):
        r0, c0, r1, c1 = box
        ri = row_index[r0]
        ci = col_index[c0]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if row_heights[ri] is None:
            row_heights[ri] = height
        else:
            assert row_heights[ri] == height, "row partition heights must align across columns"
        if col_widths[ci] is None:
            col_widths[ci] = width
        else:
            assert col_widths[ci] == width, "column partition widths must align across rows"
        small[ri, ci] = int(comp["color"])

    assert all(v is not None for v in row_heights)
    assert all(v is not None for v in col_widths)
    return small, tuple(int(v) for v in row_heights), tuple(int(v) for v in col_widths)


def _majority_grid(frames: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(frames[0])
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            values = [int(frame[r, c]) for frame in frames]
            ranked = Counter(values).most_common()
            assert ranked[0][1] >= 2, "each base cell must have a unique majority color"
            out[r, c] = ranked[0][0]
    return out


def _render_partition_grid(
    small: np.ndarray,
    row_heights: tuple[int, int, int],
    col_widths: tuple[int, int, int],
) -> np.ndarray:
    out = np.zeros((sum(row_heights), sum(col_widths)), dtype=int)
    row_starts = np.cumsum((0, *row_heights[:-1]))
    col_starts = np.cumsum((0, *col_widths[:-1]))
    for r in range(3):
        for c in range(3):
            r0 = int(row_starts[r])
            c0 = int(col_starts[c])
            out[r0 : r0 + row_heights[r], c0 : c0 + col_widths[c]] = int(small[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each unequal partition frame, extrapolate the hidden constant-motion pixel, then expand it back to full partition geometry.

    Input structure:
    - The input is a black background with four full-height separator columns that split it into three side-by-side frame panels.
    - Inside each panel, the non-black content is a 3x3 tiling of monochromatic rectangular partitions with unequal row heights and column widths.
    - The three panels share the same partition geometry, while one compressed partition color acts as a moving marker across frames.

    Rule:
    - Detect the separator columns and crop the three panel contents.
    - In each panel, recover the 3x3 partition grid by finding rectangular 4-connected color components and compressing each partition to one pixel.
    - Take the cellwise majority across the three compressed frames as the static base grid, identify the one off-majority marker position in each frame, and continue its constant motion by one more step in the compressed 3x3 domain.
    - Expand the extrapolated compressed frame back with the same row heights and column widths, and output only that standalone content panel.

    Key insight:
    - Unequal partition sizes distort the moving region in pixel space, so the motion becomes simple and uniquely readable only after first collapsing each partition to a single cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _separator_color, separator_cols = _full_height_separators(g)
    panels = [g[:, separator_cols[i] + 1 : separator_cols[i + 1]] for i in range(3)]

    compressed_frames: list[np.ndarray] = []
    content_spans: list[tuple[int, int]] = []
    frame_geometries: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for panel in panels:
        content, span = _extract_content(panel)
        small, row_heights, col_widths = _compress_frame(content)
        compressed_frames.append(small)
        content_spans.append(span)
        frame_geometries.append((row_heights, col_widths))

    assert len(set(content_spans)) == 1, "all panels must share the same vertical content span"
    assert len(set(frame_geometries)) == 1, "all panels must share identical partition geometry"
    row_heights, col_widths = frame_geometries[0]

    base = _majority_grid(compressed_frames)
    marker_positions: list[tuple[int, int]] = []
    marker_colors: list[int] = []
    for frame in compressed_frames:
        diffs = np.argwhere(frame != base)
        assert diffs.shape == (1, 2), "each compressed frame must differ from the base in exactly one cell"
        r, c = map(int, diffs[0])
        marker_positions.append((r, c))
        marker_colors.append(int(frame[r, c]))

    assert len(set(marker_colors)) == 1, "the moving marker color must stay constant"
    assert len(set(marker_positions)) == 3, "the three observed marker positions must be distinct"

    dr1 = marker_positions[1][0] - marker_positions[0][0]
    dc1 = marker_positions[1][1] - marker_positions[0][1]
    dr2 = marker_positions[2][0] - marker_positions[1][0]
    dc2 = marker_positions[2][1] - marker_positions[1][1]
    assert (dr1, dc1) == (dr2, dc2) and (dr1, dc1) != (0, 0), "motion must be constant and non-zero"

    out_small = base.copy()
    next_r = marker_positions[2][0] + dr1
    next_c = marker_positions[2][1] + dc1
    if 0 <= next_r < 3 and 0 <= next_c < 3:
        out_small[next_r, next_c] = marker_colors[0]

    return _render_partition_grid(out_small, row_heights, col_widths)


def _random_partition(total: int) -> tuple[int, int, int]:
    while True:
        cuts = sorted(random.sample(range(1, total), 2))
        parts = (cuts[0], cuts[1] - cuts[0], total - cuts[1])
        if len(set(parts)) > 1:
            return parts


def _build_base_small() -> tuple[np.ndarray, int]:
    pattern_kind = random.choice(["checker", "latin"])
    if pattern_kind == "checker":
        colors = random.sample(PALETTE, 3)
        a, b, marker = colors
        phase = random.randint(0, 1)
        base = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                base[r, c] = a if (r + c + phase) % 2 == 0 else b
        return base, marker

    colors = random.sample(PALETTE, 4)
    cycle = colors[:3]
    marker = colors[3]
    phase = random.randint(0, 2)
    base = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            base[r, c] = cycle[(r + 2 * c + phase) % 3]
    return base, marker


def _vanishing_position_choices() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr, dc in MOTION_VECTORS:
        for r0 in range(3):
            for c0 in range(3):
                p0 = (r0, c0)
                p1 = (r0 + dr, c0 + dc)
                p2 = (r0 + 2 * dr, c0 + 2 * dc)
                p3 = (r0 + 3 * dr, c0 + 3 * dc)
                in012 = all(0 <= r < 3 and 0 <= c < 3 for r, c in (p0, p1, p2))
                in3 = 0 <= p3[0] < 3 and 0 <= p3[1] < 3
                if in012 and not in3:
                    choices.append((p0, (dr, dc)))
    return choices


def _frames_from_motion(
    base: np.ndarray,
    marker_color: int,
    start: tuple[int, int],
    delta: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray]:
    dr, dc = delta
    positions = [(start[0] + i * dr, start[1] + i * dc) for i in range(4)]

    frames = []
    for r, c in positions[:3]:
        frame = base.copy()
        frame[r, c] = marker_color
        frames.append(frame)

    out_small = base.copy()
    r3, c3 = positions[3]
    if 0 <= r3 < 3 and 0 <= c3 < 3:
        out_small[r3, c3] = marker_color
    return frames, out_small


def _render_input(
    frames: list[np.ndarray],
    row_heights: tuple[int, int, int],
    col_widths: tuple[int, int, int],
    top_pad: int,
    bottom_pad: int,
    separator_color: int,
) -> np.ndarray:
    content_h = sum(row_heights)
    content_w = sum(col_widths)
    total_h = top_pad + content_h + bottom_pad
    total_w = 4 + 3 * content_w
    grid = np.full((total_h, total_w), Color.BLACK, dtype=int)

    separator_cols = [0, content_w + 1, 2 * content_w + 2, 3 * content_w + 3]
    for c in separator_cols:
        grid[:, c] = separator_color

    for i, frame in enumerate(frames):
        rendered = _render_partition_grid(frame, row_heights, col_widths)
        c0 = separator_cols[i] + 1
        grid[top_pad : top_pad + content_h, c0 : c0 + content_w] = rendered
    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    content_h = random.randint(6, 9)
    content_w = random.randint(6, 8)
    row_heights = _random_partition(content_h)
    col_widths = _random_partition(content_w)

    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    if top_pad == 0 and bottom_pad == 0:
        if random.random() < 0.5:
            top_pad = 1
        else:
            bottom_pad = 1

    base, marker_color = _build_base_small()
    start, delta = random.choice(_vanishing_position_choices())
    frames, out_small = _frames_from_motion(base, marker_color, start, delta)

    separator_color = Color.GREY
    inp = _render_input(frames, row_heights, col_widths, top_pad, bottom_pad, separator_color)
    expected = _render_partition_grid(out_small, row_heights, col_widths)

    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert len(set(row_heights)) > 1 and len(set(col_widths)) > 1
    assert not np.array_equal(expected, _render_partition_grid(frames[-1], row_heights, col_widths))
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _generate_case()
        try:
            solved = main(inp)
        except Exception:
            continue

        if not np.array_equal(solved, expected):
            continue

        separator_cols = _full_height_separators(inp)[1]
        last_panel = inp[:, separator_cols[2] + 1 : separator_cols[3]]
        small_last, _, _ = _compress_frame(_extract_content(last_panel)[0])
        small_out, _, _ = _compress_frame(expected)
        if np.array_equal(small_last, small_out):
            continue
        return inp

    raise RuntimeError("failed to generate valid motion-after-compression example")

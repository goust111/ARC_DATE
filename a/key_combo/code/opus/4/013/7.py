from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_7_013",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BAR_PALETTE = [Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE]
GEN_ATTEMPTS = 400


def _groups(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(v) for v in np.asarray(indices, dtype=int).tolist()]
    if not vals:
        return []
    out = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        out.append((start, prev))
        start = prev = v
    out.append((start, prev))
    return out


def _cell_spans(line_mask: np.ndarray) -> list[tuple[int, int]]:
    groups = _groups(np.flatnonzero(line_mask))
    assert len(groups) >= 2, "expected at least two grid lines"
    spans = []
    for (_a0, a1), (b0, _b1) in zip(groups, groups[1:]):
        start = a1 + 1
        end = b0 - 1
        assert start <= end, "grid cells must have positive size"
        spans.append((start, end))
    return spans


def _extract_lattice(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    grey = find_components(
        grid,
        bg=get_bg(grid),
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert grey, "expected one grey lattice"
    lattice = max(grey, key=lambda comp: int(comp["area"]))
    patch = crop_bbox(grid, lattice["bbox"])
    row_spans = _cell_spans(np.all(patch == Color.GREY, axis=1))
    col_spans = _cell_spans(np.all(patch == Color.GREY, axis=0))
    return patch, row_spans, col_spans


def _decode_bar_component(grid: np.ndarray, comp: dict) -> dict:
    patch = crop_bbox(grid, comp["bbox"])
    color = int(comp["color"])
    mask = patch == color
    row_cols = [np.flatnonzero(mask[r]) for r in range(mask.shape[0])]
    assert all(cols.size > 0 for cols in row_cols), "component bbox should be tight"

    bar_h = 0
    center_local = None
    for cols in row_cols:
        if cols.size != 1:
            break
        center_here = int(cols[0])
        if center_local is None:
            center_local = center_here
        else:
            assert center_here == center_local, "bar stem must stay vertical"
        bar_h += 1
    assert bar_h >= 1 and center_local is not None, "expected a 1-pixel-wide vertical bar"

    motif_rows = row_cols[bar_h:]
    assert motif_rows, "expected a comb motif below the bar"
    solid_rows = 0
    prev_width = None
    for cols in motif_rows:
        width = int(cols.size)
        assert width in (2, 3), "comb rows must be side-pairs or solid triples"
        if prev_width is not None:
            assert width != prev_width, "comb rows must alternate"
        if width == 2:
            expected = np.array([center_local - 1, center_local + 1])
        else:
            expected = np.array([center_local - 1, center_local, center_local + 1])
            solid_rows += 1
        assert np.array_equal(cols, expected), "comb rows must stay centered under the bar"
        prev_width = width

    return {"color": color, "count": solid_rows}


def _fill_sorted_columns(
    lattice_patch: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    out = lattice_patch.copy()
    ordered = sorted(pairs, key=lambda item: item[1])
    for i, (color, count) in enumerate(ordered):
        c0, c1 = col_spans[i]
        for r0, r1 in row_spans[:count]:
            out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return np.fliplr(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode comb counts, sort the bar colors by those counts, fill a grey lattice, and mirror the finished lattice.

    Input structure:
    - A black background contains one grey rectangular lattice in the upper part of the grid.
    - The lattice consists of full grey horizontal and vertical lines enclosing empty black cells.
    - Below it are several distinct-color vertical bars; each bar continues into a same-color comb motif whose solid rows encode a count.

    Rule:
    - Crop the grey lattice and recover its row and column cell partitions from the full grey divider lines.
    - For each colored bar+comb component, count the number of 3-pixel solid comb rows.
    - Sort the colors by that count from small to large; fill the lattice columns left to right in that sorted order, coloring the top `k` cells of each column.
    - Leave all remaining lattice cells black and then horizontally mirror the whole cropped lattice, including the grey frame.

    Key insight:
    - Each comb height serves twice: it is both the sorting key that chooses the destination column and the fill depth painted into that column before the final mirror.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    lattice_patch, row_spans, col_spans = _extract_lattice(g)
    rows = len(row_spans)
    cols = len(col_spans)

    palette = colors_present(g, exclude={bg, Color.GREY})
    comps = find_components(
        g,
        bg=bg,
        colors=palette,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    decoded = [_decode_bar_component(g, comp) for comp in comps]
    counts = [item["count"] for item in decoded]

    assert len(decoded) == cols, "number of bars must match the lattice columns"
    assert len(set(counts)) == len(counts), "bar counts must be distinct"
    assert all(1 <= count <= rows for count in counts), "bar counts must fit inside the lattice"

    return _fill_sorted_columns(
        lattice_patch,
        row_spans,
        col_spans,
        [(item["color"], item["count"]) for item in decoded],
    )


def _render_lattice(rows: int, cols: int, cell_h: int = 2, cell_w: int = 2) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    patch = np.zeros((height, width), dtype=int)
    for r in range(rows + 1):
        patch[r * (cell_h + 1), :] = Color.GREY
    for c in range(cols + 1):
        patch[:, c * (cell_w + 1)] = Color.GREY
    row_spans = [(r * (cell_h + 1) + 1, r * (cell_h + 1) + cell_h) for r in range(rows)]
    col_spans = [(c * (cell_w + 1) + 1, c * (cell_w + 1) + cell_w) for c in range(cols)]
    return patch, row_spans, col_spans


def _sample_counts(rows: int, cols: int) -> list[int]:
    while True:
        counts = random.sample(list(range(1, rows + 1)), cols)
        if counts != sorted(counts) and counts != sorted(counts, reverse=True):
            return counts


def _oracle_output(
    lattice_patch: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    colors: list[int],
    counts: list[int],
) -> np.ndarray:
    return _fill_sorted_columns(lattice_patch, row_spans, col_spans, list(zip(colors, counts)))


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(4, 5)
    cols = random.randint(3, min(5, rows))
    lattice_patch, row_spans, col_spans = _render_lattice(rows, cols, cell_h=2, cell_w=2)
    lattice_h, lattice_w = lattice_patch.shape

    colors = random.sample(BAR_PALETTE, cols)
    counts = _sample_counts(rows, cols)
    bar_heights = [random.randint(2, 3) for _ in range(cols)]
    max_comp_h = max(bar_h + 2 * count - 1 for bar_h, count in zip(bar_heights, counts))

    free_h = 30 - (lattice_h + max_comp_h)
    if free_h < 0:
        return None
    top = random.randint(0, min(2, free_h))
    free_h -= top
    gap = random.randint(0, min(1, free_h))
    free_h -= gap
    bottom = random.randint(0, min(2, free_h))
    total_h = top + lattice_h + gap + max_comp_h + bottom

    step = random.choice([4, 5])
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    first_center = left_margin + 1
    centers = [first_center + step * i for i in range(cols)]
    min_w = centers[-1] + 2 + right_margin
    base_w = max(lattice_w + 1, min_w)
    if base_w > 30:
        return None
    total_w = base_w + random.randint(0, min(3, 30 - base_w))
    lattice_left = random.randint(0, total_w - lattice_w)

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[top : top + lattice_h, lattice_left : lattice_left + lattice_w] = lattice_patch

    bar_top = top + lattice_h + gap
    for color, count, bar_h, center in zip(colors, counts, bar_heights, centers):
        grid[bar_top : bar_top + bar_h, center] = int(color)
        comb_top = bar_top + bar_h
        for t in range(2 * count - 1):
            rr = comb_top + t
            if t % 2 == 0:
                grid[rr, center - 1 : center + 2] = int(color)
            else:
                grid[rr, center - 1] = int(color)
                grid[rr, center + 1] = int(color)

    expected = _oracle_output(lattice_patch, row_spans, col_spans, colors, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

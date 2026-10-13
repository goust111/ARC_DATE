from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
DIVIDER_COLOR = 5
ROW_MARKER_COLORS = (2, 1, 3)

TEMPLATE_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)

GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    start = 0
    pieces: list[tuple[int, int]] = []
    for divider in dividers:
        assert start < divider < length - 1
        pieces.append((start, divider))
        start = divider + 1
    pieces.append((start, length))
    return pieces


def _find_divider_layout(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    H, W = grid.shape
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        rows = [r for r in range(H) if np.all(grid[r, :] == color)]
        cols = [c for c in range(W) if np.all(grid[:, c] == color)]
        if len(rows) != 2 or len(cols) != 2:
            continue
        row_segments = _segments(H, rows)
        col_segments = _segments(W, cols)
        if len(row_segments) != 3 or len(col_segments) != 3:
            continue
        if len({r1 - r0 for r0, r1 in row_segments}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_segments}) != 1:
            continue
        candidates.append((color, row_segments, col_segments))
    assert len(candidates) == 1, "expected one 3x3 divider grid"
    return candidates[0]


def _cell(grid: np.ndarray, row_seg: tuple[int, int], col_seg: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_seg
    c0, c1 = col_seg
    return grid[r0:r1, c0:c1]


def _extract_row_markers(
    grid: np.ndarray,
    bg: int,
    divider_color: int,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> list[int]:
    row_colors: list[int] = []
    for row_seg in row_segments:
        cell = _cell(grid, row_seg, col_segments[0])
        colors = [int(c) for c in np.unique(cell) if int(c) not in (int(bg), int(divider_color))]
        assert len(colors) == 1, "each marker cell must contain one marker color"
        color = colors[0]
        assert np.count_nonzero(cell == color) == 1, "row marker must be a single pixel"
        row_colors.append(color)
    return row_colors


def _extract_legend(
    grid: np.ndarray, bg: int, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    cell = _cell(grid, row_segments[0], col_segments[1])
    coords = np.argwhere(cell != bg)
    assert coords.size > 0, "legend cell must not be empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r0) == 0 and int(c0) == 0, "legend block must start in the cell's top-left corner"
    legend = cell[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert legend.shape[1] == 2, "legend must be two columns wide"
    assert legend.shape[0] in (2, 3), "legend must have 2 or 3 rows"
    assert np.all(legend != bg), "legend block must be fully filled"
    return [(int(legend[r, 0]), int(legend[r, 1])) for r in range(legend.shape[0])]


def _extract_template(
    grid: np.ndarray, bg: int, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int]], int]:
    cell = _cell(grid, row_segments[0], col_segments[2])
    coords = np.argwhere(cell != bg)
    assert len(coords) in (3, 4), "template must contain 3 or 4 colored pixels"
    colors = Counter(int(cell[r, c]) for r, c in coords)
    assert len(colors) == 2, "template must have one main color and one auxiliary color"
    (main_color, main_count), (_, aux_count) = colors.most_common()
    assert main_count > aux_count, "template main color must be uniquely most frequent"
    pixels = sorted((int(r), int(c), int(cell[r, c])) for r, c in coords)
    return pixels, int(main_color)


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> tuple[int, list[int]]:
    current = int(color)
    hits: list[int] = []
    for index, (src, dst) in enumerate(pairs):
        if current == int(src):
            current = int(dst)
            hits.append(index)
    return current, hits


def _paint_template(
    cell: np.ndarray,
    bg: int,
    template_pixels: list[tuple[int, int, int]],
    template_main_color: int,
    row_color: int,
) -> None:
    cell[:, :] = int(bg)
    for r, c, color in template_pixels:
        cell[r, c] = int(row_color if color == template_main_color else color)


def _target_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(1, 3) if not (r == 0 and c == 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a multicolor template into the non-marker cells, then recolor it by the ordered legend chain.

    Input structure:
    - A black background contains one grey divider grid made of two full horizontal and two
      full vertical bars, splitting the board into 3x3 equal square cells.
    - The first column cells each hold one single-pixel row marker color; the top-middle cell
      holds a 2-column legend block; the top-right cell holds a 3-4 pixel two-color template.
    - All remaining cell interiors are black.

    Rule:
    - Read the three row marker colors from the first column, the ordered legend pairs from the
      top-middle cell, and the template pixels from the top-right cell.
    - For every cell except the marker column and the legend cell, clear the cell and stamp the
      template into it; replace the template's unique majority color with that row's marker color
      while keeping the auxiliary template color unchanged.
    - Then, for each stamped non-background pixel, scan the legend rows from top to bottom and
      repeatedly replace matching source colors with their row's destination color; divider bars,
      marker cells, and the legend cell stay unchanged.

    Key insight:
    - The row marker does not directly give the final color: it first enters the stamped template,
      and only then can the ordered legend chain rewrite that injected color one or more times.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    divider_color, row_segments, col_segments = _find_divider_layout(g, bg)
    row_colors = _extract_row_markers(g, bg, divider_color, row_segments, col_segments)
    legend_pairs = _extract_legend(g, bg, row_segments, col_segments)
    template_pixels, template_main_color = _extract_template(g, bg, row_segments, col_segments)

    out = g.copy()
    for row_index, col_index in _target_cells():
        cell = _cell(out, row_segments[row_index], col_segments[col_index])
        _paint_template(cell, bg, template_pixels, template_main_color, row_colors[row_index])
        for r, c in np.argwhere(cell != bg):
            cell[r, c], _ = _apply_chain(int(cell[r, c]), legend_pairs)

    return out


def _layout_segments(cell_size: int) -> list[tuple[int, int]]:
    return [
        (0, cell_size),
        (cell_size + 1, 2 * cell_size + 1),
        (2 * cell_size + 2, 3 * cell_size + 2),
    ]


def _blank_grid(cell_size: int) -> np.ndarray:
    size = 3 * cell_size + 2
    grid = np.full((size, size), BG_COLOR, dtype=int)
    for divider in (cell_size, 2 * cell_size + 1):
        grid[divider, :] = DIVIDER_COLOR
        grid[:, divider] = DIVIDER_COLOR
    return grid


def _place_markers(grid: np.ndarray, cell_size: int) -> None:
    segments = _layout_segments(cell_size)
    center = cell_size // 2
    for row_index, row_color in enumerate(ROW_MARKER_COLORS):
        r0, _r1 = segments[row_index]
        c0, _c1 = segments[0]
        grid[r0 + center, c0 + center] = row_color


def _place_legend(grid: np.ndarray, cell_size: int, legend_pairs: list[tuple[int, int]]) -> None:
    segments = _layout_segments(cell_size)
    r0, _r1 = segments[0]
    c0, _c1 = segments[1]
    for row, (src, dst) in enumerate(legend_pairs):
        grid[r0 + row, c0 + 0] = src
        grid[r0 + row, c0 + 1] = dst


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return list(shape), h, w


def _make_template_pixels(
    cell_size: int, main_color: int, aux_color: int
) -> tuple[list[tuple[int, int, int]], int]:
    shape = random.choice(TEMPLATE_SHAPES)
    coords, height, width = _normalize_shape(shape)
    row_offset = random.randint(0, cell_size - height)
    col_offset = random.randint(0, cell_size - width)
    aux_index = random.randrange(len(coords))
    pixels: list[tuple[int, int, int]] = []
    for index, (r, c) in enumerate(coords):
        color = aux_color if index == aux_index else main_color
        pixels.append((row_offset + r, col_offset + c, color))
    return sorted(pixels), main_color


def _choose_rule_spec() -> tuple[list[tuple[int, int]], int, int]:
    extra_colors = [4, 6, 7, 8, 9]
    family = random.choice(("linear2", "direct3", "cascade3"))
    if family == "linear2":
        row_start = random.choice(ROW_MARKER_COLORS)
        mid, end, main_color = random.sample(extra_colors, 3)
        aux_color = mid
        pairs = [(row_start, mid), (mid, end)]
        return pairs, main_color, aux_color

    if family == "direct3":
        row_start, row_direct = random.sample(list(ROW_MARKER_COLORS), 2)
        chain_mid, direct_final, chain_final, main_color = random.sample(extra_colors, 4)
        aux_color = row_direct
        pairs = [(row_start, chain_mid), (row_direct, direct_final), (chain_mid, chain_final)]
        return pairs, main_color, aux_color

    row_start, row_mid = random.sample(list(ROW_MARKER_COLORS), 2)
    aux_color, final_color, main_color = random.sample(extra_colors, 3)
    pairs = [(row_start, row_mid), (row_mid, aux_color), (aux_color, final_color)]
    return pairs, main_color, aux_color


def _build_input(spec: dict) -> np.ndarray:
    cell_size = int(spec["cell_size"])
    grid = _blank_grid(cell_size)
    _place_markers(grid, cell_size)
    _place_legend(grid, cell_size, spec["legend_pairs"])

    segments = _layout_segments(cell_size)
    r0, _r1 = segments[0]
    c0, _c1 = segments[2]
    for r, c, color in spec["template_pixels"]:
        grid[r0 + r, c0 + c] = color
    return grid


def _build_output(spec: dict, *, apply_chain: bool) -> np.ndarray:
    inp = _build_input(spec)
    out = inp.copy()
    segments = _layout_segments(int(spec["cell_size"]))
    for row_index, col_index in _target_cells():
        row_seg = segments[row_index]
        col_seg = segments[col_index]
        cell = _cell(out, row_seg, col_seg)
        _paint_template(
            cell,
            BG_COLOR,
            spec["template_pixels"],
            spec["template_main_color"],
            ROW_MARKER_COLORS[row_index],
        )
        if not apply_chain:
            continue
        for r, c in np.argwhere(cell != BG_COLOR):
            cell[r, c], _ = _apply_chain(int(cell[r, c]), spec["legend_pairs"])
    return out


def _rule_usage(spec: dict) -> tuple[list[int], int]:
    hits_per_rule = [0] * len(spec["legend_pairs"])
    max_chain_len = 0
    for row_index, _col_index in _target_cells():
        for _r, _c, color in spec["template_pixels"]:
            stamped = ROW_MARKER_COLORS[row_index] if color == spec["template_main_color"] else color
            _final, hits = _apply_chain(stamped, spec["legend_pairs"])
            max_chain_len = max(max_chain_len, len(hits))
            for hit in hits:
                hits_per_rule[hit] += 1
    return hits_per_rule, max_chain_len


def _target_signatures(grid: np.ndarray, cell_size: int) -> set[bytes]:
    segments = _layout_segments(cell_size)
    signatures: set[bytes] = set()
    for row_index, col_index in _target_cells():
        cell = _cell(grid, segments[row_index], segments[col_index])
        signatures.add(cell.tobytes())
    return signatures


def _valid_spec(spec: dict) -> bool:
    inp = _build_input(spec)
    expected = _build_output(spec, apply_chain=True)
    stamp_only = _build_output(spec, apply_chain=False)

    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(expected, stamp_only):
        return False

    hits_per_rule, max_chain_len = _rule_usage(spec)
    if any(count == 0 for count in hits_per_rule):
        return False
    if max_chain_len < 2:
        return False
    if len(_target_signatures(expected, int(spec["cell_size"]))) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        cell_size = random.randint(4, 6)
        legend_pairs, template_main_color, template_aux_color = _choose_rule_spec()
        template_pixels, template_main_color = _make_template_pixels(
            cell_size, template_main_color, template_aux_color
        )
        spec = {
            "cell_size": cell_size,
            "legend_pairs": legend_pairs,
            "template_pixels": template_pixels,
            "template_main_color": template_main_color,
        }
        if _valid_spec(spec):
            return _build_input(spec)
    raise RuntimeError("failed to generate a valid ARC input")

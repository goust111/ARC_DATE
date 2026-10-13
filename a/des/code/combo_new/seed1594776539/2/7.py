from __future__ import annotations

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _cell_ranges(separators: list[int], limit: int) -> list[tuple[int, int]]:
    assert separators[0] == 0 and separators[-1] == limit - 1, "legend must be framed by separator rows"
    ranges = []
    for a, b in zip(separators, separators[1:]):
        assert b > a + 1, "legend cells must have non-empty interiors"
        ranges.append((a + 1, b - 1))
    return ranges


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = np.asarray(coords, dtype=int)
    top, left = pts.min(axis=0)
    rel = tuple(sorted((int(r - top), int(c - left)) for r, c in pts))
    height = max(r for r, _ in rel) + 1
    width = max(c for _, c in rel) + 1
    return rel, int(height), int(width)


def _legend_cells(g: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int]]]]:
    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.size, "expected a gray legend frame"
    top, left = map(int, gray_coords.min(axis=0))
    bottom, right = map(int, gray_coords.max(axis=0))
    legend = g[top : bottom + 1, left : right + 1]

    sep_rows = [r for r in range(legend.shape[0]) if np.all(legend[r] == GRAY)]
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == GRAY)]
    assert len(sep_rows) >= 3, "legend must contain multiple stacked cells"
    assert sep_cols == [0, legend.shape[1] - 1], "legend should use only outer vertical gray borders"

    row_ranges = _cell_ranges(sep_rows, legend.shape[0])
    inner_left = 1
    inner_right = legend.shape[1] - 2
    assert inner_left <= inner_right, "legend cells need interior columns"

    cells: list[dict[str, int | tuple[int, int]]] = []
    seen_colors: set[int] = set()
    for r0, r1 in row_ranges:
        cell = legend[r0 : r1 + 1, inner_left : inner_right + 1]
        dots = np.argwhere((cell != BLACK) & (cell != GRAY))
        assert len(dots) == 1, "each legend cell must contain exactly one colored dot"
        dr, dc = map(int, dots[0])
        color = int(cell[dr, dc])
        assert color not in seen_colors, "legend colors must be unique"
        seen_colors.add(color)
        cells.append(
            {
                "color": color,
                "anchor": (top + r0 + dr, left + inner_left + dc),
                "bounds": (top + r0, top + r1, left + inner_left, left + inner_right),
            }
        )

    return right + 1, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each color's scattered pixels into a normalized shape and redraw that shape in the matching legend slot.

    Input structure:
    - The background is black.
    - The left side contains a vertically stacked legend framed by gray separator lines, with one colored dot inside each cell.
    - The right side contains scattered pixels in those same colors, with each color appearing several times as one coded group.

    Rule:
    - Parse the gray-framed legend and record the unique colored anchor dot in each cell.
    - For every legend color, gather all matching pixels from the right-side coding field and normalize them by their bounding-box top-left corner to obtain that color's shape.
    - Erase the coding-field colored pixels, then redraw each recovered shape in gray inside the matching legend cell with the shape's top-left aligned to the original colored dot.

    Key insight:
    - The right-side pixels are not separate objects; all pixels of one color jointly encode a translation-normalized shape that must be transferred back into the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    code_start, cells = _legend_cells(g)
    assert code_start < g.shape[1], "grid must contain a coding region to the right of the legend"

    code = g[:, code_start:]
    legend_colors = {int(cell["color"]) for cell in cells}
    code_colors = {int(color) for color in np.unique(code) if int(color) != BLACK}
    assert code_colors == legend_colors, "coding region colors must match the legend colors"

    shapes: dict[int, tuple[tuple[tuple[int, int], ...], int, int, np.ndarray]] = {}
    for color in legend_colors:
        local = np.argwhere(code == color)
        assert 2 <= len(local) <= 5, "each coded color must use between 2 and 5 pixels"
        abs_coords = local.copy()
        abs_coords[:, 1] += code_start
        rel, height, width = _normalized_shape(abs_coords)
        shapes[color] = (rel, height, width, abs_coords)

    out = g.copy()
    for color in legend_colors:
        _rel, _height, _width, abs_coords = shapes[color]
        out[abs_coords[:, 0], abs_coords[:, 1]] = BLACK

    for cell in cells:
        color = int(cell["color"])
        rel, height, width, _abs_coords = shapes[color]
        anchor_r, anchor_c = cell["anchor"]  # type: ignore[misc]
        r0, r1, c0, c1 = cell["bounds"]  # type: ignore[misc]
        assert anchor_r + height - 1 <= r1 and anchor_c + width - 1 <= c1, "decoded shape must fit its legend cell"
        for dr, dc in rel:
            out[anchor_r + dr, anchor_c + dc] = GRAY

    return out


def _sample_shape(rng: np.random.Generator, cell_h: int, cell_w: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    max_h = min(cell_h, 4)
    max_w = min(cell_w, 4)

    for _ in range(200):
        height = int(rng.integers(1, max_h + 1))
        width = int(rng.integers(1, max_w + 1))
        area = height * width
        if area < 2:
            continue

        cover_anchor = bool(rng.integers(0, 2))
        if not cover_anchor and (height == 1 or width == 1):
            continue

        required: set[tuple[int, int]] = set()
        if cover_anchor:
            required.add((0, 0))
        else:
            required.add((0, int(rng.integers(1, width))))
            required.add((int(rng.integers(1, height)), 0))
        required.add((height - 1, int(rng.integers(0, width))))
        required.add((int(rng.integers(0, height)), width - 1))

        min_cells = max(2, len(required))
        max_cells = min(5, area)
        if min_cells > max_cells:
            continue

        target = int(rng.integers(min_cells, max_cells + 1))
        cells = set(required)
        remaining = [(r, c) for r in range(height) for c in range(width) if (r, c) not in cells]
        if target > len(cells):
            picked = rng.choice(len(remaining), size=target - len(cells), replace=False)
            for idx in np.atleast_1d(picked):
                cells.add(remaining[int(idx)])

        if not cover_anchor and (0, 0) in cells:
            continue

        rel, bbox_h, bbox_w = _normalized_shape(np.asarray(sorted(cells), dtype=int))
        if bbox_h == height and bbox_w == width:
            return rel, height, width

    raise RuntimeError("failed to sample a valid coded shape")


def _pick_offset(rng: np.random.Generator, room: int) -> int:
    assert room >= 0
    if room <= 1:
        return int(rng.integers(0, room + 1))
    return int(rng.integers(1, room))


def _place_shapes_in_code(
    rng: np.random.Generator,
    shapes: list[tuple[tuple[tuple[int, int], ...], int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []

    for rel, shape_h, shape_w in shapes:
        candidates = [
            (r, c)
            for r in range(height - shape_h + 1)
            for c in range(width - shape_w + 1)
        ]
        order = rng.permutation(len(candidates))
        chosen: tuple[int, int] | None = None
        for idx in order:
            top, left = candidates[int(idx)]
            abs_cells = [(top + dr, left + dc) for dr, dc in rel]
            if any(blocked[r, c] for r, c in abs_cells):
                continue
            chosen = (int(top), int(left))
            for r, c in abs_cells:
                r0 = max(0, r - 1)
                r1 = min(height, r + 2)
                c0 = max(0, c - 1)
                c1 = min(width, c + 2)
                blocked[r0:r1, c0:c1] = True
            break
        if chosen is None:
            return None
        placements.append(chosen)

    return placements


def _build_expected(
    grid: np.ndarray,
    code_start: int,
    cell_specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in cell_specs:
        color = int(spec["color"])
        abs_coords = np.asarray(spec["code_coords"], dtype=int)
        out[abs_coords[:, 0], abs_coords[:, 1]] = BLACK
        anchor_r, anchor_c = spec["anchor"]
        for dr, dc in spec["shape"]:
            out[anchor_r + dr, anchor_c + dc] = GRAY
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_colors = int(rng.integers(2, 6))
    max_cell_h = min(5, (30 - (n_colors + 1)) // n_colors)
    if max_cell_h < 4:
        return None
    cell_h = int(rng.integers(4, max_cell_h + 1))
    cell_w = int(rng.integers(4, 7))
    legend_w = cell_w + 2
    gap_w = int(rng.integers(2, 5))
    height = n_colors * cell_h + n_colors + 1

    colors = [int(c) for c in rng.choice(np.asarray(PALETTE), size=n_colors, replace=False)]
    shapes = [_sample_shape(rng, cell_h, cell_w) for _ in range(n_colors)]
    max_shape_w = max(shape_w for _rel, _shape_h, shape_w in shapes)
    min_code_w = max(8, max_shape_w + 3)
    max_code_w = 30 - legend_w - gap_w
    if min_code_w > max_code_w:
        return None
    code_w = int(rng.integers(min_code_w, max_code_w + 1))

    placements = _place_shapes_in_code(rng, shapes, height, code_w)
    if placements is None:
        return None

    grid = np.full((height, legend_w + gap_w + code_w), BLACK, dtype=int)
    grid[:, 0] = GRAY
    grid[:, legend_w - 1] = GRAY
    for row in range(0, height, cell_h + 1):
        grid[row, :legend_w] = GRAY
    grid[height - 1, :legend_w] = GRAY

    cell_specs: list[dict[str, object]] = []
    code_start = legend_w + gap_w
    for idx, color in enumerate(colors):
        top = 1 + idx * (cell_h + 1)
        rel, shape_h, shape_w = shapes[idx]
        anchor_r = top + _pick_offset(rng, cell_h - shape_h)
        anchor_c = 1 + _pick_offset(rng, cell_w - shape_w)
        grid[anchor_r, anchor_c] = color

        code_top, code_left = placements[idx]
        abs_cells = []
        for dr, dc in rel:
            rr = code_top + dr
            cc = code_start + code_left + dc
            grid[rr, cc] = color
            abs_cells.append((rr, cc))

        cell_specs.append(
            {
                "color": color,
                "anchor": (anchor_r, anchor_c),
                "shape": rel,
                "code_coords": abs_cells,
            }
        )

    expected = _build_expected(grid, code_start, cell_specs)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

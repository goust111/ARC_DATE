import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 300
COL_COUNT_CHOICES = (2, 3)
ROW_COUNT_BY_COLS = {2: 3, 3: 5}
CELL_HEIGHT_CHOICES = {3: (5, 7, 9), 5: (5,)}
CELL_WIDTH_CHOICES = {2: (5, 7, 9), 3: (5, 7, 9)}
SHAPE_FAMILIES = ("l", "t", "u", "plus", "stair", "frame", "notch")


def _separator_color(grid: np.ndarray) -> int:
    colors = []
    for color in np.unique(grid):
        if np.any(np.all(grid == color, axis=1)) and np.any(np.all(grid == color, axis=0)):
            colors.append(int(color))
    assert len(colors) == 1, "expected a unique separator color"
    return colors[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in separators) + [length]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _background_color(grid: np.ndarray, separator: int) -> int:
    colors, counts = np.unique(grid[grid != separator], return_counts=True)
    return int(colors[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _center_paste(canvas: np.ndarray, mask: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    mh, mw = mask.shape
    top = (h - mh) // 2
    left = (w - mw) // 2
    canvas[top : top + mh, left : left + mw][mask] = int(color)


def _partition_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    separator = _separator_color(grid)
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == separator)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == separator)]
    row_spans = _spans(grid.shape[0], sep_rows)
    col_spans = _spans(grid.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one partition row and column"
    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "partitions must be equal-sized"
    background = _background_color(grid, separator)
    return separator, background, row_spans, col_spans


def _collect_objects(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], np.ndarray], list[tuple[int, int, int]]]:
    separator, background, row_spans, col_spans = _partition_structure(grid)
    cell_h = row_spans[0][1] - row_spans[0][0] + 1
    cell_w = col_spans[0][1] - col_spans[0][0] + 1
    center = np.array([cell_h // 2, cell_w // 2])

    templates: dict[tuple[int, int], np.ndarray] = {}
    anchors: list[tuple[int, int, int]] = []
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(tile == separator), "separator color may only appear on divider lines"
            colors = [int(color) for color in np.unique(tile) if int(color) != background]
            assert len(colors) <= 1, "each partition must contain at most one foreground color"
            if not colors:
                continue
            color = colors[0]
            mask = tile == color
            coords = np.argwhere(mask)
            if len(coords) == 1 and np.array_equal(coords[0], center):
                anchors.append((pr, pc, color))
            else:
                templates[(pr, pc)] = _crop_mask(mask)

    return separator, background, row_spans, col_spans, templates, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each matched template shape into anchor partitions and recolor it by the anchor pixel.

    Input structure:
    - The grid has a black-like background and full single-pixel separator lines that split it into equal rectangular partitions.
    - Some partitions contain a single-color template shape; other partitions are empty except for one centered anchor pixel in a different color.
    - Template shapes are identified only by their partition row and column, not by their original offset inside the partition.

    Rule:
    - Detect the partition row and column of every centered anchor pixel.
    - If that partition's row contains a template partition, use that unique row template; otherwise use the unique template in the same column.
    - Copy the matched template's shape mask, crop it to its occupied bbox, center it in the anchor partition, recolor it to the anchor color, and erase the original anchor pixel.

    Key insight:
    - The anchor color tells you which color to paint, while the anchor partition's row or column alignment tells you which template shape to copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "grid must be a non-empty 2D array"

    separator, background, row_spans, col_spans, templates, anchors = _collect_objects(g)
    assert templates and anchors, "expected both template and anchor partitions"

    row_templates: dict[int, np.ndarray] = {}
    col_templates: dict[int, np.ndarray] = {}
    for (pr, pc), mask in templates.items():
        assert pr not in row_templates, "each template row must contain at most one template"
        assert pc not in col_templates, "each template column must contain at most one template"
        row_templates[pr] = mask
        col_templates[pc] = mask

    out = g.copy()
    for pr, pc, color in anchors:
        tile_r0, tile_r1 = row_spans[pr]
        tile_c0, tile_c1 = col_spans[pc]
        target = out[tile_r0 : tile_r1 + 1, tile_c0 : tile_c1 + 1]
        target[:] = background
        if pr in row_templates:
            shape = row_templates[pr]
        else:
            assert pc in col_templates, "anchor must match a template by row or column"
            shape = col_templates[pc]
        _center_paste(target, shape, color)

    return out


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _make_shape(max_h: int, max_w: int, used: set[tuple[tuple[int, int], bytes]], rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        family = SHAPE_FAMILIES[int(rng.integers(len(SHAPE_FAMILIES)))]
        mask = None

        if family == "plus":
            heights = [h for h in range(3, max_h + 1) if h % 2 == 1]
            widths = [w for w in range(3, max_w + 1) if w % 2 == 1]
            if not heights or not widths:
                continue
            h = int(rng.choice(heights))
            w = int(rng.choice(widths))
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 2, :] = True
            mask[:, w // 2] = True
        else:
            if max_h < 3 or max_w < 3:
                continue
            h = int(rng.integers(3, max_h + 1))
            w = int(rng.integers(3, max_w + 1))
            mask = np.zeros((h, w), dtype=bool)
            if family == "l":
                mask[0, :] = True
                mask[:, 0] = True
            elif family == "t":
                mask[0, :] = True
                mask[:, w // 2] = True
            elif family == "u":
                mask[:, 0] = True
                mask[:, -1] = True
                mask[-1, :] = True
            elif family == "stair":
                r = c = 0
                mask[r, c] = True
                while r < h - 1 or c < w - 1:
                    if c < w - 1:
                        c += 1
                        mask[r, c] = True
                    if r < h - 1:
                        r += 1
                        mask[r, c] = True
            elif family == "frame":
                mask[:] = True
                mask[1:-1, 1:-1] = False
            elif family == "notch":
                mask[:] = True
                cut_h = int(rng.integers(1, min(3, h - 1) + 1))
                cut_w = int(rng.integers(1, min(3, w - 1) + 1))
                corner = int(rng.integers(4))
                if corner == 0:
                    mask[:cut_h, :cut_w] = False
                elif corner == 1:
                    mask[:cut_h, w - cut_w :] = False
                elif corner == 2:
                    mask[h - cut_h :, :cut_w] = False
                else:
                    mask[h - cut_h :, w - cut_w :] = False
            else:
                raise AssertionError("unknown shape family")

        mask = _crop_mask(_random_transform(mask, rng))
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if mask.sum() < 3:
            continue
        sig = _shape_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        return mask

    raise RuntimeError("failed to build a distinct shape")


def _non_center_position(cell_h: int, cell_w: int, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    mh, mw = mask.shape
    centered = ((cell_h - mh) // 2, (cell_w - mw) // 2)
    choices = [(r, c) for r in range(cell_h - mh + 1) for c in range(cell_w - mw + 1)]
    off_center = [pos for pos in choices if pos != centered]
    if off_center:
        return off_center[int(rng.integers(len(off_center)))]
    return centered


def _tile_bounds(row: int, col: int, cell_h: int, cell_w: int) -> tuple[slice, slice]:
    r0 = row * (cell_h + 1)
    c0 = col * (cell_w + 1)
    return slice(r0, r0 + cell_h), slice(c0, c0 + cell_w)


def _build_input(rng: np.random.Generator) -> np.ndarray:
    cols = int(rng.choice(COL_COUNT_CHOICES))
    rows = ROW_COUNT_BY_COLS[cols]
    cell_h = int(rng.choice(CELL_HEIGHT_CHOICES[rows]))
    cell_w = int(rng.choice(CELL_WIDTH_CHOICES[cols]))

    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    assert max(height, width) <= 30

    grid = np.full((height, width), BG, dtype=int)
    for r in range(1, rows):
        grid[r * cell_h + (r - 1), :] = SEP
    for c in range(1, cols):
        grid[:, c * cell_w + (c - 1)] = SEP

    template_colors = list(rng.choice(COLOR_POOL, size=cols, replace=False))
    anchor_candidates = [color for color in COLOR_POOL if color not in template_colors]
    anchor_palette_size = int(rng.integers(3, min(len(anchor_candidates), 5) + 1))
    anchor_palette = list(rng.choice(anchor_candidates, size=anchor_palette_size, replace=False))

    template_rows = list(rng.choice(rows, size=cols, replace=False))
    column_order = list(rng.permutation(cols))
    template_at: dict[tuple[int, int], int] = {}
    row_to_template_col: dict[int, int] = {}
    col_to_template_row: dict[int, int] = {}
    for row, col, color in zip(template_rows, column_order, template_colors):
        template_at[(int(row), int(col))] = int(color)
        row_to_template_col[int(row)] = int(col)
        col_to_template_row[int(col)] = int(row)

    max_shape_h = min(cell_h - 1, 5)
    max_shape_w = min(cell_w - 1, 5)
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    shape_by_template: dict[tuple[int, int], np.ndarray] = {}
    for key in template_at:
        shape_by_template[key] = _make_shape(max_shape_h, max_shape_w, used_shapes, rng)

    for (row, col), color in template_at.items():
        rs, cs = _tile_bounds(row, col, cell_h, cell_w)
        tile = grid[rs, cs]
        mask = shape_by_template[(row, col)]
        top, left = _non_center_position(cell_h, cell_w, mask, rng)
        tile[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color

    center_r = cell_h // 2
    center_c = cell_w // 2
    for row in range(rows):
        for col in range(cols):
            if (row, col) in template_at:
                continue
            rs, cs = _tile_bounds(row, col, cell_h, cell_w)
            grid[rs.start + center_r, cs.start + center_c] = int(anchor_palette[int(rng.integers(len(anchor_palette)))])

    separator, background, row_spans, col_spans, templates, anchors = _collect_objects(grid)
    assert separator == SEP and background == BG
    assert len(row_spans) == rows and len(col_spans) == cols
    assert len(templates) == cols
    assert len(anchors) == rows * cols - cols
    assert len({pr for pr, _pc in templates}) == cols
    assert len({pc for _pr, pc in templates}) == cols

    row_branch = 0
    col_branch = 0
    for pr, pc, _color in anchors:
        if pr in row_to_template_col:
            assert len([1 for tr, _tc in templates if tr == pr]) == 1
            row_branch += 1
        else:
            assert pc in col_to_template_row
            assert len([1 for _tr, tc in templates if tc == pc]) == 1
            col_branch += 1
    assert row_branch == col_branch and row_branch > 0
    assert np.any(main(grid) != grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _build_input(np.random.default_rng())
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")

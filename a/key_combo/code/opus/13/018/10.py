import numpy as np


BG = 0
LEGEND_BORDER = 5
SOURCE_COLORS = (1, 2, 3, 4, 7)
TARGET_COLORS = (6, 8, 9)
GRID_SIZE_RANGE = (18, 24)
RECT_COUNT_RANGE = (3, 5)
RECT_SIZE_RANGE = (4, 7)
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []
    for comp in _components(grid == LEGEND_BORDER):
        r0, c0, r1, c1 = _bbox(comp)
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = box.shape
        if h < 4 or w != 4:
            continue
        if not np.all(box[0, :] == LEGEND_BORDER):
            continue
        if not np.all(box[-1, :] == LEGEND_BORDER):
            continue
        if not np.all(box[:, 0] == LEGEND_BORDER):
            continue
        if not np.all(box[:, -1] == LEGEND_BORDER):
            continue
        if np.any(box[1:-1, 1:-1] == LEGEND_BORDER):
            continue

        sources = box[1:-1, 1]
        targets = box[1:-1, 2]
        if len(sources) < 2:
            continue
        if np.any(sources == BG) or np.any(targets == BG):
            continue
        mapping = {int(src): int(dst) for src, dst in zip(sources, targets)}
        if len(mapping) != len(sources):
            continue
        candidates.append(((r0, c0, r1, c1), mapping))

    assert len(candidates) == 1, "expected exactly one grey legend box"
    return candidates[0]


def _paint_inner_border(out: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    if r1 - r0 + 1 <= 2 or c1 - c0 + 1 <= 2:
        return
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return
    inner[0, :] = color
    inner[-1, :] = color
    inner[:, 0] = color
    inner[:, -1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner legend and paint selective inner borders inside matching filled rectangles.

    Input structure:
    - Black background surrounds several non-touching solid rectangles in source colors.
    - A single grey rectangular frame in one corner encloses legend rows with two colored cells per row.
    - Each legend row stores a source color on the left and an inner-border color on the right.

    Rule:
    - Find the grey-framed legend and read its row-wise source->target color mapping.
    - For every solid rectangular component whose fill color appears in the mapping, repaint the pixels exactly one step inside its outer edge with the mapped target color.
    - Leave rectangles whose colors are absent from the legend unchanged, then erase the entire legend box to black.

    Key insight:
    - The grey box is disposable metadata: it specifies which rectangle colors should receive an inner border and which color that border should use, then disappears.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_bbox, mapping = _find_legend(g)
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    for comp in _components(out != BG):
        rect_color = int(out[comp[0, 0], comp[0, 1]])
        if rect_color not in mapping:
            continue
        if np.any(out[comp[:, 0], comp[:, 1]] != rect_color):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        block = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block == rect_color):
            continue
        _paint_inner_border(out, (r0, c0, r1, c1), mapping[rect_color])

    return out


def _overlaps(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int], padding: int = 1) -> bool:
    r0, c0, r1, c1 = box1
    a0, b0, a1, b1 = box2
    return not (r1 + padding < a0 or a1 + padding < r0 or c1 + padding < b0 or b1 + padding < c0)


def _legend_box(h: int, w: int, rows: int, corner: int) -> tuple[int, int, int, int]:
    lh, lw = rows + 2, 4
    if corner == 0:
        return 0, 0, lh - 1, lw - 1
    if corner == 1:
        return 0, w - lw, lh - 1, w - 1
    if corner == 2:
        return h - lh, 0, h - 1, lw - 1
    return h - lh, w - lw, h - 1, w - 1


def _place_rectangles(
    h: int,
    w: int,
    sizes: list[tuple[int, int]],
    forbidden: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    order = np.argsort([-rh * rw for rh, rw in sizes])
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    occupied = [forbidden]
    for idx in order:
        rh, rw = sizes[int(idx)]
        found = None
        for _ in range(120):
            r0 = int(np.random.randint(0, h - rh + 1))
            c0 = int(np.random.randint(0, w - rw + 1))
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_overlaps(box, other, padding=1) for other in occupied):
                continue
            found = box
            break
        if found is None:
            return None
        placed[int(idx)] = found
        occupied.append(found)
    return [box for box in placed if box is not None]


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _draw_legend(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    rows: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = LEGEND_BORDER
    grid[r1, c0 : c1 + 1] = LEGEND_BORDER
    grid[r0 : r1 + 1, c0] = LEGEND_BORDER
    grid[r0 : r1 + 1, c1] = LEGEND_BORDER
    for i, (src, dst) in enumerate(rows):
        grid[r0 + 1 + i, c0 + 1] = src
        grid[r0 + 1 + i, c0 + 2] = dst


def _valid_instance(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    mapped: dict[int, int],
    legend: tuple[int, int, int, int],
) -> bool:
    out = main(grid)
    lr0, lc0, lr1, lc1 = legend
    if not np.all(out[lr0 : lr1 + 1, lc0 : lc1 + 1] == BG):
        return False

    saw_mapped = False
    saw_unmapped = False
    for box, color in zip(rects, colors):
        r0, c0, r1, c1 = box
        inp_block = grid[r0 : r1 + 1, c0 : c1 + 1]
        out_block = out[r0 : r1 + 1, c0 : c1 + 1]
        if color in mapped:
            saw_mapped = True
            if np.array_equal(inp_block, out_block):
                return False
            expected = np.full_like(inp_block, color)
            _paint_inner_border(expected, (0, 0, expected.shape[0] - 1, expected.shape[1] - 1), mapped[color])
            if not np.array_equal(out_block, expected):
                return False
        else:
            saw_unmapped = True
            if not np.array_equal(inp_block, out_block):
                return False
    return saw_mapped and saw_unmapped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = int(np.random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))
        w = int(np.random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))
        rect_count = int(np.random.randint(RECT_COUNT_RANGE[0], RECT_COUNT_RANGE[1] + 1))
        rect_colors = list(np.random.choice(SOURCE_COLORS, size=rect_count, replace=False))

        max_rows = min(3, rect_count - 1)
        mapped_count = int(np.random.randint(2, max_rows + 1))
        mapped_sources = list(np.random.choice(rect_colors, size=mapped_count, replace=False))
        mapped_targets = list(np.random.choice(TARGET_COLORS, size=mapped_count, replace=False))
        mapping = {int(src): int(dst) for src, dst in zip(mapped_sources, mapped_targets)}

        corner = int(np.random.randint(0, 4))
        legend_box = _legend_box(h, w, mapped_count, corner)
        sizes = [
            (
                int(np.random.randint(RECT_SIZE_RANGE[0], RECT_SIZE_RANGE[1] + 1)),
                int(np.random.randint(RECT_SIZE_RANGE[0], RECT_SIZE_RANGE[1] + 1)),
            )
            for _ in range(rect_count)
        ]

        rects = _place_rectangles(h, w, sizes, legend_box)
        if rects is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for box, color in zip(rects, rect_colors):
            _draw_rect(grid, box, color)

        legend_rows = [(src, mapping[src]) for src in mapped_sources]
        np.random.shuffle(legend_rows)
        _draw_legend(grid, legend_box, legend_rows)

        if _valid_instance(grid, rects, rect_colors, mapping, legend_box):
            return grid

    raise RuntimeError("failed to generate a valid selective-inner-border legend puzzle")

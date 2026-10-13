import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
SOURCE_COLORS = (1, 2, 3, 6, 7, 8)

MIN_SHAPES = 2
MAX_SHAPES = 4
MAX_GENERATE_ATTEMPTS = 400
SHAPE_RETRIES = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, height: int, width: int, connectivity: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    if connectivity == 8:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < height and 0 <= cc < width:
                out.append((rr, cc))
    return out


def _find_monochrome_components(
    grid: np.ndarray,
    mask: np.ndarray,
    connectivity: int,
) -> list[dict[str, object]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if visited[r, c] or not bool(mask[r, c]):
                continue

            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors(cr, cc, height, width, connectivity):
                    if visited[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _shape_center_sums(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + r1), int(c0 + c1)


def _arrow_tip_col(comp: dict[str, object]) -> int:
    coords = np.asarray(comp["coords"], dtype=int)
    counts: dict[int, int] = {}
    for _r, c in coords:
        counts[int(c)] = counts.get(int(c), 0) + 1
    tip_cols = [col for col, count in counts.items() if count == 1]
    assert len(tip_cols) == 1, "arrow must have a unique tip column"
    return int(tip_cols[0])


def _is_arrow_component(comp: dict[str, object]) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    if coords.shape[0] != 3:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if r0 != 0 or r1 != 2 or c1 - c0 != 1:
        return False
    norm = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return norm in ([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 0), (2, 1)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    height, width = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, width - 1):
        value = int(grid[0, c])
        if value == bg:
            continue
        if not np.all(grid[:, c] == value):
            continue
        if not np.all(grid[:, c + 1 :] == bg):
            continue
        candidates.append((int(c), value))
    assert len(candidates) == 1, "expected one full-height separator with an empty canvas on its right"
    return candidates[0]


def _project_shapes(
    grid: np.ndarray,
    shapes: list[dict[str, object]],
    separator_col: int,
    bg: int,
) -> np.ndarray:
    height, width = grid.shape
    canvas_left = int(separator_col + 1)
    canvas_right = int(width - 1)
    canvas_row_sum = int(height - 1)
    canvas_col_sum = int(canvas_left + canvas_right)

    out = np.array(grid, dtype=int, copy=True)
    out[:, canvas_left:] = bg
    cover_count = np.zeros((height, width), dtype=int)
    first_color = np.full((height, width), bg, dtype=int)

    for shape in shapes:
        color = int(shape["color"])
        bbox = shape["bbox"]
        row_sum, col_sum = _shape_center_sums(bbox)
        dr2 = int(canvas_row_sum - row_sum)
        dc2 = int(canvas_col_sum - col_sum)
        assert dr2 % 2 == 0 and dc2 % 2 == 0, "shape centers must align to the canvas center by integer translation"
        dr = int(dr2 // 2)
        dc = int(dc2 // 2)

        for r, c in np.asarray(shape["coords"], dtype=int):
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < height and canvas_left <= cc <= canvas_right, "projected shape must fit inside the canvas"
            if cover_count[rr, cc] == 0:
                first_color[rr, cc] = color
            cover_count[rr, cc] += 1

    single = cover_count == 1
    overlap = cover_count >= 2
    out[single] = first_color[single]
    out[overlap] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the left-side congruent source shapes onto the blank right-side canvas, coloring multi-shape overlaps yellow.

    Input structure:
    - The background is black.
    - A single full-height grey column separates a populated source region on the left from an initially empty canvas on the right.
    - The source region contains 2-4 monochrome shapes that are congruent up to rigid motion and use different colors.
    - Along the top edge, each source shape has a 3-pixel maroon arrow marker aligned with it horizontally and pointing toward the canvas.

    Rule:
    - Keep the source region, the separator, and the arrow markers unchanged.
    - Translate every source shape so that its bounding-box center coincides with the center of the canvas region.
    - On the canvas, keep black where no translated shape lands, keep the original shape color where exactly one translated shape lands, and paint yellow where two or more translated shapes overlap.

    Key insight:
    - The arrows indicate that the left-side shapes are all projected to one shared canvas anchor, so the output depends on overlap counts after centering rather than on the shapes' original relative positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    separator_col, _separator_color = _find_separator(g, bg)

    left_mask = np.zeros_like(g, dtype=bool)
    left_mask[:, :separator_col] = g[:, :separator_col] != bg
    arrow_components = [
        comp
        for comp in _find_monochrome_components(g, left_mask, connectivity=8)
        if _is_arrow_component(comp)
    ]
    assert MIN_SHAPES <= len(arrow_components) <= MAX_SHAPES, "expected 2-4 top-edge arrow markers"

    arrow_mask = np.zeros_like(g, dtype=bool)
    for comp in arrow_components:
        coords = np.asarray(comp["coords"], dtype=int)
        arrow_mask[coords[:, 0], coords[:, 1]] = True

    shape_mask = left_mask & ~arrow_mask
    shapes = _find_monochrome_components(g, shape_mask, connectivity=4)
    assert len(shapes) == len(arrow_components), "expected one source shape per arrow"

    ordered_arrows = sorted(arrow_components, key=_arrow_tip_col)
    ordered_shapes = sorted(shapes, key=lambda comp: _shape_center_sums(comp["bbox"])[1])
    for arrow, shape in zip(ordered_arrows, ordered_shapes):
        tip_col = _arrow_tip_col(arrow)
        _row_sum, shape_col_sum = _shape_center_sums(shape["bbox"])
        assert abs(2 * tip_col - shape_col_sum) <= 2, "arrow must align horizontally with its shape"

    return _project_shapes(g, ordered_shapes, separator_col, bg)


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    height, width = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, height, width, connectivity=4):
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((int(nr), int(nc)))
            stack.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = variant.astype(np.uint8, copy=False).tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(np.array(variant, dtype=bool, copy=True))
    return variants


def _random_shape(size: int) -> list[np.ndarray] | None:
    max_area = min(size * size - 2, size * size // 2 + size)
    min_area = size + 1

    for _ in range(SHAPE_RETRIES):
        area = random.randint(min_area, max_area)
        cells = {(random.randrange(size), random.randrange(size))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors(r, c, size, size, connectivity=4):
                    if (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        variants = _unique_transforms(mask)
        if len(variants) < 2:
            continue
        return variants
    return None


def _choose_variants(variants: list[np.ndarray], count: int) -> list[np.ndarray]:
    if len(variants) >= count:
        return [np.array(v, dtype=bool, copy=True) for v in random.sample(variants, count)]

    picks = [np.array(v, dtype=bool, copy=True) for v in random.sample(variants, 2)]
    while len(picks) < count:
        picks.append(np.array(random.choice(variants), dtype=bool, copy=True))
    random.shuffle(picks)
    return picks


def _split_extra(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _pick_count_and_size() -> tuple[int, int]:
    count = random.randint(MIN_SHAPES, MAX_SHAPES)
    if count == 4:
        return count, 3
    return count, random.choice((3, 5))


def _paint_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    height, width = mask.shape
    patch = grid[top : top + height, left : left + width]
    patch[mask] = int(color)


def _build_input() -> np.ndarray | None:
    shape_count, size = _pick_count_and_size()
    variants = _random_shape(size)
    if variants is None:
        return None

    chosen_masks = _choose_variants(variants, shape_count)
    overlay_counts = sum(mask.astype(int) for mask in chosen_masks)
    if np.count_nonzero(overlay_counts == 1) == 0 or np.count_nonzero(overlay_counts >= 2) == 0:
        return None

    min_source_width = shape_count * size + (shape_count + 1)
    max_canvas_width = min(15, 30 - 1 - min_source_width)
    canvas_width_choices = [w for w in range(size + 2, max_canvas_width + 1) if w % 2 == size % 2]
    if not canvas_width_choices:
        return None
    canvas_width = random.choice(canvas_width_choices)

    extra_source_budget = min(7, 30 - (min_source_width + 1 + canvas_width))
    if extra_source_budget < 0:
        return None
    source_width = min_source_width + random.randint(0, extra_source_budget)
    total_width = source_width + 1 + canvas_width
    assert total_width <= 30

    height_choices = [h for h in range(size + 4, 21) if h % 2 == size % 2]
    if not height_choices:
        return None
    height = random.choice(height_choices)

    row_lo = 4
    row_hi = height - size
    if row_hi < row_lo:
        return None

    extra_gaps = source_width - min_source_width
    gaps = [1 + x for x in _split_extra(extra_gaps, shape_count + 1)]
    cols: list[int] = []
    col = gaps[0]
    for index in range(shape_count):
        cols.append(int(col))
        col += size + gaps[index + 1]

    rows = [random.randint(row_lo, row_hi) for _ in range(shape_count)]
    colors = random.sample(SOURCE_COLORS, shape_count)

    grid = np.full((height, total_width), BLACK, dtype=int)
    separator_col = source_width
    grid[:, separator_col] = GREY

    order = sorted(range(shape_count), key=lambda i: cols[i] + size // 2)
    for idx in order:
        top = rows[idx]
        left = cols[idx]
        mask = chosen_masks[idx]
        color = colors[idx]
        _paint_shape(grid, top, left, mask, color)

        arrow_left = left + size // 2 - 1
        grid[0, arrow_left] = MAROON
        grid[1, arrow_left + 1] = MAROON
        grid[2, arrow_left] = MAROON

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        canvas = out[:, np.where(grid[0] == GREY)[0][0] + 1 :]
        if np.count_nonzero(canvas == YELLOW) == 0:
            continue
        non_bg_non_yellow = np.count_nonzero((canvas != BLACK) & (canvas != YELLOW))
        if non_bg_non_yellow == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")

import numpy as np


BACKGROUND = 0
LINE_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RATIO_CLASSES = ("wide", "tall", "square")
GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_from_mask(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rule_table(grid: np.ndarray, bg: int):
    h, w = grid.shape
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_from_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r0 > h // 3 or c1 < (2 * w) // 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            color_mask = patch == color
            row_lines = np.flatnonzero(np.all(color_mask, axis=1))
            col_lines = np.flatnonzero(np.all(color_mask, axis=0))
            if len(row_lines) < 4 or len(col_lines) < 5:
                continue
            if row_lines[0] != 0 or row_lines[-1] != patch.shape[0] - 1:
                continue
            if col_lines[0] != 0 or col_lines[-1] != patch.shape[1] - 1:
                continue
            if np.min(np.diff(row_lines)) < 2 or np.min(np.diff(col_lines)) < 2:
                continue
            score = (
                (len(row_lines) - 1) * (len(col_lines) - 1),
                patch.shape[0] * patch.shape[1],
                c1,
                -r0,
            )
            if best is None or score > best["score"]:
                best = {
                    "bbox": (r0, c0, r1, c1),
                    "line_color": color,
                    "row_lines": row_lines,
                    "col_lines": col_lines,
                    "score": score,
                }
    if best is None:
        raise AssertionError("rule table not found")
    return best


def _cell_bounds(table, row_idx: int, col_idx: int):
    r0, c0, _, _ = table["bbox"]
    row_lines = table["row_lines"]
    col_lines = table["col_lines"]
    rr0 = r0 + int(row_lines[row_idx]) + 1
    rr1 = r0 + int(row_lines[row_idx + 1]) - 1
    cc0 = c0 + int(col_lines[col_idx]) + 1
    cc1 = c0 + int(col_lines[col_idx + 1]) - 1
    return rr0, cc0, rr1, cc1


def _extract_mask(grid: np.ndarray, bounds, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def _classify_ratio_icon(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if h == 1 and w >= 2:
        return "wide"
    if w == 1 and h >= 2:
        return "tall"
    if h >= 2 and w >= 2 and np.all(sub):
        return "square"
    return None


def _classify_hole_icon(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if h >= 3 and w >= 3:
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border]) and np.any(~sub[1:-1, 1:-1]):
            return True
    if h >= 2 and w >= 2 and np.all(sub):
        return False
    return None


def _cell_color(grid: np.ndarray, bounds, line_color: int, bg: int) -> int:
    r0, c0, r1, c1 = bounds
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    values, counts = np.unique(cell[cell != line_color], return_counts=True)
    if len(values) == 0:
        return bg
    return int(values[np.argmax(counts)])


def _read_lookup(grid: np.ndarray, table, bg: int):
    row_count = len(table["row_lines"]) - 1
    col_count = len(table["col_lines"]) - 1
    line_color = table["line_color"]

    ratio_cols = {}
    for col_idx in range(1, col_count):
        icon = _classify_ratio_icon(_extract_mask(grid, _cell_bounds(table, 0, col_idx), line_color))
        if icon is not None:
            ratio_cols[icon] = col_idx
    if set(ratio_cols) != set(RATIO_CLASSES):
        raise AssertionError("missing ratio headers")

    hole_rows = {}
    for row_idx in range(1, row_count):
        icon = _classify_hole_icon(_extract_mask(grid, _cell_bounds(table, row_idx, 0), line_color))
        if icon is not None:
            hole_rows[icon] = row_idx
    if set(hole_rows) != {True, False}:
        raise AssertionError("missing hole headers")

    lookup = {}
    for has_hole, row_idx in hole_rows.items():
        for ratio, col_idx in ratio_cols.items():
            lookup[(has_hole, ratio)] = _cell_color(grid, _cell_bounds(table, row_idx, col_idx), line_color, bg)
    return lookup


def _ratio_from_bbox(coords) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if width > height:
        return "wide"
    if height > width:
        return "tall"
    return "square"


def _has_hole(coords) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(~mask & ~outside))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by looking up a rule-table entry selected by
    its bbox aspect category and whether it contains enclosed background holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    table = _find_rule_table(g, bg)
    lookup = _read_lookup(g, table, bg)

    out = g.copy()
    blocked = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = table["bbox"]
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    for color in np.unique(g):
        color = int(color)
        if color == bg or color == table["line_color"]:
            continue
        mask = (g == color) & ~blocked
        for comp in _components_from_mask(mask):
            ratio = _ratio_from_bbox(comp)
            has_hole = _has_hole(comp)
            new_color = lookup[(has_hole, ratio)]
            for r, c in comp:
                out[r, c] = new_color
    return out


def _make_solid_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    intervals = []
    pivot = int(rng.integers(0, width))
    left = int(rng.integers(0, pivot + 1))
    right = int(rng.integers(pivot, width))
    intervals.append([left, right])
    for _ in range(1, height):
        prev_left, prev_right = intervals[-1]
        pivot = int(rng.integers(prev_left, prev_right + 1))
        left = int(rng.integers(0, pivot + 1))
        right = int(rng.integers(pivot, width))
        intervals.append([left, right])
    intervals[int(rng.integers(0, height))][0] = 0
    intervals[int(rng.integers(0, height))][1] = width - 1

    mask = np.zeros((height, width), dtype=bool)
    for r, (left, right) in enumerate(intervals):
        mask[r, left : right + 1] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    holes = (~mask) & (~outside)
    return len(_components_from_mask(holes))


def _make_holey_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if height >= 5 and width >= 4 and rng.random() < 0.5:
        col = int(rng.integers(1, width - 1))
        max_len = height - 3
        if max_len >= 1:
            length = int(rng.integers(1, max_len + 1))
            mask[1 : 1 + length, col] = True
    if width >= 5 and height >= 4 and rng.random() < 0.5:
        row = int(rng.integers(1, height - 1))
        max_len = width - 3
        if max_len >= 1:
            length = int(rng.integers(1, max_len + 1))
            mask[row, 1 : 1 + length] = True

    if _hole_count(mask) == 0:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    return mask


def _shape_for_class(has_hole: bool, ratio: str, rng: np.random.Generator) -> np.ndarray:
    if ratio == "wide":
        height = int(rng.integers(3, 7))
        width = int(rng.integers(height + 1, min(height + 5, 10)))
    elif ratio == "tall":
        width = int(rng.integers(3, 7))
        height = int(rng.integers(width + 1, min(width + 5, 10)))
    else:
        side = int(rng.integers(3, 8))
        height = width = side

    mask = _make_holey_mask(height, width, rng) if has_hole else _make_solid_mask(height, width, rng)
    coords = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
    if _ratio_from_bbox(coords) != ratio:
        raise AssertionError("shape ratio mismatch")
    if _has_hole(coords) != has_hole:
        raise AssertionError("shape hole mismatch")
    return mask


def _draw_ratio_icon(grid: np.ndarray, bounds, ratio: str):
    r0, c0, r1, c1 = bounds
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if ratio == "wide":
        length = min(3, w)
        rr = r0 + h // 2
        cc = c0 + (w - length) // 2
        grid[rr, cc : cc + length] = LINE_COLOR
    elif ratio == "tall":
        length = min(3, h)
        rr = r0 + (h - length) // 2
        cc = c0 + w // 2
        grid[rr : rr + length, cc] = LINE_COLOR
    else:
        size = 2 if min(h, w) >= 4 else min(h, w)
        rr = r0 + (h - size) // 2
        cc = c0 + (w - size) // 2
        grid[rr : rr + size, cc : cc + size] = LINE_COLOR


def _draw_hole_icon(grid: np.ndarray, bounds, has_hole: bool):
    r0, c0, r1, c1 = bounds
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if has_hole:
        rr = r0 + (h - 3) // 2
        cc = c0 + (w - 3) // 2
        grid[rr, cc : cc + 3] = LINE_COLOR
        grid[rr + 2, cc : cc + 3] = LINE_COLOR
        grid[rr : rr + 3, cc] = LINE_COLOR
        grid[rr : rr + 3, cc + 2] = LINE_COLOR
    else:
        size = 2 if min(h, w) >= 4 else min(h, w)
        rr = r0 + (h - size) // 2
        cc = c0 + (w - size) // 2
        grid[rr : rr + size, cc : cc + size] = LINE_COLOR


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int):
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _assign_object_colors(classes, lookup, rng: np.random.Generator):
    for _ in range(50):
        pool = list(rng.permutation(PALETTE))
        chosen = []
        ok = True
        for has_hole, ratio in classes:
            target = lookup[(has_hole, ratio)]
            candidates = [color for color in pool if color != target]
            if not candidates:
                ok = False
                break
            color = int(candidates[0])
            pool.remove(color)
            chosen.append(color)
        if ok:
            return chosen
    raise RuntimeError("failed to derange object colors")


def _gen_attempt(rng: np.random.Generator):
    cell_size = int(rng.integers(4, 6))
    table_rows, table_cols = 3, 4
    table_h = table_rows * cell_size + (table_rows + 1)
    table_w = table_cols * cell_size + (table_cols + 1)

    grid_h = int(rng.integers(max(28, table_h + 8), 37))
    grid_w = int(rng.integers(max(34, table_w + 10), 43))
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    table_r0 = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    table_c0 = grid_w - table_w - right_margin
    table_r1 = table_r0 + table_h - 1
    table_c1 = table_c0 + table_w - 1

    row_order = [bool(x) for x in rng.permutation([0, 1])]
    col_order = [RATIO_CLASSES[idx] for idx in rng.permutation(len(RATIO_CLASSES))]
    target_colors = [int(x) for x in rng.choice(PALETTE, size=6, replace=False)]

    lookup = {}
    idx = 0
    for has_hole in row_order:
        for ratio in col_order:
            lookup[(has_hole, ratio)] = target_colors[idx]
            idx += 1

    row_lines = [table_r0 + i * (cell_size + 1) for i in range(table_rows + 1)]
    col_lines = [table_c0 + i * (cell_size + 1) for i in range(table_cols + 1)]
    for r in row_lines:
        grid[r, table_c0 : table_c1 + 1] = LINE_COLOR
    for c in col_lines:
        grid[table_r0 : table_r1 + 1, c] = LINE_COLOR

    table = {
        "bbox": (table_r0, table_c0, table_r1, table_c1),
        "row_lines": np.array([r - table_r0 for r in row_lines], dtype=int),
        "col_lines": np.array([c - table_c0 for c in col_lines], dtype=int),
    }

    for col_idx, ratio in enumerate(col_order, start=1):
        _draw_ratio_icon(grid, _cell_bounds(table, 0, col_idx), ratio)
    for row_idx, has_hole in enumerate(row_order, start=1):
        _draw_hole_icon(grid, _cell_bounds(table, row_idx, 0), has_hole)
    for row_idx, has_hole in enumerate(row_order, start=1):
        for col_idx, ratio in enumerate(col_order, start=1):
            rr0, cc0, rr1, cc1 = _cell_bounds(table, row_idx, col_idx)
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = lookup[(has_hole, ratio)]

    classes = [(has_hole, ratio) for has_hole in (True, False) for ratio in RATIO_CLASSES]
    rng.shuffle(classes)
    extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        classes.append((bool(rng.integers(0, 2)), RATIO_CLASSES[int(rng.integers(0, 3))]))

    object_colors = _assign_object_colors(classes, lookup, rng)
    shapes = []
    for class_key, color in zip(classes, object_colors):
        has_hole, ratio = class_key
        mask = _shape_for_class(has_hole, ratio, rng)
        shapes.append((mask, color))

    shapes.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)
    reserved = np.zeros_like(grid, dtype=bool)
    _reserve(reserved, table_r0, table_c0, table_r1, table_c1, pad=2)

    for mask, color in shapes:
        placed = False
        sh, sw = mask.shape
        for _ in range(300):
            r0 = int(rng.integers(1, grid_h - sh - 1))
            c0 = int(rng.integers(1, grid_w - sw - 1))
            r1 = r0 + sh - 1
            c1 = c0 + sw - 1
            if reserved[max(0, r0 - 1) : min(grid_h, r1 + 2), max(0, c0 - 1) : min(grid_w, c1 + 2)].any():
                continue
            for r in range(sh):
                for c in range(sw):
                    if mask[r, c]:
                        grid[r0 + r, c0 + c] = color
            _reserve(reserved, r0, c0, r1, c1, pad=1)
            placed = True
            break
        if not placed:
            return None

    if _background_color(grid) != BACKGROUND:
        return None

    try:
        parsed_table = _find_rule_table(grid, BACKGROUND)
        parsed_lookup = _read_lookup(grid, parsed_table, BACKGROUND)
        if parsed_lookup != lookup:
            return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

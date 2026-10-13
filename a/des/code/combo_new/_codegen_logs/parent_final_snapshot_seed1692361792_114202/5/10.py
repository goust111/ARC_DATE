import random

import numpy as np


BLACK = 0
GRAY = 5
ORANGE = 7
AZURE = 8
OBJECT_COLORS = (1, 2, 3, 4, 6, 9)
CLASS_NAMES = ("wide", "tall", "square")
DIR_NAMES = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid, bg=BLACK):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return objects


def _normalize_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _parse_class_symbol(cell):
    coords = [tuple(map(int, rc)) for rc in np.argwhere(cell == ORANGE)]
    assert coords, "top rule cell must contain an orange symbol"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    area = len(coords)
    if area == 2 and h == 1 and w == 2:
        return "wide"
    if area == 2 and h == 2 and w == 1:
        return "tall"
    if area == 4 and h == 2 and w == 2:
        return "square"
    raise AssertionError("unrecognized class symbol")


def _parse_direction_symbol(cell):
    coords = [tuple(map(int, rc)) for rc in np.argwhere(cell == AZURE)]
    assert coords, "bottom rule cell must contain an azure symbol"
    shape = _normalize_coords(coords)
    patterns = {
        ((0, 0), (1, 0), (1, 1)): "up",
        ((0, 0), (0, 1), (1, 0)): "down",
        ((0, 1), (1, 0), (1, 1)): "left",
        ((0, 0), (0, 1), (1, 1)): "right",
    }
    assert shape in patterns, "unrecognized direction symbol"
    return patterns[shape]


def _find_rule_table(grid):
    h, w = grid.shape
    for top in range(h - 8):
        span = h - 1 - top
        if span % 2 != 0:
            continue
        delta = span // 2
        if delta < 4:
            continue
        mid = top + delta
        if not (np.all(grid[top] == GRAY) and np.all(grid[mid] == GRAY) and np.all(grid[-1] == GRAY)):
            continue
        band = grid[top:, :]
        cols = [c for c in range(w) if np.all(band[:, c] == GRAY)]
        if len(cols) < 4 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        gaps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        if len(set(gaps)) != 1:
            continue
        cell_w = gaps[0] - 1
        cell_h = delta - 1
        if cell_w < 2 or cell_h < 2:
            continue
        return top, mid, h - 1, cols
    raise AssertionError("failed to detect rule table")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object to a boundary according to the bottom rule table defined by its bbox aspect class.

    Input structure:
    - The background is black.
    - The upper area contains multiple separated monochrome 4-connected objects.
    - The bottom band is a grey-framed 2-row table whose top symbols are orange class markers and whose bottom symbols are azure direction markers.
    - Each orange marker is a 2-pixel horizontal line, a 2-pixel vertical line, or a 2x2 block; each azure marker is a 3-pixel wedge-like symbol.

    Rule:
    - Detect the rule table and discard it from the output.
    - For each object in the area above the table, compute the width and height of its bounding box.
    - Match wide objects to the horizontal orange symbol, tall objects to the vertical orange symbol, and square-bbox objects to the 2x2 orange block; then read the azure direction in the same table column.
    - Translate the entire object in that direction until its leading bbox edge is flush with the corresponding boundary of the active area, keeping color and shape unchanged.

    Key insight:
    - The lower strip is a visual lookup table: the top row classifies objects by bbox aspect ratio, and the bottom row tells each class which boundary to slide toward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, mid, bottom, cols = _find_rule_table(g)
    active = g[:top, :]
    assert active.size > 0, "active area must be non-empty"

    mapping = {}
    for i in range(len(cols) - 1):
        c0 = cols[i]
        c1 = cols[i + 1]
        top_cell = g[top + 1 : mid, c0 + 1 : c1]
        bottom_cell = g[mid + 1 : bottom, c0 + 1 : c1]
        cls = _parse_class_symbol(top_cell)
        direction = _parse_direction_symbol(bottom_cell)
        assert cls not in mapping, "rule classes must be unique"
        mapping[cls] = direction
    assert set(mapping) == set(CLASS_NAMES), "table must define wide, tall, and square cases"

    out = np.full(active.shape, BLACK, dtype=int)
    for obj in _find_components(active, bg=BLACK):
        r0, c0, r1, c1 = obj["bbox"]
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        if bbox_w > bbox_h:
            cls = "wide"
        elif bbox_h > bbox_w:
            cls = "tall"
        else:
            cls = "square"
        direction = mapping[cls]
        if direction == "up":
            dr, dc = -r0, 0
        elif direction == "down":
            dr, dc = active.shape[0] - 1 - r1, 0
        elif direction == "left":
            dr, dc = 0, -c0
        else:
            assert direction == "right"
            dr, dc = 0, active.shape[1] - 1 - c1
        for r, c in obj["coords"]:
            nr = r + dr
            nc = c + dc
            assert out[nr, nc] == BLACK, "translated objects must stay disjoint"
            out[nr, nc] = int(obj["color"])
    return out


def _connected(mask):
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    coord_set = set(coords)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _mask_valid(mask):
    if not np.any(mask):
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    return _connected(mask)


def _random_shape(class_name, rng):
    if class_name == "wide":
        height = rng.randint(2, 4)
        width = rng.randint(height + 1, min(height + 3, 6))
    elif class_name == "tall":
        width = rng.randint(2, 4)
        height = rng.randint(width + 1, min(width + 3, 6))
    else:
        side = rng.randint(2, 5)
        height = side
        width = side

    mask = np.ones((height, width), dtype=bool)
    edge_cells = []
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                edge_cells.append((r, c))
    rng.shuffle(edge_cells)
    removals = rng.randint(0, min(len(edge_cells), max(1, (height * width) // 4)))
    removed = 0
    for r, c in edge_cells:
        if removed >= removals:
            break
        mask[r, c] = False
        if not _mask_valid(mask):
            mask[r, c] = True
            continue
        removed += 1
    return mask


def _placement_points(mask, top, left):
    pts = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                pts.add((top + r, left + c))
    return pts


def _separated(occupied, pts):
    for r, c in pts:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _paint_mask(grid, mask, top, left, color):
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = int(color)


def _symbol_mask_class(class_name):
    if class_name == "wide":
        return np.array([[ORANGE, ORANGE]], dtype=int)
    if class_name == "tall":
        return np.array([[ORANGE], [ORANGE]], dtype=int)
    return np.array([[ORANGE, ORANGE], [ORANGE, ORANGE]], dtype=int)


def _symbol_mask_dir(direction):
    if direction == "up":
        return np.array([[AZURE, 0], [AZURE, AZURE]], dtype=int)
    if direction == "down":
        return np.array([[AZURE, AZURE], [AZURE, 0]], dtype=int)
    if direction == "left":
        return np.array([[0, AZURE], [AZURE, AZURE]], dtype=int)
    return np.array([[AZURE, AZURE], [0, AZURE]], dtype=int)


def _place_symbol(cell, symbol):
    h, w = cell.shape
    sh, sw = symbol.shape
    top = (h - sh) // 2
    left = (w - sw) // 2
    for r in range(sh):
        for c in range(sw):
            if symbol[r, c] != 0:
                cell[top + r, left + c] = int(symbol[r, c])


def _build_rule_strip(cell_size, column_order, mapping):
    width = 3 * cell_size + 4
    height = 2 * cell_size + 3
    band = np.full((height, width), BLACK, dtype=int)
    band[0, :] = GRAY
    band[cell_size + 1, :] = GRAY
    band[-1, :] = GRAY
    for c in range(0, width, cell_size + 1):
        band[:, c] = GRAY
    for i, class_name in enumerate(column_order):
        c0 = i * (cell_size + 1) + 1
        c1 = c0 + cell_size
        top_cell = band[1 : 1 + cell_size, c0:c1]
        bottom_cell = band[cell_size + 2 : cell_size + 2 + cell_size, c0:c1]
        _place_symbol(top_cell, _symbol_mask_class(class_name))
        _place_symbol(bottom_cell, _symbol_mask_dir(mapping[class_name]))
    return band


def _sample_case(rng):
    cell_size = rng.randint(3, 5)
    width = 3 * cell_size + 4
    rule_h = 2 * cell_size + 3
    active_h = rng.randint(10, 30 - rule_h)

    mapping = {class_name: rng.choice(DIR_NAMES) for class_name in CLASS_NAMES}
    column_order = list(CLASS_NAMES)
    rng.shuffle(column_order)

    object_count = rng.choices([3, 4, 5], weights=[2, 2, 1], k=1)[0]
    classes = list(CLASS_NAMES)
    while len(classes) < object_count:
        classes.append(rng.choice(CLASS_NAMES))
    rng.shuffle(classes)
    colors = rng.sample(OBJECT_COLORS, object_count)

    active_in = np.full((active_h, width), BLACK, dtype=int)
    active_out = np.full((active_h, width), BLACK, dtype=int)
    occupied_in = set()
    occupied_out = set()

    for class_name, color in zip(classes, colors):
        placed = False
        for _ in range(250):
            mask = _random_shape(class_name, rng)
            h, w = mask.shape
            if h > active_h or w > width:
                continue
            direction = mapping[class_name]
            if direction == "up":
                if active_h - h < 1:
                    continue
                top = rng.randint(1, active_h - h)
                left = rng.randint(0, width - w)
                out_top, out_left = 0, left
            elif direction == "down":
                if active_h - h < 1:
                    continue
                out_top = active_h - h
                top = rng.randint(0, out_top - 1)
                left = rng.randint(0, width - w)
                out_left = left
            elif direction == "left":
                if width - w < 1:
                    continue
                left = rng.randint(1, width - w)
                top = rng.randint(0, active_h - h)
                out_top, out_left = top, 0
            else:
                assert direction == "right"
                if width - w < 1:
                    continue
                out_left = width - w
                left = rng.randint(0, out_left - 1)
                top = rng.randint(0, active_h - h)
                out_top = top

            pts_in = _placement_points(mask, top, left)
            pts_out = _placement_points(mask, out_top, out_left)
            if not _separated(occupied_in, pts_in):
                continue
            if not _separated(occupied_out, pts_out):
                continue
            _paint_mask(active_in, mask, top, left, color)
            _paint_mask(active_out, mask, out_top, out_left, color)
            occupied_in |= pts_in
            occupied_out |= pts_out
            placed = True
            break
        if not placed:
            return None

    rule_strip = _build_rule_strip(cell_size, column_order, mapping)
    grid = np.vstack([active_in, rule_strip])
    return grid, active_out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _sample_case(rng)
        if result is None:
            continue
        grid, expected = result
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for task 10")

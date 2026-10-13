import random

import numpy as np


BG = 0
GREY = 5
DEPTHS = (1, 2, 3, 4)
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
OBJECT_RETRIES = 250
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    for col in (0, w - 1):
        rows = np.flatnonzero(grid[:, col] != bg)
        if len(rows) != 4 or not np.all(np.diff(rows) == 1):
            continue
        colors = [int(grid[r, col]) for r in rows]
        if len(set(colors)) != 4:
            continue
        coords = {(int(r), int(col)) for r in rows}
        return {depth: colors[depth - 1] for depth in DEPTHS}, coords
    raise AssertionError("expected a vertical 4-cell legend on the left or right edge")


def _object_color(grid: np.ndarray, bg: int, legend_coords: set[tuple[int, int]]) -> int:
    mask = grid != bg
    for r, c in legend_coords:
        mask[r, c] = False
    values = grid[mask]
    assert values.size > 0, "expected at least one foreground object"
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _notch_depth(coords: np.ndarray, side: str, h: int, w: int) -> int:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if side == "top":
        return int(rows.max()) + 1
    if side == "bottom":
        return int(h - rows.min())
    if side == "left":
        return int(cols.max()) + 1
    if side == "right":
        return int(w - cols.min())
    raise ValueError(side)


def _analyze_notches(mask: np.ndarray) -> tuple[list[dict[str, object]], int]:
    h, w = mask.shape
    notches: list[dict[str, object]] = []
    rejected = 0
    for coords in _components(~mask):
        sides = []
        if np.any(coords[:, 0] == 0):
            sides.append("top")
        if np.any(coords[:, 0] == h - 1):
            sides.append("bottom")
        if np.any(coords[:, 1] == 0):
            sides.append("left")
        if np.any(coords[:, 1] == w - 1):
            sides.append("right")
        if len(sides) != 1:
            rejected += 1
            continue
        side = sides[0]
        depth = _notch_depth(coords, side, h, w)
        notches.append({"coords": coords, "side": side, "depth": depth})
    return notches, rejected


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend, legend_coords = _find_legend(g, bg)
    obj_color = _object_color(g, bg, legend_coords)

    out = g.copy()
    for coords in _components(g == obj_color):
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        local[rel[:, 0], rel[:, 1]] = True
        notches, _ = _analyze_notches(local)
        for notch in notches:
            depth = int(notch["depth"])
            assert depth in legend, "notch depth must map to one legend color"
            pts = notch["coords"]
            out[r0 + pts[:, 0], c0 + pts[:, 1]] = legend[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each concave notch of every grey object using the edge legend's depth-to-color mapping.

    Input structure:
    - The background is black. One edge contains a vertical 4-cell color legend whose top-to-bottom order encodes depths 1, 2, 3, 4.
    - Away from the edge legend, the grid contains several 4-connected grey foreground objects on the black background.
    - Inside each object's bounding box, some black regions are inward dents of the outline rather than separate objects.

    Rule:
    - Read the legend as depth 1 -> top color, depth 2 -> next color, depth 3 -> next color, depth 4 -> bottom color.
    - For each grey object, inspect black connected regions inside its tight bounding box.
    - Any such region that touches exactly one side of the bounding box is a notch; measure its depth as the maximum number of pixel layers it extends inward from that touched side.
    - Fill every notch with the legend color for its depth, while leaving the grey object and the legend unchanged.

    Key insight:
    - The black cells to recolor are not arbitrary background patches: they are bbox-internal background components whose geometry reveals both that they are true concavities and how deep they go.
    """
    return _transform(grid)


def _top_notch_cells(h: int, w: int, depth: int, rng: random.Random) -> np.ndarray | None:
    if depth < 1 or depth > 4 or depth >= h or w < 3:
        return None
    span = rng.randint(1, w - 2)
    start = rng.randint(1, w - span - 1)
    left = start
    right = start + span - 1
    taper = depth > 1 and rng.random() < 0.6
    cells: list[tuple[int, int]] = []
    for layer in range(depth):
        if taper and layer > 0 and left < right:
            shrink_left = rng.randint(0, 1)
            shrink_right = rng.randint(0, 1)
            if left + shrink_left <= right - shrink_right:
                left += shrink_left
                right -= shrink_right
        for c in range(left, right + 1):
            cells.append((layer, c))
    return np.array(cells, dtype=int)


def _random_notch_cells(h: int, w: int, side: str, depth: int, rng: random.Random) -> np.ndarray | None:
    if side == "top":
        return _top_notch_cells(h, w, depth, rng)
    if side == "bottom":
        base = _top_notch_cells(h, w, depth, rng)
        if base is None:
            return None
        out = base.copy()
        out[:, 0] = h - 1 - out[:, 0]
        return out
    if side == "left":
        base = _top_notch_cells(w, h, depth, rng)
        if base is None:
            return None
        return base[:, ::-1].copy()
    if side == "right":
        base = _top_notch_cells(w, h, depth, rng)
        if base is None:
            return None
        out = base[:, ::-1].copy()
        out[:, 1] = w - 1 - out[:, 1]
        return out
    raise ValueError(side)


def _touches_carved(cells: np.ndarray, carved: np.ndarray) -> bool:
    h, w = carved.shape
    for r, c in cells:
        if carved[r, c]:
            return True
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and carved[nr, nc]:
                return True
    return False


def _make_object(depths: list[int], rng: random.Random) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    min_dim = max(6, max(depths) + 2, 4 + len(depths))
    max_dim = min(11, min_dim + 4)
    for _ in range(OBJECT_RETRIES):
        h = rng.randint(min_dim, max_dim)
        w = rng.randint(min_dim, max_dim)
        carved = np.zeros((h, w), dtype=bool)
        order = list(depths)
        rng.shuffle(order)
        ok = True
        for depth in order:
            placed = False
            for _ in range(80):
                side = rng.choice(SIDES)
                cells = _random_notch_cells(h, w, side, depth, rng)
                if cells is None or _touches_carved(cells, carved):
                    continue
                carved[cells[:, 0], cells[:, 1]] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        mask = ~carved
        if len(_components(mask)) != 1 or int(mask.sum()) < 12:
            continue
        notches, rejected = _analyze_notches(mask)
        if rejected != 0 or len(notches) != len(depths):
            continue
        if sorted(int(notch["depth"]) for notch in notches) != sorted(depths):
            continue
        return mask, notches
    return None


def _sample_depth_groups(rng: random.Random, object_count: int) -> list[list[int]]:
    total_notches = 4 + rng.randint(0, 2)
    counts = [1] * object_count
    while sum(counts) < total_notches:
        choices = [i for i, count in enumerate(counts) if count < 3]
        counts[rng.choice(choices)] += 1
    depths = list(DEPTHS) + [rng.randint(1, 4) for _ in range(total_notches - 4)]
    rng.shuffle(depths)
    groups: list[list[int]] = []
    idx = 0
    for count in counts:
        groups.append(depths[idx : idx + count])
        idx += count
    return groups


def _place_object(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    notches: list[dict[str, object]],
    legend: dict[int, int],
    legend_side: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    min_r = 1
    max_r = h - ph - 1
    if legend_side == "left":
        min_c = 2
        max_c = w - pw - 1
    else:
        min_c = 1
        max_c = w - pw - 2
    if max_r < min_r or max_c < min_c:
        return False

    positions = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    rng.shuffle(positions)
    for r, c in positions:
        rr0 = max(0, r - 1)
        rr1 = min(h, r + ph + 1)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + pw + 1)
        if np.any(blocked[rr0:rr1, cc0:cc1]):
            continue

        sub = grid[r : r + ph, c : c + pw]
        sub[patch] = GREY

        exp_sub = expected[r : r + ph, c : c + pw]
        exp_sub[patch] = GREY
        for notch in notches:
            pts = notch["coords"]
            exp_sub[pts[:, 0], pts[:, 1]] = legend[int(notch["depth"])]

        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    legend_side = rng.choice(("left", "right"))
    legend_col = 0 if legend_side == "left" else w - 1
    legend_top = rng.randint(0, h - 4)

    legend_colors = rng.sample(LEGEND_COLORS, 4)
    legend = {depth: legend_colors[depth - 1] for depth in DEPTHS}

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for offset, color in enumerate(legend_colors):
        grid[legend_top + offset, legend_col] = color
        expected[legend_top + offset, legend_col] = color

    blocked = np.zeros((h, w), dtype=bool)
    rr0 = max(0, legend_top - 1)
    rr1 = min(h, legend_top + 5)
    if legend_side == "left":
        blocked[rr0:rr1, 0 : min(2, w)] = True
    else:
        blocked[rr0:rr1, max(0, w - 2) : w] = True

    object_count = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    depth_groups = _sample_depth_groups(rng, object_count)
    rng.shuffle(depth_groups)

    for group in depth_groups:
        built = _make_object(group, rng)
        if built is None:
            return None
        patch, notches = built
        if not _place_object(grid, expected, blocked, patch, notches, legend, legend_side, rng):
            return None

    if not np.any(expected != grid):
        return None
    for depth, color in legend.items():
        if not np.any((expected == color) & (grid == BG)):
            return None
    if not np.array_equal(_transform(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

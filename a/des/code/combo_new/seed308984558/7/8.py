import random
from collections import Counter, deque

import numpy as np


GREY = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
L_SHAPES = {
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
}

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 220
SHAPE_ATTEMPTS = 120

GRID_MIN = 16
GRID_MAX = 28
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_HEIGHT_RANGE = (2, 5)
SHAPE_WIDTH_RANGE = (2, 5)
EXTRA_SHAPE_CELLS = (0, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _is_arrow_component(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 3:
        return False
    return frozenset(_normalize(coords)) in L_SHAPES


def _row_mode(row: np.ndarray) -> int:
    values, counts = np.unique(row, return_counts=True)
    return int(values[np.argmax(counts)])


def _stripe_colors(grid: np.ndarray) -> tuple[int, int]:
    row_modes = [_row_mode(grid[r]) for r in range(grid.shape[0])]
    even_counter = Counter(row_modes[::2])
    odd_counter = Counter(row_modes[1::2])
    even_color = int(even_counter.most_common(1)[0][0])
    odd_color = int(odd_counter.most_common(1)[0][0]) if odd_counter else even_color
    assert even_color != odd_color, "expected two alternating stripe colors"
    return even_color, odd_color


def _direction_from_pair(arrow: list[tuple[int, int]], obj: list[tuple[int, int]]) -> str:
    ar0, ar1, ac0, ac1 = _bbox(arrow)
    or0, or1, oc0, oc1 = _bbox(obj)
    candidates: list[str] = []
    if ac1 + 2 == oc0 and _overlap_1d(ar0, ar1, or0, or1):
        candidates.append("right")
    if oc1 + 2 == ac0 and _overlap_1d(ar0, ar1, or0, or1):
        candidates.append("left")
    if ar1 + 2 == or0 and _overlap_1d(ac0, ac1, oc0, oc1):
        candidates.append("down")
    if or1 + 2 == ar0 and _overlap_1d(ac0, ac1, oc0, oc1):
        candidates.append("up")
    assert len(candidates) == 1, "arrow/object side relation must determine one direction"
    return candidates[0]


def _shadow_length_from_stripes(obj: list[tuple[int, int]]) -> int:
    r0, r1, _c0, _c1 = _bbox(obj)
    return r1 - r0 + 1


def _paint_shadow(
    out: np.ndarray,
    src: np.ndarray,
    bg_colors: set[int],
    obj: list[tuple[int, int]],
    direction: str,
    shadow_len: int,
) -> None:
    h, w = src.shape
    rows = sorted({r for r, _ in obj})
    cols = sorted({c for _, c in obj})
    _r0, r1, c0, c1 = _bbox(obj)

    if direction == "right":
        for step in range(shadow_len):
            cc = c1 + 1 + step
            if not (0 <= cc < w):
                continue
            for rr in rows:
                if int(src[rr, cc]) in bg_colors:
                    out[rr, cc] = GREY
    elif direction == "left":
        for step in range(shadow_len):
            cc = c0 - 1 - step
            if not (0 <= cc < w):
                continue
            for rr in rows:
                if int(src[rr, cc]) in bg_colors:
                    out[rr, cc] = GREY
    elif direction == "down":
        for step in range(shadow_len):
            rr = r1 + 1 + step
            if not (0 <= rr < h):
                continue
            for cc in cols:
                if int(src[rr, cc]) in bg_colors:
                    out[rr, cc] = GREY
    else:
        for step in range(shadow_len):
            rr = _r0 - 1 - step
            if not (0 <= rr < h):
                continue
            for cc in cols:
                if int(src[rr, cc]) in bg_colors:
                    out[rr, cc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a grey shadow from each colored object in the direction indicated by its nearby L-shaped arrow.

    Input structure:
    - The background is made of one-pixel-tall horizontal stripes that alternate between two colors.
    - Several monochrome 4-connected foreground objects sit on top of the stripes, and each object uses a color that is different from the stripe colors.
    - Each object has a same-colored 3-pixel L arrow on one cardinal side with a one-pixel gap between arrow and object.

    Rule:
    - Recover the two stripe colors from the row-wise background pattern.
    - For each foreground color, split its pixels into the 3-cell L arrow component and the object component, then read the projection direction from which side the arrow occupies relative to the object.
    - Count how many horizontal stripe rows the object's bounding box crosses; call that number N.
    - Starting one cell beyond the object's outer edge, draw the object's orthogonal projection silhouette in grey for N columns or rows in the arrow direction, but only onto background-stripe cells.

    Key insight:
    - The arrow gives only the casting direction; the hidden quantity is the object's stripe-span, which is encoded by how many alternating background rows its bounding box covers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    even_color, odd_color = _stripe_colors(g)
    bg_colors = {even_color, odd_color}

    foreground_colors = [int(color) for color in np.unique(g) if int(color) not in bg_colors]
    assert foreground_colors, "expected at least one object color"

    out = g.copy()
    for color in foreground_colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each object color must form exactly one object and one arrow"

        arrow_candidates = [comp for comp in comps if _is_arrow_component(comp)]
        assert len(arrow_candidates) == 1, "expected exactly one 3-cell L arrow per color"
        arrow = arrow_candidates[0]
        obj = comps[0] if comps[1] is arrow else comps[1]
        assert len(obj) >= 4, "foreground object must be distinguishable from the arrow"

        direction = _direction_from_pair(arrow, obj)
        shadow_len = _shadow_length_from_stripes(obj)
        _paint_shadow(out, g, bg_colors, obj, direction, shadow_len)

    return out


def _random_shape() -> list[tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(*SHAPE_HEIGHT_RANGE)
        w = random.randint(*SHAPE_WIDTH_RANGE)

        steps = ["D"] * (h - 1) + ["R"] * (w - 1)
        random.shuffle(steps)
        cells = {(0, 0)}
        r = c = 0
        for step in steps:
            if step == "D":
                r += 1
            else:
                c += 1
            cells.add((r, c))

        min_cells = max(4, len(cells))
        max_extra = min(h * w - min_cells, random.randint(*EXTRA_SHAPE_CELLS))
        target = min_cells + max_extra
        while len(cells) < target:
            frontier = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        shape = list(cells)
        if random.random() < 0.5:
            shape = [(rr, w - 1 - cc) for rr, cc in shape]
        if random.random() < 0.5:
            shape = [(h - 1 - rr, cc) for rr, cc in shape]
        if random.random() < 0.35:
            shape = [(cc, rr) for rr, cc in shape]

        norm = _normalize(shape)
        rows = {r0 for r0, _ in norm}
        cols = {c0 for _, c0 in norm}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(norm) < 4:
            continue
        return norm
    raise RuntimeError("failed to generate a valid object shape")


def _arrow_shape(direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        options = [
            [(0, 0), (0, 1), (1, 1)],
            [(1, 0), (1, 1), (0, 1)],
        ]
    elif direction == "left":
        options = [
            [(0, 1), (0, 0), (1, 0)],
            [(1, 1), (1, 0), (0, 0)],
        ]
    elif direction == "down":
        options = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (1, 0)],
        ]
    else:
        options = [
            [(1, 0), (0, 0), (0, 1)],
            [(1, 1), (0, 1), (0, 0)],
        ]
    return random.choice(options)


def _shadow_cells(shape: list[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    rows = sorted({r for r, _ in shape})
    cols = sorted({c for _, c in shape})
    h = max(rows) + 1
    w = max(cols) + 1
    shadow_len = h
    cells: set[tuple[int, int]] = set()
    if direction == "right":
        for step in range(shadow_len):
            for rr in rows:
                cells.add((rr, w + step))
    elif direction == "left":
        for step in range(shadow_len):
            for rr in rows:
                cells.add((rr, -1 - step))
    elif direction == "down":
        for step in range(shadow_len):
            for cc in cols:
                cells.add((h + step, cc))
    else:
        for step in range(shadow_len):
            for cc in cols:
                cells.add((-1 - step, cc))
    return cells


def _layout_parts(
    shape: list[tuple[int, int]],
    direction: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    h = max(rows) + 1
    w = max(cols) + 1
    arrow = _arrow_shape(direction)

    if direction in ("right", "left"):
        align = random.randint(0, h - 2)
    else:
        align = random.randint(0, w - 2)

    if direction == "right":
        arrow = [(align + r, -3 + c) for r, c in arrow]
        gap = {(align + rr, -1) for rr in range(2)}
    elif direction == "left":
        arrow = [(align + r, w + 1 + c) for r, c in arrow]
        gap = {(align + rr, w) for rr in range(2)}
    elif direction == "down":
        arrow = [(-3 + r, align + c) for r, c in arrow]
        gap = {(-1, align + cc) for cc in range(2)}
    else:
        arrow = [(h + 1 + r, align + c) for r, c in arrow]
        gap = {(h, align + cc) for cc in range(2)}

    obj = list(shape)
    shadow = _shadow_cells(shape, direction)
    reserved = set(obj) | set(arrow) | gap | shadow
    return obj, arrow, gap, reserved


def _fits(
    reserved: set[tuple[int, int]],
    top: int,
    left: int,
    blocked: np.ndarray,
) -> bool:
    h, w = blocked.shape
    for r, c in reserved:
        rr = top + r
        cc = left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if bool(blocked[rr, cc]):
            return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


def _row_background_is_visible(grid: np.ndarray, even_color: int, odd_color: int) -> bool:
    h, w = grid.shape
    for r in range(h):
        expected = even_color if r % 2 == 0 else odd_color
        if int(np.count_nonzero(grid[r] == expected)) <= w // 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_MIN, 6 + object_count * 4)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    palette = [color for color in range(10) if color != GREY]
    even_color, odd_color = random.sample(palette, 2)
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        grid[r, :] = even_color if r % 2 == 0 else odd_color

    remaining = [color for color in palette if color not in {even_color, odd_color}]
    object_colors = random.sample(remaining, object_count)
    directions = random.sample(list(DIRS), object_count)

    blocked = np.zeros((h, w), dtype=bool)
    for color, direction in zip(object_colors, directions):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            shape = _random_shape()
            obj, arrow, _gap, reserved = _layout_parts(shape, direction)
            rel_rows = [r for r, _ in reserved]
            rel_cols = [c for _, c in reserved]
            min_r, max_r = min(rel_rows), max(rel_rows)
            min_c, max_c = min(rel_cols), max(rel_cols)

            top = random.randint(-min_r, h - 1 - max_r)
            left = random.randint(-min_c, w - 1 - max_c)
            if not _fits(reserved, top, left, blocked):
                continue

            _paint_cells(grid, obj, top, left, color)
            _paint_cells(grid, arrow, top, left, color)
            for rr, cc in reserved:
                blocked[top + rr, left + cc] = True
            placed = True
            break
        if not placed:
            return None

    if not _row_background_is_visible(grid, even_color, odd_color):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-shadow ARC input")

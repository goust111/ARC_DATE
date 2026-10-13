from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
OBJECT_PLACEMENT_ATTEMPTS = 100
SHAPE_ATTEMPTS = 120
MAX_NOTCHES = 3
MIN_INNER_SIZE = 3
MIN_INNER_AREA = 8


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _dominant_color(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, f"expected a unique dominant edge color, got {counts}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed orthogonal polygon interior according to opposite-edge color agreement and mark conflict corners.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick orthogonal polygon borders enclose black interior regions.
    - Each horizontal or vertical border segment has a uniform non-black color, and different segments of the same polygon may use different colors.

    Rule:
    - For every enclosed black region, read the colors on the outermost top and bottom border segments.
    - If those opposite horizontal sides match, fill the whole interior with that shared color; otherwise fill the interior row by row from the top, alternating the outermost left-side and right-side colors.
    - After filling, any interior pixel that sits diagonally inside a corner whose adjacent horizontal and vertical border colors differ is recolored yellow.

    Key insight:
    - The first split depends only on top-vs-bottom color equality, but the stripe colors come from the left and right sides; yellow marks are purely local corner-conflict signals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = g.copy()
    bg_mask = g == BG

    for coords in _component_coords(bg_mask):
        if _touches_border(coords, h, w):
            continue

        region = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            region[r, c] = True

        top_edges: list[tuple[int, int, int]] = []
        bottom_edges: list[tuple[int, int, int]] = []
        left_edges: list[tuple[int, int, int]] = []
        right_edges: list[tuple[int, int, int]] = []

        for r, c in coords:
            if r > 0 and not region[r - 1, c]:
                color = int(g[r - 1, c])
                assert color != BG
                top_edges.append((r - 1, c, color))
            if r + 1 < h and not region[r + 1, c]:
                color = int(g[r + 1, c])
                assert color != BG
                bottom_edges.append((r + 1, c, color))
            if c > 0 and not region[r, c - 1]:
                color = int(g[r, c - 1])
                assert color != BG
                left_edges.append((r, c - 1, color))
            if c + 1 < w and not region[r, c + 1]:
                color = int(g[r, c + 1])
                assert color != BG
                right_edges.append((r, c + 1, color))

        assert top_edges and bottom_edges and left_edges and right_edges, "region must be enclosed by colored edges"

        top_row = min(r for r, _c, _color in top_edges)
        bottom_row = max(r for r, _c, _color in bottom_edges)
        left_col = min(c for _r, c, _color in left_edges)
        right_col = max(c for _r, c, _color in right_edges)

        top_color = _dominant_color([color for r, _c, color in top_edges if r == top_row])
        bottom_color = _dominant_color([color for r, _c, color in bottom_edges if r == bottom_row])
        left_color = _dominant_color([color for _r, c, color in left_edges if c == left_col])
        right_color = _dominant_color([color for _r, c, color in right_edges if c == right_col])

        first_row = min(r for r, _c in coords)
        if top_color == bottom_color:
            for r, c in coords:
                out[r, c] = top_color
        else:
            for r, c in coords:
                out[r, c] = left_color if (r - first_row) % 2 == 0 else right_color

        for r, c in coords:
            has_top = r > 0 and not region[r - 1, c]
            has_bottom = r + 1 < h and not region[r + 1, c]
            has_left = c > 0 and not region[r, c - 1]
            has_right = c + 1 < w and not region[r, c + 1]

            if has_top and has_left and int(g[r - 1, c]) != int(g[r, c - 1]):
                out[r, c] = YELLOW
            if has_top and has_right and int(g[r - 1, c]) != int(g[r, c + 1]):
                out[r, c] = YELLOW
            if has_bottom and has_left and int(g[r + 1, c]) != int(g[r, c - 1]):
                out[r, c] = YELLOW
            if has_bottom and has_right and int(g[r + 1, c]) != int(g[r, c + 1]):
                out[r, c] = YELLOW

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1


def _sample_shape(max_size: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(MIN_INNER_SIZE, max_size)
        w = random.randint(MIN_INNER_SIZE, max_size)
        mask = np.ones((h, w), dtype=bool)

        for _ in range(random.randint(0, MAX_NOTCHES)):
            temp = mask.copy()
            side = random.choice(("top", "bottom", "left", "right"))

            if side in ("top", "bottom"):
                max_depth = temp.shape[0] - 2
                max_span = temp.shape[1] - 1
                if max_depth < 1 or max_span < 1:
                    continue
                depth = random.randint(1, max_depth)
                span = random.randint(1, max_span)
                start = random.randint(0, temp.shape[1] - span)
                rows = slice(0, depth) if side == "top" else slice(temp.shape[0] - depth, temp.shape[0])
                temp[rows, start : start + span] = False
            else:
                max_depth = temp.shape[1] - 2
                max_span = temp.shape[0] - 1
                if max_depth < 1 or max_span < 1:
                    continue
                depth = random.randint(1, max_depth)
                span = random.randint(1, max_span)
                start = random.randint(0, temp.shape[0] - span)
                cols = slice(0, depth) if side == "left" else slice(temp.shape[1] - depth, temp.shape[1])
                temp[start : start + span, cols] = False

            if not temp.any():
                continue
            temp = _crop_mask(temp)
            if temp.shape[0] < MIN_INNER_SIZE or temp.shape[1] < MIN_INNER_SIZE:
                continue
            if int(temp.sum()) < MIN_INNER_AREA:
                continue
            if _connected(temp):
                mask = temp

        mask = _crop_mask(mask)
        if mask.shape[0] < MIN_INNER_SIZE or mask.shape[1] < MIN_INNER_SIZE:
            continue
        if int(mask.sum()) < MIN_INNER_AREA:
            continue
        if _connected(mask):
            return mask
    return None


def _group_horizontal_runs(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    rows: dict[int, list[int]] = {}
    for r, c in cells:
        rows.setdefault(int(r), []).append(int(c))

    runs: list[list[tuple[int, int]]] = []
    for r, cols in rows.items():
        cols.sort()
        start = prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
                continue
            runs.append([(r, cc) for cc in range(start, prev + 1)])
            start = prev = c
        runs.append([(r, cc) for cc in range(start, prev + 1)])
    return runs


def _group_vertical_runs(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    cols: dict[int, list[int]] = {}
    for r, c in cells:
        cols.setdefault(int(c), []).append(int(r))

    runs: list[list[tuple[int, int]]] = []
    for c, rows in cols.items():
        rows.sort()
        start = prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
                continue
            runs.append([(rr, c) for rr in range(start, prev + 1)])
            start = prev = r
        runs.append([(rr, c) for rr in range(start, prev + 1)])
    return runs


def _choose_color(exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    options = [color for color in PALETTE if color not in exclude]
    assert options
    return int(random.choice(options))


def _mask_side_cells(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = mask.shape
    top: set[tuple[int, int]] = set()
    bottom: set[tuple[int, int]] = set()
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pr, pc = r + 1, c + 1
            if r == 0 or not mask[r - 1, c]:
                top.add((pr - 1, pc))
            if r == h - 1 or not mask[r + 1, c]:
                bottom.add((pr + 1, pc))
            if c == 0 or not mask[r, c - 1]:
                left.add((pr, pc - 1))
            if c == w - 1 or not mask[r, c + 1]:
                right.add((pr, pc + 1))
    return top, bottom, left, right


def _build_patch(mask: np.ndarray, mode: str, force_conflict: bool) -> tuple[np.ndarray, bool] | None:
    top_cells, bottom_cells, left_cells, right_cells = _mask_side_cells(mask)
    horizontal = top_cells | bottom_cells
    vertical = left_cells | right_cells
    assert horizontal and vertical

    top_row = min(r for r, _c in horizontal)
    bottom_row = max(r for r, _c in horizontal)
    left_col = min(c for _r, c in vertical)
    right_col = max(c for _r, c in vertical)

    top_color = _choose_color()
    bottom_color = top_color if mode == "pure" else _choose_color({top_color})

    right_color = _choose_color()
    if mode == "stripe":
        left_exclude = {right_color}
        if force_conflict:
            left_exclude.add(top_color)
        left_color = _choose_color(left_exclude)
    else:
        left_color = _choose_color({top_color} if force_conflict else set())

    horizontal_colors: dict[tuple[int, int], int] = {}
    vertical_colors: dict[tuple[int, int], int] = {}

    for run in _group_horizontal_runs(horizontal):
        row = run[0][0]
        if row == top_row:
            color = top_color
        elif row == bottom_row:
            color = bottom_color
        else:
            color = _choose_color()
        for cell in run:
            horizontal_colors[cell] = color

    for run in _group_vertical_runs(vertical):
        col = run[0][1]
        if col == left_col:
            color = left_color
        elif col == right_col:
            color = right_color
        else:
            color = _choose_color()
        for cell in run:
            vertical_colors[cell] = color

    patch = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=int)
    for cell, color in horizontal_colors.items():
        patch[cell] = color
    for cell, color in vertical_colors.items():
        if patch[cell] == 0:
            patch[cell] = color

    has_conflict = False
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            pr, pc = r + 1, c + 1
            top = horizontal_colors.get((pr - 1, pc))
            bottom = horizontal_colors.get((pr + 1, pc))
            left = vertical_colors.get((pr, pc - 1))
            right = vertical_colors.get((pr, pc + 1))

            if top is not None and left is not None and top != left:
                has_conflict = True
            if top is not None and right is not None and top != right:
                has_conflict = True
            if bottom is not None and left is not None and bottom != left:
                has_conflict = True
            if bottom is not None and right is not None and bottom != right:
                has_conflict = True

    if force_conflict and not has_conflict:
        return None

    return patch, has_conflict


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)

    for r, c in candidates[:OBJECT_PLACEMENT_ATTEMPTS]:
        rr0 = max(0, r - 1)
        rr1 = min(h, r + ph + 1)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + pw + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        if np.any((patch != 0) & (grid[r : r + ph, c : c + pw] != 0)):
            continue
        grid[r : r + ph, c : c + pw][patch != 0] = patch[patch != 0]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _sample_scene() -> np.ndarray | None:
    object_count = random.randint(2, 4)
    min_side = {2: 16, 3: 20, 4: 24}[object_count]
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    size_cap = {2: 10, 3: 8, 4: 7}[object_count]
    modes = ["pure", "stripe"] + [random.choice(("pure", "stripe")) for _ in range(object_count - 2)]
    random.shuffle(modes)
    conflict_target = random.randrange(object_count)

    for idx, mode in enumerate(modes):
        force_conflict = idx == conflict_target
        placed = False
        for _ in range(SHAPE_ATTEMPTS):
            mask = _sample_shape(size_cap)
            if mask is None:
                continue
            built = _build_patch(mask, mode, force_conflict=force_conflict)
            if built is None:
                continue
            patch, _has_conflict = built
            if _place_patch(grid, reserved, patch):
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_scene()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        if not np.any(out == YELLOW):
            continue
        return grid
    raise RuntimeError("failed to generate a valid orthogonal-polygon ARC sample")

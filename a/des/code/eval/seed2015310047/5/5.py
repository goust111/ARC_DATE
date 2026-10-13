from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PROFILE_VALUES = (3, 5, 7)
MAX_GEN_ATTEMPTS = 300
GRID_MIN = 18
GRID_MAX = 24
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
PLACEMENT_RETRIES = 120
COLORS = tuple(range(10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _edge_runs(line: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        color = int(line[i])
        if color == bg:
            i += 1
            continue
        j = i
        while j + 1 < n and int(line[j + 1]) == color:
            j += 1
        runs.append((i, j, color))
        i = j + 1
    return runs


def _extract_walls(grid: np.ndarray, bg: int):
    h, w = grid.shape
    segments = []
    for start, end, color in _edge_runs(grid[0, :], bg):
        segments.append({"side": "top", "fixed": 0, "start": start, "end": end, "color": color})
    for start, end, color in _edge_runs(grid[h - 1, :], bg):
        segments.append({"side": "bottom", "fixed": h - 1, "start": start, "end": end, "color": color})
    for start, end, color in _edge_runs(grid[:, 0], bg):
        segments.append({"side": "left", "fixed": 0, "start": start, "end": end, "color": color})
    for start, end, color in _edge_runs(grid[:, w - 1], bg):
        segments.append({"side": "right", "fixed": w - 1, "start": start, "end": end, "color": color})

    walls: dict[int, dict[str, int | str]] = {}
    wall_cells: set[tuple[int, int]] = set()
    for seg in segments:
        color = int(seg["color"])
        assert color not in walls, "each wall color must be unique"
        walls[color] = seg
        if seg["side"] in ("top", "bottom"):
            row = int(seg["fixed"])
            for col in range(int(seg["start"]), int(seg["end"]) + 1):
                wall_cells.add((row, col))
        else:
            col = int(seg["fixed"])
            for row in range(int(seg["start"]), int(seg["end"]) + 1):
                wall_cells.add((row, col))

    assert walls, "expected at least one boundary wall"
    return walls, wall_cells


def _is_l_triomino(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != 3:
        return False
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) != 2 or len(cols) != 2:
        return False
    return len(set(cells)) == 3


def _segment_bounds(center: int, length: int) -> tuple[int, int]:
    start = center - length // 2
    end = start + length - 1
    return start, end


def _project_body(
    body_cells: list[tuple[int, int]],
    wall: dict[str, int | str],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    side = str(wall["side"])
    paint: list[tuple[int, int]] = []

    # A 3-cell L is symmetric, so the matched wall side resolves the projection axis.
    if side in ("top", "bottom"):
        target_row = 1 if side == "top" else h - 2
        by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in body_cells:
            by_col[c].append(r)
        for col in sorted(by_col):
            span = max(by_col[col]) - min(by_col[col]) + 1
            start, end = _segment_bounds(col, span)
            assert 0 <= start and end < w
            assert int(wall["start"]) <= start and end <= int(wall["end"])
            for cc in range(start, end + 1):
                paint.append((target_row, cc))
    else:
        target_col = 1 if side == "left" else w - 2
        by_row: dict[int, list[int]] = defaultdict(list)
        for r, c in body_cells:
            by_row[r].append(c)
        for row in sorted(by_row):
            span = max(by_row[row]) - min(by_row[row]) + 1
            start, end = _segment_bounds(row, span)
            assert 0 <= start and end < h
            assert int(wall["start"]) <= start and end <= int(wall["end"])
            for rr in range(start, end + 1):
                paint.append((rr, target_col))
    return paint


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each embedded arrow to a same-colored boundary wall, then compress the
    object's body onto that wall's inner adjacent stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    walls, wall_cells = _extract_walls(g, bg)
    object_mask = (g != bg).copy()
    for r, c in wall_cells:
        object_mask[r, c] = False

    components = _components(object_mask)
    assert components, "expected at least one interior object"

    out = g.copy()
    out[object_mask] = bg
    paint_map: dict[tuple[int, int], int] = {}

    for comp in components:
        by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c in comp:
            by_color[int(g[r, c])].append((r, c))

        assert len(by_color) == 2, "each object must contain one body color and one arrow color"
        arrow_colors = [color for color, cells in by_color.items() if len(cells) == 3 and _is_l_triomino(cells)]
        assert len(arrow_colors) == 1, "expected exactly one 3-cell L arrow per object"
        arrow_color = arrow_colors[0]
        assert arrow_color in walls, "arrow color must match a wall color"

        body_color = next(color for color in by_color if color != arrow_color)
        body_cells = by_color[body_color]
        assert body_cells, "object body must be non-empty"

        for cell in _project_body(body_cells, walls[arrow_color], g.shape):
            prev = paint_map.get(cell)
            assert prev is None or prev == body_color, "projection collision must be color-consistent"
            paint_map[cell] = body_color

    for (r, c), color in paint_map.items():
        assert (r, c) not in wall_cells
        out[r, c] = color
    return out


def _sample_wall_ranges(h: int, w: int) -> dict[str, tuple[int, int]]:
    ranges = {
        "top": (random.randint(1, 3), w - 1 - random.randint(1, 3)),
        "bottom": (random.randint(1, 3), w - 1 - random.randint(1, 3)),
        "left": (random.randint(1, 3), h - 1 - random.randint(1, 3)),
        "right": (random.randint(1, 3), h - 1 - random.randint(1, 3)),
    }
    if any(end - start + 1 < 10 for start, end in ranges.values()):
        raise ValueError("wall span too short")
    return ranges


def _paint_walls(grid: np.ndarray, wall_ranges: dict[str, tuple[int, int]], wall_colors: dict[str, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    wall_cells: set[tuple[int, int]] = set()
    for side in SIDES:
        start, end = wall_ranges[side]
        color = wall_colors[side]
        if side == "top":
            row = 0
            grid[row, start : end + 1] = color
            wall_cells.update((row, c) for c in range(start, end + 1))
        elif side == "bottom":
            row = h - 1
            grid[row, start : end + 1] = color
            wall_cells.update((row, c) for c in range(start, end + 1))
        elif side == "left":
            col = 0
            grid[start : end + 1, col] = color
            wall_cells.update((r, col) for r in range(start, end + 1))
        else:
            col = w - 1
            grid[start : end + 1, col] = color
            wall_cells.update((r, col) for r in range(start, end + 1))
    return wall_cells


def _expanded_bbox(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = max(0, min(rows) - pad)
    r1 = min(h - 1, max(rows) + pad)
    c0 = max(0, min(cols) - pad)
    c1 = min(w - 1, max(cols) + pad)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _biased_center_range(lo: int, hi: int, side: str, axis_len: int) -> tuple[int, int]:
    if side == "top":
        candidate_hi = min(hi, axis_len // 2 - 1)
        if lo <= candidate_hi:
            return lo, candidate_hi
    elif side == "bottom":
        candidate_lo = max(lo, axis_len // 2)
        if candidate_lo <= hi:
            return candidate_lo, hi
    elif side == "left":
        candidate_hi = min(hi, axis_len // 2 - 1)
        if lo <= candidate_hi:
            return lo, candidate_hi
    else:
        candidate_lo = max(lo, axis_len // 2)
        if candidate_lo <= hi:
            return candidate_lo, hi
    return lo, hi


def _sample_profile(length: int, force_idx: int, wing_idx: int) -> list[int]:
    profile = [random.choice(PROFILE_VALUES)]
    for _ in range(1, length):
        prev = profile[-1]
        candidates = [value for value in PROFILE_VALUES if abs(value - prev) <= 2]
        profile.append(random.choice(candidates))

    profile[force_idx] = max(profile[force_idx], 5)
    profile[wing_idx] = max(profile[wing_idx], 3)
    if len(set(profile)) == 1:
        idx = random.randrange(length)
        alternatives = [value for value in PROFILE_VALUES if value != profile[idx]]
        if alternatives:
            profile[idx] = random.choice(alternatives)
            profile[force_idx] = max(profile[force_idx], 5)
            profile[wing_idx] = max(profile[wing_idx], 3)
    return profile


def _make_object(
    side: str,
    wall_range: tuple[int, int],
    body_color: int,
    arrow_color: int,
    shape: tuple[int, int],
):
    h, w = shape
    wing_sign = random.choice((-1, 1))

    if side in ("top", "bottom"):
        max_half = max(PROFILE_VALUES) // 2
        usable_start = wall_range[0] + max_half
        usable_end = wall_range[1] - max_half
        if usable_end - usable_start + 1 < 4:
            return None

        width = random.randint(4, min(7, usable_end - usable_start + 1))
        valid_shafts = [idx for idx in range(width) if 0 <= idx + wing_sign < width]
        if not valid_shafts:
            return None
        shaft_idx = random.choice(valid_shafts)
        wing_idx = shaft_idx + wing_sign
        profile = _sample_profile(width, shaft_idx, wing_idx)
        half = max(profile) // 2
        col0_hi = usable_end - width + 1
        if usable_start > col0_hi:
            return None
        col0 = random.randint(usable_start, col0_hi)

        row_lo = 2 + half
        row_hi = h - 3 - half
        row_lo, row_hi = _biased_center_range(row_lo, row_hi, side, h)
        if row_lo > row_hi:
            return None
        center_row = random.randint(row_lo, row_hi)

        body = set()
        for offset, span in enumerate(profile):
            col = col0 + offset
            span_half = span // 2
            for row in range(center_row - span_half, center_row + span_half + 1):
                body.add((row, col))

        shaft_col = col0 + shaft_idx
        wing_col = col0 + wing_idx
        if side == "top":
            arrow = {(center_row, shaft_col), (center_row - 1, shaft_col), (center_row, wing_col)}
        else:
            arrow = {(center_row, shaft_col), (center_row + 1, shaft_col), (center_row, wing_col)}
    else:
        max_half = max(PROFILE_VALUES) // 2
        usable_start = wall_range[0] + max_half
        usable_end = wall_range[1] - max_half
        if usable_end - usable_start + 1 < 4:
            return None

        height = random.randint(4, min(7, usable_end - usable_start + 1))
        valid_shafts = [idx for idx in range(height) if 0 <= idx + wing_sign < height]
        if not valid_shafts:
            return None
        shaft_idx = random.choice(valid_shafts)
        wing_idx = shaft_idx + wing_sign
        profile = _sample_profile(height, shaft_idx, wing_idx)
        half = max(profile) // 2
        row0_hi = usable_end - height + 1
        if usable_start > row0_hi:
            return None
        row0 = random.randint(usable_start, row0_hi)

        col_lo = 2 + half
        col_hi = w - 3 - half
        col_lo, col_hi = _biased_center_range(col_lo, col_hi, side, w)
        if col_lo > col_hi:
            return None
        center_col = random.randint(col_lo, col_hi)

        body = set()
        for offset, span in enumerate(profile):
            row = row0 + offset
            span_half = span // 2
            for col in range(center_col - span_half, center_col + span_half + 1):
                body.add((row, col))

        shaft_row = row0 + shaft_idx
        wing_row = row0 + wing_idx
        if side == "left":
            arrow = {(shaft_row, center_col), (shaft_row, center_col - 1), (wing_row, center_col)}
        else:
            arrow = {(shaft_row, center_col), (shaft_row, center_col + 1), (wing_row, center_col)}

    if not arrow <= body:
        return None

    colored_cells = {}
    for cell in body - arrow:
        colored_cells[cell] = body_color
    for cell in arrow:
        colored_cells[cell] = arrow_color

    if not colored_cells:
        return None
    return colored_cells


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(COLORS)
    palette = [color for color in COLORS if color != bg]
    random.shuffle(palette)

    wall_colors = {side: palette[idx] for idx, side in enumerate(SIDES)}
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    chosen_sides = random.sample(SIDES, n_objects)
    body_colors = {side: palette[4 + idx] for idx, side in enumerate(chosen_sides)}

    grid = np.full((h, w), bg, dtype=int)
    try:
        wall_ranges = _sample_wall_ranges(h, w)
    except ValueError:
        return None
    wall_cells = _paint_walls(grid, wall_ranges, wall_colors)
    blocked = set(wall_cells)

    placement_order = chosen_sides[:]
    random.shuffle(placement_order)
    for side in placement_order:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            obj = _make_object(side, wall_ranges[side], body_colors[side], wall_colors[side], grid.shape)
            if obj is None:
                continue
            cells = set(obj)
            halo = _expanded_bbox(cells, grid.shape, pad=1)
            if halo & blocked:
                continue
            for (r, c), color in obj.items():
                grid[r, c] = color
            blocked.update(halo)
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

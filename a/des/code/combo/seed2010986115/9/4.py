from __future__ import annotations

import random

import numpy as np


BLACK = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 4)
LINE_LENGTH_RANGE = (3, 5)
PAIR_PLACEMENT_ATTEMPTS = 500
GENERATION_ATTEMPTS = 300

NO_HOLE_FALLBACKS = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
]
HOLE_MASK = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, r0, c0


def _hole_relative(mask: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if r not in {0, h - 1} and c not in {0, w - 1}:
                continue
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        cr, cc = stack.pop()
        for dr, dc in DIRS4:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    holes = [(r, c) for r in range(h) for c in range(w) if empty[r, c] and not seen[r, c]]
    if not holes:
        return None
    assert len(holes) == 1, "expected at most one enclosed hole"
    return holes[0]


def _line_axis(coords: list[tuple[int, int]]) -> tuple[str, int] | None:
    if not (3 <= len(coords) <= 5):
        return None
    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)

    if len(set(rows)) == 1:
        c0 = cols[0]
        if cols == list(range(c0, c0 + len(cols))):
            return ("horizontal", rows[0])
    if len(set(cols)) == 1:
        r0 = rows[0]
        if rows == list(range(r0, r0 + len(rows))):
            return ("vertical", cols[0])
    return None


def _reflect_point(r: int, c: int, axis_kind: str, axis_index: int) -> tuple[int, int]:
    if axis_kind == "horizontal":
        return 2 * axis_index - r, c
    assert axis_kind == "vertical"
    return r, 2 * axis_index - c


def _copy_centroid(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return _round_half_up(sum(rows) / len(rows)), _round_half_up(sum(cols) / len(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored shape across the same-colored line segment, then carve a hole in the reflected copy.

    Input structure
    - Background is black.
    - Each foreground color appears in exactly two 4-connected components:
      one small shape and one straight horizontal or vertical line segment.
    - The line segment marks the mirror axis for the shape of the same color.

    Rule
    - Keep the original grid unchanged.
    - Mirror each shape across its same-colored line segment and draw the copy in the same color.
    - If the original shape has an enclosed one-cell hole, preserve that hole in the mirrored copy.
    - Otherwise erase the mirrored copy's rounded centroid cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _background_color(g)

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    for color in colors:
        comps = _components(g == color, connectivity=4)
        assert len(comps) == 2, "each color must have exactly one shape and one line segment"

        line_idx = None
        line_axis = None
        for idx, comp in enumerate(comps):
            axis = _line_axis(comp)
            if axis is not None:
                assert line_idx is None, "line segment must be unique"
                line_idx = idx
                line_axis = axis
        assert line_idx is not None and line_axis is not None, "expected one straight mirror segment per color"

        shape_coords = comps[1 - line_idx]
        axis_kind, axis_index = line_axis

        copy_coords = []
        for r, c in shape_coords:
            rr, cc = _reflect_point(r, c, axis_kind, axis_index)
            assert 0 <= rr < h and 0 <= cc < w, "reflected shape must remain inside the grid"
            copy_coords.append((rr, cc))

        for rr, cc in copy_coords:
            assert int(out[rr, cc]) in {bg, color}, "reflected copy cannot overwrite another foreground color"
            out[rr, cc] = color

        shape_mask, r0, c0 = _component_mask(shape_coords)
        hole_rel = _hole_relative(shape_mask)
        if hole_rel is not None:
            hr, hc = hole_rel
            rr, cc = _reflect_point(r0 + hr, c0 + hc, axis_kind, axis_index)
            out[rr, cc] = bg
        else:
            rr, cc = _copy_centroid(copy_coords)
            out[rr, cc] = bg

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randrange(4))
    if random.choice([False, True]):
        out = np.fliplr(out)
    return _trim_mask(out)


def _generate_no_hole_mask() -> np.ndarray:
    for _ in range(120):
        raw_h = random.randint(2, 4)
        raw_w = random.randint(2, 4)
        area = random.randint(3, min(8, raw_h * raw_w))

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        start = (random.randrange(raw_h), random.randrange(raw_w))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        if int(mask.sum()) != area:
            continue

        mask = _random_transform(mask)
        if min(mask.shape) < 2:
            continue
        if _hole_relative(mask) is not None:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if _line_axis(coords) is not None:
            continue

        cr, cc = _copy_centroid(coords)
        if not mask[cr, cc]:
            continue
        return mask.copy()

    fallback = _random_transform(random.choice(NO_HOLE_FALLBACKS))
    return fallback.copy()


def _shape_mask(with_hole: bool) -> np.ndarray:
    if with_hole:
        return HOLE_MASK.copy()
    return _generate_no_hole_mask()


def _offset_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(r + top), int(c + left)) for r, c in np.argwhere(mask)]


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _cells_clear(blocked: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _hole_or_centroid(mask: np.ndarray, shape_cells: list[tuple[int, int]], axis_kind: str, axis_index: int) -> tuple[int, int]:
    hole_rel = _hole_relative(mask)
    if hole_rel is not None:
        hr, hc = hole_rel
        return _reflect_point(hr, hc, axis_kind, axis_index)  # placeholder, fixed by caller
    copy_cells = [_reflect_point(r, c, axis_kind, axis_index) for r, c in shape_cells]
    return _copy_centroid(copy_cells)


def _sample_horizontal_position(grid_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[int, int, str] | None:
    h, _w = grid_shape
    sh, _sw = mask_shape
    axis = random.randint(2, h - 3)
    side = random.choice(["above", "below"])
    if side == "above":
        lo = max(0, 2 * axis - (h - 1))
        hi = axis - sh - 1
    else:
        lo = axis + 2
        hi = min(h - sh, 2 * axis + sh - 1)
    if lo > hi:
        return None
    return random.randint(lo, hi), axis, side


def _sample_vertical_position(grid_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[int, int, str] | None:
    _h, w = grid_shape
    _sh, sw = mask_shape
    axis = random.randint(2, w - 3)
    side = random.choice(["left", "right"])
    if side == "left":
        lo = max(0, 2 * axis - (w - 1))
        hi = axis - sw - 1
    else:
        lo = axis + 2
        hi = min(w - sw, 2 * axis + sw - 1)
    if lo > hi:
        return None
    return random.randint(lo, hi), axis, side


def _build_pair(
    grid_shape: tuple[int, int],
    mask: np.ndarray,
    color: int,
) -> dict[str, object] | None:
    h, w = grid_shape
    sh, sw = mask.shape
    axis_kind = random.choice(["horizontal", "vertical"])

    if axis_kind == "horizontal":
        sample = _sample_horizontal_position(grid_shape, mask.shape)
        if sample is None:
            return None
        top, axis_index, _side = sample
        left = random.randint(0, w - sw)
        line_len = random.randint(*LINE_LENGTH_RANGE)
        line_left = random.randint(0, w - line_len)
        shape_cells = _offset_coords(mask, top, left)
        line_cells = {(axis_index, c) for c in range(line_left, line_left + line_len)}
    else:
        sample = _sample_vertical_position(grid_shape, mask.shape)
        if sample is None:
            return None
        left, axis_index, _side = sample
        top = random.randint(0, h - sh)
        line_len = random.randint(*LINE_LENGTH_RANGE)
        line_top = random.randint(0, h - line_len)
        shape_cells = _offset_coords(mask, top, left)
        line_cells = {(r, axis_index) for r in range(line_top, line_top + line_len)}

    shape_set = set(shape_cells)
    if shape_set & line_cells:
        return None

    copy_full = {_reflect_point(r, c, axis_kind, axis_index) for r, c in shape_cells}
    if len(copy_full) != len(shape_cells):
        return None

    if _hole_relative(mask) is not None:
        hole_r, hole_c = _hole_relative(mask)
        erase = _reflect_point(top + hole_r, left + hole_c, axis_kind, axis_index)
    else:
        erase = _copy_centroid(list(copy_full))
        if erase not in copy_full:
            return None

    if erase in shape_set or erase in line_cells:
        return None

    reserve_cells = set(line_cells) | shape_set | copy_full | {erase}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in reserve_cells):
        return None

    final_copy = set(copy_full)
    final_copy.discard(erase)
    if not final_copy:
        return None

    return {
        "color": int(color),
        "shape": shape_set,
        "line": set(line_cells),
        "copy_full": set(copy_full),
        "erase": erase,
        "reserve": reserve_cells,
        "painted": set(line_cells) | shape_set | final_copy,
    }


def _render_expected(grid: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["copy_full"]:
            out[r, c] = color
        rr, cc = pair["erase"]
        out[rr, cc] = BLACK
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    if h < 8 or w < 8:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    blocked_input = np.zeros((h, w), dtype=bool)
    blocked_output = np.zeros((h, w), dtype=bool)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(list(range(1, 10)), object_count)
    hole_flags = [random.choice([False, True]) for _ in range(object_count)]
    if object_count >= 2:
        if all(hole_flags):
            hole_flags[random.randrange(object_count)] = False
        if not any(hole_flags):
            hole_flags[random.randrange(object_count)] = True

    pairs: list[dict[str, object]] = []
    for color, with_hole in zip(colors, hole_flags):
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            mask = _shape_mask(with_hole)
            pair = _build_pair((h, w), mask, color)
            if pair is None:
                continue
            if not _cells_clear(blocked_input, pair["shape"] | pair["line"]):
                continue
            if not _cells_clear(blocked_output, pair["reserve"]):
                continue

            for r, c in pair["shape"] | pair["line"]:
                grid[r, c] = color

            _mark_blocked(blocked_input, pair["shape"] | pair["line"])
            _mark_blocked(blocked_output, pair["reserve"])
            pairs.append(pair)
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-and-hole puzzle")

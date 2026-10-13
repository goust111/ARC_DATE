from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _grey_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected non-background pixels"
    multi = [color for color, count in counts.items() if count > 1]
    if multi:
        return max(multi, key=lambda color: counts[color])
    return max(counts, key=lambda color: counts[color])


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not empty[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if empty[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _component_hole_count(points: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(points)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        sub[r - r0, c - c0] = True
    return _hole_count(sub)


def _legend_coords(grid: np.ndarray, bg: int, grey: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    coords = [tuple(map(int, p)) for p in np.argwhere((grid != bg) & (grid != grey))]
    assert coords, "expected a color legend on the grid edge"

    if all(r == 0 for r, _c in coords):
        ordered = sorted(coords, key=lambda p: p[1])
        cols = [c for _r, c in ordered]
        assert cols == list(range(cols[0], cols[-1] + 1))
        return ordered
    if all(r == h - 1 for r, _c in coords):
        ordered = sorted(coords, key=lambda p: p[1])
        cols = [c for _r, c in ordered]
        assert cols == list(range(cols[0], cols[-1] + 1))
        return ordered
    if all(c == 0 for r, c in coords):
        ordered = sorted(coords, key=lambda p: p[0])
        rows = [r for r, _c in ordered]
        assert rows == list(range(rows[0], rows[-1] + 1))
        return ordered
    if all(c == w - 1 for r, c in coords):
        ordered = sorted(coords, key=lambda p: p[0])
        rows = [r for r, _c in ordered]
        assert rows == list(range(rows[0], rows[-1] + 1))
        return ordered

    raise AssertionError("legend pixels must form one contiguous edge run")


def _legend_mapping(grid: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    coords = _legend_coords(grid, bg, grey)
    return {idx + 1: int(grid[r, c]) for idx, (r, c) in enumerate(coords)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey contour component using the legend color indexed by its hole count.
    Components with zero holes remain grey, and legend pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    grey = _grey_color(g, bg)
    legend = _legend_mapping(g, bg, grey)

    out = g.copy()
    for comp in _components(g == grey):
        holes = _component_hole_count(comp)
        if holes == 0:
            continue
        color = legend.get(holes)
        if color is None:
            continue
        for r, c in comp:
            out[r, c] = int(color)
    return out


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_MIN = 16
GRID_MAX = 30
LEGEND_LEN_RANGE = (2, 4)
GEN_ATTEMPTS = 400
SPRITE_ATTEMPTS = 120
PLACE_ATTEMPTS = 200


def _ring_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _exterior_empty(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return exterior


def _one_neighbor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    exterior = _exterior_empty(mask)
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or not exterior[r, c]:
                continue
            deg = sum(1 for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc])
            if deg == 1:
                candidates.append((r, c))
    return candidates


def _add_spikes(mask: np.ndarray, n_cells: int) -> None:
    for _ in range(n_cells):
        candidates = _one_neighbor_candidates(mask)
        if not candidates:
            return
        r, c = random.choice(candidates)
        mask[r, c] = True


def _make_tree_shape() -> np.ndarray | None:
    for _ in range(SPRITE_ATTEMPTS):
        height = random.randint(6, 9)
        width = random.randint(6, 9)
        target = random.randint(5, 11)
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randint(2, height - 3), random.randint(2, width - 3))
        mask[start] = True
        while int(mask.sum()) < target:
            candidates = _one_neighbor_candidates(mask)
            if not candidates:
                break
            r, c = random.choice(candidates)
            mask[r, c] = True
        if int(mask.sum()) < target:
            continue
        _add_spikes(mask, random.randint(0, 2))
        cropped = _crop_mask(mask)
        if _hole_count(cropped) == 0 and int(cropped.sum()) >= 5:
            return cropped
    return None


def _make_holey_shape(hole_count: int) -> np.ndarray | None:
    assert hole_count >= 1
    for _ in range(SPRITE_ATTEMPTS):
        orientation = random.choice(["horizontal", "vertical"])
        rings = [(random.randint(3, 4), random.randint(3, 4)) for _ in range(hole_count)]
        gaps = [random.randint(1, 2) for _ in range(hole_count - 1)]

        if orientation == "horizontal":
            canvas_h = max(h for h, _w in rings) + 4
            canvas_w = sum(w for _h, w in rings) + sum(gaps) + 2
            mask = np.zeros((canvas_h, canvas_w), dtype=bool)
            center = canvas_h // 2
            left = 1
            prev_top = prev_h = prev_right = None
            ok = True
            for idx, (height, width) in enumerate(rings):
                top_lo = max(1, center - (height - 2))
                top_hi = min(canvas_h - height - 1, center - 1)
                if top_lo > top_hi:
                    ok = False
                    break
                top = random.randint(top_lo, top_hi)
                mask[top : top + height, left : left + width] |= _ring_mask(height, width)
                if prev_right is not None:
                    row_lo = max(prev_top + 1, top + 1)
                    row_hi = min(prev_top + prev_h - 2, top + height - 2)
                    if row_lo > row_hi:
                        ok = False
                        break
                    bridge_row = random.randint(row_lo, row_hi)
                    mask[bridge_row, prev_right : left + 1] = True
                prev_top, prev_h, prev_right = top, height, left + width - 1
                left += width
                if idx < len(gaps):
                    left += gaps[idx]
        else:
            canvas_h = sum(h for h, _w in rings) + sum(gaps) + 2
            canvas_w = max(w for _h, w in rings) + 4
            mask = np.zeros((canvas_h, canvas_w), dtype=bool)
            center = canvas_w // 2
            top = 1
            prev_left = prev_w = prev_bottom = None
            ok = True
            for idx, (height, width) in enumerate(rings):
                left_lo = max(1, center - (width - 2))
                left_hi = min(canvas_w - width - 1, center - 1)
                if left_lo > left_hi:
                    ok = False
                    break
                left = random.randint(left_lo, left_hi)
                mask[top : top + height, left : left + width] |= _ring_mask(height, width)
                if prev_bottom is not None:
                    col_lo = max(prev_left + 1, left + 1)
                    col_hi = min(prev_left + prev_w - 2, left + width - 2)
                    if col_lo > col_hi:
                        ok = False
                        break
                    bridge_col = random.randint(col_lo, col_hi)
                    mask[prev_bottom : top + 1, bridge_col] = True
                prev_left, prev_w, prev_bottom = left, width, top + height - 1
                top += height
                if idx < len(gaps):
                    top += gaps[idx]

        if not ok:
            continue

        _add_spikes(mask, random.randint(0, hole_count + 1))
        cropped = _crop_mask(mask)
        if _hole_count(cropped) == hole_count:
            return cropped
    return None


def _make_component(hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        return _make_tree_shape()
    return _make_holey_shape(hole_count)


def _sample_hole_counts(max_holes: int) -> list[int]:
    mode = random.choice(["mixed", "positive", "repeat"])
    if mode == "mixed":
        n_objects = random.randint(2, 4)
        counts = [random.randint(0, max_holes) for _ in range(n_objects)]
        if all(count == 0 for count in counts):
            counts[random.randrange(n_objects)] = random.randint(1, max_holes)
        if 0 not in counts and random.random() < 0.7:
            counts[random.randrange(n_objects)] = 0
        if len({count for count in counts if count > 0}) < 2 and n_objects >= 2:
            counts[0] = random.randint(1, max_holes)
            counts[1] = random.randint(1, max_holes)
        return counts
    if mode == "positive":
        n_objects = random.randint(2, 4)
        counts = [random.randint(1, max_holes) for _ in range(n_objects)]
        if len(set(counts)) == 1 and n_objects >= 2 and max_holes >= 2:
            counts[0] = random.randint(1, max_holes)
            counts[1] = random.randint(1, max_holes)
        return counts

    n_objects = random.randint(2, 3)
    base = random.randint(1, max_holes)
    counts = [base] * n_objects
    if random.random() < 0.6:
        counts[random.randrange(n_objects)] = 0
    return counts


def _legend_side() -> str:
    return random.choice(["top", "bottom", "left", "right"])


def _shape_limits(side: str, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    r0, r1 = 1, grid_h - 2
    c0, c1 = 1, grid_w - 2
    if side == "top":
        r0 = 2
    elif side == "bottom":
        r1 = grid_h - 3
    elif side == "left":
        c0 = 2
    else:
        c1 = grid_w - 3
    return r0, r1, c0, c1


def _place_legend(grid: np.ndarray, side: str, colors: list[int]) -> None:
    h, w = grid.shape
    n = len(colors)
    if side == "top":
        start = random.randint(1, w - n - 1)
        for i, color in enumerate(colors):
            grid[0, start + i] = int(color)
    elif side == "bottom":
        start = random.randint(1, w - n - 1)
        for i, color in enumerate(colors):
            grid[h - 1, start + i] = int(color)
    elif side == "left":
        start = random.randint(1, h - n - 1)
        for i, color in enumerate(colors):
            grid[start + i, 0] = int(color)
    else:
        start = random.randint(1, h - n - 1)
        for i, color in enumerate(colors):
            grid[start + i, w - 1] = int(color)


def _can_place(blocked: np.ndarray, top: int, left: int, shape: np.ndarray) -> bool:
    h, w = blocked.shape
    sh, sw = shape.shape
    rr0 = max(0, top - 1)
    rr1 = min(h, top + sh + 1)
    cc0 = max(0, left - 1)
    cc1 = min(w, left + sw + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _place_components(
    grid: np.ndarray, side: str, shapes: list[tuple[np.ndarray, int]]
) -> list[tuple[int, int, np.ndarray, int]] | None:
    grid_h, grid_w = grid.shape
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int, np.ndarray, int]] = []
    r0, r1, c0, c1 = _shape_limits(side, grid_h, grid_w)

    for shape, holes in sorted(shapes, key=lambda item: int(item[0].sum()), reverse=True):
        sh, sw = shape.shape
        row_choices = list(range(r0, r1 - sh + 2))
        col_choices = list(range(c0, c1 - sw + 2))
        if not row_choices or not col_choices:
            return None
        positions = [(top, left) for top in row_choices for left in col_choices]
        random.shuffle(positions)
        placed = False
        for top, left in positions[:PLACE_ATTEMPTS]:
            if not _can_place(blocked, top, left, shape):
                continue
            grid[top : top + sh, left : left + sw][shape] = GREY
            rr0 = max(0, top - 1)
            rr1 = min(grid_h, top + sh + 1)
            cc0 = max(0, left - 1)
            cc1 = min(grid_w, left + sw + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            placements.append((top, left, shape, holes))
            placed = True
            break
        if not placed:
            return None
    return placements


def _expected_output(
    grid: np.ndarray, placements: list[tuple[int, int, np.ndarray, int]], legend_colors: list[int]
) -> np.ndarray:
    out = grid.copy()
    for top, left, shape, holes in placements:
        if holes == 0:
            continue
        color = int(legend_colors[holes - 1])
        out[top : top + shape.shape[0], left : left + shape.shape[1]][shape] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend_colors = random.sample(PALETTE, legend_len)
    hole_counts = _sample_hole_counts(legend_len)

    shapes: list[tuple[np.ndarray, int]] = []
    for holes in hole_counts:
        shape = _make_component(holes)
        if shape is None:
            return None
        if _hole_count(shape) != holes:
            return None
        shapes.append((shape, holes))

    max_h = max(shape.shape[0] for shape, _holes in shapes)
    max_w = max(shape.shape[1] for shape, _holes in shapes)
    total_area = sum((shape.shape[0] + 2) * (shape.shape[1] + 2) for shape, _holes in shapes)

    side = _legend_side()
    min_h = max(GRID_MIN, max_h + 4, int(np.ceil(np.sqrt(total_area))) + 4)
    min_w = max(GRID_MIN, max_w + 4, int(np.ceil(np.sqrt(total_area))) + 4)
    if side in ("top", "bottom"):
        min_w = max(min_w, legend_len + 2)
    else:
        min_h = max(min_h, legend_len + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    _place_legend(grid, side, legend_colors)
    placements = _place_components(grid, side, shapes)
    if placements is None:
        return None

    expected = _expected_output(grid, placements, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

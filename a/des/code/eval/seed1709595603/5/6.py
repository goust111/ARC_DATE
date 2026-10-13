from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))

LEAF_MIN_SPAN = 5
MAX_DEPTH = 4
TOP_LEVEL_MARGIN = 2
OUTER_SIZE_EXTRA = 6
CLEARANCE = 1
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _flood_from_border(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not bool(blocked[r, c]) and not bool(visited[r, c]):
                visited[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(blocked[r, c]) and not bool(visited[r, c]):
                visited[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not bool(blocked[nr, nc]) and not bool(visited[nr, nc]):
                visited[nr, nc] = True
                q.append((nr, nc))

    return visited


def _strip_info(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    coords = np.argwhere((g != BLACK) & (g != GREY))
    assert coords.shape[0] > 0, "expected a non-empty edge palette"
    assert np.all(
        (coords[:, 0] == 0)
        | (coords[:, 0] == h - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == w - 1)
    ), "palette pixels must all lie on the grid edge"

    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})

    ordered_coords: list[tuple[int, int]]
    if len(rows) == 1:
        r = rows[0]
        positions = sorted(int(c) for c in coords[:, 1])
        assert positions == list(range(positions[0], positions[-1] + 1)), "palette must be contiguous"
        left_dist = min(positions[0], w - 1 - positions[0])
        right_dist = min(positions[-1], w - 1 - positions[-1])
        if left_dist <= right_dist:
            ordered_coords = [(r, c) for c in positions]
        else:
            ordered_coords = [(r, c) for c in reversed(positions)]
    else:
        assert len(cols) == 1, "palette must occupy a single border row or column"
        c = cols[0]
        positions = sorted(int(r) for r in coords[:, 0])
        assert positions == list(range(positions[0], positions[-1] + 1)), "palette must be contiguous"
        top_dist = min(positions[0], h - 1 - positions[0])
        bottom_dist = min(positions[-1], h - 1 - positions[-1])
        if top_dist <= bottom_dist:
            ordered_coords = [(r, c) for r in positions]
        else:
            ordered_coords = [(r, c) for r in reversed(positions)]

    colors = [int(g[r, c]) for r, c in ordered_coords]
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    return ordered_coords, colors


def _inside_mask_for_component(shape: tuple[int, int], coords: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "shapes must stay off the outer border"

    top = r0 - 1
    left = c0 - 1
    bottom = r1 + 1
    right = c1 + 1

    blocked = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
    for r, c in coords:
        blocked[int(r) - top, int(c) - left] = True

    outside = _flood_from_border(blocked)
    inside = (~blocked) & (~outside)
    assert inside.any(), "each grey component must be a closed shape"
    return (top, left), inside


def _component_depths(shape: tuple[int, int], comps: list[np.ndarray]) -> list[int]:
    inside_data = [_inside_mask_for_component(shape, comp) for comp in comps]
    depths = [0] * len(comps)

    for outer_index, (origin, inside) in enumerate(inside_data):
        top, left = origin
        ih, iw = inside.shape
        for inner_index, comp in enumerate(comps):
            if inner_index == outer_index:
                continue
            contained = False
            for r, c in comp:
                rr = int(r) - top
                cc = int(c) - left
                if 0 <= rr < ih and 0 <= cc < iw and bool(inside[rr, cc]):
                    contained = True
                    break
            if contained:
                depths[inner_index] += 1

    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed grey outline by its nesting depth using the edge palette.

    The palette is read from the contiguous non-grey, non-black segment on one
    border, starting from the end that is closest to a corner. The k-th palette
    color is assigned to shapes contained in exactly k other shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_coords, palette_colors = _strip_info(g)
    comps = _components(g == GREY)
    assert comps, "expected at least one grey closed shape"

    depths = _component_depths(g.shape, comps)
    assert max(depths) < len(palette_colors), "palette must cover every nesting depth"

    out = g.copy()
    for r, c in palette_coords:
        out[r, c] = BLACK
    for comp, depth in zip(comps, depths):
        out[comp[:, 0], comp[:, 1]] = palette_colors[depth]

    return out


def _min_span(depth: int) -> int:
    return LEAF_MIN_SPAN + 4 * depth


def _expand_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_free(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _choose_chain_sizes(depth: int, h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    sizes: list[tuple[int, int]] = []
    outer_min = _min_span(depth)
    outer_h_max = min(h - 2 * TOP_LEVEL_MARGIN, outer_min + OUTER_SIZE_EXTRA)
    outer_w_max = min(w - 2 * TOP_LEVEL_MARGIN, outer_min + OUTER_SIZE_EXTRA)
    if outer_h_max < outer_min or outer_w_max < outer_min:
        return None

    current_h = rng.randint(outer_min, outer_h_max)
    current_w = rng.randint(outer_min, outer_w_max)
    sizes.append((current_h, current_w))

    remaining = depth
    while remaining > 0:
        child_depth = remaining - 1
        child_min = _min_span(child_depth)
        child_h_max = min(current_h - 4, child_min + OUTER_SIZE_EXTRA)
        child_w_max = min(current_w - 4, child_min + OUTER_SIZE_EXTRA)
        if child_h_max < child_min or child_w_max < child_min:
            return None
        current_h = rng.randint(child_min, child_h_max)
        current_w = rng.randint(child_min, child_w_max)
        sizes.append((current_h, current_w))
        remaining -= 1

    return sizes


def _instantiate_chain(
    sizes: list[tuple[int, int]],
    outer_r0: int,
    outer_c0: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    current_r0 = outer_r0
    current_c0 = outer_c0

    for index, (height, width) in enumerate(sizes):
        box = (current_r0, current_c0, current_r0 + height - 1, current_c0 + width - 1)
        boxes.append(box)
        if index == len(sizes) - 1:
            break

        child_h, child_w = sizes[index + 1]
        row_min = current_r0 + 2
        row_max = current_r0 + height - child_h - 2
        col_min = current_c0 + 2
        col_max = current_c0 + width - child_w - 2
        if row_min > row_max or col_min > col_max:
            return None
        current_r0 = rng.randint(row_min, row_max)
        current_c0 = rng.randint(col_min, col_max)

    return boxes


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_palette(grid: np.ndarray, colors: list[int], side: str, corner_gap: int) -> None:
    h, w = grid.shape
    length = len(colors)

    if side == "top":
        for offset, color in enumerate(colors):
            grid[0, corner_gap + offset] = color
        return
    if side == "bottom":
        for offset, color in enumerate(colors):
            grid[h - 1, corner_gap + offset] = color
        return
    if side == "right":
        for offset, color in enumerate(colors):
            grid[corner_gap + offset, w - 1] = color
        return
    for offset, color in enumerate(colors):
        grid[corner_gap + offset, 0] = color


def _place_palette_from_corner(
    grid: np.ndarray,
    colors: list[int],
    edge: str,
    near_first_corner: bool,
    rng: random.Random,
) -> None:
    h, w = grid.shape
    length = len(colors)

    if edge in ("top", "bottom"):
        max_gap = max(0, min(2, w - length - 1))
        gap = rng.randint(0, max_gap)
        if near_first_corner:
            _place_palette(grid, colors, edge, gap)
        else:
            row = 0 if edge == "top" else h - 1
            start = w - 1 - gap
            for offset, color in enumerate(colors):
                grid[row, start - offset] = color
        return

    max_gap = max(0, min(2, h - length - 1))
    gap = rng.randint(0, max_gap)
    if near_first_corner:
        _place_palette(grid, colors, edge, gap)
    else:
        col = 0 if edge == "left" else w - 1
        start = h - 1 - gap
        for offset, color in enumerate(colors):
            grid[start - offset, col] = color


def _validate_generated(grid: np.ndarray) -> bool:
    palette_coords, palette_colors = _strip_info(grid)
    comps = _components(grid == GREY)
    depths = _component_depths(grid.shape, comps)
    if len(comps) < 2 or max(depths) < 1 or max(depths) >= len(palette_colors):
        return False
    out = main(grid)
    if np.any(out == GREY):
        return False
    return all(int(out[r, c]) == BLACK for r, c in palette_coords)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_depth = rng.randint(1, MAX_DEPTH)
    min_dim = 9 + 4 * max_depth
    h = rng.randint(min_dim, 32)
    w = rng.randint(min_dim, 32)

    palette_len = max_depth + 1 + rng.randint(0, 2)
    if max(h, w) <= palette_len:
        return None
    palette_colors = rng.sample(PALETTE_COLORS, palette_len)

    if max_depth >= 3:
        chain_count = rng.randint(1, 2)
    else:
        chain_count = rng.randint(1, 3)
    chain_depths = [max_depth] + [rng.randint(0, max_depth) for _ in range(chain_count - 1)]

    chain_specs: list[dict[str, object]] = []
    for depth in chain_depths:
        sizes = _choose_chain_sizes(depth, h, w, rng)
        if sizes is None:
            return None
        outer_h, outer_w = sizes[0]
        chain_specs.append(
            {
                "depth": depth,
                "sizes": sizes,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "area": outer_h * outer_w,
            }
        )

    chain_specs.sort(key=lambda spec: int(spec["area"]), reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    all_boxes: list[tuple[int, int, int, int]] = []

    for spec in chain_specs:
        sizes = spec["sizes"]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        row_min = TOP_LEVEL_MARGIN
        row_max = h - TOP_LEVEL_MARGIN - outer_h
        col_min = TOP_LEVEL_MARGIN
        col_max = w - TOP_LEVEL_MARGIN - outer_w
        if row_min > row_max or col_min > col_max:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            outer_box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            reserve = _expand_box(outer_box, CLEARANCE, (h, w))
            if not _box_is_free(occupied, reserve):
                continue
            boxes = _instantiate_chain(sizes, r0, c0, rng)
            if boxes is None:
                continue
            _mark_box(occupied, reserve)
            all_boxes.extend(boxes)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BLACK, dtype=int)
    for box in all_boxes:
        _draw_ring(grid, box, GREY)

    edge = rng.choice(("top", "bottom", "left", "right"))
    near_first_corner = bool(rng.randint(0, 1))
    _place_palette_from_corner(grid, palette_colors, edge, near_first_corner, rng)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

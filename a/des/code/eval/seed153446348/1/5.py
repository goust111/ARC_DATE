from __future__ import annotations

import random

import numpy as np


def _infer_frame_and_checker_colors(grid: np.ndarray) -> tuple[int, int, int]:
    colors = [int(color) for color in np.unique(grid)]
    assert len(colors) == 3, "expected exactly two checkerboard colors and one frame color"

    parity = np.indices(grid.shape).sum(axis=0) % 2
    for frame_color in colors:
        even_vals = np.unique(grid[(parity == 0) & (grid != frame_color)])
        odd_vals = np.unique(grid[(parity == 1) & (grid != frame_color)])
        if even_vals.size == 1 and odd_vals.size == 1 and int(even_vals[0]) != int(odd_vals[0]):
            return int(frame_color), int(even_vals[0]), int(odd_vals[0])

    raise AssertionError("could not identify frame color against the checkerboard background")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    sub = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True

    expected = np.zeros_like(sub)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _collect_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == int(frame_color)):
        assert _is_rect_frame(coords), "frame color must form only 1-pixel rectangular frames"
        frames.append(_bbox(coords))
    assert frames, "expected at least one frame"
    return frames


def _enclosure_depth(shape: tuple[int, int], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    depth = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in frames:
        depth[r0 + 1 : r1, c0 + 1 : c1] += 1
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangular frames on a checkerboard and flip the checkerboard phase
    inside odd enclosure depths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, even_color, odd_color = _infer_frame_and_checker_colors(g)
    frames = _collect_frames(g, frame_color)
    depth = _enclosure_depth(g.shape, frames)

    out = g.copy()
    toggle = (g != frame_color) & (depth % 2 == 1)
    out[toggle & (g == even_color)] = odd_color
    out[toggle & (g == odd_color)] = even_color
    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_GRID_MIN = 20
_GRID_MAX = 30
_BORDER = 1
_PLACEMENT_PADDING = 1
_MAX_GENERATION_ATTEMPTS = 400


def _checkerboard(rows: int, cols: int, even_color: int, odd_color: int) -> np.ndarray:
    parity = np.indices((rows, cols)).sum(axis=0) % 2
    grid = np.empty((rows, cols), dtype=int)
    grid[parity == 0] = int(even_color)
    grid[parity == 1] = int(odd_color)
    return grid


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _min_outer_size(depth: int) -> int:
    return 3 + 4 * (depth - 1)


def _sample_chain(depth: int, max_h: int, max_w: int, *, bonus: int) -> list[tuple[int, int, int, int]] | None:
    min_outer = _min_outer_size(depth)
    if max_h < min_outer or max_w < min_outer:
        return None

    outer_h = random.randint(min_outer, min(max_h, min_outer + bonus))
    outer_w = random.randint(min_outer, min(max_w, min_outer + bonus + 2))

    boxes: list[tuple[int, int, int, int]] = []
    r0 = 0
    c0 = 0
    h = outer_h
    w = outer_w
    for level in range(depth):
        boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        if level == depth - 1:
            break

        remaining = depth - level - 1
        min_inner = _min_outer_size(remaining)
        max_margin_r = (h - min_inner) // 2
        max_margin_c = (w - min_inner) // 2
        if max_margin_r < 2 or max_margin_c < 2:
            return None

        shrink_r = random.randint(2, max_margin_r)
        shrink_c = random.randint(2, max_margin_c)
        r0 += shrink_r
        c0 += shrink_c
        h -= 2 * shrink_r
        w -= 2 * shrink_c
    return boxes


def _outer_area(chain: list[tuple[int, int, int, int]]) -> int:
    r0, c0, r1, c1 = chain[0]
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _place_chain(
    occupied: np.ndarray,
    chain: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    rows, cols = occupied.shape
    _r0, _c0, _r1, _c1 = chain[0]
    h = _r1 - _r0 + 1
    w = _c1 - _c0 + 1

    candidates: list[tuple[int, int]] = []
    max_r = rows - _BORDER - h
    max_c = cols - _BORDER - w
    for r in range(_BORDER, max_r + 1):
        for c in range(_BORDER, max_c + 1):
            r1 = r + h - 1
            c1 = c + w - 1
            er0 = max(0, r - _PLACEMENT_PADDING)
            ec0 = max(0, c - _PLACEMENT_PADDING)
            er1 = min(rows - 1, r1 + _PLACEMENT_PADDING)
            ec1 = min(cols - 1, c1 + _PLACEMENT_PADDING)
            if not occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                candidates.append((r, c))

    if not candidates:
        return None

    base_r, base_c = random.choice(candidates)
    placed = [
        (base_r + r0, base_c + c0, base_r + r1, base_c + c1)
        for r0, c0, r1, c1 in chain
    ]
    pr0, pc0, pr1, pc1 = placed[0]
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
    return placed


def _choose_shape() -> tuple[int, int]:
    rows = random.randint(_GRID_MIN, _GRID_MAX)
    cols = random.randint(_GRID_MIN, _GRID_MAX)
    if random.random() < 0.6:
        cols = min(_GRID_MAX, max(_GRID_MIN, rows + random.randint(-5, 5)))
    return rows, cols


def _sample_depths(max_depth: int) -> list[int]:
    primary_choices = [depth for depth in (2, 3, 4) if depth <= max_depth]
    primary_depth = random.choice(primary_choices)

    if primary_depth >= 4:
        extra_count = random.randint(0, 1)
    else:
        extra_count = random.randint(0, 2)

    extra_choices = [depth for depth in (1, 2, 3) if depth <= max_depth]
    depths = [primary_depth]
    for _ in range(extra_count):
        depths.append(random.choice(extra_choices))
    return depths


def _build_sample() -> np.ndarray | None:
    rows, cols = _choose_shape()
    max_depth = min(4, 1 + (min(rows, cols) - 2 * _BORDER - 3) // 4)
    if max_depth < 2:
        return None

    frame_color, color_a, color_b = random.sample(_COLORS, 3)
    even_color, odd_color = (color_a, color_b) if random.random() < 0.5 else (color_b, color_a)
    grid = _checkerboard(rows, cols, even_color, odd_color)

    depths = _sample_depths(max_depth)
    chains: list[list[tuple[int, int, int, int]]] = []
    crowded = len(depths) > 1
    for index, depth in enumerate(depths):
        bonus = 6 if index == 0 and not crowded else 4 if index == 0 else 3
        chain = _sample_chain(
            depth,
            rows - 2 * _BORDER,
            cols - 2 * _BORDER,
            bonus=bonus,
        )
        if chain is None:
            return None
        chains.append(chain)

    chains.sort(key=_outer_area, reverse=True)
    occupied = np.zeros((rows, cols), dtype=bool)
    placed_chains: list[list[tuple[int, int, int, int]]] = []
    for chain in chains:
        placed = _place_chain(occupied, chain)
        if placed is None:
            return None
        placed_chains.append(placed)

    for chain in placed_chains:
        for bbox in chain:
            _draw_frame(grid, bbox, frame_color)

    detected_frame, _even, _odd = _infer_frame_and_checker_colors(grid)
    if detected_frame != frame_color:
        return None

    frames = _collect_frames(grid, frame_color)
    if len(frames) != sum(len(chain) for chain in placed_chains):
        return None

    depth = _enclosure_depth(grid.shape, frames)
    non_frame = grid != frame_color
    enclosed = depth[non_frame]
    if enclosed.max(initial=0) < 2:
        return None
    if not np.any(enclosed == 0):
        return None
    if not np.any(enclosed % 2 == 1):
        return None
    if not np.any((enclosed > 0) & (enclosed % 2 == 0)):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _build_sample()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-frame sample")

from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "expected a grey frame"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    assert side_h == side_w and side_h >= 3 and side_h % 2 == 1, "frame must be an odd square"

    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    border = np.zeros((side_h, side_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(patch[border] == GREY), "frame border must be grey"
    assert not np.any(patch[~border] == GREY), "frame interior must not contain grey"
    assert int(grey_coords.shape[0]) == int(border.sum()), "unexpected extra grey cells"
    return int(r0), int(c0), int(r1), int(c1)


def _count_components(region: np.ndarray, color: int) -> int:
    mask = region == int(color)
    if not np.any(mask):
        return 0

    h, w = map(int, region.shape)
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _depth_map(side: int) -> np.ndarray:
    inner = side - 2
    rows, cols = np.indices((inner, inner))
    return np.minimum.reduce([rows, cols, inner - 1 - rows, inner - 1 - cols]) + 1


def _render_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    depth_to_color: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    side = int(r1 - r0 + 1)
    out = np.full(shape, BLACK, dtype=int)

    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    depth = _depth_map(side)
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    for k, color in depth_to_color.items():
        inner[depth == int(k)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each interior color's 4-connected components and redraw that color as the matching concentric depth ring.

    Input structure:
    - A black background contains one odd-sized one-cell-thick grey hollow square frame.
    - Inside the frame, several non-black non-grey colors appear as one or more 4-connected components.
    - The same colors may also appear outside the frame as distractor pixels or clusters.

    Rule:
    - Keep the grey frame fixed and erase every colored distractor outside the frame to black.
    - For each interior color, count its 4-connected component count k inside the frame only.
    - Rewrite the entire frame interior so that every cell at Chebyshev depth exactly k from the inner wall becomes that color.
    - Any depth with no matching color stays black, and the original interior shapes are not preserved.

    Key insight:
    - The meaningful signal is the number of interior connected components per color, not the original pixel count or shape, and outside same-color noise must be ignored before counting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_colors = sorted(int(v) for v in np.unique(inner) if int(v) not in {BLACK, GREY})

    depth_to_color: dict[int, int] = {}
    for color in inner_colors:
        comp_count = _count_components(inner, color)
        if comp_count == 0:
            continue
        assert comp_count not in depth_to_color, "component counts must be unique across colors"
        depth_to_color[comp_count] = int(color)

    return _render_output(g.shape, frame_bbox, depth_to_color)


def _can_start_component(region: np.ndarray, color: int, r: int, c: int) -> bool:
    if region[r, c] != BLACK:
        return False
    h, w = map(int, region.shape)
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and region[nr, nc] == int(color):
            return False
    return True


def _can_grow_component(
    region: np.ndarray,
    color: int,
    r: int,
    c: int,
    component: set[tuple[int, int]],
) -> bool:
    h, w = map(int, region.shape)
    if not (0 <= r < h and 0 <= c < w):
        return False
    if region[r, c] != BLACK or (r, c) in component:
        return False
    touches_current = False
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in component:
            touches_current = True
        elif region[nr, nc] == int(color):
            return False
    return touches_current


def _place_component(region: np.ndarray, color: int, want_size: int) -> int | None:
    starts = [(r, c) for r, c in np.argwhere(region == BLACK) if _can_start_component(region, color, int(r), int(c))]
    if not starts:
        return None

    seed = random.choice(starts)
    component = {(int(seed[0]), int(seed[1]))}
    while len(component) < int(want_size):
        growth: list[tuple[int, int]] = []
        for cr, cc in list(component):
            for nr, nc in _neighbors4(cr, cc):
                if _can_grow_component(region, color, nr, nc, component):
                    growth.append((int(nr), int(nc)))
        if not growth:
            break
        component.add(random.choice(growth))

    for r, c in component:
        region[r, c] = int(color)
    return len(component)


def _paint_random_components(
    region: np.ndarray,
    color: int,
    count: int,
    *,
    favor_multicell: bool,
) -> list[int] | None:
    sizes: list[int] = []
    for _ in range(int(count)):
        if favor_multicell and random.random() < 0.6:
            want_size = random.choice([2, 2, 3, 4])
        else:
            want_size = random.choice([1, 1, 1, 2, 3])
        placed = _place_component(region, color, want_size)
        if placed is None:
            return None
        sizes.append(int(placed))
    return sizes


def _paint_outside_distractors(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    colors: list[int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    h, w = map(int, grid.shape)
    for color in colors:
        cluster_count = random.randint(1, 3)
        placed_any = False
        for _ in range(cluster_count):
            candidates = [(r, c) for r, c in np.argwhere(outside & (grid == BLACK))]
            if not candidates:
                break
            sr, sc = random.choice(candidates)
            cluster = {(int(sr), int(sc))}
            grid[int(sr), int(sc)] = int(color)
            target_size = random.choice([1, 1, 2, 2, 3])
            while len(cluster) < target_size:
                growth: list[tuple[int, int]] = []
                for cr, cc in list(cluster):
                    for nr, nc in _neighbors4(cr, cc):
                        if 0 <= nr < h and 0 <= nc < w and outside[nr, nc] and grid[nr, nc] == BLACK:
                            growth.append((int(nr), int(nc)))
                if not growth:
                    break
                nr, nc = random.choice(growth)
                cluster.add((int(nr), int(nc)))
                grid[nr, nc] = int(color)
            placed_any = True
        assert placed_any, "expected room for at least one outside distractor per color"


def _build_input_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(list(range(7, 22, 2)))
    inner_side = side - 2
    max_depth = (inner_side + 1) // 2

    margins = [random.randint(1, 4) for _ in range(4)]
    top, bottom, left, right = margins
    grid_h = side + top + bottom
    grid_w = side + left + right
    if grid_h > 30 or grid_w > 30:
        return None

    num_colors = random.randint(2, min(4, max_depth))
    colors = random.sample(PALETTE, num_colors)
    depths = random.sample(list(range(1, max_depth + 1)), num_colors)
    depth_to_color = {int(depth): int(color) for depth, color in zip(depths, colors)}

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    frame_bbox = (top, left, top + side - 1, left + side - 1)
    expected = _render_output(grid.shape, frame_bbox, depth_to_color)

    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    size_log: list[int] = []
    require_multicell = random.random() < 0.75
    for depth, color in depth_to_color.items():
        sizes = _paint_random_components(
            interior,
            color,
            depth,
            favor_multicell=require_multicell or random.random() < 0.5,
        )
        if sizes is None:
            return None
        size_log.extend(sizes)

    if require_multicell and max(size_log, default=1) == 1:
        return None

    for depth, color in depth_to_color.items():
        if _count_components(interior, color) != int(depth):
            return None

    _paint_outside_distractors(grid, frame_bbox, colors)
    for color in colors:
        outside_mask = np.ones(grid.shape, dtype=bool)
        outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
        if not np.any((grid == int(color)) & outside_mask):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        built = _build_input_attempt()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")

from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
GREY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_CHOICES = (1, 2, 3, 4, 5)

MIN_COLOR_COUNT = 3
MAX_COLOR_COUNT = 5
INPUT_ROWS = (15, 16, 17, 18)
INPUT_COLS = (15, 16, 17, 18)
GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 120
PLACEMENT_GAP = 1


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(v) for v in np.unique(np.asarray(grid, dtype=int)))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def count_components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _square_specs(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    specs: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={BLACK}):
        count = int(np.count_nonzero(g == color))
        side = math.isqrt(count)
        assert side * side == count, f"color {color} count {count} is not a square"
        specs.append((color, side))
    sides = [side for _color, side in specs]
    assert len(specs) >= 1
    assert len(set(sides)) == len(sides), "square side lengths must be unique"
    return specs


def _render_output(
    ordered_specs: list[tuple[int, int]],
    *,
    mode: str = "local",
    global_phase: int = 0,
) -> np.ndarray:
    height = max(side for _color, side in ordered_specs)
    width = sum(side for _color, side in ordered_specs) + len(ordered_specs) - 1
    out = np.full((height, width), BLACK, dtype=int)

    col = 0
    for index, (color, side) in enumerate(ordered_specs):
        r0 = height - side
        c0 = col
        for lr in range(side):
            for lc in range(side):
                if mode == "solid":
                    fill = color
                elif mode == "local":
                    fill = color if (lr + lc) % 2 == 0 else GREY
                elif mode == "global":
                    fill = color if (r0 + lr + c0 + lc) % 2 == global_phase else GREY
                else:
                    raise ValueError(f"unknown render mode: {mode}")
                out[r0 + lr, c0 + lc] = fill
        col += side
        if index != len(ordered_specs) - 1:
            col += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repack scattered color pixels into ascending squares and recolor each square with a local checkerboard.

    Input structure
    - A black background contains 3 to 5 non-background colors scattered as arbitrary pixels.
    - Each non-background color has a unique perfect-square pixel count, so each color determines one square side length.

    Rule
    - Count each non-background color and compute its square side as the integer square root of that count.
    - Sort the colors by side length from small to large, then place their filled squares left-to-right with one black gap column and all bottoms aligned.
    - Inside each square, use the square's own top-left corner as local origin: cells with even local row+col stay the square color, and odd local row+col become grey.

    Key insight
    - The original input geometry is discarded after counting, and the checkerboard phase restarts independently inside every output square rather than following the global grid parity.
    """
    ordered_specs = sorted(_square_specs(np.asarray(grid, dtype=int)), key=lambda item: item[1])
    return _render_output(ordered_specs, mode="local")


def _zone_shapes(side: int) -> list[tuple[int, int]]:
    n = side * side
    base = [
        (side, side + 1),
        (side + 1, side),
        (side + 1, side + 1),
        (side, side + 2),
        (side + 2, side),
    ]
    if side == 1:
        base.extend([(1, 3), (2, 2), (2, 3)])
    out = []
    for h, w in base:
        if h * w >= n + 1:
            out.append((h, w))
    return sorted(set(out))


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _grow_blob(shape: tuple[int, int], target: int, blocked: np.ndarray) -> np.ndarray | None:
    h, w = shape
    free_cells = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if len(free_cells) < target:
        return None

    seed = random.choice(free_cells)
    blob = np.zeros((h, w), dtype=bool)
    blob[seed] = True

    while int(blob.sum()) < target:
        frontier = []
        for r, c in np.argwhere(blob):
            r = int(r)
            c = int(c)
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not blob[nr, nc]:
                    same_neighbors = 0
                    for rr, cc in neighbors4(nr, nc):
                        if 0 <= rr < h and 0 <= cc < w and blob[rr, cc]:
                            same_neighbors += 1
                    frontier.extend([(nr, nc)] * max(1, same_neighbors))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        blob[nr, nc] = True

    return blob


def _expand4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    expanded = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                expanded[nr, nc] = True
    return expanded


def _mask_is_solid_rectangle(mask: np.ndarray) -> bool:
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _make_zone_mask(side: int, zone_h: int, zone_w: int) -> np.ndarray | None:
    count = side * side
    for _ in range(MASK_ATTEMPTS):
        use_split = count >= 4 and min(zone_h, zone_w) >= 3 and random.random() < 0.55
        if use_split:
            max_parts = 3 if count >= 6 else 2
            parts = random.randint(2, min(max_parts, count))
            sizes = _random_partition(count, parts)
            union = np.zeros((zone_h, zone_w), dtype=bool)
            blocked = np.zeros((zone_h, zone_w), dtype=bool)
            ok = True
            for size in sizes:
                blob = _grow_blob((zone_h, zone_w), size, blocked)
                if blob is None:
                    ok = False
                    break
                union |= blob
                blocked = _expand4(union)
            if not ok:
                continue
            mask = union
        else:
            mask = _grow_blob((zone_h, zone_w), count, np.zeros((zone_h, zone_w), dtype=bool))
            if mask is None:
                continue

        if _mask_is_solid_rectangle(mask):
            continue
        return mask
    return None


def _rect_conflicts(
    candidate: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    gap: int = PLACEMENT_GAP,
) -> bool:
    r0, c0, h, w = candidate
    for or0, oc0, oh, ow in placed:
        separated = (
            r0 + h + gap <= or0
            or or0 + oh + gap <= r0
            or c0 + w + gap <= oc0
            or oc0 + ow + gap <= c0
        )
        if not separated:
            return True
    return False


def _place_zones(
    grid_h: int,
    grid_w: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    placements: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * len(sizes)
    placed_rects: list[tuple[int, int, int, int]] = []
    order = list(range(len(sizes)))
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        h, w = sizes[idx]
        candidates = []
        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                rect = (r0, c0, h, w)
                if not _rect_conflicts(rect, placed_rects):
                    candidates.append(rect)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placements[idx] = chosen
        placed_rects.append(chosen)

    return placements


def _first_occurrence_order(grid: np.ndarray) -> list[int]:
    seen = set()
    order = []
    for value in grid.flat:
        color = int(value)
        if color == BLACK or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _sample_specs() -> list[tuple[int, int]]:
    n_colors = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    sides = sorted(random.sample(SIDE_CHOICES, n_colors))
    colors = random.sample(INPUT_COLORS, n_colors)
    return list(zip(colors, sides))


def _independent_expected(specs: list[tuple[int, int]]) -> np.ndarray:
    return _render_output(sorted(specs, key=lambda item: item[1]), mode="local")


def _semantic_checks(grid: np.ndarray, specs: list[tuple[int, int]]) -> bool:
    expected = _independent_expected(specs)
    if not np.array_equal(main(grid), expected):
        return False

    ordered = sorted(specs, key=lambda item: item[1])
    if np.array_equal(expected, _render_output(ordered, mode="solid")):
        return False
    if np.array_equal(expected, _render_output(ordered, mode="global", global_phase=0)):
        return False
    if np.array_equal(expected, _render_output(ordered, mode="global", global_phase=1)):
        return False

    color_sorted = sorted(specs, key=lambda item: item[0])
    if np.array_equal(expected, _render_output(color_sorted, mode="local")):
        return False

    descending = sorted(specs, key=lambda item: item[1], reverse=True)
    if np.array_equal(expected, _render_output(descending, mode="local")):
        return False

    size_order = [color for color, _side in ordered]
    if _first_occurrence_order(grid) == size_order:
        return False

    component_counts = [count_components(grid == color) for color, _side in specs]
    if max(component_counts) < 2:
        return False

    irregular = False
    for color, _side in specs:
        mask = grid == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        if int(mask.sum()) < (r1 - r0 + 1) * (c1 - c0 + 1):
            irregular = True
            break
    if not irregular:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    zone_sizes = [random.choice(_zone_shapes(side)) for _color, side in specs]
    total_zone_area = sum(h * w for h, w in zone_sizes)
    total_pixels = sum(side * side for _color, side in specs)

    valid_dims = [
        (rows, cols)
        for rows in INPUT_ROWS
        for cols in INPUT_COLS
        if rows * cols >= total_zone_area + 8
        and rows * cols >= total_pixels + 80
    ]
    random.shuffle(valid_dims)

    for grid_h, grid_w in valid_dims:
        placements = _place_zones(grid_h, grid_w, zone_sizes)
        if placements is None:
            continue

        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        for (color, side), (zone_h, zone_w), (r0, c0, _h, _w) in zip(specs, zone_sizes, placements):
            mask = _make_zone_mask(side, zone_h, zone_w)
            if mask is None:
                break
            grid[r0 : r0 + zone_h, c0 : c0 + zone_w][mask] = color
        else:
            bg_fraction = float(np.count_nonzero(grid == BLACK)) / float(grid.size)
            if bg_fraction < 0.55:
                continue
            if _semantic_checks(grid, specs):
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
